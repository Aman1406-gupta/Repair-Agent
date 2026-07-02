"""
Shared execution pipeline for ``/invoke`` (sync and streaming).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import tornado.web

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage,SystemMessage
from langchain_core.runnables import RunnableConfig
from bson import ObjectId

from agent_builder.handlers.core.base_handler import BaseBuilderHandler
from agent_builder.handlers.core.requests import AgentInvokeRequest
from agent_builder.handlers.utils.platform_fetch import fetch_and_persist_agent_from_platform
from agent_builder.utils.constants import (
    AGENT_COLLECTION,
    AGENT_DOC,
    AGENT_ID,
    AGENT_METADATA_COLLECTION,
    CLIENT_HTTP_HEADERS,
    CLIENT_IDENTIFIER,
    CONFIG_VARIABLES,
    DEFAULT_PARTNER_ID,
    FEATURE_ID,
    INVOKE_MESSAGE_COUNT,
    MESSAGES,
    PARTNER_ID,
    VERSION,
)
from agent_builder.mcp_client.client import load_mcp_tools, rebuild_mcp_tools
from agent_builder.base.tools_mock import MockStructuredTool
from agent_builder.storage.utils.builders import build_agent_from_doc
from agent_builder.llm_client.utils.remote_adapter import (
    _api_message_to_lc_message,
    get_plain_text_from_content,
    request_to_state,
)
from agent_builder.utils.misc import convert_objectid_to_str

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Everything produced during the preparation phase."""
    agent_id: str
    agent_doc: dict
    metadata: Optional[dict]
    state: dict
    partner_id: Optional[int] = None
    version: Optional[int] = None
    merged_mocks: dict = field(default_factory=dict)
    is_stateful: bool = False
    mcp_descriptors_by_task: dict = field(default_factory=dict)
    response_id: str = ""


_RESPONSE_ALREADY_WRITTEN = object()


def _resolve_last_active_task(raw: Any, doc: dict) -> dict:
    """Normalise ``lastActiveTask`` from the request into ``{"path": [...], "depth": N}``.

    Accepts either the canonical dict (returned as-is) or a flat list of task
    ``_id`` strings / task names.  IDs are resolved to names via the agent doc's
    ``tasks`` array; entries that don't match any ``_id`` are kept verbatim
    (they may already be task names).
    """
    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, list) or not raw:
        return {"path": [], "depth": 0}

    tasks = doc.get("tasks") or []
    id_to_name: dict[str, str] = {
        str(t["_id"]): t["name"]
        for t in tasks
        if isinstance(t, dict) and "_id" in t and "name" in t
    }

    resolved = [id_to_name.get(str(entry), entry) for entry in raw]
    return {"path": resolved, "depth": max(len(resolved) - 1, 0)}


class AgentExecutionHandler(BaseBuilderHandler):
    """Reusable building blocks for ``/invoke``.

    NOT a framework — just shared helpers.
    Subclasses write their own ``process()`` calling these as needed.
    """

    async def post(self) -> None:
        """Capture client HTTP headers for downstream remote proxy calls and state."""
        self._client_http_headers = self.request.headers
        await super().post()

    # ── public interface ─────────────────────────────────────────────────

    def validate_payload(self, payload: Dict[str, Any]):
        raise NotImplementedError("Must override this method in subclasses")

    def write_response(self, response: Any) -> None:
        if response is _RESPONSE_ALREADY_WRITTEN:
            return
        super().write_response(response)

    # ═══════════════════════════════════════════════════════════════════
    #  Orchestration
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_stateful(request: AgentInvokeRequest) -> bool:
        """``conversationState: stateful`` uses Redis session for multi-turn."""
        return request.conversationState == "stateful"

    async def _prepare_execution(
        self, request: AgentInvokeRequest,
    ) -> Tuple[Any, ExecutionContext]:
        """Build an `ExecutionContext` with everything needed to run the agent.

        Internally delegates to the stateful or stateless path based on
        ``_is_stateful(request)``.  Callers never need to know the mode.
        """
        def apply_client_headers_to_state(state: dict) -> None:
            headers = getattr(self, "_client_http_headers", None)
            if not headers:
                return
            state.setdefault(CONFIG_VARIABLES, {})[CLIENT_HTTP_HEADERS] = headers
        is_stateful = self._is_stateful(request)
        logger.debug("Preparing execution | agent_id=%s mode=%s", request.agentId, "stateful" if is_stateful else "stateless")
        if is_stateful:
            agent, ctx = await self._prepare_stateful_execution(request)
        else:
            agent, ctx = await self._prepare_stateless_execution(request)
            await self.redis_client.set_exec_status(request.sessionId, "running", request.id)

        apply_client_headers_to_state(ctx.state)
        cfgv = ctx.state[CONFIG_VARIABLES]
        cfgv["invokeDelivery"] = request.delivery
        n_messages = len([m for m in ctx.state.get(MESSAGES, []) if not isinstance(m, SystemMessage)])
        cfgv[INVOKE_MESSAGE_COUNT] = n_messages

        ctx.response_id = request.responseId or uuid.uuid4().hex
        ctx.state["response_id"] = ctx.response_id

        return agent, ctx

    @staticmethod
    def _enter_background(ctx: ExecutionContext) -> None:
        """Mark a context as background so finalization persists the session envelope.

        Background-mode replies must land in the canonical message history (the
        session envelope written by the stateful branch of ``_finalize_execution``),
        because the backend does not resend history. Forcing ``is_stateful`` makes the
        background completion persist ``graph_state.messages`` exactly like a stateful
        turn, regardless of the request's declared ``conversationState``.
        """
        ctx.is_stateful = True

    async def _finalize_execution(
        self,
        new_state: dict,
        request: AgentInvokeRequest,
        ctx: ExecutionContext,
    ) -> bool:
        """Persist results after agent execution.

        Returns ``True`` if normal finalization happened, ``False`` if the
        execution was stopped (output suppressed, state not updated).

        * **Stateful** — persists the session envelope (graph state, agent
          triple, merged mock behaviours) to Redis, and asynchronously
          dumps a copy to Mongo for durability.
        * **Stateless** — persists only ``last_active_task`` (lightweight).
        """
        should_proceed = await self._check_exec_status(request)
        if not should_proceed:
            logger.info(
                "Execution stopped — suppressing output | agent_id=%s session_id=%s request_id=%s",
                ctx.agent_id, request.sessionId, request.id,
            )
            return False

        logger.debug("Finalizing execution | agent_id=%s stateful=%s", ctx.agent_id, ctx.is_stateful)
        if ctx.is_stateful:
            from agent_builder.storage.utils.state_serializer import serialize_session_data

            session_data = {
                "graph_state": new_state,
                "agent_id": ctx.agent_id,
                "partner_id": ctx.partner_id,
                "version": ctx.version,
                "persistent_mock_behaviors": ctx.merged_mocks or {},
                "mcp_descriptors_by_task": ctx.mcp_descriptors_by_task or {},
            }
            serialized = serialize_session_data(session_data)
            await self.redis_client.set_extended_session_data(request.sessionId, serialized)
            asyncio.ensure_future(self._save_session_to_mongo(request.sessionId, session_data))
        else:
            last_active = new_state.get("last_active_task")
            if last_active is not None:
                await self.redis_client.set_last_active_task(request.sessionId, last_active)

        await self.redis_client.set_exec_status(request.sessionId, "completed", request.id)
        return True

    async def _finalize_interrupted_execution(
        self,
        partial_state: dict,
        request: AgentInvokeRequest,
        ctx: ExecutionContext,
    ) -> None:
        """Persist partial state after an interrupt and mark session completed.

        Unlike ``_finalize_execution``, this always persists (no re-check of
        exec_status) and only writes state in stateful mode.
        """
        self._sanitize_interrupted_state(partial_state)
        logger.info(
            "Finalizing interrupted execution | agent_id=%s session_id=%s request_id=%s stateful=%s",
            ctx.agent_id, request.sessionId, request.id, ctx.is_stateful,
        )
        if ctx.is_stateful:
            from agent_builder.storage.utils.state_serializer import serialize_session_data

            session_data = {
                "graph_state": partial_state,
                "agent_id": ctx.agent_id,
                "partner_id": ctx.partner_id,
                "version": ctx.version,
                "persistent_mock_behaviors": ctx.merged_mocks or {},
                "mcp_descriptors_by_task": ctx.mcp_descriptors_by_task or {},
            }
            serialized = serialize_session_data(session_data)
            await self.redis_client.set_extended_session_data(request.sessionId, serialized)
            asyncio.ensure_future(self._save_session_to_mongo(request.sessionId, session_data))

        await self.redis_client.set_exec_status(request.sessionId, "completed", request.id)

    @staticmethod
    def _sanitize_interrupted_state(state: dict) -> None:
        """Strip unmatched tool_calls from the last AIMessage.

        Safety net for all interrupt timings.  If the last AIMessage has
        tool_calls without corresponding ToolMessages, strip ALL tool_calls
        and orphaned ToolMessages to ensure a clean resumable state.
        """
        messages = state.get("messages")
        if not messages:
            return

        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and getattr(messages[i], "tool_calls", None):
                last_ai_idx = i
                break

        if last_ai_idx is None:
            return

        ai_msg = messages[last_ai_idx]
        result_ids = {
            m.tool_call_id
            for m in messages[last_ai_idx + 1:]
            if isinstance(m, ToolMessage)
        }
        unmatched = [tc for tc in ai_msg.tool_calls if tc["id"] not in result_ids]

        if unmatched:
            clean_msg = AIMessage(
                content=ai_msg.content or "",
                id=ai_msg.id,
                response_metadata=ai_msg.response_metadata,
            )
            state["messages"] = list(messages[:last_ai_idx]) + [clean_msg]

    @staticmethod
    def _tag_response_messages_with_id(state: dict, response_id: str) -> None:
        """Stamp ``responseId`` on all messages after the last ``HumanMessage``."""
        messages = state.get("messages", [])
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        for msg in messages[last_human_idx + 1:]:
            msg.additional_kwargs["responseId"] = response_id

    # ── core execution ───────────────────────────────────────────────────

    @staticmethod
    async def _invoke_agent(agent: Any, state: dict, agent_id: str, config: Optional[RunnableConfig] = None) -> dict:
        """Call ``agent.ainvoke()`` with uniform error handling."""
        try:
            result = await agent.ainvoke(state, config=config)
            if not isinstance(result, dict):
                raise ValueError("Agent returned non-dict state")
            return result
        except asyncio.TimeoutError:
            raise tornado.web.HTTPError(504, "Agent invocation timed out")
        except tornado.web.HTTPError:
            raise
        except Exception as exc:
            logger.exception("Agent invocation failed | agent_id=%s", agent_id)
            raise tornado.web.HTTPError(500, f"Agent failed: {exc}") from exc

    # ═══════════════════════════════════════════════════════════════════
    #  Stateless path
    # ═══════════════════════════════════════════════════════════════════

    async def _prepare_execution_from_request_body(
        self, request: AgentInvokeRequest, *, is_stateful: bool,
    ) -> Tuple[Any, ExecutionContext]:
        """Build agent + state from the invoke body (no Redis ``graph_state`` yet)."""
        doc, metadata = await self._resolve_agent_doc(request)

        partner_id = int(doc.get(PARTNER_ID)) if doc.get(PARTNER_ID) is not None else request.partnerId
        jwt = self._extract_jwt(request)
        mcp_tools_by_task, mcp_descriptors_by_task = await load_mcp_tools(doc, partner_id, jwt, self.redis_client)

        agent = await self._build_agent(doc, metadata, extra_tools_by_task=mcp_tools_by_task, session_id=request.sessionId)

        state = request_to_state(request)

        context_management = (metadata or {}).get("contextManagement", False)
        if context_management and doc:
            state.setdefault(CONFIG_VARIABLES, {})[AGENT_DOC] = convert_objectid_to_str(doc)

        if mcp_descriptors_by_task:
            state.setdefault(CONFIG_VARIABLES, {})["mcp_config"] = {
                "tools_by_task": mcp_descriptors_by_task,
            }

        if not request.lastActiveTask:
            cached = await self.redis_client.get_last_active_task(request.sessionId)
            if cached is not None:
                state["last_active_task"] = cached
        else:
            state["last_active_task"] = _resolve_last_active_task(request.lastActiveTask, doc)

        request_mocks = request.mockToolBehaviour or {}
        if request_mocks:
            self._apply_mock_tool_behavior(agent, request_mocks)

        logical_agent_id = doc.get(AGENT_ID) or request.agentId
        doc_partner_id = doc.get(PARTNER_ID)
        doc_version = doc.get(VERSION)

        if is_stateful:
            await self._persist_input_and_set_running(
                request, state, doc, request_mocks, mcp_descriptors_by_task,
            )

        return agent, ExecutionContext(
            agent_id=str(logical_agent_id),
            partner_id=int(doc_partner_id) if doc_partner_id is not None else request.partnerId,
            version=int(doc_version) if doc_version is not None else request.version,
            agent_doc=doc,
            metadata=metadata,
            state=state,
            merged_mocks=request_mocks,
            is_stateful=is_stateful,
            mcp_descriptors_by_task=mcp_descriptors_by_task,
        )

    async def _prepare_stateless_execution(
        self, request: AgentInvokeRequest,
    ) -> Tuple[Any, ExecutionContext]:
        return await self._prepare_execution_from_request_body(request, is_stateful=False)

    async def _resolve_agent_doc(self, request: AgentInvokeRequest) -> Tuple[dict, dict]:
        """Fetch agent doc + metadata, validate required fields.

        Returns ``(doc, metadata)``.
        Raises 400 if ``agent_type`` is missing from the agent doc, and 404 if
        the agent or its metadata row is missing.
        """
        if request.partnerId is not None and request.version is not None:
            doc = await self.mongo_client._find_one_cached(
                AGENT_COLLECTION,
                {AGENT_ID: request.agentId, PARTNER_ID: int(request.partnerId), VERSION: int(request.version)},
                partner_id=int(request.partnerId),
            )
            if not doc:
                logger.warning(
                    "Agent not in store, attempting platform fetch fallback | agentId=%s partnerId=%s version=%s",
                    request.agentId, request.partnerId, request.version,
                )
                try:
                    doc = await fetch_and_persist_agent_from_platform(
                        self.mongo_client,
                        request.agentId,
                        int(request.partnerId),
                        int(request.version),
                    )
                except tornado.web.HTTPError:
                    raise
                except Exception as exc:
                    raise tornado.web.HTTPError(
                        404,
                        f"Agent '{request.agentId}' not found (platform fallback failed: {exc})",
                    ) from exc
        else:
            doc = await self.mongo_client._find_one_cached(AGENT_COLLECTION, {"_id": ObjectId(request.agentId)}, partner_id=request.partnerId)
        if not doc:
            raise tornado.web.HTTPError(404, f"Agent '{request.agentId}' not found")

        logger.debug("Agent doc resolved | agent_id=%s name=%s", request.agentId, doc.get("name"))

        agent_type = doc.get("agent_type")

        if not agent_type:
            raise tornado.web.HTTPError(400, "agent_type is required")

        if doc.get(PARTNER_ID) is None:
            raise tornado.web.HTTPError(400, f"no {PARTNER_ID} in agent document for agent '{request.agentId}'")

        metadata = await self.mongo_client._find_one_cached(
            AGENT_METADATA_COLLECTION, {"name": agent_type}, partner_id=DEFAULT_PARTNER_ID,
        )
        if not metadata:
            raise tornado.web.HTTPError(
                404, f"Agent metadata not found for agent_type '{agent_type}'",
            )

        feature_id = metadata.get(FEATURE_ID)
        client_identifier = metadata.get(CLIENT_IDENTIFIER)

        if feature_id is None or client_identifier is None:
            raise tornado.web.HTTPError(400, f"no {FEATURE_ID} or {CLIENT_IDENTIFIER} in agent metadata for agent_type '{agent_type}'")

        return doc, metadata

    async def _build_agent(self, doc: dict, metadata: dict, extra_tools_by_task=None, session_id: str | None = None) -> Any:
        """Resolve LLM configs, build agent, propagate overrides."""
        agent = build_agent_from_doc(doc, extra_tools_by_task=extra_tools_by_task, session_id=session_id)


        feature_id = metadata.get(FEATURE_ID)
        partner_id = doc.get(PARTNER_ID)
        client_identifier = metadata.get(CLIENT_IDENTIFIER)
        self.propagate_llm_overrides_to_tasks(agent, feature_id, partner_id, client_identifier)

        return agent

    def _extract_jwt(self, request: AgentInvokeRequest) -> str:
        token = getattr(request, "sprMcpAuthToken", "") or ""
        return token.removeprefix("Bearer ").strip() if token else ""

    # ═══════════════════════════════════════════════════════════════════
    #  Stateful path
    # ═══════════════════════════════════════════════════════════════════

    async def _prepare_stateful_execution(
        self, request: AgentInvokeRequest,
    ) -> Tuple[Any, ExecutionContext]:
        if not request.sessionId:
            raise tornado.web.HTTPError(400, "sessionId is required for stateful requests")

        session_data = await self.redis_client.get_extended_session_data(request.sessionId)
        if session_data is None:
            session_data = await self.mongo_client.load_session(request.sessionId, partner_id=int(request.partnerId))
            if session_data is not None:
                logger.info("Session restored from Mongo | session_id=%s", request.sessionId)
        if session_data is None:
            logger.debug(
                "Stateful bootstrap | session_id=%s agent_id=%s",
                request.sessionId,
                request.agentId,
            )
            return await self._prepare_execution_from_request_body(request, is_stateful=True)

        agent, state, doc, persisted_mocks, mcp_descriptors_by_task = await self._reconstruct_from_session(
            request, request.sessionId, session_data,
        )

        metadata = await self._resolve_metadata_and_propagate(agent, doc)

        context_management = (metadata or {}).get("contextManagement", False)
        if context_management and doc:
            state.setdefault(CONFIG_VARIABLES, {})[AGENT_DOC] = convert_objectid_to_str(doc)

        request_mocks = request.mockToolBehaviour or {}
        merged_mocks = {**persisted_mocks, **request_mocks}
        if merged_mocks:
            self._apply_mock_tool_behavior(agent, merged_mocks)

        user_api_msg = next(
            (m for m in reversed(request.messages) if m.role == "user"), None,
        )
        if user_api_msg:
            human_msg = _api_message_to_lc_message(user_api_msg)
            human_msg.id = user_api_msg.id or request.id
            human_msg.additional_kwargs["responseId"] = human_msg.id
            state["messages"].append(human_msg)

        if request.lastActiveTask:
            state["last_active_task"] = _resolve_last_active_task(request.lastActiveTask, doc)

        # Eagerly persist state (with user input) and mark execution as running.
        # This ensures a subsequent request always sees this user message even if
        # the current execution is stopped before finalization.
        await self._persist_input_and_set_running(
            request, state, doc, merged_mocks, mcp_descriptors_by_task,
        )


        return agent, ExecutionContext(
            agent_id=str(doc.get(AGENT_ID) or request.agentId),
            partner_id=int(doc.get(PARTNER_ID)) if doc.get(PARTNER_ID) is not None else request.partnerId,
            version=int(doc.get(VERSION)) if doc.get(VERSION) is not None else request.version,
            agent_doc=doc,
            metadata=metadata,
            state=state,
            merged_mocks=merged_mocks,
            is_stateful=True,
            mcp_descriptors_by_task=mcp_descriptors_by_task
        )

    async def _reconstruct_from_session(
        self,
        request: AgentInvokeRequest,
        session_id: Optional[str],
        session_data: Dict[str, Any],
    ) -> Tuple[Any, dict, dict, dict, dict]:
        """Reload agent + state from a stored Redis or Mongo session.

        Returns ``(agent, state, agent_doc, persisted_mock_behaviors, mcp_descriptors_by_task)``.
        When the requested ``agent_id`` differs from the stored one the
        session is reset to a fresh initial state.

        Overwrites ``state["request_id"]`` with ``request.id`` so restored graph state matches
        this HTTP invoke (persisted state still holds the previous turn's id until here).
        """
        stored_agent_id = session_data.get("agent_id")
        stored_partner_id = session_data.get("partner_id")
        stored_version = session_data.get("version")

        same_agent = stored_agent_id in (None, request.agentId)
        same_partner = stored_partner_id in (None, request.partnerId)
        same_version = stored_version in (None, request.version)
        same_triple = same_agent and same_partner and same_version

        doc = await self._fetch_agent_doc_for_invoke(request)
        if not doc:
            raise tornado.web.HTTPError(404, f"Agent '{request.agentId}' not found")

        partner_id = int(doc.get(PARTNER_ID)) if doc.get(PARTNER_ID) is not None else request.partnerId
        jwt = self._extract_jwt(request)

        if same_triple:
            mcp_descriptors_by_task = session_data.get("mcp_descriptors_by_task", {})
            mcp_tools_by_task = await rebuild_mcp_tools(mcp_descriptors_by_task, doc, partner_id, jwt)
        else:
            mcp_tools_by_task, mcp_descriptors_by_task = await load_mcp_tools(doc, partner_id, jwt, self.redis_client)

        agent = build_agent_from_doc(doc, extra_tools_by_task=mcp_tools_by_task, session_id=request.sessionId)


        state = session_data["graph_state"]
        mocks = session_data.get("persistent_mock_behaviors", {})

        if mcp_descriptors_by_task:
            state.setdefault(CONFIG_VARIABLES, {})["mcp_config"] = {
                "tools_by_task": mcp_descriptors_by_task,
            }

        if not same_triple:
            logger.info(
                "Switching agent within session %s : %s -> %s",
                session_id, stored_agent_id, request.agentId,
            )

        # Persisted graph_state carries the prior invoke's id; overwrite with this invoke's id.
        state["request_id"] = request.id

        return agent, state, doc, mocks, mcp_descriptors_by_task

    async def _fetch_agent_doc_for_invoke(self, request: AgentInvokeRequest) -> dict:
        """Fetch agent doc for invoke resolution (triple when provided, else legacy ObjectId)."""
        if request.partnerId is not None and request.version is not None:
            doc = await self.mongo_client._find_one_cached(
                AGENT_COLLECTION,
                {AGENT_ID: request.agentId, PARTNER_ID: int(request.partnerId), VERSION: int(request.version)},
                partner_id=int(request.partnerId),
            )
            if not doc:
                raise tornado.web.HTTPError(
                    404,
                    f"Agent '{request.agentId}' not found for partnerId={request.partnerId} version={request.version}",
                )
            return doc
        # Legacy: agentId is Mongo ObjectId
        return await self.mongo_client._find_one_cached(
            AGENT_COLLECTION, {"_id": ObjectId(request.agentId)},
            partner_id=request.partnerId,
        )

    async def _resolve_metadata_and_propagate(
        self, agent: Any, doc: dict,
    ) -> Optional[dict]:
        """Fetch metadata for the agent's type and propagate LLM overrides."""
        agent_type = doc.get("agent_type")
        if not agent_type:
            return None

        if doc.get(PARTNER_ID) is None:
            agent_oid = doc.get("_id")
            aid = str(agent_oid) if agent_oid is not None else ""
            raise tornado.web.HTTPError(400, f"no {PARTNER_ID} in agent document for agent '{aid}'")

        metadata = await self.mongo_client._find_one_cached(
            AGENT_METADATA_COLLECTION, {"name": agent_type}, partner_id=DEFAULT_PARTNER_ID,
        )
        if not metadata:
            raise tornado.web.HTTPError(404, f"Agent metadata not found for agent_type '{agent_type}'")

        feature_id = metadata.get(FEATURE_ID)
        partner_id = doc.get(PARTNER_ID)
        client_identifier = metadata.get(CLIENT_IDENTIFIER)
        if feature_id is None or client_identifier is None:
            raise tornado.web.HTTPError(400, f"no {FEATURE_ID} or {CLIENT_IDENTIFIER} in agent metadata for agent_type '{agent_type}'")

        self.propagate_llm_overrides_to_tasks(agent, feature_id, partner_id, client_identifier)

        return metadata

    # ── exec_status helpers ───────────────────────────────────────────────

    async def _persist_input_and_set_running(
        self,
        request: AgentInvokeRequest,
        state: dict,
        agent_doc: dict,
        merged_mocks: dict,
        mcp_descriptors_by_task: Optional[dict] = None,
    ) -> None:
        """Eagerly persist state (with user input appended) and set exec_status to running.

        Pipelined into one logical step so a subsequent request always sees
        this turn's user message, even if the current execution is stopped.
        """
        from agent_builder.storage.utils.state_serializer import serialize_session_data

        session_data = {
            "graph_state": state,
            "agent_id": request.agentId,
            "partner_id": request.partnerId,
            "version": request.version,
            "persistent_mock_behaviors": merged_mocks or {},
            "mcp_descriptors_by_task": mcp_descriptors_by_task or {},
        }
        serialized = serialize_session_data(session_data)
        await self.redis_client.set_extended_session_data(request.sessionId, serialized)
        asyncio.ensure_future(self._save_session_to_mongo(request.sessionId, session_data))
        await self.redis_client.set_exec_status(request.sessionId, "running", request.id)

    async def _check_exec_status(self, request: AgentInvokeRequest) -> bool:
        """Return True if this request should proceed with normal finalization.

        Normal finalization happens only when exec_status is ``running`` AND
        the stored requestId matches the current request.  Any other state
        (stopped, mismatched requestId) means this execution was superseded.
        """
        status_data = await self.redis_client.get_exec_status(request.sessionId)
        if status_data is None:
            return True
        return (
            status_data.get("status") == "running"
            and status_data.get("requestId") == request.id
        )

    # ── shared helpers ───────────────────────────────────────────────────

    @staticmethod
    def _apply_mock_tool_behavior(agent_instance, mock_behaviors: dict) -> None:
        """Apply mock tool behaviors to agent tools."""
        try:
            for task in agent_instance.tasks:
                for tool in task.tools:
                    if tool.tags and tool.tags[0] in mock_behaviors and isinstance(tool, MockStructuredTool):
                        tool._behavior = mock_behaviors[tool.tags[0]]
        except Exception:
            logger.exception("Failed to apply mock tool behavior")

    def propagate_llm_overrides_to_tasks( self, agent_instance, feature_id: str = None, partner_id: str = None, client_identifier: str = None) -> None:
        """Recursively propagate feature_id, partner_id, and client_identifier to LLM configs."""
        tasks_list = getattr(agent_instance, "tasks", None) or []
        sub_tasks_list = getattr(agent_instance, "sub_tasks", None) or []
        if tasks_list or sub_tasks_list:
            # ``Agent`` / ``agent_wrapper`` expose children via ``tasks`` / ``sub_tasks``.
            all_tasks = list(tasks_list) + list(sub_tasks_list)
        elif getattr(agent_instance, "task_config", None) is not None:
            # A nested task from ``task_as_tool`` is a plain ``Task`` with neither list, so propagate that node itself.
            all_tasks = [agent_instance]
        else:
            all_tasks = []
        for task in all_tasks:
            if llm_cfg := getattr(getattr(task, 'task_config', None), 'llm_config', None):
                if partner_id is not None:
                    llm_cfg.partner_id = int(partner_id)
                if feature_id is not None:
                    llm_cfg.tracking_params = {**getattr(llm_cfg, 'tracking_params', {}), "feature": feature_id}
                if client_identifier is not None:
                    llm_cfg.client_identifier = client_identifier
            if getattr(task, 'task_type', None) == 'agent_wrapper':
                self.propagate_llm_overrides_to_tasks(
                    task, feature_id, partner_id, client_identifier,
                )

            # task_as_tools / agent_as_tools.
            for task_as_tool in getattr(task, "tools", []) or []:
                extras = getattr(task_as_tool, "extras", None)
                nested = extras.get("nested_as_tool") if isinstance(extras, dict) else None
                if nested is None:
                    continue
                self.propagate_llm_overrides_to_tasks(
                    nested, feature_id, partner_id, client_identifier,
                )

    # ── response building ────────────────────────────────────────────────

    @staticmethod
    def _build_stopped_response(request: AgentInvokeRequest) -> dict:
        now = int(time.time())
        return {
            "apiVersion": "1.0",
            "sessionId": request.sessionId,
            "id": request.id,
            "createdAt": now,
            "updatedAt": now,
            "content": [],
            "status": "STOPPED",
            "index": 0,
            "text": "",
            "usage": {},
        }

