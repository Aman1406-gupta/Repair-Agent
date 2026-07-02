import copy
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from asgard_mongo_connector.sdk.client import AsgardStoreMongoConnectorClient

from agent_builder.handlers.core.requests import (
    RegisterTaskRequest,
    RegisterReleaseTaskRequest,
    RegisterAgentRequest,
    UpdateAgentRequest,
    RegisterAgentMetadataRequest,
    parse_task_input,
    parse_task_input_list,
    TaskInput,
)
from agent_builder.utils.constants import (
    AGENT_AS_TASK,
    AGENT_AS_TOOLS,
    AGENT_COLLECTION,
    AGENT_DOC,
    AGENT_ID,
    AGENT_METADATA_AGENT_ID,
    AGENT_METADATA_COLLECTION,
    AGENT_TYPE,
    CONFIG_VARIABLES,
    DEFAULT_PARTNER_ID,
    DEFAULT_SWARM_TYPE,
    DESCRIPTION,
    ENABLED,
    LLM_CONFIG,
    NAME,
    PARTNER_ID,
    REMOTE_REQUEST,
    REMOTE_RESPONSE,
    ROUTER_MODEL_CONFIG,
    SERVER_TYPE,
    SKILLS_ZIP,
    SUBAGENTS,
    SESSIONS_COLLECTION,
    SWARM_TYPE,
    TASK_AS_ROUTER,
    TASK_AS_TOOLS,
    TASKS,
    TOOLS,
    VERSION,
    WORKFLOW_EDGES,
)
from agent_builder.storage.utils.mongo_topology import (
    AgentBuilderStoreError,
    DuplicateAgentMetadataError,
    EmbedStoreMixin,
    agent_request_to_doc,
    assign_missing_ids,
    collect_embedded_task_ids,
    convert_objectids,
    maybe_object_id,
    store_error_handler,
    task_request_to_doc,
    validate_workflow_edges,
)


logger = logging.getLogger(__name__)


def _summarize_agent_registration(agent_doc: Dict[str, Any]) -> Dict[str, Any]:
    """IDs from embedded tasks/tools in the agent document."""
    agent_id = str(agent_doc["_id"])
    task_ids: List[str] = []
    tool_ids: List[str] = []
    for task in agent_doc.get(TASKS) or []:
        tid = task.get("_id")
        task_ids.append(str(tid) if tid else None)
        for nested in task.get(TASK_AS_TOOLS) or []:
            nid = nested.get("_id")
            task_ids.append(str(nid))
        for tool in task.get(TOOLS) or []:
            oid = tool.get("_id")
            tool_ids.append(str(oid))

    router = agent_doc.get(TASK_AS_ROUTER)
    router_task_id: Optional[str] = None
    if isinstance(router, dict) and router.get("_id") is not None:
        router_task_id = str(router["_id"])
    return {
        "agent_id": agent_id,
        "task_ids": task_ids,
        "tool_ids": tool_ids,
        "router_task_id": router_task_id,
        "success": True,
    }


def _map_requests(items, converter, **kwargs):
    if not items:
        return []
    return [converter(item, **kwargs) for item in items]


class AgentBuilderMongoStore(EmbedStoreMixin):
    def __init__(
        self,
        *,
        server_type: str = SERVER_TYPE,
        partner_id: int = DEFAULT_PARTNER_ID,
        redis_client: Any = None,
        **connector_kwargs: Any,
    ) -> None:
        self._client = AsgardStoreMongoConnectorClient(
            store_host="asgard-master-store-v2:10000",
            **connector_kwargs,
        )
        self._server_type = server_type
        self._partner_id = partner_id
        self._redis_client = redis_client

    def set_redis_client(self, redis_client: Any) -> None:
        """Set the Redis client for distributed caching."""
        self._redis_client = redis_client
        logger.debug("Redis client set for AgentBuilderMongoStore")

    # ── Agent Registration, Update & Delete ─────────────────────────

    async def _resolve_template_agent_doc(self, agent_type: str) -> Optional[Dict[str, Any]]:
        meta = await self._find_one_cached(
            AGENT_METADATA_COLLECTION, {"name": agent_type}, partner_id=DEFAULT_PARTNER_ID,
        )
        if not meta:
            raise AgentBuilderStoreError(f"Agent metadata not found for agent type: {agent_type}")

        raw = meta.get(AGENT_METADATA_AGENT_ID)
        return await self._validate_and_fetch_metadata_agent_doc(raw, agent_type) if raw else None

    @store_error_handler({"name": "request.name"})
    async def register_agent(self, request: RegisterAgentRequest):
        if agent_template := await self._resolve_template_agent_doc(request.agent_type):
            return await self._register_agent_from_template(request, agent_template)
        return await self._register_agent_fresh(request)

    async def _register_agent_from_template(
        self, request: RegisterAgentRequest, template: Dict[str, Any],
    ) -> Dict[str, Any]:
        doc = template

        if request.tasks is not None:
            doc[TASKS] = _map_requests(request.tasks, task_request_to_doc)
        if request.agent_as_task is not None:
            doc[AGENT_AS_TASK] = _map_requests(
                request.agent_as_task, agent_request_to_doc,
                default_agent_type=request.agent_type,
                default_partner_id=request.partner_id,
            )
        if request.task_as_router is not None:
            doc[TASK_AS_ROUTER] = task_request_to_doc(request.task_as_router)
        elif "task_as_router" in request.model_fields_set and request.task_as_router is None:
            doc[TASK_AS_ROUTER] = None

        assign_missing_ids(doc)
        doc.pop("_id", None)
        doc.pop(AGENT_ID, None)

        inserted_id = await self._insert_one(AGENT_COLLECTION, doc, partner_id=request.partner_id)
        if not inserted_id:
            raise AgentBuilderStoreError(f"Insert agent failed for template: {request.name}")

        doc["_id"] = inserted_id
        doc[AGENT_ID] = str(inserted_id)

        update_data = {k: v for k, v in request.model_dump().items() if v is not None}
        update_data["agent_id"] = str(inserted_id)
        out = await self.update_agent(UpdateAgentRequest(**update_data))
        logger.debug("Agent registered from template: %s (id=%s)", request.name, inserted_id)
        return out

    async def _register_agent_fresh(self, request: RegisterAgentRequest) -> Dict[str, Any]:
        if not request.agent_type:
            raise AgentBuilderStoreError(
                "register_agent without a template requires 'agent_type'",
            )
        if not request.name or not str(request.name).strip():
            raise AgentBuilderStoreError(
                "register_agent without a template requires a non-empty 'name'",
            )

        tasks = _map_requests(request.tasks, task_request_to_doc)
        task_as_router_doc = task_request_to_doc(request.task_as_router) if request.task_as_router else None
        agent_as_task_docs = _map_requests(
            request.agent_as_task, agent_request_to_doc,
            default_agent_type=request.agent_type,
            default_partner_id=request.partner_id,
        )
        workflow_edges = request.workflow_edges or []

        doc = {
            NAME: request.name,
            DESCRIPTION: request.description,
            TASKS: tasks,
            WORKFLOW_EDGES: workflow_edges,
            ROUTER_MODEL_CONFIG: request.llm_config,
            SWARM_TYPE: request.swarm_type or DEFAULT_SWARM_TYPE,
            AGENT_AS_TASK: agent_as_task_docs,
            TASK_AS_ROUTER: task_as_router_doc,
            AGENT_TYPE: request.agent_type,
            PARTNER_ID: request.partner_id,
            VERSION: 0,
        }
        assign_missing_ids(doc)

        if workflow_edges:
            validate_workflow_edges(workflow_edges, collect_embedded_task_ids(doc))

        inserted_id = await self._insert_one(AGENT_COLLECTION, doc, partner_id=request.partner_id)
        if not inserted_id:
            raise AgentBuilderStoreError(f"Insert agent failed for: {request.name}")

        doc["_id"] = inserted_id
        doc[AGENT_ID] = str(inserted_id)
        await self._update_one(
            AGENT_COLLECTION, inserted_id, {AGENT_ID: doc[AGENT_ID]}, base_doc=doc,
            partner_id=request.partner_id,
        )

        logger.debug("Agent registered: %s (id=%s)", request.name, inserted_id)
        doc["_id"] = inserted_id
        return _summarize_agent_registration(doc)

    @store_error_handler({"id": "request.agent_id"})
    async def update_agent(self, request: UpdateAgentRequest) -> Dict[str, Any]:
        """Update an agent (name, description, tasks, llm_config, swarm_type, …)."""
        partner_id = request.partner_id
        updates = request.model_dump(exclude_unset=True, exclude={"agent_id", "partner_id"})
        agent_oid = maybe_object_id(request.agent_id)
        agent_doc = await self._find_one_cached(
            AGENT_COLLECTION, {"_id": agent_oid}, partner_id=partner_id,
        )
        if not agent_doc:
            raise AgentBuilderStoreError(f"Agent not found: {request.agent_id}")

        update_fields: Dict[str, Any] = {}

        simple_fields = {NAME, DESCRIPTION, SWARM_TYPE, AGENT_TYPE, WORKFLOW_EDGES}
        for field in simple_fields:
            if field in updates:
                update_fields[field] = updates[field]

        if LLM_CONFIG in updates:
            update_fields[ROUTER_MODEL_CONFIG] = updates[LLM_CONFIG]

        if TASKS in updates:
            tasks_raw = updates[TASKS]
            if tasks_raw and isinstance(tasks_raw[0], dict):
                tasks_raw = parse_task_input_list(tasks_raw)
            task_docs = _map_requests(tasks_raw, task_request_to_doc)
            assign_missing_ids({TASKS: task_docs})
            update_fields[TASKS] = task_docs

        if AGENT_AS_TASK in updates:
            raw = updates[AGENT_AS_TASK] or []
            if raw and isinstance(raw[0], dict):
                raw = [RegisterAgentRequest(**d) for d in raw]
            agent_docs = _map_requests(
                raw, agent_request_to_doc,
                default_agent_type=updates.get(AGENT_TYPE) or agent_doc.get(AGENT_TYPE),
                default_partner_id=partner_id,
            )
            assign_missing_ids({AGENT_AS_TASK: agent_docs})
            update_fields[AGENT_AS_TASK] = agent_docs

        if TASK_AS_ROUTER in updates:
            if updates[TASK_AS_ROUTER] is None:
                update_fields[TASK_AS_ROUTER] = None
            else:
                router_in = updates[TASK_AS_ROUTER]
                if isinstance(router_in, dict):
                    router_in = parse_task_input(router_in)
                router_doc = task_request_to_doc(router_in)
                assign_missing_ids({TASK_AS_ROUTER: router_doc})
                update_fields[TASK_AS_ROUTER] = router_doc

        if not agent_doc.get(AGENT_ID):
            update_fields[AGENT_ID] = str(agent_oid)
        if agent_doc.get(VERSION) is None:
            update_fields[VERSION] = 0
        if agent_doc.get(PARTNER_ID) is None:
            update_fields[PARTNER_ID] = partner_id

        stamped_updates = await self._update_one(
            AGENT_COLLECTION, agent_oid, update_fields, base_doc=agent_doc,
            partner_id=partner_id,
        )
        updated_doc = {**agent_doc, **stamped_updates}
        logger.debug("Agent updated: %s", request.agent_id)
        out = _summarize_agent_registration(updated_doc)
        out["success"] = True
        return out

    @store_error_handler({"id": "agent_id"})
    async def delete_agent(self, agent_id: Union[str, ObjectId], *, partner_id: Optional[int] = None) -> bool:
        agent_oid = maybe_object_id(agent_id)
        deleted_count = await self._delete_one(AGENT_COLLECTION, agent_oid, partner_id=partner_id)
        if deleted_count != 1:
            logger.warning("delete_agent: No document deleted for id=%s", agent_oid)
        else:
            logger.debug("Agent deleted: %s", agent_oid)
        return deleted_count == 1

    @store_error_handler()
    async def delete_agent_by_triple(
        self, agent_id: str, partner_id: int, version: int,
    ) -> bool:
        """Delete an agent by logical ``agent_id`` + ``partner_id`` + ``version``.

        Resolves the Mongo document, then reuses :meth:`delete_agent` (which runs
        ``_delete_one`` and drops the composite Redis cache key for that document).
        """
        doc = await self._find_one_cached(
            AGENT_COLLECTION,
            {AGENT_ID: str(agent_id).strip(), PARTNER_ID: partner_id, VERSION: version},
            partner_id=partner_id,
        )
        if not doc or not doc.get("_id"):
            return False
        return await self.delete_agent(str(doc["_id"]), partner_id=partner_id)

    # ── Agent Metadata ──────────────────────────────────────────────

    async def _validate_and_fetch_metadata_agent_doc(self, agent_id, agent_type):
        if not agent_id:
            raise AgentBuilderStoreError(f"AgentId is missing for agentType: {agent_type}")

        aid = maybe_object_id(agent_id)
        existing_agent_doc = await self._find_one_cached(
            AGENT_COLLECTION, {"_id": aid}, partner_id=DEFAULT_PARTNER_ID,
        )
        if not existing_agent_doc:
            raise AgentBuilderStoreError(
                f"Agent not found for agentId: {agent_id}, agentType: {agent_type}",
            )

        return existing_agent_doc

    @store_error_handler({"name": "request.name"})
    async def register_agent_metadata(self, request: RegisterAgentMetadataRequest) -> Dict[str, Any]:
        existing = await self._find_one_cached(
            AGENT_METADATA_COLLECTION, {"name": request.name}, partner_id=DEFAULT_PARTNER_ID,
        )
        if existing:
            raise DuplicateAgentMetadataError(
                f"Agent metadata with name '{request.name}' already exists",
            )

        if request.agent_id:
            await self._validate_and_fetch_metadata_agent_doc(request.agent_id, request.name)

        doc = {
            "name": request.name,
            "type": request.agent_type,
            "featureId": request.feature_id,
            "contextManagement": request.context_management,
            "clientIdentifier": request.client_identifier,
            "agentId": request.agent_id,
        }

        inserted_id = await self._insert_one(
            AGENT_METADATA_COLLECTION, doc, partner_id=DEFAULT_PARTNER_ID,
        )
        if not inserted_id:
            raise AgentBuilderStoreError(f"Insert agent metadata failed for: {request.name}")

        logger.debug("Agent metadata registered: %s (id=%s)", request.name, inserted_id)
        return {"metadata_id": str(inserted_id), "name": request.name}

    @store_error_handler({"name": "name"})
    async def update_agent_metadata(self, name: str, **updates: Any) -> Dict[str, Any]:
        metadata_doc = await self._find_one_cached(
            AGENT_METADATA_COLLECTION, {"name": name}, partner_id=DEFAULT_PARTNER_ID,
        )
        if not metadata_doc:
            raise AgentBuilderStoreError(f"Agent metadata not found: {name}")

        allowed = {"type", "featureId", "contextManagement", "clientIdentifier", "agentId"}
        unknown = set(updates) - allowed
        if unknown:
            raise AgentBuilderStoreError(f"Unknown update keys: {sorted(unknown)}")

        if updates.get("agentId"):
            await self._validate_and_fetch_metadata_agent_doc(updates.get("agentId"), name)

        update_fields = {k: v for k, v in updates.items() if v is not None}
        meta_oid = maybe_object_id(metadata_doc["_id"])
        await self._update_one(
            AGENT_METADATA_COLLECTION, meta_oid, update_fields, base_doc=metadata_doc,
            partner_id=DEFAULT_PARTNER_ID,
        )

        logger.debug("Agent metadata updated: %s", name)
        return {"name": name}

    @store_error_handler(fallback=[])
    async def get_all_agent_metadata(self, populate_cache: bool = True) -> List[Dict[str, Any]]:
        results = await self._find_many(AGENT_METADATA_COLLECTION, {}, partner_id=DEFAULT_PARTNER_ID)

        populated = []
        for metadata_doc in results:
            result = convert_objectids(metadata_doc)
            populated.append(result)

            if populate_cache and result.get("name") and self._redis_client:
                await self._sync_cache(
                    AGENT_METADATA_COLLECTION,
                    {"name": metadata_doc["name"]},
                    metadata_doc,
                )

        return populated

    # ── Session Persistence ──────────────────────────────────────────

    @store_error_handler()
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Persist a session to Mongo as a structured document.

        Stores ``agent_id``, ``partner_id``, ``version``, ``messages``,
        and ``persistent_mock_behaviors`` as top-level fields so sessions
        can be queried and inspected directly.
        """
        from langchain_core.messages import BaseMessage
        from agent_builder.storage.utils.state_serializer import sanitize

        def to_primitives(obj: Any) -> Any:
            """Round-trip through JSON to coerce non-primitive types (UUID, set, etc.) to strings/lists."""
            return json.loads(json.dumps(obj, default=str))

        _TRANSIENT_STATE_KEYS = {REMOTE_REQUEST, REMOTE_RESPONSE, "session_id", "request_id"}

        graph_state = session_data.get("graph_state", {})
        raw_messages = graph_state.get("messages", [])

        messages = [
            to_primitives(sanitize(msg.model_dump())) if isinstance(msg, BaseMessage) else to_primitives(sanitize(msg))
            for msg in raw_messages
        ]

        config_vars = graph_state.get(CONFIG_VARIABLES)
        if isinstance(config_vars, dict):
            config_vars = {k: v for k, v in config_vars.items() if k != AGENT_DOC}

        graph_state_extras = to_primitives({
            k: sanitize(config_vars if k == CONFIG_VARIABLES else v)
            for k, v in graph_state.items()
            if k != "messages" and k not in _TRANSIENT_STATE_KEYS
        })

        fields: Dict[str, Any] = {
            "agent_id": session_data.get("agent_id"),
            "partner_id": session_data.get("partner_id"),
            "version": session_data.get("version"),
            "messages": messages,
            "message_count": len(messages),
            "persistent_mock_behaviors": to_primitives(sanitize(session_data.get("persistent_mock_behaviors", {}))),
            "graph_state_extras": graph_state_extras,
            "last_updated_at": time.time(),
        }

        pid = session_data.get("partner_id")
        existing = await self._find_one(SESSIONS_COLLECTION, {"session_id": session_id}, partner_id=pid)
        if existing:
            await self._update_one(
                SESSIONS_COLLECTION,
                existing["_id"],
                fields,
                base_doc=existing,
                partner_id=pid,
            )
        else:
            await self._insert_one(SESSIONS_COLLECTION, {
                "session_id": session_id,
                **fields,
            }, partner_id=pid)

        logger.debug("Session saved to Mongo | session_id=%s", session_id)
        return True

    @store_error_handler()
    async def load_session(self, session_id: str, partner_id: int) -> Optional[Dict[str, Any]]:
        from agent_builder.storage.utils.state_serializer import unsanitize
        from agent_builder.llm_client.utils.remote_adapter import LANGCHAIN_TO_API_ROLE, _ROLE_TO_MESSAGE_CLASS

        doc = await self._find_one(SESSIONS_COLLECTION, {"session_id": session_id}, partner_id=partner_id)
        if not doc:
            return None

        raw_messages = doc.get("messages")
        if raw_messages is None:
            return None

        def _clean_for_constructor(d: dict) -> dict:
            """Strip empty-dict values that the protobuf layer substitutes for null."""
            return {k: v for k, v in d.items() if v != {}}

        raw_messages = json.loads(json.dumps(raw_messages, default=str))

        messages = [
            _ROLE_TO_MESSAGE_CLASS[LANGCHAIN_TO_API_ROLE[m["type"]]](**_clean_for_constructor(unsanitize(m)))
            for m in raw_messages
        ]

        extras = doc.get("graph_state_extras", {})
        extras = json.loads(json.dumps(extras, default=str))

        graph_state = {
            **unsanitize(extras),
            "messages": messages,
            "session_id": session_id,
        }

        return {
            "graph_state": graph_state,
            "agent_id": doc.get("agent_id"),
            "partner_id": doc.get("partner_id"),
            "version": doc.get("version"),
            "persistent_mock_behaviors": unsanitize(
                json.loads(json.dumps(doc.get("persistent_mock_behaviors", {}), default=str))
            ),
        }

    @store_error_handler()
    async def clone_session(self, source_session_id: str, target_session_id: str, partner_id: int) -> bool:
        """Clone a session document under a new session_id."""
        source_doc = await self._find_one(SESSIONS_COLLECTION, {"session_id": source_session_id}, partner_id=partner_id)
        if not source_doc:
            raise AgentBuilderStoreError(f"Source session not found: {source_session_id}")

        cloned_fields = {
            k: v for k, v in source_doc.items()
            if k not in ("_id", "session_id")
        }
        cloned_fields["last_updated_at"] = time.time()

        inserted_id = await self._insert_one(SESSIONS_COLLECTION, {
            "session_id": target_session_id,
            **cloned_fields,
        }, partner_id=partner_id)
        if not inserted_id:
            raise AgentBuilderStoreError(f"Clone session failed: {source_session_id} -> {target_session_id}")

        logger.debug("Session cloned | source=%s target=%s", source_session_id, target_session_id)
        return True

    # ── Generic Query Helpers ───────────────────────────────────────

    @store_error_handler({"collection": "collection"}, fallback=[])
    async def get_all_docs(self, collection: str, *, partner_id: Optional[int] = None) -> List[Dict[str, Any]]:
        results = await self._find_many(collection, {}, partner_id=partner_id)
        for doc in results:
            try:
                await self._sync_cache(collection, {"_id": doc["_id"]}, doc)
            except Exception as e:
                logger.error("Error syncing cache for collection: %s and doc: %s", collection, doc, e)
        if not results:
            logger.debug("No docs found in %s", collection)
        return [convert_objectids(doc) for doc in results]
