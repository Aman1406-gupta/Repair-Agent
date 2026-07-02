"""Release task that delegates conversation handling to a remote endpoint."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict

logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, tool
from langchain_core.runnables.config import ensure_config
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent_builder.base.configs import HttpConfig, RemoteReleaseMetadata, TaskConfig
from agent_builder.base.state import State
from agent_builder.base.task import Task
from agent_builder.utils.constants import (
    DESCRIPTION,
    ENVELOPE_KEY,
    HTTP_CONFIG,
    LAST_ACTIVE_TASK,
    MESSAGES,
    NAME,
    RAW_RESPONSE_KEY,
    ATTRIBUTES,
    TASK_TYPE_REMOTE_RELEASE,
)
from agent_builder.utils.misc import remove_system_prompt
from agent_builder.llm_client.remote_chat_model import RemoteChatModel
from agent_builder.llm_client.utils.remote_adapter import state_to_response
from agent_builder.llm_client.utils.remote_chat_helpers import is_valid_last_active_task_obj


class RemoteRelease(Task):
    """Release task constructed from a fully resolved Mongo document.

    The document contains an ``http_config`` for the remote endpoint and
    optional ``attributes``. A ``task_form`` may be present on the stored
    document for external consumers; it is not read by this class.
    """

    def __init__(self, release_doc: Dict[str, Any], memory=None):
        http_cfg_raw = release_doc.get(HTTP_CONFIG)

        if not http_cfg_raw or not isinstance(http_cfg_raw, dict) or not http_cfg_raw.get("url").strip():
            raise ValueError("Invalid HTTP configuration")

        http_config = HttpConfig(
            url=http_cfg_raw.get("url"),
            proxy_server=http_cfg_raw.get("proxy_server", ""),
            proxy_port=http_cfg_raw.get("proxy_port", ""),
        )

        self.remote_release_metadata = RemoteReleaseMetadata(
            release_name=release_doc[NAME],
            release_description=release_doc[DESCRIPTION],
            http_config=http_config,
        )

        self.task_config = TaskConfig(
            name=release_doc[NAME],
            description=release_doc[DESCRIPTION],
        )
        self.task_config.task_type = TASK_TYPE_REMOTE_RELEASE
        
        self.memory = memory
        self._current_state_class = State
        self.tools = []

        self.llm = RemoteChatModel(
            http_config=self.remote_release_metadata.http_config,
        )
        self.llm_node = self._get_remote_chatbot_node()

        self._attributes: Dict[str, Any] = release_doc.get(ATTRIBUTES, {})
        self._allowed_handoff_targets: set = set()

        self.graph = self._build_graph()
        logger.debug("RemoteRelease task created | name=%s url=%s",
                      release_doc[NAME], http_config.url)

    # ── preprocessing ──────────────────────────────────────────────────

    def preprocess_state(self, state, **kwargs):
        state = remove_system_prompt(state)
        return state

    async def _call_remote_llm(self, state: dict, config: RunnableConfig | None = None):
        """``agenerate`` against remote copilot; returns a single ``AIMessage``."""
        cfg = ensure_config(config)
        llm_result = await self.llm.agenerate(
            [state[MESSAGES]],
            callbacks=cfg.get("callbacks"),
            tags=cfg.get("tags"),
            metadata=cfg.get("metadata"),
            full_state=state,
        )
        return llm_result.generations[0][0].message

    # ── chatbot node ───────────────────────────────────────────────────

    def _get_remote_chatbot_node(self):
        """Build the async function that proxies to the remote LLM."""
        task = self

        async def chatbot(state: dict, **kwargs):
            cfg = ensure_config(kwargs.get("config"))
            msg = await task._call_remote_llm(state, config=cfg)

            new_state = dict(state)
            new_state[MESSAGES] = list(state[MESSAGES]) + [msg]
            return new_state

        return chatbot

    # ── output-state builder ───────────────────────────────────────────
    def postprocess_state(self, output_state: dict):
        """Release handoff from envelope ``lastActiveTask`` (remote body already on state)."""

        current_last_active_task = output_state[LAST_ACTIVE_TASK]

        for msg in reversed(output_state.get(MESSAGES, [])):
            additional_kwargs = getattr(msg, "additional_kwargs", None) or {}
            envelope = additional_kwargs.get(ENVELOPE_KEY)
            if not envelope:
                continue
            if not envelope.get("is_remote_response"):
                return output_state
            last_active = envelope.get("lastActiveTask")
            if not last_active or not is_valid_last_active_task_obj(last_active):
                output_state[LAST_ACTIVE_TASK] = {
                    "path": current_last_active_task["path"][:-1],
                    "depth": current_last_active_task["depth"] - 1,
                }
                return output_state
            if (
                last_active != current_last_active_task
                and last_active["depth"] == current_last_active_task["depth"]
                and last_active["path"][-1] in self._allowed_handoff_targets
            ):
                target = last_active["path"][-1]
                output_state[LAST_ACTIVE_TASK]["path"][-1] = target

                additional_kwargs['remote_handoff'] = (True, f"handing off to {last_active}")
                return Command(
                    update=output_state,
                    goto=target,
                    graph=Command.PARENT,
                )
            return output_state

        output_state[LAST_ACTIVE_TASK] = {
            "path": current_last_active_task["path"][:-1],
            "depth": current_last_active_task["depth"] - 1,
        }

        return output_state
        

    # ── single turn ────────────────────────────────────────────────────

    async def _async_single_turn(self, input_state, **kwargs):
        preprocessed = self.preprocess_state(input_state)
        output_state = await self.llm_node(preprocessed, **kwargs)
        return self.postprocess_state(
            output_state,
        )

    # ── graph ──────────────────────────────────────────────────────────

    def _build_graph(self, state_class=None):
        state_class = state_class or self._current_state_class
        builder = StateGraph(state_class)
        builder.add_node("chatbot", self._async_single_turn)
        builder.set_entry_point("chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile(checkpointer=self.memory)

    # ── invoke / ainvoke ───────────────────────────────────────────────

    def invoke(self, input_state, config: RunnableConfig = None, **kwargs):
        return self.graph.invoke(input_state, config=config, **kwargs)

    async def ainvoke(self, input_state, config: RunnableConfig = None, **kwargs):
        return await self.graph.ainvoke(input_state, config=config, **kwargs)

    # ── no-ops ─────────────────────────────────────────────────────────

    def add_tools(self, new_tools):
        non_transfer = []
        for t in new_tools:
            if t.name == "transfer_tool":
                self._allowed_handoff_targets = set(
                    getattr(t, '_allowed_tasks', [])
                )
            else:
                non_transfer.append(t.name)
        if non_transfer:
            logger.debug(
                "RemoteRelease '%s' skipping local tool binding (forwarded via mcpConfig in state): %s",
                self.task_config.name, non_transfer,
            )

    def update_system_prompt(self, tmpl):
        pass

    def as_tool(self) -> StructuredTool:
        task = self
        name = f"run_{task.task_config.name.replace(' ', '_')}"
        description = (
            f"Invoke the remote release «{task.task_config.name}»: "
            f"{task.task_config.description}. "
            "The full current conversation state is forwarded to the remote "
            "service. The tool output is the raw remote response body."
        )

        async def _run(state: Annotated[dict, InjectedState]) -> Any:
            cfg = ensure_config(None)
            isolated = dict(state)
            isolated[MESSAGES] = list(state.get(MESSAGES, []))

            preprocessed = task.preprocess_state(isolated)
            msg = await task._call_remote_llm(preprocessed, config=cfg)

            raw_response = (msg.response_metadata or {}).get(RAW_RESPONSE_KEY)
            if isinstance(raw_response, dict):
                return raw_response

            preprocessed[MESSAGES] = list(preprocessed[MESSAGES]) + [msg]
            return state_to_response(preprocessed)

        _run.__name__ = name
        _run.__doc__ = description

        return tool(_run)

    def compile_with_state(self, state_class=None):
        self._current_state_class = state_class or State
        self.graph = self._build_graph(state_class)
