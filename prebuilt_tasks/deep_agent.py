import base64
import contextvars
import copy
import io
import logging
import os
import zipfile
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, List, cast

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.types import Command

from agent_builder.base.configs import TaskConfig
from agent_builder.base.state import State
from agent_builder.base.task import Task
from agent_builder.llm_client.sprinklr_chat_model import SprinklrChatModel
from agent_builder.utils.constants import PARENT_ROUTER_NODE
from agent_builder.utils.misc import strip_ephemeral_metadata
from agent_builder.utils.preprocessors import preprocessors_dict

try:
    from deepagents import create_deep_agent
    from deepagents.backends.local_shell import LocalShellBackend
except ImportError as _exc:
    raise ImportError(
        "deepagents is required for DeepAgentsTask. "
        "Install it with: pip install deepagents"
    ) from _exc

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.todo import PlanningState

logger = logging.getLogger(__name__)

DEEP_TRANSFER_TOOL_NAME = "deep_agent_transfer"

_SANDBOX_ENABLED = os.environ.get("SANDBOX_ENABLED", "false").lower() in ("true", "1", "yes")
_SANDBOX_SERVICE_URL = os.environ.get("SANDBOX_SERVICE_URL", "http://localhost:10000")
_TODO_SCOPE = os.environ.get("DEEP_AGENT_TODO_SCOPE", "local")  # "local" = per-task, "global" = shared
_COMPACTION_MAX_INPUT_TOKENS = int(os.environ.get("DEEP_AGENT_COMPACTION_MAX_INPUT_TOKENS", "0"))

_SAVED_TODOS: contextvars.ContextVar[list] = contextvars.ContextVar("_deep_agent_saved_todos", default=[])


class TodoInjectionMiddleware(AgentMiddleware[PlanningState[Any], Any, Any]):
    """Middleware that injects the current todo items into the system prompt.

    The built-in TodoListMiddleware only injects *instructions* about the
    write_todos tool but never shows the model the actual todo items stored
    in state.  This middleware fills that gap: on every model call it reads
    ``request.state["todos"]`` and appends a formatted snapshot to the
    system message so the model always sees its current task list — even
    after context compaction removes earlier write_todos tool-call messages.
    """

    state_schema = PlanningState  # type: ignore[assignment]

    def _format_todos(self, todos: list[dict]) -> str:
        if not todos:
            return ""
        lines = ["## Current Todo List"]
        for i, todo in enumerate(todos, 1):
            status = todo.get("status", "pending")
            content = todo.get("content", "")
            marker = {"completed": "x", "in_progress": "~", "pending": " "}.get(status, " ")
            lines.append(f"  {i}. [{marker}] {content} ({status})")
        lines.append(
            "\nKeep this list up to date using the `write_todos` tool as you "
            "make progress."
        )
        return "\n".join(lines)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        request = self._inject(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        request = self._inject(request)
        return await handler(request)

    def _inject(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        todos = request.state.get("todos") or _SAVED_TODOS.get([])
        if not todos:
            return request
        snippet = self._format_todos(todos)
        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{snippet}"},
            ]
        else:
            new_content = [{"type": "text", "text": snippet}]
        new_system = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_content)
        )
        return request.override(system_message=new_system)


class DeepAgentsTask(Task):
    """Task backed by the LangChain Deep Agents harness.

    Provides built-in harness features on top of user-supplied tools:
      - Filesystem access  (ls, read_file, write_file, edit_file, glob, grep)
      - Shell execution     (execute — via LocalShellBackend)
      - Planning            (write_todos)
      - Sub-agent spawning  (task tool)
      - Context summarisation

    The deep agent manages its own tool-calling loop internally, so this
    task's graph is a single "chatbot" node (no outer ToolNode).

    Handoffs are supported via a lightweight ``transfer_tool`` injected
    into the deep agent — mirroring ``special_transfer_tool`` from
    ``agent_builder.base.tools``.  After the deep agent completes, the
    chatbot node detects the transfer call and re-emits a ``Command``
    (with the correct ``last_active_task`` state updates) at the
    Task-graph level so ``Command.PARENT`` reaches the Agent graph.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        tools,
        handoffs,
        memory,
        callbacks=None,
        *,
        root_dir: str | None = None,
        inherit_env: bool = True,
        shell_timeout: int = 120,
        session_id: str | None = None,
        subagents: List[dict] | None = None,
        skills_zip_b64: List[str] | None = None,
    ):
        self._root_dir = root_dir or os.getcwd()
        self._inherit_env = inherit_env
        self._shell_timeout = shell_timeout
        self._session_id = session_id
        self._subagents = subagents
        self._skills_zip_b64 = skills_zip_b64

        self._transfer_allowed_tasks: List[str] = [
            t.task_config.name for t in (handoffs or [])
        ]

        task_config.task_type = "deep_agent"
        # super().__init__ calls _build_graph() which rebuilds deep_agent_graph
        super().__init__(task_config, tools, handoffs, memory, callbacks=callbacks)

    # ------------------------------------------------------------------
    # Transfer tool  (mirrors special_transfer_tool)
    # ------------------------------------------------------------------

    def _make_deep_transfer_tool(self):
        """Build a transfer tool for the deep agent.

        Uses ``return_direct=True`` so ``create_agent``'s routing exits
        the tool loop without an extra LLM call (first line of defense).
        Truncation in the chatbot node is the second line of defense for
        parallel-tool-call scenarios where ``return_direct`` is ignored.
        """
        allowed = list(self._transfer_allowed_tasks)

        @tool(return_direct=True)
        def deep_agent_transfer(
            id_: Annotated[str, "Domain to transfer to"],
        ):
            """Transfer conversation to a domain specialist."""
            all_allowed = allowed + ["<PARENT>", "<MANUAL_TRANSFER>"]
            if id_ not in all_allowed:
                return (
                    f"Invalid task: {id_}! "
                    f"Allowed tasks are: {','.join(all_allowed)}"
                )
            return f"Successfully transferred to {id_}"

        return deep_agent_transfer

    # ------------------------------------------------------------------
    # Transfer detection + Command construction
    # ------------------------------------------------------------------

    def _detect_transfer(self, messages) -> str | None:
        """Check if the most recent tool-calling AI message is a transfer.

        Only inspects the *last* AI message that has ``tool_calls`` —
        if that message doesn't contain ``deep_agent_transfer``, no
        transfer happened in this step.  Skips self-transfers (the call
        that handed control *to* this task).
        """
        for msg in reversed(messages):
            if not (isinstance(msg, AIMessage) and msg.tool_calls):
                continue
            for tc in msg.tool_calls:
                if tc.get("name") == DEEP_TRANSFER_TOOL_NAME:
                    dest = tc.get("args", {}).get("id_")
                    if dest == self.task_config.name:
                        return None
                    return dest
            return None
        return None

    @staticmethod
    def _truncate_after_transfer(messages):
        """Remove messages produced after the last transfer_tool response.

        The deep-agent loop may not terminate immediately when
        ``transfer_tool`` returns ``Command(goto=END)`` — any trailing
        AI messages are artifacts and must be discarded so only the
        destination task answers the user.

        Iterates in reverse to find the *last* transfer — earlier ones
        may have been rejected (invalid target) and retried.
        """
        last_transfer_ai_idx = None
        last_transfer_call_id = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == DEEP_TRANSFER_TOOL_NAME:
                        last_transfer_ai_idx = i
                        last_transfer_call_id = tc.get("id")
                        break
            if last_transfer_ai_idx is not None:
                break

        if last_transfer_ai_idx is None:
            return messages

        for i in range(last_transfer_ai_idx + 1, len(messages)):
            msg = messages[i]
            if getattr(msg, "tool_call_id", None) == last_transfer_call_id:
                return messages[: i + 1]

        return messages[: last_transfer_ai_idx + 1]

    @staticmethod
    def _build_transfer_command(state, new_messages, dest):
        """Build a ``Command`` that mirrors ``special_transfer_tool`` behaviour.

        Updates ``last_active_task`` the same way the real transfer tool
        does, then emits ``Command(graph=Command.PARENT)``.
        """
        updated = {**state, "messages": state["messages"] + new_messages}

        if dest == "<PARENT>":
            path = updated["last_active_task"]["path"]
            updated["last_active_task"] = {
                **updated["last_active_task"],
                "path": path[:-2],
            }
            target = PARENT_ROUTER_NODE
        elif dest == "<MANUAL_TRANSFER>":
            target = PARENT_ROUTER_NODE
        else:
            depth = updated["last_active_task"]["depth"]
            path = list(updated["last_active_task"]["path"])
            path[depth] = dest
            updated["last_active_task"] = {
                **updated["last_active_task"],
                "path": path,
            }
            target = dest

        return Command(update=updated, goto=target, graph=Command.PARENT)

    # ------------------------------------------------------------------
    # Skills ZIP upload
    # ------------------------------------------------------------------

    _SKILLS_DEST = "/skills"

    def _upload_and_extract_skills(self, backend) -> list[str]:
        """Extract skills ZIPs locally and upload individual files to the backend."""
        files = []
        for zip_b64 in self._skills_zip_b64 or []:
            try:
                zip_bytes = base64.b64decode(zip_b64)
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    for name in zf.namelist():
                        if name.endswith("/"):
                            continue
                        dest = f"{self._SKILLS_DEST}/{name}"
                        files.append((dest, zf.read(name)))
            except zipfile.BadZipFile:
                continue

        if not files:
            return []

        responses = backend.upload_files(files)
        failed = [r for r in responses if r.error]
        if failed:
            logger.warning("Skills upload partially failed: %s", [(r.path, r.error) for r in failed])
        return [f"{self._SKILLS_DEST}/"]

    # ------------------------------------------------------------------
    # Deep-agent construction
    # ------------------------------------------------------------------

    def _create_deep_agent(self, session_id: str | None = None):
        if _SANDBOX_ENABLED:
            from agent_builder.prebuilt_tasks.utils.remote_sandbox import RemoteSandboxBackend
            sandbox_name = f"{session_id}-sandbox" if session_id else self.task_config.name
            backend = RemoteSandboxBackend(
                sandbox_name=sandbox_name,
                sandbox_service_url=_SANDBOX_SERVICE_URL,
                default_timeout=self._shell_timeout,
            )
        else:
            backend = LocalShellBackend(
                root_dir=self._root_dir,
                virtual_mode=False,
                inherit_env=self._inherit_env,
                timeout=self._shell_timeout,
            )
        llm = SprinklrChatModel(llm_config=self.task_config.llm_config)

        if _COMPACTION_MAX_INPUT_TOKENS > 0:
            llm.profile = {"max_input_tokens": _COMPACTION_MAX_INPUT_TOKENS}

        all_tools = list(self._orig_tools)
        if self._transfer_allowed_tasks:
            all_tools.append(self._make_deep_transfer_tool())

        skills = None
        if self._skills_zip_b64:
            skills = self._upload_and_extract_skills(backend)

        kwargs = dict(
            model=llm,
            tools=all_tools,
            system_prompt=self.task_config.system_template,
            backend=backend,
            middleware=[TodoInjectionMiddleware()],
        )

        if self._subagents:
            kwargs["subagents"] = self._subagents
        if skills:
            kwargs["skills"] = skills

        return create_deep_agent(**kwargs)

    # ------------------------------------------------------------------
    # add_tools — intercept special_transfer_tool from Agent
    # ------------------------------------------------------------------

    def add_tools(self, new_tools: List):
        """Add tools at runtime.

        Intercepts ``special_transfer_tool`` (detected via its
        ``_allowed_tasks`` attr) and merges the targets into the
        deep agent's lightweight transfer tool.  All other tools
        are forwarded normally.
        """
        transfer_targets = []
        other_tools = []

        for t in new_tools:
            if hasattr(t, "_allowed_tasks"):
                transfer_targets.extend(t._allowed_tasks)
            else:
                other_tools.append(t)

        if transfer_targets:
            existing = set(self._transfer_allowed_tasks)
            self._transfer_allowed_tasks.extend(
                t for t in transfer_targets if t not in existing
            )

        if other_tools:
            existing_names = {t.name for t in self._orig_tools}
            self._orig_tools.extend(
                t for t in other_tools if t.name not in existing_names
            )

        # super().add_tools calls self._build_graph() which rebuilds deep_agent_graph
        super().add_tools(new_tools)

    # ------------------------------------------------------------------
    # Chatbot node — delegates entire turn to the deep agent
    # ------------------------------------------------------------------

    def _todos_key(self) -> str:
        if _TODO_SCOPE == "global":
            return "_todos:global"
        return f"_todos:{self.task_config.name}"

    def get_default_chatbot_node(self):
        task_name = self.task_config.name

        async def chatbot(state: State, config: RunnableConfig = None, **kwargs):
            input_messages = [
                m for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]

            todos_key = self._todos_key()
            saved_todos = state.get("config_variables", {}).get(todos_key, [])

            logger.info("[%s] invoke START | saved_todos=%s", task_name, saved_todos)

            invoke_input = {"messages": input_messages}

            _SAVED_TODOS.set(saved_todos)
            result = await self.deep_agent_graph.ainvoke(
                invoke_input,
                config=config,
            )

            new_messages = result["messages"][len(input_messages):]
            new_messages = [
                strip_ephemeral_metadata(m) if isinstance(m, AIMessage) else m
                for m in new_messages
            ]

            new_todos = result.get("todos") or saved_todos
            config_vars = {**state.get("config_variables", {}), todos_key: new_todos}

            logger.info("[%s] invoke END | updated_todos=%s", task_name, new_todos)

            dest = self._detect_transfer(new_messages)
            if dest:
                logger.info("[%s] transfer detected | dest=%s todos=%s", task_name, dest, new_todos)
                new_messages = self._truncate_after_transfer(new_messages)
                state_with_todos = {**state, "config_variables": config_vars}
                return self._build_transfer_command(state_with_todos, new_messages, dest)

            return {**state, "messages": state["messages"] + new_messages, "config_variables": config_vars}

        return chatbot

    # ------------------------------------------------------------------
    # System prompt — rebuild deep agent after update
    # ------------------------------------------------------------------

    def update_system_prompt(self, new_sys_template):
        super().update_system_prompt(new_sys_template)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Preprocessing — skip system prompt (deep agent owns it)
    # ------------------------------------------------------------------

    def preprocess_state(self, state, **kwargs):
        if self.task_config.preprocessor in preprocessors_dict:
            state = preprocessors_dict[self.task_config.preprocessor](state)
        return state

    # ------------------------------------------------------------------
    # Graph — single node, no outer ToolNode
    # ------------------------------------------------------------------

    def _build_graph(self, state_class=None):
        self.deep_agent_graph = self._create_deep_agent(self._session_id)

        if state_class is None:
            state_class = self._current_state_class

        builder = StateGraph(state_class)
        builder.add_node("chatbot", self._async_single_turn)
        builder.set_entry_point("chatbot")
        return builder.compile(checkpointer=self.memory)

    # ------------------------------------------------------------------
    # Deep-copy support
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for attr, value in self.__dict__.items():
            if attr in {"graph", "llm", "llm_node", "deep_agent_graph"}:
                continue
            setattr(new, attr, copy.deepcopy(value, memo))

        new.llm = SprinklrChatModel(
            llm_config=new.task_config.llm_config,
        ).bind_tools(new.tools)
        new.llm_node = new.get_default_chatbot_node()
        # _build_graph rebuilds deep_agent_graph internally
        new.graph = new._build_graph()
        return new
