import copy
import logging
import uuid
from typing import Annotated, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, StructuredTool, InjectedToolCallId
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langgraph.types import Command

from agent_builder.base.state import State, get_initial_state
from agent_builder.base.streaming import StreamableMixin
from agent_builder.utils.misc import add_system_prompt, restore_original_messages, strip_ephemeral_metadata
from agent_builder.utils.preprocessors import preprocessors_dict
from agent_builder.llm_client.sprinklr_chat_model import SprinklrChatModel
from langchain_core.runnables.config import merge_configs

logger = logging.getLogger(__name__)


class Task(StreamableMixin):
    def __init__(self, task_config, tools, handoffs, memory=None, callbacks=None):
        self.task_config = task_config

        self._orig_tools = tools or []
        self.handoff_targets = handoffs or []
        self.memory = memory
        self._handoff_tools: List[Tool] = [
            self._make_handoff_tool(target.task_config.name)
            for target in self.handoff_targets
        ]

        self.tools: List[Tool] = self._orig_tools + self._handoff_tools
        self.llm = SprinklrChatModel(llm_config=self.task_config.llm_config).bind_tools(self.tools)
        self.callbacks = callbacks or []

        self._current_state_class = State
        self.llm_node = self.get_default_chatbot_node()

        self.graph = self._build_graph()
        logger.debug("Task created | name=%s type=%s tools=%d", task_config.name, task_config.task_type, len(self.tools))

    def compile_with_state(self, state_class=None):
        """
        Rebuilds the graph with a custom state class
        """

        state_class = state_class if state_class is not None else State

        self._current_state_class = state_class
        self.graph = self._build_graph(state_class)

    def as_tool(self, ) -> StructuredTool:
        task = self
        name = f"run_{task.task_config.name.replace(' ', '_')}"
        description = f"""Use the following sub-agent as tool/function: '{task.task_config.description}'. 
Pass the relevant information along with the query using the information_for_tool argument. 
Keep in mind that this tool will only have access to the information_for_tool information and nothing else."""

        async def _run(information_for_tool: str):
            sub_state = get_initial_state(session_id=str(uuid.uuid4()))
            sub_state['messages'] = [{'role': 'user', 'content': information_for_tool}]
            tool_res = await task.ainvoke(sub_state, config=None)
            return {"tool_result": tool_res['messages'][-1].content}

        _run.__name__ = name
        _run.__doc__ = description

        return StructuredTool.from_function(
            coroutine=_run,
            extras={"nested_as_tool": task},
        )

    def add_tools(self, new_tools: List[StructuredTool]):  # ← add self
        """
        Add tools at runtime and rebuild the internal LLM binding + graph.
        """
        # Avoid duplicates
        existing_names = {t.name for t in self.tools}
        self.tools.extend(t for t in new_tools if t.name not in existing_names)

        # Re-bind LLM with the augmented toolset
        self.llm = SprinklrChatModel(llm_config=self.task_config.llm_config).bind_tools(self.tools)

        # Re-compile the task's private graph
        self.graph = self._build_graph()

    def update_system_prompt(self, new_sys_template):
        """ wraps the current system prompt in new template"""
        self.task_config.system_template = new_sys_template.format(self.task_config.system_template)

    def _make_handoff_tool(self, dest: str) -> StructuredTool:
        def _inner(state: Annotated[dict, InjectedState],
                  tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
            # To remove post debugging
            tool_message = {
            "role": "tool",
            "content": f"Successfully transferred control to task `{dest}`.",
            "tool_call_id": tool_call_id,
        }
            return Command(goto=dest, graph=Command.PARENT, update={"messages": state['messages'] + [tool_message]})

        _inner.__name__ = f"handoff_to_{dest}"
        _inner.__doc__ = f"Transfer control to Task `{dest}`."

        return tool(_inner)

    def preprocess_state(self, state, **kwargs):
        if self.task_config.preprocessor in preprocessors_dict:
            state = preprocessors_dict[self.task_config.preprocessor](state)
        state = add_system_prompt(state, self.task_config.system_template)
        return state

    def postprocess_state(self, original_messages, preprocessed_len, output_state, **kwargs):
        return restore_original_messages(original_messages, preprocessed_len, output_state)

    async def _async_single_turn(self, input_state, **kwargs):
        original_messages = input_state['messages']
        preprocessed_state = self.preprocess_state(input_state)
        preprocessed_len = len(preprocessed_state['messages'])
        output_state = await self.llm_node(preprocessed_state, **kwargs)
        return self.postprocess_state(original_messages, preprocessed_len, output_state)

    def _single_turn(self, input_state, **kwargs):
        original_messages = input_state['messages']
        preprocessed_state = self.preprocess_state(input_state)
        preprocessed_len = len(preprocessed_state['messages'])
        output_state = self.llm_node(preprocessed_state, **kwargs)
        return self.postprocess_state(original_messages, preprocessed_len, output_state)

    def invoke(self, input_state, config: RunnableConfig = None, **kwargs):
        # Merge callbacks into config if present
        if self.callbacks:
            config = merge_configs(config, RunnableConfig(callbacks=self.callbacks))
        return self.graph.invoke(input_state, config=config, **kwargs)

    async def ainvoke(self, input_state, config: RunnableConfig = None, **kwargs):
        # Merge callbacks into config if present
        if self.callbacks:
            config = merge_configs(config, RunnableConfig(callbacks=self.callbacks))
        return await self.graph.ainvoke(input_state, config=config , **kwargs)

    def _build_graph(self, state_class = None):

        if state_class is None:
            state_class = self._current_state_class
        
        graph_builder = StateGraph(state_class)
        graph_builder.add_node("chatbot", self._async_single_turn)

        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.set_entry_point("chatbot")

        return graph_builder.compile(checkpointer=self.memory)

    def get_default_chatbot_node(self):
        async def chatbot(state: self._current_state_class, **kwargs):
            new_message = await self.llm.ainvoke(state["messages"], **kwargs)
            new_message = strip_ephemeral_metadata(new_message)
            state['messages'].append(new_message)
            return state

        return chatbot

    @property
    def task_type(self) -> str:
        return self.task_config.task_type

    @task_type.setter
    def task_type(self, value: str):
        self.task_config.task_type = value

    def __deepcopy__(self, memo):
        """
        Custom deepcopy that skips the compiled sub-graph and the bound
        LLM (both hold non-copyable sentinels).  Everything else is
        copied, then the LLM is rebound and the private graph rebuilt.
        """
        cls = self.__class__
        new_task = cls.__new__(cls)
        memo[id(self)] = new_task

        for attr, value in self.__dict__.items():
            if attr in {"graph", "llm", "llm_node"}:  # <-- skip
                continue
            setattr(new_task, attr, copy.deepcopy(value, memo))

        # Re-bind the LLM with the copied tool list
        new_task.llm = SprinklrChatModel(
            llm_config=new_task.task_config.llm_config
        ).bind_tools(new_task.tools)
        if new_task.task_type == 'code':
            new_task.llm_node = self.llm_node
        else:
            new_task.llm_node = new_task.get_default_chatbot_node()

        # Build a fresh internal graph(uses the _current_state_class)
        new_task.graph = new_task._build_graph()
        return new_task

