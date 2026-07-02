import uuid
import copy
from dataclasses import asdict
from typing import List, Callable, Annotated, Any, get_args, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.tools import StructuredTool

from agent_builder.base.state import State, return_right
from agent_builder.base.streaming import StreamableMixin
from agent_builder.base.task import Task  # ← your Task class
from agent_builder.base.tools import special_transfer_tool  # ← default tool
from agent_builder.base.configs import TaskConfig, AgentConfig
from agent_builder.utils.misc import convert_objectid_to_str
from agent_builder.prebuilt_tasks.code import CodeTask
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from agent_builder.utils.constants import PARENT_ROUTER_NODE, TASK_COORDINATOR_PROMPT

try:
    from agent_builder.prebuilt_tasks.deep_agent import DeepAgentsTask
    _HAS_DEEP_AGENTS = True
except ImportError:
    _HAS_DEEP_AGENTS = False


import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                                 AGENT                                       #
# --------------------------------------------------------------------------- #
class Agent(StreamableMixin):
    """
    A thin orchestration wrapper around a collection of Task objects.
    It:
      1. builds a *router* task that is able to forward the conversation to any
         specialised task by calling the `transfer_tool`;
      2. wires every task (and the router) together inside one parent
         StateGraph;
      3. optionally (swarm_type='all_connected') equips *all* tasks with the
         capability (and prompting) to hand-off to any other task.
    """

    # ---------- construction ------------------------------------------------ #
    def __init__(
            self,
            agent_config: AgentConfig,
            tasks: List[Task],
            use_task_as_router: Task = None,  # optional parent-level check-pointer
            memory=None,  # optional parent-level check-pointer
            callbacks=None,  # optional callbacks for tracing/logging
    ):
        self.agent_config = agent_config
        self.memory = memory
        self.callbacks = callbacks or []

        self._current_state_class = State

        self.tasks = list(tasks)

        if use_task_as_router is None and len(self.tasks) == 1:
            use_task_as_router = self.tasks[0]

        if not use_task_as_router:
            self.agent_config.router_model_config.kwargs.update(parallel_tool_calls=False)

        if use_task_as_router:
            self.router_task_name = use_task_as_router.task_config.name
            self.router_task = use_task_as_router
            other_tasks = [t for t in self.tasks if t is not self.router_task]
            if other_tasks:
                specialist_desc = "\n".join(
                    f"{t.task_config.name}: {t.task_config.description}" for t in other_tasks
                )
                addon_prompt = (
                    "{}\n\n"
                    + TASK_COORDINATOR_PROMPT.format(specialist_tasks=specialist_desc)
                )
                self.router_task.update_system_prompt(addon_prompt)
                if not any(tool.name == "transfer_tool" for tool in self.router_task.tools):
                    self.router_task.add_tools([special_transfer_tool(allowed_tasks=[t.task_config.name for t in other_tasks])])
        else:
            self.router_task_name = "default_router_task"
            self.router_task = self._build_router_task()
        
        if self.router_task not in self.tasks:
            self.tasks.append(self.router_task)

        # Propagate callbacks to all tasks (including router task)
        for task in self.tasks:
            if not hasattr(task, 'callbacks') or not task.callbacks:
                task.callbacks = self.callbacks

        self.config = RunnableConfig(recursion_limit=25)

        if self.agent_config.swarm_type == "all_connected":
            self._enable_full_connectivity()
        elif self.agent_config.swarm_type == "router_back_connection":
            self._enable_router_connectivity()

        self.is_task_instance = False
        self.graph = self._build_graph()
        logger.debug("Agent created | name=%s swarm=%s tasks=%d connectivity=%s",
                      agent_config.name, agent_config.swarm_type, len(self.tasks),
                      "full" if agent_config.swarm_type == "all_connected" else "router_back")

    def compile_with_state(self, state_class=None):
        """
        Rebuild the agent's graph with a custom state class and propagate 
        the state class to all tasks within the agent.
        """
        state_class = state_class if state_class is not None else State
        self._current_state_class = state_class


        self.graph = self._build_graph(state_class)
    
    def add_tools(self, new_tools: List[StructuredTool]) -> None:
        """
        Add tools to *every* task that lives inside this agent and make sure the
        internal router gets them as well (if desired).
        """
        for t in self.tasks:
            t.add_tools(new_tools)

    def update_system_prompt(self, new_sys_template: str) -> None:
        """
        Wrap every task’s system prompt with the supplied template.
        """
        for t in self.tasks:
            t.update_system_prompt(new_sys_template)

    def invoke(self, state: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Entry-point called by the outside world.
        Simply forwards to the compiled LangGraph.
        """

        # Merge callbacks into config
        if self.callbacks:
            callback_config = RunnableConfig(callbacks=self.callbacks)
            config = merge_configs(self.config, callback_config, config)
        else:
            config = merge_configs(self.config, config)
        
        return self.graph.invoke(state, config=config, **kwargs)
        # return self.graph.invoke(state, **kwargs)

    async def ainvoke(self, state: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Entry-point called by the outside world.
        Simply forwards to the compiled LangGraph.
        """

        # Merge callbacks into config
        if self.callbacks:
            callback_config = RunnableConfig(callbacks=self.callbacks)
            config = merge_configs(self.config, callback_config, config)
        else:
            config = merge_configs(self.config, config)
        
        return await self.graph.ainvoke(state, config=config, **kwargs)
        # return await self.graph.ainvoke(state, **kwargs)
    
    def as_task(
        self,
        task_type: str = "normal",
        task_name: str = None,
        task_id: str = None,
        update_subtask_prompts: bool = False,
    ) -> "Task":
        """
        Return a Task instance that acts as a **proxy** to this agent.

        • Any `invoke/ainvoke` on the returned task are delegated to this
          Agent's own `invoke/ainvoke`.
        • A call to `add_tools` / `update_system_prompt` on the returned task
          simply forwards the request to *all* inner tasks.
        """
        agent_copy = self

        if update_subtask_prompts:
            agent_copy.router_task.update_system_prompt("{}\n[IMPORTANT] If NONE of the existing journeys match the user's query, please call the `transfer_tool` function with the special argument `<PARENT>`. This will hand the conversation back to the parent agent who will do appropriate routing.")
            agent_copy.router_task.task_config.description = f"{agent_copy.router_task.task_config.description}. This task can also hand the conversation back to the parent agent who will do appropriate routing. "
            
            for t in agent_copy.tasks:
                if t == agent_copy.router_task:
                    continue
                if self.agent_config.swarm_type == "all_connected":
                    t.update_system_prompt("{}\n[IMPORTANT] If none of the existing journeys match the user's query, please call the `transfer_tool` function with the special argument `<PARENT>`. This will hand the conversation back to the parent agent who will do appropriate routing.")
        
        agent_copy.is_task_instance = True        
        
        wrapper_cfg = TaskConfig(
            name         = task_name if task_name else f"{self.agent_config.name}_wrapper",
            description  = f"Composite task that internally runs the agent {self.agent_config.name}: {self.agent_config.description}",
            _id          = task_id if task_id else f"task_{str(uuid.uuid4())}",
        )
        
        if task_type == "code":
            wrapper_task = CodeTask(
                task_config = wrapper_cfg,
                tools       = [],
                handoffs    = [],          # the agent handles hand-offs
                memory      = self.memory
            )
        else:
            wrapper_task = Task(
                task_config = wrapper_cfg,
                tools       = [],
                handoffs    = [],          # the agent handles hand-offs
                memory      = self.memory
            )

        wrapper_task._current_state_class = agent_copy._current_state_class
        wrapper_task._build_graph = agent_copy._build_graph

        def _wrapper_compile_with_state(state_class=None):
            """Propagate state compilation to internal agent"""
            state_class = state_class if state_class is not None else agent_copy._current_state_class
            agent_copy.compile_with_state(state_class)
            wrapper_task.graph = agent_copy.graph
            wrapper_task.sub_tasks = agent_copy.tasks
            wrapper_task._current_state_class = state_class
        
        wrapper_task.compile_with_state = _wrapper_compile_with_state

        # ---------------- add_tools / update_system_prompt ------------- #
        # They forward to the *agent copy* which in turn broadcasts downwards.
        wrapper_task.add_tools = agent_copy.add_tools            # type: ignore
        wrapper_task.update_system_prompt = lambda x: None
        wrapper_task.task_config.task_type = "agent_wrapper"
        wrapper_task.sub_tasks = agent_copy.tasks

        wrapper_task.graph = wrapper_task._build_graph()

        return wrapper_task

    def _build_router_task(self) -> Task:
        """
        Creates a Task whose only purpose is to decide which specialist
        task should handle the user request and then call `transfer_tool`.
        """

        if self.tasks:
            specialist_desc = "\n".join(
                f"{t.task_config.name}: {t.task_config.description}" for t in self.tasks
            )
            router_prompt = TASK_COORDINATOR_PROMPT.format(specialist_tasks=specialist_desc)
            router_tools = [special_transfer_tool(allowed_tasks=[x.task_config.name for x in self.tasks])]
        else:
            router_prompt = TASK_COORDINATOR_PROMPT.format(specialist_tasks="None — no specialist tasks are configured.")
            router_tools = []

        router_task_cfg = TaskConfig(
            name=self.router_task_name,
            description="Top-level Default router that forwards the conversation to the correct journey.",
            system_template=router_prompt,
            llm_config=self.agent_config.router_model_config,
        )

        if _HAS_DEEP_AGENTS:
            router = DeepAgentsTask(
                task_config=router_task_cfg,
                tools=[],
                handoffs=[],
                memory=self.memory,
            )
            if router_tools:
                router.add_tools(router_tools)
            return router

        return Task(
            task_config=router_task_cfg,
            tools=router_tools,
            handoffs=[],
            memory=self.memory,
        )

    # ---------- all-connected mode --------------------------------------- #
    def _enable_full_connectivity(self):
        """
        Give every non-router task a) the transfer_tool and
        b) an augmented system prompt listing *other* journeys.
        """
        for task in self.tasks:
            if task is self.router_task:
                continue  # router already has the tool

            # Build a mapping that excludes the *current* task only
            mapping = "\n".join(
                f"{t.task_config.name}: {t.task_config.description}"
                for t in self.tasks if t is not task
            )
            allowed_tasks = [x.task_config.name for x in self.tasks if x is not task]

            addon_prompt = (
                "{}\n\n"
                "Moreover, you have access to other tasks or journeys as well, "
                "which are created to solve specific user issues. "
                "Following are the journey id/name and description:\n"
                f"{mapping}\n\n"
                "[IMPORTANT] If the user query is related to any journey "
                "description provided, please call the `transfer_tool` function "
                "with the proper journey name."
            )

            task.update_system_prompt(addon_prompt)
            task.add_tools([special_transfer_tool(allowed_tasks=allowed_tasks)])

    def _enable_router_connectivity(self):
        """
        Equip every specialist task with:
          • the `transfer_tool`
          • a short additive prompt that documents ONLY the router task
        """
        router_name = self.router_task.task_config.name
        router_description = self.router_task.task_config.description

        for task in self.tasks:
            if task is self.router_task:  # skip the router itself
                continue
            _tmp = f"{router_name}: {router_description}\n\n"

            # --- 1) prompt augmentation -------------------------------- #
            addon_prompt = (
                "{}\n\n"  # slot for the task's existing system prompt
                "You have the option to hand the conversation back to the "
                "central Router if you think another pathway is more suitable.\n\n"
                f"{router_name}: {router_description}\n\n"
                "[IMPORTANT] If you decide to delegate, call the "
                f"`transfer_tool` function with id_='{router_name}'."
            )
            task.update_system_prompt(addon_prompt)

            # --- 2) make sure the tool is present ---------------------- #
            task.add_tools([special_transfer_tool(allowed_tasks=[router_name])])

    def _create_agent_level_state(self, state_class):
        """
        Create agent-level state with ALL custom field reducers converted to override.
        
        This prevents double-accumulation at agent boundaries while allowing tasks
        to use their original reducers internally.
        """
        
        base_fields = set(State.__annotations__.keys())
        new_annotations = {}        
        for field_name, field_type in state_class.__annotations__.items():
            if field_name in base_fields:
                # Keep base State fields unchanged
                new_annotations[field_name] = field_type
            else:
                # Convert custom fields to override reducers
                if hasattr(field_type, '__metadata__'):
                    args = get_args(field_type)
                    base_type = args[0] if args else field_type
                    new_annotations[field_name] = Annotated[base_type, return_right]
                else:
                    # No reducer - keep as-is
                    new_annotations[field_name] = field_type
        
        #TypedDict with override reducers
        AgentState = TypedDict(
            f'Agent_{state_class.__name__}',
            new_annotations
        )
        
        return AgentState

    # ---------- parent graph --------------------------------------------- #
    def _build_graph(self, state_class=None):
        """
        Build the LangGraph that orchestrates the entire swarm of tasks.
        """
        if state_class is None:
            state_class = self._current_state_class

        # Compile all tasks with the current state class
        for task in self.tasks:
            task.compile_with_state(state_class)

        # Create agent-level state with override reducers for custom fields
        agent_state_class = self._create_agent_level_state(state_class)

        # Agent graph uses override reducers
        graph_builder = StateGraph(agent_state_class)
        key_to_task_dict = {}
        for t in self.tasks:
            key_to_task_dict[t.task_config._id] = t


        for t in self.tasks:
            node_name = t.task_config.name
            def _make_node(task: Task) -> Callable[[state_class], state_class]:
                async def _node(state: agent_state_class, config: RunnableConfig = None, **kwargs):
                    new_state = await task.ainvoke(state, config=merge_configs(self.config, config), **kwargs)
                    return new_state

                return _node

            graph_builder.add_node(node_name, _make_node(t))

        def _entry_passthrough(state: agent_state_class, **kwargs):
            """
            Does nothing: the *real* routing is performed by the conditional
            edge immediately after this node.
            """
            return state

        graph_builder.add_node("ENTRY", _entry_passthrough)
        graph_builder.add_node(PARENT_ROUTER_NODE, self.parent_router_node_producer(self.agent_config.name))

        task_names = {t.task_config.name for t in self.tasks}

        def _entry_condition(state: agent_state_class, **kwargs) -> str:
            """
            1. if `last_active_task` is set and we know that task → jump there;
            2. otherwise start at the router.
            """
            depth = state.get("last_active_task").get("depth")
            dest_path = state.get("last_active_task").get("path")
            if (not self.is_task_instance) or (len(dest_path) == 0):
                depth = 0 
            else:
                depth = depth + 1
            state['last_active_task']['depth'] = depth
            
            if len(dest_path) < depth + 1:
                dest_path.append(self.router_task_name)
                state['last_active_task']['path'] = dest_path
            
            dest = dest_path[depth]
            logger.debug("Entry routing | agent=%s dest=%s depth=%d", self.agent_config.name, dest, depth)
            return dest

        graph_builder.add_conditional_edges("ENTRY", _entry_condition,{k:k for k in list(set(list(task_names)+[self.router_task_name]))})

        graph_builder.set_entry_point("ENTRY")

        for (t1,t2) in self.agent_config.workflow_edges:
            if t1 not in key_to_task_dict or t2 not in key_to_task_dict:
                raise ValueError("Task keys not attached with this task")
            logger.debug("Adding workflow edge | %s -> %s", key_to_task_dict[t1].task_config.name, key_to_task_dict[t2].task_config.name)
            graph_builder.add_edge(key_to_task_dict[t1].task_config.name , key_to_task_dict[t2].task_config.name)


        graph = graph_builder.compile(checkpointer=self.memory)
        return graph

    def __deepcopy__(self, memo):
        """
        Custom deepcopy that avoids cloning the compiled LangGraph.

        1. Copy every attribute except `graph`.
        2. Deep-copy the list of tasks – each Task has its own
           __deepcopy__ below and will rebuild its private graph.
        3. Finally build a *new* parent graph that wires the copied
           tasks together.
        """
        cls = self.__class__
        new_agent = cls.__new__(cls)
        memo[id(self)] = new_agent

        for attr, value in self.__dict__.items():
            if attr == "graph":  # <-- skip compiled graph
                continue
            setattr(new_agent, attr, copy.deepcopy(value, memo))

        # Re-compile a new parent graph for the clone
        new_agent.graph = new_agent._build_graph(self._current_state_class)
        return new_agent

    def to_dict(self) -> dict:
        """
        Convert the Agent instance into a simple dictionary.
        """

        # Collect normal tasks (skip router_task)
        tasks_data = []
        for task in self.tasks:
            if task == self.router_task:
                continue

            task_dict = {
                "task_config": convert_objectid_to_str(asdict(task.task_config)),
                "tools_metadata": convert_objectid_to_str(getattr(task, '_tools_metadata', [])),
                "task_type": getattr(task, 'task_type', 'normal'),
                "has_transfer_tool": any(tool.name == "transfer_tool" for tool in task.tools)
            }
            tasks_data.append(task_dict)

        # Collect router task (only if it's a custom_router)
        router_task_data = None
        if getattr(self.router_task, 'task_type', None) == 'custom_router':
            router_task_data = {
                "task_config": convert_objectid_to_str(asdict(self.router_task.task_config)),
                "tools_metadata": convert_objectid_to_str(getattr(self.router_task, '_tools_metadata', [])),
                "task_type": "custom_router"
            }

        # Return final dictionary
        return {
            "agent_config": convert_objectid_to_str(asdict(self.agent_config)),
            "tasks": tasks_data,
            "router_task": router_task_data,
            "memory": None
        }


    @staticmethod
    def parent_router_node_producer(agent_name: str):
        def parent_router_node(state: dict[str, Any], **kwargs):
            logger.debug("Parent router | agent=%s last_active=%s", agent_name, state['last_active_task'])
            depth = state['last_active_task']['depth']
            path = state['last_active_task']['path']
            
            if len(path) == 0 and depth == 0:
                return Command(update=state, goto = 'ENTRY')
            
            
            if len(path) < depth + 1:
                state['last_active_task']['depth'] -= 1
                return Command(update=state, goto=PARENT_ROUTER_NODE, graph=Command.PARENT)
            
            contains_end = False
            while path and path[-1] == END:
                contains_end = True
                path.pop()
                depth -= 1
            target_node = END if contains_end else path[-1]
                
            return Command(update=state, goto=target_node)
        return parent_router_node