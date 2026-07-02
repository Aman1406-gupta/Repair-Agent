from typing import Any
from agent_builder.base.configs import TaskConfig
from agent_builder.base.task import Task
from langchain_core.tools import StructuredTool,tool
from agent_builder.base.state import get_initial_state
from langchain_core.runnables import RunnableConfig


import uuid
import inspect
import functools
import asyncio


class CodeTask(Task):
    def __init__(self, task_config: TaskConfig, tools=[], handoffs=[], memory=None, chatbot_fn=None, callbacks=None):
        super().__init__(task_config, tools, handoffs, memory, callbacks)
        task_config.task_type = "code"
        
        if chatbot_fn is None:
            self.llm_node = self.get_default_chatbot_node()
        else:
            self.llm_node = self._wrap_chatbot_fn(chatbot_fn)
    
    def _wrap_chatbot_fn(self, fn):
        """
        Return a version of *fn* that
        1. injects `self` iff *fn* has a parameter called 'self';
        2. injects `config` iff *fn* has a parameter called 'config'.
        Works for sync or async functions.
        """
        sig = inspect.signature(fn)
        params = sig.parameters

        needs_self   = 'self'   in params
        needs_config = 'config' in params

        @functools.wraps(fn)
        async def _wrapper(state: dict[str, Any], *args, config: RunnableConfig = None,
                           **kwargs):
            # Build positional/keyword arguments that *fn* expects
            call_args = []
            call_kwargs = {}

            if needs_self:
                call_args.append(self)           # first positional -> self

            call_args.append(state)             # state is always passed

            if needs_config and config is not None:
                call_kwargs['config'] = config

            # Forward any extra user kwargs that match fn’s signature
            for name in kwargs:
                if name in params:
                    call_kwargs[name] = kwargs[name]

            result = fn(*call_args, **call_kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        return _wrapper
    
    def as_tool(self, ) -> StructuredTool:
        task = self
        name = f"run_{task.task_config.name.replace(' ', '_')}"
        description = f"""Use the following sub-agent as tool/function: '{task.task_config.description}'. 
Pass the relevant information along with the query using the information_for_tool argument. 
Keep in mind that this tool will only have access to the information_for_tool information and nothing else."""

        async def _run(information_for_tool: str):
            sub_state = get_initial_state(session_id=str(uuid.uuid4()))
            sub_state['config_variables']['_input'] = information_for_tool
            output_state = await task.ainvoke(sub_state, config=None)
            return output_state['config_variables']['_output']
        _run.__name__ = name
        _run.__doc__ = description

        return tool(_run)
