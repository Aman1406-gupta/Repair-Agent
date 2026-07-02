"""
Tests for custom state handling in Tasks and Agents.

Tests focus on:
- Task/Agent compile_with_state and state class propagation
- Custom state fields with reducers (merge, append, add, return_right)
- State preservation through handoffs, workflow edges, and transfers
- CodeTask integration with custom state
- Agent.as_task wrapper state handling
"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent_builder.base.agent import Agent
from agent_builder.base.configs import AgentConfig, TaskConfig
from agent_builder.base.state import State
from agent_builder.base.task import Task
from agent_builder.prebuilt_tasks.code import CodeTask
from agent_builder.tests.conftest import (
    CounterObj,
    mock_task_llm_response,
    mock_router_to_transfer,
    mock_task_to_call_transfer,
)

CUSTOM_STATE_FIELDS = {'analytics', 'action_history', 'step_counter', 'counter', 'history'}


@pytest.mark.asyncio
class TestCustomStateForTasks:
    async def test_compile_with_state_and_none(self, base_task_config, custom_state_cls):
        """
        Verify compile_with_state correctly sets the internal state class,
        and compile_with_state(None) resets to default State class.
        """
        task = Task(task_config=base_task_config, tools=[], handoffs=[], memory=None)
        assert task._current_state_class is State
        
        task.compile_with_state(custom_state_cls)
        assert task._current_state_class is custom_state_cls
        
        task.compile_with_state(None)
        assert task._current_state_class is State

    async def test_invoke_with_reducers_and_state_mutation(self, base_task_config, custom_state_cls, custom_initial_state):
        """
        Verify custom state reducers are applied during task.ainvoke.
        Tests merge_nested_dict, add_counter, append_list, and return_right reducers.
        """
        cfg = deepcopy(base_task_config)
        cfg.name = "reducer_test_task"
        
        task = Task(task_config=cfg, tools=[], handoffs=[], memory=None)
        task.compile_with_state(custom_state_cls)
        
        async def custom_llm_node(state, config=None, **kwargs):
            out = deepcopy(state)
            out['messages'] = out['messages'] + [AIMessage(content="LLM node response")]
            out['analytics'] = {"metrics": {"from_llm": 42}}
            out['counter'] = CounterObj(7)
            out['history'] = ["llm_action"]
            out['step_counter'] = 99
            out['action_history'] = out['action_history'] + ["llm_processed"]
            return out
        
        task.llm_node = custom_llm_node
        task.graph = task._build_graph()
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Test LLM reducers")],
            'analytics': {"metrics": {"existing": 10}},
            'counter': CounterObj(3),
            'history': ["initial"],
            'step_counter': 1,
            'action_history': ["setup"],
        })
        
        result = await task.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys()), \
            f"Missing fields: {CUSTOM_STATE_FIELDS - set(result.keys())}"
        
        assert len(result['messages']) >= 2
        assert any("LLM node response" in m.content for m in result['messages'] if hasattr(m, 'content'))
        
        assert result['analytics']['metrics'] == {"existing": 10, "from_llm": 42}  # Merged
        assert result['counter'].value == 10  # 3 + 7 = 10
        assert set(result['history']) == {"initial", "llm_action"}  # Appended
        assert result['step_counter'] == 99  # Replaced (return_right)
        assert set(result['action_history']) == {'setup', 'llm_processed'}  # Appended

    async def test_code_task_with_reducers(self, base_task_config, custom_state_cls, custom_initial_state):
        """
        Verify CodeTask correctly applies custom state reducers.
        Tests that counter values are added and history lists are appended.
        """
        cfg = deepcopy(base_task_config)
        cfg.name = "code_task_reducer_test"
        
        async def code_fn(state: custom_state_cls):
            out = deepcopy(state)
            out['analytics'] = {"metrics": {"code_executed": True}}
            out['counter'] = CounterObj(10)
            out['history'] = ["code_action"]
            
            out['messages'].append(AIMessage(content="code done"))
            return out
        
        code_task = CodeTask(cfg, tools=[], handoffs=[], memory=None, chatbot_fn=code_fn)
        code_task.compile_with_state(custom_state_cls)
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="run code")],
            'analytics': {"metrics": {"existing": 1}},
            'counter': CounterObj(5),
            'history': ["initial"],
        })
        
        result = await code_task.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        assert result['counter'].value == 15  # 5 + 10
        assert set(result['history']) == {"initial", "code_action"}  # Appended
        assert result['analytics']['metrics']['code_executed'] == True  # Merged
        
        assert any("code done" in m.content for m in result['messages'] if hasattr(m, 'content'))

    async def test_task_with_handoff_preserves_custom_state(self, base_task_config, custom_state_cls, custom_initial_state):
        """
        Verify custom state fields are preserved when routing through handoffs.
        Flow: router -> target_task, checking analytics and action_history persist.
        """
        target_config = deepcopy(base_task_config)
        target_config.name = "target_task"
        target_config.description = "Handles target requests"
        target_task = Task(task_config=target_config, tools=[], handoffs=[], memory=None)
        
        # Create source task
        source_config = deepcopy(base_task_config)
        source_config.name = "source_task"
        source_config.description = "Handles source requests"
        source_task = Task(task_config=source_config, tools=[], handoffs=[], memory=None)
        
        # Create agent with both tasks
        agent_config = AgentConfig(
            name="test_handoff_agent",
            description="Agent for testing handoffs"
        )
        agent = Agent(agent_config=agent_config, tasks=[source_task, target_task])
        agent.compile_with_state(custom_state_cls)
        
        # Mock router to transfer to target_task
        router_response = AIMessage(
            content="",
            tool_calls=[{
                "name": "transfer_tool",
                "args": {"id_": "target_task"},
                "id": "router_call_1"
            }]
        )
        mock_task_llm_response(agent.router_task, router_response)
        mock_task_llm_response(target_task, "Target task executed successfully")
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Route me to target")],
            'analytics': {"metrics": {"calls": 1}, "events": ["request"]},
            'action_history': ["init", "authenticate"],
            'counter': CounterObj(7),
            'step_counter': 3,
        })
        
        result = await agent.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        assert result['analytics']['metrics']['calls'] == 1
        assert set(result['action_history']) == {'init', 'authenticate'}
        
        message_contents = [m.content for m in result['messages'] if hasattr(m, 'content')]
        assert any("Target task executed" in c for c in message_contents)

    async def test_tool_command_updates_state(self, base_task_config, custom_state_cls, custom_initial_state):
        """
        Verify tools returning Command can update custom state fields via reducers.
        Tests that counter, history, and analytics are correctly merged from tool Commands.
        """
        cfg = deepcopy(base_task_config)
        cfg.name = "command_tool_test_task"
        
        @tool
        def state_modifier_tool(
            increment: Annotated[int, "Value to add to counter"],
            new_history_entry: Annotated[str, "Entry to add to history"],
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId] = None
        ):
            """Modifies custom state fields and returns updated state via Command."""
            updated_state = deepcopy(state)
            updated_state['counter'] = CounterObj(increment)
            updated_state['history'] = [new_history_entry]
            updated_state['analytics'] = {"metrics": {"tool_modified": True}}
            
            tool_message = {
                "role": "tool",
                "content": f"State modified: counter +{increment}, history: {new_history_entry}",
                "tool_call_id": tool_call_id,
            }
            if tool_call_id:
                updated_state['messages'] = updated_state['messages'] + [tool_message]
            
            return Command(
                update=updated_state
            )
        
        task = Task(task_config=cfg, tools=[state_modifier_tool], handoffs=[], memory=None)
        task.compile_with_state(custom_state_cls)
        
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{
                "name": "state_modifier_tool",
                "args": {"increment": 25, "new_history_entry": "tool_executed"},
                "id": "tool_call_1"
            }]
        )
        final_response = AIMessage(content="State modification complete")
        mock_task_llm_response(task, [tool_call_response, final_response])
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Modify my state")],
            'analytics': {"metrics": {"queries": 1}},
            'counter': CounterObj(10),
            'history': ["initial"],
        })
        
        result = await task.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        assert result['counter'].value == 35  # 10 + 25 via add_counter
        assert "tool_executed" in result['history']  # Appended via append_list
        assert result['analytics']['metrics'].get('tool_modified') == True  # Merged
        
        tool_messages = [m for m in result['messages'] if isinstance(m, ToolMessage)]
        assert len(tool_messages) > 0


@pytest.mark.asyncio
class TestCustomStateForAgents:

    async def test_compile_with_state_and_none_propagates_to_all_tasks(self, base_llm_config, weather_task, booking_task, custom_state_cls):
        """
        Verify compile_with_state correctly sets state class on agent and all tasks,
        and compile_with_state(None) resets everything to default State class.
        """
        config = AgentConfig(
            name="compile_test_agent",
            description="Agent for compile testing",
            swarm_type="default"
        )
        
        agent = Agent(agent_config=config, tasks=[weather_task, booking_task], memory=None)
        assert agent._current_state_class is State
        for task in agent.tasks:
            assert task._current_state_class is State
        
        agent.compile_with_state(custom_state_cls)
        assert agent._current_state_class is custom_state_cls
        for task in agent.tasks:
            assert task._current_state_class is custom_state_cls
        
        agent.compile_with_state(None)
        assert agent._current_state_class is State
        for task in agent.tasks:
            assert task._current_state_class is State

    async def test_custom_state_flows_through_workflow_edges(self, base_llm_config, custom_state_cls, custom_initial_state):
        """
        Verify custom state correctly flows through sequential workflow edges.
        Task A -> Task B (via workflow edge)
        """
        task_a_config = TaskConfig(
            name="task_a",
            description="First task in workflow",
            llm_config=base_llm_config
        )
        task_a = Task(task_config=task_a_config, tools=[], handoffs=[], memory=None)
        
        task_b_config = TaskConfig(
            name="task_b",
            description="Second task in workflow",
            llm_config=base_llm_config
        )
        task_b = Task(task_config=task_b_config, tools=[], handoffs=[], memory=None)
        
        config = AgentConfig(
            name="workflow_custom_state_agent",
            description="Agent with workflow edges and custom state",
            workflow_edges=[(task_a.task_config._id, task_b.task_config._id)],
            swarm_type="default"
        )
        
        agent = Agent(agent_config=config, tasks=[task_a, task_b], memory=None)
        agent.compile_with_state(custom_state_cls)
        
        mock_router_to_transfer(agent, "task_a")
        mock_task_llm_response(task_a, "Task A complete")
        mock_task_llm_response(task_b, "Task B complete")
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Start workflow")],
            'counter': CounterObj(100),
            'history': ["workflow_start"],
            'step_counter': 0,
        })
        
        result = await agent.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        # assert 'task_a' in result['log']
        # assert 'task_b' in result['log']
        ai_contents = [m.content for m in result['messages'] if isinstance(m, AIMessage)]
        assert any("Task A complete" in c for c in ai_contents), "task_a should have processed messages"
        assert any("Task B complete" in c for c in ai_contents), "task_b should have processed messages"
        
        assert result['counter'].value == 100
        assert 'workflow_start' in result['history']
    
    async def test_internal_task_transfers_preserve_custom_state(self, base_llm_config, custom_state_cls, custom_initial_state):
        """
        Verify custom state is preserved when one task internally transfers to 
        another task using transfer_tool in all_connected swarm mode.
        """
        task_x_config = TaskConfig(
            name="task_x",
            description="First task",
            llm_config=base_llm_config
        )
        task_x = Task(task_config=task_x_config, tools=[], handoffs=[], memory=None)
        
        task_y_config = TaskConfig(
            name="task_y",
            description="Second task",
            llm_config=base_llm_config
        )
        task_y = Task(task_config=task_y_config, tools=[], handoffs=[], memory=None)
        
        config = AgentConfig(
            name="all_connected_custom_agent",
            description="All-connected agent with custom state",
            swarm_type="all_connected"
        )
        
        agent = Agent(agent_config=config, tasks=[task_x, task_y], memory=None)
        agent.compile_with_state(custom_state_cls)
        
        # Start at task_x, have it transfer to task_y
        mock_task_to_call_transfer(task_x, "task_y")
        mock_task_llm_response(task_y, "Task Y executed")
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Process")],
            'last_active_task': {'path': ['task_x'], 'depth': 0},
            'analytics': {"metrics": {"processed": True}},
            'counter': CounterObj(25),
        })
        
        result = await agent.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        # assert 'task_x' in result['log']
        # assert 'task_y' in result['log']
        assert any(isinstance(m, AIMessage) and "Task Y executed" in m.content for m in result['messages']), \
            "task_y should have executed after transfer from task_x"
        assert 'task_y' in result['last_active_task']['path']
        
        assert result['analytics']['metrics']['processed'] == True
        assert result['counter'].value == 25
    
    async def test_as_task_wrapper_preserves_custom_state(self, base_llm_config, custom_state_cls, custom_initial_state):
        """
        Verify as_task() wrapper correctly handles custom state:
        1. compile_with_state propagates to internal agent and all tasks
        2. State mutations from inner task's llm_node are applied via reducers
        """
        inner_config = TaskConfig(
            name="inner_task",
            description="Inner task of wrapped agent",
            llm_config=base_llm_config
        )
        inner_task = Task(task_config=inner_config, tools=[], handoffs=[], memory=None)

        async def tampering_llm_node(state, config=None, **kwargs):
            out = deepcopy(state)
            out['messages'] = out['messages'] + [AIMessage(content="Inner task tampered response")]
            out['counter'] = CounterObj(50)
            out['history'] = ["tampered_by_inner"]
            out['step_counter'] = 777
            return out
        
        inner_task.llm_node = tampering_llm_node
        inner_task.graph = inner_task._build_graph()

        config = AgentConfig(
            name="wrappable_agent",
            description="Agent to be wrapped as task",
            swarm_type="default"
        )
        
        agent = Agent(agent_config=config, tasks=[inner_task], memory=None)
        # mock_router_to_transfer(agent, "inner_task")

        wrapper_task = agent.as_task(task_name="agent_as_task")

        wrapper_task.compile_with_state(custom_state_cls)
        assert wrapper_task._current_state_class is custom_state_cls
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Execute wrapped agent")],
            'counter': CounterObj(33),
            'analytics': {"metrics": {"wrapper_test": True}},
            'history': ["initial"],
            'step_counter': 1,
        })
        
        result = await wrapper_task.ainvoke(state)
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        assert result['counter'].value == 83  # 33 + 50 via add_counter
        assert 'tampered_by_inner' in result['history']
        assert result['step_counter'] == 777  # Replaced via return_right
        assert result['analytics']['metrics']['wrapper_test'] == True
    
    async def test_as_task_inherits_state_from_precompiled_agent(self, base_llm_config, custom_state_cls, custom_initial_state):
        """
        Verify that when an agent is compiled with custom state BEFORE calling as_task(),
        the resulting wrapper task automatically inherits that state class.
        """
        task_config = TaskConfig(
            name="precompiled_task",
            description="Task for precompiled agent test",
            llm_config=base_llm_config
        )
        task = Task(task_config=task_config, tools=[], handoffs=[], memory=None)
        mock_task_llm_response(task, "Precompiled task response")
        
        config = AgentConfig(
            name="precompiled_agent",
            description="Agent compiled before as_task",
            swarm_type="default"
        )
        
        agent = Agent(agent_config=config, tasks=[task], memory=None)
        # mock_router_to_transfer(agent, "precompiled_task")
        
        agent.compile_with_state(custom_state_cls)
        
        wrapper_task = agent.as_task(task_name="inherited_wrapper")
        
        assert wrapper_task._current_state_class is custom_state_cls
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Test inherited state")],
            'counter': CounterObj(42),
            'history': ["inherited_test"],
        })
        
        result = await wrapper_task.ainvoke(state)
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
                
    async def test_code_task_in_agent_with_custom_state(self, base_llm_config, custom_state_cls, custom_initial_state):
        """
        Verify CodeTask sandwiched between normal tasks correctly handles custom state.
        Flow: router -> pre_task -> code_task -> post_task (via workflow edges)
        """
        pre_config = TaskConfig(
            name="pre_task",
            description="Task before CodeTask",
            llm_config=base_llm_config
        )
        pre_task = Task(task_config=pre_config, tools=[], handoffs=[], memory=None)
        
        code_config = TaskConfig(
            name="code_task",
            description="CodeTask that modifies custom state",
            llm_config=base_llm_config
        )
        
        async def code_fn(state):
            out = deepcopy(state)
            out['counter'] = CounterObj(100)
            out['history'] = out['history'] + ["code_executed"]
            out['step_counter'] = 999
            out['messages'].append(AIMessage(content="Code task done"))
            return out
        
        code_task = CodeTask(code_config, tools=[], handoffs=[], memory=None, chatbot_fn=code_fn)
        
        post_config = TaskConfig(
            name="post_task",
            description="Task after CodeTask",
            llm_config=base_llm_config
        )
        post_task = Task(task_config=post_config, tools=[], handoffs=[], memory=None)
        
        config = AgentConfig(
            name="sandwich_agent",
            description="Agent with CodeTask sandwiched between normal tasks",
            workflow_edges=[
                (pre_task.task_config._id, code_task.task_config._id),
                (code_task.task_config._id, post_task.task_config._id)
            ],
            swarm_type="default"
        )
        
        agent = Agent(agent_config=config, tasks=[pre_task, code_task, post_task], memory=None)
        
        mock_router_to_transfer(agent, "pre_task")
        mock_task_llm_response(pre_task, "Pre task complete")
        mock_task_llm_response(post_task, "Post task complete")
        
        agent.compile_with_state(custom_state_cls)
        
        state = custom_initial_state()
        state.update({
            'messages': [HumanMessage(content="Run sandwich workflow")],
            'counter': CounterObj(50),
            'history': ["before_workflow"],
            'step_counter': 1,
        })
        
        result = await agent.ainvoke(state)
        
        assert CUSTOM_STATE_FIELDS <= set(result.keys())
        
        # assert 'pre_task' in result['log']
        # assert 'code_task' in result['log']
        # assert 'post_task' in result['log']
        ai_contents = [m.content for m in result['messages'] if isinstance(m, AIMessage)]
        assert any("Pre task complete" in c for c in ai_contents), "pre_task should have processed messages"
        assert any("Code task done" in c for c in ai_contents), "code_task should have processed messages"
        assert any("Post task complete" in c for c in ai_contents), "post_task should have processed messages"
        
        # Verify CodeTask modified state correctly
        assert result['counter'].value == 150  # 50 + 100 via add_counter
        assert "code_executed" in result['history']
        assert result['step_counter'] == 999


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
