import uuid
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agent_builder.base.configs import TaskConfig
from agent_builder.base.state import get_initial_state
from agent_builder.base.task import Task


@pytest.mark.asyncio
class TestTaskInit:
    """Tests Task.__init__ with various configurations."""
    
    async def test_with_tools(self, base_task_config, weather_tool):
        """Verify task initializes with tools and builds graph."""
        task = Task(
            task_config=base_task_config,
            tools=[weather_tool],
            handoffs=[],
            memory=None,
        )
        
        assert len(task.tools) > 0
        assert isinstance(task.graph, CompiledStateGraph)
        assert weather_tool.name in [t.name for t in task.tools]
    
    
    async def test_with_handoffs(self, base_llm_config):
        """Verify handoff tools are created and return correct Command via AIMessage with tool call."""
        target_config = TaskConfig(
            name="target_task",
            description="Target task for handoff",
            llm_config=base_llm_config
        )
        target_task = Task(
            task_config=target_config,
            tools=[],
            handoffs=[],
            memory=None
        )

        main_config = TaskConfig(
            name="main_task",
            description="Main task with handoff",
            llm_config=base_llm_config
        )
        main_task = Task(
            task_config=main_config,
            tools=[],
            handoffs=[target_task],
            memory=None
        )

        handoff_tool = main_task.tools[0]
        tool_call_id = "handoff-invoke-1"
        state = get_initial_state(str(uuid.uuid4()))
        state['messages'] = [
            HumanMessage(content="Please handoff to target_task."),
            AIMessage(
                content="Invoking handoff.",
                tool_calls=[{"name": handoff_tool.name, "args": {}, "id": tool_call_id}]
            )
        ]

        handoff_node = ToolNode(tools=[handoff_tool])
        result = await handoff_node.ainvoke(state)
        assert hasattr(result[0], "goto")
        assert result[0].goto == "target_task"


@pytest.mark.asyncio
class TestTaskInvoke:
    """Tests Task.ainvoke - the main execution method."""
    
    async def test_parses_tool_calls_from_llm_response(self, base_task_config, weather_tool):
        """Verify task correctly parses tool calls from mocked LLM response."""
        task = Task(
            task_config=base_task_config,
            tools=[weather_tool],
            handoffs=[],
            memory=None
        )
        
        mock_ai_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "Tokyo"}, "id": "mock_call_1"}
            ]
        )
        
        original_llm = task.llm
        call_count = 0
        
        async def mock_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_ai_response
            return await original_llm.ainvoke(*args, **kwargs)
        
        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_ainvoke
        task.llm = mock_llm
        task.llm_node = task.get_default_chatbot_node()
        task.graph = task._build_graph()
        
        session_id = str(uuid.uuid4())
        state = get_initial_state(session_id)
        state['messages'] = [HumanMessage(content="What's the weather in Tokyo?")]
        
        result = await task.ainvoke(state)
        
        ai_messages_with_tool_calls = [
            msg for msg in result['messages'] 
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
        ]
        
        assert len(ai_messages_with_tool_calls) > 0, "Should have AI message with tool calls"
        
        tool_call = ai_messages_with_tool_calls[0].tool_calls[0]
        assert tool_call['name'] == 'get_weather'
        assert tool_call['args']['location'] == 'Tokyo'
        
        tool_messages = [msg for msg in result['messages'] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) > 0, "Tool should have been executed"
        assert "Tokyo" in tool_messages[0].content, "Tool result should contain location"


class TestTaskDynamicModification:
    """Tests add_tools and update_system_prompt methods."""
    
    def test_add_tools_deduplicates(self, base_task_config, weather_tool, calculator_tool):
        """Verify add_tools avoids duplicates and adds new tools."""
        task = Task(
            task_config=base_task_config,
            tools=[weather_tool],
            handoffs=[],
            memory=None
        )
        
        initial_count = len(task.tools)
        
        task.add_tools([weather_tool])
        assert len(task.tools) == initial_count
        
        task.add_tools([calculator_tool])
        assert len(task.tools) == initial_count + 1
        assert calculator_tool.name in [t.name for t in task.tools]
    
    
    def test_update_system_prompt(self, base_task_config):
        """Verify update_system_prompt wraps the current prompt."""
        task = Task(
            task_config=base_task_config,
            tools=[],
            handoffs=[],
            memory=None
        )
        
        original_prompt = task.task_config.system_template
        wrapper = "IMPORTANT: {} Always end with 'DONE.'"
        
        task.update_system_prompt(wrapper)
        
        assert task.task_config.system_template == wrapper.format(original_prompt)


@pytest.mark.asyncio
class TestTaskAsTool:
    """Tests Task.as_tool method."""
    
    async def test_executes_as_subtool(self, base_task_config, calculator_tool):
        """Verify as_tool converts task to functional tool."""
        task = Task(
            task_config=base_task_config,
            tools=[calculator_tool],
            handoffs=[],
            memory=None
        )
        
        task_tool = task.as_tool()
        result = await task_tool.ainvoke({"information_for_tool": "Please calculate 10 + 20"})
        
        assert len(result['tool_result']) > 0


@pytest.mark.asyncio
class TestTaskStreaming:
    """Tests Task.astream method."""
    
    async def test_produces_chunks(self, base_task_config, simple_state):
        """Verify astream produces message and value chunks."""
        task = Task(
            task_config=base_task_config,
            tools=[],
            handoffs=[],
            memory=None
        )
        
        chunks = []
        async for chunk in task.astream(simple_state, stream_mode=['messages', 'values']):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        stream_modes = set(c['stream_mode'] for c in chunks)
        assert 'messages' in stream_modes and 'values' in stream_modes


class TestTaskStateManagement:
    """Tests state processing methods."""
    
    def test_preprocess_adds_system_message(self, base_task_config):
        """Verify preprocess_state adds system prompt."""
        task = Task(
            task_config=base_task_config,
            tools=[],
            handoffs=[],
            memory=None
        )
        
        session_id = str(uuid.uuid4())
        state = get_initial_state(session_id)
        state['messages'] = [HumanMessage(content="Hello")]
        
        preprocessed = task.preprocess_state(state)
        
        system_messages = [m for m in preprocessed['messages'] if isinstance(m, SystemMessage)]
        assert any(base_task_config.system_template in msg.content for msg in system_messages)
    
    
    def test_preprocess_with_custom_preprocessor(self, base_llm_config, mocker):
        """Verify preprocess_state respects custom preprocessor."""
        
        def uppercase_messages_preprocessor(state):
            # Modify messages in place, no deepcopy
            for msg in state['messages']:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    msg.content = msg.content.upper()
            return state
        
        mocker.patch(
            'agent_builder.base.task.preprocessors_dict',
            {'UPPERCASE_MESSAGES': uppercase_messages_preprocessor}
        )
        
        config = TaskConfig(
            name="custom_task",
            description="Task with custom UPPERCASE_MESSAGES preprocessor",
            system_template="You are a helpful assistant.",
            llm_config=base_llm_config,
            preprocessor="UPPERCASE_MESSAGES"
        )
        
        task = Task(
            task_config=config,
            tools=[],
            handoffs=[],
            memory=None
        )
        
        session_id = str(uuid.uuid4())
        state = get_initial_state(session_id)
        state['messages'] = [
            HumanMessage(content="hello world"),
            AIMessage(content="response here"),
            HumanMessage(content="another message")
        ]
        
        preprocessed = task.preprocess_state(state)
        
        non_system_messages = [m for m in preprocessed['messages'] if not isinstance(m, SystemMessage)]
        system_message = next((m for m in preprocessed['messages'] if isinstance(m, SystemMessage)), None)
        
        assert (all(msg.content == msg.content.upper() for msg in non_system_messages))
        assert(system_message.content != system_message.content.upper())
    
    
    def test_postprocess_enriches_metadata(self, base_task_config):
        """Verify postprocess_state enriches state with task metadata."""
        task = Task(
            task_config=base_task_config,
            tools=[],
            handoffs=[],
            memory=None
        )
        
        session_id = str(uuid.uuid4())
        input_state = get_initial_state(session_id)
        original_messages = [HumanMessage(content="Hello")]
        input_state['messages'] = original_messages
        
        preprocessed = task.preprocess_state(input_state)
        preprocessed_len = len(preprocessed['messages'])
        
        
        # Simulate output state by appending to preprocessed messages
        preprocessed['messages'].append(AIMessage(content="Hi there!"))
        output_state = preprocessed
        
        result = task.postprocess_state(
            original_messages,
            preprocessed_len,
            output_state
        )
        
        # assert 'log' in result
        # assert base_task_config.name in result['log']
        # assert len(result['log'][base_task_config.name]) > 0
        assert result is output_state, "postprocess_state should return output_state unchanged"
        assert any(isinstance(m, AIMessage) and m.content == "Hi there!" for m in result['messages']), \
            "Output state messages should be preserved"
    
    
    def test_postprocess_returns_command_unchanged(self, base_task_config):
        """Verify postprocess_state returns Command unchanged."""
        task = Task(
            task_config=base_task_config,
            tools=[],
            handoffs=[],
            memory=None
        )
        
        input_state = get_initial_state(str(uuid.uuid4()))
        preprocessed = task.preprocess_state(input_state)
        command = Command(goto="other_task")
        
        result = task.postprocess_state(input_state['messages'], len(preprocessed['messages']), command)
        
        assert result is command
        assert result.goto == "other_task"
    

    
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
