"""
Shared fixtures for all tests in the agent_builder test suite.

This module consolidates common fixtures to avoid duplication across test files.
"""

# Register pytest plugins early to avoid PytestAssertRewriteWarning
pytest_plugins = ["pytest_asyncio", "anyio"]

import pytest
import uuid
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Dict, Any, List
from unittest.mock import MagicMock, AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent_builder.base.configs import LLMConfig, TaskConfig
from agent_builder.base.state import State, get_initial_state, return_right, with_nonetype_check
from agent_builder.base.task import Task
from agent_builder.prebuilt_tasks.code import CodeTask


# ============================================================================
# LLM CONFIG FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def base_llm_config():
    """
    Standard LLM config for testing.
    Uses gpt-4o-mini via the Sprinklr LLM router.
    """
    return LLMConfig(
        model="gpt-4.1-mini-2025-04-14",
        provider="AZURE_OPEN_AI",
        temperature=0.1,
        max_tokens=1024,
        partner_id=66000000,
        llm_router_url="intuitionx-llm-router-v2.qa6-k8singress-intuition-gke.sprinklr.com",
        # llm_router_url="qa6-intuitionx-llm-router-v2.sprinklr.com",
        tracking_params={"release": "ca_research", "feature": "AGENT_BUILDER"},
        timeout=60
    )


@pytest.fixture(scope="module")
def advance_llm_config():
    """
    Advanced LLM config for tool calling and structured output tests.
    Uses gpt-4o for better accuracy with tool calls.
    """
    return LLMConfig(
        model="gpt-4.1-2025-04-14",
        provider="AZURE_OPEN_AI",
        temperature=0.0,
        max_tokens=1024,
        partner_id=66000000,
        llm_router_url="intuitionx-llm-router-v2.qa6-k8singress-intuition-gke.sprinklr.com",
        # llm_router_url="qa6-intuitionx-llm-router-v2.sprinklr.com",
        tracking_params={"release": "ca_research", "feature": "AGENT_BUILDER"},
        timeout=60
    )


@pytest.fixture(scope="module")
def local_llm_config():
    """
    Config for the 'LOCAL' provider.
    Requires a live, running service URL for tests to pass.
    """
    return LLMConfig(
        provider="LOCAL",
        model="local-model",
        llm_router_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8080"),
        timeout=30
    )


@pytest.fixture(scope="module")
def voice_llm_config():
    """
    Config for the 'VOICE' provider.
    Requires a valid OPENAI_API_KEY environment variable.
    """
    return LLMConfig(provider="VOICE", model="gpt-realtime")


# ============================================================================
# TOOL FIXTURES
# ============================================================================

@pytest.fixture
def weather_tool():
    """Weather tool that returns deterministic result."""
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location.
        
        Args:
            location: The city name
        """
        return f"The weather in {location} is sunny and 72°F"
    return get_weather


@pytest.fixture
def booking_tool():
    """Booking tool that returns deterministic result."""
    @tool
    def book_appointment(date: str, time: str) -> str:
        """Book an appointment.
        
        Args:
            date: The date for the appointment
            time: The time for the appointment
        """
        return f"Appointment booked for {date} at {time}"
    return book_appointment


@pytest.fixture
def calculator_tool():
    """Calculator tool for testing math operations."""
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression.
        
        Args:
            expression: The mathematical expression to evaluate
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    return calculate


# ============================================================================
# TASK CONFIG FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def base_task_config(base_llm_config):
    """Base task configuration for testing"""
    return TaskConfig(
        name="test_task",
        description="A test task that answers questions concisely",
        system_template="You are a helpful assistant. Be concise and direct.",
        llm_config=base_llm_config,
        preprocessor="DEFAULT"
    )


# ============================================================================
# TASK FIXTURES
# ============================================================================

@pytest.fixture
def weather_task(base_llm_config, weather_tool):
    """A weather specialist task with tool."""
    config = TaskConfig(
        name="weather_task",
        description="Handles weather queries",
        system_template="You are a weather specialist.",
        llm_config=base_llm_config
    )
    return Task(task_config=config, tools=[weather_tool], handoffs=[], memory=None)


@pytest.fixture
def booking_task(base_llm_config, booking_tool):
    """A booking specialist task with tool."""
    config = TaskConfig(
        name="booking_task",
        description="Handles appointment booking",
        system_template="You are a booking assistant.",
        llm_config=base_llm_config
    )
    return Task(task_config=config, tools=[booking_tool], handoffs=[], memory=None)


@pytest.fixture
def simple_task(base_llm_config):
    """A simple task without tools."""
    config = TaskConfig(
        name="simple_task",
        description="Handles general questions",
        system_template="You are a helpful assistant.",
        llm_config=base_llm_config
    )
    return Task(task_config=config, tools=[], handoffs=[], memory=None)


@pytest.fixture
def code_task(base_llm_config):
    """A CodeTask with custom chatbot function."""
    config = TaskConfig(
        name="code_task",
        description="Executes code-based logic",
        system_template="You are a code executor.",
        llm_config=base_llm_config
    )
    
    # Custom chatbot function that returns deterministic output
    async def custom_chatbot(state: Dict[str, Any]) -> Dict[str, Any]:
        ret_state = deepcopy(state)
        ret_state['messages'] = ret_state['messages'] + [
            AIMessage(content="Code task executed successfully")
        ]
        ret_state['config_variables']['_output'] = "code_result"
        return ret_state
    
    return CodeTask(
        task_config=config,
        tools=[],
        handoffs=[],
        memory=None,
        chatbot_fn=custom_chatbot
    )


# ============================================================================
# STATE FIXTURES
# ============================================================================

@pytest.fixture
def simple_state():
    """Simple conversation state for testing."""
    session_id = str(uuid.uuid4())
    state = get_initial_state(session_id)
    state['messages'] = [HumanMessage(content="Hello, how are you?")]
    return state


# ============================================================================
# OPENAPI / SCHEMA FIXTURES
# ============================================================================

@pytest.fixture
def openai_schema():
    """Standard OpenAI function schema for testing."""
    return {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["city"]
        }
    }


@pytest.fixture
def openapi_spec():
    """OpenAPI spec with GET and POST operations for testing."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users/{userId}": {
                "get": {
                    "operationId": "getUser",
                    "summary": "Get user by ID",
                    "parameters": [
                        {"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}}
                    ]
                }
            },
            "/users": {
                "post": {
                    "operationId": "createUser",
                    "summary": "Create user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"}
                                    },
                                    "required": ["name"]
                                }
                            }
                        }
                    }
                }
            }
        }
    }


# ============================================================================
# STATE HELPER FUNCTIONS
# ============================================================================

def make_state_with_tool_call(tool_name: str, args: dict, call_id: str = "call_1"):
    """Helper to create state with a tool call."""
    state = get_initial_state(str(uuid.uuid4()))
    state['messages'] = [
        HumanMessage(content="test"),
        AIMessage(content="", tool_calls=[
            {"name": tool_name, "args": args, "id": call_id}
        ])
    ]
    return state


def make_state_with_transfer(target: str, tool_call_id: str = "call_1"):
    """Helper to create state with a transfer tool call."""
    state = get_initial_state(str(uuid.uuid4()))
    state['last_active_task'] = {'path': ['current'], 'depth': 0}
    state['messages'] = [
        HumanMessage(content="test"),
        AIMessage(content="", tool_calls=[
            {"name": "transfer_tool", "args": {"id_": target}, "id": tool_call_id}
        ])
    ]
    return state


# ============================================================================
# MOCKING HELPERS
# ============================================================================

def mock_task_to_call_transfer(task, target_task_name: str, fallback_response: str = "Done"):
    """
    Mock a task's LLM to call transfer_tool targeting another task.
    
    Args:
        task: The task to mock
        target_task_name: Target task for transfer_tool
        fallback_response: If provided, returns this on subsequent calls to prevent loops
    """
    transfer_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "transfer_tool",
            "args": {"id_": target_task_name},
            "id": "mock_transfer_call"
        }]
    )
    
    if fallback_response is not None:
        call_count = 0
        fallback = AIMessage(content=fallback_response)
        
        async def mock_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return transfer_response
            return fallback
    else:
        async def mock_ainvoke(*args, **kwargs):
            return transfer_response
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = mock_ainvoke
    task.llm = mock_llm
    task.llm_node = task.get_default_chatbot_node()
    task.graph = task._build_graph()
    
    return mock_llm


def mock_router_to_transfer(agent, target_task_name: str):
    """
    Mock the router's LLM to force a transfer_tool call to the target task.
    Returns the mock so it can be verified.
    """
    mock_llm = mock_task_to_call_transfer(agent.router_task, target_task_name, fallback_response=None)
    
    # Rebuild agent graph to pick up the mocked router
    agent.graph = agent._build_graph()
    
    return mock_llm


def mock_task_llm_response(task, response):
    
    # Normalize response(s) to AIMessage
    def to_ai_message(r):
        return AIMessage(content=r) if isinstance(r, str) else r
    
    if isinstance(response, list):
        normalized = [to_ai_message(r) for r in response]
    else:
        normalized = to_ai_message(response)
    
    mock_llm = MagicMock()
    
    # Handle both single response and sequence of responses
    if isinstance(normalized, list):
        mock_llm.ainvoke = AsyncMock(side_effect=normalized)
    else:
        mock_llm.ainvoke = AsyncMock(return_value=normalized)
    
    task.llm = mock_llm
    task.llm_node = task.get_default_chatbot_node()
    task.graph = task._build_graph()
    return mock_llm


@dataclass
class CounterObj:
    """Custom dataclass for testing counter reducers."""
    value: int = 0


@with_nonetype_check
def merge_nested_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that deeply merges nested dictionaries.
    
    Handles the case where complete states are passed by checking for identity.
    If the same object is returned (left is right), no merge is performed.
    """
    if left is right:
        return right
    out = deepcopy(left)
    for k, v in right.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_nested_dict(out[k], v)
        else:
            out[k] = v
    return out



@with_nonetype_check
def append_list(left: List[Any], right: List[Any]) -> List[Any]:
    """Reducer that appends lists together.
    
    Handles the case where complete states are passed by checking for identity.
    If the same list object is returned (left is right), no append is performed.
    """
    if left is right:
        return right
    return left + right


@with_nonetype_check
def add_counter(left: CounterObj, right: CounterObj) -> CounterObj:
    """Reducer that adds counter values.
    
    Handles the case where complete states are passed by checking for identity.
    If the same counter object is returned (left is right), no addition is performed.
    """
    if left is right:
        return right
    return CounterObj(left.value + right.value)


class CustomState(State):
    """Custom state class with custom fields for testing."""
    analytics: Annotated[Dict[str, Any], merge_nested_dict]
    # user_profile: Annotated[UserProfile, merge_user_profile]
    action_history: Annotated[List[str], append_list]
    step_counter: Annotated[int, return_right]
    counter: Annotated[CounterObj, add_counter]
    history: Annotated[List[str], append_list]


# ============================================================================
# CUSTOM STATE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def custom_state_cls():
    """Custom state class fixture."""
    return CustomState


@pytest.fixture
def custom_initial_state():
    """Factory fixture that returns a function to create custom state instances."""
    def _factory(session_id: str = None):
        sid = session_id or str(uuid.uuid4())
        base = get_initial_state(sid)
        base.update({
            'analytics': {"metrics": {}, "events": []},
            # 'user_profile': UserProfile("guest", "Guest"),
            'action_history': [],
            'step_counter': 0,
            'counter': CounterObj(0),
            'history': [],
        })
        return base
    return _factory

