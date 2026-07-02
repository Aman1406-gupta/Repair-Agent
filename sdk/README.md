# Agent Builder SDK

> Build intelligent, multi-task AI agents in Python with tools, routing, and state management.

The SDK is the engine that powers the [Agent Builder API](../README.md). You can also use it directly as a Python library to build agents programmatically — useful for local development, testing, notebooks, and embedding agents in your own services.

## Installation

```bash
pip install agent-builder \
  --index-url https://{NEXUS_USERNAME}:{NEXUS_PASSWORD}@prod-nexus.sprinklr.com/nexus/repository/spypi/simple \
  --extra-index-url https://pypi.python.org/simple/ \
  --no-cache-dir
```

---

## Concepts

The SDK is built around four core abstractions, composed bottom-up:

```
Tool → Task → Agent → Invoke
```

| Abstraction | SDK Class | What it does |
|------------|-----------|-------------|
| **Tool** | `StructuredTool` / `MockStructuredTool` | A callable function — real API or LLM-powered mock |
| **Task** | `Task` / `RemoteRelease` / `CodeTask` | A specialist with tools + system prompt + LLM. Handles one kind of conversation |
| **Agent** | `Agent` | Orchestrates multiple tasks with a router that delegates messages |
| **State** | `State` (TypedDict) | The conversation state that flows through the graph |

![SDK Architecture](../assets/sdk_composition.png)

---

## Quick Start

A minimal agent with one task, no external tools:

```python
from langchain_core.messages import HumanMessage

from agent_builder.base.agent import Agent
from agent_builder.base.task import Task
from agent_builder.base.configs import TaskConfig, AgentConfig, LLMConfig
from agent_builder.base.state import get_initial_state

task = Task(
    task_config=TaskConfig(
        name="assistant",
        description="General-purpose assistant",
        system_template="You are a helpful assistant.",
        llm_config=LLMConfig(model="gpt-4.1-2025-04-14"),
    ),
    tools=[],
    handoffs=[],
)

agent = Agent(
    agent_config=AgentConfig(name="my_agent"),
    tasks=[task],
)

state = get_initial_state(session_id="demo")
state["messages"] = [HumanMessage(content="Hello!")]

result = await agent.ainvoke(state)
print(result["messages"][-1].content)
```

---

## Tools

The SDK is built on LangGraph and LangChain, so **any tool that works with LangGraph works here** — including plain Python functions decorated with `@tool`, `StructuredTool` instances, or anything that conforms to LangChain's tool interface.

There are four ways to equip tasks with tools:

### 1. Any Python Function (LangGraph Native)

The simplest way — decorate any Python function with LangChain's `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def calculate_discount(price: float, percent: float) -> float:
    """Calculate the discounted price."""
    return price * (1 - percent / 100)

task = Task(
    task_config=TaskConfig(name="pricing", description="Handles pricing queries", system_template="..."),
    tools=[calculate_discount],
    handoffs=[],
)
```

You can also create tools from `StructuredTool` directly, use async functions, or wrap any callable — the SDK accepts anything LangGraph does. Tools can be added at runtime too via `task.add_tools([new_tool])`.

### 2. API Tools from OpenAPI Specs

Convert an OpenAPI spec into callable `StructuredTool` instances. Each operation becomes a tool that makes real HTTP requests.

```python
from agent_builder.base.tools import openapi_spec_to_tools

spec = {
    "openapi": "3.0.0",
    "info": {"title": "Weather", "version": "1.0"},
    "servers": [{"url": "https://api.weather.com"}],
    "paths": {
        "/forecast": {
            "get": {
                "operationId": "get_forecast",
                "summary": "Get weather forecast",
                "parameters": [
                    {"name": "city", "in": "query", "required": True, "schema": {"type": "string"}}
                ]
            }
        }
    }
}
tools = openapi_spec_to_tools(spec)
```

For multiple YAML files at once, use `openapi_yaml_list_to_tools(["spec1.yaml", "spec2.yaml"])`.

### 3. Mock Tools (LLM-Powered)

When you don't have a real API yet, `MockStructuredTool` uses an LLM to simulate responses based on a behavior description.

```python
from agent_builder.base.tools_mock import MockStructuredTool

tool = MockStructuredTool.from_openai_schema(
    schema={
        "name": "lookup_order",
        "description": "Look up an order by ID",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    behavior="Return a JSON order object with status 'shipped', estimated delivery in 3 days",
    llm_config=LLMConfig(model="gpt-4.1-2025-04-14"),
)
```

You can also create mock tools from OpenAPI operations with `MockStructuredTool.from_openapi_operation()`.

### 4. Tasks and Agents as Tools

Any task or agent can be exposed as a callable tool for other tasks:

```python
sub_task = Task(task_config=..., tools=[], handoffs=[])

parent_task = Task(
    task_config=TaskConfig(name="parent", description="...", system_template="..."),
    tools=[sub_task.as_tool()],
    handoffs=[],
)
```

When called as a tool, the sub-task runs in an isolated state — it only sees the `information_for_tool` string passed to it.

---

## Tasks

A Task is the core building block: an LLM with a system prompt and a set of tools, compiled into a LangGraph state machine.

### TaskConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | — | Task identifier |
| `description` | str | — | Used by the router to decide when to delegate to this task |
| `system_template` | str | `""` | System prompt that guides the LLM |
| `llm_config` | LLMConfig | defaults | Model configuration |
| `tool_keys` | list[str] | `[]` | Tool IDs (for DB-backed tasks) |
| `task_type` | str | `"normal"` | `"normal"`, `"release"`, `"remote_release"`, or `"code"` |
| `preprocessor` | str | `"DEFAULT"` | `"DEFAULT"`, `"CLEAR_ALL_MESSAGES"`, or `"KEEP_ONLY_LAST_MESSAGE"` |
| `postprocessor` | str | `"DEFAULT"` | Postprocessing strategy |

### Key Methods

```python
task.invoke(state)                    # Synchronous execution
await task.ainvoke(state)             # Async execution
task.add_tools([tool1, tool2])        # Add tools at runtime
task.update_system_prompt(template)   # Wrap existing prompt in a new template
task.as_tool()                        # Expose as a callable tool
task.compile_with_state(MyState)      # Rebuild graph with a custom state class
```

### Task Types

| Type | Class | When to Use |
|------|-------|------------|
| **Normal** | `Task` | Standard LLM + tools task |
| **Remote Release** | `RemoteRelease` | Proxy to a remote HTTP copilot endpoint |
| **Code** | `CodeTask` | Run a custom Python function instead of the default LLM loop |

#### RemoteRelease

A release task delegates conversation handling to a remote endpoint. Instead of running tools locally, it forwards the full state to the remote URL and incorporates the response.

```python
from agent_builder.prebuilt_tasks.remote_release import RemoteRelease

release_task = RemoteRelease(release_doc={
    "name": "external_copilot",
    "description": "Handles billing questions via external service",
    "http_config": {"url": "https://billing-copilot.internal/chat"},
    "task_form": "507f1f77bcf86cd799439011",
    "attributes": {},
})
```

Release tasks are read-only wrappers — `add_tools()` and `update_system_prompt()` are no-ops.

#### CodeTask

Replaces the default LLM-tool loop with a custom Python function. Useful for tasks that need deterministic logic, external SDKs, or custom orchestration.

```python
from agent_builder.prebuilt_tasks.code import CodeTask

async def my_logic(state, config=None):
    user_input = state["config_variables"].get("_input", "")
    state["config_variables"]["_output"] = f"Processed: {user_input}"
    return state

code_task = CodeTask(
    task_config=TaskConfig(name="processor", description="Custom processing"),
    chatbot_fn=my_logic,
)
```

When used via `as_tool()`, CodeTask reads from `config_variables["_input"]` and writes to `config_variables["_output"]`.

---

## Agents

An Agent orchestrates multiple tasks by routing conversations to the right specialist.

### AgentConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | — | Agent identifier |
| `description` | str | `""` | Agent description |
| `agent_type` | str | `""` | Must match registered agent-metadata name |
| `partner_id` | int | `0` | Partner ID for LLM tracking |
| `router_model_config` | LLMConfig | defaults | LLM config for the auto-generated router |
| `swarm_type` | str | `"router_back_connection"` | Topology (see below) |
| `task_keys` | list[str] | `[]` | Task IDs (for DB-backed agents) |
| `workflow_edges` | list[tuple] | `[]` | Explicit `(from_id, to_id)` edges |

### Swarm Topologies

![Agent Swarm Topologies](../assets/swarm_topologies.png)

| Topology | Behavior |
|----------|----------|
| `router_back_connection` | Router dispatches to tasks. Each task can hand back to the router. Most common pattern |
| `all_connected` | Every task can hand off to every other task. Router is still the entry point |
| `default` | Router dispatches to tasks. No back-connections |

For explicit wiring, use `workflow_edges`:
```python
agent_config = AgentConfig(
    name="pipeline_agent",
    workflow_edges=[("task_a_id", "task_b_id"), ("task_b_id", "task_c_id")],
)
```

### Key Methods

```python
agent.invoke(state)                   # Synchronous execution
await agent.ainvoke(state)            # Async execution
agent.add_tools([tool1, tool2])       # Add tools to ALL tasks
agent.compile_with_state(MyState)     # Rebuild with custom state (propagates to all tasks)
agent.as_task(task_name="wrapper")    # Convert agent into a Task for nesting
```

### Nesting Agents

Agents can be nested: wrap an inner agent as a task and include it in an outer agent.

```python
inner_agent = Agent(agent_config=AgentConfig(name="billing"), tasks=[billing_task, refund_task])
outer_agent = Agent(
    agent_config=AgentConfig(name="main"),
    tasks=[general_task, inner_agent.as_task(task_name="billing_dept", update_subtask_prompts=True)],
)
```

When `update_subtask_prompts=True`, inner tasks get instructions to call `transfer_tool` with `<PARENT>` to hand back to the outer agent's router.

### Custom Router

Instead of the auto-generated router, you can provide your own task as the router:

```python
custom_router = Task(
    task_config=TaskConfig(name="my_router", description="Custom routing logic", ...),
    tools=[],
    handoffs=[],
)
agent = Agent(agent_config=..., tasks=[task_a, task_b], use_task_as_router=custom_router)
```

---

## State

State is a `TypedDict` that flows through every node in the agent graph.

### Built-in Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list[AnyMessage] | Conversation messages |
| `session_id` | str | Session identifier |
| `request_id` | str | Request identifier for correlation |
| `timestamp` | datetime | Last update time |
| `log` | dict | Task-specific execution logs |
| `last_active_task` | dict | Tracks routing path and depth for multi-level agents |
| `config_variables` | dict | Arbitrary key-value store (used by CodeTask for `_input`/`_output`) |
| `remote_request` | dict | The original API request (for remote adapter roundtripping) |
| `remote_response` | dict | Response from remote endpoints (for release tasks) |

### Initialization

```python
from langchain_core.messages import HumanMessage
from agent_builder.base.state import get_initial_state

state = get_initial_state(session_id="user-123")
state["messages"] = [HumanMessage(content="Hello")]
```

### Custom State

Extend the base `State` with domain-specific fields. Custom state classes automatically propagate to all sub-tasks when compiled.

```python
from agent_builder.base.state import State

class EcommerceState(State):
    cart_items: list
    user_id: str

agent.compile_with_state(EcommerceState)

state = get_initial_state(session_id="user-123")
state["cart_items"] = []
state["user_id"] = "customer-456"
result = agent.invoke(state)
```

Custom fields use an **override reducer** at the agent boundary (the value returned by the last task wins), while base `State` fields keep their original reducers (e.g., messages are merged, timestamps take the latest).

---

## Streaming

Both `Task` and `Agent` support async streaming via the `StreamableMixin`:

```python
async for chunk in agent.astream(state, stream_mode=["messages"]):
    mode = chunk["stream_mode"]

    if mode == "messages":
        msg = chunk["message"]
        print(msg.content, end="", flush=True)

    elif mode == "values":
        full_state = chunk["value"]
```

| Stream Mode | What You Get |
|------------|-------------|
| `messages` | LLM token deltas (`AIMessageChunk`) and tool results (`ToolMessage`) as they happen |
| `values` | Full state snapshots after each graph node completes |

---

## Configuration

### LLMConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"gpt-4.1-2025-04-14"` | Model identifier |
| `provider` | str | `"AZURE_OPEN_AI"` | `AZURE_OPEN_AI`, `OPEN_AI`, `VERTEX`, `LOCAL`, `VOICE` |
| `temperature` | float | `0.1` | Sampling temperature |
| `max_tokens` | int | `4096` | Max output tokens |
| `top_p` | float | `1.0` | Nucleus sampling |
| `timeout` | int | `60` | Request timeout in seconds |
| `reasoning_effort` | str | `MINIMAL_REASONING` | Reasoning effort level |
| `pii_masking_templates` | list[str] | `[]` | PII masking template names |
| `guardrails` | list[str] | `[]` | Guardrail names to apply |
| `tracking_params` | dict | `{"release": "ca_research", "feature": "AGENT_BUILDER"}` | Tracking metadata |
| `partner_id` | int | `0` | Partner ID (overridden at runtime by agent metadata) |

---

## Preprocessors

Preprocessors transform the state before a task's LLM sees it. Set via `TaskConfig.preprocessor`.

| Name | Behavior |
|------|----------|
| `DEFAULT` | Pass through (no transformation) |
| `CLEAR_ALL_MESSAGES` | Remove all messages — task starts fresh |
| `KEEP_ONLY_LAST_MESSAGE` | Keep only the most recent message |

---

## Prebuilt Tasks

| Task | Module | Purpose |
|------|--------|---------|
| `ThinkActTask` | `prebuilt_tasks.think_act` | Two-phase reasoning: thinks first (generates a plan), then acts (executes with tools) |
| `RemoteRelease` | `prebuilt_tasks.remote_release` | Proxies conversation to a remote copilot endpoint |
| `CodeTask` | `prebuilt_tasks.code` | Runs a custom Python function instead of the LLM-tool loop |

### ThinkActTask

Implements a two-stage pattern for complex reasoning:

1. **Think phase**: Analyzes the conversation and generates an execution plan
2. **Act phase**: Executes the plan using tools and generates the response

```python
from agent_builder.prebuilt_tasks.think_act import ThinkActTask

task = ThinkActTask(
    task_config=TaskConfig(name="analyst", description="Deep analysis tasks"),
    tools=[search_tool, calculator_tool],
)
```
---

## Storage

Persistence and runtime object construction for Mongo-backed agents.

| Module | Purpose |
|--------|---------|
| `storage.mongo_client` | `AgentBuilderMongoStore` — MongoDB CRUD for agents, tasks, tools, metadata |
| `storage.redis_client` | Sessions, document caching, `last_active_task` persistence |
| `storage.registry` | `InMemoryRegistry` — in-memory agent/task/tool cache |
| `storage.utils.builders` | `build_task_from_doc`, `build_agent_from_doc` — Mongo docs → runtime objects |
| `storage.utils.mongo_topology` | Registration topology (embed tasks, tools, nested agents) |
| `storage.utils.state_serializer` | Serialize/deserialize graph state for Redis |

### Building from MongoDB

```python
from agent_builder.storage.utils.builders import build_agent_from_doc

agent = build_agent_from_doc(agent_doc, memory=None)
result = await agent.ainvoke(state)
```

### In-Memory Registry

The `InMemoryRegistry` caches built agents, tasks, and tools to avoid repeated MongoDB fetches and graph compilations.

```python
from agent_builder.storage.registry import InMemoryRegistry

registry = InMemoryRegistry()
# Fetches from Mongo, builds the agent, and caches it
agent = await registry.register_agent(agent_id="...", partner_id=66000000)

# Subsequent calls return the cached instance
cached_agent = registry.get_agent("...")
```

---

## MCP Client

The Model Context Protocol (MCP) lets you dynamically attach tools from remote servers. 

For SDK usage, you can connect directly to an MCP server, fetch its tools, and attach them to your tasks.

```python
import asyncio
from agent_builder.mcp_client.client import fetch_tools_for_connection

async def attach_mcp_tools(task):
    # 1. Define your MCP server connection
    connection_config = {
        "transport": "streamable_http",
        "url": "https://my-mcp-server/mcp",
        "headers": {"Authorization": "Bearer token"}
    }
    
    # 2. Fetch tools from the MCP server
    mcp_tools = await fetch_tools_for_connection(
        server_name="my_server",
        connection=connection_config
    )
    
    # 3. Attach them to your task
    task.add_tools(mcp_tools)
```

| Function | Purpose |
|----------|---------|
| `fetch_tools_for_connection` | `tools/list` against a single MCP server connection |
| `load_mcp_tools` | Fetch tools for all tasks under an agent (used by API) |
| `rebuild_mcp_tools` | Rebuild lazy MCP tool wrappers after cache refresh |

---

## Utilities Reference

For contributors and advanced users — a quick map of what each utility module does.

| Module | Purpose |
|--------|---------|
| `utils/misc.py` | System prompt helpers, message restore/strip, YAML loading |
| `utils/preprocessors.py` | State transformation functions |
| `utils/visualize.py` | Mermaid diagram generation, debug chat loops |
| `utils/log_context.py` | Correlation IDs (`session_id`, `request_id`) in log records |
| `utils/constants.py` | Collection names, field names, default configurations |
| `utils/openapi_utils.py` | OpenAPI spec parsing, Pydantic model generation |
| `utils/application.py` | `BaseHandler` / `AsyncApplication` (used by API handlers) |
| `utils/es_logging_handler.py` | Elasticsearch logging handler |

---

## Module Structure

```
agent_builder/
├── base/                     # Core abstractions
│   ├── agent.py              # Agent orchestration
│   ├── task.py               # Task graph
│   ├── state.py              # State TypedDict + reducers
│   ├── configs.py            # LLMConfig, TaskConfig, AgentConfig, HttpConfig
│   ├── tools.py              # OpenAPI → StructuredTool, transfer_tool
│   ├── tools_mock.py         # MockStructuredTool (LLM-powered)
│   └── streaming.py          # StreamableMixin
├── prebuilt_tasks/           # Specialized task types
│   ├── remote_release.py     # RemoteRelease
│   ├── think_act.py          # ThinkActTask
│   └── code.py               # CodeTask
├── llm_client/               # LLM clients and adapters
│   ├── sprinklr_chat_model.py
│   ├── remote_chat_model.py
│   ├── unified_client.py
│   ├── voicebot.py
│   └── utils/
├── handlers/                 # HTTP handlers (API layer)
│   ├── invoke.py
│   ├── base.py
│   ├── metadata.py
│   ├── list.py
│   ├── session.py
│   ├── conversation.py
│   ├── sync.py
│   └── core/
├── storage/                  # Persistence and object builders
│   ├── mongo_client.py
│   ├── redis_client.py
│   ├── registry.py
│   └── utils/
├── mcp_client/               # MCP tool discovery
│   └── client.py
└── utils/                    # Infrastructure and helpers
    ├── preprocessors.py
    ├── openapi_utils.py
    ├── visualize.py
    └── ...
```

---

## See Also

- [API README](../README.md) — for using Agent Builder via HTTP endpoints
