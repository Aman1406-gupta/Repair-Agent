# Agent Builder

> Build, compose, and invoke multi-task AI agents through a REST + SSE API.

Agent Builder is an API-first backend for orchestrating AI workflows. You register **agents** with embedded **tasks** (specialist conversation handlers) and **tools** (from OpenAPI specs or LLM-powered prompts), then **invoke** agents over HTTP or **stream** responses via Server-Sent Events (SSE).

If you're looking to use Agent Builder as a Python library instead of via the API, see the [SDK documentation](sdk/README.md).

---

## Concepts

Agent Builder uses a **unified registration model** — everything is embedded on the agent document in a single `POST /register/agents` call:

```
Agent
 ├── Tasks (embedded)
 │    ├── Tools (embedded)
 │    ├── task_as_tools (nested tasks)
 │    └── agent_as_tools (cross-agent refs)
 ├── agent_as_task (nested agents)
 └── task_as_router (custom router)
```

| Concept | What it is | Analogy |
|---------|-----------|---------|
| **Tool** | A callable function — either a real API endpoint (from an OpenAPI spec) or an LLM-powered mock (prompt tool). Embedded on tasks | A single skill |
| **Task** | A tool set + system prompt + LLM config. Represents a specialist that handles one kind of conversation. Embedded on agents | A department |
| **Agent** | A group of tasks + a router that decides which task handles each message. Supports multiple wiring topologies | A company |
| **Agent Metadata** | Deployment-level config: which `featureId` to track, which partner, context management settings. Pre-registered for each deployment | The company's operating license |

### LLM Configuration

Each task and agent accepts an `llm_config` field as an optional dict. To use a platform-managed configuration, include `llm_configuration_id` as a key in the dict alongside any other parameters:

```json
{"model": "gpt-4.1", "provider": "AZURE_OPEN_AI", "llm_configuration_id": "platform-cfg-12345", ...}
```

When `llm_configuration_id` is set, the LLM router uses the platform-managed config; the remaining fields (model, provider, partner_id, etc.) are still needed for client setup. When `llm_config` is omitted, defaults apply at runtime.

### Standard Tasks (Release Tasks)

A **standard task** doesn't run tools locally — it proxies the conversation to a **remote HTTP endpoint** (another copilot service). Registered as part of the agent with `http_config` instead of `system_template`/`tools`.

### Swarm Topologies

When an agent has multiple tasks, the **swarm type** controls how tasks hand off conversations:

| Swarm Type | Behavior |
|-----------|----------|
| `router_back_connection` (default) | Central router dispatches to tasks. Each task can hand back to the router only |
| `all_connected` | Every task can hand off to any other task directly |
| `default` | Router dispatches to tasks. No back-connections |

For explicit control, use `workflow_edges` to define custom `[from_task, to_task]` pairs.

### Composing Entities

Agent Builder supports nesting and cross-referencing at every level:

| Feature | Where | What it does |
|---------|-------|-------------|
| `task_as_tools` | Task (embedded) | Expose other tasks as callable tools within a task |
| `agent_as_tools` | Task (embedded) | Expose entire agents as callable tools within a task (embedded agent docs) |
| `agent_as_task` | Agent registration | Nest an agent inside another agent as if it were a task |
| `task_as_router` | Agent registration | Replace the auto-generated router with a custom task that you control |

### MCP Tools

At invoke time, the API can dynamically attach **Model Context Protocol (MCP)** tools discovered from partner-specific MCP servers. MCP integration is handled by the `mcp_client/` module and is transparent to registration — tools are loaded automatically when `AGENT_BUILDER_MCP_CONFIG_BASE_URL` is configured.

---

## Quick Start

Register an agent with embedded tasks and tools, then invoke it — all in two API calls:

```python
import requests

BASE = "http://<service-url>"

# 1. Register an agent with embedded tasks and tools
agent = requests.post(f"{BASE}/register/agents", json={
    "name": "research_agent",
    "partner_id": 66000000,
    "agent_type": "MY_COPILOT",
    "swarm_type": "router_back_connection",
    "llm_config": {
        "model": "gpt-4.1-2025-04-14",
        "provider": "AZURE_OPEN_AI",
        "partner_id": 66000000,
    },
    "tasks": [
        {
            "name": "search_task",
            "description": "Researches information using the search tool",
            "system_template": "You are a research assistant. Use the search tool to find information.",
            "llm_config": {
                "model": "gpt-4.1-2025-04-14",
                "provider": "AZURE_OPEN_AI",
                "partner_id": 66000000,
            },
            "tools": [
                {
                    "openapi_schema": {
                        "openapi": "3.0.0",
                        "info": {"title": "Search API", "version": "1.0"},
                        "servers": [{"url": "https://api.example.com"}],
                        "paths": {
                            "/search": {
                                "get": {
                                    "operationId": "search",
                                    "summary": "Search for information",
                                    "parameters": [
                                        {"name": "query", "in": "query", "required": True, "schema": {"type": "string"}}
                                    ]
                                }
                            }
                        }
                    }
                }
            ],
        },
        {
            "name": "general_task",
            "description": "Handles general questions",
            "system_template": "You are a helpful assistant.",
            "tools": [],
        },
    ],
}).json()

agent_id = agent["agent_id"]

# 2. Invoke the agent
response = requests.post(f"{BASE}/invoke", json={
    "agentId": agent_id,
    "partnerId": 66000000,
    "version": 0,
    "sessionId": "session-001",
    "id": "request-001",
    "conversationState": "stateless",
    "messages": [
        {
            "role": "user",
            "content": [{"type": "input.text", "text": "What is LangGraph?"}],
        }
    ],
}).json()

print(response["text"])
```

---

## API Reference

### Route Overview

| Method | Endpoint | Purpose |
|--------|---------|---------|
| `POST` | `/register/agents` | Register an agent with embedded tasks, tools, and nested agents |
| `POST` | `/register/agent-metadata` | Register deployment metadata |
| `POST` | `/update/agents` | Update an agent |
| `POST` | `/update/agent-metadata` | Update metadata |
| `POST` | `/invoke` | Invoke an agent (sync JSON, SSE, or webhook) |
| `GET` | `/session` | Generate a new session ID |
| `POST` | `/session/clone` | Clone an existing session |
| `GET` | `/agents` | List or fetch agents |
| `GET` | `/agent-metadata` | Get metadata |
| `POST` | `/platform/sync/agent` | Sync an agent from the external platform |
| `POST` | `/conversation/interrupt` | Interrupt an in-progress conversation |
| `POST` | `/log-level` | Change server log level at runtime |

---

### Registration Endpoints

#### `POST /register/agents`

Register an agent with all tasks, tools, and nested entities embedded in a single request.

**Top-level fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Agent name |
| `partner_id` | int | Yes | Partner ID for LLM tracking |
| `agent_type` | string | Yes | Must match a registered agent-metadata `name` |
| `description` | string | No | Agent description |
| `llm_config` | object \| string | No | Router LLM config — inline dict or `llm_configuration_id` string |
| `tasks` | TaskInput[] | No | Embedded task definitions (see below) |
| `swarm_type` | string | No | `router_back_connection` (default), `all_connected`, or `default` |
| `workflow_edges` | [string, string][] | No | Explicit edges as `[from_task_id, to_task_id]` pairs |
| `agent_as_task` | RegisterAgentRequest[] | No | Nested agent definitions (recursive) |
| `task_as_router` | TaskInput | No | Custom router task definition |

**Embedded task fields (custom task):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Task name |
| `description` | string | Yes | Short description (used by the router to decide when to delegate) |
| `system_template` | string | Yes | System prompt that guides the task's LLM |
| `llm_config` | object \| string | No | Task LLM config — inline dict or `llm_configuration_id` string |
| `tools` | ToolInput[] | No | Embedded tool definitions (OpenAPI or prompt tools) |
| `preprocessor` | string | No | `DEFAULT`, `CLEAR_ALL_MESSAGES`, or `KEEP_ONLY_LAST_MESSAGE` |
| `postprocessor` | string | No | Postprocessing strategy |
| `task_as_tools` | TaskInput[] | No | Nested task definitions exposed as tools |
| `agent_as_tools` | RegisterAgentRequest[] | No | Embedded agent definitions exposed as tools |

**Embedded task fields (release/standard task — detected when `http_config` is present):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Task name |
| `description` | string | Yes | Short description |
| `http_config` | object | Yes | `{"url": "...", "proxy_server": "...", "proxy_port": "..."}` |
| `task_form` | string | No | Opaque reference for external consumers |
| `attributes` | object | No | Optional key-value metadata |

**Embedded tool fields (API tool):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `openapi_schema` | object | Yes | OpenAPI 3.0+ or Swagger 2.0 spec |
| `filter_name` | string | No | Only register the operation matching this `operationId` |

**Embedded tool fields (prompt tool):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `openapi_schema` | object | One of these | Tool schema in OpenAPI format |
| `openai_schema` | object | One of these | Tool schema in OpenAI function format |
| `llm_behavior` | string | No | Natural language instruction for how the LLM should behave |
| `llm_config` | object | Yes | Model configuration for the prompt tool |

**Response:**
```json
{
  "success": true,
  "agent_id": "68a6c70375e584a205fc66e4",
  "task_ids": ["68a6c7038880c2ba6d46b9ea", "68a6c7038880c2ba6d46b9eb"],
  "tool_ids": ["68a6c703e57bf28c58d97242"],
  "router_task_id": "68a6c7038880c2ba6d46b9ec"
}
```

#### `POST /register/agent-metadata`

Register deployment-level metadata. The `name` here is what you set as `agent_type` on your agent.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Metadata identifier (must match `agent_type` on the agent) |
| `type` | string | Yes | Metadata type label |
| `featureId` | string | Yes | Feature ID for LLM tracking and billing |
| `clientIdentifier` | string | Yes | Client identifier for remote requests |
| `contextManagement` | boolean | No | Whether to inject agent config into remote requests (default: `true`) |
| `agentId` | string | No | Template agent document ID |

**Response:**
```json
{"success": true, "metadata_id": "68a6c703e57bf28c58d97243", "name": "MY_COPILOT"}
```

---

### Update Endpoints

All update endpoints follow the same pattern: provide the entity ID and only the fields you want to change. Omitted fields are left untouched.

| Endpoint | ID Field | Updatable Fields |
|---------|---------|-----------------|
| `POST /update/agents` | `agent_id` | `name`, `description`, `tasks`, `llm_config`, `swarm_type`, `workflow_edges`, `agent_as_task`, `task_as_router`, `agent_type`, `partner_id` |
| `POST /update/agent-metadata` | `name` | `type`, `featureId`, `contextManagement`, `clientIdentifier`, `agentId` |

**Example — update an agent's tasks:**
```json
{
  "agent_id": "68a6c70375e584a205fc66e4",
  "tasks": [
    {
      "name": "updated_task",
      "description": "Updated task description",
      "system_template": "You are a helpful agent. Always cite your sources.",
      "tools": []
    }
  ]
}
```

**Response:** `{"success": true, "agent_id": "68a6c70375e584a205fc66e4"}`

---

### List / Get Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agents` | GET | List all agents, or fetch by `ids` param or `(agentId, partnerId, version)` triple |
| `/agent-metadata` | GET | Fetch metadata by `name` |

**Agent list response** includes a `mermaid_code` field with a Mermaid diagram of the agent's task graph.

**Response shape:**
```json
{
  "success": true,
  "response": [{"_id": "...", "name": "...", ...}]
}
```

---

### `GET /session`

Generate a new session ID (stored in Redis).

```json
{"success": true, "session_id": "unique_session_id"}
```

---

### Invoke & Stream (unified `POST /invoke`)

Use **`POST /invoke`** for both blocking JSON and SSE streaming. Set **`"stream": true`** in the JSON body to enable streaming; optionally set **`streamMode`** (`["messages"]`, `["values"]`, or both).

#### Request Contract

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agentId` | string | Yes | Agent ID (Mongo `_id` or logical agent ID) |
| `partnerId` | int | No | Partner ID (for logical agent resolution) |
| `version` | int | No | Version (for logical agent resolution) |
| `sessionId` | string | Yes | Session ID (from `/session` or your own) |
| `id` | string | Yes | Unique request identifier (for tracking/correlation) |
| `messages` | ApiMessage[] | Yes | Conversation messages |
| `conversationState` | string | No | `stateless` (default) or `stateful` (Redis-backed multi-turn) |
| `context` | object | No | Request context (screen context, user context) |
| `webhookData` | object | No | Webhook delivery config |
| `mockToolBehaviour` | object | No | Runtime overrides for prompt-tool behaviors |
| `stream` | boolean | No | If **true**, response is **SSE**; default **false** for JSON |
| `streamMode` | string[] | No | When `stream` is true: `["messages"]` (default), `["values"]`, or both |

**ApiMessage structure:**
```json
{
  "role": "user",
  "content": [{"type": "input.text", "text": "Your question here"}]
}
```

#### Response Contract (`stream`: false)

```json
{
  "apiVersion": "1.0",
  "sessionId": "session-001",
  "id": "req-001",
  "content": [
    {"type": "response.text", "index": 0, "text": "Here is what I found..."}
  ],
  "status": "COMPLETED",
  "text": "Here is what I found...",
  "error": {"message": null, "retryable": false},
  "usage": {
    "totalCost": 0.001,
    "numCalls": 2,
    "timing": {"ttft": 0.5, "totalTime": 2.1},
    "modelBreakdown": [{"modelId": "gpt-4.1-2025-04-14", "provider": "OPEN_AI", "inputTokens": 150, "outputTokens": 80}]
  }
}
```

#### Stream Response (SSE)

With **`"stream": true`**, `POST /invoke` returns Server-Sent Events:

1. **`stream.start`** — first frame
2. **`content.delta`** — text chunks via `response.text.delta` items
3. **`stream.completed`** — final metrics frame with `usage`
4. **`stream.failed`** — on server-side errors

```
data: {"type":"stream.start","sequence":0,"id":"req-001","created":1710000000}

data: {"type":"content.delta","sequence":1,"id":"req-001","content":[{"type":"response.text.delta","index":1,"text":"Hello "}]}

data: {"type":"stream.completed","sequence":42,"id":"req-001","usage":{...}}

data: [DONE]
```

#### Webhook Mode

When `webhookData` is provided, the server responds with **202 Accepted** and delivers results asynchronously:

```json
{
  "webhookData": {
    "webhookUrl": "https://your-service.com/callback",
    "webhookHeaders": {"Authorization": "Bearer ..."},
    "webhookRetryConfig": {"maxRetries": 3, "retryDelaySeconds": 2, "exponentialBackoff": true}
  }
}
```

---

### Platform Sync

#### `POST /platform/sync/agent`

Sync an agent definition from the external platform. Checks Mongo cache first; if not found, fetches from the platform sync service.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agentId` | string | Yes | Platform agent ID |
| `partnerId` | int | Yes | Partner ID |
| `version` | int | Yes | Agent version |

---

## Error Responses

All errors follow a consistent shape:

```json
{
  "success": false,
  "error": {
    "code": 400,
    "message": "description of the error"
  }
}
```

| Code | Meaning |
|------|---------|
| 400 | Validation error (missing/invalid fields) |
| 404 | Entity not found |
| 409 | Duplicate name (agents, metadata) |
| 504 | Timeout during agent execution |
| 500 | Internal server error |

---

## Architecture

- **Tornado** async web server (`POST /invoke` for JSON or SSE via the `stream` flag)
- **MongoDB** for persistent storage of agents and metadata
- **Redis** for session management, `last_active_task` tracking, and document caching
- **LangGraph** under the hood for agent orchestration and task graphs

---

## See Also

- [SDK README](sdk/README.md) — for using Agent Builder as a Python library
