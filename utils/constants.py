# Application constants

import os

# Elasticsearch telemetry indices (fixed; Kibana dashboards target these names)
ES_TRACES_INDEX = "agent-builder-traces"
ES_LLM_CALLS_INDEX = "agent-builder-llm-calls"
ES_HANDLER_IO_INDEX = "agent-builder-handler-io"
ES_EVENT_LOOP_HEALTH_INDEX = "agent-builder-event-loop-health"

# Telemetry environment (loaded once at import)
TELEMETRY_ENV_CONFIG = {
    "enabled": os.environ.get("ENABLE_TELEMETRY", "false").lower() == "true",
    "logs": os.environ.get("ES_TELEMETRY_LOGS", "false").lower() == "true",
    "batch_size": int(os.environ.get("ES_BATCH_SIZE", "100")),
    "flush_interval": float(os.environ.get("ES_FLUSH_INTERVAL", "1.5")),
    "max_queue_size": int(os.environ.get("ES_MAX_QUEUE_SIZE", "10000")),
    "max_ops": int(os.environ.get("ES_HANDLER_IO_MAX_OPS", "100")),
    "health_interval": float(os.environ.get("ES_HEALTH_SAMPLE_INTERVAL_SEC", "3.0")),
}

# Collections
AGENT_COLLECTION = "agentBuilderAgents"
LOG_COLLECTION = "agentBuilderLogs"
AGENT_METADATA_COLLECTION = "agentBuilderMetadata"
SESSIONS_COLLECTION = "agentBuilderSessions"
# Optional reference on metadata (camelCase in Mongo) to a template agent in AGENT_COLLECTION.
AGENT_METADATA_AGENT_ID = "agentId"

# Payload field names
NAME = "name"
OPENAPI_SCHEMA = "openapi_schema"
DESCRIPTION = "description"
PROMPT = "prompt"
TOOLS = "tools"
LLM_CONFIG = "llm_config"
PLANNING_METHOD = "planning_method"
NAME_DESC = "name_desc"
OPENAPI_SPEC = "openapi_spec"
OPENAI_SCHEMAS = "openai_schemas"
OPENAI_SCHEMA = "openai_schema"
BEHAVIOR_OVERRIDES = "behavior_overrides"
DEFAULT_BEHAVIOR = "default_behavior"
LLM_BEHAVIOR = "llm_behavior"
MODEL = "model"
PROVIDER = "provider"
SYSTEM_TEMPLATE = "system_template"
AGENT_ID = "agent_id"
SESSION_ID = "session_id"
USER_MESSAGE = "user_message"
MOCK_TOOL_BEHAVIOR = "mock_tool_behavior"
TASKS = "tasks"
FILTER_NAME = "filter_name"
PREPROCESSOR = "preprocessor"
POSTPROCESSOR = "postprocessor"
SWARM_TYPE = "swarm_type"
WORKFLOW_EDGES = "workflow_edges"
AGENT_AS_TASK = "agent_as_task"
TASK_AS_TOOLS = "task_as_tools"
AGENT_AS_TOOLS = "agent_as_tools"
LIVENESS_SERVICE = "liveness_service"
READINESS_SERVICE = "readiness_service"
SERVER_PORT = 10000
TASK_AS_ROUTER = "task_as_router"
AGENT_TYPE = "agent_type"
PARTNER_ID = "partner_id"
VERSION = "version"
# Stored on task ``llm_config`` / :class:`LLMConfig` when syncing from platform (see sync handler).
LLM_CONFIGURATION_ID = "llm_configuration_id"
# Platform task JSON field for pre-resolved LLM configuration id (camelCase).
PLATFORM_LLM_CONFIGURATION_ID = "llmConfigurationId"
# Platform agent JSON field for skill definitions [{name, type, url}].
PLATFORM_SKILLS_ZIPPED = "skills"
# When truthy, sync maps ``PLATFORM_LLM_CONFIGURATION_ID`` from platform tasks into stored llm_config.
ENV_SYNC_POPULATE_LLM_CONFIGURATION_ID = "AGENT_BUILDER_SYNC_POPULATE_LLM_CONFIGURATION_ID"

# Agent metadata field names (for error messages; values match Mongo / API camelCase)
FEATURE_ID = "featureId"
CLIENT_IDENTIFIER = "clientIdentifier"

# Default configurations
DEFAULT_LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000
}
DEFAULT_QA_PARTNER_ID = 66000000
MINIMAL_REASONING = "minimal"
# Default swarm topology when registering without a template (matches AgentConfig).
DEFAULT_SWARM_TYPE = "all_connected"

# Server settings
DEFAULT_PARTNER_ID = 0
SERVER_TYPE = "GLOBAL"
DEFAULT_SERVER_TYPE = "DEFAULT"

PARENT_ROUTER_NODE = "parent_router_node"

TASK_TYPE = "task_type"
ENABLED = "enabled"
SUBAGENTS = "subagents"
# List of base64-encoded skill ZIP blobs on embedded tasks.
SKILLS_ZIP = "skills_zip"

RAW_RESPONSE_KEY = "__raw_response__"

# Task type values
TASK_TYPE_REMOTE = "remote"
TASK_TYPE_RELEASE = "release"
TASK_TYPE_NORMAL = "normal"
TASK_TYPE_DEEP = "deep_agent"
TASK_TYPE_REMOTE_RELEASE = "remote_release"
DEFAULT_TASK_TYPE = TASK_TYPE_DEEP if os.environ.get("TASK_TYPE", "").strip().lower() == "deep" else TASK_TYPE_NORMAL

# State / document field names
MESSAGES = "messages"
REMOTE_REQUEST = "remote_request"
REMOTE_RESPONSE = "remote_response"
TASK_FORM = "task_form"
ATTRIBUTES = "attributes"
HTTP_CONFIG = "http_config"
LAST_ACTIVE_TASK = "last_active_task"
ROUTER_MODEL_CONFIG = "router_model_config"
STREAM = "stream"
CONFIG_VARIABLES = "config_variables"
# Non-system ``messages`` count at ``/invoke`` prepare; :func:`~agent_builder.llm_client.utils.remote_adapter.state_to_response` uses this for slicing instead of ``len(remote_request["messages"])``.
INVOKE_MESSAGE_COUNT = "invokeInputNonSystemMessageCount"
# Inbound HTTP headers from /invoke, forwarded to remote LLM proxy calls.
CLIENT_HTTP_HEADERS = "clientHttpHeaders"
# Agent document carried in config_variables, injected into outbound remote requests.
AGENT_DOC = "agentDoc"


# Tool document fields
TOOL_TYPE_API = "api_tool"
TOOL_TYPE_KEY = "toolType"

# Message metadata keys – stored in ``additional_kwargs`` for round-trip fidelity.
ENVELOPE_KEY = "__envelope__"   # response-level: usage, error (and lastActiveTask from ENVELOPE_FIELDS)
ENVELOPE_FIELDS = ("usage", "error", "lastActiveTask")
# LangChain ``additional_kwargs`` key for invoke ``content[]`` block list (request/response round-trip).
_RAW_REMOTE_CONTENT = "_raw_remote_content_"
# Set on :exc:`RuntimeError` from remote ``stream.failed`` (see ``agent_builder.llm_client.utils.remote_chat_helpers.is_remote_stream_failed``).
REMOTE_STREAM_FAILED_ATTR = "__agent_builder_remote_stream_failed__"
# OpenAI-style ``choices[]`` streaming metadata (``SprinklrChatModel``); not used for invoke-agent remote.
OPENAI_CHOICES_KEY = "__choice__"

MONGO_BATCH_SIZE = 5

# External agent sync service URL (set via environment variable)
AGENT_SYNC_SERVICE_URL = "AGENT_SYNC_SERVICE_URL"

# Task types for platform sync
PLATFORM_TASK_TYPE_STANDARD = "STANDARD"
PLATFORM_TASK_TYPE_CUSTOM = "CUSTOM"

DETAIL = "detail"
ENABLED = "enabled"

# Generic task-coordinator system prompt for router tasks.
# Contains a {specialist_tasks} placeholder that must be formatted with the
# list of available specialist task descriptions before use.
TASK_COORDINATOR_PROMPT = (
    "You are a Task Coordinator. You delegate work to specialists — you never answer specialist questions yourself.\n\n"

    "## SPECIALISTS\n"
    "{specialist_tasks}\n\n"

    "## READING YOUR TODO LIST\n"
    "Your current todo list is always visible in this system prompt under the heading \"## Current Todo List\".\n"
    "Read it every time you receive control to know exactly which items are pending, in_progress, or completed.\n\n"

    "## UPDATING THE TODO LIST\n"
    "Use `write_todos` to create or update the list. Always pass the FULL list — it replaces the entire list.\n"
    "Each item has: `content` (short description), `status` (pending | in_progress | completed).\n\n"

    "## NEW USER MESSAGE\n"
    "1. Break the request into actions and map each to a specialist.If the query is simple in itself, keep it as a single item in TODO List[IMPORTANT]\n"
    "2. Call `write_todos` — first item `in_progress`, rest `pending`.\n"
    "3. Transfer to the specialist for the `in_progress` item. Do not answer yourself.\n\n"

    "## EVERY TIME YOU REGAIN CONTROL (after a transfer returns)\n"
    "This is critical — you MUST continue the flow without waiting for the user:\n"
    "1. Read \"## Current Todo List\" in this prompt.\n"
    "2. Mark the item the specialist just handled as `completed`.\n"
    "3. If any items are still `pending` or `in_progress`:\n"
    "   a. Pick the first `in_progress` item; if none, set the next `pending` item to `in_progress`.\n"
    "   b. Call `write_todos` with the updated full list.\n"
    "   c. Transfer immediately to the specialist for that item.\n"
    "4. If ALL items are `completed`: respond to the user with a concise combined summary. Do not transfer.\n\n"

    "## RULES\n"
    "- Never answer specialist questions yourself — always delegate.\n"
    "- Always call `write_todos` before every transfer.\n"
    "- Follow-up or new questions get a fresh todo list.\n"
    "- If unsure which specialist to route to, ask the user.\n"
)
