"""
Pydantic request models for all handler endpoints.

Each model validates + coerces the raw JSON payload so that handlers
and the mongo service layer receive typed, pre-validated objects.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union, get_type_hints

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator, AliasChoices
from typing_extensions import get_args

from agent_builder.base.configs import AgentConfig

from agent_builder.utils.misc import _loads_if_json_str


# ---------------------------------------------------------------------------
# Reusable types
# ---------------------------------------------------------------------------
def _check_non_empty(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must be a non-empty string")
    return v


NonEmptyStr = Annotated[str, AfterValidator(_check_non_empty)]

ConversationState = Literal["stateful", "stateless"]


# ---------------------------------------------------------------------------
# Reusable validator helpers
# ---------------------------------------------------------------------------
def _parse_json_dict(v: Any, field_name: str) -> Any:
    """Parse a JSON string into a dict, passing ``None`` through unchanged."""
    if v is None:
        return v
    return _loads_if_json_str(v, field_name) if isinstance(v, str) else v


def _validate_non_empty_str_list(v: Optional[List[str]], field_name: str) -> Optional[List[str]]:
    """Ensure every item in an (optional) list is a non-empty string."""
    if v is not None:
        for idx, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
    return v


def _validate_boolean(v: Any, field_name: str, optional: bool = False) -> Optional[bool]:
    """Ensure value is a boolean. If optional=True, allows None."""
    if optional and v is None:
        return v
    if not isinstance(v, bool):
        suffix = " if provided" if optional else ""
        raise ValueError(f"{field_name} must be a boolean{suffix}")
    return v


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
class RegisterToolRequest(BaseModel):
    openapi_schema: Dict[str, Any]
    filter_name: Optional[NonEmptyStr] = None

    @field_validator("openapi_schema", mode="before")
    @classmethod
    def parse_openapi_schema(cls, v):
        return _parse_json_dict(v, "openapi_schema")

    @field_validator("openapi_schema", mode="after")
    @classmethod
    def validate_openapi_structure(cls, v):
        version = v.get("openapi") or v.get("swagger")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(
                "openapi_schema must include an 'openapi' (v3+) or 'swagger' (v2) version string"
            )

        info = v.get("info")
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("title"), str)
            or not isinstance(info.get("version"), str)
        ):
            raise ValueError(
                "openapi_schema.info.title and openapi_schema.info.version are required"
            )

        paths = v.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("openapi_schema.paths must be an object")

        return v


# ---------------------------------------------------------------------------
# Prompt-tool registration
# ---------------------------------------------------------------------------
class RegisterPromptToolRequest(BaseModel):
    openapi_schema: Optional[Dict[str, Any]] = None
    openai_schema: Optional[Dict[str, Any]] = None
    filter_name: Optional[NonEmptyStr] = None
    llm_behavior: str = ""
    llm_config: Dict[str, Any]

    @field_validator("openapi_schema", mode="before")
    @classmethod
    def parse_openapi_schema(cls, v):
        return _parse_json_dict(v, "openapi_schema")

    @field_validator("openai_schema", mode="before")
    @classmethod
    def parse_openai_schema(cls, v):
        return _parse_json_dict(v, "openai_schema")

    @model_validator(mode="after")
    def exactly_one_schema(self):
        has_openapi = self.openapi_schema is not None
        has_openai = self.openai_schema is not None
        if has_openapi == has_openai:
            raise ValueError(
                "Provide exactly ONE of 'openapi_schema' or 'openai_schema'"
            )
        return self


# ---------------------------------------------------------------------------
# Task registration (embedded under agents)
# ---------------------------------------------------------------------------
TaskToolInput = Union["RegisterToolRequest", "RegisterPromptToolRequest"]
TaskInput = Union["RegisterTaskRequest", "RegisterReleaseTaskRequest"]


class RegisterTaskRequest(BaseModel):
    """Normal task embedded on an agent (or nested via ``task_as_tools``)."""
    name: NonEmptyStr
    description: NonEmptyStr
    system_template: NonEmptyStr
    task_type: Optional[str] = None
    tools: List[TaskToolInput] = Field(default_factory=list)
    llm_config: Optional[Dict[str, Any]] = None
    preprocessor: NonEmptyStr = "DEFAULT"
    postprocessor: Optional[NonEmptyStr] = None
    task_as_tools: Optional[List["RegisterTaskRequest"]] = None
    agent_as_tools: Optional[List["RegisterAgentRequest"]] = None
    enabled: bool = True
    skills_zip: Optional[List[str]] = None
    subagents: Optional[List[Dict[str, Any]]] = None
    id: Optional[str] = Field(default=None, description="Optional client-supplied embedded task id")


class HttpConfigPayload(BaseModel):
    url: NonEmptyStr
    proxy_server: str = ""
    proxy_port: str = ""


def parse_task_input(data: Dict[str, Any]) -> TaskInput:
    """Parse a task payload as normal or release task request."""
    if "http_config" in data:
        return RegisterReleaseTaskRequest(**data)
    return RegisterTaskRequest(**data)


def parse_task_input_list(items: Optional[List[Any]]) -> Optional[List[TaskInput]]:
    if items is None:
        return None
    return [parse_task_input(item) if isinstance(item, dict) else item for item in items]


class RegisterReleaseTaskRequest(BaseModel):
    """Register a release task: thin wrapper grouping sub-tasks behind a remote endpoint."""
    name: NonEmptyStr
    description: NonEmptyStr
    http_config: HttpConfigPayload
    task_form: Optional[str] = None
    attributes: Dict = Field(default_factory=dict)
    enabled: bool = True
    id: Optional[str] = Field(default=None, description="Optional client-supplied embedded task id")


# ---------------------------------------------------------------------------
# Agent registration / update
# ---------------------------------------------------------------------------
class RegisterAgentRequest(BaseModel):
    agent_type: NonEmptyStr
    name: Optional[str] = None
    partner_id: int
    tasks: Optional[List[TaskInput]] = None
    description: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None
    workflow_edges: Optional[List[Tuple[str, str]]] = None
    swarm_type: Optional[str] = None
    agent_as_task: Optional[List["RegisterAgentRequest"]] = None
    task_as_router: Optional[TaskInput] = None
    id: Optional[str] = Field(default=None, description="Optional client-supplied embedded agent id")

    @field_validator("swarm_type")
    @classmethod
    def validate_swarm_type(cls, v):
        if v is None:
            return v

        type_hints = get_type_hints(AgentConfig)
        valid = list(get_args(type_hints.get("swarm_type", type(None))))
        if v not in valid:
            raise ValueError(f"swarm_type must be one of: {', '.join(valid)}")
        return v

    @field_validator("workflow_edges", mode="before")
    @classmethod
    def coerce_edges(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("workflow_edges must be a list")
        converted = []
        for idx, edge in enumerate(v):
            if not isinstance(edge, (list, tuple)):
                raise ValueError(f"workflow_edges[{idx}] must be a list or tuple")
            if len(edge) != 2:
                raise ValueError(
                    f"workflow_edges[{idx}] must have exactly 2 elements (from_node, to_node)"
                )
            converted.append(tuple(edge))
        return converted


class UpdateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: NonEmptyStr
    name: Optional[str] = None
    description: Optional[str] = None
    tasks: Optional[List[TaskInput]] = None
    llm_config: Optional[Dict[str, Any]] = None
    workflow_edges: Optional[List[Tuple[str, str]]] = None
    swarm_type: Optional[str] = None
    agent_as_task: Optional[List[RegisterAgentRequest]] = None
    task_as_router: Optional[TaskInput] = None
    agent_type: Optional[str] = None
    partner_id: int

    @field_validator("workflow_edges", mode="before")
    @classmethod
    def coerce_edges(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("workflow_edges must be a list")
        converted = []
        for idx, edge in enumerate(v):
            if not isinstance(edge, (list, tuple)):
                raise ValueError(f"workflow_edges[{idx}] must be a list or tuple")
            if len(edge) != 2:
                raise ValueError(
                    f"workflow_edges[{idx}] must have exactly 2 elements (from_node, to_node)"
                )
            converted.append(tuple(edge))
        return converted

    @model_validator(mode="after")
    def validate_nonnullable_when_set(self):
        """name and swarm_type cannot be null when explicitly provided."""
        if "name" in self.model_fields_set:
            if self.name is None or not self.name.strip():
                raise ValueError("'name' must be a non-empty string")
        if "swarm_type" in self.model_fields_set:
            type_hints = get_type_hints(AgentConfig)
            valid = list(get_args(type_hints.get("swarm_type", type(None))))
            if self.swarm_type is None or self.swarm_type not in valid:
                raise ValueError(
                    f"'swarm_type' must be one of: {', '.join(valid)}"
                )
        return self


# ---------------------------------------------------------------------------
# Agent metadata registration / update
# ---------------------------------------------------------------------------
class RegisterAgentMetadataRequest(BaseModel):
    name: NonEmptyStr
    agent_type: NonEmptyStr = Field(alias="type")
    feature_id: NonEmptyStr = Field(alias="featureId")
    client_identifier: NonEmptyStr = Field(alias="clientIdentifier")
    context_management: bool = Field(default=True, alias="contextManagement")
    agent_id: Optional[str] = Field(default=None, alias="agentId")

    @field_validator("context_management", mode="before")
    @classmethod
    def validate_context_management(cls, v):
        return _validate_boolean(v, "contextManagement")


class UpdateAgentMetadataRequest(BaseModel):
    name: NonEmptyStr
    agent_type: Optional[NonEmptyStr] = Field(default=None, alias="type")
    feature_id: Optional[NonEmptyStr] = Field(default=None, alias="featureId")
    client_identifier: Optional[NonEmptyStr] = Field(default=None, alias="clientIdentifier")
    context_management: Optional[bool] = Field(default=None, alias="contextManagement")
    agent_id: Optional[str] = Field(default=None, alias="agentId")

    @field_validator("context_management", mode="before")
    @classmethod
    def validate_context_management(cls, v):
        return _validate_boolean(v, "contextManagement", optional=True)



# Unified API Contract – shared sub-models
# ---------------------------------------------------------------------------

class ContentPart(BaseModel):
    """A single part in a multipart content message (file, inline data, URL, etc.)."""
    kind: str
    mimeType: Optional[str] = None
    encoding: Optional[str] = None
    data: str


class MessageContent(BaseModel):
    """Content envelope following the COPILOT_MULTIPART_CONTENT schema."""
    type: Optional[str] = None
    text: Optional[str] = None
    textFormat: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ToolCallObject(BaseModel):
    """A single tool call or action in the unified contract."""
    id: str
    index: int
    type: str
    name: str
    arguments: Any = Field(default_factory=dict)

    def to_langchain_tool_call(self):
        return {"id": self.id, "name": self.name, "args": self.arguments or {}, "type": self.type}


class ApiMessage(BaseModel):
    """A single message in the unified API contract (used in both request and response)."""
    id: Optional[str] = None
    role: str
    content: Optional[List[MessageContent]] = None
    tool_calls: List[ToolCallObject] = Field(default_factory=list)
    actions: List[ToolCallObject] = Field(default_factory=list)
    tool_call_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Unified API Contract – context models
# ---------------------------------------------------------------------------

class ScreenContext(BaseModel):
    """Screen context information accompanying a request."""
    type: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    activeSelection: Optional[List[Dict[str, Any]]] = None
    moduleType: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class UserContext(BaseModel):
    userId: int
    clientId: int
    partnerId: int
    partnerName: Optional[str] = None


class RequestContext(BaseModel):
    
    screenContext: Optional[ScreenContext] = None
    userContext: Optional[UserContext] = None
    attachedAgentId: Optional[str] = None

    model_config = ConfigDict(extra="allow")

# ---------------------------------------------------------------------------
# Unified API Contract – webhook models
# ---------------------------------------------------------------------------

class WebhookRetryConfig(BaseModel):
    maxRetries: int
    retryDelaySeconds: int
    exponentialBackoff: bool


class WebhookData(BaseModel):
    webhookUrl: str
    webhookHeaders: Dict[str, str]
    webhookRetryConfig: Optional[WebhookRetryConfig] = None

class SyncAgentRequest(BaseModel):
    """Request model for ``POST /platform/sync/agent``."""
    model_config = ConfigDict(extra="forbid")

    agentId: NonEmptyStr
    partnerId: int
    version: int


class ConversationInterruptRequest(BaseModel):
    """Request model for ``POST /conversation/interrupt``."""
    model_config = ConfigDict(extra="forbid")

    sessionId: NonEmptyStr
    requestId: NonEmptyStr

ConversationStopRequest = ConversationInterruptRequest


# ---------------------------------------------------------------------------
# Unified API Contract – background-mode ``/message`` polling + ingest
# ---------------------------------------------------------------------------

class MessagePollRequest(BaseModel):
    """Request model for ``POST /message`` (backend polls accumulated chunks)."""
    model_config = ConfigDict(extra="allow")

    # The backend sends ``Id``; accept ``id``/``pollingId``/``requestId`` aliases too.
    requestId: NonEmptyStr = Field(validation_alias=AliasChoices("Id", "id", "pollingId", "messageId"))
    sessionId: NonEmptyStr
    agentId: Optional[str] = None
    seq: int = 0
    generationId: Optional[str] = None

    @field_validator("seq", mode="before")
    @classmethod
    def _default_seq(cls, v: Any) -> int:
        if v is None:
            return 0
        return v


class MessageIngestRequest(BaseModel):
    """Request model for ``POST /message/ingest`` (remote task pushes chunks)."""
    model_config = ConfigDict(extra="allow")

    requestId: NonEmptyStr = Field(validation_alias=AliasChoices("id", "Id", "requestId"))
    sessionId: NonEmptyStr
    agentId: Optional[str] = None
    generationId: NonEmptyStr
    events: List[Dict[str, Any]]

    @field_validator("events")
    @classmethod
    def _validate_events(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("events must be a non-empty list")
        for i, ev in enumerate(v):
            if not isinstance(ev, dict):
                raise ValueError(f"events[{i}] must be an object")
            seq = ev.get("sequence")
            if not isinstance(seq, int) or isinstance(seq, bool):
                raise ValueError(f"events[{i}] must have an integer 'sequence'")
            if not ev.get("type"):
                raise ValueError(f"events[{i}] must have a 'type'")
        return v


# ---------------------------------------------------------------------------
# Unified API Contract – top-level request
# ---------------------------------------------------------------------------

class AgentInvokeRequest(BaseModel):
    """Top-level request model for ``/invoke`` (JSON or SSE via ``stream``).

    Use ``conversationState`` to choose Redis-backed multi-turn (``stateful``) vs
    each-call full context (``stateless``, default).
    """
    agentId: NonEmptyStr
    # Invoke now supports logical agent resolution using (agentId, partnerId, version).
    # These are optional for backward compatibility with Mongo _id based resolution.
    partnerId: Optional[int] = Field(default=None, validation_alias=AliasChoices("partnerId", "partner_id"))
    version: Optional[int] = Field(default=None, validation_alias=AliasChoices("version", "versionId", "version_id"))
    sessionId: NonEmptyStr
    id: NonEmptyStr
    context: Optional[RequestContext] = None
    messages: List[ApiMessage]
    webhookData: Optional[WebhookData] = None
    mockToolBehaviour: Optional[Dict[str, str]] = None
    lastActiveTask: Optional[Any] = None
    delivery: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    streamMode: Optional[List[str]] = None
    conversationState: ConversationState = "stateless"
    responseId: Optional[str] = None
    sprMcpAuthToken: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("conversationState", mode="before")
    @classmethod
    def _normalize_conversation_state(cls, v: Any) -> str:
        if v is None:
            return "stateless"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("stateful", "stateless"):
                return s
        raise ValueError("conversationState must be 'stateful' or 'stateless'")
    @field_validator("streamMode", mode="before")
    @classmethod
    def validate_stream_mode(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]

        if not isinstance(v, list) or not v:
            raise ValueError("streamMode must be a non-empty list containing 'messages', 'values', or both")

        allowed = {"messages", "values"}
        extras = set(v) - allowed
        if extras:
            raise ValueError(f"streamMode must contain only 'messages' and/or 'values'. Extra values: {extras}")

        return v

    @field_validator("delivery", mode="before")
    @classmethod
    def _normalize_delivery(cls, v: Any) -> Any:
        if v is None:
            v = {}

        v.setdefault("mode", "foreground")
        return v

    @model_validator(mode="after")
    def _default_stream_mode_when_streaming(self):
        if self.stream and (self.streamMode is None or len(self.streamMode) == 0):
            self.streamMode = ["messages"]
        return self






    


# Backward compatibility for imports; same shape as ``AgentInvokeRequest`` with ``stream=True``.
AgentStreamRequest = AgentInvokeRequest

RegisterTaskRequest.model_rebuild()
RegisterAgentRequest.model_rebuild()

