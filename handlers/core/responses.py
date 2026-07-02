"""
Pydantic response models for all handler endpoints.

Each model defines the shape of a successful JSON response, ensuring
consistent structure and providing documentation for API consumers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

class ToolInfo(BaseModel):
    name: str
    id: str


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------
class RegisterAgentResponse(BaseModel):
    success: bool = True
    agent_id: str
    task_ids: List[str]
    tool_ids: List[str]
    router_task_id: Optional[str] = None


class UpdateAgentResponse(BaseModel):
    success: bool = True
    agent_id: str


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------
class RegisterAgentMetadataResponse(BaseModel):
    success: bool = True
    metadata_id: str
    name: str


class UpdateAgentMetadataResponse(BaseModel):
    success: bool = True
    name: str


class GetAgentMetadataResponse(BaseModel):
    success: bool = True
    response: Any


# ---------------------------------------------------------------------------
# Platform sync
# ---------------------------------------------------------------------------
class SyncAgentResponse(BaseModel):
    success: bool = True
    agentId: str
    partnerId: int
    version: int


# ---------------------------------------------------------------------------
# List / generic
# ---------------------------------------------------------------------------
class ListResponse(BaseModel):
    success: bool = True
    response: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Unified API Contract – response models
# ---------------------------------------------------------------------------

# Shared shapes for nested ``content`` / widget payloads (citations, safety, etc.).


class SafetyCategory(BaseModel):
    category: str
    probability: float
    severity: Optional[str] = None


class SafetyMetadata(BaseModel):
    overallStatus: str
    categories: Optional[List[SafetyCategory]] = None


class Citation(BaseModel):
    id: Any
    type: str
    name: Optional[str] = None
    sourceId: Optional[str] = None
    number: Optional[Any] = None
    link: Optional[str] = None
    relevanceScore: Optional[float] = None
    additional: Optional[Dict[str, Any]] = None


class TimingMetrics(BaseModel):
    ttft: Optional[float] = None
    ttlt: Optional[float] = None
    totalTime: Optional[float] = None
    thinkingTime: Optional[float] = None


class ModelBreakdownEntry(BaseModel):
    modelId: str
    provider: str
    cost: Optional[float] = None
    inputTokens: Optional[float] = None
    outputTokens: Optional[float] = None
    numCalls: Optional[float] = None
    timeTaken: Optional[float] = None


class ComponentBreakdownEntry(BaseModel):
    name: str
    cost: Optional[float] = None
    timeMs: Optional[float] = None


class UsageMetrics(BaseModel):
    totalCost: float = 0.0
    numCalls: int = 0
    timing: Optional[TimingMetrics] = None
    modelBreakdown: Optional[List[ModelBreakdownEntry]] = None
    componentBreakdown: Optional[List[ComponentBreakdownEntry]] = None
    additionalMetrics: Optional[Dict[str, Any]] = None


class ErrorInfo(BaseModel):
    code: int
    type: Optional[str] = None
    message: Optional[str] = None


class AgentInvokeError(BaseModel):
    """``error`` object on accumulated ``/invoke`` JSON (matches remote ``{message, retryable}``)."""

    message: Optional[str] = None
    retryable: bool = False


class AgentInvokeContentBlock(BaseModel):
    """Single ordered row in ``messages[].content`` / accumulated ``/invoke`` ``content`` array.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    index: int
    text: Optional[str] = None
    textFormat: Optional[str] = None

class AgentInvokeResponse(BaseModel):
    """Top-level non-streaming ``/invoke`` response (accumulated ``content`` contract)."""

    apiVersion: str = "1.0"
    sessionId: str
    id: str
    createdAt: int = 0
    updatedAt: int = 0
    content: List[AgentInvokeContentBlock] = Field(default_factory=list)
    status: str = "COMPLETED"
    index: int = 0
    text: str = ""
    error: AgentInvokeError = Field(default_factory=AgentInvokeError)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    lastActiveTask: Optional[Dict[str, Any]] = None
