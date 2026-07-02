import os
import uuid
from dataclasses import dataclass, field
from typing import Optional
from typing_extensions import List, Literal, Tuple
from agent_builder.utils.constants import DEFAULT_SWARM_TYPE, MINIMAL_REASONING


@dataclass
class LLMConfig:
    model: str = "gpt-4.1-2025-04-14"
    llm_router_url: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_ROUTER_URL", 
            "qa6-intuitionx-llm-router-v2.sprinklr.com",
            # "prod0-intuitionx-llm-router-v2.sprinklr.com",
            # "intuitionx-llm-router-v2.qa6-k8singress-intuition-gke.sprinklr.com"
            # "azrqa-k8singress-intuition-aks.sprinklr.com"
        )
    )
    provider: Literal["AZURE_OPEN_AI", "OPEN_AI", "VERTEX", "LOCAL", "VOICE"] = "AZURE_OPEN_AI"
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 1.0
    partner_id: int = 66000000
    reasoning_effort: str = MINIMAL_REASONING
    tracking_params: dict = field(default_factory=lambda: {"release": "ca_research", "feature": "AGENT_BUILDER"})
    client_identifier: str = "ml-ca-dev"
    timeout: int = 60
    pii_masking_templates: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    #: When set (and not ``LOCAL`` provider), Sprinklr router requests send only this id plus
    #: tracking/client/partner/kwargs — no model-level generation parameters.
    llm_configuration_id: Optional[str] = None

@dataclass
class TaskConfig:
    """Unified config for all task types (normal, remote, release)."""
    name: str
    description: str
    _id: str = field(default_factory=lambda: f"task_{str(uuid.uuid4())}")
    task_type: str = "normal"
    system_template: str = field(default_factory=str)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    tool_keys: List[str] = field(default_factory=list)
    preprocessor: Literal['DEFAULT', 'CLEAR_ALL_MESSAGES', 'KEEP_ONLY_LAST_MESSAGE'] = 'DEFAULT'
    postprocessor: Literal['DEFAULT'] = 'DEFAULT'

@dataclass
class AgentConfig:
    """
    Configuration object for an Agent (a swarm of tasks).
    """
    name: str
    description: str = field(default_factory=str)
    _id: str = field(default_factory=lambda: f"agent_{str(uuid.uuid4())}") #is only used locally, overwritten by the mongo_id when saved to the DB
    agent_type: str = field(default_factory=str)
    partner_id: int = 66000000
    router_model_config:LLMConfig  = field(default_factory=LLMConfig)
    workflow_edges: List[Tuple[str, str]] = field(default_factory=list)
    swarm_type: Literal["all_connected", "router_back_connection","default"] = DEFAULT_SWARM_TYPE
    task_keys: List[str] =  field(default_factory=list)
    version: int = 0
    agent_id: str = field(default_factory=str)
    
@dataclass
class HttpConfig:
    url: str = ""
    proxy_server: str = ""
    proxy_port: str = ""

@dataclass
class RemoteReleaseMetadata:
    release_name: str
    release_description: str
    http_config: HttpConfig = field(default_factory=HttpConfig)