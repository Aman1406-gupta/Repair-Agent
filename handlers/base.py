"""
CRUD handlers for agents.

Each handler is a thin declarative subclass of ``CrudHandler`` — set
class attributes, override ``validate_payload`` or ``build_response``
only when the default pattern doesn't fit.
"""
from typing import Any, Dict

import logging

from agent_builder.handlers.core.crud_handler import CrudHandler
from agent_builder.handlers.core.requests import (
    RegisterAgentRequest,
    UpdateAgentRequest,
    parse_task_input,
    parse_task_input_list,
)
from agent_builder.handlers.core.responses import (
    RegisterAgentResponse,
    UpdateAgentResponse,
)

logger = logging.getLogger(__name__)


def _parse_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce embedded task dicts in agent register/update payloads."""
    out = dict(payload)
    if "tasks" in out and out["tasks"] is not None:
        out["tasks"] = parse_task_input_list(out["tasks"])
    if "task_as_router" in out and isinstance(out["task_as_router"], dict):
        out["task_as_router"] = parse_task_input(out["task_as_router"])
    return out


class RegisterAgentsHandler(CrudHandler):
    request_model = RegisterAgentRequest
    mongo_method = "register_agent"
    response_model = RegisterAgentResponse

    def validate_payload(self, payload: Dict[str, Any]):
        return RegisterAgentRequest(**_parse_agent_payload(payload))


class UpdateAgentsHandler(CrudHandler):
    request_model = UpdateAgentRequest
    response_model = UpdateAgentResponse
    mongo_method = "update_agent"

    def validate_payload(self, payload: Dict[str, Any]):
        return UpdateAgentRequest(**_parse_agent_payload(payload))

    def build_response(self, request, result):
        return UpdateAgentResponse(
            success=result.get("success", True),
            agent_id=result["agent_id"],
        ).model_dump()
