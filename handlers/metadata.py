import tornado
from typing import Dict, Any

from agent_builder.handlers.core.base_handler import BaseBuilderHandler
from agent_builder.utils.constants import AGENT_METADATA_COLLECTION, DEFAULT_PARTNER_ID
from agent_builder.handlers.core.requests import (
    RegisterAgentMetadataRequest,
    UpdateAgentMetadataRequest,
)
from agent_builder.handlers.core.responses import (
    RegisterAgentMetadataResponse,
    UpdateAgentMetadataResponse,
    GetAgentMetadataResponse,
)

class RegisterAgentMetadataHandler(BaseBuilderHandler):
    """Handler for POST /register/agent-metadata"""

    def validate_payload(self, payload: Dict[str, Any]) -> RegisterAgentMetadataRequest:
        return RegisterAgentMetadataRequest(**payload)

    async def process(self, request: RegisterAgentMetadataRequest) -> Dict[str, Any]:
        result = await self.mongo_client.register_agent_metadata(request)
        return RegisterAgentMetadataResponse(**result).model_dump()


class UpdateAgentMetadataHandler(BaseBuilderHandler):
    """Handler for POST /update/agent-metadata"""

    def validate_payload(self, payload: Dict[str, Any]) -> UpdateAgentMetadataRequest:
        return UpdateAgentMetadataRequest(**payload)

    async def process(self, request: UpdateAgentMetadataRequest) -> Dict[str, Any]:
        updates = request.model_dump(
            by_alias=True, exclude_none=True, exclude={"name"}
        )
        result = await self.mongo_client.update_agent_metadata(
            name=request.name, **updates
        )
        return UpdateAgentMetadataResponse(**result).model_dump()


class GetAgentMetadataHandler(BaseBuilderHandler):
    """Handler for GET /agent-metadata"""

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload.get("name")

        # If no name provided, return all metadata
        if not name:
            results = await self.mongo_client.get_all_agent_metadata()
            return GetAgentMetadataResponse(response=results).model_dump()

        # Otherwise, get specific metadata by name
        result = await self.mongo_client._find_one_cached(
            AGENT_METADATA_COLLECTION, {"name": name}, partner_id=DEFAULT_PARTNER_ID,
        )

        if result is None:
            raise tornado.web.HTTPError(status_code=404, reason="Agent metadata not found")

        return GetAgentMetadataResponse(response=result).model_dump()
