"""
POST /platform/sync/agent

Syncs an agent from an external platform into the local agent builder.

Flow:
  1. Check Redis cache for the agent (by agentId, partnerId, version).
  2. On cache miss, check Mongo for the same triple.
  3. On Mongo miss, fetch the agent definition from the external
     platform service (URL from env ``AGENT_SYNC_SERVICE_URL``) via
     ``/fetchAgent``.
  4. Build the internal agent document from the platform response
     and persist it to Mongo + Redis.

Custom tasks merge ``llm_config`` over full ``LLMConfig`` defaults (see ``_merged_llm_config_dict_from_platform_task``).
Set ``AGENT_BUILDER_SYNC_POPULATE_LLM_CONFIGURATION_ID`` to ``1`` / ``true`` to also map
platform ``llmConfigurationId`` into stored ``llm_configuration_id``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent_builder.handlers.core.base_handler import BaseBuilderHandler
from agent_builder.handlers.core.requests import SyncAgentRequest
from agent_builder.handlers.core.responses import SyncAgentResponse
from agent_builder.handlers.utils.platform_fetch import (
    build_and_persist_agent,
    fetch_agent_from_platform,
    _merged_llm_config_dict_from_platform_task,  # noqa: F401 – re-exported for backward compat
)
from agent_builder.utils.constants import (
    AGENT_COLLECTION,
    AGENT_ID,
    PARTNER_ID,
    VERSION,
)

logger = logging.getLogger(__name__)


class SyncAgentHandler(BaseBuilderHandler):
    """Handles ``POST /platform/sync/agent``."""

    def validate_payload(self, payload: Dict[str, Any]):
        return SyncAgentRequest(**payload)

    async def process(self, request: SyncAgentRequest) -> Dict[str, Any]:
        agent_id = request.agentId
        partner_id = request.partnerId
        version = request.version

        logger.info(
            "Sync agent | agentId=%s partnerId=%s version=%s",
            agent_id, partner_id, version,
        )

        def _response() -> Dict[str, Any]:
            return SyncAgentResponse(
                agentId=agent_id, partnerId=partner_id, version=version,
            ).model_dump()

        existing = await self._find_existing_agent(agent_id, partner_id, version)
        if existing:
            logger.info("Agent found | agentId=%s version=%s", agent_id, version)
            return _response()

        platform_response = await fetch_agent_from_platform(agent_id, partner_id, version)
        await build_and_persist_agent(
            self.mongo_client, platform_response, agent_id, partner_id, version,
        )

        return _response()

    # ── cache-aware lookup (Redis → Mongo) ────────────────────────

    async def _find_existing_agent(
        self, agent_id: str, partner_id: int, version: int,
    ) -> Optional[Dict]:
        return await self.mongo_client._find_one_cached(
            AGENT_COLLECTION,
            {AGENT_ID: agent_id, PARTNER_ID: partner_id, VERSION: version},
            partner_id=partner_id,
        )
