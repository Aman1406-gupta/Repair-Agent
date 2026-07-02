"""
``/conversation/interrupt`` endpoint — signal an active session to interrupt.

Sets ``exec_status`` to ``stopped`` for the given session so the running
agent's interrupt poller detects the signal and halts execution.
"""

import logging
from typing import Any, Dict

from agent_builder.handlers.core.base_handler import BaseBuilderHandler
from agent_builder.handlers.core.requests import ConversationInterruptRequest

logger = logging.getLogger(__name__)


class ConversationInterruptHandler(BaseBuilderHandler):

    def validate_payload(self, payload: Dict[str, Any]) -> ConversationInterruptRequest:
        return ConversationInterruptRequest(**payload)

    async def process(self, request: ConversationInterruptRequest) -> Dict[str, Any]:
        current = await self.redis_client.get_exec_status(request.sessionId)

        if current is None:
            return {
                "sessionId": request.sessionId,
                "status": "no_active_session",
            }

        current_status = current.get("status")

        if current_status == "completed":
            return {
                "sessionId": request.sessionId,
                "status": "already_completed",
            }

        if current_status == "stopped":
            return {
                "sessionId": request.sessionId,
                "status": "already_stopped",
            }

        await self.redis_client.set_exec_status(
            request.sessionId, "stopped", current.get("requestId", ""),
        )
        logger.info(
            "Conversation interrupted | session_id=%s request_id=%s",
            request.sessionId, current.get("requestId"),
        )
        return {
            "sessionId": request.sessionId,
            "status": "interrupted",
        }
