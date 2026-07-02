import logging

import tornado.web

from agent_builder.handlers.core.base_handler import BaseBuilderHandler

logger = logging.getLogger(__name__)


class SessionHandler(BaseBuilderHandler):

    async def get(self) -> None:
        session_id = await self.redis_client.generate_session_id()
        logger.debug("Generated new session_id=%s", session_id)

        self.write_json(
            {
                "success": True,
                "session_id": session_id,
            },
            status=200,
        )
        logger.info("Session %s created successfully", session_id)


class CloneSessionHandler(BaseBuilderHandler):

    def validate_payload(self, payload):
        source = payload.get("sourceSessionId")
        target = payload.get("targetSessionId")
        partner_id = payload.get("partnerId")
        if not source or not target:
            raise tornado.web.HTTPError(400, "sourceSessionId and targetSessionId are required")
        if partner_id is None:
            raise tornado.web.HTTPError(400, "partnerId is required")
        if source == target:
            raise tornado.web.HTTPError(400, "sourceSessionId and targetSessionId must be different")
        return payload

    async def process(self, request):
        source_id = request["sourceSessionId"]
        target_id = request["targetSessionId"]
        partner_id = int(request["partnerId"])

        await self.mongo_client.clone_session(source_id, target_id, partner_id=partner_id)

        logger.info("Session cloned | %s -> %s", source_id, target_id)
        return {
            "success": True,
            "sourceSessionId": source_id,
            "targetSessionId": target_id,
        }
