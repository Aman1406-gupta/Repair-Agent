from agent_builder.utils.application import BaseHandler
import logging
import tornado.web
import tornado.escape

logger = logging.getLogger(__name__)

VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class LogLevelHandler(BaseHandler):
    """GET returns the current root log level; POST changes it at runtime."""

    def get(self):
        current_level = logging.getLogger().getEffectiveLevel()
        self.finish({"level": logging.getLevelName(current_level)})

    def post(self):
        try:
            body = tornado.escape.json_decode(self.request.body)
        except Exception:
            raise tornado.web.HTTPError(400, reason="Invalid JSON in request body")

        level = body.get("level", "").upper()
        if level not in VALID_LEVELS:
            raise tornado.web.HTTPError(
                400, reason=f"Invalid log level '{level}'. Must be one of: {', '.join(sorted(VALID_LEVELS))}"
            )

        logging.getLogger().setLevel(level)
        logger.info("Root log level changed to %s", level)
        self.finish({"level": level})
