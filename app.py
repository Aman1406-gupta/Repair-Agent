# from agent_builder.local import os_exports
import asyncio
from tornado.ioloop import IOLoop
import logging
import sys

from agent_builder.utils.application import AsyncApplication
from agent_builder.utils.es_logging_handler import ElasticsearchCallbackHandler
from agent_builder.utils.telemetry import initialize_telemetry

# Import handlers from specific submodules
from agent_builder.handlers.base import (
    RegisterAgentsHandler,
    UpdateAgentsHandler,
)
from agent_builder.handlers.metadata import (
    RegisterAgentMetadataHandler,
    UpdateAgentMetadataHandler,
    GetAgentMetadataHandler,
)
from agent_builder.handlers.invoke import InvokeAgentHandler
from agent_builder.handlers.message import (
    MessageHandler,
    MessageIngestHandler,
)
from agent_builder.handlers.conversation import ConversationInterruptHandler
from agent_builder.handlers.session import SessionHandler, CloneSessionHandler
from agent_builder.handlers.log_level import LogLevelHandler
from agent_builder.handlers.sync import SyncAgentHandler
from agent_builder.handlers.list import (
    ListAgentsHandler,
)

# Import utilities
from agent_builder.storage.mongo_client import AgentBuilderMongoStore
from agent_builder.storage.redis_client import RedisClient

from agent_builder.utils.log_context import CorrelationFilter

_handler = logging.StreamHandler()
_handler.addFilter(CorrelationFilter())
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s %(levelname)s] %(name)s [sid=%(session_id)s rid=%(request_id)s]: %(message)s',
    handlers=[_handler]
)
logger = logging.getLogger(__name__)


PORT = 10000
server_started = False


handlers = {
    r"/register/agents": RegisterAgentsHandler,
    r"/register/agent-metadata": RegisterAgentMetadataHandler,
    r"/update/agents": UpdateAgentsHandler,
    r"/update/agent-metadata": UpdateAgentMetadataHandler,
    r"/session": SessionHandler,
    r"/session/clone": CloneSessionHandler,
    r"/invoke": InvokeAgentHandler,
    r"/message": MessageHandler,
    r"/message/ingest": MessageIngestHandler,
    r"/agents": ListAgentsHandler,
    r"/agent-metadata": GetAgentMetadataHandler,
    r"/conversation/interrupt": ConversationInterruptHandler,
    r"/log-level": LogLevelHandler,
    r"/platform/sync/agent": SyncAgentHandler,
}


async def initialize_connectors(app):
    """Initialize MongoDB and Redis connections.
    """
    # Initialize MongoDB client
    mongo_client = AgentBuilderMongoStore()

    # Initialize Redis client (used for serialization utilities and metadata caching)
    redis_client = RedisClient()

    # Set Redis client on mongo_client for distributed caching
    mongo_client.set_redis_client(redis_client)

    # Attach to application
    app.mongo_client = mongo_client
    app.redis_client = redis_client

    logger.info("Database connectors initialized successfully (stateless mode)")

    # ES telemetry (opt-in via ENABLE_TELEMETRY=true); see utils/telemetry.py
    await initialize_telemetry(app)


def main(port=PORT):
    global server_started

    try:
        pipeline_app = AsyncApplication()

        # Add all handlers to the application
        for pattern, handler in handlers.items():
            pipeline_app.add_handler(pattern=pattern, handler=handler)

        # Get the application instance and start the server
        app = pipeline_app.app_instance()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(initialize_connectors(app))
        app.listen(port)
        logger.info("Starting tornado server at port: " + str(port))
        server_started = True
        IOLoop.current().start()
    except Exception as e:
        logger.error("Error in application {0}".format(e), exc_info=True)
        atlas_logger.stop()
        sys.exit(0)
    finally:
        ElasticsearchCallbackHandler.stop_background_worker()


if __name__ == "__main__":
    main()
