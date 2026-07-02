from agent_builder.utils.application import BaseHandler
import asyncio
import logging
import time
from typing import Any, Dict
import tornado
from pydantic import ValidationError
from agent_builder.storage.utils.mongo_topology import (
    AgentBuilderStoreError,
    DuplicateAgentError,
    DuplicateAgentMetadataError,
)
from agent_builder.utils.log_context import set_log_context, clear_log_context
from agent_builder.utils.handler_io_telemetry import (
    bind_request_context,
    flush_handler_io_doc,
    start_request,
)
from agent_builder.utils.telemetry import init_handler_telemetry, tlog_exception

logger = logging.getLogger(__name__)

class BaseBuilderHandler(BaseHandler):
    def initialize(self, mongo_client=None) -> None:
        self.mongo_client = mongo_client or getattr(self.application, "mongo_client", None)
        if self.mongo_client is None:
            raise RuntimeError("mongo_client is not configured on the application")

        self.redis_client = getattr(self.application, "redis_client", None)
        if self.redis_client is None:
            raise RuntimeError("Redis client not configured on the application")

        self.telemetry_enabled, self.es_handler = init_handler_telemetry(self.application)

    def prepare(self):
        if self.telemetry_enabled:
            try:
                start_request(self.request.path)
            except Exception:
                tlog_exception("Failed to start handler I/O telemetry for %s", self.request.path)

    def on_finish(self):
        if getattr(self, "_handler_io_flush_deferred", False):
            return
        try:
            flush_handler_io_doc(self)
        except Exception:
            tlog_exception("Failed to flush handler I/O telemetry for %s", self.request.path)

    async def _save_session_to_mongo(self, session_id: str, session_data: Dict[str, Any]) -> None:
        try:
            await self.mongo_client.save_session(session_id, session_data)
        except Exception:
            logger.exception("Background Mongo session save failed | session_id=%s", session_id)

    def write_json(self, data: Any, status: int = 200) -> None:
        self.set_status(status)
        if isinstance(data, (dict, list)):
            self.finish(data)
        else:
            self.finish(data)

    def write_error(self, status_code: int, **kwargs) -> None:
        exc = kwargs.get("exc_info", (None, None, None))[1]
        message = getattr(exc, "reason", None) or getattr(exc, "log_message", None) or self._reason
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.finish(
            {
                "success": False,
                "error": {
                    "code": status_code,
                    "message": message,
                },
            }
        )

    def get_json_payload(self) -> Dict[str, Any]:
        if not self.request.body:
            raise tornado.web.HTTPError(status_code=400, reason="Request body is empty")
        try:
            payload = tornado.escape.json_decode(self.request.body)
        except Exception:
            raise tornado.web.HTTPError(status_code=400, reason="Invalid JSON in request body")
        if not isinstance(payload, dict):
            raise tornado.web.HTTPError(status_code=400, reason="JSON request body must be an object")
        return payload

    def _set_log_context_from_payload(self, payload: dict) -> None:
        sid = str(payload.get("sessionId") or payload.get("session_id") or "-")
        rid = str(payload.get("id") or payload.get("requestId") or "-")
        set_log_context(session_id=sid, request_id=rid)
        bind_request_context(session_id=sid, request_id=rid)

    async def post(self) -> None:
        start = time.perf_counter()
        path = self.request.path
        try:
            payload = self.get_json_payload()
            self._set_log_context_from_payload(payload)
            logger.info("POST %s | content_length=%s", path, len(self.request.body or b""))
            logger.debug("Request payload keys: %s", list(payload.keys()))
            request = self.validate_payload(payload)
            response = await self.do_process(request)
            self.write_response(response)
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info("POST %s | status=%d | latency=%.0fms", path, self.get_status(), latency_ms)
        except tornado.web.HTTPError:
            raise
        except (asyncio.TimeoutError, TimeoutError) as ex:
            logger.warning("Timeout on POST %s: %s", path, ex)
            raise tornado.web.HTTPError(status_code=504, reason="A timeout occurred in Agent Builder server")
        except Exception as ex:
            if isinstance(ex, ValidationError):
                parts = []
                for err in ex.errors():
                    field = " → ".join(str(loc) for loc in err.get("loc", []))
                    msg = err.get("msg", str(err))
                    parts.append(f"{field}: {msg}" if field else msg)
                reason = "; ".join(parts)
                logger.warning("Validation error on POST %s: %s", path, reason)
                raise tornado.web.HTTPError(status_code=400, reason=reason)

            if isinstance(ex, (DuplicateAgentError, DuplicateAgentMetadataError)):
                logger.warning("Duplicate entry on POST %s: %s", path, ex)
                raise tornado.web.HTTPError(status_code=409, reason=str(ex))
            if isinstance(ex, AgentBuilderStoreError):
                logger.error("Store error on POST %s: %s", path, ex)
                raise tornado.web.HTTPError(status_code=500, reason=str(ex))
            logger.exception("Unhandled error on POST %s", path)
            raise tornado.web.HTTPError(status_code=500, reason="An exception occurred in Agent Builder server")
        finally:
            clear_log_context()

    def write_response(self, response: Any) -> None:
        """Hook for subclasses to customise how the response is written.

        Override this (e.g. to a no-op) in handlers that write directly
        to the transport during ``process()`` (streaming, SSE, etc.).
        """
        if response is None:
            raise tornado.web.HTTPError(status_code=502, reason="No response from server")
        self.write_json(response, status=200)

    async def get(self):
        start = time.perf_counter()
        path = self.request.path
        try:
            query_params = {
                key: self.get_arguments(key) if len(self.get_arguments(key)) > 1 else self.get_argument(key)
                for key in self.request.arguments
            }
            logger.debug("GET %s | params=%s", path, list(query_params.keys()))
            response = await self.do_process(query_params)
            if response is None:
                raise tornado.web.HTTPError(status_code=502, reason="No response from server")

            self.write_json(response, status=200)
            latency_ms = (time.perf_counter() - start) * 1000
            logger.debug("GET %s | status=%d | latency=%.0fms", path, self.get_status(), latency_ms)
        except tornado.web.HTTPError:
            raise
        except (asyncio.TimeoutError, TimeoutError) as ex:
            logger.warning("Timeout on GET %s: %s", path, ex)
            raise tornado.web.HTTPError(status_code=504, reason="A timeout occurred in Agent Builder server")
        except Exception as ex:
            if isinstance(ex, (DuplicateAgentError, DuplicateAgentMetadataError)):
                logger.warning("Duplicate entry on GET %s: %s", path, ex)
                raise tornado.web.HTTPError(status_code=409, reason=str(ex))
            if isinstance(ex, AgentBuilderStoreError):
                logger.error("Store error on GET %s: %s", path, ex)
                raise tornado.web.HTTPError(status_code=500, reason=str(ex))
            logger.exception("Unhandled error on GET %s", path)
            raise tornado.web.HTTPError(status_code=500, reason="An exception occurred in Agent Builder server")

    def validate_payload(self, payload: Dict[str, Any]) -> Any:
        """Override in subclasses to construct and return a Pydantic request model.

        The returned object is passed directly to ``process()``.
        Default implementation returns the raw payload dict (for handlers
        that don't need structured validation, e.g. GET-only handlers).
        """
        return payload

    async def process(self, request: Any) -> Any:
        raise NotImplementedError

    async def do_process(self, request: Any) -> Any:
        return await self.process(request)
