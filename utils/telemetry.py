"""Elasticsearch telemetry: opt-in logging and app startup wiring."""

import logging
from typing import Any, Dict, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from agent_builder.utils.constants import (
    ES_EVENT_LOOP_HEALTH_INDEX,
    ES_HANDLER_IO_INDEX,
    ES_LLM_CALLS_INDEX,
    ES_TRACES_INDEX,
    TELEMETRY_ENV_CONFIG,
)


# ── startup (ENABLE_TELEMETRY=true) ────────────────────────────────────


def _index_already_exists(exc: BaseException) -> bool:
    """True when ES index create lost a race to another pod/process."""
    if type(exc).__name__ == "ResourceAlreadyExistsException":
        return True
    if getattr(exc, "error", None) == "resource_already_exists_exception":
        return True
    meta = getattr(exc, "meta", None)
    body = getattr(meta, "body", None) if meta is not None else None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("type") == "resource_already_exists_exception":
            return True
    return "resource_already_exists" in str(exc).lower()


async def _ensure_es_index(es_client, index: str, body: Dict[str, Any]) -> None:
    """Create an ES index if missing; tolerate concurrent startup (K8s pods)."""
    if await es_client.indices.exists(index=index):
        return
    try:
        await es_client.indices.create(index=index, body=body, ignore=400)
        tlog_info("Created ES index: %s", index)
    except Exception as exc:
        if _index_already_exists(exc):
            return
        raise


async def initialize_telemetry(app) -> None:
    """Connect to ES, ensure indices exist, start writer + health sampler."""
    # Lazy imports avoid circular dependency with es_logging_handler.
    from tornado.ioloop import IOLoop

    from agent_builder.utils.es_logging_handler import ElasticsearchCallbackHandler
    from agent_builder.utils.handler_io_telemetry import (
        EVENT_LOOP_HEALTH_MAPPING,
        HANDLER_IO_MAPPING,
        HEALTH_SAMPLE_INTERVAL_SEC,
        start_event_loop_health_sampler,
    )

    app.telemetry_enabled = TELEMETRY_ENV_CONFIG["enabled"]
    app.es_client = None

    if not app.telemetry_enabled:
        tlog_info("ES telemetry disabled (set ENABLE_TELEMETRY=true to enable)")
        return

    try:
        from tools_io.elasticsearch.es_client_provider import ESClientProvider

        # Quiet third-party ES HTTP logs unless ES_TELEMETRY_LOGS=true
        maybe_silence_es_client_logs()

        es = await ESClientProvider.get_es_client_by_es_server_type("MONITORING")
        app.es_client = getattr(es, "es", es)

        # Create indices on first run if they do not exist yet (ignore=400 for multi-pod races)
        await _ensure_es_index(app.es_client, ES_TRACES_INDEX, ElasticsearchCallbackHandler.INDEX_MAPPING)
        await _ensure_es_index(app.es_client, ES_LLM_CALLS_INDEX, ElasticsearchCallbackHandler.LLM_CALLS_INDEX_MAPPING)
        await _ensure_es_index(app.es_client, ES_HANDLER_IO_INDEX, HANDLER_IO_MAPPING)
        await _ensure_es_index(app.es_client, ES_EVENT_LOOP_HEALTH_INDEX, EVENT_LOOP_HEALTH_MAPPING)

        batch_size = TELEMETRY_ENV_CONFIG["batch_size"]
        flush_interval = TELEMETRY_ENV_CONFIG["flush_interval"]
        max_queue_size = TELEMETRY_ENV_CONFIG["max_queue_size"]

        # Background writer runs on its own thread/loop (isolated from Tornado IOLoop)
        ElasticsearchCallbackHandler.start_background_worker(
            es_client_config=ElasticsearchCallbackHandler.client_config_from(app.es_client),
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
        )
        tlog_info(
            "ES telemetry initialized (batch_size=%d, flush_interval=%.2fs, max_queue_size=%d)",
            batch_size,
            flush_interval,
            max_queue_size,
        )

        app._health_sampler_thread = start_event_loop_health_sampler(
            interval_sec=HEALTH_SAMPLE_INTERVAL_SEC,
            io_loop=IOLoop.current(), 
        )
        tlog_info(
            "Event loop health sampler started (interval=%.1fs)",
            HEALTH_SAMPLE_INTERVAL_SEC,
        )
    except Exception as exc:
        _logger.error("ES telemetry initialization failed: %s", exc, exc_info=True)
        app.telemetry_enabled = False
        app.es_client = None


# ── invoke tracing (LangChain RunnableConfig) ──────────────────────────


def extract_agent_id_mappings(agent_doc: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Walk an agent doc and build name→MongoID maps for tasks and tools."""
    task_id_map: Dict[str, str] = {}
    tool_id_map: Dict[str, str] = {}

    def process_task_doc(task_doc: Dict[str, Any], is_task_as_tool: bool = False) -> None:
        if not task_doc:
            return

        task_name = task_doc.get("name")
        task_id = task_doc.get("_id")
        if task_name and task_id:
            task_id_map[task_name] = str(task_id)
            if is_task_as_tool:
                tool_id_map[f"run_{task_name}"] = str(task_id)

        for tool_meta in task_doc.get("tools", []) or []:
            tname = tool_meta.get("name")
            tid = tool_meta.get("_id")
            if tname and tid:
                tool_id_map[tname] = str(tid)

        for nested_task in task_doc.get("task_as_tools", []) or []:
            process_task_doc(nested_task, is_task_as_tool=True)

    for task_doc in agent_doc.get("tasks", []) or []:
        process_task_doc(task_doc)

    if agent_doc.get("task_as_router"):
        process_task_doc(agent_doc["task_as_router"])

    for embedded_agent in agent_doc.get("agent_as_task", []) or []:
        for task_doc in embedded_agent.get("tasks", []) or []:
            process_task_doc(task_doc)

    return task_id_map, tool_id_map


# ── per-request handler (trace callback) ───────────────────────────────


def init_handler_telemetry(application: Any) -> Tuple[bool, Any]:
    """Create a per-request ES callback handler, or (False, None) if unavailable."""
    if not getattr(application, "telemetry_enabled", False):
        return False, None

    es_client = getattr(application, "es_client", None)
    if es_client is None:
        return False, None

    try:
        from agent_builder.utils.es_logging_handler import ElasticsearchCallbackHandler

        return True, ElasticsearchCallbackHandler(es_client=es_client)
    except Exception:
        _logger.warning(
            "Failed to create ElasticsearchCallbackHandler; disabling telemetry for this request",
            exc_info=True,
        )
        return False, None


def build_invoke_telemetry_config(handler: Any, request: Any, ctx: Any) -> Optional[RunnableConfig]:
    """Build RunnableConfig that wires the per-request ES callback handler."""
    if not getattr(handler, "telemetry_enabled", False) or not getattr(handler, "es_handler", None):
        return None

    agent_doc = getattr(ctx, "agent_doc", None) or {}
    task_id_map, tool_id_map = extract_agent_id_mappings(agent_doc)

    return RunnableConfig(
        callbacks=[handler.es_handler],
        metadata={
            "session_id": request.sessionId,
            "request_id": getattr(request, "id", None),
            "agent_id": str(agent_doc.get("_id", ctx.agent_id)),
            "agent_name": agent_doc.get("name", "unknown_agent"),
            "task_id_map": task_id_map,
            "tool_id_map": tool_id_map,
        },
    )



# ── opt-in verbose logging (ES_TELEMETRY_LOGS=true) ────────────────────
# debug/info → agent_builder.telemetry (gated). warning+ → module logger (always).

TELEMETRY_LOGS = TELEMETRY_ENV_CONFIG["logs"]

_tlog = logging.getLogger("agent_builder.telemetry")
_logger = logging.getLogger(__name__)


def tlog_debug(msg: str, *args, **kwargs) -> None:
    if TELEMETRY_LOGS:
        _tlog.debug(msg, *args, **kwargs)


def tlog_info(msg: str, *args, **kwargs) -> None:
    if TELEMETRY_LOGS:
        _tlog.info(msg, *args, **kwargs)


def tlog_warning(msg: str, *args, **kwargs) -> None:
    _logger.warning(msg, *args, **kwargs)


def tlog_error(msg: str, *args, **kwargs) -> None:
    _logger.error(msg, *args, **kwargs)


def tlog_exception(msg: str, *args, **kwargs) -> None:
    _logger.exception(msg, *args, **kwargs)


def maybe_silence_es_client_logs() -> None:
    """Quiet third-party ES HTTP noise unless ES_TELEMETRY_LOGS=true."""
    if TELEMETRY_LOGS:
        return
    for name in (
        "elasticsearch",
        "elastic_transport",
        "tools_io.elasticsearch",
        "tools_frameworks",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


