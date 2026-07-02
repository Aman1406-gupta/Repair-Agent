"""
Per-request handler IO telemetry and event-loop lag sampling.

Accumulates Mongo/Redis operation timings via ContextVar-scoped state,
then flushes a single ES document per request on completion.
"""

import asyncio
import functools
import os
import socket
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agent_builder.utils.constants import (
    ES_EVENT_LOOP_HEALTH_INDEX,
    ES_HANDLER_IO_INDEX,
    TELEMETRY_ENV_CONFIG,
)
from agent_builder.utils.es_logging_handler import ElasticsearchCallbackHandler
from agent_builder.utils.telemetry import tlog_exception

# ── env-configurable caps (see TELEMETRY_ENV_CONFIG in constants.py) ───────
MAX_OPS = TELEMETRY_ENV_CONFIG["max_ops"]
HEALTH_SAMPLE_INTERVAL_SEC = TELEMETRY_ENV_CONFIG["health_interval"]

_INSTANCE_ID = os.environ.get("HOSTNAME", socket.gethostname())

# ── ES index mappings ──────────────────────────────────────────────────

HANDLER_IO_MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "timestamp": {"type": "date"},
            "route": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "request_id": {"type": "keyword"},
            "http_status": {"type": "short"},
            "success": {"type": "boolean"},
            "elapsed_ms": {"type": "float"},
            "mongo_total_ms": {"type": "float"},
            "redis_total_ms": {"type": "float"},
            "mongo_count": {"type": "integer"},
            "redis_count": {"type": "integer"},
            "mongo_ops": {
                "type": "object",
                "properties": {
                    "op": {"type": "keyword"},
                    "collection": {"type": "keyword"},
                    "duration_ms": {"type": "float"},
                },
            },
            "redis_ops": {
                "type": "object",
                "properties": {
                    "op": {"type": "keyword"},
                    "key_kind": {"type": "keyword"},
                    "duration_ms": {"type": "float"},
                },
            },
            "mongo_truncated": {"type": "boolean"},
            "redis_truncated": {"type": "boolean"},
            "instance_id": {"type": "keyword"},
            "error_type": {"type": "keyword"},
            "error_message": {"type": "text"},
        },
    }
}

EVENT_LOOP_HEALTH_MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "timestamp": {"type": "date"},
            "task_count": {"type": "integer"},
            "tasks_pending": {"type": "integer"},
            "top_coroutine_types": {
                "type": "object",
                "properties": {
                    "name": {"type": "keyword"},
                    "count": {"type": "integer"},
                },
            },
            "instance_id": {"type": "keyword"},
        },
    }
}

# ── accumulator ────────────────────────────────────────────────────────


@dataclass
class _OpRecord:
    op: str
    target: str
    duration_ms: float


@dataclass
class IoAccumulator:
    route: str
    session_id: str = "-"
    request_id: str = "-"
    start_ns: int = field(default_factory=time.perf_counter_ns)
    mongo_ops: List[_OpRecord] = field(default_factory=list)
    redis_ops: List[_OpRecord] = field(default_factory=list)
    mongo_total_ms: float = 0.0
    redis_total_ms: float = 0.0
    mongo_count: int = 0
    redis_count: int = 0
    mongo_truncated: bool = False
    redis_truncated: bool = False
    flushed: bool = False


_accumulator: ContextVar[Optional[IoAccumulator]] = ContextVar(
    "handler_io_accumulator", default=None
)

# ── public helpers ─────────────────────────────────────────────────────


def is_active() -> bool:
    return _accumulator.get() is not None


def start_request(route: str) -> None:
    _accumulator.set(IoAccumulator(route=route))


def bind_request_context(*, session_id: str, request_id: str) -> None:
    """Snapshot correlation IDs on the active accumulator (survives log_context clear)."""
    acc = _accumulator.get()
    if acc is None:
        return
    acc.session_id = session_id
    acc.request_id = request_id


def request_elapsed_ms() -> Optional[float]:
    """Milliseconds since HTTP ``prepare()`` / ``start_request()`` for the active request."""
    acc = _accumulator.get()
    if acc is None:
        return None
    return (time.perf_counter_ns() - acc.start_ns) / 1_000_000


def record_mongo_op(op: str, collection: str, duration_ms: float) -> None:
    acc = _accumulator.get()
    if acc is None:
        return
    acc.mongo_total_ms += duration_ms
    acc.mongo_count += 1
    if len(acc.mongo_ops) < MAX_OPS:
        acc.mongo_ops.append(_OpRecord(op=op, target=collection, duration_ms=duration_ms))
    else:
        acc.mongo_truncated = True


def record_redis_op(op: str, key_kind: str, duration_ms: float) -> None:
    acc = _accumulator.get()
    if acc is None:
        return
    acc.redis_total_ms += duration_ms
    acc.redis_count += 1
    if len(acc.redis_ops) < MAX_OPS:
        acc.redis_ops.append(_OpRecord(op=op, target=key_kind, duration_ms=duration_ms))
    else:
        acc.redis_truncated = True


# ── doc builder + flush ────────────────────────────────────────────────


def build_handler_io_doc(
    handler: Any,
    success: bool = True,
    error_info: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    acc = _accumulator.get()
    if acc is None:
        return None

    elapsed_ms = (time.perf_counter_ns() - acc.start_ns) / 1_000_000
    http_status = handler.get_status() if hasattr(handler, "get_status") else 0

    doc: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "route": acc.route,
        "session_id": acc.session_id,
        "request_id": acc.request_id,
        "http_status": http_status,
        "success": success if error_info is None else False,
        "elapsed_ms": round(elapsed_ms, 2),
        "mongo_total_ms": round(acc.mongo_total_ms, 2),
        "redis_total_ms": round(acc.redis_total_ms, 2),
        "mongo_count": acc.mongo_count,
        "redis_count": acc.redis_count,
        "mongo_ops": [
            {"op": r.op, "collection": r.target, "duration_ms": round(r.duration_ms, 2)}
            for r in acc.mongo_ops
        ],
        "redis_ops": [
            {"op": r.op, "key_kind": r.target, "duration_ms": round(r.duration_ms, 2)}
            for r in acc.redis_ops
        ],
        "mongo_truncated": acc.mongo_truncated,
        "redis_truncated": acc.redis_truncated,
        "instance_id": _INSTANCE_ID,
    }

    if error_info:
        doc["error_type"] = error_info.get("type", "")
        doc["error_message"] = error_info.get("message", "")[:500]

    return doc


def flush_handler_io_doc(handler: Any) -> None:
    try:
        acc = _accumulator.get()
        if acc is None or acc.flushed:
            return

        if not getattr(handler, "telemetry_enabled", False):
            return

        acc.flushed = True

        doc = build_handler_io_doc(handler)
        if doc is None:
            return

        ElasticsearchCallbackHandler.enqueue_telemetry(ES_HANDLER_IO_INDEX, doc)
    except Exception:
        tlog_exception("Failed to flush handler I/O telemetry document")


# ── timing decorators ──────────────────────────────────────────────────


def _timed_mongo(op_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self, collection: str, *args, **kwargs):
            if not is_active():
                return await fn(self, collection, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return await fn(self, collection, *args, **kwargs)
            finally:
                try:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    record_mongo_op(op_name, collection, duration_ms)
                except Exception:
                    pass

        return wrapper

    return decorator


def _timed_redis(op_label: str, key_kind: str = "") -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if not is_active():
                return await fn(self, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return await fn(self, *args, **kwargs)
            finally:
                try:
                    resolved = key_kind or (str(args[0]) if args else "")
                    duration_ms = (time.perf_counter() - t0) * 1000
                    record_redis_op(op_label, resolved, duration_ms)
                except Exception:
                    pass

        return wrapper

    return decorator


# ── event-loop health sampler (non-intrusive) ─────────────────────────

_MAX_COROUTINE_TYPES = 10


def _snapshot_event_loop_health() -> Dict[str, Any]:
    """Read-only inspection of all asyncio tasks"""
    all_tasks = asyncio.all_tasks()

    pending = 0
    coro_counts: Dict[str, int] = {}

    for task in all_tasks:
        if not task.done():
            pending += 1

        coro = task.get_coro()
        name = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", "unknown")
        coro_counts[name] = coro_counts.get(name, 0) + 1

    top_types = sorted(coro_counts.items(), key=lambda x: x[1], reverse=True)[:_MAX_COROUTINE_TYPES]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_count": len(all_tasks),
        "tasks_pending": pending,
        "top_coroutine_types": [
            {"name": name, "count": count} for name, count in top_types
        ],
        "instance_id": _INSTANCE_ID,
    }


def start_event_loop_health_sampler(
    interval_sec: float = HEALTH_SAMPLE_INTERVAL_SEC,
    io_loop=None,
) -> threading.Thread:
    """Start a daemon thread that periodically snapshots event loop health.

    The sampler thread only schedules work; ``asyncio.all_tasks()`` runs on
    ``io_loop`` so it sees the request-serving event loop.
    """
    if io_loop is None:
        from tornado.ioloop import IOLoop

        io_loop = IOLoop.current()

    def _enqueue_snapshot() -> None:
        try:
            doc = _snapshot_event_loop_health()
            ElasticsearchCallbackHandler.enqueue_telemetry(
                ES_EVENT_LOOP_HEALTH_INDEX, doc,
            )
        except Exception:
            tlog_exception("Failed to enqueue event-loop health snapshot")

    def _sampler() -> None:
        while True:
            time.sleep(interval_sec)
            try:
                io_loop.add_callback(_enqueue_snapshot)
            except Exception:
                tlog_exception("Failed to schedule event-loop health snapshot")

    t = threading.Thread(
        target=_sampler,
        daemon=True,
        name="event-loop-health-sampler",
    )
    t.start()
    return t
