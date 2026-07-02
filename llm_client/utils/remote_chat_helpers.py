"""Remote copilot HTTP/SSE helpers (stream failures, state merge, headers)."""

from __future__ import annotations

from typing import Any, Dict

from agent_builder.utils.constants import REMOTE_STREAM_FAILED_ATTR


def is_remote_stream_failed(exc: BaseException) -> bool:
    """True when *exc* was raised by :func:`raise_remote_stream_failed` (remote ``stream.failed``)."""
    return getattr(exc, REMOTE_STREAM_FAILED_ATTR, False) is True


def raise_remote_stream_failed(message: str, *, retryable: bool = False) -> None:
    """Raise a marked :exc:`RuntimeError` for remote ``stream.failed`` (``retryable`` on the instance)."""
    e = RuntimeError(message)
    setattr(e, REMOTE_STREAM_FAILED_ATTR, True)
    setattr(e, "retryable", retryable)
    raise e


def remote_typed_event_kind(chunk_data: dict) -> str:
    t = chunk_data.get("type")
    if not isinstance(t, str):
        return ""
    return t.replace(" ", "").lower()


def is_valid_last_active_task_obj(x: Any) -> bool:
    """``lastActiveTask`` with non-empty ``path`` and ``depth`` (empty path = invalid)."""
    if not isinstance(x, dict) or set(x.keys()) != {"path", "depth"}:
        return False
    return (isinstance(x["path"], list) and len(x["path"]) > 0 and isinstance(x["depth"], int))


def sanitize_forwarded_headers(headers: Dict[str, str]) -> dict[str, str]:
    """Drop headers that must not be forwarded to the downstream remote endpoint."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("content-length", "transfer-encoding", "host")
    }
