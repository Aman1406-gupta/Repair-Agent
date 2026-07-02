"""
Correlation logging infrastructure using contextvars.

Sets session_id and request_id on every log line automatically via a logging Filter.
"""

import contextvars
import logging

_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def set_log_context(*, session_id: str = None, request_id: str = None):
    if session_id is not None:
        _session_id.set(session_id)
    if request_id is not None:
        _request_id.set(request_id)


def clear_log_context():
    _session_id.set("-")
    _request_id.set("-")


class CorrelationFilter(logging.Filter):
    """Injects session_id and request_id into every LogRecord."""

    def filter(self, record):
        record.session_id = _session_id.get("-")
        record.request_id = _request_id.get("-")
        return True
