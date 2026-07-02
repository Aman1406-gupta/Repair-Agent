"""ContextVar-based signal for LLM-level interrupt and stream-failure handling."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_builder.utils.constants import ENVELOPE_KEY

logger = logging.getLogger(__name__)


@dataclass
class StreamSignals:
    """Per-request signal bundle shared between handler and LLM layer.

    * ``interrupt_event`` — set by the handler (via InterruptPoller); read by
      ``_astream()`` to cut the HTTP stream early.
    * ``status`` / ``error`` — set by ``_astream()`` when it terminates
      abnormally; read by ``_agenerate_with_cache()`` to strip tool calls.
    """

    interrupt_event: Optional[asyncio.Event] = None
    status: Optional[str] = None
    error: Optional[Dict[str, Any]] = field(default=None)
    # Set by ``_astream()`` when the remote emits ``mode.background``: carries
    # ``{"generationId", "messageUrl"}`` so the handler can hand off to ``/message``.
    background: Optional[Dict[str, Any]] = field(default=None)

    def check_interrupted(self) -> bool:
        return self.interrupt_event is not None and self.interrupt_event.is_set()

    def reset_status(self) -> None:
        self.status = None
        self.error = None
        self.background = None


STREAM_SIGNALS: contextvars.ContextVar[StreamSignals] = contextvars.ContextVar(
    "_stream_signals", default=StreamSignals()
)


def strip_tool_calls_from_result(
    result: ChatResult,
    status: str,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    """Remove tool calls from all generations and tag ``response_metadata``.

    Ensures at least one generation exists so downstream ``ainvoke()``
    never receives an empty ``ChatResult``.
    """
    if not result.generations:
        result.generations = [
            ChatGeneration(message=AIMessage(content="")),
        ]

    for gen in result.generations:
        msg = gen.message
        if hasattr(msg, "tool_calls"):
            msg.tool_calls = []
        if hasattr(msg, "invalid_tool_calls"):
            msg.invalid_tool_calls = []
        if hasattr(msg, "tool_call_chunks"):
            msg.tool_call_chunks = []

        meta = dict(msg.response_metadata or {})
        meta["stream_status"] = status
        if error:
            meta["stream_error"] = error
        msg.response_metadata = meta

        if error:
            envelope = msg.additional_kwargs.setdefault(ENVELOPE_KEY, {})
            envelope["error"] = error
