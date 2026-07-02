"""Monotonic ``sequence`` counter for invoke-agent typed streaming (``content.delta``, etc.)."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from agent_builder.llm_client.utils.response_formats import (
    image_block,
    text_block,
    thinking_block,
    video_block,
)


class TypedStreamSequencer:
    """Monotonic ``sequence`` for invoke typed SSE frames."""

    __slots__ = ("_request_id", "_response_id", "_seq", "_thinking_content_idx")

    def __init__(self, request_id: str, response_id: str | None = None) -> None:
        self._request_id = request_id
        self._response_id = response_id
        self._seq = 0
        self._thinking_content_idx = 0

    def _take_sequence(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def consume_passthrough_sse(self, raw_sse: str) -> None:
        """Bump ``sequence`` past the highest remote ``sequence`` in forwarded SSE."""
        for line in raw_sse.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                seq = obj.get("sequence")
                if isinstance(seq, int) and not isinstance(seq, bool):
                    self._seq = max(self._seq, seq + 1)

    def stream_start(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": "stream.start",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "created": int(time.time()),
        }
        if self._response_id is not None:
            event["responseId"] = self._response_id
        return event

    def content_delta(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        """``content.delta`` carrying one or more typed blocks.

        Blocks are forwarded verbatim; a missing ``index`` is filled from the block's
        position in the list, an explicit ``index`` is never overwritten.
        """
        content = []
        for i, block in enumerate(blocks):
            blk = dict(block)
            blk.setdefault("index", i)
            content.append(blk)
        return {
            "type": "content.delta",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "content": content,
        }

    def content_delta_text(self, text: str, *, content_index: int = 0) -> dict[str, Any]:
        return self.content_delta_block(text_block(text), content_index=content_index)

    def content_delta_block(
        self,
        block: dict[str, Any],
        *,
        content_index: int = 0,
    ) -> dict[str, Any]:
        """``content.delta`` wrapping one arbitrary typed block (e.g. ``response.image``).

        The single primitive for streaming any non-text ``response.*`` block. The block is
        forwarded verbatim; ``index`` is injected when absent. Type-agnostic so future block
        kinds need no change here.
        """
        blk = dict(block)
        blk.setdefault("index", content_index)
        return {
            "type": "content.delta",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "content": [blk],
        }

    def content_delta_image(
        self,
        url: str,
        *,
        mime_type: Optional[str] = None,
        alt_text: Optional[str] = None,
        content_index: int = 0,
    ) -> dict[str, Any]:
        return self.content_delta_block(
            image_block(url, mime_type=mime_type, alt_text=alt_text),
            content_index=content_index,
        )

    def content_delta_video(
        self,
        url: str,
        *,
        mime_type: Optional[str] = None,
        content_index: int = 0,
    ) -> dict[str, Any]:
        return self.content_delta_block(
            video_block(url, mime_type=mime_type), content_index=content_index,
        )

    def raw_additional(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Re-emit a stored non-``content.delta`` frame (e.g. ``entity.chunk``).

        The frame is forwarded verbatim except ``sequence``/``id``, which are re-stamped
        into this stream's space — the persisted values are ordering/origin metadata only.
        """
        return {**frame, "sequence": self._take_sequence(), "id": self._request_id}

    def content_delta_thinking_text(
        self,
        text: str,
        *,
        content_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """``content.delta`` with ``thinking.text.delta`` (auto ``index`` if omitted)."""
        if content_index is None:
            idx = self._thinking_content_idx
            self._thinking_content_idx += 1
        else:
            idx = content_index
        return self.content_delta_block(thinking_block(text), content_index=idx)

    def mode_background(
        self,
        generation_id: str,
        *,
        polling_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Non-terminal control frame: the live wire is closing; poll ``/message``.

        Carries ``generationId`` so the backend can poll and, on a pod restart that
        produces a fresh generation, detect the change. ``pollingId`` (the id to poll
        ``/message`` with) is included when provided. Never signals completion.
        """
        frame: dict[str, Any] = {
            "type": "mode.background",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "generationId": generation_id,
        }
        if polling_id is not None:
            frame["pollingId"] = polling_id
        return frame

    def stream_completed(self, usage: Optional[dict[str, Any]]) -> dict[str, Any]:
        return {
            "type": "stream.completed",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "usage": usage if isinstance(usage, dict) else {},
        }

    def stream_failed(self, message: str, *, retryable: bool = False) -> dict[str, Any]:
        return {
            "type": "stream.failed",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "error": {"message": message, "retryable": retryable},
        }
    
    def stream_interrupted(self, usage: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "type": "stream.interrupted",
            "sequence": self._take_sequence(),
            "id": self._request_id,
            "usage": usage if isinstance(usage, dict) else {},
        }

