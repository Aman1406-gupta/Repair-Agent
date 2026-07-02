from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessageChunk, BaseMessage

from agent_builder.llm_client.utils.response_formats import is_non_text_block_type
from agent_builder.utils.constants import _RAW_REMOTE_CONTENT


# Typed invoke streaming ↔ LangChain


def _delta_item_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or "").replace(" ", "").lower()


def join_text_deltas(event: dict[str, Any]) -> str:
    """Join ``*.text.delta`` ``text`` fields from one typed SSE ``content`` array."""
    if not isinstance(event, dict):
        return ""

    content = event.get("content")
    if not isinstance(content, list):
        return ""

    fragments: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = _delta_item_type(block)
        if not block_type.endswith(".text.delta"):
            continue

        text = block.get("text")
        if isinstance(text, str) and text:
            fragments.append(text)

    return "".join(fragments)




def extract_response_text_delta(
    msg: BaseMessage,
) -> Optional[Tuple[str, int]]:
    """``(text, index)`` for a non-empty local ``AIMessageChunk``, else ``None``."""
    if not isinstance(msg, AIMessageChunk):
        return None
    text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    if not text:
        return None
    idx = int(msg.additional_kwargs.get("content_delta_index", 0))
    return text, idx


def extract_response_blocks(msg: BaseMessage) -> List[dict[str, Any]]:
    """Non-text typed blocks carried on a locally-produced message.

    Reads ``additional_kwargs[_RAW_REMOTE_CONTENT]`` (the canonical block carrier) and
    returns every non-text block — media ``response.*`` (e.g. ``response.image`` /
    ``response.video``), ``thinking.*`` and ``citation.*`` — excluding the plain-text body
    (``response.text`` / ``response.text.delta``). Type-agnostic: new block kinds under those
    prefixes need no change here. Remote passthrough blocks are forwarded via raw SSE instead,
    so this is only meaningful for locally-produced messages.
    """
    raw = (getattr(msg, "additional_kwargs", None) or {}).get(_RAW_REMOTE_CONTENT)
    if not isinstance(raw, list):
        return []
    blocks: List[dict[str, Any]] = []
    for block in raw:
        if not isinstance(block, dict):
            continue
        if is_non_text_block_type(str(block.get("type") or "")):
            blocks.append(block)
    return blocks


def format_sse_event(event_dict: dict[str, Any]) -> str:
    """Format one typed event as an SSE ``data:`` line."""
    return f"data: {json.dumps(event_dict, default=str)}\n\n"
