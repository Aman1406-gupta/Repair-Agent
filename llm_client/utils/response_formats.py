"""Typed response block formats for invoke-agent streaming.

Blocks are plain dicts that go inside a ``content.delta`` event's ``content`` list.
Build them with these helpers (or write the dict yourself — they pass through
verbatim). Each builder takes an optional ``index``; when omitted, the event
builder assigns it from the block's position in the list.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def text_block(text: str, *, index: Optional[int] = None) -> Dict[str, Any]:
    """``response.text.delta`` block."""
    block: Dict[str, Any] = {"type": "response.text.delta", "text": text}
    if index is not None:
        block["index"] = index
    return block


def thinking_block(text: str, *, index: Optional[int] = None) -> Dict[str, Any]:
    """``thinking.text.delta`` block."""
    block: Dict[str, Any] = {"type": "thinking.text.delta", "text": text}
    if index is not None:
        block["index"] = index
    return block


def image_block(
    url: str,
    *,
    mime_type: Optional[str] = None,
    alt_text: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """``response.image`` block."""
    block: Dict[str, Any] = {"type": "response.image", "url": url}
    if mime_type is not None:
        block["mimeType"] = mime_type
    if alt_text is not None:
        block["altText"] = alt_text
    if index is not None:
        block["index"] = index
    return block


def video_block(
    url: str,
    *,
    mime_type: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """``response.video`` block."""
    block: Dict[str, Any] = {"type": "response.video", "url": url}
    if mime_type is not None:
        block["mimeType"] = mime_type
    if index is not None:
        block["index"] = index
    return block


def content_block(type: str, **fields: Any) -> Dict[str, Any]:
    """Generic block for any other/future type, e.g. ``content_block("response.idea", summary="x")``."""
    return {"type": type, **fields}


#: Plain response-text block types. These form the assistant message body (joined as
#: text) and are never preserved/forwarded as standalone typed blocks.
_TEXT_BLOCK_TYPES = frozenset({"response.text", "response.text.delta"})

#: ``type`` prefixes of typed blocks preserved as standalone blocks — forwarded as their
#: own ``content.delta`` block and folded into durable history. Covers media
#: (``response.image`` / ``response.video`` / future ``response.*``), ``thinking.*``
#: (reasoning) and ``citation.*`` (sources). Plain response text is excluded.
_NON_TEXT_BLOCK_PREFIXES = ("response.", "thinking.", "citation.")


def is_non_text_block_type(block_type: str) -> bool:
    """True for a typed block preserved as a standalone (non-plain-text) block.

    Matches media ``response.*`` (e.g. ``response.image`` / ``response.video``),
    ``thinking.*`` and ``citation.*`` blocks; excludes the plain-text body blocks
    ``response.text`` / ``response.text.delta``. Type-agnostic, so new block kinds under
    these prefixes need no change at the call sites.
    """
    return (
        block_type.startswith(_NON_TEXT_BLOCK_PREFIXES)
        and block_type not in _TEXT_BLOCK_TYPES
    )
