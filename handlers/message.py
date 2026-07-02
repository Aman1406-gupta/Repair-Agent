"""
``/message`` endpoints for background-mode long-running tasks.

* ``POST /message``        — backend polls accumulated typed chunks (``content.delta`` …)
  produced after a ``mode.background`` switch, ending with the run-level ``stream.completed``.
* ``POST /message/ingest`` — a long-running remote task pushes its own chunks into Redis so
  ``/message`` can serve them even though they originate outside this pod.

The transient chunk buffer is purely for stream replay; the canonical message history is
persisted in the session envelope by the invoke finalize path.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage

from agent_builder.handlers.core.base_handler import BaseBuilderHandler
from agent_builder.handlers.core.requests import (
    MessageIngestRequest,
    MessagePollRequest,
)
from agent_builder.llm_client.utils.response_formats import is_non_text_block_type
from agent_builder.storage.utils.state_serializer import (
    serialize_session_data,
)
from agent_builder.utils.constants import _RAW_REMOTE_CONTENT

logger = logging.getLogger(__name__)

#: terminal frame type → ``stream_status`` to stamp on the assembled assistant message.
_TERMINAL_STATUS = {
    "stream.completed": "completed",
    "stream.failed": "failed",
    "stream.interrupted": "interrupted",
}


def _content_blocks_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All content blocks across the generation's ``content.delta`` frames, each keeping the
    verbatim ``index`` from its packet, in streamed order.

    Streaming ``*.text.delta`` blocks that share an index are accumulated into a single block
    with the ``.delta`` suffix dropped — ``response.text.delta`` → one ``response.text`` (the
    assistant body), ``thinking.text.delta`` → one ``thinking.text`` (the reasoning) —
    mirroring how the live/drain path merges chunk lists by index. Every other block
    (``citation.*``, media ``response.image`` / ``response.video``) passes through unchanged.
    No index is reassigned and nothing is re-sorted, so the order matches exactly what the
    remote streamed (e.g. ``thinking@0`` then ``text@1`` then media at ``2+``).
    """
    out: List[Dict[str, Any]] = []
    accum: Dict[tuple, Dict[str, Any]] = {}  # (base_type, index) -> the merged block held in `out`
    for event in chunks:
        if not isinstance(event, dict) or event.get("type") != "content.delta":
            continue
        for blk in event.get("content") or []:
            if not isinstance(blk, dict):
                continue
            btype = str(blk.get("type") or "")
            text = blk.get("text")
            if btype.endswith(".delta") and isinstance(text, str):
                if not text:
                    continue
                base = btype[: -len(".delta")]
                idx = _safe_index(blk)
                key = (base, idx)
                merged = accum.get(key)
                if merged is not None:
                    merged["text"] += text
                else:
                    merged = {"type": base, "index": idx, "text": text}
                    accum[key] = merged
                    out.append(merged)
            elif is_non_text_block_type(btype):
                out.append(dict(blk))
    return out


def _fold_into_session_messages(
    messages: List[BaseMessage], request_id: str, text: str, status: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    response_id: Optional[str] = None,
) -> bool:
    """Fold the assembled background reply into the canonical history in place.

    Targets the background placeholder ``AIMessage`` for this ``request_id`` (the empty
    message yielded on the remote's ``mode.background`` handoff, which carries the
    request id). When the placeholder is missing (e.g. the terminal ingest raced the
    handoff persist), a new ``AIMessage`` with this ``request_id`` is appended instead —
    no positional guessing, so a previous turn's reply is never touched. Sets the content
    (only if still empty — never clobbers a real reply), merges ``blocks`` (the full
    streamed content list, each carrying its verbatim packet ``index``) onto the canonical
    ``_RAW_REMOTE_CONTENT`` carrier, and stamps the terminal ``stream_status``. Returns
    ``True`` if changed.
    """
    blocks = blocks or []
    target: Optional[AIMessage] = next(
        (m for m in messages if isinstance(m, AIMessage) and m.id == request_id), None,
    )

    if target is None:
        if not text and not blocks:
            return False
        new_msg = AIMessage(content=text, id=request_id,
                            response_metadata={"stream_status": status})
        raw = _merge_blocks([], blocks)
        if raw:
            new_msg.additional_kwargs[_RAW_REMOTE_CONTENT] = raw
        if response_id:
            new_msg.additional_kwargs["responseId"] = response_id
        messages.append(new_msg)
        return True

    changed = False
    if response_id and not target.additional_kwargs.get("responseId"):
        target.additional_kwargs["responseId"] = response_id
        changed = True
    if text and not (isinstance(target.content, str) and target.content.strip()):
        target.content = text
        changed = True
    if blocks:
        # Append the streamed blocks after whatever the carrier already holds (e.g. the
        # inline preamble's thinking.* folded at the stream→background handoff), keeping
        # each block's verbatim index and streamed order. De-duplicated by ``_block_key``
        # so a retried/duplicate terminal ingest (the SDK retries on transport/5xx) does
        # not append the same block twice.
        existing = target.additional_kwargs.get(_RAW_REMOTE_CONTENT)
        existing = existing if isinstance(existing, list) else []
        merged = _merge_blocks(existing, blocks)
        if merged != existing:
            target.additional_kwargs[_RAW_REMOTE_CONTENT] = merged
            changed = True
    meta = dict(target.response_metadata or {})
    if meta.get("stream_status") != status:
        meta["stream_status"] = status
        target.response_metadata = meta
        changed = True
    return changed


def _block_key(b: Dict[str, Any]) -> tuple:
    """Identity for de-duping content blocks across folds (handoff carrier + terminal
    ingest, and any re-finalize). Citations carry ``id``, media carry ``url``, text/thinking
    fall back to ``(index, text)``."""
    return (b.get("type"), b.get("id"), b.get("url"), b.get("index"), b.get("text"))


def _safe_index(blk: Dict[str, Any]) -> int:
    idx = blk.get("index")
    return idx if isinstance(idx, int) and not isinstance(idx, bool) else 0


def _merge_blocks(
    existing: List[Dict[str, Any]], new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Carrier blocks followed by the newly-streamed blocks, de-duplicated by ``_block_key``.

    Order and each block's verbatim ``index`` are preserved exactly as received — nothing is
    sorted or re-numbered. The carrier (e.g. an inline thinking preamble at index 0) comes
    first, then the terminal-ingest blocks (answer text at its own index, media after)."""
    out: List[Dict[str, Any]] = [dict(b) for b in existing if isinstance(b, dict)]
    seen = {_block_key(b) for b in out}
    for b in new:
        if not isinstance(b, dict):
            continue
        key = _block_key(b)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(b))
    return out


class MessageHandler(BaseBuilderHandler):
    """``POST /message``: return accumulated background chunks with ``sequence > seq``."""

    def validate_payload(self, payload: Dict[str, Any]) -> MessagePollRequest:
        return MessagePollRequest(**payload)

    async def process(self, request: MessagePollRequest) -> Dict[str, Any]:
        meta = await self.redis_client.get_bg_meta(request.sessionId, request.requestId)

        if meta is None:
            # No background generation recorded (not switched yet, or TTL-expired).
            return {
                "generationId": request.generationId,
                "status": "unknown",
                "chunks": [],
            }

        current_gen = meta.get("generationId")
        status = meta.get("status", "running")

        # Same (or unspecified) generation → incremental from the caller's cursor.
        # Different generation (pod restarted from scratch) → full replay so the
        # backend resets to the new generation's sequence space.
        same_generation = request.generationId in (None, current_gen)
        after_seq = request.seq if same_generation else None

        chunks = await self.redis_client.get_bg_chunks(
            request.sessionId, request.requestId, current_gen, after_seq,
        )

        logger.debug(
            "Message poll | rid=%s gen=%s after_seq=%s returned=%d status=%s",
            request.requestId, current_gen, after_seq, len(chunks), status,
        )
        return {
            "generationId": current_gen,
            "status": status,
            "chunks": chunks,
        }


class MessageIngestHandler(BaseBuilderHandler):
    """``POST /message/ingest``: persist chunks pushed by a long-running remote task."""

    def validate_payload(self, payload: Dict[str, Any]) -> MessageIngestRequest:
        return MessageIngestRequest(**payload)

    async def process(self, request: MessageIngestRequest) -> Dict[str, Any]:
        accepted = await self.redis_client.append_bg_chunks(
            request.sessionId, request.requestId, request.generationId, request.events,
        )

        # When the run terminates (the remote pushed stream.completed/failed/interrupted),
        # assemble the generation's chunks into one assistant message and persist it into the
        # canonical session envelope (agent_builder:{sessionId}) + Mongo, so the next stateful
        # turn has the reply (the empty background placeholder gets filled).
        terminal_status: Optional[str] = None
        for ev in request.events:
            t = ev.get("type") if isinstance(ev, dict) else None
            if t in _TERMINAL_STATUS:
                terminal_status = _TERMINAL_STATUS[t]
        if terminal_status is not None:
            await self._finalize_session_from_chunks(
                request.sessionId, request.requestId, request.generationId, terminal_status,
            )

        meta = await self.redis_client.get_bg_meta(request.sessionId, request.requestId)
        status = (meta or {}).get("status", "running")

        logger.debug(
            "Message ingest | rid=%s gen=%s accepted=%d status=%s terminal=%s",
            request.requestId, request.generationId, accepted, status, terminal_status,
        )
        return {
            "accepted": accepted,
            "generationId": request.generationId,
            "status": status,
        }

    async def _finalize_session_from_chunks(
        self, session_id: str, request_id: str, generation_id: str, status: str,
    ) -> None:
        """Combine the generation's ``content.delta`` chunks and fold the reply into the
        session envelope's assistant message, then persist (Redis + Mongo)."""
        chunks = await self.redis_client.get_bg_chunks(
            session_id, request_id, generation_id, after_seq=None,
        )
        content_blocks = _content_blocks_from_chunks(chunks)
        text = "".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "response.text"
        )

        session_data = await self.redis_client.get_extended_session_data(session_id)
        if not session_data:
            logger.warning(
                "Ingest terminal but no session envelope to finalize | sid=%s rid=%s",
                session_id, request_id,
            )
            return

        graph_state = session_data.get("graph_state") or {}
        messages = graph_state.get("messages")
        if not isinstance(messages, list):
            logger.warning("Session envelope has no messages list | sid=%s", session_id)
            return

        msg_id = graph_state.get("response_id") or None
        changed = _fold_into_session_messages(messages, request_id, text, status, content_blocks, response_id=msg_id)

        if not changed:
            return

        await self.redis_client.set_extended_session_data(
            session_id, serialize_session_data(session_data),
        )
        # Mark execution completed for parity with the drain/sync finalize paths — but only
        # when this request still owns exec_status (don't clobber a newer running turn).
        exec_status = await self.redis_client.get_exec_status(session_id)
        if exec_status is None or exec_status.get("requestId") == request_id:
            await self.redis_client.set_exec_status(session_id, "completed", request_id)

        asyncio.ensure_future(self._save_session_to_mongo(session_id, session_data))

        logger.info(
            "Background session finalized from ingest | sid=%s rid=%s status=%s chars=%d",
            session_id, request_id, status, len(text),
        )
