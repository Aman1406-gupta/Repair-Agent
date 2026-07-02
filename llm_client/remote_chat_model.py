"""BaseChatModel that proxies calls to a remote copilot API endpoint."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time as _time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from pydantic import ConfigDict, Field

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agent_builder.base.configs import HttpConfig
from agent_builder.utils.constants import (
    CLIENT_HTTP_HEADERS,
    CONFIG_VARIABLES,
    ENVELOPE_FIELDS,
    ENVELOPE_KEY,
    RAW_RESPONSE_KEY,
    STREAM,
    _RAW_REMOTE_CONTENT,
)

from agent_builder.llm_client.utils.remote_adapter import  state_to_request_payload,get_plain_text_from_content, api_response_to_lc_message
from agent_builder.llm_client.utils.interrupt import STREAM_SIGNALS, strip_tool_calls_from_result
from agent_builder.llm_client.utils.remote_chat_helpers import (
    raise_remote_stream_failed,
    remote_typed_event_kind,
    sanitize_forwarded_headers,
)
import asyncio
import time as _time


logger = logging.getLogger(__name__)

_CHUNK_LIST: contextvars.ContextVar[list | None] = (
    contextvars.ContextVar("_remote_chunk_list", default=None)
)


class RemoteChatModel(BaseChatModel):
    """Chat model that proxies to a remote copilot endpoint."""

    http_config: HttpConfig = Field(...)
    timeout: float = Field(default=120.0)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "remote_copilot"

    # ── HTTP plumbing ──────────────────────────────────────────────────

    def _proxy_url(self) -> Optional[str]:
        cfg = self.http_config
        if cfg.proxy_server and cfg.proxy_port:
            return f"http://{cfg.proxy_server}:{cfg.proxy_port}"
        return None

    def _client_kwargs(self) -> dict:
        kwargs: Dict[str, Any] = {"timeout": self.timeout}
        proxy = self._proxy_url()
        if proxy:
            kwargs["proxy"] = proxy
        return kwargs

    # ── _generate (sync fallback, required by ABC) ─────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(
            self._agenerate(messages, stop, run_manager, **kwargs),
        )

    # ── _agenerate (non-streaming) ─────────────────────────────────────

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_state = kwargs.pop("full_state", {})
        payload = state_to_request_payload(full_state)
        payload[STREAM] = False
        fwd_headers = (full_state.get(CONFIG_VARIABLES) or {}).get(CLIENT_HTTP_HEADERS)

        logger.debug("Remote LLM call (non-stream) | url=%s", self.http_config.url)
        t0 = _time.perf_counter()
        try:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                post_kwargs: Dict[str, Any] = {"json": payload}
                if fwd_headers:
                    post_kwargs["headers"] = sanitize_forwarded_headers(fwd_headers)
                resp = await client.post(self.http_config.url, **post_kwargs)
                resp.raise_for_status()
                data = resp.json()
            latency = (_time.perf_counter() - t0) * 1000
            logger.debug("Remote LLM response | status=%d latency=%.0fms", resp.status_code, latency)
        except httpx.TimeoutException:
            logger.warning("Remote LLM timeout | url=%s", self.http_config.url)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("Remote LLM HTTP error | status=%d url=%s", e.response.status_code, self.http_config.url)
            raise

        message = api_response_to_lc_message(data)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ── _astream (SSE streaming) ───────────────────────────────────────

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Yield one ChatGenerationChunk per SSE ``data:`` line."""
        signals = STREAM_SIGNALS.get()
        full_state = kwargs.pop("full_state", {})
        payload = state_to_request_payload(full_state)
        payload[STREAM] = True
        fwd_headers = (full_state.get(CONFIG_VARIABLES) or {}).get(CLIENT_HTTP_HEADERS)

        logger.debug("Remote LLM stream | url=%s", self.http_config.url)

        buffer = ""
        stream_kwargs: Dict[str, Any] = {"json": payload}
        if fwd_headers:

            stream_kwargs["headers"] = sanitize_forwarded_headers(fwd_headers)

        try:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                async with client.stream(
                    "POST", self.http_config.url, **stream_kwargs,
                ) as resp:
                    resp.raise_for_status()

                    async for chunk_bytes in resp.aiter_bytes():
                        if signals.check_interrupted():
                            signals.status = "interrupted"
                            return

                        raw_sse = chunk_bytes.decode("utf-8", errors="replace")
                        buffer += raw_sse
                        *lines, buffer = buffer.splitlines()
                        sent_raw_meta = False

                        for line in lines:
                            if not line.startswith("data: "):
                                continue

                            data_str = line[len("data: "):].strip()

                            if data_str == "[DONE]":
                                continue

                            try:
                                chunk_data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            kind = remote_typed_event_kind(chunk_data)
                            if kind == "stream.start":
                                # Handler already wrote ``stream.start``; do not yield duplicate passthrough.
                                continue
                            if kind == "stream.completed":
                                if isinstance(chunk_data, dict):
                                    envelope = {k: chunk_data.get(k) for k in ENVELOPE_FIELDS}
                                    envelope["is_remote_response"] = True
                                    term = AIMessageChunk(
                                        content="",
                                        id=str(chunk_data.get("id") or "") or None,
                                        chunk_position="last",
                                        additional_kwargs={ENVELOPE_KEY: envelope},
                                        response_metadata={RAW_RESPONSE_KEY: dict(chunk_data)},
                                    )
                                    gen_chunk = ChatGenerationChunk(message=term)
                                    self._capture_chunk(gen_chunk)
                                    yield gen_chunk
                                continue

                            if kind == "stream.failed":
                                signals.status = "failed"
                                signals.error = chunk_data.get("error", {})
                                return

                            if kind == "mode.background":
                                # The remote is handing the rest of the run off to
                                # ``/message`` (it will push chunks via /message/ingest).
                                # Record the handoff and stop reading the wire. Yield one
                                # empty chunk so the streaming aggregation has a generation
                                # (avoids langchain's "No generations found in stream").
                                signals.status = "background"
                                signals.background = {
                                    "generationId": chunk_data.get("generationId"),
                                    "pollingId": chunk_data.get("pollingId"),
                                    "messageUrl": chunk_data.get("messageUrl") or "/message",
                                }
                                logger.info(
                                    "Remote requested background handoff | gen=%s url=%s",
                                    chunk_data.get("generationId"),
                                    chunk_data.get("messageUrl") or "/message",
                                )
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content="",
                                        id=str(chunk_data.get("id") or "") or None,
                                    ),
                                )
                                return

                            content_blocks = chunk_data.get("content")
                            if not isinstance(content_blocks, list):
                                content_blocks = []

                            response_text_parts: list[str] = []
                            for _blk in content_blocks:
                                if isinstance(_blk, dict) and str(_blk.get("type") or "").replace(" ", "").lower() == "response.text.delta":
                                    _t = _blk.get("text")
                                    if isinstance(_t, str) and _t:
                                        response_text_parts.append(_t)
                            text_delta = "".join(response_text_parts)

                            passthrough_msg = AIMessageChunk(
                                content=text_delta,
                                id=str(chunk_data.get("id") or "") or None,
                                additional_kwargs={
                                    ENVELOPE_KEY: {k: chunk_data.get(k) for k in ENVELOPE_FIELDS},
                                    _RAW_REMOTE_CONTENT: content_blocks,
                                },
                                response_metadata={
                                    "passthrough_sse": True,
                                    "raw_sse": line + "\n\n",
                                },
                            )
                            gen_chunk = ChatGenerationChunk(message=passthrough_msg)
                            self._capture_chunk(gen_chunk)
                            yield gen_chunk

                            if run_manager and passthrough_msg.content:
                                await run_manager.on_llm_new_token(passthrough_msg.content)
        except Exception as exc:
            logger.warning("Remote LLM stream error | %s", exc)
            signals.status = "failed"
            signals.error = {"message": str(exc), "retryable": True}

    # ── Streaming chunk capture and merge (``agenerate`` via ``_astream``) ──

    def _capture_chunk(self, chunk: ChatGenerationChunk) -> None:
        """Collect streamed chunks for post-merge (no-op outside ``_agenerate_with_cache``)."""
        chunks = _CHUNK_LIST.get()
        if chunks is None:
            return
        chunks.append(chunk)

    async def _agenerate_with_cache(self, *args, **kwargs) -> ChatResult:
        """Capture SSE chunks during ``super()``; merge and apply envelope when any were recorded."""
        signals = STREAM_SIGNALS.get()
        signals.reset_status()
        chunks: list[ChatGenerationChunk] = []
        token = _CHUNK_LIST.set(chunks)
        try:
            result = await super()._agenerate_with_cache(*args, **kwargs)
        finally:
            _CHUNK_LIST.reset(token)

        if chunks:
            result = self._rebuild_generation(chunks)
            self._apply_envelope(result, chunks)

        if signals.status:
            strip_tool_calls_from_result(result, signals.status, signals.error)

        return result

    @staticmethod
    def _stamp_block_ids(chunks: list[ChatGenerationChunk]) -> None:
        """Ensure every ``_RAW_REMOTE_CONTENT`` block has an ``id`` so that
        ``merge_lists`` never merges blocks of different types at the same index.

        Blocks that already carry an ``id`` get it prefixed with their type
        (``"{type}_{id}"``); blocks without an ``id`` receive their ``type``
        as the id.  This keeps same-type blocks (e.g. consecutive
        ``response.text.delta``) mergeable while preventing cross-type
        collisions (e.g. ``response.text.delta`` + ``citation.widget``).
        """
        for chunk in chunks:
            raw = (chunk.message.additional_kwargs or {}).get(_RAW_REMOTE_CONTENT)
            if not isinstance(raw, list):
                continue
            for blk in raw:
                if not isinstance(blk, dict):
                    continue
                blk_type = blk.get("type", "")
                blk_id = blk.get("id")
                if blk_id:
                    blk["id"] = f"{blk_type}_{blk_id}"
                else:
                    blk["id"] = blk_type

    @staticmethod
    def _restore_block_ids(blocks: list[dict[str, Any]]) -> None:
        """Undo the synthetic ``id`` values added by ``_stamp_block_ids``
        and strip the ``.delta`` suffix from block types.

        After merging, each block's ``id`` is either the bare type (was id-less)
        or ``"{type}_{original_id}"`` (had a real id).  This restores the
        original: deletes synthetic ids, strips the type prefix from real ones,
        and normalises ``response.text.delta`` → ``response.text``, etc.
        """
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            blk_type = blk.get("type", "")
            if isinstance(blk_type, str) and blk_type.endswith(".delta"):
                blk["type"] = blk_type[:-len(".delta")]
            blk_id = blk.get("id", "")
            blk_type = blk.get("type", "")
            if blk_id == blk_type or blk_id == f"{blk_type}.delta":
                del blk["id"]
            elif blk_id.startswith(f"{blk_type}.delta_"):
                blk["id"] = blk_id[len(f"{blk_type}.delta_"):]
            elif blk_id.startswith(f"{blk_type}_"):
                blk["id"] = blk_id[len(f"{blk_type}_"):]

    @staticmethod
    def _rebuild_generation(
        chunks: list[ChatGenerationChunk],
    ) -> ChatResult:
        """Merge captured chunks into a single generation; normalize raw content blocks to match sync contract."""
        RemoteChatModel._stamp_block_ids(chunks)
        merged = chunks[0]
        for c in chunks[1:]:
            merged = merged + c

        raw_payload: dict[str, Any] | None = None
        for c in reversed(chunks):
            md = c.message.response_metadata or {}
            rp = md.get(RAW_RESPONSE_KEY)
            if isinstance(rp, dict):
                raw_payload = rp
                break

        msg = message_chunk_to_message(merged.message)

        raw_blocks = (msg.additional_kwargs or {}).get(_RAW_REMOTE_CONTENT)
        if isinstance(raw_blocks, list) and raw_blocks:
            RemoteChatModel._restore_block_ids(raw_blocks)
            msg.content = get_plain_text_from_content(raw_blocks, for_remote_invoke=True)

        if raw_payload is not None:
            meta = dict(msg.response_metadata or {})
            meta[RAW_RESPONSE_KEY] = raw_payload
            msg.response_metadata = meta

        gen = ChatGeneration(
            message=msg,
            generation_info=merged.generation_info,
        )

        gen.message.response_metadata = {
            **(gen.generation_info or {}),
            **(gen.message.response_metadata or {}),
        }

        gen.message.response_metadata.pop("raw_sse", None)
        gen.message.response_metadata.pop("passthrough_sse", None)

        return ChatResult(generations=[gen])

    @staticmethod
    def _apply_envelope(
        result: ChatResult,
        chunks: list[ChatGenerationChunk],
    ) -> None:
        """Copy latest envelope fields from captured chunks onto the merged generation."""
        if not result.generations:
            return

        last: dict[str, Any] = {k: None for k in ENVELOPE_FIELDS}
        is_remote = False
        for c in chunks:
            chunk_envelope = (c.message.additional_kwargs or {}).get(ENVELOPE_KEY)
            if not chunk_envelope:
                continue
            if chunk_envelope.get("is_remote_response"):
                is_remote = True
            for k in ENVELOPE_FIELDS:
                if chunk_envelope.get(k) is not None:
                    last[k] = chunk_envelope[k]

        if is_remote:
            last["is_remote_response"] = True
        result.generations[0].message.additional_kwargs[ENVELOPE_KEY] = last
