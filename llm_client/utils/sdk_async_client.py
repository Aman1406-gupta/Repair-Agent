"""SDK client a remote runtime uses to talk back to Agent Builder.

A task registered behind an ``http_config.url`` runs out-of-process. A remote runtime
can use :class:`BackgroundStreamer` to stream events in two phases through a single,
unified API:

**Phase 1 — inline SSE** (before ``switch_to_background()``):
  :meth:`send_event` / :meth:`complete` / … build typed frames and return them as SSE
  strings. The caller writes these to its HTTP response.

**Phase 2 — background** (after ``switch_to_background()``):
  The same methods automatically POST frames to ``/message/ingest`` instead. The
  caller's HTTP response is already closed; the client polls ``/message``.

Sequence numbers are monotonic across both phases — the
:class:`TypedStreamSequencer` is shared.

With a ``writer`` callback (recommended), the caller never touches SSE framing::

    from agent_builder.llm_client.utils.sdk_async_client import agent_builder_client

    run = agent_builder_client.get_background_streamer(
        session_id=sid, request_id=rid, writer=resp.write,
    )

    await run.start()                        # writes stream.start to resp
    await run.send_event("inline chunk 1")   # writes content.delta to resp
    await run.send_event("inline chunk 2")   # writes content.delta to resp
    await run.switch_to_background()         # writes mode.background to resp
    await resp.write_eof()

    await run.send_event("background chunk") # POSTs to /message/ingest
    await run.complete()                     # POSTs stream.completed

Without a ``writer``, inline methods return SSE strings (caller writes them manually).
Legacy ``handshake_sse()`` still works for pure-background usage.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import random
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import httpx

from agent_builder.llm_client.utils.response_formats import text_block
from agent_builder.llm_client.utils.typed_stream_sequencer import TypedStreamSequencer


class AgentBuilderClientError(RuntimeError):
    """Raised when an outbound call to Agent Builder fails after exhausting retries."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _sse(event: Dict[str, Any]) -> str:
    """Format one typed event as an SSE ``data:`` frame."""
    return "data: " + json.dumps(event) + "\n\n"


class BackgroundStreamer:
    """Async client for one run — inline SSE *then* background, through one API.

    With a *writer* callback, all methods write/post automatically — the caller
    never touches SSE framing. Without a writer, inline methods return SSE strings.
    """

    def __init__(
        self,
        agent_builder_url: Optional[str] = None,
        *,
        session_id: str,
        request_id: str,
        agent_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        writer: Optional[Callable[[bytes], Awaitable[None]]] = None,
        client: Optional[httpx.AsyncClient] = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 30.0,
    ) -> None:
        base_url = agent_builder_url or os.getenv("AGENT_BUILDER_URL")
        if not base_url:
            raise ValueError(
                "agent_builder_url is required (pass it explicitly or set AGENT_BUILDER_URL)"
            )
        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._request_id = request_id
        self._agent_id = agent_id
        self._generation_id = generation_id or uuid.uuid4().hex
        self._seq = TypedStreamSequencer(request_id)
        self._content_index = 0
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._writer = writer
        self._is_background = False

    @property
    def generation_id(self) -> str:
        return self._generation_id

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def session_id(self) -> str:
        return self._session_id

    # -- content index -----------------------------------------------------

    def get_index(self) -> int:
        """Current content block index for this run (starts at 0).

        Pass it to a block builder (``text_block("...", index=run.get_index())``) when
        blocks spread across multiple ``content.delta`` events must share one position
        — e.g. many text deltas appending to the same block.
        """
        return self._content_index

    def increase_index(self) -> int:
        """Advance the content block index by one and return the new value.

        Call it when the next block should render as a new position in the message
        (e.g. after finishing a text block, before starting an image).
        """
        self._content_index += 1
        return self._content_index

    @property
    def is_background(self) -> bool:
        return self._is_background

    # -- lifecycle: start / switch / handshake -----------------------------

    async def start(self) -> Optional[str]:
        """Emit ``stream.start``. With a writer, writes directly; otherwise returns SSE string."""
        return await self._emit(self._seq.stream_start())

    async def switch_to_background(self) -> Optional[str]:
        """Emit ``mode.background`` and flip to background mode.

        After this call, :meth:`send_event` / :meth:`complete` / etc. POST to
        ``/message/ingest`` instead of writing inline SSE.
        """
        if self._is_background:
            return None
        event = self._seq.mode_background(self._generation_id, polling_id=self._request_id)
        sse = _sse(event)
        if self._writer is not None:
            await self._writer(sse.encode())
        self._is_background = True
        return None if self._writer else sse

    def handshake_sse(self) -> str:
        """Legacy sync shortcut: ``stream.start`` + ``mode.background`` in one string.

        Flips to background mode. Does NOT use the writer — returns a raw SSE string
        for callers that manage their own response wire.
        """
        start = _sse(self._seq.stream_start())
        bg = _sse(self._seq.mode_background(self._generation_id, polling_id=self._request_id))
        self._is_background = True
        return start + bg

    # -- core emit (inline writer / return SSE / background POST) ----------

    async def _emit(self, event: Dict[str, Any]) -> Optional[Union[str, Dict[str, Any]]]:
        """Route one event through the current mode.

        * **Inline + writer**: writes SSE bytes via the callback, returns ``None``.
        * **Inline, no writer**: returns the SSE string.
        * **Background**: POSTs to ``/message/ingest``, returns the response dict.
        """
        if self._is_background:
            return await self.send_events([event])
        sse = _sse(event)
        if self._writer is not None:
            await self._writer(sse.encode())
            return None
        return sse

    # -- streaming events (unified interface) ------------------------------

    def _build_content_event(
        self,
        content: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        *,
        event_type: str = "content.delta",
    ) -> Dict[str, Any]:
        if event_type != "content.delta":
            if not isinstance(content, dict):
                raise ValueError(f"event_type={event_type!r} requires content to be a dict")
            return self._seq.raw_additional({**content, "type": event_type})
        if isinstance(content, str):
            blocks = [text_block(content)]
        elif isinstance(content, dict):
            blocks = [content]
        elif isinstance(content, list):
            blocks = content
        else:
            raise ValueError("content must be a str, a block dict, or a list of block dicts")
        blocks = [{"index": self.get_index(), **block} for block in blocks]
        return self._seq.content_delta(blocks)

    async def send_event(
        self,
        content: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        *,
        event_type: str = "content.delta",
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Build one typed event and emit it.

        With a **writer**: writes inline or POSTs background — nothing to do with the
        return value (``None`` inline, response dict background).

        Without a writer: returns SSE string (inline) or response dict (background).
        """
        return await self._emit(self._build_content_event(content, event_type=event_type))

    async def complete(self, usage: Optional[Dict[str, Any]] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Terminal ``stream.completed`` frame (triggers server-side finalize)."""
        return await self._emit(self._seq.stream_completed(usage))

    async def fail(self, message: str, *, retryable: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Terminal ``stream.failed`` frame."""
        return await self._emit(self._seq.stream_failed(message, retryable=retryable))

    async def interrupt(self, usage: Optional[Dict[str, Any]] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Terminal ``stream.interrupted`` frame."""
        return await self._emit(self._seq.stream_interrupted(usage))

    async def send_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Low-level batch push of 1..n pre-built events to ``POST /message/ingest``."""
        body: Dict[str, Any] = {
            "id": self._request_id,
            "sessionId": self._session_id,
            "generationId": self._generation_id,
            "events": events,
        }
        if self._agent_id:
            body["agentId"] = self._agent_id
        return await self._post("/message/ingest", body)

    # -- transport with retry ---------------------------------------------

    async def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON with retry+backoff on transient failures (5xx/429/transport/timeout)."""
        url = f"{self._base_url}{path}"
        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(url, json=body)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
            else:
                if resp.status_code < 400:
                    return resp.json()
                last_status = resp.status_code
                # 4xx (other than 429) are contract errors — fail fast.
                if resp.status_code != 429 and resp.status_code < 500:
                    raise AgentBuilderClientError(
                        f"POST {path} failed with {resp.status_code}: {resp.text}",
                        status_code=resp.status_code,
                    )

            if attempt < self._max_retries:
                delay = self._backoff_base * (2 ** attempt) + random.uniform(0, self._backoff_base)
                await asyncio.sleep(delay)

        if last_exc is not None:
            raise AgentBuilderClientError(
                f"POST {path} failed after {self._max_retries + 1} attempts: {last_exc}",
            ) from last_exc
        raise AgentBuilderClientError(
            f"POST {path} failed after {self._max_retries + 1} attempts (last status {last_status})",
            status_code=last_status,
        )

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "BackgroundStreamer":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


class AgentBuilderClient:
    """Process-wide singleton owning one shared ``httpx`` pool for the pod's lifetime.

    Import the ready-made :data:`agent_builder_client` instance once and call :meth:`run` per
    request to get a cheap :class:`BackgroundStreamer` that borrows the shared pool. The
    per-run client owns no pool, so it needs no closing — just let it be garbage-collected.
    The shared pool is created lazily on first use and closed automatically at interpreter
    exit (call :meth:`aclose` yourself if your framework has an async shutdown hook).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 30.0,
    ) -> None:
        # Resolved lazily in get_background_streamer so an AGENT_BUILDER_URL set after
        # import (the singleton is created at import time) is still picked up.
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._atexit_registered = False

    def configure(
        self,
        *,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> "AgentBuilderClient":
        """Override defaults once, at startup (before the shared pool is created)."""
        if self._client is not None:
            raise RuntimeError("configure() must be called before the first run()/request")
        if base_url is not None:
            self._base_url = base_url
        if max_retries is not None:
            self._max_retries = max_retries
        if backoff_base is not None:
            self._backoff_base = backoff_base
        if timeout is not None:
            self._timeout = timeout
        return self

    def _shared_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            if not self._atexit_registered:
                atexit.register(self._close_atexit)
                self._atexit_registered = True
        return self._client

    def get_background_streamer(
        self,
        *,
        session_id: str,
        request_id: str,
        agent_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        writer: Optional[Callable[[bytes], Awaitable[None]]] = None,
    ) -> BackgroundStreamer:
        """Return a per-run client bound to the shared pool. Cheap; nothing to close.

        Pass *writer* (e.g. ``resp.write``) so inline methods write directly
        instead of returning SSE strings.
        """
        return BackgroundStreamer(
            self._base_url or os.getenv("AGENT_BUILDER_URL"),
            session_id=session_id,
            request_id=request_id,
            agent_id=agent_id,
            generation_id=generation_id,
            writer=writer,
            client=self._shared_client(),
            max_retries=self._max_retries,
            backoff_base=self._backoff_base,
        )

    async def aclose(self) -> None:
        """Close the shared pool. Safe to call from a framework shutdown hook."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _close_atexit(self) -> None:
        """Best-effort pool teardown at interpreter exit (Option C)."""
        client = self._client
        if client is None:
            return
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                loop.create_task(client.aclose())
            else:
                asyncio.run(client.aclose())
        except Exception:
            pass


#: Ready-made process-wide singleton. Import once, ``agent_builder_client.run(...)`` per request.
agent_builder_client = AgentBuilderClient()
