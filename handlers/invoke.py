"""``POST /invoke``: sync JSON or typed SSE when ``stream`` is true."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time as _time
import uuid
from typing import Any, Dict

import tornado.web

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent_builder.handlers.core.agent_execution import (
    AgentExecutionHandler,
    ExecutionContext,
    _RESPONSE_ALREADY_WRITTEN,
)
from agent_builder.llm_client.utils.remote_adapter import extract_usage_from_state, state_to_response
from agent_builder.handlers.core.requests import AgentInvokeRequest
from agent_builder.utils.constants import RAW_RESPONSE_KEY
from agent_builder.llm_client.utils.remote_chat_helpers import is_remote_stream_failed
from agent_builder.llm_client.utils.sse_adapter import (
    extract_response_blocks,
    extract_response_text_delta,
    format_sse_event,
)
from agent_builder.llm_client.utils.typed_stream_sequencer import TypedStreamSequencer
from agent_builder.storage.utils.state_serializer import get_json_serializable_graph_state
from agent_builder.handlers.core.interrupt_poller import InterruptPoller
from agent_builder.handlers.utils.background_stream import (
    fold_assistant_text_into_state,
    response_text_from_delta_event,
)
from agent_builder.llm_client.utils.interrupt import STREAM_SIGNALS, StreamSignals
from agent_builder.utils.handler_io_telemetry import flush_handler_io_doc
from agent_builder.utils.telemetry import build_invoke_telemetry_config

logger = logging.getLogger(__name__)

#: Wall-clock seconds a streaming invoke may run live before auto-switching to
#: background mode (emit ``mode.background`` and continue via ``/message`` polling).
BACKGROUND_SWITCH_SECONDS: int = int(os.getenv("BACKGROUND_SWITCH_SECONDS", 3600))



def _process_message_chunk(
    msg: Any, typed: TypedStreamSequencer, *, allow_passthrough: bool = True,
) -> list[tuple[str, Any]]:
    """Map one graph message chunk to typed ``content.delta`` events.

    Returns ``(kind, payload)`` pairs where *kind* is ``"passthrough_raw"`` (only when
    *allow_passthrough* is ``True`` and the chunk carries remote SSE) or ``"typed_event"``.

    When *allow_passthrough* is ``False`` (background buffer path), remote passthrough
    chunks are re-sequenced locally so their ``sequence`` doesn't break ``/message`` cursoring.
    """
    if isinstance(msg, AIMessageChunk) and getattr(msg, "chunk_position", None) == "last":
        meta = getattr(msg, "response_metadata", None) or {}
        if isinstance(meta.get(RAW_RESPONSE_KEY), dict) or not msg.content:
            return []

    if allow_passthrough and msg.response_metadata.get("passthrough_sse"):
        raw = msg.response_metadata.get("raw_sse")
        if raw:
            typed.consume_passthrough_sse(raw)
            return [("passthrough_raw", raw)]
        return []

    out: list[tuple[str, Any]] = []
    delta = extract_response_text_delta(msg)
    if delta is not None:
        text, idx = delta
        out.append(("typed_event", typed.content_delta_text(text, content_index=idx)))
    for block in extract_response_blocks(msg):
        out.append(("typed_event", typed.content_delta_block(block)))
    return out


_usage_dict_from_final_state = extract_usage_from_state


class InvokeAgentHandler(AgentExecutionHandler):
    """Agent execution: blocking JSON or streaming SSE controlled by ``request.stream``."""

    async def _build_terminal_frame(
        self,
        final_state: dict,
        request: AgentInvokeRequest,
        ctx: ExecutionContext,
        typed: TypedStreamSequencer,
        *,
        interrupted: bool,
        stream_failed: bool,
        stream_error: dict,
        fold_text: str | None = None,
    ) -> dict:
        """Build the terminal SSE frame after streaming completes.

        Handles interrupted / failed / completed branching and calls the
        appropriate finalization method. Pass *fold_text* (from background drain)
        to fold assembled text into state before finalizing.
        """
        if interrupted:
            await self._finalize_interrupted_execution(final_state, request, ctx)
            usage = _usage_dict_from_final_state(final_state)
            return typed.stream_interrupted(usage)

        if stream_failed:
            await self._finalize_interrupted_execution(final_state, request, ctx)
            return typed.stream_failed(
                stream_error.get("message", "LLM stream failed"),
                retryable=stream_error.get("retryable", True),
            )

        if fold_text is not None:
            fold_assistant_text_into_state(final_state, fold_text)
            self._tag_response_messages_with_id(final_state, ctx.response_id)

        proceeded = await self._finalize_execution(final_state, request, ctx)
        if proceeded:
            usage = _usage_dict_from_final_state(final_state)
            return typed.stream_completed(usage)
        return typed.stream_failed("Conversation stopped", retryable=False)

    async def _handle_stream_exception(
        self,
        poller: InterruptPoller,
        stream_opened: bool,
        final_state: dict,
        request: AgentInvokeRequest,
        ctx: ExecutionContext,
        typed: TypedStreamSequencer,
        *,
        error_message: str,
        retryable: bool,
        raise_if_closed: Exception | None = None,
    ) -> object:
        """Shared exception handling for ``_stream_to_client``.

        If the interrupt fired, writes ``stream.interrupted``. Otherwise writes
        ``stream.failed``. If the stream was never opened, raises *raise_if_closed*
        (or returns the sentinel).
        """
        if poller.is_interrupted and stream_opened:
            try:
                await self._finalize_interrupted_execution(final_state, request, ctx)
                usage = _usage_dict_from_final_state(final_state)
                self.write(format_sse_event(typed.stream_interrupted(usage)))
                await self.flush()
            except Exception:
                logger.exception("Failed to write stream.interrupted")
            return _RESPONSE_ALREADY_WRITTEN
        if stream_opened:
            try:
                self.write(format_sse_event(typed.stream_failed(error_message, retryable=retryable)))
                await self.flush()
            except Exception:
                logger.exception("Failed to write stream.failed")
            return _RESPONSE_ALREADY_WRITTEN
        if raise_if_closed is not None:
            raise raise_if_closed
        return _RESPONSE_ALREADY_WRITTEN

    def on_connection_close(self):
        if getattr(self, "_invoke_was_streaming", False):
            logger.warning("Client disconnected during streaming")
        super().on_connection_close()

    def validate_payload(self, payload: Dict[str, Any]) -> AgentInvokeRequest:
        return AgentInvokeRequest(**payload)

    def write_response(self, response: Any) -> None:
        if response is _RESPONSE_ALREADY_WRITTEN:
            return
        super().write_response(response)

    async def process(self, request: AgentInvokeRequest) -> Any:
        logger.debug("Invoke request payload | %s", request.dict())
        self._invoke_was_streaming = bool(request.stream)
        agent, ctx = await self._prepare_execution(request)
        telemetry_config = build_invoke_telemetry_config(self, request, ctx)

        if request.stream:
            return await self._process_streaming(agent, ctx, request, telemetry_config)

        poller = InterruptPoller(self.redis_client, request.sessionId, request.id)
        poller.start()
        try:
            new_state = await self._invoke_with_interrupt(
                agent, ctx, request, poller, telemetry_config=telemetry_config,
            )
            self._tag_response_messages_with_id(new_state, ctx.response_id)
            if poller.is_interrupted:
                await self._finalize_interrupted_execution(ctx.state, request, ctx)
                return self._build_stopped_response(request)

            proceeded = await self._finalize_execution(new_state, request, ctx)
            if not proceeded:
                return self._build_stopped_response(request)
            return state_to_response(new_state)
        finally:
            await poller.stop()

    # ── streaming (SSE) ─────────────────────────────────────────────────

    async def _process_streaming(
        self, agent: Any, ctx: ExecutionContext, request: AgentInvokeRequest,
        telemetry_config: Any = None,
    ) -> Any:
        if not callable(getattr(agent, "astream", None)):
            raise tornado.web.HTTPError(
                500, f"Agent '{ctx.agent_id}' does not support streaming",
            )

        if (request.delivery or {}).get("mode") == "background":
            return await self._start_background_immediately(agent, ctx, request, telemetry_config)

        poller = InterruptPoller(self.redis_client, request.sessionId, request.id)
        poller.start()
        try:
            return await self._stream_to_client(agent, ctx, request, poller, telemetry_config)
        finally:
            await poller.stop()

    async def _stream_to_client(
        self, agent: Any, ctx: ExecutionContext, request: AgentInvokeRequest,
        poller: InterruptPoller,
        telemetry_config: Any = None,
    ) -> object:
        """Consume ``agent.astream()`` and flush each chunk as an SSE event."""
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")

        final_state = ctx.state
        client_stream_mode = set(request.streamMode or ["messages"])
        internal_stream_mode = list(client_stream_mode | {"values"})
        chunk_count = 0
        t_start = _time.perf_counter()
        t_first_chunk = None
        typed = TypedStreamSequencer(request.id, response_id=ctx.response_id)
        stream_opened = False
        interrupted = False
        stream_failed = False
        stream_error: dict = {}
        switched = False

        signals = StreamSignals(interrupt_event=poller.interrupted)
        interrupt_token = STREAM_SIGNALS.set(signals)
        agen = agent.astream(ctx.state, stream_mode=internal_stream_mode, config=telemetry_config)
        try:
            self.write(format_sse_event(typed.stream_start()))
            await self.flush()
            stream_opened = True

            async for chunk in agen:
                try:
                    mode = chunk.get("stream_mode")
                    chunk_count += 1

                    if t_first_chunk is None:
                        t_first_chunk = _time.perf_counter()
                        logger.debug("TTFT=%.0fms | agent_id=%s", (t_first_chunk - t_start) * 1000, ctx.agent_id)

                    if chunk_count % 20 == 0:
                        logger.debug("Streaming progress | chunks=%d agent_id=%s", chunk_count, ctx.agent_id)

                    if mode == "values":
                        final_state = chunk["value"]

                        if signals.check_interrupted():
                            interrupted = True
                            break

                        last_ai = self._find_last_ai_message(final_state)
                        if last_ai is not None:
                            meta = last_ai.response_metadata or {}
                            if meta.get("stream_status") == "failed":
                                stream_failed = True
                                stream_error = meta.get("stream_error", {})
                                break

                        if "values" in client_stream_mode:
                            serialized = json.dumps(get_json_serializable_graph_state(final_state))
                            self.write(f"state: {serialized}\n")
                            await self.flush()

                        if not switched and (_time.perf_counter() - t_start) > BACKGROUND_SWITCH_SECONDS:
                            switched = True
                            gen_id = uuid.uuid4().hex
                            logger.info(
                                "Auto-switching to background mode | agent_id=%s elapsed>%ss",
                                ctx.agent_id, BACKGROUND_SWITCH_SECONDS,
                            )
                            self.write(format_sse_event(typed.mode_background(gen_id, polling_id=request.id)))
                            await self.flush()
                            self._enter_background(ctx)
                            await self.redis_client.set_bg_meta(
                                request.sessionId, request.id, gen_id, "running",
                            )
                            await self._persist_input_and_set_running(
                                request, final_state, ctx.agent_doc, ctx.merged_mocks, ctx.mcp_descriptors_by_task,
                            )
                            asyncio.ensure_future(
                                self._drain_background(agen, ctx, request, typed, gen_id, final_state),
                            )
                            return _RESPONSE_ALREADY_WRITTEN

                    elif mode == "messages":
                        msg = chunk["message"]
                        deliverables = _process_message_chunk(msg, typed)
                        if not deliverables:
                            continue
                        for kind, payload in deliverables:
                            if kind == "passthrough_raw":
                                self.write(payload)
                            else:
                                self.write(format_sse_event(payload))
                        await self.flush()

                except Exception:
                    logger.exception("Error writing SSE chunk")
                    continue

            self._tag_response_messages_with_id(final_state, ctx.response_id)

            # Remote handed the run off to /message (it emitted mode.background and
            # will push the rest via /message/ingest). Forward mode.background using
            # the remote's generationId, mark background, and stop — do NOT finalize
            # with stream.completed, since the run continues out-of-band.
            if signals.background:
                gen_id = signals.background.get("generationId") or uuid.uuid4().hex
                polling_id = signals.background.get("pollingId") or request.id
                logger.info(
                    "Remote handoff to background | agent_id=%s gen=%s",
                    ctx.agent_id, gen_id,
                )
                self.write(format_sse_event(typed.mode_background(gen_id, polling_id=polling_id)))
                await self.flush()
                self._enter_background(ctx)
                await self.redis_client.set_bg_meta(
                    request.sessionId, request.id, gen_id, "running",
                )
                await self._persist_input_and_set_running(
                    request, final_state, ctx.agent_doc, ctx.merged_mocks, ctx.mcp_descriptors_by_task,
                )
                return _RESPONSE_ALREADY_WRITTEN

            terminal_frame = await self._build_terminal_frame(
                final_state, request, ctx, typed,
                interrupted=interrupted, stream_failed=stream_failed, stream_error=stream_error,
            )
            self.write(format_sse_event(terminal_frame))
            await self.flush()

            outcome = "interrupted" if interrupted else ("failed" if stream_failed else "completed")
            logger.debug(
                "Stream %s | chunks=%d total=%.0fms agent_id=%s",
                outcome, chunk_count, (_time.perf_counter() - t_start) * 1000, ctx.agent_id,
            )

        except asyncio.TimeoutError:
            logger.error("Agent streaming timed out | agent_id=%s", ctx.agent_id)
            return await self._handle_stream_exception(
                poller, stream_opened, final_state, request, ctx, typed,
                error_message="Agent streaming timed out", retryable=True,
                raise_if_closed=tornado.web.HTTPError(504, "Agent streaming timed out"),
            )
        except RuntimeError as exc:
            if not is_remote_stream_failed(exc):
                raise
            logger.warning(
                "Remote task stream.failed | agent_id=%s retryable=%s",
                ctx.agent_id, exc.retryable,
            )
            return await self._handle_stream_exception(
                poller, stream_opened, final_state, request, ctx, typed,
                error_message=str(exc), retryable=exc.retryable,
            )
        except tornado.web.HTTPError:
            raise
        except Exception as exc:
            logger.exception("Agent streaming failed | agent_id=%s", ctx.agent_id)
            return await self._handle_stream_exception(
                poller, stream_opened, final_state, request, ctx, typed,
                error_message=str(exc), retryable=False,
                raise_if_closed=tornado.web.HTTPError(500, f"Agent streaming failed: {exc}"),
            )
        finally:
            STREAM_SIGNALS.reset(interrupt_token)

        return _RESPONSE_ALREADY_WRITTEN

    @staticmethod
    def _find_last_ai_message(state: dict):
        """The current turn's assistant message, or ``None``.

        Scans history backwards but stops at the first ``HumanMessage`` (the just-appended
        user input) so a previous turn's reply is never returned. This matters for the
        ``stream_status == "failed"`` check: a failed AIMessage persisted by an earlier turn
        must not be mistaken for a failure of the current turn (which would emit a spurious
        ``stream.failed`` and permanently wedge the session).
        """
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage):
                return m
            if isinstance(m, HumanMessage):
                return None
        return None

    # ── background mode (long-running / poll-via-/message) ──────────────

    async def _start_background_immediately(
        self, agent: Any, ctx: ExecutionContext, request: AgentInvokeRequest,
        telemetry_config: Any = None,
    ) -> object:
        """``delivery.mode == "background"``: open the wire only long enough to emit
        ``stream.start`` + ``mode.background``, then detach a drain task.

        The run continues entirely in the background and is served via ``/message``.
        """
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")

        typed = TypedStreamSequencer(request.id, response_id=ctx.response_id)
        gen_id = uuid.uuid4().hex

        self.write(format_sse_event(typed.stream_start()))
        self.write(format_sse_event(typed.mode_background(gen_id, polling_id=request.id)))
        await self.flush()

        self._enter_background(ctx)
        await self.redis_client.set_bg_meta(request.sessionId, request.id, gen_id, "running")
        # Eagerly persist the user message so a pod restart can resume from graph_state.
        await self._persist_input_and_set_running(
            request, ctx.state, ctx.agent_doc, ctx.merged_mocks, ctx.mcp_descriptors_by_task,
        )

        client_stream_mode = set(request.streamMode or ["messages"])
        internal_stream_mode = list(client_stream_mode | {"values"})
        agen = agent.astream(ctx.state, stream_mode=internal_stream_mode, config=telemetry_config)
        asyncio.ensure_future(
            self._drain_background(agen, ctx, request, typed, gen_id, ctx.state),
        )
        return _RESPONSE_ALREADY_WRITTEN

    async def _drain_background(
        self,
        agen: Any,
        ctx: ExecutionContext,
        request: AgentInvokeRequest,
        typed: TypedStreamSequencer,
        gen_id: str,
        final_state: dict,
    ) -> None:
        """Drain the remainder of an ``astream()`` iterator into the Redis chunk buffer.

        Runs detached after the live wire switched to ``mode.background``. Persists each
        text delta as a locally-sequenced ``content.delta`` frame (continuing the same
        ``typed`` sequence space), then finalizes the session envelope (canonical history)
        and writes the single run-level terminal frame + meta status.
        """
        sid, rid = request.sessionId, request.id
        poller = InterruptPoller(self.redis_client, sid, rid)
        poller.start()
        signals = StreamSignals(interrupt_event=poller.interrupted)
        token = STREAM_SIGNALS.set(signals)
        interrupted = False
        stream_failed = False
        stream_error: dict = {}
        assembled_parts: list[str] = []
        try:
            async for chunk in agen:
                mode = chunk.get("stream_mode")
                if mode == "values":
                    final_state = chunk["value"]
                    if signals.check_interrupted():
                        interrupted = True
                        break
                    last_ai = self._find_last_ai_message(final_state)
                    if last_ai is not None:
                        meta = last_ai.response_metadata or {}
                        if meta.get("stream_status") == "failed":
                            stream_failed = True
                            stream_error = meta.get("stream_error", {})
                            break
                elif mode == "messages":
                    pairs = _process_message_chunk(chunk["message"], typed, allow_passthrough=False)
                    if pairs:
                        events = [payload for _, payload in pairs]
                        for event in events:
                            assembled_parts.append(response_text_from_delta_event(event))
                        await self.redis_client.append_bg_chunks(sid, rid, gen_id, events)

            self._tag_response_messages_with_id(final_state, ctx.response_id)

            terminal = await self._build_terminal_frame(
                final_state, request, ctx, typed,
                interrupted=interrupted, stream_failed=stream_failed, stream_error=stream_error,
                fold_text="".join(assembled_parts),
            )
            await self.redis_client.append_bg_chunks(sid, rid, gen_id, [terminal])

        except RuntimeError as exc:
            if is_remote_stream_failed(exc):
                logger.warning(
                    "Background remote stream.failed | agent_id=%s retryable=%s",
                    ctx.agent_id, exc.retryable,
                )
                terminal = typed.stream_failed(str(exc), retryable=exc.retryable)
            else:
                logger.exception("Background drain failed | agent_id=%s", ctx.agent_id)
                terminal = typed.stream_failed(str(exc), retryable=False)
            await self._safe_append_terminal(sid, rid, gen_id, terminal)
        except Exception as exc:
            logger.exception("Background drain failed | agent_id=%s", ctx.agent_id)
            await self._safe_append_terminal(
                sid, rid, gen_id, typed.stream_failed(str(exc), retryable=False),
            )
        finally:
            STREAM_SIGNALS.reset(token)
            await poller.stop()
            try:
                await agen.aclose()
            except Exception:
                pass

    async def _safe_append_terminal(
        self, sid: str, rid: str, gen_id: str, terminal: dict,
    ) -> None:
        try:
            await self.redis_client.append_bg_chunks(sid, rid, gen_id, [terminal])
        except Exception:
            logger.exception("Failed to persist background terminal frame | rid=%s", rid)

    # ── interrupt-aware invoke ──────────────────────────────────────────

    @staticmethod
    def _suppress_task_exception(task: asyncio.Task) -> None:
        if not task.cancelled():
            try:
                task.exception()
            except Exception:
                pass

    async def _invoke_with_interrupt(
        self,
        agent: Any,
        ctx: ExecutionContext,
        request: AgentInvokeRequest,
        poller: InterruptPoller,
        telemetry_config: Any = None,
    ) -> dict:
        """Race ``agent.ainvoke()`` against the interrupt signal.

        If the interrupt fires first the invoke task is cancelled and
        ``ctx.state`` (pre-execution, with user message already appended)
        is returned.  Caller checks ``poller.is_interrupted`` to distinguish.
        """
        signals = StreamSignals(interrupt_event=poller.interrupted)
        interrupt_token = STREAM_SIGNALS.set(signals)
        try:
            invoke_task = asyncio.create_task(
                self._invoke_agent(agent, ctx.state, ctx.agent_id, config=telemetry_config),
            )
            interrupt_wait = asyncio.create_task(poller.interrupted.wait())

            done, pending = await asyncio.wait(
                {invoke_task, interrupt_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                task.add_done_callback(self._suppress_task_exception)

            if invoke_task in done:
                return invoke_task.result()

            return ctx.state
        finally:
            STREAM_SIGNALS.reset(interrupt_token)
