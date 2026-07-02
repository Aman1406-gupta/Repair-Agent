from __future__ import annotations
import asyncio
import json
import logging
import uuid
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
from typing import (Any, AsyncIterator, Callable, Dict, Iterator, List,
                    Optional, Sequence, Type, Union)

from langchain_core.callbacks import (AsyncCallbackManagerForLLMRun,
                                      CallbackManagerForLLMRun)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ChatMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, ConfigDict, Field, root_validator, model_validator
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from llm_router.sdk.client import LLMClient
from agent_builder.base.configs import LLMConfig
from agent_builder.utils.constants import ENVELOPE_KEY, OPENAI_CHOICES_KEY
from openai import AsyncOpenAI

from agent_builder.llm_client.utils.message_converters import _convert_message_to_dict, _convert_dict_to_message
from agent_builder.llm_client.utils.interrupt import STREAM_SIGNALS, strip_tool_calls_from_result
from agent_builder.llm_client.voicebot import RealtimeVoiceBot

import time as _time


# --- Refactored Custom Chat Model ---

class SprinklrChatModel(BaseChatModel):
    """
    A refactored custom LangChain chat model that wraps the user's LLMClient,
    with inspiration from `ChatSprinklrAI` for robustness and tool-calling.
    """
    llm_config: LLMConfig = Field(...)
    client: Any = Field(default=None, exclude=True)
    max_retries: int = Field(default=2)
    disable_streaming: bool = Field(default=False)

    model_config = ConfigDict(
        validate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode='after')
    def validate_environment(self, values: Dict) -> Dict:
        """Initialise the LLMRouter client from the dataclass."""
        if self.client is None:
            if self.llm_config.provider=='LOCAL':
                self.client = AsyncOpenAI(
                            api_key= "EMPTY",
                            base_url=self.llm_config.llm_router_url,
                        )
            elif self.llm_config.provider == "VOICE":
                self.client = RealtimeVoiceBot()
            else:
                llm_router_url = self.llm_config.llm_router_url
                # Prepend scheme if missing, so urlparse correctly identifies the hostname
                if not llm_router_url.startswith(("http://", "https://")):
                    llm_router_url = f"http://{llm_router_url}"
                parsed_result = urlparse(llm_router_url)
                proxy = self.llm_config.kwargs.get("proxy")
                timeout = self.llm_config.timeout
                scheme, hostname, port = (
                    parsed_result.scheme,
                    parsed_result.hostname,
                    parsed_result.port,
                )
                self.client = LLMClient(
                    service_url=f"{scheme}://{hostname}",
                    service_port=port or (443 if scheme == "https" else 80),
                    proxy_url=proxy,
                    sdk_timeout=timeout,
                )
        return self

    @property
    def _llm_type(self) -> str:
        """A required property to identify this as a custom LLM."""
        return "custom_chat_model_v2"

    def _router_uses_llm_configuration_id_only(self) -> bool:
        """Router mode: send only config id + tracking/client/partner/kwargs (non-LOCAL)."""
        cfg = self.llm_config
        return bool(cfg.llm_configuration_id) and cfg.provider != "LOCAL"

    @property
    def _default_params(self) -> Dict[str, Any]:
        cfg = self.llm_config
        if self._router_uses_llm_configuration_id_only():
            return {
                "model": "LLM CONFIG MODEL",
                "llm_config_id": cfg.llm_configuration_id,
                "tracking_params": cfg.tracking_params,
                "client_identifier": cfg.client_identifier,
                "partner_id": cfg.partner_id,
                **cfg.kwargs,
            }

        if self.llm_config.provider=='LOCAL':
            return {
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                **cfg.kwargs,  # dataclass-level extras
            }

        if "gpt-5" in cfg.model or "o3" in cfg.model or "o4" in cfg.model:
            return {
                "model": cfg.model,
                "max_completion_tokens": cfg.max_tokens,
                "temperature": 1.0,
                "top_p": cfg.top_p,
                "provider": cfg.provider,
                "tracking_params": cfg.tracking_params,
                "client_identifier": cfg.client_identifier,
                "partner_id": cfg.partner_id,
                **cfg.kwargs,  # dataclass-level extras
            }
        else:
            return {
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "provider": cfg.provider,
                "tracking_params": cfg.tracking_params,
                "client_identifier": cfg.client_identifier,
                "partner_id": cfg.partner_id,
                **cfg.kwargs,  # dataclass-level extras
            }

    def _create_chat_result(self, payload: Dict[str, Any], response: Dict[str, Any]) -> ChatResult:
        """Creates a ChatResult from a non-streaming API response."""
        generations = []
        choices = response.get("choices", [])
        raw_usage = response.get("usage") or {}

        envelope = {
            "usage": self._to_usage_metrics(raw_usage, response),
            #TODO: determine how to propagate errors from here
            "error": None,
            "choices_count": len(choices),
        }

        for choice in choices:
            choice_meta = {
                "id": choice.get("id") or response.get("id") or str(uuid.uuid4()),
                "finishReason": choice.get("finish_reason"),
                "index": choice.get("index", 0),
            }

            message = _convert_dict_to_message(
                choice["message"],
                disable_streaming=self.disable_streaming,
                choice_meta=choice_meta,
                envelope=envelope,
            )
            gen_info = {
                "finish_reason": choice.get("finish_reason"),
            }
            generations.append(ChatGeneration(message=message, generation_info=gen_info))

        llm_output = {
            "token_usage": raw_usage,
            "model_name": self._model_label_for_logs(),
            "spending": response.get("spending", 0.0)
        }
        res = ChatResult(generations=generations, llm_output=llm_output)
        if len(generations) == 0:
            logger.warning("LLM returned 0 generations | %s", self._model_label_for_logs())
        return res

    def _model_label_for_logs(self) -> str:
        cfg = self.llm_config
        if cfg.llm_configuration_id:
            return f"llm_configuration_id={cfg.llm_configuration_id}"
        return cfg.model

    def _to_usage_metrics(self, raw_usage: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM-router OpenAI-format usage to the API-contract UsageMetrics shape."""
        cfg = self.llm_config
        model_id = cfg.llm_configuration_id or cfg.model
        return {
            "totalCost": response.get("spending", 0.0),
            "numCalls": "1",
            "timing": {
                "totalTime": response.get("total_time"),
            } if response.get("total_time") else None,
            "modelBreakdown": [{
                "modelId": model_id,
                "provider": cfg.provider,
                "inputTokens": raw_usage.get("prompt_tokens"),
                "outputTokens": raw_usage.get("completion_tokens"),
                "numCalls": 1,
                "timeTaken": response.get("provider_total_time"),
            }] if raw_usage else None,
        }

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Sync, non-streaming requests. Wraps the async implementation."""
        # Note: Using asyncio.run() in a sync method can have performance
        # implications in some environments.
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async, non-streaming requests with a simple retry mechanism.
        A response is considered *bad* when the dict contains the key 'error'.
        """
        if self.llm_config.provider == "VOICE":
            sys_msg_content = next(
                (m.content for m in messages if isinstance(m, SystemMessage)), None
            )

            tail: list[BaseMessage] = []
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    break
                tail.append(m)
            tail.reverse()                               # chronological order

            if not tail:
                logger.error("VOICE mode: no user/tool messages found after last AIMessage")
                for m in messages:
                    logger.debug("  msg type=%s content=%.100s", m.type, m.content)
                raise ValueError("VOICE mode needs at least one user / tool message")

            for idx, msg in enumerate(tail):
                if isinstance(msg, HumanMessage):
                    audio_bytes = (msg.additional_kwargs or {}).get("audio",None)
                    if audio_bytes:
                        await self.client._asend_audio(audio_bytes)
                    if msg.content.strip()!="":
                        await self.client._asend_text(msg.content)

                elif isinstance(msg, ToolMessage):
                    await self.client._asend_function_output(msg.tool_call_id, msg.content)
                else:
                    pass


            tools = [ {"type":"function","name":tool['function']['name'],'description':tool['function']['description'],'parameters':tool['function']['parameters']} for tool in kwargs.get("tools",[])]
            text, _transcript, audio_pcm, f_calls,usage = await self.client._areceive_response(sys_msg_content,tools)


            tool_calls_for_lc = [
                {
                    "index": 0,
                    "id": fc["call_id"],
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": fc["arguments"]
                        }
                }
                for fc in f_calls
            ]
            ai_msg = AIMessage(
                content=_transcript,
                additional_kwargs={"audio": audio_pcm,
                                "transcript": _transcript,
                                "tool_calls": tool_calls_for_lc})

            return ChatResult(
                generations=[ChatGeneration(message=ai_msg,
                                            generation_info={"finish_reason": "stop"})],
                llm_output={
                    "model_name": self._model_label_for_logs(),
                    "spending": 0.0,
                    "token_usage": usage
                },
            )
        else:
            async def _do_request(payload: Dict[str, Any]) -> Dict[str, Any]:
                return await self.client.completion(payload=payload)
            async def _do_request_local(payload: Dict[str, Any]) -> Dict[str, Any]:
                return (await self.client.chat.completions.create(**payload)).to_dict()


            sys_msg = next((m for m in messages if isinstance(m, SystemMessage)), None)
            if sys_msg:
                sys_text = sys_msg.content if isinstance(sys_msg.content, str) else str(sys_msg.content)
                logger.debug("LLM system prompt | %s\n%s", self._model_label_for_logs(), sys_text)

            payload = self._create_payload(messages, stream=False, stop=stop, **kwargs)
            logger.debug(
                "LLM call | %s provider=%s msg_count=%d",
                self._model_label_for_logs(), self.llm_config.provider, len(messages),
            )
            t0 = _time.perf_counter()
            last_response: Union[Dict[str, Any],None] = None
            for attempt in range(1, self.max_retries + 1):
                if self.llm_config.provider=='LOCAL':
                    last_response = await _do_request_local(payload)
                else:
                    last_response = await _do_request(payload)
                if last_response.get("error") is None:
                    latency = (_time.perf_counter() - t0) * 1000
                    usage = last_response.get("usage", {})
                    logger.debug(
                        "LLM response | %s latency=%.0fms prompt_tokens=%s completion_tokens=%s",
                        self._model_label_for_logs(), latency,
                        usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    )
                    return self._create_chat_result(payload, last_response)
                logger.warning(
                    "LLM retry %d/%d | %s error=%s",
                    attempt, self.max_retries, self._model_label_for_logs(), last_response.get("error"),
                )

            logger.error("LLM max retries exceeded | %s", self._model_label_for_logs())
            error = last_response.get("error") if last_response else None
            error_msg = ""
            if isinstance(error, dict):
                error_msg = error.get("message", str(error))
            elif error is not None:
                error_msg = str(error)
            signals = STREAM_SIGNALS.get()
            signals.status = "failed"
            signals.error = {"message": error_msg or "LLM call failed after retries", "retryable": True}
            return self._create_chat_result(payload, last_response)



    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Sync streaming that prevents deadlocks by using the running event loop.
        """
        loop = asyncio.get_event_loop()
        agen = self._astream(messages, stop, run_manager, **kwargs)

        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break


    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async streaming logic, now with tool call chunk support."""
        signals = STREAM_SIGNALS.get()
        payload = self._create_payload(messages, stream=True, stop=stop, **kwargs)

        try:
            response_iterator = await self.client.chat_completion(payload=payload)
        except Exception as exc:
            logger.warning("LLM stream connection failed | %s", exc)
            signals.status = "failed"
            signals.error = {"message": str(exc), "retryable": True}
            return

        buffer = ""
        first_chunk = True

        try:
            async for chunk_bytes in response_iterator:
                if signals.check_interrupted():
                    signals.status = "interrupted"
                    return

                raw_chunk = chunk_bytes.decode("utf-8", errors="replace")
                curr_chunk = buffer + raw_chunk
                *curr_chunk_lines, buffer = curr_chunk.splitlines()

                for line in curr_chunk_lines:
                    if not line.startswith('data: '):
                        continue
                    payload_str = line[6:].strip()
                    if payload_str == '[DONE]':
                        continue
                    try:
                        js = json.loads(payload_str)
                    except json.JSONDecodeError:
                        logger.warning("LLM stream JSON decode error: %.200s", payload_str)
                        continue

                    chunk_error = js.get("error")
                    if chunk_error:
                        error_msg = chunk_error.get("message", str(chunk_error)) if isinstance(chunk_error, dict) else str(chunk_error)
                        logger.warning("LLM stream returned error | %s error=%s", self._model_label_for_logs(), error_msg)
                        signals.status = "failed"
                        signals.error = {"message": error_msg, "retryable": True}
                        return

                    choices = js.get("choices")
                    if not choices:
                        raw_usage = js.get("usage")
                        if isinstance(raw_usage, dict) and raw_usage.get("total_tokens"):
                            resp_for_metrics = js
                            if not resp_for_metrics.get("total_time"):
                                lc = resp_for_metrics.get("latency_checkpoint") or {}
                                if lc.get("total_duration_ms") is not None:
                                    resp_for_metrics = {
                                        **js,
                                        "total_time": lc["total_duration_ms"],
                                    }
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    chunk_position="last",
                                    additional_kwargs={
                                        ENVELOPE_KEY: {
                                            "usage": self._to_usage_metrics(
                                                raw_usage, resp_for_metrics,
                                            ),
                                            "error": None,
                                            "choices_count": 1,
                                        }
                                    },
                                    response_metadata={
                                        "stream_usage": raw_usage,
                                        "stream_spending": float(
                                            resp_for_metrics.get("spending") or 0.0,
                                        ),
                                        "model_name": self._model_label_for_logs(),
                                    },
                                )
                            )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content: str = delta.get("content") or ""
                    tool_call_chunks = delta.get("tool_calls") or []

                    additional_kwargs: Dict[str, Any] = {}

                    if first_chunk:
                        additional_kwargs[ENVELOPE_KEY] = {
                            "usage": None,
                            "error": None,
                            "choices_count": 1,
                        }
                        first_chunk = False

                    additional_kwargs[OPENAI_CHOICES_KEY] = {
                        "id": choice.get("id") or js.get("id"),
                        "finishReason": choice.get("finish_reason"),
                        "index": choice.get("index", 0),
                    }

                    parsed_tool_call_chunks = self._parse_tool_call_chunks(tool_call_chunks)
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=content,
                            tool_call_chunks=parsed_tool_call_chunks,
                            additional_kwargs=additional_kwargs,
                            response_metadata={"raw_sse": line},

                        )
                        )

                    if run_manager and content:
                        await run_manager.on_llm_new_token(content)
        except Exception as exc:
            logger.warning("LLM stream error mid-flight | %s", exc)
            signals.status = "failed"
            signals.error = {"message": str(exc), "retryable": True}


    async def _agenerate_with_cache(self, *args, **kwargs) -> ChatResult:
        signals = STREAM_SIGNALS.get()
        signals.reset_status()
        try:
            result = await super()._agenerate_with_cache(*args, **kwargs)
        except (ValueError, IndexError):
            if signals.status:
                result = ChatResult(generations=[
                    ChatGeneration(message=AIMessage(content="")),
                ])
            else:
                raise
        if signals.status:
            strip_tool_calls_from_result(result, signals.status, signals.error)
        return result

    def _parse_tool_call_chunks(self, tool_call_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not tool_call_chunks:
            return []

        parsed_tool_call_chunks = []
        for single_tool_call_chunk in tool_call_chunks:
            valid_tool_call_chunk_keys = ['id', 'name', 'arguments', 'index']
            parsed_single_tool_call_chunk ={
                k: single_tool_call_chunk.get(
                    k, single_tool_call_chunk.get('function', {}).get(k, None)
                )
                for k in valid_tool_call_chunk_keys
            }
            parsed_single_tool_call_chunk['args'] = parsed_single_tool_call_chunk.pop('arguments')
            parsed_tool_call_chunks.append(parsed_single_tool_call_chunk)
        return parsed_tool_call_chunks

    def _create_payload(
            self,
            messages: List[BaseMessage],
            stream: bool,
            stop: Optional[List[str]],
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Constructs the final request payload for the API without calling the
        non-existent 'get_invocation_params' method.
        """
        # Start with a copy of the default parameters.
        payload = self._default_params.copy()

        # Update the payload with any runtime arguments. This is critical
        # for allowing overrides and for passing bound parameters like 'tools'.
        payload.update(kwargs)

        # Add stop sequences if they are provided for this specific call.
        if stop is not None:
            payload["stop"] = stop

        # Add the formatted messages and the stream flag.
        payload["messages"] = [_convert_message_to_dict(m) for m in messages]
        payload["stream"] = stream

        return payload

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Binds tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind.
            **kwargs: Additional parameters to pass to the runnable constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        """
        A convenience method for binding a single tool for structured output.
        This version is corrected to handle Pydantic models and dicts.
        """
        # First, convert the provided schema (Pydantic model, dict, etc.)
        # into the standard OpenAI tool dictionary format.
        formatted_tool = convert_to_openai_tool(schema)

        # Now we can reliably extract the function's name from the standardized dict.
        tool_name = formatted_tool.get("function", {}).get("name")
        if not tool_name:
            # Fallback or raise an error if the name can't be found
            raise ValueError(f"Could not determine tool name from schema: {schema}")

        # The 'tool_choice' parameter forces the model to call the specified tool.
        tool_choice = {
            "type": "function",
            "function": {"name": tool_name}
        }

        # Bind the original schema and the new tool_choice argument to the model.
        return self.bind_tools(
            [schema],  # Pass the original schema here
            tool_choice=tool_choice,
            **kwargs
        )
