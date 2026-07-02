import logging
from typing import Any, Dict, List, Optional, NamedTuple
from llm_router.sdk.client import LLMClient
from litellm import acompletion as litellm_async_completion
from copy import deepcopy
import json

logger = logging.getLogger(__name__)

def add_system_prompt_to_messages(kwargs):
    system_prompt = kwargs.get("system_prompt", None)
    messages = kwargs.get("messages")
    assert messages, f"Please pass 'messages' argument to add system prompt"

    if system_prompt:
        messages = [{'role': 'system', 'content': system_prompt}] + messages
        kwargs["messages"] = messages
        del kwargs['system_prompt']

class ParsedChatResponse(NamedTuple):
    """
    A standardized container for the model's response.
    """
    raw_response: Any
    agent_message_for_history: Any
    agent_thoughts: Optional[str] = None
    text_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage_stats: Optional[Dict] = None

class UnifiedLLMClient():
    def __init__(self, llm_router_url=None, vertex_credentials_file=None):
        assert llm_router_url or vertex_credentials_file, f"Either llm_router_url or vertex_credentials_file is required."
        assert not (llm_router_url and vertex_credentials_file), f"Only pass one of either llm_router_url or vertex_credentials_file."

        self.using_llm_router = (llm_router_url is not None)
        self._universal_defaults = {
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 1.0,
        }

        if self.using_llm_router:
            def _llm_router_completion_wrapper(**kwargs):
                return llm_router_client.completion(kwargs)

            def _modify_kwargs(**kwargs):
                add_system_prompt_to_messages(kwargs)
                return kwargs

            llm_router_client =  LLMClient(service_url=llm_router_url)
            self._completion = _llm_router_completion_wrapper
            self._wrapper_defaults = {
                "client_identifier": "ml-ca-dev",
                "tracking_params": {"release": "ca_research", "feature": "AGENT_BUILDER"},
                "provider": "AZURE_OPEN_AI",
            }
            self.modify_kwargs = _modify_kwargs
        else:
            with open(vertex_credentials_file, 'r') as file:
                vertex_credentials = json.load(file)
            vertex_credentials_json = json.dumps(vertex_credentials)

            def _modify_kwargs(**kwargs):
                add_system_prompt_to_messages(kwargs)
                kwargs['model'] = f"vertex_ai/{kwargs['model']}"
                return kwargs

            self._completion = litellm_async_completion
            self._wrapper_defaults = {
                # "tool_choice": "auto",
                # "parallel_tool_calls": True,
                "vertex_location": "us-east5",
                "vertex_credentials": vertex_credentials_json
            }
            self.modify_kwargs = _modify_kwargs


    async def completion(self, **kwargs):
        curr_kwargs = deepcopy(self._universal_defaults)
        curr_kwargs.update(deepcopy(self._wrapper_defaults))
        curr_kwargs.update(kwargs)
        curr_kwargs = self.modify_kwargs(**curr_kwargs)

        logger.debug("UnifiedLLM call | model=%s", curr_kwargs.get("model"))
        resp = await self._completion(**curr_kwargs)
        resp = resp if self.using_llm_router else resp.model_dump()
        agent_message = resp.get("choices", [{}])[0].get("message", {})
        raw_tool_calls = agent_message.get("tool_calls") or []
        parsed_tool_calls = []
        for raw_call in raw_tool_calls:
            call = deepcopy(raw_call)
            try:
                call["function"]["arguments"] = json.loads(raw_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                call['function']['arguments'] = {}
            parsed_tool_calls.append(call)

        usage_stats = resp.get("usage")
        parsed_usage_stats = {}
        for k, v in usage_stats.items():
            if isinstance(v, int):
                parsed_usage_stats[k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, int):
                        parsed_usage_stats[f"{k}_{k2}"] = v2

        return ParsedChatResponse(
            raw_response=resp,
            agent_message_for_history=agent_message,
            agent_thoughts=agent_message.get("reasoning_content", None),
            text_content=agent_message.get("content", None),
            tool_calls=parsed_tool_calls,
            usage_stats=parsed_usage_stats
        )
