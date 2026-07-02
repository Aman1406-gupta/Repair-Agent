import json
from typing import Any, Dict, List
from langchain_core.messages import (
    AIMessage, AIMessageChunk, BaseMessage,
    HumanMessage, SystemMessage, ToolMessage, ChatMessage,
)
from agent_builder.utils.constants import ENVELOPE_KEY, OPENAI_CHOICES_KEY


def parse_langgraph_tool_calls(langgraph_tool_calls):
    parsed_tool_calls = []
    for x in langgraph_tool_calls:
        parsed_tool_calls.append({
            'index': 0,
            'id': x['id'],
            'type': 'function',
            'function': {
                'name': x['name'],
                'arguments': json.dumps(x['args'])
            }
        })

    return parsed_tool_calls

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Converts a LangChain message to a dictionary expected by the API."""
    # Mapping from LangChain message types to API role strings
    role_mapping = {
        "human": "user",
        "ai": "assistant",
        "AIMessageChunk": "assistant"
    }
    role = role_mapping.get(message.type, message.type)

    message_dict = {"role": role, "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        if isinstance(message, AIMessageChunk) or 'tool_calls' not in message.model_dump()['additional_kwargs']:
            message_dict["tool_calls"] = parse_langgraph_tool_calls(message.tool_calls)
        else:
            message_dict["tool_calls"] = message.model_dump()['additional_kwargs']['tool_calls']
        if not message.content:
            message_dict["content"] = None

    if isinstance(message, ToolMessage):
        message_dict["tool_call_id"] = message.tool_call_id
    return message_dict


def _convert_dict_to_message(
    d: Dict[str, Any],
    disable_streaming: bool,
    *,
    choice_meta: Dict[str, Any] | None = None,
    envelope: Dict[str, Any] | None = None,
) -> BaseMessage:
    """Converts a dictionary from the API response to a LangChain message."""
    role = d.get("role")
    content = d.get("content", "") or ""

    if role == "assistant":
        kwargs: Dict[str, Any] = {'disable_streaming': disable_streaming}
        if tool_calls := d.get("tool_calls"):
            kwargs["tool_calls"] = tool_calls
        if choice_meta:
            kwargs[OPENAI_CHOICES_KEY] = choice_meta
        if envelope:
            kwargs[ENVELOPE_KEY] = envelope
        return AIMessage(content=content, additional_kwargs=kwargs)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content, tool_call_id=d["tool_call_id"])
    else:
        return ChatMessage(role=role, content=content)

