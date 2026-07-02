from __future__ import annotations
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.types import Command
from typing import Any
import json
import yaml

import logging
logger = logging.getLogger(__name__)

def convert_objectid_to_str(obj):
    """Recursively convert BSON ObjectId instances to strings for JSON serialization."""
    from bson import ObjectId
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    return obj


def restore_original_messages(original_messages, preprocessed_len, output_state):
    """Replace preprocessed messages with originals, keeping only newly generated messages."""
    if isinstance(output_state, Command):
        return output_state
    new_messages = output_state['messages'][preprocessed_len:]
    output_state['messages'] = original_messages + new_messages
    return output_state

def strip_ephemeral_metadata(message: BaseMessage) -> BaseMessage:
    """Remove streaming-only metadata (e.g. raw_sse) that should never persist in state."""
    _EPHEMERAL_KEYS = {"raw_sse", "passthrough_sse"}
    if hasattr(message, "response_metadata") and message.response_metadata:
        message.response_metadata = {
            k: v
            for k, v in message.response_metadata.items()
            if k not in _EPHEMERAL_KEYS
        }
    return message


def add_system_prompt(input_state, system_template):
    messages = input_state.get("messages", [])
    if messages and isinstance(messages[0], SystemMessage):
        messages = messages[1:]
    system_msg = SystemMessage(content=system_template)
    input_state["messages"] = [system_msg] + messages
    return input_state

def remove_system_prompt(updated_state):
    messages = updated_state.get("messages", [])
    if messages and isinstance(messages[0], SystemMessage):
        updated_state["messages"] = messages[1:]
    return updated_state



def clean_json_string(content: str) -> str:
    """Remove markdown code block markers from JSON strings."""
    content = content.strip()
    for prefix in ["```json", "```"]:
        if content.startswith(prefix):
            content = content[len(prefix):]
            break

    for suffix in ["```"]:
        if content.endswith(suffix):
            content = content[:-len(suffix)]
            break

    return content.strip()


def _loads_if_json_str(value: Any, field_name: str) -> Any:
    """
    Convert a string value to a Python object.
    Supports both JSON and YAML formats.
    If the value is already a dict/object, returns it as-is.
    """
    if isinstance(value, str):
        # First try JSON parsing
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # If JSON fails, try YAML parsing
            try:
                result = yaml.safe_load(value)
                # yaml.safe_load can return None for empty strings,
                # strings, numbers, etc. We want dict objects for schemas.
                if not isinstance(result, dict):
                    raise ValueError(f"{field_name} must be a JSON object or YAML document that represents an object")
                return result
            except yaml.YAMLError as e:
                raise ValueError(f"{field_name} must be a valid JSON object or YAML document. YAML error: {str(e)}")
            except Exception as e:
                raise ValueError(f"{field_name} must be a valid JSON object or YAML document. Error: {str(e)}")
    return value
