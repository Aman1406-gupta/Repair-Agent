"""
Storage-agnostic serialization / deserialization for graph state and session data.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage
from agent_builder.llm_client.utils.remote_adapter import LANGCHAIN_TO_API_ROLE, _ROLE_TO_MESSAGE_CLASS

# ── Helpers ────────────────────────────────────────────────────────────────────

def sanitize(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, bytes):
        return {"__type__": "bytes", "value": base64.b64encode(value).decode()}
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def unsanitize(value: Any) -> Any:
    if isinstance(value, dict) and "__type__" in value:
        t = value["__type__"]
        if t == "datetime":
            return datetime.fromisoformat(value["value"])
        if t == "bytes":
            return base64.b64decode(value["value"])
    if isinstance(value, dict):
        return {k: unsanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unsanitize(v) for v in value]
    return value


# ── Serialization ──────────────────────────────────────────────────────────────

def get_json_serializable_graph_state(state: Dict[str, Any]) -> str:
    """Convert a runtime graph-state dict into a JSON string."""
    serializable = {k: sanitize(v) for k, v in state.items() if k != "messages"}
    serializable["messages"] = [sanitize(msg.model_dump()) if isinstance(msg, BaseMessage) else sanitize(msg) for msg in state['messages']]
    return serializable


def serialize_session_data(session_data: Dict[str, Any]) -> str:
    """Serialize a full session envelope (which embeds a graph-state)."""
    serializable = {k: get_json_serializable_graph_state(v) if k == "graph_state" else sanitize(v) for k, v in session_data.items()}
    return json.dumps(serializable, default=str)

# ── Deserialization ────────────────────────────────────────────────────────────

def deserialize_session_data(data: str) -> Dict[str, Any]:
    """Deserialize a full session envelope, reconstructing the embedded graph-state."""
    try:
        session_data = json.loads(data)
        session_data = {k: unsanitize(v) for k, v in session_data.items()}
        session_data['graph_state']['messages'] = [_ROLE_TO_MESSAGE_CLASS[LANGCHAIN_TO_API_ROLE[m['type']]](**m) for m in session_data['graph_state']['messages']]
        return session_data
    except Exception:
        logger.error("Failed to deserialize session data")
        raise
