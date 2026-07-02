from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage, AIMessageChunk,
    BaseMessage, BaseMessageChunk,
    HumanMessage, HumanMessageChunk,
    SystemMessage, SystemMessageChunk,
    ToolMessage, ToolMessageChunk,
)

from agent_builder.utils.constants import (
    AGENT_DOC,
    CONFIG_VARIABLES,
    ENVELOPE_FIELDS,
    ENVELOPE_KEY,
    INVOKE_MESSAGE_COUNT,
    LAST_ACTIVE_TASK,
    MESSAGES,
    RAW_RESPONSE_KEY,
    _RAW_REMOTE_CONTENT,
    REMOTE_REQUEST,
    STREAM,
)
from agent_builder.utils.misc import remove_system_prompt
from agent_builder.handlers.core.requests import (
    AgentInvokeRequest,
    ApiMessage,
    MessageContent,
)
from agent_builder.handlers.core.responses import (
    AgentInvokeResponse,
    TimingMetrics,
    UsageMetrics,
)
from agent_builder.base.state import get_initial_state


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Role maps and message-class dispatch
# ---------------------------------------------------------------------------

API_TO_LANGCHAIN_ROLE: dict[str, str] = {
    "assistant": "ai",
    "system": "system",
    "user": "human",
    "tool": "tool",
}

LANGCHAIN_TO_API_ROLE: dict[str, str] = {
    "AIMessageChunk": "assistant",
    **{v: k for k, v in API_TO_LANGCHAIN_ROLE.items()},
}

_ROLE_TO_MESSAGE_CLASS: dict[str, type[BaseMessage]] = {
    "assistant": AIMessage,
    "user": HumanMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}

_ROLE_TO_CHUNK_CLASS: dict[str, type[BaseMessageChunk]] = {
    "assistant": AIMessageChunk,
    "user": HumanMessageChunk,
    "system": SystemMessageChunk,
    "tool": ToolMessageChunk,
}


def message_class_for_role(role: str) -> type[BaseMessage]:
    """Return the LangChain message class for a copilot API role string."""
    return _ROLE_TO_MESSAGE_CLASS.get(role, AIMessage)


def chunk_class_for_role(role: str) -> type[BaseMessageChunk]:
    """Return the LangChain message *chunk* class for a copilot API role string."""
    return _ROLE_TO_CHUNK_CLASS.get(role, AIMessageChunk)


# ---------------------------------------------------------------------------
#  Private helpers
# ---------------------------------------------------------------------------


def _sanitize_datetimes(d: dict) -> dict:
    """Return a shallow copy of *d* with datetime values converted to ISO strings."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        elif isinstance(value, dict):
            out[key] = _sanitize_datetimes(value)
        elif isinstance(value, list):
            out[key] = [
                _sanitize_datetimes(item) if isinstance(item, dict)
                else item.isoformat() if isinstance(item, datetime)
                else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def get_plain_text_from_content(
    content: Optional[Iterable[dict[str, Any] | MessageContent]],
    *,
    for_remote_invoke: bool,
) -> str:
    """Get plain text from a message ``content`` list; remote bodies use ``response.text`` only."""
    if content is None:
        return ""
    allowed_types = {"response.text"} if for_remote_invoke else {"", "response.text", "input.text"}
    parts: list[str] = []
    for item in content:
        if isinstance(item, MessageContent):
            typ, txt = str(item.type or ""), item.text
        else:
            typ, txt = str(item.get("type") or ""), item.get("text")
        if isinstance(txt, str) and txt and typ in allowed_types:
            parts.append(txt)
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  1. REQUEST → STATE  (inbound: invoke handler receives a request)
# ═══════════════════════════════════════════════════════════════════════════

def _api_message_to_lc_message(msg: ApiMessage) -> BaseMessage:
    """Convert one inbound ``ApiMessage`` into a LangChain ``BaseMessage`` (``request_to_state``)."""
    role = msg.role
    content_list = msg.content or []
    additional_kwargs: dict[str, Any] = {}

    rows = [m.model_dump() for m in content_list]
    additional_kwargs[_RAW_REMOTE_CONTENT] = rows
    text = get_plain_text_from_content(content_list, for_remote_invoke=False)

    cls = message_class_for_role(role)
    lc_msg = cls(content=text, additional_kwargs=additional_kwargs)
    if msg.id:
        lc_msg.id = msg.id
    return lc_msg


def request_to_state(
    request: AgentInvokeRequest
) -> dict[str, Any]:
    """Map ``AgentInvokeRequest`` to agent state (``messages``, ``remote_request``, session)."""
    request.messages = [m for m in request.messages if m.role.lower() != "system"]
    state = get_initial_state(request.sessionId)

    state['request_id'] = request.id
    state["remote_request"] = request.model_dump()

    #copilots expect this
    state['remote_request']['apiVersion'] = state['remote_request'].get('apiVersion', '1.0')

    state[MESSAGES] = [_api_message_to_lc_message(m) for m in request.messages]

    for msg in reversed(state[MESSAGES]):
        if msg.type == "human":
            if not msg.id:
                msg.id = request.id
            msg.additional_kwargs["responseId"] = msg.id
            break

    logger.debug("request_to_state | messages=%d", len(request.messages))
    return state


# ═══════════════════════════════════════════════════════════════════════════
#  2. STATE → REQUEST  (outbound: sending a remote request)
# ═══════════════════════════════════════════════════════════════════════════


def _content_blocks_for_remote_invoke(
    role: str,
    *,
    base_text: str,
    raw_items: Any,
) -> list[dict[str, Any]]:
    """Build invoke-agent ``content[]`` for one outbound message row.

    When the LangChain message carries the original request/response block list in
    ``additional_kwargs[_RAW_REMOTE_CONTENT]`` (any role), that list is re-emitted so widgets,
    images, and other non-text fields are preserved. Otherwise synthesize a single typed text
    block matching the contract (``input.text`` / ``response.text``, ``index``, ``textFormat``).
    """
    if raw_items:
        blocks = [x for x in raw_items]
        if blocks:
            return blocks

    if role == "user":
        block_type = "input.text"
    else:
        block_type = "response.text"

    return [
        {
            "type": block_type,
            "index": 0,
            "text": base_text,
            "textFormat": "markdown",
        }
    ]


def lc_message_to_api_message(msg: BaseMessage) -> dict:
    """Convert one LangChain ``BaseMessage`` to an invoke-agent ``messages[]`` row (``role`` + ``content``).

    Used when assembling ``messages`` inside ``state_to_request_payload`` (or any other outbound
    envelope that expects unified API message rows).
    ``content`` is always a block list per the invoke-agent contract. Tool-call metadata is not
    serialized on the wire; the contract does not define ``tool_calls`` / ``actions`` /
    ``tool_call_id`` on messages.
    """
    role = LANGCHAIN_TO_API_ROLE.get(msg.type, msg.type)
    extra = msg.additional_kwargs or {}
    base_text = msg.content if isinstance(msg.content, str) else str(msg.content)
    raw_items = extra.get(_RAW_REMOTE_CONTENT)

    content: Any = _content_blocks_for_remote_invoke(
        role, base_text=base_text, raw_items=raw_items
    )

    row: dict[str, Any] = {
        "role": role,
        "content": content,
    }
    if msg.id:
        row["id"] = msg.id
    response_id = extra.get("responseId")
    if response_id:
        row["responseId"] = response_id
    return row


def state_to_request_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Build downstream remote ``/invoke`` JSON from state (reuse ``remote_request`` envelope fields)."""
    _exclude = {MESSAGES, STREAM}
    remote_req = state.get(REMOTE_REQUEST)

    state_messages = [m for m in state.get(MESSAGES, []) if not isinstance(m, (SystemMessage, ToolMessage))]

    if remote_req:
        n_state_messages = len(state_messages)
        n_remote_messages = len(remote_req.get("messages", []))

        if n_remote_messages == n_state_messages:
            out = {**remote_req, "lastActiveTask": state.get(LAST_ACTIVE_TASK)}
        else:
            out = {k: v for k, v in remote_req.items() if k not in _exclude}
            out["messages"] = [lc_message_to_api_message(m) for m in state_messages]
            out["lastActiveTask"] = state.get(LAST_ACTIVE_TASK)
    else:
        out = {
            "agentId":        state.get("agent_id", ""),
            "sessionId":      state.get("session_id", ""),
            "id":             state.get("request_id", ""),
            "apiVersion":     "1.0",
            "messages":       [lc_message_to_api_message(m) for m in state_messages],
            "lastActiveTask": state.get(LAST_ACTIVE_TASK),
        }

    rid = state.get("request_id")
    if rid:
        out["id"] = rid

    mid = state.get("response_id")
    if mid:
        out["responseId"] = mid

    out["conversationState"] = "stateless"

    out["delivery"] = dict((state.get(CONFIG_VARIABLES) or {}).get("invokeDelivery") or {"mode": "foreground"})

    config_vars = state.get(CONFIG_VARIABLES) or {}

    if agent_doc := config_vars.get(AGENT_DOC):
        out[AGENT_DOC] = agent_doc

    if mcp_config := config_vars.get("mcp_config"):
        out["mcpConfig"] = mcp_config

    return _sanitize_datetimes(out)


# ═══════════════════════════════════════════════════════════════════════════
#  3. API RESPONSE → LC MESSAGE  (inbound: remote HTTP JSON, not handler ``state_to_response``)
# ═══════════════════════════════════════════════════════════════════════════


def api_response_to_lc_message(
    data: dict[str, Any] | AgentInvokeResponse,
) -> AIMessage:
    """Map remote sync invoke JSON to one ``AIMessage`` (empty if no ``content``)."""
    if isinstance(data, AgentInvokeResponse):
        data = data.model_dump(exclude_none=True)

    envelope = {k: data.get(k) for k in ENVELOPE_FIELDS}
    envelope["is_remote_response"] = True
    raw_content = data.get("content")
    if isinstance(raw_content, list) and len(raw_content) > 0:
        content = list(raw_content)
        msg = AIMessage(
            content=get_plain_text_from_content(content, for_remote_invoke=True),
        )
        msg.additional_kwargs[ENVELOPE_KEY] = envelope
        msg.additional_kwargs[_RAW_REMOTE_CONTENT] = content
        msg.response_metadata[RAW_RESPONSE_KEY] = data
        return msg

    empty = AIMessage(content="")
    empty.additional_kwargs[ENVELOPE_KEY] = envelope
    empty.response_metadata[RAW_RESPONSE_KEY] = data
    return empty


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_timing(timings: List[Optional[TimingMetrics]]) -> Optional[TimingMetrics]:
    if not timings:
        return None

    ttft_vals        = [t["ttft"]         for t in timings if t and t.get("ttft")         is not None]
    ttlt_vals        = [t["ttlt"]         for t in timings if t and t.get("ttlt")         is not None]
    totaltime_vals   = [t["totalTime"]    for t in timings if t and t.get("totalTime")    is not None]
    thinking_vals    = [t["thinkingTime"] for t in timings if t and t.get("thinkingTime") is not None]

    if not (ttft_vals or ttlt_vals or totaltime_vals or thinking_vals):
        return None

    return TimingMetrics(
        ttft         = min(ttft_vals)      if ttft_vals      else None,
        ttlt         = max(ttlt_vals)      if ttlt_vals      else None,
        totalTime    = sum(totaltime_vals) if totaltime_vals else None,
        thinkingTime = sum(thinking_vals)  if thinking_vals  else None,
    )


def _aggregate_additional(metrics_list: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, List[Any]]]:
    if not metrics_list:
        return None

    agg: Dict[str, List[Any]] = {}
    for d in metrics_list:
        if not d:
            continue
        for k, v in d.items():
            agg.setdefault(k, []).append(v)
    return agg or None


# HTTP codes treated as retryable when building ``error`` for local handler failures (not remote SSE).
INVOKE_PUBLIC_ERROR_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset(
    {408, 425, 429, 500, 502, 503, 504},
)


def collapse_errors_and_usage(
    errors: list[Any],
    usages: list[Any],
) -> Tuple[dict[str, Any], dict[str, Any]]:
    # One top-level ``error`` on the invoke body: last non-empty ``message`` wins; ``retryable`` is true if any envelope had it.
    agg_error: dict[str, Any] = {"message": None, "retryable": False}
    for e in errors:
        if not isinstance(e, dict):
            continue
        if e.get("retryable") is True:
            agg_error["retryable"] = True
        m = e.get("message")
        if isinstance(m, str) and m.strip():
            agg_error["message"] = m.strip()
        elif m is not None:
            s = str(m).strip()
            if s:
                agg_error["message"] = s

    total_cost = sum(um["totalCost"] for um in usages if um.get("totalCost") is not None)

    total_calls_int = sum(int(um["numCalls"]) if um.get("numCalls") is not None else 0 for um in usages)

    timing_agg = _aggregate_timing([um.get("timing") for um in usages])

    model_breakdown = [
        mbe
        for um in usages
        for mbe in (um.get("modelBreakdown") or [])
    ] or None

    component_breakdown = [
        cbe
        for um in usages
        for cbe in (um.get("componentBreakdown") or [])
    ] or None

    additional = _aggregate_additional([um.get("additionalMetrics") for um in usages])

    agg_usage = UsageMetrics(
        totalCost          = total_cost,
        numCalls           = total_calls_int,
        timing             = timing_agg,
        modelBreakdown     = model_breakdown,
        componentBreakdown = component_breakdown,
        additionalMetrics  = additional,
    ).model_dump()

    return agg_error, agg_usage


# ═══════════════════════════════════════════════════════════════════════════
#  5. MESSAGES → ACCUMULATED INVOKE JSON  (outbound: sync ``/invoke`` ``content`` contract)
# ═══════════════════════════════════════════════════════════════════════════



def lc_messages_to_content_blocks(
    messages: list[BaseMessage],
) -> tuple[list[dict[str, Any]], str]:
    """Build invoke ``content[]`` from LangChain messages; return blocks and joined ``response.text``."""
    content: list[dict[str, Any]] = []
    next_index = 0
    primary_text = ""

    for msg in messages:
        if isinstance(msg, (HumanMessage, HumanMessageChunk, SystemMessage, SystemMessageChunk, ToolMessage, ToolMessageChunk)):
            continue

        raw_items = (msg.additional_kwargs or {}).get(_RAW_REMOTE_CONTENT)
        if isinstance(raw_items, list) and raw_items:
            for item in raw_items:
                row = item.copy()
                row["index"] = next_index
                next_index += 1
                content.append(row)
                if (
                    str(row.get("type") or "") == "response.text"
                    and isinstance(row.get("text"), str)
                    and row["text"].strip()
                ):
                    primary_text = row["text"].strip()
            continue

        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        if body.strip():
            content.append({"type": "response.text", "index": next_index, "text": body.strip()})
            primary_text = body.strip()
            next_index += 1

    rt_parts: list[str] = []
    for row in content:
        if str(row.get("type") or "") != "response.text":
            continue
        t = row.get("text")
        if isinstance(t, str) and t.strip():
            rt_parts.append(t.strip())
    if rt_parts:
        primary_text = "\n".join(rt_parts)

    return content, primary_text


# ═══════════════════════════════════════════════════════════════════════════
#  4. STATE → RESPONSE  (outbound: invoke handler sends a response)
# ═══════════════════════════════════════════════════════════════════════════


def extract_usage_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Extract aggregated usage metrics from graph state without building the full response."""
    remote_req = state.get(REMOTE_REQUEST, {})
    cfg = state.get(CONFIG_VARIABLES) or {}
    slice_n = cfg.get(INVOKE_MESSAGE_COUNT)
    if isinstance(slice_n, int) and slice_n >= 0:
        n_req_messages = slice_n
    else:
        n_req_messages = len(remote_req.get("messages", []))
    usage_raw: list[Any] = []
    for msg in state.get(MESSAGES, [])[n_req_messages:]:
        envelope = (msg.additional_kwargs or {}).get(ENVELOPE_KEY)
        if envelope and envelope.get("usage"):
            usage_raw.append(envelope["usage"])
    if not usage_raw:
        return {}
    _, agg_usage = collapse_errors_and_usage([], usage_raw)
    return agg_usage if isinstance(agg_usage, dict) else {}


def state_to_response(state: dict[str, Any]) -> dict[str, Any]:
    """Accumulated invoke response from state (output messages after request prefix)."""
    state = remove_system_prompt(state)
    remote_req = state.get(REMOTE_REQUEST, {})
    rid = state.get("request_id")
    sid = state.get("session_id")
    cfg = state.get(CONFIG_VARIABLES) or {}
    slice_n = cfg.get(INVOKE_MESSAGE_COUNT)
    if isinstance(slice_n, int) and slice_n >= 0:
        n_req_messages = slice_n
    else:
        n_req_messages = len(remote_req.get("messages", []))
    all_msgs = state.get(MESSAGES, [])
    output_msgs = all_msgs[n_req_messages:]

    usage_raw: list[Any] = []
    error_raw: list[Any] = []
    for msg in output_msgs:
        envelope = (msg.additional_kwargs or {}).get(ENVELOPE_KEY)
        if envelope:
            if envelope.get("usage"):
                usage_raw.append(envelope["usage"])
            if envelope.get("error"):
                error_raw.append(envelope["error"])

    agg_error, agg_usage = collapse_errors_and_usage(error_raw, usage_raw)

    content, primary_text = lc_messages_to_content_blocks(output_msgs)
    err_out = agg_error
    status = "FAILED" if err_out.get("message") else "COMPLETED"
    usage = agg_usage if isinstance(agg_usage, dict) else {}

    now = int(time.time())
    mid = state.get("response_id") or rid or ""
    out: dict[str, Any] = {
        "apiVersion": "1.0",
        "sessionId": sid or "",
        "id": rid or "",
        "responseId": mid,
        "createdAt": now,
        "updatedAt": now,
        "content": content,
        "status": status,
        "index": 0,
        "text": primary_text,
        "error": err_out,
        "usage": usage,
    }
    return _sanitize_datetimes(out)
