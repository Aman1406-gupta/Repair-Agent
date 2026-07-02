"""Helpers for folding background-streamed typed frames back into graph state."""

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage


def response_text_from_delta_event(event: dict) -> str:
    """Concatenate the ``response.text.delta`` text carried by a ``content.delta`` frame.

    Non-``content.delta`` frames (e.g. terminal ``stream.*`` events) yield ``""``.
    """
    if event.get("type") != "content.delta":
        return ""
    parts: list[str] = []
    for blk in event.get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "response.text.delta":
            t = blk.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def fold_assistant_text_into_state(state: dict, text: str) -> None:
    """Ensure the streamed assistant text lives in ``state['messages']`` as one ``AIMessage``.

    Background replies are streamed as ``content.delta`` chunks; the graph's ``values``
    state may carry an empty (or no) assistant message for the current turn. This folds the
    assembled text into the canonical history so a later stateful turn sees the reply. It is
    a no-op when the graph already assembled a non-empty assistant message (no duplication).
    """
    if not text:
        return
    messages = state.get("messages")
    if not isinstance(messages, list):
        return
    last_ai: Optional[AIMessage] = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break
        if isinstance(m, HumanMessage):
            break
    if last_ai is not None:
        content = last_ai.content
        if isinstance(content, str) and content.strip():
            return  # graph already assembled the reply
        if not content:  # empty placeholder → fill in place
            last_ai.content = text
            return
    messages.append(AIMessage(content=text))
