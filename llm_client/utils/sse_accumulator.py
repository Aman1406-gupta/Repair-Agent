"""
StreamMetadataAccumulator
~~~~~~~~~~~~~~~~~~~~~~~~~

Schema-driven accumulator for streaming JSON payloads (SSE chunks).

The schema mirrors the API response structure as a nested dict.
Strategies are registered via ``@strategy`` decorators.  The
accumulator is API-agnostic — swap the schema for any JSON shape.

Schema nodes
~~~~~~~~~~~~
Each key in a schema dict maps to one of:

* ``field("strategy_name", "group")`` — accumulate this value using
  the named strategy and file the result under the named output group.
* ``(DESCEND, sub_schema)`` — walk into a dict wrapper without storing it.
* ``(DESCEND_FIRST, sub_schema)`` — take ``array[0]`` and walk into it.
* ``SKIP`` — ignore this key entirely.

A ``__default__`` key at any schema level provides a fallback rule for
unknown fields (forward-compatible with new API fields).

Usage::

    acc = StreamMetadataAccumulator()
    for chunk in sse_chunks:
        acc.update(chunk)
    groups = acc.snapshot()
    # groups["envelope"] → {id, sessionId, usage, …}
    # groups["choice"]   → {citations, safetyMetadata, …}
    # groups["content"]  → {textFormat, parts, actions, …}
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional


# ── Sentinels ─────────────────────────────────────────────────────────

class _Sentinel:
    __slots__ = ("_name",)
    def __init__(self, name: str):
        self._name = name
    def __repr__(self) -> str:
        return self._name

SKIP          = _Sentinel("SKIP")
DESCEND       = _Sentinel("DESCEND")
DESCEND_FIRST = _Sentinel("DESCEND_FIRST")


def field(strategy: str, group: str) -> tuple:
    """Declare a leaf field: accumulate with *strategy*, file under *group*."""
    return (strategy, group)


# ── Strategy registry ─────────────────────────────────────────────────

_STRATEGIES: Dict[str, Callable[[Any, Any], Any]] = {}


def strategy(name: str):
    """Decorator that registers a two-arg ``(old, new) → merged`` function."""
    def decorator(fn: Callable[[Any, Any], Any]):
        _STRATEGIES[name] = fn
        return fn
    return decorator


def _is_empty(v: Any) -> bool:
    return v is None or v == {} or v == []


@strategy("first_non_null")
def _first_non_null(old: Any, new: Any) -> Any:
    return old if not _is_empty(old) else new


@strategy("last_non_null")
def _last_non_null(old: Any, new: Any) -> Any:
    return new if not _is_empty(new) else old


@strategy("last")
def _last(_old: Any, new: Any) -> Any:
    return new


@strategy("deep_merge")
def _deep_merge(old: Any, new: Any) -> Any:
    if _is_empty(old):
        return new
    if _is_empty(new):
        return old
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for k, v in new.items():
            if not _is_empty(v):
                merged[k] = v
        return merged
    return new


@strategy("collect_unique")
def _collect_unique(old: Any, new: Any) -> Any:
    s = old if isinstance(old, set) else set()
    if not _is_empty(new):
        s.add(
            json.dumps(new, sort_keys=True)
            if isinstance(new, (dict, list)) else new
        )
    return s


# ── Default schema (Copilot + OpenAI envelope) ───────────────────────

_CONTENT_SCHEMA = {
    "__default__":         field("last_non_null", "content"),
    "text":                SKIP,
    "textFormat":          field("first_non_null", "content"),
    "parts":               field("last_non_null",  "content"),
    "actions":             field("last_non_null",  "content"),
    "suggestedQuestions":   field("last_non_null",  "content"),
    "promptVariables":     field("last_non_null",  "content"),
}

_MSG_WRAPPER = (DESCEND, {"content": (DESCEND, _CONTENT_SCHEMA)})

DEFAULT_SCHEMA = {
    "__default__":        field("last_non_null",  "envelope"),
    "id":                 field("first_non_null", "envelope"),
    "object":             field("first_non_null", "envelope"),
    "created":            field("first_non_null", "envelope"),
    "model":              field("first_non_null", "envelope"),
    "apiVersion":         field("first_non_null", "envelope"),
    "sessionId":          field("first_non_null", "envelope"),
    "service_tier":       field("first_non_null", "envelope"),
    "system_fingerprint": field("first_non_null", "envelope"),
    "usage":              field("last_non_null",  "envelope"),
    "error":              field("last_non_null",  "envelope"),
    "seqNo":              field("last",           "envelope"),
    "additional":         field("deep_merge",     "envelope"),
    "choices": (DESCEND_FIRST, {
        "__default__":    field("last_non_null",  "choice"),
        "citations":      field("last_non_null",  "choice"),
        "safetyMetadata": field("last_non_null",  "choice"),
        "finishReason":   field("last_non_null",  "choice"),
        "delta":          _MSG_WRAPPER,
        "message":        _MSG_WRAPPER,
    }),
}


# ── Accumulator ───────────────────────────────────────────────────────

class StreamMetadataAccumulator:
    """Generic schema-driven accumulator for streaming JSON payloads."""

    def __init__(self, schema: Optional[dict] = None):
        self._schema = schema or DEFAULT_SCHEMA
        self._groups: Dict[str, Dict[str, Any]] = {}

    def update(self, chunk: dict) -> None:
        """Ingest one raw SSE chunk dict."""
        self._walk(chunk, self._schema)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return accumulated fields grouped by output bucket."""
        return {
            group: {
                k: sorted(v) if isinstance(v, set) else v
                for k, v in fields.items()
            }
            for group, fields in self._groups.items()
        }

    def _walk(self, data: Any, schema: dict) -> None:
        if not isinstance(data, dict):
            return
        default_rule = schema.get("__default__")
        for key, value in data.items():
            rule = schema.get(key)

            if rule is SKIP:
                continue

            if isinstance(rule, tuple) and rule[0] in (DESCEND, DESCEND_FIRST):
                nav, sub_schema = rule
                target = value
                if nav is DESCEND_FIRST and isinstance(value, list):
                    target = value[0] if value else None
                if isinstance(target, dict):
                    self._walk(target, sub_schema)
                continue

            if isinstance(rule, tuple):
                strat_name, group = rule
            elif default_rule:
                strat_name, group = default_rule
            else:
                continue

            bucket = self._groups.setdefault(group, {})
            bucket[key] = _STRATEGIES[strat_name](bucket.get(key), value)
