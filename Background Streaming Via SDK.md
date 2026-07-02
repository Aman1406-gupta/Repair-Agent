# Background Streaming SDK

> Stream results from a long-running remote task back into Agent Builder — text, thinking, images, videos, or any custom typed frame.

A task registered behind an `http_config.url` runs out-of-process. Instead of hand-rolling typed frames and HTTP calls (see [`mock_remote_task_server.py`](../../mock_remote_task_server.py)), your runtime uses [`BackgroundStreamer`](sdk_async_client.py) to:

1. **Handshake** — emit `stream.start` + `mode.background` on the invoke response wire, telling Agent Builder / the UI to switch to polling `/message`.
2. **Stream** — build typed events locally and push them to `POST /message/ingest`.
3. **Finish** — push one terminal frame (`complete` / `fail` / `interrupt`).

---

## The mental model — blocks in, one event out

```
blocks   →   send_event
(dicts)      (build one typed event + post it)
```

| Layer | What you call | Network? |
|---|---|---|
| **Blocks** | `text_block`, `thinking_block`, `image_block`, `video_block`, `content_block` from [`response_formats.py`](response_formats.py) | No — plain dict factories |
| **Send** | `await run.send_event(content, event_type=...)` — builds **one** frame (stamps `sequence`/`id`), posts it | Yes |

Index rule: a block without an `index` automatically gets the run's current `get_index()`; call `increase_index()` to move to the next block position. An explicit `index` on a block always wins.

(`await run.send_events([...])` also exists as a low-level escape hatch for posting pre-built frames in one batch.)

---

## Quickstart

```python
from agent_builder.llm_client.utils.response_formats import (
    text_block, thinking_block, image_block, video_block, content_block,
)
from agent_builder.llm_client.utils.sdk_async_client import agent_builder_client

# One per request. Borrows a process-wide connection pool — nothing to close.
run = agent_builder_client.get_background_streamer(
    session_id=session_id,
    request_id=request_id,
    agent_id=agent_id,          # optional
)

# 1. Return the handshake on your invoke HTTP response (SSE body), then keep working.
handshake = run.handshake_sse()

# 2. Stream as you go.
await run.send_event("working on it...")

# 3. Finish.
await run.complete()
```

Configure the target service once at startup — pass `base_url` explicitly or set the `AGENT_BUILDER_URL` env var:

```python
agent_builder_client.configure(base_url="https://agent-builder.my-env.internal")
```

---

## Examples

### Text

```python
# A str is shorthand for one text_block.
await run.send_event("here is the answer")
```

### Thinking

```python
await run.send_event(thinking_block("checking sources..."))
```

### Streaming one block across many events

Deltas for the *same* block (e.g. token-by-token text) must share one index — that's why blocks without an `index` get the run's current counter automatically. Bump it when the next block should render as a new position:

```python
# both deltas append to block 0 (counter starts at 0)
await run.send_event(text_block("hel"))
await run.send_event(text_block("lo"))

run.increase_index()                                  # next block renders as position 1
await run.send_event(image_block("https://cdn/x.png"))
```

### Mixed content in ONE `content.delta`

Put any number of blocks in one event. Give each its own position — blocks without an `index` all share the *current* counter value, so bump it per block:

```python
await run.send_event([
    text_block("here is the result", index=run.get_index()),
    image_block("https://cdn/x.png", mime_type="image/png", alt_text="chart", index=run.increase_index()),
    video_block("https://cdn/x.mp4", mime_type="video/mp4", index=run.increase_index()),
])
```

An explicit index always wins and is never overwritten.

### Any other / future block type

Use the generic `content_block` — or a hand-written dict; blocks are plain dicts forwarded verbatim:

```python
await run.send_event(content_block("response.idea", summary="try variant B", score=3))
```

### Custom top-level frames (not `content.delta`)

Pass `event_type` to emit any typed frame as its own event — e.g. an `entity.chunk`:

```python
insight = {"type": "response.insight", "value": {"id": "ins-1", "title": "Spike in mentions"}}
await run.send_event({"entity": insight}, event_type="entity.chunk")
```

`content` must be a dict here; it is sent verbatim with `type`, `sequence`, and `id` stamped by the run.

### Finishing the run

Exactly one terminal frame ends the run (triggers server-side finalize):

```python
await run.complete({"totalTokens": 1234})        # success (usage optional)
await run.fail("upstream timed out", retryable=True)
await run.interrupt()                            # user/system cancelled
```

---

## End-to-end: a remote task endpoint

```python
import asyncio

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from agent_builder.llm_client.utils.response_formats import text_block, image_block
from agent_builder.llm_client.utils.sdk_async_client import (
    AgentBuilderClientError, agent_builder_client,
)

app = FastAPI()

@app.post("/invoke")
async def invoke(payload: dict):
    run = agent_builder_client.get_background_streamer(
        session_id=payload["sessionId"],
        request_id=payload["id"],
    )
    asyncio.create_task(do_work(run, payload))          # keep working after responding
    return PlainTextResponse(run.handshake_sse(), media_type="text/event-stream")

async def do_work(run, payload):
    try:
        await run.send_event("crunching the numbers...")
        result_png = await crunch(payload)
        await run.send_event([
            text_block("done — here's the chart", index=run.get_index()),
            image_block(result_png, mime_type="image/png", index=run.increase_index()),
        ])
        await run.complete()
    except AgentBuilderClientError:
        raise                                            # delivery failed after retries
    except Exception as exc:
        await run.fail(str(exc), retryable=False)
```

---

## Behavior notes

- **One sequence space.** The run owns a single `generation_id` (auto-minted unless you pass one) and one monotonic `sequence` counter starting at the handshake — every frame it produces is ordered.
- **Retries built in.** Every send retries 5xx / 429 / transport errors with exponential backoff (default 3 retries); other 4xx fail fast. After exhausting retries it raises `AgentBuilderClientError` (carries `.status_code`).
- **Nothing to close.** The `agent_builder_client` singleton owns one shared `httpx` pool for the pod's lifetime, closed automatically at exit (or call `await agent_builder_client.aclose()` from your shutdown hook). Per-run streamers borrow it.
- **Standalone use.** For scripts/tests you can construct `BackgroundStreamer(...)` directly — it then owns its own client, so close it (`async with BackgroundStreamer(...) as run:`).

## API reference

| Call | Sync/Async | Purpose |
|---|---|---|
| `text_block(text, *, index=None)` | sync | `response.text.delta` block |
| `thinking_block(text, *, index=None)` | sync | `thinking.text.delta` block |
| `image_block(url, *, mime_type=None, alt_text=None, index=None)` | sync | `response.image` block |
| `video_block(url, *, mime_type=None, index=None)` | sync | `response.video` block |
| `content_block(type, **fields)` | sync | any other block type |
| `run.handshake_sse()` | sync | `stream.start` + `mode.background` SSE for the invoke response |
| `run.get_index()` | sync | current content block index (starts at 0); fills any block missing an `index` |
| `run.increase_index()` | sync | advance to the next block position; returns the new index |
| `await run.send_event(content, *, event_type="content.delta")` | async | build one event from a str / block / block list (or a custom frame) and post it |
| `await run.send_events(events)` | async | low-level: post 1..n pre-built frames to `/message/ingest` |
| `await run.complete(usage=None)` | async | terminal `stream.completed` |
| `await run.fail(message, *, retryable=False)` | async | terminal `stream.failed` |
| `await run.interrupt(usage=None)` | async | terminal `stream.interrupted` |
