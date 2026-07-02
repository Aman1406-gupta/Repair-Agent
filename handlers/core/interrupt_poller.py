"""
Per-request interrupt poller.

Polls Redis ``exec_status`` at a configurable interval and fires an
``asyncio.Event`` when the execution is marked as ``stopped``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

INTERRUPT_POLL_ENABLED = os.getenv("ENABLE_INTERRUPT_POLLING", "").lower() in ("1", "true", "yes", "on")
INTERRUPT_POLL_INTERVAL = float(os.getenv("INTERRUPT_POLL_INTERVAL", "2.0"))


class InterruptPoller:
    """Poll Redis for an interrupt signal on a specific session/request.

    One instance per request execution — no shared state between sessions.
    """

    def __init__(
        self,
        redis_client: Any,
        session_id: str,
        request_id: str,
        *,
        poll_interval: Optional[float] = None,
    ) -> None:
        self._redis_client = redis_client
        self._session_id = session_id
        self._request_id = request_id
        self._poll_interval = poll_interval or INTERRUPT_POLL_INTERVAL
        self._interrupted = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    @property
    def interrupted(self) -> asyncio.Event:
        return self._interrupted

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted.is_set()

    def start(self) -> None:
        if not INTERRUPT_POLL_ENABLED or self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(self._poll_interval)
                try:
                    status_data = await self._redis_client.get_exec_status(
                        self._session_id,
                    )
                    if status_data is None:
                        continue
                    if (
                        status_data.get("status") == "stopped"
                        and status_data.get("requestId") == self._request_id
                    ):
                        logger.info(
                            "Interrupt detected | session_id=%s request_id=%s",
                            self._session_id,
                            self._request_id,
                        )
                        self._interrupted.set()
                        return
                except Exception:
                    logger.exception(
                        "InterruptPoller Redis poll error | session_id=%s",
                        self._session_id,
                    )
        except asyncio.CancelledError:
            pass
