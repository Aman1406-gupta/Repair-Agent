import os
import logging
import json
import uuid
import redis.asyncio as redis
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from agent_builder.utils.handler_io_telemetry import _timed_redis
from agent_builder.utils.constants import (
    AGENT_COLLECTION,
    AGENT_METADATA_COLLECTION,
)
from agent_builder.storage.utils.state_serializer import (
    get_json_serializable_graph_state,
    serialize_session_data,
    deserialize_session_data,
)


class CacheSpec(NamedTuple):
    """Maps a collection to its Redis cache namespace and key field(s).

    ``key_field`` is either a single field name (str) or a tuple of field
    names whose values are joined with ``":"`` to form a composite cache key.
    """
    ns: str
    key_field: Union[str, Tuple[str, ...]]


def build_cache_key(spec: CacheSpec, source: Dict[str, Any]) -> Optional[str]:
    """Extract the cache key from *source* (a query filter or document).

    Returns ``None`` when any required field is missing.
    """
    if isinstance(spec.key_field, str):
        val = source.get(spec.key_field)
        return str(val) if val is not None else None
    parts = []
    for f in spec.key_field:
        val = source.get(f)
        if val is None:
            return None
        parts.append(str(val))
    return ":".join(parts)


_CACHE_NS: Dict[str, CacheSpec] = {
    AGENT_COLLECTION:          CacheSpec("agent_doc", ("agent_id", "version", "partner_id")),
    AGENT_METADATA_COLLECTION: CacheSpec("metadata",  "name"),
}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

_DEFAULT_REDIS_HOST = "qa6-redis-intuition.sprinklr.com"
class RedisClient(metaclass=Singleton):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        read_host = os.getenv("REDIS_READ_HOST") or os.getenv("REDIS_HOST") or _DEFAULT_REDIS_HOST
        write_host = os.getenv("REDIS_WRITE_HOST") or os.getenv("REDIS_HOST") or _DEFAULT_REDIS_HOST
        # Only send AUTH when REDIS_PASSWORD is set. Many managed / QA Redis instances have no password required causing AUTH errors there.
        redis_pass: Optional[str] = os.getenv("REDIS_PASSWORD") or None
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        def _make_client(host: str) -> redis.Redis:
            kwargs = {
                "host": host,
                "port": redis_port,
                "decode_responses": False,
            }
            if redis_pass is not None:
                kwargs["password"] = redis_pass
            return redis.Redis(**kwargs)

        if read_host == write_host:
            self._read_client = self._write_client = _make_client(read_host)
        else:
            self._read_client = _make_client(read_host)
            self._write_client = _make_client(write_host)
            self.logger.info(
                "Redis split read/write | read_host=%s write_host=%s port=%s",
                read_host,
                write_host,
                redis_port,
            )
        
        self.key_prefix = "agent_builder"

        self.session_ttl = int(os.getenv("REDIS_SESSION_TTL", 86400))
        self.metadata_ttl = int(os.getenv("REDIS_METADATA_TTL", 3600))
        #: TTL (seconds) for the transient background chunk buffer / meta. Defaults to
        #: the session TTL so a poller can drain for as long as the session lives.
        self.bg_chunk_ttl = int(os.getenv("BACKGROUND_CHUNK_TTL", self.session_ttl))

    @_timed_redis("set_session_state", "session")
    async def set_session_state(self, session_id: str, data: Dict[str, Any]) -> bool:
        try:
            basic_session_data = {
                "graph_state": data,
                "agent_id": None,
                "persistent_mock_behaviors": {}
            }
            serialized = serialize_session_data(basic_session_data)
            await self._write_client.set(f"{self.key_prefix}:{session_id}", serialized, ex=self.session_ttl)
            self.logger.debug("Session state set | session_id=%s ttl=%d", session_id, self.session_ttl)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to set session state for {session_id}: {e}")
            return False

    @_timed_redis("get_session_state", "session")
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get basic session state from Redis. Returns only the graph_state for backward compatibility."""
        try:
            extended_data = await self.get_extended_session_data(session_id)
            if extended_data is not None:
                return extended_data.get("graph_state")
            return None
        except Exception as e:
            self.logger.exception(f"Failed to get session state for {session_id}: {e}")
            return None

    async def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        session_id = str(uuid.uuid4())
        return session_id

    @_timed_redis("set_extended_session_data", "session")
    async def set_extended_session_data(
        self,
        session_id: str,
        serialized_data: str,
    ) -> bool:
        try:
            await self._write_client.set(f"{self.key_prefix}:{session_id}", serialized_data, ex=self.session_ttl)
            self.logger.debug("Extended session data set | session_id=%s", session_id)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to set extended session data for {session_id}: {e}")
            return False

    @_timed_redis("get_extended_session_data", "session")
    async def get_extended_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:

            data = await self._read_client.get(f"{self.key_prefix}:{session_id}")

            if data is None:
                self.logger.debug("Session not found | session_id=%s", session_id)
                return None

            try:
                session_data = deserialize_session_data(data.decode())
                self.logger.debug("Session restored | session_id=%s", session_id)
                return session_data
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self.logger.exception(f"Error deserializing extended session data for {session_id}: {e}")
                return None

        except Exception as e:
            self.logger.exception(f"Failed to get extended session data for {session_id}: {e}")
            return None

    @_timed_redis("update_graph_state", "session")
    async def update_graph_state(self, session_id: str, graph_state: Dict[str, Any]) -> bool:
        """Update only the graph_state for a session. Returns True on success, False on failure."""
        try:
            serialized_graph_state = json.dumps(get_json_serializable_graph_state(graph_state))
            await self._write_client.set(f"{self.key_prefix}:{session_id}:graph_state", serialized_graph_state, ex=self.session_ttl)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to update graph state for {session_id}: {e}")
            return False

    # ==========================================
    # Execution Status (per session)
    # ==========================================

    def _exec_status_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}:exec_status"

    @_timed_redis("set_exec_status", "exec_status")
    async def set_exec_status(
        self,
        session_id: str,
        status: str,
        request_id: str,
    ) -> bool:
        try:
            payload = json.dumps({"status": status, "requestId": request_id})
            await self._write_client.set(
                self._exec_status_key(session_id), payload, ex=self.session_ttl,
            )
            return True
        except Exception as e:
            self.logger.exception("Failed to set exec_status for %s: %s", session_id, e)
            return False

    @_timed_redis("get_exec_status", "exec_status")
    async def get_exec_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = await self._read_client.get(self._exec_status_key(session_id))
            if data is None:
                return None
            return json.loads(data.decode())
        except Exception as e:
            self.logger.exception("Failed to get exec_status for %s: %s", session_id, e)
            return None

    # ==========================================
    # Background-mode chunk buffer (per request / generation)
    # ==========================================
    #
    # Transient replay buffer for the ``/message`` poll endpoint. The canonical
    # message history still lives in the session envelope (see
    # ``set_extended_session_data``); this only stores the streamed typed frames
    # so a poller can resume the stream after ``mode.background``.
    #
    #   meta   : agent_builder:{sid}:bg:{rid}:meta            -> JSON {generationId,status,usage}
    #   chunks : agent_builder:{sid}:bg:{rid}:{gen}:chunks    -> ZSET score=sequence member=json(event)

    _BG_TERMINAL_TYPES = {"stream.completed", "stream.failed", "stream.interrupted"}

    def _bg_meta_key(self, session_id: str, request_id: str) -> str:
        return f"{self.key_prefix}:{session_id}:bg:{request_id}:meta"

    def _bg_chunks_key(self, session_id: str, request_id: str, generation_id: str) -> str:
        return f"{self.key_prefix}:{session_id}:bg:{request_id}:{generation_id}:chunks"

    async def set_bg_meta(
        self,
        session_id: str,
        request_id: str,
        generation_id: str,
        status: str,
        usage: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            payload = json.dumps({
                "generationId": generation_id,
                "status": status,
                "usage": usage or {},
            })
            await self._write_client.set(
                self._bg_meta_key(session_id, request_id), payload, ex=self.bg_chunk_ttl,
            )
            return True
        except Exception as e:
            self.logger.exception("Failed to set bg meta for %s/%s: %s", session_id, request_id, e)
            return False

    async def get_bg_meta(self, session_id: str, request_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = await self._read_client.get(self._bg_meta_key(session_id, request_id))
            if data is None:
                return None
            return json.loads(data.decode())
        except Exception as e:
            self.logger.exception("Failed to get bg meta for %s/%s: %s", session_id, request_id, e)
            return None

    async def append_bg_chunks(
        self,
        session_id: str,
        request_id: str,
        generation_id: str,
        events: List[Dict[str, Any]],
    ) -> int:
        """Persist typed events to the generation's sorted set (score = ``sequence``).

        Idempotent per sequence: any existing member at that score is removed first so
        a re-ingested seq replaces rather than duplicates. Updates meta (current
        generation + status/usage) and refreshes TTL. Returns the number persisted.
        """
        if not events:
            return 0
        key = self._bg_chunks_key(session_id, request_id, generation_id)
        try:
            terminal_status: Optional[str] = None
            terminal_usage: Optional[Dict[str, Any]] = None
            count = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                seq = event.get("sequence")
                if not isinstance(seq, int) or isinstance(seq, bool):
                    self.logger.warning("Skipping bg chunk without int sequence | rid=%s", request_id)
                    continue
                await self._write_client.zremrangebyscore(key, seq, seq)
                await self._write_client.zadd(key, {json.dumps(event): seq})
                count += 1
                etype = event.get("type")
                if etype == "stream.completed":
                    terminal_status, terminal_usage = "completed", event.get("usage") or {}
                elif etype == "stream.failed":
                    terminal_status = "failed"
                elif etype == "stream.interrupted":
                    terminal_status, terminal_usage = "interrupted", event.get("usage") or {}
            await self._write_client.expire(key, self.bg_chunk_ttl)

            status = terminal_status or "running"
            await self.set_bg_meta(session_id, request_id, generation_id, status, terminal_usage)
            return count
        except Exception as e:
            self.logger.exception("Failed to append bg chunks for %s/%s: %s", session_id, request_id, e)
            return 0

    async def get_bg_chunks(
        self,
        session_id: str,
        request_id: str,
        generation_id: str,
        after_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return events with ``sequence >= after_seq`` (or all when ``after_seq`` is None)."""
        key = self._bg_chunks_key(session_id, request_id, generation_id)
        try:
            lo = "-inf" if after_seq is None else f"{after_seq}"
            raw = await self._read_client.zrangebyscore(key, lo, "+inf")
            out: List[Dict[str, Any]] = []
            for item in raw:
                try:
                    out.append(json.loads(item.decode() if isinstance(item, bytes) else item))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.logger.warning("Skipping undecodable bg chunk | rid=%s", request_id)
            return out
        except Exception as e:
            self.logger.exception("Failed to get bg chunks for %s/%s: %s", session_id, request_id, e)
            return []

    # ==========================================
    # Last Active Task (per session)
    # ==========================================

    def _get_last_active_task_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}:last_active_task"

    @_timed_redis("set_last_active_task", "session")
    async def set_last_active_task(self, session_id: str, last_active_task: Dict[str, Any]) -> bool:
        try:
            cache_key = self._get_last_active_task_key(session_id)
            serialized = json.dumps(last_active_task)
            await self._write_client.set(cache_key, serialized, ex=self.session_ttl)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to set last_active_task for {session_id}: {e}")
            return False

    @_timed_redis("get_last_active_task", "session")
    async def get_last_active_task(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            cache_key = self._get_last_active_task_key(session_id)
            data = await self._read_client.get(cache_key)
            if data is None:
                return None
            return json.loads(data.decode())
        except Exception as e:
            self.logger.exception(f"Failed to get last_active_task for {session_id}: {e}")
            return None

    # ==========================================
    # Generic Document Cache
    # ==========================================

    def _cache_key(self, ns: str, key: str) -> str:
        return f"{self.key_prefix}:{ns}:{key}"

    @_timed_redis("cache_get")
    async def cache_get(self, ns: str, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached document by namespace and key."""
        try:
            data = await self._read_client.get(self._cache_key(ns, key))
            return None if data is None else json.loads(data.decode())
        except Exception as e:
            self.logger.exception("cache_get failed ns=%s key=%s: %s", ns, key, e)
            return None

    @_timed_redis("cache_mget")
    async def cache_mget(self, ns: str, keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch-fetch cached documents (MGET). Missing keys are omitted."""
        if not keys:
            return {}
        try:
            redis_keys = [self._cache_key(ns, k) for k in keys]
            values = await self._read_client.mget(redis_keys)
            out: Dict[str, Dict[str, Any]] = {}
            for k, raw in zip(keys, values):
                if raw is not None:
                    out[k] = json.loads(raw.decode())
            return out
        except Exception as e:
            self.logger.exception("cache_mget failed ns=%s: %s", ns, e)
            return {}

    @_timed_redis("cache_set")
    async def cache_set(self, ns: str, key: str, doc: Dict[str, Any], ttl: int = None) -> bool:
        """Store a document in cache."""
        try:
            await self._write_client.set(self._cache_key(ns, key), json.dumps(doc), ex=ttl or self.metadata_ttl)
            return True
        except Exception as e:
            self.logger.exception("cache_set failed ns=%s key=%s: %s", ns, key, e)
            return False

    @_timed_redis("cache_del")
    async def cache_del(self, ns: str, key: str) -> bool:
        """Delete a document from cache."""
        try:
            await self._write_client.delete(self._cache_key(ns, key))
            return True
        except Exception as e:
            self.logger.exception("cache_del failed ns=%s key=%s: %s", ns, key, e)
            return False

