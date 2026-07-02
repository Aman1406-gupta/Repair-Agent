"""
Elasticsearch Callback Handler for Agent Builder Tracing.

Logs agent execution traces to Elasticsearch for visualization and analysis.
Each span (agent, task, tool, LLM call) is stored as a document.

"""

# Version marker
ES_HANDLER_VERSION = "v3.3"  # http_ttft_ms: HTTP prepare() → first token per LLM call

import asyncio
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from uuid import UUID

from elasticsearch.helpers import async_bulk
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from agent_builder.utils.constants import ENVELOPE_KEY, ES_LLM_CALLS_INDEX, ES_TRACES_INDEX
from agent_builder.utils.telemetry import (
    tlog_debug,
    tlog_error,
    tlog_info,
    tlog_warning,
)


class ElasticsearchCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that logs traces to Elasticsearch.
    
    Tracks LLM calls, tool executions, and chain/task execution with timing,
    token usage, costs, and hierarchical parent-child relationships.
    """
    
    INDEX_MAPPING = {
        "mappings": {
            "properties": {
                # Trace visualization
                "session_id": {"type": "keyword"},
                "request_id": {"type": "keyword"},
                "span_id": {"type": "keyword"},
                "parent_span_id": {"type": "keyword"},
                "component_type": {"type": "keyword"},
                "component_name": {"type": "keyword"},
                
                # MongoDB IDs
                "agent_id": {"type": "keyword"},
                "task_id": {"type": "keyword"},
                "tool_id": {"type": "keyword"},
                
                # Time telemetry
                "start_time": {"type": "date"},
                "end_time": {"type": "date"},
                "duration_ms": {"type": "long"},
                "ttft_ms": {"type": "long"},
                "http_ttft_ms": {"type": "long"},
                "node_name": {"type": "keyword"},
                
                # LLM cost (only for LLM spans)
                "input_tokens": {"type": "integer"},
                "output_tokens": {"type": "integer"},
                "cost": {"type": "float"},
                "model": {"type": "keyword"},
                
                # LLM output (only for LLM spans)
                "llm_output": {"type": "text"},
                "llm_tool_calls": {"type": "text"},
                
                # Tool I/O (only for tool spans)
                "tool_input": {"type": "text"},
                "tool_output": {"type": "text"},
                
                # Error tracking
                "status_code": {"type": "integer"},
                "error_type": {"type": "keyword"},
                "error_message": {"type": "text"},
                
                # Hierarchy
                "depth": {"type": "integer"},
                
                # Accumulated tokens (for agents, tasks, tools - sum of child LLM calls)
                "total_input_tokens": {"type": "integer"},
                "total_output_tokens": {"type": "integer"},
                "total_cost": {"type": "float"},
                "llm_calls": {"type": "integer"}
            }
        }
    }
    
    # LLM Calls Flow Index Mapping 
    LLM_CALLS_INDEX_MAPPING = {
        "mappings": {
            "properties": {
                "session_id": {"type": "keyword"},
                "request_id": {"type": "keyword"},
                "partner_id": {"type": "keyword"},
                "agent_id": {"type": "keyword"},
                "agent_name": {"type": "keyword"},
                "query": {"type": "text"},
                "response": {"type": "text"},
                "created_time": {"type": "date"},
                "model": {"type": "keyword"},
                "total_input_tokens": {"type": "integer"},
                "total_output_tokens": {"type": "integer"},
                "total_cost": {"type": "float"},
                "total_duration_ms": {"type": "long"},
                "llm_call_count": {"type": "integer"},
                "message_flow": {"type": "object", "enabled": False},
                "status_code": {"type": "integer"},
                "error_type": {"type": "keyword"},
                "error_message": {"type": "text"}
            }
        }
    }
    
    # ========== Background Worker (Class-level, shared across all handler instances) ==========
    _queue: queue.Queue = None
    _worker_thread: threading.Thread = None
    _worker_started: bool = False
    _shutdown_event: threading.Event = threading.Event()

    @classmethod
    def client_config_from(cls, es_client) -> Dict[str, Any]:
        """Build kwargs for a new AsyncElasticsearch client (same cluster/auth as ``es_client``)."""
        transport = es_client.transport
        return {"hosts": transport.hosts, **transport.kwargs}

    @classmethod
    def start_background_worker(
        cls,
        es_client_config: Dict[str, Any],
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_queue_size: int = 10000,
    ):
        """Start the background ES writer thread (call once at app startup).

        The writer runs a dedicated asyncio event loop in a daemon thread and uses
        AsyncElasticsearch + async_bulk — isolated from the Tornado IOLoop.
        """
        if cls._worker_started:
            tlog_debug("ES writer already running")
            return

        cls._batch_size = batch_size
        cls._flush_interval = flush_interval
        cls._queue = queue.Queue(maxsize=max_queue_size)
        cls._shutdown_event.clear()

        cls._worker_thread = threading.Thread(
            target=cls._writer_thread_entry,
            args=(es_client_config,),
            daemon=True,
            name="es-telemetry-writer",
        )
        cls._worker_thread.start()
        cls._worker_started = True
        tlog_debug(
            "ES writer started (batch_size=%s, flush_interval=%ss)",
            batch_size,
            flush_interval,
        )

    @classmethod
    def stop_background_worker(cls, timeout: float = 5.0):
        """Stop the background worker gracefully (call at app shutdown)."""
        if not cls._worker_started:
            return
        tlog_info(
            "Stopping ES telemetry background worker (queue size: %s)...",
            cls._queue.qsize() if cls._queue else 0,
        )
        cls._shutdown_event.set()
        if cls._worker_thread:
            cls._worker_thread.join(timeout=timeout)
        cls._worker_started = False
        tlog_info("ES telemetry background worker stopped")

    @classmethod
    def _writer_thread_entry(cls, es_client_config: Dict[str, Any]) -> None:
        """Sync thread entry: create an event loop and run the async ES writer on it."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cls._async_es_writer_loop(es_client_config))
        finally:
            loop.close()

    @classmethod
    async def _async_es_writer_loop(cls, es_client_config: Dict[str, Any]) -> None:
        """Async loop: drain the queue and bulk-write to ES until shutdown."""
        from elasticsearch import AsyncElasticsearch

        es_client = AsyncElasticsearch(**es_client_config)
        try:
            while not cls._shutdown_event.is_set():
                batch = cls._drain_batch()
                if batch:
                    await cls._bulk_write_batch(es_client, batch)
                else:
                    await asyncio.to_thread(
                        cls._shutdown_event.wait, cls._flush_interval,
                    )

            batch = cls._drain_batch()
            if batch:
                # Final flush on shutdown
                count = await cls._bulk_write_batch(es_client, batch)
                tlog_debug("ES writer final flush: %s docs", count)
        finally:
            await es_client.close()

    @classmethod
    async def _bulk_write_batch(cls, es_client, batch: List[tuple]) -> int:
        actions = [{"_index": idx, "_source": doc} for idx, doc in batch]
        try:
            success, failed = await async_bulk(es_client, actions, raise_on_error=False)
            if failed:
                tlog_warning(
                    "ES bulk write partial failure: %s ok, %s failed",
                    success,
                    len(failed),
                )
            else:
                tlog_debug("ES bulk write: %s docs", success)
            return success
        except Exception as e:
            tlog_error("ES bulk write error: %s", e)
            return 0

    @classmethod
    def _drain_batch(cls):
        batch = []
        while len(batch) < cls._batch_size:
            try:
                batch.append(cls._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    @classmethod
    def get_queue_size(cls) -> int:
        """Get current queue size (for monitoring)."""
        return cls._queue.qsize() if cls._queue else 0

    @classmethod
    def enqueue_telemetry(cls, index: str, doc: dict) -> None:
        if cls._queue is None or not cls._worker_started:
            return
        try:
            cls._queue.put_nowait((index, doc))
        except queue.Full:
            tlog_warning("ES telemetry queue full, dropping doc for index=%s", index)
    
    def __init__(
        self,
        es_client=None,
        es_url: str = None,
        traces_index: str = ES_TRACES_INDEX,
        llm_calls_index: str = ES_LLM_CALLS_INDEX,
    ):
        """Initialize the ES callback handler.
        
        Args:
            es_client: Existing Elasticsearch client (optional)
            es_url: Elasticsearch URL (used if es_client not provided). 
                    Falls back to ELASTICSEARCH_URL env var.
            traces_index: Name of the ES index for agent traces
            llm_calls_index: Name of the ES index for LLM call flow
        """
        super().__init__()
        
        self.es_url = es_url or os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
        self.traces_index = traces_index
        self.llm_calls_index = llm_calls_index
        
        self._es_client = es_client
        self._es_initialized = False
        
        # Trace state tracking
        self._session_id: Optional[str] = None
        self._request_id: Optional[str] = None  # New request_id for each invoke
        self._start_times: Dict[UUID, float] = {}
        self._start_timestamps: Dict[UUID, str] = {}
        self._run_names: Dict[UUID, str] = {}
        self._run_types: Dict[UUID, str] = {}
        self._depth_stack: Dict[UUID, int] = {}
        self._parent_map: Dict[UUID, UUID] = {}
        self._tool_inputs: Dict[UUID, str] = {}  # Store tool inputs
        self._span_tokens: Dict[UUID, Dict] = {}  # Track tokens per span (accumulated from child LLMs)
        
        # MongoDB ID mappings (for linking spans to registered entities)
        self._agent_id: Optional[str] = None
        self._task_id_map: Dict[str, str] = {}  # task_name -> task_id
        self._tool_id_map: Dict[str, str] = {}  # tool_name -> tool_id
        
        # LLM Call Flow tracking 
        self._partner_id: Optional[str] = None
        self._user_query: Optional[str] = None
        self._final_response: Optional[str] = None
        self._llm_message_flow: List[Dict] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self._llm_call_count: int = 0
        self._conversation_start_time: Optional[float] = None
        self._root_agent_run_id: Optional[UUID] = None
        self._first_token_times: Dict[UUID, float] = {}
        self._http_ttft_ms: Dict[UUID, int] = {}
        
        # Buffer ALL documents for flush at request end (avoids async context issues)
        self._pending_docs: List[tuple] = []  # List of (index_name, doc) tuples
    
    @property
    def es_client(self):
        """Lazy initialization of ES client."""
        if self._es_client is None:
            try:
                from elasticsearch import AsyncElasticsearch
                self._es_client = AsyncElasticsearch(self.es_url)
                self._es_initialized = True
            except ImportError:
                raise ImportError(
                    "elasticsearch package not installed. "
                    "Install with: pip install elasticsearch"
                )
        return self._es_client
    
    def _get_depth(self, run_id: UUID, parent_run_id: Optional[UUID] = None) -> int:
        """Calculate the depth of this run in the execution tree."""
        if parent_run_id is None:
            return 0
        parent_depth = self._depth_stack.get(parent_run_id, 0)
        return parent_depth + 1
    
    def _get_meaningful_depth(self, parent_run_id: Optional[UUID]) -> int:
        """Calculate depth based on meaningful parent chain (skipping internal nodes).
        
        Note: default_router_task is NOT skipped here to maintain accurate depth hierarchy.
        Items inside default_router_task will have higher depth than sibling tasks.
        """
        if parent_run_id is None:
            return 0
        
        # Note: default_router_task is NOT in skip_names to preserve depth hierarchy
        skip_names = {"ENTRY", "tools", "chatbot", "unknown"}
        depth = 0
        current_id = parent_run_id
        
        while current_id:
            run_name = self._run_names.get(current_id, "")
            run_type = self._run_types.get(current_id, "")
            
            # Count if it's a meaningful node (will be written to ES)
            if run_type == "tool":
                depth += 1
            elif run_name and run_name not in skip_names:
                # Check it's not a wrapper duplicate
                parent_of_current = self._parent_map.get(current_id)
                parent_name = self._run_names.get(parent_of_current, "") if parent_of_current else ""
                if run_name != parent_name:
                    depth += 1
            
            current_id = self._parent_map.get(current_id)
        
        return depth
    
    def _get_parent_span_id(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        """Find nearest ancestor that will be written to ES.
        
        Traverses up the parent chain, skipping:
        - Internal LangGraph nodes (chatbot, tools, ENTRY, etc.)
        - Wrapper duplicates (nodes with same name as their parent)
        """
        if parent_run_id is None:
            return None
        
        skip_names = {"ENTRY", "tools", "chatbot", "unknown"}
        current_id = parent_run_id
        
        while current_id:
            run_name = self._run_names.get(current_id, "")
            run_type = self._run_types.get(current_id, "")
            
            if run_type == "tool":
                return str(current_id)
            
            # Chains: check if not internal AND not a wrapper duplicate
            if run_name and run_name not in skip_names:
                parent_of_current = self._parent_map.get(current_id)
                parent_name = self._run_names.get(parent_of_current, "") if parent_of_current else ""
                if run_name != parent_name:
                    return str(current_id)
            
            current_id = self._parent_map.get(current_id)
        
        return None
    
    def _get_meaningful_parent_info(self, parent_run_id: Optional[UUID]) -> tuple:
        """Get meaningful parent name and type, skipping internal nodes.
        
        Returns: (parent_name, parent_type) tuple
        """
        if parent_run_id is None:
            return ("root", "agent")
        
        skip_names = {"ENTRY", "tools", "chatbot", "unknown"}
        current_id = parent_run_id
        
        while current_id:
            run_name = self._run_names.get(current_id, "")
            run_type = self._run_types.get(current_id, "")
            
            # Clean the name
            clean_name = run_name.replace("LLM:", "").replace("Tool:", "")
            
            # Tools are meaningful
            if run_type == "tool":
                return (clean_name, "tool")
            
            # Check if this is a meaningful chain/task
            if clean_name and clean_name not in skip_names:
                parent_of_current = self._parent_map.get(current_id)
                parent_name = self._run_names.get(parent_of_current, "") if parent_of_current else ""
                if clean_name != parent_name:
                    depth = self._depth_stack.get(current_id, 1)
                    comp_type = "agent" if depth == 0 else "task"
                    return (clean_name, comp_type)
            
            current_id = self._parent_map.get(current_id)
        
        return ("root", "agent")

    def _node_name_for_llm(self, parent_run_id: Optional[UUID]) -> str:
        """Immediate node (tool or task) where the LLM ran."""
        parent_name, _ = self._get_meaningful_parent_info(parent_run_id)
        return parent_name if parent_name and parent_name != "root" else "unknown"
    
    def _accumulate_tokens_to_ancestors(self, run_id: UUID, input_tokens: int, output_tokens: int, cost: float):
        """Accumulate LLM tokens to all ancestor spans (agents, tasks, tools)."""
        current_id = self._parent_map.get(run_id)
        
        while current_id:
            # Initialize token tracking for this span if not exists
            if current_id not in self._span_tokens:
                self._span_tokens[current_id] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "llm_calls": 0}
            
            # Accumulate tokens
            self._span_tokens[current_id]["input_tokens"] += input_tokens
            self._span_tokens[current_id]["output_tokens"] += output_tokens
            self._span_tokens[current_id]["cost"] += cost
            self._span_tokens[current_id]["llm_calls"] += 1
            
            # Move to parent
            current_id = self._parent_map.get(current_id)
    
    def _get_span_tokens(self, run_id: UUID) -> Dict:
        """Get accumulated tokens for a span."""
        return self._span_tokens.get(run_id, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "llm_calls": 0})
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format for ES."""
        return datetime.now(timezone.utc).isoformat()
    
    def _to_iso_timestamp(self, epoch: float) -> str:
        """Convert epoch time to ISO format string for ES."""
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    @staticmethod
    def _model_name_from_serialized(serialized: Optional[Dict[str, Any]]) -> Optional[str]:
        if not serialized:
            return None
        kwargs_dict = serialized.get("kwargs") or {}
        llm_config = kwargs_dict.get("llm_config")
        if isinstance(llm_config, dict):
            return llm_config.get("model") or llm_config.get("llm_configuration_id")
        if llm_config is not None:
            return getattr(llm_config, "model", None) or getattr(
                llm_config, "llm_configuration_id", None,
            )
        return (
            kwargs_dict.get("model_name")
            or kwargs_dict.get("model")
            or serialized.get("model_name")
            or serialized.get("model")
        )

    @staticmethod
    def _token_usage_from_llm_result(
        response: LLMResult,
    ) -> tuple[int, int, float, Optional[str]]:
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage") or {}
        input_tokens = int(token_usage.get("prompt_tokens") or 0)
        output_tokens = int(token_usage.get("completion_tokens") or 0)
        spending = float(llm_output.get("spending") or 0)
        model_hint = llm_output.get("model_name")

        if input_tokens or output_tokens:
            return input_tokens, output_tokens, spending, model_hint

        if not response.generations:
            return 0, 0, spending, model_hint

        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                envelope = (getattr(msg, "additional_kwargs", None) or {}).get(ENVELOPE_KEY) or {}
                usage = envelope.get("usage") or {}
                breakdown = usage.get("modelBreakdown") or []
                if breakdown:
                    input_tokens = sum(int(row.get("inputTokens") or 0) for row in breakdown)
                    output_tokens = sum(int(row.get("outputTokens") or 0) for row in breakdown)
                    if not model_hint:
                        model_hint = breakdown[0].get("modelId")
                if usage.get("totalCost") is not None and not spending:
                    spending = float(usage.get("totalCost") or 0)
                if input_tokens or output_tokens:
                    return input_tokens, output_tokens, spending, model_hint

                resp_meta = getattr(msg, "response_metadata", None) or {}
                stream_usage = resp_meta.get("stream_usage")
                if isinstance(stream_usage, dict):
                    input_tokens = int(stream_usage.get("prompt_tokens") or 0)
                    output_tokens = int(stream_usage.get("completion_tokens") or 0)
                    if not model_hint:
                        model_hint = resp_meta.get("model_name")
                    if resp_meta.get("stream_spending") is not None and not spending:
                        spending = float(resp_meta.get("stream_spending") or 0)
                    if input_tokens or output_tokens:
                        return input_tokens, output_tokens, spending, model_hint

        return 0, 0, spending, model_hint
    
    def _create_span_doc(
        self,
        span_id: UUID,
        parent_span_id: Optional[str],
        component_type: str,
        component_name: str,
        start_time: str,
        end_time: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """Create a span document for ES."""
        doc = {
            "session_id": self._session_id or "unknown",
            "request_id": self._request_id or "unknown",
            "span_id": str(span_id),
            "parent_span_id": parent_span_id,
            "component_type": component_type,
            "component_name": component_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
        }
        
        # Add extra fields (LLM data, tool data, etc.)
        for key, value in extra_fields.items():
            if value is not None:
                doc[key] = value
        
        return doc
    
    def _buffer_doc(self, doc: Dict[str, Any], traces_index: str = None):
        """Fire-and-forget: queue document for background worker. Returns immediately."""
        target_index = traces_index or self.traces_index
        if type(self)._worker_started:
            # Use background worker queue if available
            type(self).enqueue_telemetry(target_index, doc)
        else:
            # Fallback to local buffer if worker not started
            self._pending_docs.append((target_index, doc))
    
    async def _write_to_es(self, doc: Dict[str, Any]):
        """Fire-and-forget: queue document for background write."""
        self._buffer_doc(doc, self.traces_index)
    
    async def _flush_all_docs(self):
        """No-op when using background worker (fire-and-forget).
        
        Falls back to sync bulk write if worker not started (e.g., local testing).
        """
        # If using background worker, nothing to do - docs are already queued
        if type(self)._worker_started:
            tlog_debug(
                "ES fire-and-forget mode, queue size=%s",
                type(self).get_queue_size(),
            )
            return
        
        # Fallback: flush local buffer if worker not started
        if not self._pending_docs:
            return
        
        doc_count = len(self._pending_docs)
        tlog_debug("ES flushing %s buffered docs (fallback mode)", doc_count)
        
        try:
            from elasticsearch.helpers import async_bulk
            
            actions = [
                {"_index": traces_index, "_source": doc}
                for traces_index, doc in self._pending_docs
            ]
            
            success, failed = await async_bulk(self.es_client, actions, raise_on_error=False)
            
            if failed:
                tlog_warning(
                    "ES bulk flush partial failure: %s ok, %s failed",
                    success,
                    len(failed),
                )
            else:
                tlog_debug("ES bulk flush: %s docs", success)

        except Exception as e:
            tlog_warning("ES bulk flush failed (%s), trying individual writes", e)
            for traces_index, doc in self._pending_docs:
                try:
                    result = self.es_client.index(index=traces_index, document=doc)
                    if hasattr(result, '__await__'):
                        await result
                except Exception as write_error:
                    tlog_error("ES individual write error: %s", write_error)
        
        self._pending_docs = []
    
    async def _write_llm_call_flow(
        self, 
        agent_name: str, 
        duration_ms: int,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        status_code: int = 200
    ):
        """Write LLM call flow document to llm-calls index."""
        try:
            # Get model from first LLM call (if available)
            model = "unknown"
            if self._llm_message_flow:
                model = self._llm_message_flow[0].get("model", "unknown")
            
            llm_call_doc = {
                "session_id": self._session_id,
                "request_id": self._request_id,
                "partner_id": self._partner_id,
                "agent_id": self._agent_id,
                "agent_name": agent_name,
                "query": self._user_query,
                "response": self._final_response if not error_type else None,
                "created_time": self._get_current_timestamp(),
                "model": model,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_cost": self._total_cost,
                "total_duration_ms": duration_ms,
                "llm_call_count": self._llm_call_count,
                "message_flow": self._llm_message_flow,
                "status_code": status_code,
                "error_type": error_type,
                "error_message": error_message
            }
            
            tlog_debug(
                "ES buffering LLM call flow session=%s messages=%s",
                (self._session_id[:8] if self._session_id else "NONE"),
                len(self._llm_message_flow),
            )
            if error_type:
                tlog_debug("ES LLM call flow error: %s status=%s", error_type, status_code)

            # Buffer for flush at end (along with all other docs)
            self._buffer_doc(llm_call_doc, self.llm_calls_index)
        except Exception as e:
            tlog_error("ES error buffering LLM call flow: %s", e)
    
    # ========== Chain/Task Callbacks ==========
    
    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Track chain/task start."""
        if metadata:
            # Extract session_id from metadata (passed via RunnableConfig)
            self._session_id = (
                metadata.get("session_id")
                or self._session_id
            )
        
        now = time.time()
        self._start_times[run_id] = now
        self._start_timestamps[run_id] = self._to_iso_timestamp(now)
        
        if parent_run_id:
            self._parent_map[run_id] = parent_run_id
        
        # Extract chain name 
        chain_name = (
            kwargs.get("langgraph_node") or
            (metadata.get("name") if metadata else None) or
            (metadata.get("langgraph_node") if metadata else None) or
            (serialized.get("name") if serialized else None) or
            (serialized.get("id") if serialized else None) or
            "unknown"
        )
        
        # Calculate depth
        depth = self._get_depth(run_id, parent_run_id)
        self._depth_stack[run_id] = depth
        
        # LLM Call Flow: Capture context at root agent start (depth 0)
        if depth == 0:
            if metadata and metadata.get("agent_name"):
                chain_name = metadata["agent_name"]
            elif chain_name == "unknown":
                chain_name = "root_agent"  # Fallback
            self._root_agent_run_id = run_id
            self._conversation_start_time = now
            
            req_id = metadata.get("request_id") if metadata else None
            # Use invoke request_id; generate one if missing
            self._request_id = str(req_id) if req_id else str(uuid.uuid4())
            
            # Extract MongoDB ID mappings from metadata (passed via RunnableConfig)
            if metadata:
                self._agent_id = metadata.get("agent_id")
                self._task_id_map = metadata.get("task_id_map", {})
                self._tool_id_map = metadata.get("tool_id_map", {})
            
            # Reset LLM tracking for new conversation/request
            self._llm_message_flow = []
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost = 0.0
            self._llm_call_count = 0
            self._user_query = None
            self._final_response = None
            
            # Extract user query from inputs - get the LAST user message (for multi-turn)
            if inputs:
                messages = inputs.get("messages", [])
                if messages:
                    # Iterate in reverse to get the LAST user message
                    for msg in reversed(messages):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            self._user_query = msg.get("content", "")
                            break
                        elif hasattr(msg, "type") and msg.type == "human":
                            self._user_query = msg.content if hasattr(msg, "content") else str(msg)
                            break
                        elif hasattr(msg, "content") and "HumanMessage" in str(type(msg)):
                            self._user_query = msg.content
                            break
        
        tlog_debug("ES on_chain_start: %s depth=%s", chain_name, depth)
        
        self._run_names[run_id] = chain_name
        self._run_types[run_id] = "chain"
    
    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        """Track chain/task completion."""
        duration_ms = int((time.time() - self._start_times.pop(run_id, time.time())) * 1000)
        start_time = self._start_timestamps.pop(run_id, None)
        end_time = self._get_current_timestamp()
        
        chain_name = self._run_names.get(run_id, "unknown")
        depth = self._depth_stack.get(run_id, 0)
        parent_name = self._run_names.get(parent_run_id, "") if parent_run_id else ""
        
        skip_names = {"ENTRY", "tools", "chatbot"}
        should_skip = (
            chain_name in skip_names or
            chain_name == parent_name or
            (chain_name == "unknown" and depth > 0)
        )
        
        if should_skip:
            tlog_debug("ES on_chain_end skip: %s", chain_name)
            self._cleanup_run(run_id)
            return
        
        component_type = "agent" if depth == 0 else "task"
        # Use meaningful depth for tree visualization
        meaningful_depth = self._get_meaningful_depth(parent_run_id)
        
        tlog_debug(
            "ES on_chain_end write: %s (%s) depth=%s",
            chain_name,
            component_type,
            meaningful_depth,
        )
        
        # Get accumulated tokens from child LLM calls
        span_tokens = self._get_span_tokens(run_id)
        
        # Lookup MongoDB IDs based on component type
        entity_id_field = None
        entity_id_value = None
        if component_type == "agent":
            entity_id_field = "agent_id"
            entity_id_value = self._agent_id
        elif component_type == "task":
            entity_id_field = "task_id"
            entity_id_value = self._task_id_map.get(chain_name)
        
        doc = self._create_span_doc(
            span_id=run_id,
            parent_span_id=self._get_parent_span_id(parent_run_id),
            component_type=component_type,
            component_name=chain_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status_code=200,
            depth=meaningful_depth,
            total_input_tokens=span_tokens["input_tokens"],
            total_output_tokens=span_tokens["output_tokens"],
            total_cost=span_tokens["cost"],
            llm_calls=span_tokens["llm_calls"],
            agent_id=self._agent_id if component_type == "agent" else None,
            task_id=entity_id_value if component_type == "task" else None
        )
        
        await self._write_to_es(doc)
        
        # When root agent ends, buffer LLM call flow then flush ALL docs
        if depth == 0:
            # Buffer LLM call flow document (will be flushed with everything else)
            if self._llm_call_count > 0:
                await self._write_llm_call_flow(
                    agent_name=chain_name,
                    duration_ms=duration_ms
                )
            
            # Flush ALL buffered documents at once (bulk write)
            await self._flush_all_docs()
        
        self._cleanup_run(run_id)
    
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        """Track chain/task errors."""
        # Skip ParentCommand (task transfer)
        error_type = type(error).__name__
        if error_type == "ParentCommand" or "ParentCommand" in str(error):
            # Task transfer — record span like on_chain_end (not a real error)
            duration_ms = int((time.time() - self._start_times.pop(run_id, time.time())) * 1000)
            start_time = self._start_timestamps.pop(run_id, None)
            chain_name = self._run_names.get(run_id, "unknown")
            depth = self._depth_stack.get(run_id, 0)
            parent_name = self._run_names.get(parent_run_id, "") if parent_run_id else ""
            skip_names = {"ENTRY", "tools", "chatbot"}
            should_skip = (
                chain_name in skip_names or
                chain_name == parent_name or
                (chain_name == "unknown" and depth > 0)
            )
            if not should_skip:
                component_type = "agent" if depth == 0 else "task"
                meaningful_depth = self._get_meaningful_depth(parent_run_id)
                span_tokens = self._get_span_tokens(run_id)
                entity_id_value = (
                    self._agent_id if component_type == "agent"
                    else self._task_id_map.get(chain_name)
                )
                doc = self._create_span_doc(
                    span_id=run_id,
                    parent_span_id=self._get_parent_span_id(parent_run_id),
                    component_type=component_type,
                    component_name=chain_name,
                    start_time=start_time,
                    end_time=self._get_current_timestamp(),
                    duration_ms=duration_ms,
                    status_code=200,
                    depth=meaningful_depth,
                    total_input_tokens=span_tokens["input_tokens"],
                    total_output_tokens=span_tokens["output_tokens"],
                    total_cost=span_tokens["cost"],
                    llm_calls=span_tokens["llm_calls"],
                    agent_id=self._agent_id if component_type == "agent" else None,
                    task_id=entity_id_value if component_type == "task" else None,
                )
                await self._write_to_es(doc)
            self._cleanup_run(run_id)
            return
        
        # Log error span
        duration_ms = int((time.time() - self._start_times.pop(run_id, time.time())) * 1000)
        start_time = self._start_timestamps.pop(run_id, None)
        
        chain_name = self._run_names.get(run_id, "unknown")
        depth = self._depth_stack.get(run_id, 0)
        # Determine component type based on depth (same logic as on_chain_end)
        component_type = "agent" if depth == 0 else "task"
        meaningful_depth = self._get_meaningful_depth(parent_run_id)
        parent_span_id = self._get_parent_span_id(parent_run_id)
        parent_name = self._run_names.get(parent_run_id, "") if parent_run_id else ""
        
        # Skip internal LangGraph nodes (same logic as on_chain_end)
        skip_names = {"ENTRY", "tools", "chatbot"}
        should_skip = (
            chain_name in skip_names or
            chain_name == parent_name or
            (chain_name == "unknown" and meaningful_depth > 0)
        )
        
        if should_skip:
            self._cleanup_run(run_id)
            return
        
        # Extract status code - first try to get from error's code attribute
        error_type = type(error).__name__
        status_code = 500  # Default server error
        if hasattr(error, 'code') and error.code:
            try:
                status_code = int(error.code)
            except (ValueError, TypeError):
                pass
        
        # If no code attribute, try to parse from error message
        if status_code == 500:
            error_str = str(error).lower()
            if "validation" in error_str or "invalid" in error_str:
                status_code = 400  # Bad Request
            elif "timeout" in error_str or "timed out" in error_str:
                status_code = 408  # Request Timeout
            elif "forbidden" in error_str or "403" in error_str:
                status_code = 403  # Forbidden
            elif "unauthorized" in error_str or "401" in error_str:
                status_code = 401  # Unauthorized
            elif "not found" in error_str or "404" in error_str:
                status_code = 404  # Not Found
        
        # Lookup MongoDB IDs based on component type
        entity_id_value = None
        if component_type == "agent":
            entity_id_value = self._agent_id
        elif component_type == "task":
            entity_id_value = self._task_id_map.get(chain_name)
        
        doc = self._create_span_doc(
            span_id=run_id,
            parent_span_id=parent_span_id,
            component_type=component_type,
            component_name=chain_name,
            start_time=start_time,
            end_time=self._get_current_timestamp(),
            duration_ms=duration_ms,
            status_code=status_code,
            error_type=error_type,
            error_message=str(error)[:1000],
            depth=meaningful_depth,
            agent_id=self._agent_id if component_type == "agent" else None,
            task_id=entity_id_value if component_type == "task" else None
        )
        
        await self._write_to_es(doc)
        
        # When root agent errors, buffer LLM call flow then flush ALL docs
        if depth == 0:
            # Buffer error record to llm-calls index
            await self._write_llm_call_flow(
                agent_name=chain_name,
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=str(error)[:500],
                status_code=status_code
            )
            
            # Flush ALL buffered documents at once (bulk write)
            await self._flush_all_docs()
        
        self._cleanup_run(run_id)
    
    # ========== LLM Callbacks ==========
    
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Track LLM call start."""

        now = time.time()
        self._start_times[run_id] = now
        self._start_timestamps[run_id] = self._to_iso_timestamp(now)
        
        if parent_run_id:
            self._parent_map[run_id] = parent_run_id
        
        # Extract model name and partner_id
        model_name = self._model_name_from_serialized(serialized) or "unknown"
        if serialized:
            kwargs_dict = serialized.get("kwargs", {})
            # Capture partner_id from LLM config (first time only)
            if not self._partner_id:
                llm_config = kwargs_dict.get("llm_config")
                partner_id = kwargs_dict.get("partner_id")
                if partner_id is None and llm_config is not None:
                    partner_id = (
                        llm_config.get("partner_id")
                        if isinstance(llm_config, dict)
                        else getattr(llm_config, "partner_id", None)
                    )
                self._partner_id = str(partner_id or "") or None
        
        self._run_names[run_id] = f"LLM:{model_name}"
        self._run_types[run_id] = "llm"
        
        depth = self._get_depth(run_id, parent_run_id)
        self._depth_stack[run_id] = depth
    
    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Record time-to-first-token for streaming LLM calls (once per run_id)."""
        if run_id not in self._first_token_times:
            self._first_token_times[run_id] = time.time()
            from agent_builder.utils.handler_io_telemetry import request_elapsed_ms

            elapsed = request_elapsed_ms()
            if elapsed is not None:
                self._http_ttft_ms[run_id] = int(elapsed)
    
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        """Track LLM call completion."""
        start_epoch = self._start_times.pop(run_id, time.time())
        start_time = self._start_timestamps.pop(run_id, None)
        end_time = self._get_current_timestamp()
        duration_ms = int((time.time() - start_epoch) * 1000)
        first_token_epoch = self._first_token_times.pop(run_id, None)
        ttft_ms = (
            int((first_token_epoch - start_epoch) * 1000)
            if first_token_epoch is not None
            else None
        )
        http_ttft_ms = self._http_ttft_ms.pop(run_id, None)
        node_name = self._node_name_for_llm(parent_run_id)
        
        llm_output = response.llm_output or {}
        input_tokens, output_tokens, spending, model_hint = self._token_usage_from_llm_result(
            response,
        )
        if spending == 0 and response.generations and len(response.generations) > 0:
            if len(response.generations[0]) > 0:
                gen_info = response.generations[0][0].generation_info or {}
                spending = gen_info.get("spending", 0)

        # Extract model name
        model_name = self._run_names.get(run_id, "LLM:unknown").replace("LLM:", "")
        if model_name == "unknown":
            model_name = model_hint or llm_output.get("model_name") or llm_output.get("model") or model_name
        
        # Check for empty generations (indicates LLM error with empty response)
        is_empty_response = not response.generations or len(response.generations) == 0 or len(response.generations[0]) == 0
        error_type = None
        error_message = None
        status_code = 200
        
        if is_empty_response:
            # LLM returned empty generations - this is typically an error
            # Try to extract error info from llm_output or kwargs
            status_code = 500  # Default error
            error_type = "EmptyLLMResponse"
            error_message = "LLM returned empty generations"
            
            # Check if there's error info in llm_output
            if "error" in llm_output:
                err = llm_output["error"]
                if isinstance(err, dict):
                    error_message = err.get("message", error_message)
                    status_code = err.get("code", err.get("type", 500))
                    if isinstance(status_code, str):
                        try:
                            status_code = int(status_code)
                        except:
                            status_code = 500
                else:
                    error_message = str(err)
            
            # Check kwargs for error info
            if "error" in kwargs:
                err = kwargs["error"]
                if isinstance(err, dict):
                    error_message = err.get("message", error_message)
                    status_code = err.get("code", err.get("type", status_code))
                    if isinstance(status_code, str):
                        try:
                            status_code = int(status_code)
                        except:
                            pass
        
        # Extract LLM output content
        llm_output_content = None
        llm_tool_calls = None
        tool_calls_list = None
        
        if response.generations and len(response.generations) > 0:
            if len(response.generations[0]) > 0:
                gen = response.generations[0][0]
                if hasattr(gen, 'message'):
                    msg = gen.message
                    llm_output_content = msg.content if hasattr(msg, 'content') else str(gen.text)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        llm_tool_calls = str(msg.tool_calls)
                        tool_calls_list = msg.tool_calls
                elif hasattr(gen, 'text'):
                    llm_output_content = gen.text
        
        # LLM Call Flow: Build message_flow and accumulate totals
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += spending
        self._llm_call_count += 1
        
        # Accumulate tokens to parent spans (agents, tasks, tools)
        self._accumulate_tokens_to_ancestors(run_id, input_tokens, output_tokens, spending)
        
        # Get meaningful parent task/agent name (skip internal nodes like chatbot)
        parent_name, parent_type = self._get_meaningful_parent_info(parent_run_id)
        
        # Convert tool_calls to serializable format
        serializable_tool_calls = None
        if tool_calls_list:
            try:
                serializable_tool_calls = [
                    {
                        "name": tc.get("name", ""),
                        "args": json.loads(json.dumps(tc.get("args", {}), default=str)),
                        "id": tc.get("id", "")
                    }
                    for tc in tool_calls_list
                ]
            except Exception:
                # Fallback to string representation
                serializable_tool_calls = [{"name": tc.get("name", ""), "args": str(tc.get("args", {})), "id": tc.get("id", "")} for tc in tool_calls_list]
        
        # Add to message flow
        message_entry = {
            "id": str(run_id),
            "parent_id": str(parent_run_id) if parent_run_id else None,
            "parent_name": parent_name,
            "parent_type": parent_type,
            "node_name": node_name,
            "model": model_name,
            "content": llm_output_content or "",
            "tool_calls": serializable_tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": spending,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
            "http_ttft_ms": http_ttft_ms,
            "timestamp": end_time,
            "status_code": status_code,
            "error_type": error_type,
            "error_message": error_message,
            "user_query": self._user_query  # Include query in each message for Kibana visualization
        }
        self._llm_message_flow.append(message_entry)
        
        
        if llm_output_content and not tool_calls_list:
            self._final_response = llm_output_content
        
        parent_span_id = self._get_parent_span_id(parent_run_id)
        llm_depth = self._get_meaningful_depth(parent_run_id)
        
        doc = self._create_span_doc(
            span_id=run_id,
            parent_span_id=parent_span_id,
            component_type="llm",
            component_name=model_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status_code=status_code,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=spending,
            model=model_name,
            llm_output=llm_output_content if not is_empty_response else error_message,
            llm_tool_calls=llm_tool_calls,
            depth=llm_depth,
            error_type=error_type,
            error_message=error_message,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=spending,
            llm_calls=1,
            node_name=node_name,
            ttft_ms=ttft_ms,
            http_ttft_ms=http_ttft_ms,
        )
        
        await self._write_to_es(doc)
        self._cleanup_run(run_id)
    
    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        """Track LLM errors."""
        error_type = type(error).__name__
        if error_type == "ParentCommand" or "ParentCommand" in str(error):
            self._cleanup_run(run_id)
            return
        
        # Calculate timing
        start_epoch = self._start_times.pop(run_id, time.time())
        start_time = self._start_timestamps.pop(run_id, None)
        duration_ms = int((time.time() - start_epoch) * 1000)
        first_token_epoch = self._first_token_times.pop(run_id, None)
        ttft_ms = (
            int((first_token_epoch - start_epoch) * 1000)
            if first_token_epoch is not None
            else None
        )
        http_ttft_ms = self._http_ttft_ms.pop(run_id, None)
        node_name = self._node_name_for_llm(parent_run_id)
        
        # Get LLM name and parent
        llm_name = self._run_names.get(run_id, "unknown_llm")
        parent_span_id = self._get_parent_span_id(parent_run_id)
        
        # Extract status code - first try to get from error's code attribute (e.g., LLMRouterError)
        status_code = 500  # Default server error
        if hasattr(error, 'code') and error.code:
            try:
                status_code = int(error.code)
            except (ValueError, TypeError):
                pass
        
        # If no code attribute, try to parse from error message
        if status_code == 500:
            error_str = str(error).lower()
            if "timeout" in error_str or "timed out" in error_str:
                status_code = 408  # Request Timeout
            elif "rate limit" in error_str or "429" in error_str:
                status_code = 429  # Too Many Requests
            elif "unauthorized" in error_str or "401" in error_str:
                status_code = 401  # Unauthorized
            elif "forbidden" in error_str or "403" in error_str:
                status_code = 403  # Forbidden
            elif "bad request" in error_str or "400" in error_str:
                status_code = 400  # Bad Request
            elif "not found" in error_str or "404" in error_str:
                status_code = 404  # Not Found
        
        llm_depth = self._get_meaningful_depth(parent_run_id)
        
        doc = self._create_span_doc(
            span_id=run_id,
            parent_span_id=parent_span_id,
            component_type="llm",
            component_name=llm_name,
            start_time=start_time,
            end_time=self._get_current_timestamp(),
            duration_ms=duration_ms,
            status_code=status_code,
            error_type=error_type,
            error_message=str(error)[:1000],
            depth=llm_depth,
            node_name=node_name,
            ttft_ms=ttft_ms,
            http_ttft_ms=http_ttft_ms,
        )
        
        await self._write_to_es(doc)
        self._cleanup_run(run_id)
    
    # ========== Tool Callbacks ==========
    
    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Track tool execution start."""

        now = time.time()
        self._start_times[run_id] = now
        self._start_timestamps[run_id] = self._to_iso_timestamp(now)
        
        if parent_run_id:
            self._parent_map[run_id] = parent_run_id
        
        tool_name = serialized.get("name", "unknown") if serialized else "unknown"
        
        tool_input = None
        if inputs:
            try:
                import json
                tool_input = json.dumps(inputs)
            except:
                tool_input = str(inputs)
        elif input_str:
            tool_input = input_str
        
        self._tool_inputs[run_id] = tool_input
        
        self._run_names[run_id] = f"Tool:{tool_name}"
        self._run_types[run_id] = "tool"
        
        depth = self._get_depth(run_id, parent_run_id)
        self._depth_stack[run_id] = depth
    
    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        """Track tool execution completion."""
        duration_ms = int((time.time() - self._start_times.pop(run_id, time.time())) * 1000)
        start_time = self._start_timestamps.pop(run_id, None)
        end_time = self._get_current_timestamp()
        
        tool_name = self._run_names.get(run_id, "Tool:unknown").replace("Tool:", "")
        
        # Get tool input from storage
        tool_input = self._tool_inputs.pop(run_id, None)
        
        # Capture tool output
        tool_output = None
        if output is not None:
            try:
                if isinstance(output, (dict, list)):
                    tool_output = json.dumps(output)
                else:
                    tool_output = str(output)
            except:
                tool_output = str(output)
        
        # Check if tool returned an error result
        status_code = 200
        error_type = None
        error_message = None
        
        if tool_output and '"result": "error"' in tool_output.lower():
            status_code = 500  # Tool returned error as result
            error_type = "ToolError"
            # Try to extract error message from output
            try:
                import re
                # Try to find "message": "..." in the output
                match = re.search(r'"message"\s*:\s*"([^"]+)"', tool_output)
                if match:
                    error_message = match.group(1)
                else:
                    error_message = "Tool returned error result"
            except:
                error_message = "Tool returned error result"
        
        parent_span_id = self._get_parent_span_id(parent_run_id)
        # Tool depth is meaningful parent depth + 1
        tool_depth = self._get_meaningful_depth(parent_run_id)
        
        # Get accumulated tokens from child LLM calls (for tools that invoke LLMs)
        span_tokens = self._get_span_tokens(run_id)
        
        # Lookup tool_id from tool name
        tool_id = self._tool_id_map.get(tool_name)
        
        doc = self._create_span_doc(
            span_id=run_id,
            parent_span_id=parent_span_id,
            component_type="tool",
            component_name=tool_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status_code=status_code,
            error_type=error_type,
            error_message=error_message,
            tool_input=tool_input,
            tool_output=tool_output,
            depth=tool_depth,
            total_input_tokens=span_tokens["input_tokens"],
            total_output_tokens=span_tokens["output_tokens"],
            total_cost=span_tokens["cost"],
            llm_calls=span_tokens["llm_calls"],
            tool_id=tool_id
        )
        
        await self._write_to_es(doc)
        self._cleanup_run(run_id)
    
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        """Track tool errors."""
        error_type = type(error).__name__
        if error_type == "ParentCommand" or "ParentCommand" in str(error):
            self._cleanup_run(run_id)
            return
        
        # Calculate timing
        duration_ms = int((time.time() - self._start_times.pop(run_id, time.time())) * 1000)
        start_time = self._start_timestamps.pop(run_id, None)
        
        # Get tool name and parent (strip "Tool:" prefix for consistency)
        tool_name = self._run_names.get(run_id, "Tool:unknown_tool").replace("Tool:", "")
        parent_span_id = self._get_parent_span_id(parent_run_id)
        
        # Get tool input if captured
        tool_input = self._tool_inputs.pop(run_id, None)
        
        # Extract status code - first try to get from error's code attribute
        status_code = 500  # Default server error
        if hasattr(error, 'code') and error.code:
            try:
                status_code = int(error.code)
            except (ValueError, TypeError):
                pass
        
        # If no code attribute, try to parse from error message
        if status_code == 500:
            error_str = str(error).lower()
            if "validation" in error_str or "invalid" in error_str:
                status_code = 400  # Bad Request
            elif "not found" in error_str or "404" in error_str:
                status_code = 404  # Not Found
            elif "timeout" in error_str or "timed out" in error_str:
                status_code = 408  # Request Timeout
            elif "permission" in error_str or "forbidden" in error_str:
                status_code = 403  # Forbidden
            elif "unauthorized" in error_str or "401" in error_str:
                status_code = 401  # Unauthorized
        
        tool_depth = self._get_meaningful_depth(parent_run_id)
        
        # Lookup tool_id from tool name
        tool_id = self._tool_id_map.get(tool_name)
        
        doc = self._create_span_doc(
            span_id=run_id,
            parent_span_id=parent_span_id,
            component_type="tool",
            component_name=tool_name,
            start_time=start_time,
            end_time=self._get_current_timestamp(),
            duration_ms=duration_ms,
            status_code=status_code,
            error_type=error_type,
            error_message=str(error)[:1000],
            tool_input=tool_input,
            depth=tool_depth,
            tool_id=tool_id
        )
        
        await self._write_to_es(doc)
        self._cleanup_run(run_id)
    
    # ========== Utility ==========
    
    def _cleanup_run(self, run_id: UUID):
        """Clean up tracking data for a completed run."""
        self._start_times.pop(run_id, None)
        self._start_timestamps.pop(run_id, None)
        self._run_names.pop(run_id, None)
        self._run_types.pop(run_id, None)
        self._depth_stack.pop(run_id, None)
        self._parent_map.pop(run_id, None)
        self._tool_inputs.pop(run_id, None)
        self._span_tokens.pop(run_id, None)
        self._first_token_times.pop(run_id, None)
        self._http_ttft_ms.pop(run_id, None)