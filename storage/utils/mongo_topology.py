"""
Mongo store primitives and embedded agent-graph helpers.

Tasks, tools, and nested agents are stored fully embedded inside agent
documents.
"""

import functools
import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Set, Union

from bson import ObjectId

from agent_builder.base.tools import openapi_spec_to_metadata
from agent_builder.base.tools_mock import (
    openapi_spec_to_prompt_tool_metadata,
    openai_schema_to_prompt_tool_metadata,
)
from agent_builder.handlers.core.requests import (
    RegisterAgentRequest,
    RegisterPromptToolRequest,
    RegisterReleaseTaskRequest,
    RegisterTaskRequest,
    RegisterToolRequest,
)
from agent_builder.storage.redis_client import _CACHE_NS, build_cache_key
from agent_builder.utils.handler_io_telemetry import _timed_mongo
from agent_builder.utils.constants import (
    AGENT_AS_TOOLS,
    AGENT_AS_TASK,
    AGENT_COLLECTION,
    AGENT_TYPE,
    ATTRIBUTES,
    DEFAULT_SERVER_TYPE,
    DEFAULT_SWARM_TYPE,
    DEFAULT_TASK_TYPE,
    DESCRIPTION,
    ENABLED,
    HTTP_CONFIG,
    LLM_CONFIG,
    NAME,
    PARTNER_ID,
    POSTPROCESSOR,
    PREPROCESSOR,
    ROUTER_MODEL_CONFIG,
    SERVER_TYPE,
    SKILLS_ZIP,
    SUBAGENTS,
    SWARM_TYPE,
    SYSTEM_TEMPLATE,
    TASK_AS_ROUTER,
    TASK_AS_TOOLS,
    TASK_FORM,
    TASK_TYPE,
    TASK_TYPE_RELEASE,
    TASKS,
    TOOL_TYPE_API,
    TOOL_TYPE_KEY,
    TOOLS,
    WORKFLOW_EDGES,
)

logger = logging.getLogger(__name__)


def get_server_type_by_pid(partner_id: int) -> str:
    return SERVER_TYPE if partner_id == 0 else DEFAULT_SERVER_TYPE


# ── Errors ──────────────────────────────────────────────────────────


class AgentBuilderStoreError(Exception):
    pass


class DuplicateAgentError(AgentBuilderStoreError):
    pass


class DuplicateAgentMetadataError(AgentBuilderStoreError):
    pass


# ── Utility Helpers ─────────────────────────────────────────────────


def add_meta(doc: Dict[str, Any]) -> None:
    """Stamp created_time / modified_time / deleted on a new document."""
    now = int(time.time() * 1000)
    doc.setdefault("created_time", now)
    doc.setdefault("modified_time", now)
    doc.setdefault("deleted", False)


def maybe_object_id(value: Union[str, ObjectId]) -> Union[str, ObjectId]:
    if isinstance(value, str):
        try:
            return ObjectId(value)
        except Exception:
            return value
    return value


def convert_objectids(doc: Any) -> Any:
    """Recursively convert ObjectId instances to strings for JSON serialization."""
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, dict):
        return {k: convert_objectids(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [convert_objectids(item) for item in doc]
    return doc


def _to_oids(ids: List[str]) -> List[ObjectId]:
    """Convert string IDs to ObjectIds, raising on invalid format."""
    result = []
    for raw in ids:
        try:
            result.append(ObjectId(raw))
        except Exception:
            raise AgentBuilderStoreError(f"Invalid ObjectId: {raw}")
    return result


def _ensure_unique_ids(items: List[Dict[str, Any]], label: str) -> None:
    seen: Set[str] = set()
    for item in items:
        raw = item.get("_id")
        if raw is None:
            continue
        sid = str(raw)
        if sid in seen:
            raise AgentBuilderStoreError(f"Duplicate {label} id: {sid}")
        seen.add(sid)


def assign_id(doc: Dict[str, Any], client_id: Optional[str] = None) -> None:
    """Assign ``_id`` on a new embedded node; prefer client-supplied id."""
    if doc.get("_id"):
        doc["_id"] = maybe_object_id(doc["_id"])
        return
    if client_id:
        doc["_id"] = maybe_object_id(client_id)
    else:
        doc["_id"] = ObjectId()
    add_meta(doc)


def tool_requests_to_metadata(
    tools: List[Union[RegisterToolRequest, RegisterPromptToolRequest]],
) -> List[Dict[str, Any]]:
    """Expand tool registration requests into metadata dicts for embedding."""
    from agent_builder.utils.constants import DEFAULT_LLM_CONFIG

    out: List[Dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, RegisterToolRequest):
            schema_title = tool.openapi_schema.get("info", {}).get("title", "<unknown>")
            metas = openapi_spec_to_metadata(
                tool.openapi_schema, filter_name=tool.filter_name,
            )
            if not metas:
                raise AgentBuilderStoreError(
                    f"No tools extracted from schema '{schema_title}'",
                )
            for meta in metas:
                meta[TOOL_TYPE_KEY] = TOOL_TYPE_API
                out.append(meta)
        elif isinstance(tool, RegisterPromptToolRequest):
            llm_config = tool.llm_config if tool.llm_config is not None else DEFAULT_LLM_CONFIG
            llm_behavior = tool.llm_behavior or "Return a plausible mock JSON response."
            if tool.openapi_schema is not None:
                metas = openapi_spec_to_prompt_tool_metadata(
                    spec=tool.openapi_schema,
                    llm_config=llm_config,
                    default_behavior=llm_behavior,
                    filter_name=tool.filter_name,
                )
                if not metas:
                    raise AgentBuilderStoreError("No tools extracted from OpenAPI spec")
            else:
                metas = [
                    openai_schema_to_prompt_tool_metadata(
                        schema=tool.openai_schema,
                        llm_config=llm_config,
                        default_behavior=llm_behavior,
                    ),
                ]
            out.extend(metas)
        else:
            raise AgentBuilderStoreError(f"Unsupported tool request type: {type(tool)}")
    return out


def task_request_to_doc(
    request: Union[RegisterTaskRequest, RegisterReleaseTaskRequest],
) -> Dict[str, Any]:
    """Build an embedded task document from a registration request."""
    if isinstance(request, RegisterReleaseTaskRequest):
        doc: Dict[str, Any] = {
            NAME: request.name,
            DESCRIPTION: request.description,
            TASK_TYPE: TASK_TYPE_RELEASE,
            HTTP_CONFIG: request.http_config.model_dump(),
            TASK_FORM: request.task_form,
            ATTRIBUTES: request.attributes,
            ENABLED: request.enabled,
        }
        if request.id:
            doc["_id"] = request.id
        return doc

    tools_meta = tool_requests_to_metadata(request.tools)
    nested = [
        task_request_to_doc(nested_req)
        for nested_req in (request.task_as_tools or [])
    ]
    doc = {
        NAME: request.name,
        DESCRIPTION: request.description,
        TASK_TYPE: request.task_type or DEFAULT_TASK_TYPE,
        SYSTEM_TEMPLATE: request.system_template,
        LLM_CONFIG: request.llm_config,
        PREPROCESSOR: request.preprocessor,
        POSTPROCESSOR: request.postprocessor,
        TOOLS: tools_meta,
        TASK_AS_TOOLS: nested,
        AGENT_AS_TOOLS: [
            agent_request_to_doc(r) for r in (request.agent_as_tools or [])
        ],
        ENABLED: request.enabled,
    }
    if request.skills_zip:
        doc[SKILLS_ZIP] = request.skills_zip
    if request.subagents:
        doc[SUBAGENTS] = request.subagents
    if request.id:
        doc["_id"] = request.id
    return doc


def agent_request_to_doc(
    request: RegisterAgentRequest,
    *,
    default_agent_type: Optional[str] = None,
    default_partner_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build an embedded agent document from a registration request."""
    agent_type = request.agent_type or default_agent_type
    partner_id = (
        request.partner_id
        if request.partner_id is not None
        else default_partner_id
    )
    nested = [
        agent_request_to_doc(
            child,
            default_agent_type=agent_type,
            default_partner_id=partner_id,
        )
        for child in (request.agent_as_task or [])
    ]
    doc: Dict[str, Any] = {
        NAME: request.name,
        DESCRIPTION: request.description,
        TASKS: [task_request_to_doc(t) for t in (request.tasks or [])],
        WORKFLOW_EDGES: list(request.workflow_edges or []),
        ROUTER_MODEL_CONFIG: request.llm_config,
        SWARM_TYPE: request.swarm_type or DEFAULT_SWARM_TYPE,
        AGENT_AS_TASK: nested,
        TASK_AS_ROUTER: (
            task_request_to_doc(request.task_as_router)
            if request.task_as_router is not None
            else None
        ),
        AGENT_TYPE: agent_type,
        PARTNER_ID: partner_id,
    }
    if request.id:
        doc["_id"] = request.id
    return doc


def assign_missing_ids(agent_doc: Dict[str, Any]) -> None:
    """Assign ``_id`` on embedded agents, tasks, and tools in an agent doc."""
    embedded_agents = agent_doc.get(AGENT_AS_TASK) or []
    _ensure_unique_ids(embedded_agents, "agent_as_task")
    for embedded in embedded_agents:
        if not embedded.get("_id"):
            assign_id(embedded)
        else:
            embedded["_id"] = maybe_object_id(embedded["_id"])
        assign_missing_ids(embedded)

    router = agent_doc.get(TASK_AS_ROUTER)
    if isinstance(router, dict):
        _assign_ids_on_task(router)

    tasks = agent_doc.get(TASKS) or []
    _ensure_unique_ids(tasks, "task")
    for task in tasks:
        _assign_ids_on_task(task)


def _assign_ids_on_task(task: Dict[str, Any]) -> None:
    if not task.get("_id"):
        assign_id(task)
    else:
        task["_id"] = maybe_object_id(task["_id"])

    tools = task.get(TOOLS) or []
    _ensure_unique_ids(tools, "tool")
    for tool in tools:
        if not tool.get("_id"):
            assign_id(tool)
        else:
            tool["_id"] = maybe_object_id(tool["_id"])

    nested_tasks = task.get(TASK_AS_TOOLS) or []
    _ensure_unique_ids(nested_tasks, "task_as_tools")
    for nested in nested_tasks:
        _assign_ids_on_task(nested)


def collect_embedded_task_ids(agent_doc: Dict[str, Any]) -> List[str]:
    """Collect embedded task and ``agent_as_task`` node ids for workflow edge validation."""
    ids: List[str] = []

    def walk_agent(agent: Dict[str, Any]) -> None:
        for embedded in agent.get(AGENT_AS_TASK) or []:
            if isinstance(embedded, dict) and embedded.get("_id"):
                ids.append(str(embedded["_id"]))
                walk_agent(embedded)
        for task in agent.get(TASKS) or []:
            walk_task(task)
        router = agent.get(TASK_AS_ROUTER)
        if isinstance(router, dict):
            walk_task(router)

    def walk_task(task: Dict[str, Any]) -> None:
        if task.get("_id"):
            ids.append(str(task["_id"]))
        for nested in task.get(TASK_AS_TOOLS) or []:
            walk_task(nested)

    walk_agent(agent_doc)
    return ids


def validate_workflow_edges(edges: List[tuple], valid_node_ids: List[str]) -> None:
    """Ensure workflow edge endpoints reference embedded task or agent-as-task ids."""
    valid = set(valid_node_ids)
    for edge in edges:
        if not all(n in valid for n in edge):
            raise AgentBuilderStoreError(
                f"Invalid workflow edges: {edges} (valid nodes: {sorted(valid)})",
            )


# ── Error Handling Decorator ─────────────────────────────────────────


_RAISE = object()


def store_error_handler(context=None, *, fallback=_RAISE):
    """Standardised error handling for async store methods."""
    context = context or {}

    def _build_context(fn, args, kwargs):
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        bound = {params[i]: v for i, v in enumerate(args) if i < len(params)}
        bound.update(kwargs)
        parts = []
        for label, path in context.items():
            root, *attrs = path.split(".")
            obj = bound.get(root)
            for attr in attrs:
                obj = getattr(obj, attr, None) if obj is not None else None
            if obj is not None:
                parts.append(f"{label}={obj}")
        return f" ({', '.join(parts)})" if parts else ""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except AgentBuilderStoreError:
                raise
            except Exception as e:
                ctx = _build_context(fn, args, kwargs)
                msg = f"{fn.__name__} failed{ctx}"
                logger.exception("%s: %s", msg, e)
                if fallback is _RAISE:
                    raise AgentBuilderStoreError(msg) from e
                return fallback
        return wrapper
    return decorator


# ── Mixin: thin DB wrappers ─────────────────────────────────────────


class EmbedStoreMixin:
    """Mongo CRUD + cache helpers for AgentBuilderMongoStore."""

    async def _sync_cache(
        self,
        collection: str,
        query_filter: Dict[str, Any],
        doc: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._redis_client or not _CACHE_NS.get(collection):
            return

        spec = _CACHE_NS[collection]
        key_val = build_cache_key(spec, query_filter) or (
            build_cache_key(spec, doc) if doc else None
        )
        if not key_val:
            return

        if doc:
            await self._redis_client.cache_set(
                spec.ns, str(key_val), convert_objectids(doc),
            )
        else:
            await self._redis_client.cache_del(spec.ns, str(key_val))

    @_timed_mongo("find_one")
    async def _find_one(self, collection: str, query: Dict, *, partner_id: Optional[int] = None) -> Optional[Dict]:
        try:
            pid = partner_id if partner_id is not None else self._partner_id
            res = await self._client.find_one_in_collection_by_server_type_and_pid(
                collection=collection,
                server_type=get_server_type_by_pid(pid),
                partner_id=pid,
                query_params=query,
            )
            doc = res.get("mongo_response")
            await self._sync_cache(collection, query, doc)
            return convert_objectids(doc)
        except Exception as e:
            logger.error("_find_one failed (collection=%s): %s", collection, e)
            return None

    @_timed_mongo("find_many")
    async def _find_many(self, collection: str, query: Dict, *, partner_id: Optional[int] = None) -> List[Dict]:
        pid = partner_id if partner_id is not None else self._partner_id
        res = await self._client.find_in_collection_by_server_type_and_pid(
            collection=collection,
            server_type=get_server_type_by_pid(pid),
            partner_id=pid,
            query_params=query,
        )
        return res.get("mongo_response", [])

    @_timed_mongo("insert_one")
    async def _insert_one(self, collection: str, doc: Dict, *, partner_id: Optional[int] = None) -> Optional[ObjectId]:
        add_meta(doc)
        pid = partner_id if partner_id is not None else self._partner_id
        res = await self._client.insert_one_in_collection_by_server_type_and_pid(
            collection=collection,
            server_type=get_server_type_by_pid(pid),
            partner_id=pid,
            document=doc,
        )
        inserted_id = res.get("mongo_response", {}).get("inserted_id")
        logger.debug("Inserted doc | collection=%s id=%s", collection, inserted_id)
        if inserted_id:
            await self._sync_cache(
                collection, {"_id": inserted_id}, {**doc, "_id": inserted_id},
            )
        return inserted_id

    @_timed_mongo("insert_many")
    async def _insert_many(self, collection: str, docs: List[Dict], *, partner_id: Optional[int] = None) -> Optional[List]:
        for d in docs:
            add_meta(d)
        pid = partner_id if partner_id is not None else self._partner_id
        res = await self._client.insert_many_in_collection_by_server_type_and_pid(
            collection=collection,
            server_type=get_server_type_by_pid(pid),
            partner_id=pid,
            documents=docs,
        )
        return res.get("mongo_response", {}).get("inserted_ids")

    @_timed_mongo("update_one")
    async def _update_one(
        self,
        collection: str,
        doc_id: Union[str, ObjectId],
        fields: Dict,
        base_doc: Optional[Dict] = None,
        *,
        partner_id: Optional[int] = None,
    ) -> Dict:
        """Write *fields* to the document and return the timestamped update dict."""
        doc_id = maybe_object_id(doc_id)
        fields.setdefault("modified_time", int(time.time() * 1000))
        pid = partner_id if partner_id is not None else self._partner_id
        await self._client.update_one_in_collection_by_server_type_and_pid(
            collection=collection,
            server_type=get_server_type_by_pid(pid),
            partner_id=pid,
            query_params={"_id": doc_id},
            update_params=fields,
        )
        logger.debug(
            "Updated doc | collection=%s id=%s fields=%s",
            collection, doc_id, list(fields.keys()),
        )
        updated_doc = {**base_doc, **fields} if base_doc is not None else None
        await self._sync_cache(collection, {"_id": doc_id}, updated_doc)
        return fields

    @_timed_mongo("delete_one")
    async def _delete_one(self, collection: str, doc_id: Union[str, ObjectId], *, partner_id: Optional[int] = None) -> int:
        doc_id = maybe_object_id(doc_id)
        pid = partner_id if partner_id is not None else self._partner_id
        doc = await self._find_one(collection, {"_id": doc_id}, partner_id=pid)
        res = await self._client.delete_one_in_collection_by_server_type_and_pid(
            collection=collection,
            server_type=get_server_type_by_pid(pid),
            partner_id=pid,
            query_params={"_id": doc_id},
        )
        if doc:
            spec = _CACHE_NS.get(collection)
            if spec and self._redis_client:
                key = build_cache_key(spec, doc)
                if key:
                    await self._redis_client.cache_del(spec.ns, key)
        return res.get("mongo_response", {}).get("deleted_count", 0)

    async def _resolve_refs(self, collection: str, ids: List[str], *, partner_id: Optional[int] = None) -> List[Dict]:
        """Fetch docs by IDs in a single query. Raises if any missing."""
        if not ids:
            return []
        oids = _to_oids(ids)
        found = await self._find_many(collection, {"_id": {"$in": oids}}, partner_id=partner_id)
        for d in found:
            await self._sync_cache(collection, {"_id": d["_id"]}, d)

        found_ids = {d["_id"] for d in found}
        missing = set(oids) - found_ids
        if missing:
            raise AgentBuilderStoreError(
                f"Not found in {collection}: {sorted(map(str, missing))}",
            )

        return [convert_objectids(d) for d in found]

    async def _find_one_cached(self, collection: str, query: Dict, *, partner_id: Optional[int] = None) -> Optional[Dict]:
        spec = _CACHE_NS.get(collection)
        if spec and self._redis_client:
            key = build_cache_key(spec, query)
            if key is not None:
                cached = await self._redis_client.cache_get(spec.ns, key)
                if cached is not None:
                    return cached

        return await self._find_one(collection, query, partner_id=partner_id)

    async def _resolve_refs_cached(self, collection: str, ids: List[str], *, partner_id: Optional[int] = None) -> List[Dict]:
        """Batch read from Redis when possible; on miss delegate to :meth:`_resolve_refs` (which syncs cache).

        Falls back to a non-cached resolve for composite cache keys since
        batch mget requires simple single-field keys.
        """
        spec = _CACHE_NS.get(collection)
        if not spec or not self._redis_client or isinstance(spec.key_field, tuple):
            return await self._resolve_refs(collection, ids, partner_id=partner_id)

        cached = await self._redis_client.cache_mget(spec.ns, ids)
        if missing_ids := [id for id in ids if id not in cached]:
            found = await self._resolve_refs(collection, missing_ids, partner_id=partner_id)
            cached.update({str(doc[spec.key_field]): doc for doc in found})

        return [cached[id] for id in ids]

