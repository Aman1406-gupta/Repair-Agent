"""MCP tool discovery and loading.

High-level API (used by invoke pipeline):
    load_mcp_tools     — first turn: discover endpoint, fetch tools via adapter
    rebuild_mcp_tools  — subsequent turns: reconstruct from cached descriptors

Composable building blocks (usable directly from SDK):
    build_mcp_connection       — build a single streamable-HTTP connection dict
    fetch_tools_for_connection — fetch tools from one MCP server connection
    extract_descriptors        — serialize LangChain tools to cacheable dicts
    descriptors_to_tools       — rebuild LangChain tools from cached descriptors
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import Tool as McpTool

from agent_builder.utils.constants import AGENT_ID, VERSION

MCP_CONFIG_BASE_URL = os.environ.get("AGENT_BUILDER_MCP_CONFIG_BASE_URL", "")
MCP_ENDPOINT_CACHE_TTL = int(os.environ.get("MCP_ENDPOINT_CACHE_TTL", 1800))

logger = logging.getLogger(__name__)

_ENDPOINT_CACHE_NS = "mcp_endpoint"


# ── High-level API (invoke pipeline) ───────────────────────────────


async def load_mcp_tools(
    doc: dict,
    partner_id: int,
    jwt: str,
    redis_client: Any,
) -> Tuple[Dict[str, list], Dict[str, dict]]:
    """Fetch MCP tools for all enabled tasks under an agent.

    Returns ``(tools_by_task, descriptors_by_task)`` where each task cache entry
    is ``{"endpoint": str, "tools": [descriptor, ...]}``.
    Both empty when MCP is not configured or no endpoints are available.
    """
    if not MCP_CONFIG_BASE_URL:
        return {}, {}

    endpoints = await _resolve_endpoints(partner_id, redis_client)
    if not endpoints:
        return {}, {}

    agent_id, version = _extract_agent_identity(doc)
    enabled_tasks = [t for t in doc.get("tasks", []) if t.get("enabled", True)]

    tools_by_task: Dict[str, list] = {}
    descriptors_by_task: Dict[str, dict] = {}

    for task in enabled_tasks:
        task_id = str(task.get("_id", ""))
        if not task_id:
            continue
        endpoint, tools = await _discover_tools_for_task(
            task_id, endpoints, partner_id, agent_id, version, jwt,
        )
        if endpoint and tools is not None:
            tools_by_task[task_id] = tools
            descriptors_by_task[task_id] = _make_task_cache(endpoint, tools)
        else:
            logger.warning("MCP tools/list failed for task %s on all candidate endpoints", task_id)

    return tools_by_task, descriptors_by_task


async def rebuild_mcp_tools(
    descriptors_by_task: Dict[str, dict],
    doc: dict,
    partner_id: int,
    jwt: str,
) -> Dict[str, list]:
    """Reconstruct LangChain tools from cached descriptors without calling tools/list.

    Reads the resolved endpoint from each task cache entry in ``descriptors_by_task``.
    Returns empty dict when descriptors are empty or entries lack endpoint/tools.
    """
    if not descriptors_by_task:
        return {}

    agent_id, version = _extract_agent_identity(doc)

    tools_by_task: Dict[str, list] = {}
    for task_id, entry in descriptors_by_task.items():
        endpoint = entry.get("endpoint")
        descriptors = entry.get("tools")
        if not endpoint or not descriptors:
            logger.warning("MCP task cache missing endpoint or tools for task %s", task_id)
            continue
        connection = build_mcp_connection(endpoint, partner_id, agent_id, version, task_id, jwt)
        tools_by_task[task_id] = descriptors_to_tools(descriptors, connection)

    return tools_by_task


# ── Composable building blocks ─────────────────────────────────────


def normalize_mcp_url(endpoint: str) -> str:
    """Ensure an MCP endpoint has an HTTP scheme."""
    endpoint = (endpoint or "").strip()
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint


def build_mcp_connection(
    endpoint: str,
    partner_id: int,
    agent_id: str,
    version: int,
    task_id: str,
    jwt: str,
) -> Dict[str, Any]:
    """Build a single streamable-HTTP connection dict for one MCP server.

    The returned dict can be passed to ``MultiServerMCPClient``,
    ``fetch_tools_for_connection``, or ``descriptors_to_tools``.

    ``Authorization`` and ``X-Spr-Ai-Task-Id`` are omitted when the
    corresponding value is missing or blank so upstream servers are not sent
    ``Bearer `` or empty routing headers.
    """
    headers: Dict[str, str] = {
        "X-Spr-Partner-Id": str(partner_id),
        "X-Spr-Ai-Agent-Version": str(version),
    }
    aid = (agent_id or "").strip()
    if aid:
        headers["X-Spr-Ai-Agent-Id"] = aid
    tid = (task_id or "").strip()
    if tid:
        headers["X-Spr-Ai-Task-Id"] = tid
    token = (jwt or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return {
        "transport": "streamable_http",
        "url": endpoint,
        "headers": headers,
    }


async def fetch_tools_for_connection(
    server_name: str,
    connection: Dict[str, Any],
) -> list:
    """Fetch LangChain tools from a single MCP server connection via ``tools/list``."""
    client = MultiServerMCPClient(connections={server_name: connection})
    return await client.get_tools(server_name=server_name)


def extract_descriptors(tools: list) -> List[Dict[str, Any]]:
    """Serialize LangChain tools into JSON-safe dicts for caching.

    Each descriptor contains ``name``, ``description``, ``inputSchema``,
    and ``annotations`` (tool metadata).
    """
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.args_schema if isinstance(t.args_schema, dict) else (t.args_schema.model_json_schema() if t.args_schema else {}),
            "annotations": t.metadata or {},
        }
        for t in tools
    ]


def descriptors_to_tools(
    descriptors: List[Dict[str, Any]],
    connection: Dict[str, Any],
) -> list:
    """Rebuild LangChain tools from cached descriptors + a live connection.

    Does NOT call ``tools/list`` — uses the cached schema and creates tools
    that open a lazy MCP session on first ``tools/call``.
    """
    tools = []
    for desc in descriptors:
        mcp_tool = McpTool(
            name=desc["name"],
            description=desc.get("description", ""),
            inputSchema=desc.get("inputSchema", {}),
        )
        lc_tool = convert_mcp_tool_to_langchain_tool(
            session=None, tool=mcp_tool, connection=connection,
        )
        cached_metadata = desc.get("annotations")
        if cached_metadata:
            lc_tool.metadata = cached_metadata
        tools.append(lc_tool)
    return tools


# ── Internal helpers ───────────────────────────────────────────────


def _make_task_cache(endpoint: str, tools: list) -> dict:
    return {"endpoint": endpoint, "tools": extract_descriptors(tools)}


def _parse_endpoints_from_response(data: dict) -> List[str]:
    """Extract normalized endpoint URLs from an mcp-config response."""
    servers = data.get("servers")
    if not isinstance(servers, list):
        logger.warning("mcp-config response missing servers list")
        return []

    endpoints: List[str] = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        raw = server.get("endpoint")
        if raw:
            endpoints.append(normalize_mcp_url(str(raw)))
    return endpoints


async def _discover_tools_for_task(
    task_id: str,
    endpoints: List[str],
    partner_id: int,
    agent_id: str,
    version: int,
    jwt: str,
) -> Tuple[Optional[str], Optional[list]]:
    """Probe candidate endpoints until tools/list succeeds for a task."""
    for endpoint in endpoints:
        connection = build_mcp_connection(endpoint, partner_id, agent_id, version, task_id, jwt)
        try:
            tools = await fetch_tools_for_connection(task_id, connection)
            return endpoint, tools
        except Exception as e:
            logger.debug(
                "MCP tools/list failed for task %s on %s: %s",
                task_id, endpoint, e,
            )
    return None, None


async def _resolve_endpoints(partner_id: int, redis_client: Any) -> List[str]:
    """Return candidate MCP endpoint URLs for a partner (Redis cache, then mcp-config API)."""
    cached = await redis_client.cache_get(_ENDPOINT_CACHE_NS, str(partner_id))
    if cached is not None:
        endpoints = cached.get("endpoints")
        if isinstance(endpoints, list) and endpoints:
            logger.info(
                "MCP endpoints from partner cache | partner_id=%s endpoints=%s",
                partner_id, endpoints,
            )
            return endpoints

    endpoints = await _fetch_endpoints_from_config(partner_id)
    if endpoints:
        await redis_client.cache_set(
            _ENDPOINT_CACHE_NS,
            str(partner_id),
            {"endpoints": endpoints},
            ttl=MCP_ENDPOINT_CACHE_TTL,
        )
    return endpoints


async def _fetch_endpoints_from_config(partner_id: int) -> List[str]:
    """POST to internal mcp-config API to discover MCP endpoints for a partner."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                MCP_CONFIG_BASE_URL, json={"partnerId": partner_id}, timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("mcp-config returned %s for partner %s", resp.status, partner_id)
                    return []
                data = await resp.json()
                endpoints = _parse_endpoints_from_response(data)
                if not endpoints:
                    logger.warning("mcp-config response has no endpoints for partner %s", partner_id)
                else:
                    logger.info(
                        "MCP endpoints from mcp-config | partner_id=%s endpoints=%s",
                        partner_id, endpoints,
                    )
                return endpoints
    except Exception as e:
        logger.exception("Failed to fetch MCP endpoints for partner %s: %s", partner_id, e)
        return []


def _extract_agent_identity(doc: dict) -> Tuple[str, int]:
    """Return (agent_id_str, version_int) from an agent document."""
    agent_id = str(doc.get(AGENT_ID, doc.get("_id", "")))
    version = int(doc.get(VERSION, 0))
    return agent_id, version
