"""Shared utilities for fetching and persisting agents from the external platform.

Used by both the ``/platform/sync/agent`` handler and the invoke-time
fallback in ``_resolve_agent``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import zipfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import aiohttp
import tornado.web

from agent_builder.base.configs import LLMConfig as LLMConfigDataclass
from agent_builder.storage.utils.mongo_topology import AgentBuilderStoreError
from agent_builder.utils.constants import (
    AGENT_COLLECTION,
    AGENT_ID,
    AGENT_TYPE,
    AGENT_SYNC_SERVICE_URL,
    ATTRIBUTES,
    DEFAULT_TASK_TYPE,
    DESCRIPTION,
    ENABLED,
    ENV_SYNC_POPULATE_LLM_CONFIGURATION_ID,
    HTTP_CONFIG,
    LLM_CONFIGURATION_ID,
    NAME,
    PARTNER_ID,
    PLATFORM_LLM_CONFIGURATION_ID,
    PLATFORM_SKILLS_ZIPPED,
    PLATFORM_TASK_TYPE_CUSTOM,
    PLATFORM_TASK_TYPE_STANDARD,
    SKILLS_ZIP,
    SUBAGENTS,
    SWARM_TYPE,
    TASKS,
    TASK_AS_ROUTER,
    TASK_FORM,
    TASK_TYPE,
    TASK_TYPE_RELEASE,
    VERSION,
    DEFAULT_SWARM_TYPE,
)

logger = logging.getLogger(__name__)


# ── LLM-config merge helpers ──────────────────────────────────────────

def _env_flag_truthy(var_name: str) -> bool:
    return os.environ.get(var_name, "").strip().lower() in ("1", "true", "yes", "on")


def _should_populate_platform_llm_configuration_id() -> bool:
    return _env_flag_truthy(ENV_SYNC_POPULATE_LLM_CONFIGURATION_ID)


def _merged_llm_config_dict_from_platform_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    raw = task_data.get("llm_config")
    if not isinstance(raw, dict):
        raw = {}
    merged: Dict[str, Any] = {**asdict(LLMConfigDataclass()), **raw}
    if _should_populate_platform_llm_configuration_id():
        lid = (
            task_data.get(PLATFORM_LLM_CONFIGURATION_ID)
            or task_data.get(LLM_CONFIGURATION_ID)
        )
        if lid:
            merged[LLM_CONFIGURATION_ID] = str(lid)
    return merged


# ── task-doc builders ─────────────────────────────────────────────────

def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def build_standard_task_doc(
    task_data: Dict[str, Any],
    task_name: str,
    task_description: str,
    enabled: bool,
) -> Dict[str, Any]:
    http_cfg_raw = task_data.get("httpConfig", {})
    return {
        NAME: task_name,
        DESCRIPTION: task_description,
        TASK_TYPE: TASK_TYPE_RELEASE,
        ENABLED: enabled,
        HTTP_CONFIG: {
            "url": http_cfg_raw.get("url", ""),
            "proxy_server": http_cfg_raw.get("proxyServer", ""),
            "proxy_port": http_cfg_raw.get("proxyPort", ""),
        },
        ATTRIBUTES: task_data.get(ATTRIBUTES, {}),
        TASK_FORM: task_data.get(TASK_FORM, ""),
    }


def build_custom_task_doc(
    task_data: Dict[str, Any],
    task_name: str,
    task_description: str,
    enabled: bool,
) -> Dict[str, Any]:
    task_prompt = task_data.get("taskPrompt", {})
    prompt = task_prompt.get("prompt", "") if isinstance(task_prompt, dict) else ""

    doc: Dict[str, Any] = {
        NAME: task_name,
        DESCRIPTION: task_description,
        TASK_TYPE: DEFAULT_TASK_TYPE,
        ENABLED: enabled,
        "system_template": prompt,
        "llm_config": _merged_llm_config_dict_from_platform_task(task_data),
        "preprocessor": "DEFAULT",
        "postprocessor": None,
        "tools": [],
        "task_as_tools": [],
        "agent_as_tools": [],
    }
    if task_data.get(SUBAGENTS):
        doc[SUBAGENTS] = task_data[SUBAGENTS]
    return doc


def build_task_doc(
    task_data: Dict[str, Any], task_type: str, enabled: bool,
) -> Dict[str, Any]:
    task_name = task_data.get(NAME, "")
    task_description = task_data.get("detail")
    if task_description is None:
        task_description = ""

    if task_type == PLATFORM_TASK_TYPE_STANDARD:
        doc = build_standard_task_doc(task_data, task_name, task_description, enabled)
    else:
        doc = build_custom_task_doc(task_data, task_name, task_description, enabled)

    platform_id = task_data.get("id")
    if platform_id:
        doc["_id"] = str(platform_id)

    return doc


# ── platform skill resolution ─────────────────────────────────────────

def _md_to_skill_zip(name: str, markdown_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/SKILL.md", markdown_text)
    return buf.getvalue()


async def resolve_platform_skills(skills_info: List[Dict[str, Any]]) -> List[str]:
    if not skills_info:
        return []

    resolved: List[str] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        for skill in skills_info:
            name = skill.get(NAME, "")
            skill_type = str(skill.get("type", "")).strip().upper()
            url = skill.get("url", "")
            if not name or not skill_type or not url:
                raise tornado.web.HTTPError(502, f"Invalid platform skill: name={name!r} url={url!r}")

            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise tornado.web.HTTPError(
                            502, f"Skill fetch name={name!r} url={url!r}: HTTP {resp.status}",
                        )
                    content = await resp.read()
            except aiohttp.ClientError as exc:
                raise tornado.web.HTTPError(
                    502, f"Skill fetch name={name!r} url={url!r}: {exc}",
                ) from exc

            if skill_type == "MD":
                zip_bytes = _md_to_skill_zip(name, content.decode("utf-8"))
            elif skill_type == "ZIP":
                zip_bytes = content
            else:
                raise tornado.web.HTTPError(
                    502, f"Unsupported skill type name={name!r} type={skill_type!r}",
                )

            resolved.append(base64.b64encode(zip_bytes).decode("ascii"))
    return resolved


# ── platform HTTP fetch ───────────────────────────────────────────────

async def fetch_agent_from_platform(
    agent_id: str, partner_id: int, version: int,
) -> Dict[str, Any]:
    """Fetch an agent definition from the external platform sync service.

    Returns the platform response dict containing ``"agent"``, ``"tasks"``,
    and optionally ``"skills"`` keys.
    """
    base_url = os.environ.get(AGENT_SYNC_SERVICE_URL)
    if not base_url:
        raise tornado.web.HTTPError(
            500,
            f"Environment variable '{AGENT_SYNC_SERVICE_URL}' is not configured",
        )

    params = {
        "agentId": agent_id,
        "partnerId": str(partner_id),
        "version": version,
    }

    logger.info("Fetching agent from platform | url=%s agentId=%s", base_url, agent_id)

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            async with session.get(base_url, params=params) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise tornado.web.HTTPError(
                        502,
                        f"Platform fetch returned {resp.status}: {body}",
                    )
                data = await resp.json()
    except aiohttp.ClientError as exc:
        raise tornado.web.HTTPError(
            502, f"Failed to reach platform service: {exc}",
        ) from exc

    if "agent" not in data or "tasks" not in data:
        raise tornado.web.HTTPError(
            502,
            "Platform response missing 'agent' or 'tasks'",
        )

    return data


# ── agent construction & persistence ──────────────────────────────────

async def build_and_persist_agent(
    mongo_client: Any,
    platform_response: Dict[str, Any],
    agent_id: str,
    partner_id: int,
    version: int,
) -> Dict[str, Any]:
    """Transform a platform response into an internal agent doc and persist to Mongo."""
    agent_info = platform_response["agent"]
    tasks_info: List[Dict[str, Any]] = platform_response["tasks"]

    agent_name = agent_info.get(NAME, "")
    agent_description = agent_info.get("detail")
    if agent_description is None:
        agent_description = ""
    agent_type = agent_info.get("agentType") or agent_info.get("agent_type") or ""

    skills_zip = await resolve_platform_skills(platform_response.get(PLATFORM_SKILLS_ZIPPED) or [])

    task_docs: List[Dict[str, Any]] = []
    router_task_doc: Optional[Dict[str, Any]] = None

    for task_data in tasks_info:
        is_router = parse_bool(task_data.get("router", False))
        task_type = task_data.get("type", PLATFORM_TASK_TYPE_CUSTOM)
        enabled = task_data.get(ENABLED, True)

        task_doc = build_task_doc(task_data, task_type, enabled)
        if skills_zip:
            task_doc[SKILLS_ZIP] = skills_zip

        if is_router:
            router_task_doc = task_doc
        else:
            task_docs.append(task_doc)

    agent_doc: Dict[str, Any] = {
        NAME: agent_name,
        DESCRIPTION: agent_description,
        AGENT_ID: agent_id,
        AGENT_TYPE: agent_type,
        TASKS: task_docs,
        TASK_AS_ROUTER: router_task_doc,
        SWARM_TYPE: DEFAULT_SWARM_TYPE,
        PARTNER_ID: partner_id,
        VERSION: version,
    }

    inserted_id = await mongo_client._insert_one(AGENT_COLLECTION, agent_doc, partner_id=partner_id)
    if not inserted_id:
        raise AgentBuilderStoreError(
            f"Failed to insert synced agent: agentId={agent_id} version={version}",
        )
    agent_doc["_id"] = str(inserted_id)

    logger.info(
        "Agent synced successfully | agentId=%s partnerId=%s version=%s mongo_id=%s",
        agent_id, partner_id, version, inserted_id,
    )
    return agent_doc


async def fetch_and_persist_agent_from_platform(
    mongo_client: Any,
    agent_id: str,
    partner_id: int,
    version: int,
) -> Dict[str, Any]:
    """Fetch from platform and persist in one call — convenience wrapper for fallback use."""
    platform_response = await fetch_agent_from_platform(agent_id, partner_id, version)
    return await build_and_persist_agent(
        mongo_client, platform_response, agent_id, partner_id, version,
    )
