from __future__ import annotations

import logging
from typing import Any, get_type_hints
from dataclasses import fields, is_dataclass
from bson import ObjectId

from agent_builder.base.tools import openapi_metadata_to_tool
from agent_builder.base.tools_mock import create_mock_tool_from_metadata
from agent_builder.base.configs import TaskConfig, AgentConfig
from agent_builder.base.task import Task
from agent_builder.base.agent import Agent
from agent_builder.prebuilt_tasks.remote_release import RemoteRelease

logger = logging.getLogger(__name__)


def mongo_to_dataclass(doc: dict, cls):
    """Convert a Mongo document into a dataclass instance, handling nested dataclasses."""
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass type")

    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])

    kwargs = {}
    for f in fields(cls):
        if f.name in doc:
            val = doc[f.name]
            if val is None and is_dataclass(f.type):
                continue
            if is_dataclass(f.type) and isinstance(val, dict):
                kwargs[f.name] = mongo_to_dataclass(val, f.type)
            elif f.type is int:
                if val is None or val == "":
                    kwargs[f.name] = 0
                elif isinstance(val, str):
                    kwargs[f.name] = int(val) if val.strip() else 0
                else:
                    kwargs[f.name] = int(val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def create_tool_from_metadata(meta: dict):
    """Create a single tool from metadata based on toolType."""

    tool_type = meta.get("toolType", "api_tool")

    if tool_type == "api_tool":
        return openapi_metadata_to_tool(meta)
    elif tool_type == "prompt_tool":
        return create_mock_tool_from_metadata(meta)
    else:
        raise ValueError(f"Unknown toolType: {tool_type}")


def build_tools_from_metadata_list(tools_metadata: list) -> list:
    """Convert a list of stored tool metadata dicts into StructuredTool instances."""
    return [create_tool_from_metadata(meta) for meta in tools_metadata]


def build_task_from_doc(task_doc: dict, memory=None, extra_tools_by_task=None, session_id: str | None = None):
    task_type = task_doc.get("task_type", "normal")
    task_name = task_doc.get("name", "<unknown>")
    logger.debug("Building task | name=%s type=%s (diversion: %s)",
                 task_name, task_type,
                 "release/remote" if task_type in ("release", "remote", "remote_release") else "normal")

    if task_type == "release":
        return RemoteRelease(release_doc=task_doc, memory=memory)


    task_config = mongo_to_dataclass(task_doc, TaskConfig)
    tools_metadata = task_doc.get("tools", [])
    tools = build_tools_from_metadata_list(tools_metadata)

    task_as_tools_list = task_doc.get("task_as_tools", [])
    for embedded_task_doc in task_as_tools_list:
        referenced_task = build_task_from_doc(embedded_task_doc, memory=memory, extra_tools_by_task=extra_tools_by_task, session_id=session_id)
        task_tool = referenced_task.as_tool()
        tools.append(task_tool)

    agent_as_tools_list = task_doc.get("agent_as_tools", [])
    for embedded_agent_doc in agent_as_tools_list:
        referenced_agent = build_agent_from_doc(embedded_agent_doc, memory=memory, extra_tools_by_task=extra_tools_by_task, session_id=session_id)
        agent_as_task = referenced_agent.as_task()
        agent_tool = agent_as_task.as_tool()
        tools.append(agent_tool)

    if task_type == "deep_agent":
        from agent_builder.prebuilt_tasks.deep_agent import DeepAgentsTask
        subagents = task_doc.get("subagents") or None
        skills_zip_b64 = task_doc.get("skills_zip") or None
        task = DeepAgentsTask(
            task_config=task_config,
            tools=tools,
            handoffs=[],
            memory=memory,
            session_id=session_id,
            subagents=subagents,
            skills_zip_b64=skills_zip_b64,
        )
        task._tools_metadata = tools_metadata
        return task

    task = Task(task_config=task_config, tools=tools, handoffs=[], memory=memory)
    task._tools_metadata = tools_metadata
    return task


def build_agent_from_doc(agent_doc: dict, memory=None, extra_tools_by_task=None, session_id: str | None = None):

    logger.debug("Building agent from doc | name=%s tasks=%d", agent_doc.get("name"), len(agent_doc.get("tasks", [])))
    agent_config = mongo_to_dataclass(agent_doc, AgentConfig)
    tasks_raw = [t for t in agent_doc.get("tasks", []) if t.get("enabled", True)]
    tasks_list = [build_task_from_doc(t, memory, extra_tools_by_task=extra_tools_by_task, session_id=session_id) for t in tasks_raw]

    if extra_tools_by_task:
        for task in tasks_list:
            task_id = str(getattr(task.task_config, '_id', ''))
            extra = extra_tools_by_task.get(task_id, [])
            if extra:
                task.add_tools(extra)

    agent_as_task_list = agent_doc.get("agent_as_task", [])
    if agent_as_task_list:
        for embedded_agent_doc in agent_as_task_list:
            referenced_agent = build_agent_from_doc(embedded_agent_doc, memory=memory, extra_tools_by_task=extra_tools_by_task, session_id=session_id)
            agent_task = referenced_agent.as_task(task_id=embedded_agent_doc.get("_id"))
            tasks_list.append(agent_task)

    router_doc = agent_doc.get("task_as_router")
    use_task_as_router = (
        build_task_from_doc(router_doc, memory=memory, extra_tools_by_task=extra_tools_by_task, session_id=session_id)
        if router_doc and router_doc.get("enabled", True)
        else None
    )

    return Agent(agent_config=agent_config, tasks=tasks_list, use_task_as_router=use_task_as_router, memory=memory)
