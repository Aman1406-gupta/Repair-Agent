from typing import Dict, Any, Optional, List

from bson import ObjectId

from agent_builder.handlers.core.requests import (
    RegisterTaskRequest,
    RegisterReleaseTaskRequest,
    RegisterAgentRequest,
)

from agent_builder.storage.mongo_client import AgentBuilderMongoStore
from agent_builder.utils.constants import AGENT_COLLECTION
from agent_builder.storage.utils.builders import (
    build_agent_from_doc,
    build_task_from_doc,
    create_tool_from_metadata,
)
from agent_builder.base.task import Task
from agent_builder.base.agent import Agent
from langchain_core.tools import BaseTool


class InMemoryRegistry:
    def __init__(self, mongo_client: Optional[AgentBuilderMongoStore] = None):
        if mongo_client is None:
            mongo_client = AgentBuilderMongoStore()
        self.mongo_client = mongo_client
        self._task_registry: Dict[str, Task] = {}
        self._agent_registry: Dict[str, Agent] = {}
        self._tool_registry: Dict[str, Any] = {}

    def get_task(self, name: str) -> Optional[Task]:
        return self._task_registry.get(name)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agent_registry.get(agent_id)

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tool_registry.get(name)

    def put(self, obj: Any) -> str:
        if isinstance(obj, Agent):
            key = getattr(obj.agent_config, "agent_id", None) or getattr(obj.agent_config, "name", None)
            if not key:
                raise ValueError("Agent is missing agent_id and name in agent_config")
            self._agent_registry[key] = obj
            return key

        if isinstance(obj, Task):
            name = getattr(obj.task_config, "name", None)
            if not name:
                raise ValueError("Task is missing a name in task_config")
            self._task_registry[name] = obj
            return name

        if isinstance(obj, BaseTool):
            name = getattr(obj, "name", None)
            if not name:
                raise ValueError("Tool object is missing a name attribute")
            self._tool_registry[name] = obj
            return name

        raise TypeError("Unsupported object type for registry storage")

    async def register_agent(self, agent_id: str, partner_id: int) -> Optional[Agent]:
        if agent_id in self._agent_registry:
            return self._agent_registry[agent_id]

        doc = await self.mongo_client._find_one_cached(
            AGENT_COLLECTION, {"_id": ObjectId(agent_id)},
            partner_id=partner_id,
        )
        if not doc:
            return None

        agent = build_agent_from_doc(doc)
        self._agent_registry[agent_id] = agent
        return agent

    async def agent_to_mongo(
        self,
        name: str,
        partner_id: int,
        agent_type: str,
        tasks: List[RegisterTaskRequest],
        description: Optional[str] = None,
        workflow_edges: Optional[List[tuple]] = None,
        router_model_config: Optional[Dict[str, Any]] = None,
        swarm_type: str = "router_back_connection",
        agent_as_task: Optional[List[RegisterAgentRequest]] = None,
        task_as_router: Optional[RegisterTaskRequest] = None,
    ) -> Dict[str, Any]:
        request = RegisterAgentRequest(
            name=name,
            partner_id=partner_id,
            agent_type=agent_type,
            tasks=tasks,
            description=description,
            workflow_edges=workflow_edges,
            llm_config=router_model_config,
            swarm_type=swarm_type,
            agent_as_task=agent_as_task,
            task_as_router=task_as_router,
        )
        return await self.mongo_client.register_agent(request)

    async def load_from_config(self, config: Dict[str, Any]) -> None:
        if "partner_id" not in config:
            raise ValueError("partner_id is required in registry config")
        partner_id = config["partner_id"]
        for agent_id in config.get("agents", []):
            await self.register_agent(agent_id, partner_id)
