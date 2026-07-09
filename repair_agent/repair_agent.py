from repair_agent.graph import build_graph
from repair_agent.task_factory import TaskFactory
from repair_agent.nodes.repository_context_node import RepositoryContextNode
from repair_agent.nodes.failure_analysis_node import FailureAnalysisNode
from repair_agent.nodes.repair_node import RepairNode
from repair_agent.nodes.grouping_node import GroupingNode
from repair_agent.nodes.patch_and_validation_node import PatchAndValidationNode
from repair_agent.nodes.pull_request_node import PullRequestNode
from repair_agent.tools.git_tool import GitTool
from repair_agent.tools.github_tool import GitHubTool
from repair_agent.tools.validation_tool import ValidationTool
from repair_agent.tools.file_tool import FileTool


class RepairAgent:

    def __init__(self):

        # ---------- Tools ----------

        self.git_tool = GitTool()

        self.github_tool = GitHubTool()

        self.validation_tool = ValidationTool()

        self.file_tool = FileTool()

        # ---------- Tasks ----------

        self.failure_analysis_task = TaskFactory.create_failure_analysis_task(
            tools=[
                *self.github_tool.as_langchain_tools(),
                *self.validation_tool.as_langchain_tools(),
            ],
        )

        self.repair_task = TaskFactory.create_repair_task(
            tools=[
                *self.file_tool.as_langchain_tools(),
            ],
        )

        self.pull_request_task = TaskFactory.pull_request_description_task(
            tools=[],
        )

        self.skeleton_pull_request_task = TaskFactory.skeleton_pull_request_description_task(
            tools=[],
        )

        # ---------- Nodes ----------

        repository_context = RepositoryContextNode(
            self.github_tool,
            self.git_tool,
        )

        failure_analysis = FailureAnalysisNode(
            self.failure_analysis_task,
        )

        repair_generation = RepairNode(
            self.github_tool,
            self.repair_task,
        )

        grouping = GroupingNode()

        patch_and_validation = PatchAndValidationNode(
            self.validation_tool,
            self.git_tool,
            self.file_tool,
        )

        pull_request = PullRequestNode(
            self.pull_request_task,
            self.skeleton_pull_request_task,
            self.git_tool,
            self.github_tool,
        )

        self.graph = build_graph(
            repository_context,
            failure_analysis,
            repair_generation,
            grouping,
            patch_and_validation,
            pull_request,
        )

    async def ainvoke(self, state):
        return await self.graph.ainvoke(state)