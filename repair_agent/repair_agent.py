from repair_agent.graph import build_graph

from repair_agent.task_factory import TaskFactory

from repair_agent.nodes.repository_context import RepositoryContextNode
from repair_agent.nodes.failure_analysis import FailureAnalysisNode
from repair_agent.nodes.repair_generation import RepairGenerationNode
from repair_agent.nodes.validation import ValidationNode
from repair_agent.nodes.pull_request import PullRequestNode

from repair_agent.tools.git_tool import GitTool
from repair_agent.tools.github_tool import GitHubTool
from repair_agent.tools.github_pr_tool import GitHubPRTool
from repair_agent.tools.gradle_tool import GradleTool
from repair_agent.tools.file_tool import FileTool


class RepairAgent:

    def __init__(self):

        # ---------- Tools ----------

        self.git_tool = GitTool()

        self.github_tool = GitHubTool()

        self.github_pr_tool = GitHubPRTool()

        self.gradle_tool = GradleTool()

        self.file_tool = FileTool()

        # ---------- Tasks ----------

        self.failure_analysis_task = TaskFactory.create_failure_analysis_task(
            tools=[
                self.github_tool.fetch_file_lines,
                self.git_tool.get_diff,
            ],
        )

        self.repair_task = TaskFactory.create_repair_task(
            tools=[
                self.file_tool.read_file,
                self.file_tool.replace_lines,
            ],
        )

        self.pull_request_task = TaskFactory.create_pull_request_task(
            tools=[],
        )

        # ---------- Nodes ----------

        repository_context = RepositoryContextNode(
            self.github_tool,
            self.git_tool,
        )

        failure_analysis = FailureAnalysisNode(
            self.failure_analysis_task,
            self.github_tool,
        )

        repair_generation = RepairGenerationNode(
            self.repair_task,
            self.file_tool,
            self.git_tool,
        )

        validation = ValidationNode(
            self.gradle_tool,
        )

        pull_request = PullRequestNode(
            self.pull_request_task,
            self.git_tool,
            self.github_pr_tool,
        )

        self.graph = build_graph(
            repository_context,
            failure_analysis,
            repair_generation,
            validation,
            pull_request,
        )

    async def ainvoke(self, state):
        return await self.graph.ainvoke(state)

    # def invoke(self, state):
    #     return self.graph.invoke(state)