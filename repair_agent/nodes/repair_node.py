import json

from repair_agent.state import RepairState

class RepairNode:
    """
    Fetch Service source code and generate repair patches only for test repairs.
    Service repairs are skipped because we generate a skeleton PR for that.
    """

    def __init__(self, github_tool, repair_task):
        self.github_tool = github_tool
        self.repair_task = repair_task

    async def __call__(self, state: RepairState) -> RepairState:

        for item in state["repair_items"]:

            if item.is_infrastructure or not item.is_reproducible or item.target_to_repair == "SERVICE":
                continue

            service_source = self.github_tool.fetch_file(
                repository_url= item.test_document.repositoryUrl,
                file_path= item.service_file_path,
                ref= item.test_document.currentCommitSha,
            )

            item.service_source_code = service_source

            result = await self.repair_task.ainvoke(
                {
                    "test_document": item.test_document,
                    "target_to_repair": item.target_to_repair,
                    "root_cause": item.root_cause,
                    "test_source_code": item.test_source_code,
                    "service_source_code": item.service_source_code,
                    "pre_repair_git_diff": item.pre_repair_git_diff,
                }
            )

            response= result["messages"][-1].content
            patch = json.loads(response)

            item.repair_patch= patch["generated_patch"]

        return state