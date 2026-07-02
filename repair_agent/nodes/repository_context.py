from repair_agent.state import RepairState


class RepositoryContextNode:
    """
    Retrieves repository context required for downstream analysis.
    """

    def __init__(self, github_tool, git_tool):
        self.github_tool = github_tool
        self.git_tool = git_tool

    async def __call__(self, state: RepairState) -> RepairState:
        test_doc = state["test_document"]

        test_source = self.github_tool.fetch_file_lines.invoke(
            {
                "repository_url": test_doc["repositoryUrl"],
                "file_path": test_doc["testCaseFilePath"],
                "start_line": test_doc["startLine"],
                "end_line": test_doc["endLine"],
                "ref": state["test_document"]["currentCommitSha"],
            }
        )

        git_diff = self.git_tool.get_diff.invoke(
            {
                "revision": "HEAD~1"
            }
        )

        state["test_source_code"] = test_source
        state["git_diff"] = git_diff

        return state