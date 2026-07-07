from repair_agent.models.repair_item import RepairItem
from repair_agent.state import RepairState

class RepositoryContextNode:
    """
    Builds repository context for every failing test.
    """

    def __init__(self, github_tool, git_tool):
        self.github_tool = github_tool
        self.git_tool = git_tool

    async def __call__(self, state: RepairState) -> RepairState:

        repair_items = []
        diff_cache = {}
        checkout_cache = set()

        for test_doc in state["test_documents"]:

            repo = test_doc.repositoryUrl
            current_commit = test_doc.currentCommitSha
            file_path = test_doc.testCaseFilePath

            checkout_key = (repo, current_commit)

            if checkout_key not in checkout_cache:
                self.git_tool.checkout_commit(
                    repository_url=repo,
                    commit_sha=current_commit,
                )
                checkout_cache.add(checkout_key)

            cache_key = (repo, current_commit, file_path)

            if cache_key not in diff_cache:
                diff_cache[cache_key] = self.git_tool.pre_repair_git_diff(
                    repo,
                    current_commit,
                    file_path
                )

            test_source = self.github_tool.fetch_file_lines(
                repository_url= repo,
                file_path= test_doc.testCaseFilePath,
                start_line= test_doc.startLine,
                end_line= test_doc.endLine,
                ref= current_commit,
            )

            item = RepairItem(
                test_document=test_doc,
                test_source_code=test_source,
                pre_repair_git_diff=diff_cache[cache_key],
            )

            repair_items.append(item)

        state["repair_items"] = repair_items

        return state