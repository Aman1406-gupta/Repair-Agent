class PullRequestNode:
    """
    Creates a commit, pushes the branch and opens a Pull Request.
    """

    def __init__(
            self,
            pull_request_task,
            git_tool,
            github_pr_tool,
    ):
        self.pull_request_task = pull_request_task
        self.git_tool = git_tool
        self.github_pr_tool = github_pr_tool

    async def __call__(self, state):

        state = await self.pull_request_task.ainvoke(state)

        description = state["messages"][-1].content

        test_id = state["test_document"]["testID"]

        branch = f"repair-{test_id}"

        self.git_tool.commit_all.invoke(
            {
                "commit_message": f"Fix failing test {test_id}"
            }
        )

        self.git_tool.push_branch.invoke(
            {
                "branch_name": branch
            }
        )

        current_branch = self.git_tool.current_branch.invoke({})

        pr_url = self.github_pr_tool.create_pull_request.invoke(
            {
                "repository_url": state["test_document"]["repositoryUrl"],
                "source_branch": branch,
                "target_branch": current_branch,
                "title": f"Automated fix for {test_id}",
                "description": description,
            }
        )

        state["pr_url"] = pr_url

        return state