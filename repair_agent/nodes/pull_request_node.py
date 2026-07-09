class PullRequestNode:
    """
    Pushes the branch and opens a Pull Request.
    If grouped by serviceFilePath -> generate skeleton pr
    If grouped by testCaseFilePath -> generate pr corresponding to test cases where repair passed the validation
    """

    def __init__(
            self,
            pull_request_description_task,
            skeleton_pull_request_description_task,
            git_tool,
            github_tool,
    ):
        self.pull_request_description_task = pull_request_description_task
        self.skeleton_pull_request_description_task = skeleton_pull_request_description_task
        self.git_tool = git_tool
        self.github_tool = github_tool

    async def __call__(self, state):

        pr_urls = []

        for group in state["repair_groups"].values():

            if group.is_skeleton_pr:

                affected_items =[
                    {
                        "test_id": item.test_document.testID,
                        "test_name": item.test_document.methodName,
                        "test_file": item.test_document.testCaseFilePath,
                        "service_file": item.service_file_path,
                        "service_start_line": item.service_start_line,
                        "service_end_line": item.service_end_line,
                        "root_cause": item.root_cause,
                    }
                    for item in group.repair_items
                ]

                pr_desc = await self.skeleton_pull_request_description_task.ainvoke(
                    {
                        "affected_items": affected_items
                    }
                )

                group.pr_description = pr_desc

                if group.owners:
                    reviewers = "\n".join(f"-@{owner}" for owner in group.owners)
                    group.pr_description += (
                        "\n\n## Suggested Reviewers\n"
                        f"{reviewers}"
                    )

            if not group.is_skeleton_pr:

                repair_items = [
                    {
                        "test_id": item.test_document.testID,
                        "test_name": item.test_document.methodName,
                        "test_file": item.test_document.testCaseFilePath,
                        "service_file": item.service_file_path,
                        "root_cause": item.root_cause,
                        "target_to_repair": item.target_to_repair,
                        "repair_patch": item.repair_patch,
                        "validation_passed": item.validation_passed,
                        "git_diff": item.post_repair_git_diff,
                    }
                    for item in group.repair_items if item.validation_passed == True
                ]

                pr_desc = await self.pull_request_description_task.ainvoke(
                    {
                        "repair_items": repair_items
                    }
                )

                group.pr_description = pr_desc

            self.git_tool.push_branch(
                repository_url= group.repository_url,
                branch_name=group.branch_name,
            )

            pr_url = self.github_tool.create_pull_request(
                repository_url= group.repository_url,
                source_branch= group.branch_name,
                target_branch= self.git_tool.current_branch(group.repository_url),
                title= group.pr_title,
                description= group.pr_description,
            )

            pr_urls.append(pr_url)

        state["pr_urls"] = pr_urls

        print("Pull request node completed")

        return state