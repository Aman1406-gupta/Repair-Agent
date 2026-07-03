from repair_agent.models.patch import Patch

class PatchAndValidationNode:
    """
    Node responsible for creating branch, patching generated repairs and
    validating the repair items by running their associated tests,
    also responsible for removing patches for failing test cases after patching.
    """

    def __init__(self, validation_tool, git_tool, file_tool):
        self.validation_tool = validation_tool
        self.git_tool = git_tool
        self.file_tool = file_tool

    async def __call__(self, state):

        for group in state["repair_groups"]:

            self.git_tool.checkout_branch.invoke(group.branch_name)

            repair_patches: list[Patch] = []
            remove_patches: list[Patch] = []

            for item in state["repair_items"]:

                if item.is_infrastructure or not item.is_reproducible or item.target_to_repair == "SERVICE":
                    continue

                repair_patch = Patch(
                    methodName= item.test_document.methodName,
                    start_line= item.test_document.start_line,
                    end_line= item.test_document.end_line,
                    replacement= item.repair_patch,
                    original= item.test_source_code
                )

                repair_patches.append(repair_patch)

            updated_patches = self.file_tool.replace_lines.invoke(
                {
                    "file_path": group.group_key,
                    "repair_patches": repair_patches,
                }
            )

            failed_validations: list[str] = []

            for item in state["repair_items"]:

                validation_result = await self.validation_tool.run_test.ainvoke(
                    {
                        "class_name": item.test_document.className,
                        "method_name": item.test_document.methodName
                    }
                )

                post_repair_git_diff = self.git_tool.post_repair_git_diff.invoke(file_path=item.test_document.testCaseFilePath)

                item.post_repair_git_diff = post_repair_git_diff

                item.validation_passed = validation_result

                if not validation_result:

                    failed_validations.append(item.test_document.methodName)

            for updated_patch in updated_patches:

                if updated_patch.methodName in failed_validations:

                    remove_patches.append(updated_patch)

            self.file_tool.remove_lines.invoke(
                {
                    "file_path": group.group_key,
                    "remove_patches": remove_patches
                }
            )

            self.git_tool.commit_all.invoke(group.commit_message, group.group_key)

        return state