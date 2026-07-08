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

        for group in state["repair_groups"].values():

            self.git_tool.checkout_branch(
                repository_url=group.repository_url,
                branch_name=group.branch_name
            )

            repair_patches: list[Patch] = []
            remove_patches: list[Patch] = []

            for item in state["repair_items"]:

                if item.is_infrastructure or not item.is_reproducible or item.target_to_repair == "SERVICE":
                    continue

                repair_patch = Patch(
                    methodName= item.test_document.methodName,
                    start_line= item.test_document.startLine,
                    end_line= item.test_document.endLine,
                    replacement= item.repair_patch,
                    original= item.test_source_code
                )

                repair_patches.append(repair_patch)

            updated_patches = self.file_tool.replace_lines(
                file_path= group.group_key,
                repair_patches= repair_patches,
            )

            failed_validations: list[str] = []

            for item in group.repair_items:

                validation_result = self.validation_tool.run_test(
                    test_class= item.test_document.className,
                    test_method= item.test_document.methodName
                )

                post_repair_git_diff = self.git_tool.post_repair_git_diff(
                    repository_url=group.repository_url,
                    file_path=item.test_document.testCaseFilePath
                )

                item.post_repair_git_diff = post_repair_git_diff

                item.validation_passed = validation_result

                if not validation_result:

                    failed_validations.append(item.test_document.methodName)

                else:
                    if item.test_document.lastModifiedBy not in group.owners:
                        group.owners.append(item.test_document.lastModifiedBy)

            for updated_patch in updated_patches:

                if updated_patch.methodName in failed_validations:

                    remove_patches.append(updated_patch)
                    print(f"Repair not commited for" + updated_patch.methodName + "due to Validation failure")

            self.file_tool.remove_lines(
                file_path= group.group_key,
                remove_patches= remove_patches
            )

            self.git_tool.commit(
                repository_url=group.repository_url,
                file_path=group.group_key,
                commit_message=group.commit_message
            )

        print("Patch and Validation node completed")

        return state