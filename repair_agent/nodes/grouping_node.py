from repair_agent.state import RepairState
from repair_agent.models.repair_group import RepairGroup

class GroupingNode:
    """
    Node responsible for grouping the repair items.
    Grouping rules:

    TEST target
        -> group by testCaseFilePath

    SERVICE target
        -> group by serviceFilePath
    """

    async def __call__(self, state: RepairState) -> RepairState:

        groups = {}

        for item in state["repair_items"]:

            if item.is_infrastructure or not item.is_reproducible:

                continue

            if item.target_to_repair == "TEST":

                group_key = item.test_document.testCaseFilePath
                is_skeleton_pr = False

            else:

                group_key = item.service_file_path
                is_skeleton_pr = True

            if group_key not in groups:

                file_name = group_key.split("/")[-1].replace(".java", "")

                groups[group_key] = RepairGroup(
                    group_key=group_key,
                    branch_name=f"ai-repair/{file_name}",
                    commit_message=f"AI repair: {file_name}",
                    pr_title=f"AI Repair: {file_name}",
                    repair_items=[],
                    is_skeleton_pr=is_skeleton_pr,
                    owner=item.test_document.lastModifiedBy,
                    repository_url=item.test_document.repositoryUrl,
                )

            groups[group_key].repair_items.append(item)

        state["repair_groups"] = groups

        return state