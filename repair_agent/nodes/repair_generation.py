from repair_agent.state import RepairState


class RepairGenerationNode:
    """
    Executes the RepairTask and applies the generated patch to the repository.
    """

    def __init__(
            self,
            repair_task,
            file_tool,
            git_tool,
    ):
        self.repair_task = repair_task
        self.file_tool = file_tool
        self.git_tool = git_tool

    async def __call__(self, state: RepairState):

        state = await self.repair_task.ainvoke(state)

        analysis = state["analysis_result"]

        file_path = analysis["target_file_path"]

        self.git_tool.checkout_branch.invoke(
            {
                "branch_name": f"repair-{state['test_document']['testID']}"
            }
        )

        self.file_tool.replace_lines.invoke(
            {
                "file_path": file_path,
                "start_line": analysis["repair_start_line"],
                "end_line": analysis["repair_end_line"],
                "replacement": state["generated_patch"],
            }
        )

        state["patch_file_path"] = file_path

        return state