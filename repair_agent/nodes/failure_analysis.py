import json

from langchain_core.messages import AIMessage

from repair_agent.state import RepairState


class FailureAnalysisNode:
    """
    Executes the FailureAnalysis ThinkAct task and updates the RepairState.
    """

    def __init__(self, failure_analysis_task, github_tool):
        self.failure_analysis_task = failure_analysis_task
        self.github_tool = github_tool

    async def __call__(self, state: RepairState) -> RepairState:

        state = await self.failure_analysis_task.ainvoke(state)

        analysis = self._extract_analysis(state)

        state["analysis_result"] = analysis

        state["is_infrastructure"] = analysis["is_infrastructure"]
        state["is_reproducible"] = analysis["is_reproducible_guess"]
        state["target_to_repair"] = analysis["target_to_repair"]
        state["target_file_path"] = analysis["target_file_path"]
        state["target_start_line"] = analysis["target_start_line"]
        state["target_end_line"] = analysis["target_end_line"]

        if not state["is_infrastructure"] and state["is_reproducible"]:
            if state["target_to_repair"].equals("SERVICE"):
                state["target_source_code"] = (
                    self.github_tool.fetch_file_lines.invoke(
                        {
                            "repository_url": state["test_document"]["repositoryUrl"],
                            "file_path": analysis["target_file_path"],
                            "start_line": analysis["target_start_line"],
                            "end_line": analysis["target_end_line"],
                            "ref": state["test_document"]["currentCommitSha"],
                        }
                    )
                )
            else:
                state["target_source_code"] = state["test_source_code"]

        return state

    def _extract_analysis(self, state: RepairState) -> dict:
        """
        Extracts the final AI JSON response from the conversation.
        """

        ai_message = None

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                ai_message = message
                break

        if ai_message is None:
            raise ValueError("FailureAnalysisTask did not produce an AI response.")

        content = ai_message.content.strip()

        if content.startswith("```json"):
            content = content[7:]

        if content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON returned by FailureAnalysisTask:\n{content}"
            ) from e

        required_fields = [
            "is_infrastructure",
            "is_reproducible_guess",
            "target_to_repair",
            "target_file_path",
            "target_start_line",
            "target_end_line",
        ]

        missing = [
            field
            for field in required_fields
            if field not in analysis
        ]

        if missing:
            raise ValueError(
                f"Failure analysis response missing fields: {missing}"
            )

        return analysis