import json
import uuid
from agent_builder.base.state import get_initial_state
from repair_agent.state import RepairState


class FailureAnalysisNode:
    def __init__(self, failure_analysis_task):
        self.failure_analysis_task = failure_analysis_task

    async def __call__(self, state: RepairState):

        for item in state["repair_items"]:
            try:
                content = f"""Test ID: {item.test_document.testID}
                        Class Name: {item.test_document.className}
                        Method Name: {item.test_document.methodName}
                        Module Name: {item.test_document.moduleName}
                        Repository URL: {item.test_document.repositoryUrl}
                        Ref (branch/commit): {item.test_document.currentCommitSha}
                        
                        Test Source Code:
                        {item.test_source_code}
                        
                        Git Diff Before Repair:
                        {item.pre_repair_git_diff}
                        
                        Failure Message:
                        {item.test_document.errorMessage}
                        
                        Stack Trace:
                        {item.test_document.stackTrace}
                        """

                task_state = get_initial_state(session_id=str(uuid.uuid4()))
                task_state["messages"] = [{"role": "user", "content": content}]

                response = await self.failure_analysis_task.ainvoke(task_state)

                ai_message = response["messages"][-1].content
                analysis = json.loads(ai_message)

                print(analysis)

                required = [
                    "is_infrastructure", "is_reproducible", "target_to_repair",
                    "service_file_path", "service_method", "root_cause_explanation",
                ]
                missing = [f for f in required if f not in analysis]
                if missing:
                    raise ValueError(f"Failure analysis response missing fields: {missing}")

                item.failure_analysis = analysis
                item.is_infrastructure = analysis["is_infrastructure"]
                item.is_reproducible = analysis["is_reproducible"]
                item.target_to_repair = analysis["target_to_repair"]
                item.service_file_path = analysis["service_file_path"]
                item.service_method = analysis["service_method"]
                item.root_cause = analysis["root_cause_explanation"]

            except Exception as e:
                item.error = str(e)

        print("Failure analysis node completed")
        return state