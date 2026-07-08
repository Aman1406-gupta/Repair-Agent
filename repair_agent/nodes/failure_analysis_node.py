import json
from repair_agent.state import RepairState

class FailureAnalysisNode:
    """
    Node responsible for performing failure analysis on repair items.
    """

    def __init__(self, failure_analysis_task):
        self.failure_analysis_task = failure_analysis_task

    async def __call__(self, state: RepairState):

        for item in state["repair_items"]:

            try:

                response = await self.failure_analysis_task.ainvoke(
                    {
                        "testID": item.test_document.testID,
                        "className": item.test_document.className,
                        "methodName": item.test_document.methodName,
                        "moduleName": item.test_document.moduleName,
                        "repositoryUrl": item.test_document.repositoryUrl,
                        "ref": item.test_document.currentCommitSha,
                        "test_source_code": item.test_source_code,
                        "pre_repair_git_diff": item.pre_repair_git_diff,
                        "errorMessage": item.test_document.errorMessage,
                        "stackTrace": item.test_document.stackTrace,
                    }
                )

                print(response)

                analysis = json.loads(response.content)

                if (
                    "is_infrastructure" not in analysis or
                    "is_reproducible" not in analysis or
                    "target_to_repair" not in analysis or
                    "service_file_path" not in analysis or
                    "service_start_line" not in analysis or
                    "service_end_line" not in analysis or
                    "root_cause_explanation" not in analysis
                ):
                    raise ValueError(
                        "Failure analysis response is missing required fields."
                )

                item.failure_analysis = analysis

                item.is_infrastructure = analysis["is_infrastructure"]
                item.is_reproducible = analysis["is_reproducible"]

                item.target_to_repair = analysis["target_to_repair"]

                item.service_file_path = analysis["service_file_path"]
                item.service_start_line = analysis["service_start_line"]
                item.service_end_line = analysis["service_end_line"]

                item.root_cause = analysis["root_cause_explanation"]

            except Exception as e:

                item.error = str(e)

            break

        print("Failure analysis node completed")

        return state