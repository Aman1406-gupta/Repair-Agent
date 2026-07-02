from repair_agent.state import RepairState


class ValidationNode:
    """
    Validates the generated repair by executing Gradle tests.
    """

    def __init__(self, gradle_tool):
        self.gradle_tool = gradle_tool

    async def __call__(self, state: RepairState):

        test_doc = state["test_document"]

        passed = self.gradle_tool.run_test.invoke(
            {
                "test_class": test_doc["className"],
                "test_method": test_doc["methodName"],
            }
        )

        state["validation_passed"] = passed

        return state