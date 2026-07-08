import subprocess

from langchain_core.tools import tool

from storage.utils.mongo_topology import tool_requests_to_metadata


class ValidationTool:
    """Wrapper around Gradle test execution."""

    @staticmethod
    def _run(*command: str) -> bool:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return process.returncode == 0

    def run_test(self, test_class: str, test_method: str) -> bool:
        """
        Execute a single test method.

        Example:
        UserServiceTest.testCreateUser
        """
        target = f"{test_class}.{test_method}"

        return self._run(
            "./gradlew",
            "test",
            "--tests",
            target,
        )

    @tool
    def run_test_tool(self, test_class: str, test_method: str) -> bool:
        """
        Execute a single test method.

        Example:
        UserServiceTest.testCreateUser
        """

        return self.run_test(
            test_class= test_class,
            test_method= test_method,
        )