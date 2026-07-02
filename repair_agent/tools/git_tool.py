import subprocess
from langchain_core.tools import tool


class GitTool:
    """Wrapper around local git commands."""

    @staticmethod
    def _run(*command: str) -> str:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    @tool
    def get_diff(self, revision: str = "HEAD~1") -> str:
        """
        Returns the git diff for the specified revision.
        """
        return self._run("git", "diff", revision)

    @tool
    def checkout_branch(self, branch_name: str, create: bool = True) -> str:
        """
        Checkout an existing branch or create a new branch.
        """
        if create:
            self._run("git", "checkout", "-b", branch_name)
        else:
            self._run("git", "checkout", branch_name)

        return f"Checked out {branch_name}"

    @tool
    def commit_all(self, commit_message: str) -> str:
        """
        Stage all files and create a commit.
        """
        self._run("git", "add", ".")
        self._run("git", "commit", "-m", commit_message)

        return "Commit created."

    @tool
    def push_branch(self, branch_name: str, remote: str = "origin") -> str:
        """
        Push current branch to remote.
        """
        self._run("git", "push", remote, branch_name)

        return f"Pushed {branch_name}"

    @tool
    def current_branch(self) -> str:
        """
        Returns the current git branch.
        """
        return self._run(
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        )