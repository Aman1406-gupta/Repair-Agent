import subprocess

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

    def pre_repair_git_diff(self, current_commit_sha: str, previous_commit_sha: str, file_path: str) -> str:
        """
        Returns the git diff for the specified revision.
        """
        return self._run("git", "diff", previous_commit_sha, current_commit_sha, "--", file_path)

    def post_repair_git_diff(self, file_path: str) -> str:
        """
        Returns the git diff for the specified file after repair.
        """
        return self._run("git", "diff", "--", file_path)

    def checkout_branch(self, branch_name: str, create: bool = True):
        """
        Checkout an existing branch or create a new branch.
        """
        if create:
            self._run("git", "checkout", "-b", branch_name)
        else:
            self._run("git", "checkout", branch_name)

    def commit_all(self, commit_message: str, file_path: str):
        """
        Stage all files and create a commit.
        """
        self._run("git", "add", file_path)
        self._run("git", "commit", "-m", commit_message)

    def push_branch(self, branch_name: str, remote: str = "origin"):
        """
        Push current branch to remote.
        """
        self._run("git", "push", remote, branch_name)

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