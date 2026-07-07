import os
import shutil
import subprocess
from pathlib import Path


class GitTool:
    """Wrapper around local git operations."""

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is missing.")

        self.workspace = Path("/tmp/repositories")
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _repo_path(self, repository_url: str) -> Path:
        repo_name = repository_url.rstrip("/").replace(".git", "").split("/")[-1]
        return self.workspace / repo_name

    def _authenticated_url(self, repository_url: str) -> str:
        return repository_url.replace(
            "https://",
            f"https://{self.token}@",
        )

    def _ensure_repo(self, repository_url: str) -> Path:
        """
        Clone repository if it is not already present locally, Otherwise fetch the latest references.
        """
        repo_path = self._repo_path(repository_url)

        if not repo_path.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    self._authenticated_url(repository_url),
                    str(repo_path),
                ],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "fetch", "--all"],
                cwd=repo_path,
                check=True,
            )

        return repo_path

    @staticmethod
    def _run(repo_path: Path, *command: str) -> str:
        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def checkout_commit(
            self,
            repository_url: str,
            commit_sha: str,
    ) -> None:

        """
        Checkout the specified commit.
        """
        repo = self._ensure_repo(repository_url)

        self._run(repo, "git", "reset", "--hard")
        self._run(repo, "git", "clean", "-fd")

        self._run(
            repo,
            "git",
            "checkout",
            commit_sha,
        )

    def pre_repair_git_diff(
            self,
            repository_url: str,
            current_commit_sha: str,
            file_path: str,
    ) -> str:
        """
        Returns the diff introduced by the given commit for a specific file.
        """
        repo = self._ensure_repo(repository_url)

        parent_commit = self._run(
            repo,
            "git",
            "rev-parse",
            f"{current_commit_sha}^",
        )

        return self._run(
            repo,
            "git",
            "diff",
            parent_commit,
            current_commit_sha,
            "--",
            file_path,
        )

    def post_repair_git_diff(
            self,
            repository_url: str,
            file_path: str,
    ) -> str:
        """
        Returns the uncommitted diff for a file.
        """
        repo = self._ensure_repo(repository_url)

        return self._run(
            repo,
            "git",
            "diff",
            "--",
            file_path,
        )

    def checkout_branch(
            self,
            repository_url: str,
            branch_name: str,
    ) -> None:
        """
        Checkout or create a branch.
        """
        repo = self._ensure_repo(repository_url)

        branches = self._run(
            repo,
            "git",
            "branch",
            "--list",
            branch_name,
        )

        if branches.strip():
            self._run(
                repo,
                "git",
                "switch",
                branch_name,
            )
        else:
            self._run(
                repo,
                "git",
                "switch",
                "-c",
                branch_name,
            )

    def commit(
            self,
            repository_url: str,
            file_path: str,
            commit_message: str,
    ) -> None:
        """
        Stage a single file and commit it.
        """
        repo = self._ensure_repo(repository_url)

        self._run(
            repo,
            "git",
            "add",
            file_path,
        )

        self._run(
            repo,
            "git",
            "commit",
            "-m",
            commit_message,
        )

    def push_branch(
            self,
            repository_url: str,
            branch_name: str,
    ) -> None:
        """
        Push a branch to origin.
        """
        repo = self._ensure_repo(repository_url)

        self._run(
            repo,
            "git",
            "push",
            "--set-upstream",
            "origin",
            branch_name,
        )

    def current_branch(
            self,
            repository_url: str,
    ) -> str:
        """
        Returns the current branch.
        """
        repo = self._ensure_repo(repository_url)

        return self._run(
            repo,
            "git",
            "branch",
            "--show-current",
        )

    def delete_local_repository(
            self,
            repository_url: str,
    ) -> None:
        """
        Delete the cached local clone.
        """
        repo = self._repo_path(repository_url)

        if repo.exists():
            shutil.rmtree(repo)