import os
import requests
from langchain_core.tools import tool

class GitHubTool:
    """Wrapper around remote GitHub API operations."""

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is missing.")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }

    @staticmethod
    def _base_url(repository_url: str) -> str:
        """Parses git URL or web URL into the GitHub API base endpoint."""
        repository_url = repository_url.rstrip("/")
        parts = repository_url.replace(".git", "").split("/")
        owner, repo = parts[-2:]
        if ":" in owner:
            owner = owner.split(":")[-1]
        return f"https://api.github.com/repos/{owner}/{repo}"

    def fetch_file_lines(
            self,
            repository_url: str,
            file_path: str,
            start_line: int,
            end_line: int,
            ref: str,
    ) -> str:
        """
        Fetch specific lines from a GitHub file.
        """
        content = self.fetch_file(repository_url, file_path, ref)
        lines = content.splitlines()

        return "\n".join(lines[start_line - 1:end_line])

    def fetch_file(self, repository_url: str, file_path: str, ref: str) -> str:
        """
        Fetch the entire content of a GitHub file. Supports files larger than 1MB
        by using the raw media type format.
        """
        base_url = self._base_url(repository_url)
        headers = self._headers()
        headers["Accept"] = "application/vnd.github.raw"

        response = requests.get(
            f"{base_url}/contents/{file_path}",
            headers=headers,
            params={"ref": ref},
            timeout=30,
        )
        response.raise_for_status()
        return response.text

    def create_pull_request(
            self,
            repository_url: str,
            source_branch: str,
            target_branch: str,
            title: str,
            description: str,
    ) -> str:
        """
        Creates a GitHub Pull Request.
        """
        base_url = self._base_url(repository_url)

        payload = {
            "title": title,
            "head": source_branch,
            "base": target_branch,
            "body": description,
        }

        response = requests.post(
            f"{base_url}/pulls",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["html_url"]

    @tool
    def fetch_file_lines_tool(
            self,
            repository_url: str,
            file_path: str,
            start_line: int,
            end_line: int,
            ref: str,
    ) -> str:
        return self.fetch_file_lines(
            repository_url= repository_url,
            file_path= file_path,
            start_line= start_line,
            end_line= end_line,
            ref= ref
        )