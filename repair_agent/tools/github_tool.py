import base64
import os

import requests
from langchain_core.tools import tool


class GitHubTool:

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }

    @staticmethod
    def _base_url(repository_url: str) -> str:
        repository_url = repository_url.rstrip("/")

        owner, repo = repository_url.split("/")[-2:]

        return f"https://api.github.com/repos/{owner}/{repo}"

    @tool
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

        base_url = self._base_url(repository_url)

        response = requests.get(
            f"{base_url}/contents/{file_path}",
            headers=self._headers(),
            params={"ref": ref},
            timeout=30,
        )

        response.raise_for_status()

        content = base64.b64decode(
            response.json()["content"]
        ).decode("utf-8")

        lines = content.splitlines()

        return "\n".join(lines[start_line - 1:end_line])