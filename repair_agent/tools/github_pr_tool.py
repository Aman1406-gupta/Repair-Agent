import os

import requests
from langchain_core.tools import tool


class GitHubPRTool:

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