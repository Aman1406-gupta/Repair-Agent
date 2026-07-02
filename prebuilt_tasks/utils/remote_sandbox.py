"""RemoteSandboxBackend — deepagents BaseSandbox backed by sandbox_sprinklr HTTP API.

Implements ``execute()``, ``upload_files()``, and ``download_files()``
by forwarding calls to the ``sandbox_sprinklr`` service over HTTP.

Overrides ``ls``, ``read``, ``glob``, and ``edit`` from ``BaseSandbox``
because the default implementations run Python3 scripts via ``execute()``,
but the just-bash sandbox does not have a Python interpreter.  The
overrides use shell builtins (find, cat, sed) or the download API instead.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    FileData,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox, FileInfo, GlobResult, LsResult

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_SERVICE_URL = os.environ.get(
    "SANDBOX_SERVICE_URL", "http://localhost:10000"
)
DEFAULT_TIMEOUT_S = 120


class RemoteSandboxBackend(BaseSandbox):
    """Sandbox backend that delegates to a remote ``sandbox_sprinklr`` service.

    On first use the backend lazily ensures the sandbox exists by calling
    ``/sandbox/create``.  Subsequent calls to ``execute``, ``upload_files``,
    and ``download_files`` are forwarded over HTTP.
    """

    def __init__(
        self,
        sandbox_name: str,
        *,
        sandbox_service_url: str = DEFAULT_SANDBOX_SERVICE_URL,
        default_timeout: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._sandbox_name = sandbox_name
        self._service_url = sandbox_service_url.rstrip("/")
        self._default_timeout = default_timeout
        self._sandbox_id: Optional[str] = None
        self._initialised = False

    @property
    def id(self) -> str:
        return self._sandbox_name

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout_s: float = 60.0) -> dict:
        url = f"{self._service_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Sandbox-Session", self._sandbox_name)
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error("HTTP %d from %s: %s", e.code, url, body[:500])
            raise
        except URLError as e:
            logger.error("Connection error to %s: %s", url, e.reason)
            raise

    def _ensure_sandbox(self) -> None:
        if self._initialised:
            return
        resp = self._post("/sandbox/create", {"name": self._sandbox_name})
        self._sandbox_id = resp.get("sandbox_id", self._sandbox_name)
        self._initialised = True
        logger.info(
            "Sandbox ready: name=%s id=%s", self._sandbox_name, self._sandbox_id
        )

    # ------------------------------------------------------------------
    # BaseSandbox abstract methods
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        self._ensure_sandbox()

        effective_timeout = timeout if timeout is not None else self._default_timeout
        resp = self._post(
            "/sandbox/execute",
            {
                "sandbox_name": self._sandbox_name,
                "command": command,
                "timeout": effective_timeout,
            },
            timeout_s=effective_timeout + 30,
        )

        if not resp.get("success", False):
            return ExecuteResponse(
                output=resp.get("error", "Unknown sandbox execution error"),
                exit_code=1,
                truncated=False,
            )

        return ExecuteResponse(
            output=resp.get("output", ""),
            exit_code=resp.get("exit_code"),
            truncated=resp.get("truncated", False),
        )

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        self._ensure_sandbox()

        file_entries = [
            {
                "path": path,
                "content_b64": base64.b64encode(content).decode("ascii"),
            }
            for path, content in files
        ]

        resp = self._post(
            "/sandbox/upload",
            {
                "sandbox_name": self._sandbox_name,
                "files": file_entries,
            },
        )

        results = resp.get("results", [])
        return [
            FileUploadResponse(
                path=r.get("path", ""),
                error=r.get("error"),
            )
            for r in results
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        self._ensure_sandbox()

        resp = self._post(
            "/sandbox/download",
            {
                "sandbox_name": self._sandbox_name,
                "paths": paths,
            },
        )

        results = resp.get("results", [])
        return [
            FileDownloadResponse(
                path=r.get("path", ""),
                content=(
                    base64.b64decode(r["content_b64"])
                    if r.get("content_b64")
                    else None
                ),
                error=r.get("error"),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Overrides — BaseSandbox defaults use python3 scripts via execute(),
    # which doesn't work in just-bash.  These use shell builtins or the
    # download/upload HTTP API instead.
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        safe = shlex.quote(path)
        result = self.execute(f"find {safe} -maxdepth 1 -mindepth 1 2>/dev/null")
        if result.exit_code and result.exit_code != 0:
            return LsResult(error=f"Cannot list '{path}': {result.output.strip()}")

        all_paths = {
            p.strip() for p in result.output.strip().splitlines() if p.strip()
        }
        dir_result = self.execute(f"find {safe} -maxdepth 1 -mindepth 1 -type d 2>/dev/null")
        dir_paths = {
            p.strip() for p in dir_result.output.strip().splitlines() if p.strip()
        }
        entries: list[FileInfo] = [
            FileInfo(path=p, is_dir=(p in dir_paths)) for p in sorted(all_paths)
        ]
        return LsResult(entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        responses = self.download_files([file_path])
        if not responses:
            return ReadResult(error=f"File '{file_path}': no response from sandbox")
        resp = responses[0]
        if resp.error:
            return ReadResult(error=f"File '{file_path}': {resp.error}")
        if resp.content is None:
            return ReadResult(error=f"File '{file_path}': empty content")
        try:
            text = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            return ReadResult(
                file_data=FileData(
                    content=base64.b64encode(resp.content).decode("ascii"),
                    encoding="base64",
                )
            )
        lines = text.splitlines(keepends=True)
        page = lines[offset : offset + limit]
        return ReadResult(file_data=FileData(content="".join(page), encoding="utf-8"))

    def write(self, file_path: str, content: str) -> WriteResult:
        safe = shlex.quote(file_path)
        check = self.execute(f"test -e {safe} && echo EXISTS || echo OK")
        if "EXISTS" in check.output:
            return WriteResult(error=f"File already exists: '{file_path}'")
        parent = os.path.dirname(file_path)
        if parent:
            self.execute(f"mkdir -p {shlex.quote(parent)}")
        responses = self.upload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            return WriteResult(error=f"Failed to write file '{file_path}': no response")
        if responses[0].error:
            return WriteResult(error=f"Failed to write file '{file_path}': {responses[0].error}")
        return WriteResult(path=file_path)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        safe_path = shlex.quote(path)
        safe_pattern = shlex.quote(pattern)
        result = self.execute(f"find {safe_path} -path {safe_pattern} 2>/dev/null || true")
        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])
        matches: list[FileInfo] = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                matches.append(FileInfo(path=line))
        return GlobResult(matches=matches)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        self._ensure_sandbox()
        resp = self._post(
            "/sandbox/edit",
            {
                "sandbox_name": self._sandbox_name,
                "file_path": file_path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            },
        )
        if not resp.get("success", False):
            return EditResult(error=f"File '{file_path}': {resp.get('error', 'unknown error')}")
        return EditResult(path=resp.get("path", file_path), occurrences=resp.get("occurrences", 1))
