from pathlib import Path

from langchain_core.tools import tool


class FileTool:
    """Utilities for reading and surgically modifying source files."""

    @tool
    def read_file(self, file_path: str) -> str:
        """
        Read an entire file.
        """
        return Path(file_path).read_text(encoding="utf-8")

    @tool
    def replace_lines(
            self,
            file_path: str,
            start_line: int,
            end_line: int,
            replacement: str,
    ) -> str:
        """
        Replace an inclusive range of lines with the supplied code.
        """

        path = Path(file_path)

        lines = path.read_text(encoding="utf-8").splitlines()

        start = max(start_line - 1, 0)
        end = min(end_line, len(lines))

        replacement_lines = replacement.splitlines()

        updated = (
                lines[:start]
                + replacement_lines
                + lines[end:]
        )

        path.write_text(
            "\n".join(updated) + "\n",
            encoding="utf-8",
            )

        return f"Updated {file_path}"

    @tool
    def write_file(
            self,
            file_path: str,
            content: str,
    ) -> str:
        """
        Overwrite a file.
        """

        Path(file_path).write_text(
            content,
            encoding="utf-8",
        )

        return f"Written {file_path}"