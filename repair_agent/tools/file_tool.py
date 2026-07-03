from pathlib import Path
from langchain_core.tools import tool
from repair_agent.models.patch import Patch

class FileTool:
    """Utilities for reading and surgically modifying source files."""

    @tool
    def read_file(self, file_path: str) -> str:
        """
        Read an entire file.
        """
        return Path(file_path).read_text(encoding="utf-8")

    def replace_lines(
            self,
            file_path: str,
            repair_patches: list[Patch],
    ) -> list[Patch]:
        """
        Replace multiple inclusive ranges of lines in a file.
        """

        path = Path(file_path)
        lines = path.read_text(encoding="utf-8").splitlines()

        # Validate that patches do not overlap
        repair_patches.sort(key=lambda p: (p.start_line, p.end_line))

        for i in range(len(repair_patches) - 1):
            if repair_patches[i].end_line >= repair_patches[i + 1].start_line:
                raise ValueError(
                    f"Overlapping patches: "
                    f"{repair_patches[i]} and {repair_patches[i + 1]}"
                )

        updated_patches: list[Patch] = []

        offset = 0

        for patch in repair_patches:

            original_start = patch.start_line - 1
            original_end = patch.end_line

            start = original_start + offset
            end = original_end + offset

            replacement_lines = patch.replacement.splitlines()

            lines = (
                    lines[:start]
                    + replacement_lines
                    + lines[end:]
            )

            new_end = start + len(replacement_lines)

            updated_patches.append(
                Patch(
                    methodName= patch.methodName,
                    start_line= start + 1,
                    end_line= new_end,
                    replacement= patch.replacement,
                    original= patch.original,
                )
            )

            offset += len(replacement_lines) - (patch.end_line - patch.start_line + 1)

        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
            )

        return updated_patches

    def remove_lines(
            self,
            file_path: str,
            remove_patches: list[Patch],
    ) -> None:
        """
        Restore previously replaced code using updated patch positions.
        """

        path = Path(file_path)
        lines = path.read_text(encoding="utf-8").splitlines()

        # Remove from bottom to top
        remove_patches.sort(
            key=lambda p: p.start_line,
            reverse=True
        )

        for patch in remove_patches:
            start = patch.start_line - 1
            end = patch.end_line

            original_lines = patch.original.splitlines()

            lines = (
                    lines[:start]
                    + original_lines
                    + lines[end:]
            )

        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
            )