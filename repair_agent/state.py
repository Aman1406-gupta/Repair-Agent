from typing import Annotated, Optional, Dict, Any, List

from base.state import State, return_right


class RepairState(State):
    """
    State shared across the complete repair workflow.

    AgentBuilder already provides:
        - messages
        - session_id
        - last_active_task
        - etc.

    We extend it with our repair-specific fields.
    """

    test_document: Annotated[Optional[Dict[str, Any]], return_right]

    test_source_code: Annotated[Optional[str], return_right]
    target_source_code: Annotated[Optional[str], return_right]

    git_diff: Annotated[Optional[str], return_right]

    analysis_result: Annotated[Optional[Dict[str, Any]], return_right]

    is_infrastructure: Annotated[Optional[bool], return_right]
    is_reproducible: Annotated[Optional[bool], return_right]

    generated_patch: Annotated[Optional[str], return_right]
    patch_file_path: Annotated[Optional[str], return_right]

    validation_passed: Annotated[Optional[bool], return_right]

    pr_url: Annotated[Optional[str], return_right]

    error_logs: Annotated[List[str], return_right]

    target_file_path: Annotated[Optional[str], return_right]

    target_start_line: Annotated[Optional[int], return_right]

    target_end_line: Annotated[Optional[int], return_right]

    retry_count: Annotated[int, return_right]

    max_retries: Annotated[int, return_right]