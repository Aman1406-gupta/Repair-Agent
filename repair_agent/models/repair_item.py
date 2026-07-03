from dataclasses import dataclass
from typing import Optional
from repair_agent.models.test_document import TestDocument

@dataclass
class RepairItem:
    """
    Represents one failing test throughout the complete workflow.
    """

    test_document: TestDocument
    test_source_code: Optional[str] = None
    pre_repair_git_diff: Optional[str] = None
    post_repair_git_diff: Optional[str] = None
    failure_analysis: Optional[str] = None
    is_infrastructure: bool = False
    is_reproducible: bool = False
    target_to_repair: Optional[str] = None
    service_file_path: Optional[str] = None
    service_start_line: Optional[int] = None
    service_end_line: Optional[int] = None
    service_source_code: Optional[str] = None
    repair_patch: Optional[str] = None
    validation_passed: bool = False
    root_cause: Optional[str] = None

    status: str = "PENDING"

    error: Optional[str] = None