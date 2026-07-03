from dataclasses import dataclass, field
from typing import List, Optional
from repair_agent.models.repair_item import RepairItem
from repair_agent.models.patch import Patch

@dataclass
class RepairGroup:
    """
    Represents one Pull Request.

    Grouping rules:

    TEST target
        -> group by testCaseFilePath

    SERVICE target
        -> group by serviceFilePath
    """

    group_key: str
    repair_items: List[RepairItem] = field(default_factory=list)
    branch_name: Optional[str] = None
    commit_message: Optional[str] = None
    pr_title: Optional[str] = None
    pr_description: Optional[str] = None
    repair_patches: List[Patch] = None
    is_skeleton_pr: bool = False
    owner: Optional[str] = None
    repository_url: Optional[str] = None