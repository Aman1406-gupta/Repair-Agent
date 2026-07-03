from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from repair_agent.models.repair_group import RepairGroup
from repair_agent.models.repair_item import RepairItem
from repair_agent.models.test_document import TestDocument

class RepairState(TypedDict, total=False):

    test_documents: List[TestDocument]
    repair_items: List[RepairItem]
    repair_groups: dict[str, RepairGroup]
    messages: List[BaseMessage]
    pr_urls: List[str]
    errors: List[str]