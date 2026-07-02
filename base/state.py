from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Annotated,Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from functools import wraps

def with_nonetype_check(merger_function):
    @wraps(merger_function)
    def wrapped(left, right):
        if not left:
            return right
        if not right:
            return left
        return merger_function(left, right)
    return wrapped

def override_with_typecasting(left, right):
    if len(right)>0 and isinstance(right[0],ToolMessage):
        return convert_to_messages(left+right)
    return convert_to_messages(right)
    
def add_new_keys_to_dict(left, right):
    # print(left, right)
    return left|right

@with_nonetype_check
def return_bigger_datetime(left, right):
    if isinstance(left, str):
        left = datetime.fromisoformat(left)
    if isinstance(right, str):
        right = datetime.fromisoformat(right)
    
    if left>right:
        return left
    return right

@with_nonetype_check
def sameid_check_strict(left, right):
    if left != right:
        raise ValueError("session_id mismatch while merging state")
    return right

@with_nonetype_check
def return_right(left, right):
    return right
    
@with_nonetype_check
def merge_non_null(left: dict, right: dict) -> dict:
    """
    Merge copilot response dicts:
    - Non-null / non-empty values in *right* override *left*
    - Null / empty values in *right* preserve *left*'s value
    
    This ensures that partial updates (e.g. a streaming chunk
    with usage but no error) don't wipe previously-set fields.
    """
    merged = dict(left)
    for key, value in right.items():
        if value is not None and value != {} and value != []:
            merged[key] = value
    return merged


class State(TypedDict):
    timestamp: Annotated[datetime,return_bigger_datetime]
    session_id: Annotated[str,sameid_check_strict]
    request_id: Annotated[str, return_right]
    response_id: Annotated[str, return_right]
    messages: Annotated[List[AnyMessage], override_with_typecasting]
    config_variables: Annotated[Dict[str,Any], add_new_keys_to_dict] 
    log:  Annotated[Dict[str, List[Tuple[datetime, BaseMessage]]],add_new_keys_to_dict]
    last_active_task: Annotated[Dict[str, Any], return_right]
    remote_request: Annotated[Dict[str, Any], return_right]
    remote_response: Annotated[Dict[str, Any], merge_non_null]

def get_initial_state(session_id):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "request_id": str(uuid.uuid4()),
        "response_id": "",
        "messages": [SystemMessage(content="This is a default system message")],
        "config_variables" : {},
        "log": {},
        'last_active_task': {
            'path': [],
            'depth': 0
        },
        "remote_request": {},
        "remote_response": {},
    }
