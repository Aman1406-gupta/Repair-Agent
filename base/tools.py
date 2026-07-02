import json
import logging

import aiohttp
from langchain_core.tools import tool, StructuredTool, InjectedToolCallId, Tool
from langgraph.types import Command
from typing_extensions import Annotated, List, Any, Dict, Union, Optional
from agent_builder.base.state import State
from langgraph.prebuilt import InjectedState
import yaml
from agent_builder.utils.openapi_utils import _build_arg_model, sanitize_tool_name
from agent_builder.utils.constants import PARENT_ROUTER_NODE

logger = logging.getLogger(__name__)

##########################################################################################################################################
#
#
#                                     DEFAULT TOOLS
#
#
##########################################################################################################################################
def special_transfer_tool(allowed_tasks: List[str] = []):
    @tool
    def transfer_tool(
        id_: Annotated[str, "Domain to transfer to"], 
        state: Annotated[State, InjectedState] ,
        tool_call_id: Annotated[str, InjectedToolCallId] = None
    ):
        """Transfer conversation to a domain specialist """
        
        all_allowed_tasks = allowed_tasks + ['<PARENT>', '<MANUAL_TRANSFER>']
        assert id_ in all_allowed_tasks, f"Invalid task: {id_}! Allowed tasks are: {','.join(allowed_tasks)}"

        logger.debug("Transfer tool invoked | target=%s", id_)

        path = state['last_active_task']['path']
        depth = state['last_active_task']['depth']

        if id_ == '<PARENT>':
            state['last_active_task']['path'] = path[:-2]
            target_node = PARENT_ROUTER_NODE
        elif id_ == '<MANUAL_TRANSFER>':
            # state['last_active_task']['path'] = path[:-1]
            id_ = path[-1]
            target_node = PARENT_ROUTER_NODE
        else:
            state['last_active_task']['path'][depth] = id_
            target_node = id_
            
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {id_}",
            "tool_call_id": tool_call_id,
        }
        if tool_call_id:    
            state['messages'] = state['messages'] + [tool_message]
        
        return Command(
            update=state,
            goto=target_node,
            graph=Command.PARENT
        )
    object.__setattr__(transfer_tool, '_allowed_tasks', list(allowed_tasks))
    return transfer_tool


##########################################################################################################################################
#
#
#                                     OPENAPI SPEC AS TOOLS
#
#
##########################################################################################################################################
def _make_caller(base_url: str, path_tmpl: str, method: str, body_fields: List[str] = None):
    """
    Return a closure that executes the HTTP request for this operation.
    
    Args:
        base_url: The base URL for the API
        path_tmpl: The path template with placeholders for path parameters
        method: HTTP method (GET, POST, etc.)
        body_fields: List of parameter names that should go into the request body.
                     If empty/None, no body is sent.
    """
    body_fields = body_fields or []

    async def _call(**kwargs):
        url = base_url.rstrip("/") + path_tmpl
        # path params → substitute in URL
        for k, v in list(kwargs.items()):
            ph = "{" + k + "}"
            if ph in url:
                url = url.replace(ph, str(v))
                kwargs.pop(k)

        # Reconstruct request body from flattened parameters
        body = None
        if body_fields:
            # Check if it's the legacy 'body' field (fallback case)
            if body_fields == ["body"]:
                body = kwargs.pop("body", None)
            else:
                # Collect all body field values into the request body
                body = {}
                for field_name in body_fields:
                    if field_name in kwargs:
                        value = kwargs.pop(field_name)
                        if value is not None:
                            body[field_name] = value
                # If body is empty after collecting, set to None
                body = body if body else None

        meth = method.upper()
        logger.debug("Tool HTTP call | %s %s has_body=%s", meth, url, body is not None)
        timeout = aiohttp.ClientTimeout(total=20)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(meth, url, params=kwargs or None, json=body) as resp:
                    status_code = resp.status
                    logger.debug("Tool HTTP response | status=%s url=%s", status_code, url)
                    if isinstance(status_code, int) and status_code >= 400:
                        tb = await resp.text()
                        logger.warning(
                            "Tool HTTP non-2xx | status=%s url=%s body=%.500s",
                            status_code,
                            url,
                            tb,
                        )
                    resp.raise_for_status()
                    text = await resp.text()
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        return text
        except TimeoutError:
            logger.warning("Tool HTTP timeout | %s %s", meth, url)
            raise
        except aiohttp.ClientError as e:
            logger.error("Tool HTTP client error | %s %s: %s", meth, url, e)
            raise

    return _call


# ─── conversion core ────────────────────────────────────────────────────────────
def openapi_spec_to_tools(spec: Dict[str, Any], filter_name=None) -> List[StructuredTool]:
    base_url = (spec.get("servers") or [{"url": ""}])[0]["url"]
    tools: List[Tool] = []

    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            raw_op_id = op.get("operationId") or f"{method}_{path.strip('/').replace('/', '_')}"
            op_id = sanitize_tool_name(raw_op_id)
            description = op.get("summary") or op.get("description") or "No description"
            parameters = op.get("parameters", [])

            body_schema = None
            if "requestBody" in op:
                content = op["requestBody"].get("content", {})
                if "application/json" in content:
                    body_schema = content["application/json"].get("schema")

            ArgsModel, body_fields = _build_arg_model(op_id, parameters, body_schema)
            caller = _make_caller(base_url, path, method, body_fields=body_fields)

            tools.append(
                StructuredTool(
                    name=op_id,
                    description=description,
                    args_schema=ArgsModel,
                    func=None,
                    coroutine=caller,
                )
            )
    return tools


def openapi_yaml_list_to_tools(yaml_strings: Union[List[str], List[bytes]]) -> List[StructuredTool]:
    """
    Accepts a list of YAML *contents* (strings/bytes or file paths)
    and returns a flat list of LangGraph Tool objects.
    """
    tools: List[Tool] = []
    for y in yaml_strings:
        # If the item is a file path, read it; else treat it as YAML content
        if isinstance(y, str) and not y.lstrip().startswith(("openapi:", "{", "[")):
            with open(y, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
        else:
            spec = yaml.safe_load(y)

        tools.extend(openapi_spec_to_tools(spec))
    return tools


def openapi_spec_to_metadata(spec: Dict[str, Any], filter_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert an OpenAPI spec into a list of JSON-serializable metadata dicts."""
    base_url = (spec.get("servers") or [{"url": ""}])[0]["url"]
    metadata_list: List[Dict[str, Any]] = []

    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            raw_op_id = op.get("operationId") or f"{method}_{path.strip('/').replace('/', '_')}"
            op_id = sanitize_tool_name(raw_op_id)
            if filter_name and filter_name != op_id:
                continue

            description = op.get("summary") or op.get("description") or "No description"
            parameters = op.get("parameters", [])

            body_schema = None
            if "requestBody" in op:
                content = op["requestBody"].get("content", {})
                if "application/json" in content:
                    body_schema = content["application/json"].get("schema")

            ArgsModel, body_fields = _build_arg_model(op_id, parameters, body_schema)

            metadata_list.append({
                "name": op_id,
                "description": description,
                "args_schema": ArgsModel.model_json_schema(),
                "httpConfig": {
                    "base_url": base_url,
                    "path": path,
                    "method": method.upper(),
                    "has_body": body_schema is not None,
                    "body_fields": body_fields,
                    "parameters": parameters
                },
                "body_schema": body_schema,
                "source": spec.get("info", {}).get("title", "unknown")
            })

    return metadata_list


def openapi_metadata_to_tool(metadata: Dict[str, Any]) -> StructuredTool:
    params = metadata["httpConfig"].get("parameters", [])
    body_schema = metadata.get("body_schema")

    # Sanitize the tool name in case metadata contains unsanitized names
    sanitized_name = sanitize_tool_name(metadata["name"])

    ArgsModel, body_fields = _build_arg_model(
        sanitized_name,
        params,
        body_schema,
    )

    # Use stored body_fields from metadata if available (for backward compatibility),
    # otherwise use the freshly computed body_fields
    stored_body_fields = metadata["httpConfig"].get("body_fields")
    if stored_body_fields is not None:
        body_fields = stored_body_fields

    caller = _make_caller(
        base_url=metadata["httpConfig"]["base_url"],
        path_tmpl=metadata["httpConfig"]["path"],
        method=str(metadata["httpConfig"]["method"]).lower(),
        body_fields=body_fields,
    )

    tool = StructuredTool(
        name=sanitized_name,
        description=metadata["description"],
        args_schema=ArgsModel,
        func=None,
        coroutine=caller,
    )

    tool.tags = [str(metadata.get("_id", ""))]
    return tool
