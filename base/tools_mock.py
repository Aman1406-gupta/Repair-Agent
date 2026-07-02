from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from agent_builder.utils.openapi_utils import _build_arg_model, sanitize_tool_name
from agent_builder.utils.misc import clean_json_string
from pydantic import BaseModel, PrivateAttr
from agent_builder.base.configs import LLMConfig
from agent_builder.utils.constants import DEFAULT_LLM_CONFIG
from agent_builder.llm_client.sprinklr_chat_model import SprinklrChatModel
import json
import textwrap

import logging
logger = logging.getLogger(__name__)

_DEFAULT_MOCK_BEHAVIOR = "Return a plausible mock JSON response."

# ──────────────────────────────────────────────────────────────
#  Part 2:  MockStructuredTool
# ──────────────────────────────────────────────────────────────
class MockStructuredTool(StructuredTool):
    """A StructuredTool whose body is faked by an LLM according to
    a natural-language *behavior* specification."""
    _behavior: str = PrivateAttr(default="")
    _llm: BaseChatModel = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        name: str,
        description: str,
        behavior: str,
        args_schema: Type[BaseModel],
        llm_config: LLMConfig,
    ):
        behavior_text = textwrap.dedent(behavior).strip()
        llm_obj = SprinklrChatModel(llm_config=llm_config)

        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
            func=None,
            coroutine=self._mock_coroutine,
        )

        self._behavior = behavior_text
        self._llm = llm_obj

    def _build_mock_messages(self, arguments: Dict[str, Any]) -> List[Any]:
        system = (
            f"You are mocking the `{self.name}` tool.\n\n"
            f"BEHAVIOR SPECIFICATION:\n{self._behavior}\n\n"
            "You will receive JSON arguments and must reply with "
            "ONLY a JSON object representing the return value. "
            "Do NOT wrap the JSON in markdown code blocks."
        )
        return [
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(arguments, indent=2)),
        ]

    def _parse_mock_llm_response(self, response: Any) -> Any:
        content = clean_json_string(response.content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM output for tool `{self.name}` was not valid JSON:\n{response.content}"
            ) from e

    async def _mock_coroutine(self, **kwargs: Any) -> Any:
        logger.debug("Mock tool invoked (async) | name=%s", self.name)
        prompt = self._build_mock_messages(kwargs)
        response = await self._llm.ainvoke(prompt)
        return self._parse_mock_llm_response(response)

    def _run(self, **kwargs):
        logger.debug("Mock tool invoked | name=%s", self.name)
        prompt = self._build_mock_messages(kwargs)
        response = self._llm.invoke(prompt)
        return self._parse_mock_llm_response(response)

    @classmethod
    def from_openai_schema(
        cls,
        *,
        schema: Dict[str, Any],
        behavior: str,
        llm_config: LLMConfig,
    ):
        name = schema["name"]
        desc = schema.get("description", "No description")
        # Build arg model from the JSON-Schema parameters field
        params_schema = schema.get("parameters", {})
        props = params_schema.get("properties", {})
        required = set(params_schema.get("required", []))
        param_list = []
        for prop_name, prop_schema in props.items():
            param_list.append(
                {
                    "name": prop_name,
                    "required": prop_name in required,
                    "schema": prop_schema,
                }
            )
        args_model, _ = _build_arg_model(name, param_list, body_schema=None)
        return cls(
            name=name,
            description=desc,
            behavior=behavior,
            args_schema=args_model,
            llm_config=llm_config,
        )

    @classmethod
    def from_openapi_operation(
        cls,
        *,
        op_dict: Dict[str, Any],
        fallback_name: str,
        behavior: str,
        llm_config: LLMConfig,
    ):
        name = op_dict.get("operationId", fallback_name)
        desc = op_dict.get("summary") or op_dict.get("description") or "No description"
        params = op_dict.get("parameters", [])
        body_schema = None
        if "requestBody" in op_dict:
            content = op_dict["requestBody"].get("content", {})
            if "application/json" in content:
                body_schema = content["application/json"].get("schema")
        args_schema, _ = _build_arg_model(name, params, body_schema)
        return cls(
            name=name,
            description=desc,
            behavior=behavior,
            args_schema=args_schema,
            llm_config=llm_config,
        )


# ──────────────────────────────────────────────────────────────
#  Part 3:  Bulk builders
# ──────────────────────────────────────────────────────────────
def make_mock_tools_from_openapi(
    spec: Dict[str, Any],
    *,
    llm_config: LLMConfig,
    llm_behavior: str = _DEFAULT_MOCK_BEHAVIOR,
    filter_name: Optional[str] = None,
):
    """Return list[MockStructuredTool] – one per operation."""
    tools = []
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            raw_op_id = op.get("operationId") or f"{method}_{path.strip('/').replace('/', '_')}"
            op_id = sanitize_tool_name(raw_op_id)
            if filter_name is not None and filter_name != op_id:
                continue
            tool = MockStructuredTool.from_openapi_operation(
                op_dict=op,
                fallback_name=op_id,
                behavior=llm_behavior,
                llm_config=llm_config,
            )
            tools.append(tool)
    return tools


def make_mock_tool_from_openai_function(
    schema: Dict[str, Any],
    *,
    llm_config: LLMConfig,
    llm_behavior: str = _DEFAULT_MOCK_BEHAVIOR,
):
    return MockStructuredTool.from_openai_schema(
        schema=schema,
        behavior=llm_behavior,
        llm_config=llm_config,
    )


# ──────────────────────────────────────────────────────────────
#  Part 4:  Metadata conversion functions
# ──────────────────────────────────────────────────────────────

def openapi_spec_to_prompt_tool_metadata(
    spec: Dict[str, Any],
    *,
    llm_config: Dict[str, Any],
    default_behavior: str = _DEFAULT_MOCK_BEHAVIOR,
    behavior_overrides: Optional[Dict[str, str]] = None,
    filter_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return List[Dict] – metadata for one tool per operation."""
    behavior_overrides = behavior_overrides or {}
    tools_metadata = []
    
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            raw_op_id = op.get("operationId") or f"{method}_{path.strip('/').replace('/', '_')}"
            op_id = sanitize_tool_name(raw_op_id)
            if filter_name and filter_name != op_id:
                continue
            behavior = behavior_overrides.get(op_id, default_behavior)
            
            tool_metadata = {
                "name": op_id,
                "toolType": "prompt_tool",
                "mockConfig": {
                    "behavior": behavior,
                    "llm_config": llm_config,
                    "source_type": "openapi"
                },
                "op_dict": op
            }
            tools_metadata.append(tool_metadata)
    
    return tools_metadata


def openai_schema_to_prompt_tool_metadata(
    schema: Dict[str, Any],
    *,
    llm_config: Dict[str, Any],
    default_behavior: str = _DEFAULT_MOCK_BEHAVIOR,
) -> Dict[str, Any]:
    """Return Dict – metadata for single tool."""
    tool_metadata = {
        "name": schema["name"],
        "toolType": "prompt_tool",
        "mockConfig": {
            "behavior": default_behavior,
            "llm_config": llm_config,
            "source_type": "openai_function"
        },
        "openai_schema": schema
    }
    return tool_metadata


def create_mock_tool_from_metadata(metadata: Dict[str, Any]):
    """Create a MockStructuredTool instance from stored metadata."""
    
    tool_type = metadata.get("toolType")
    if tool_type != "prompt_tool":
        raise ValueError(f"Metadata is not for a prompt tool: {tool_type}")
    
    mock_config = metadata.get("mockConfig", {})
    source_type = mock_config.get("source_type")
    behavior = mock_config.get("behavior", _DEFAULT_MOCK_BEHAVIOR)
    llm_config = mock_config.get("llm_config", DEFAULT_LLM_CONFIG)
    
    # Convert dict llm_config to LLMConfig object if needed
    if not isinstance(llm_config, LLMConfig):
        llm_config_obj = LLMConfig(**llm_config)
    else:
        llm_config_obj = llm_config
    
    # Sanitize the tool name in case metadata contains unsanitized names
    sanitized_name = sanitize_tool_name(metadata["name"])
    
    if source_type == "openapi":
        # For OpenAPI tools, use the stored operation dict directly
        op_dict = metadata.get("op_dict", {})
        if not op_dict:
            raise ValueError("OpenAPI metadata missing op_dict")
        
        tool = MockStructuredTool.from_openapi_operation(
            op_dict=op_dict,
            fallback_name=sanitized_name,
            behavior=behavior,
            llm_config=llm_config_obj,
        )
    elif source_type == "openai_function":
        # For OpenAI function tools, use the stored schema directly
        openai_schema = metadata.get("openai_schema", {})
        if not openai_schema:
            raise ValueError("OpenAI metadata missing openai_schema")
        
        # Create a copy of the schema with sanitized name
        sanitized_schema = openai_schema.copy()
        sanitized_schema["name"] = sanitized_name
        
        tool = MockStructuredTool.from_openai_schema(
            schema=sanitized_schema,
            behavior=behavior,
            llm_config=llm_config_obj,
        )
    else:
        raise ValueError(f"Unknown prompt tool source type: {source_type}")
    
    tool.tags = [str(metadata.get("_id", ""))]
    
    return tool