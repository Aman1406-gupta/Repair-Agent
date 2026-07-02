import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agent_builder.base.tools import (
    special_transfer_tool,
    openapi_spec_to_tools,
    openapi_yaml_list_to_tools,
    openapi_spec_to_metadata,
    openapi_metadata_to_tool,
)
from agent_builder.utils.constants import PARENT_ROUTER_NODE
from agent_builder.tests.conftest import make_state_with_transfer


def _aiohttp_client_session_patch(response_json_text: str):
    """Build patch for ``aiohttp.ClientSession`` used by OpenAPI HTTP tools."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = AsyncMock(return_value=response_json_text)

    req_cm = MagicMock()
    req_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    req_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.request = MagicMock(return_value=req_cm)

    sess_cm = MagicMock()
    sess_cm.__aenter__ = AsyncMock(return_value=mock_session)
    sess_cm.__aexit__ = AsyncMock(return_value=None)

    return patch("agent_builder.base.tools.aiohttp.ClientSession", return_value=sess_cm), mock_session


class TestTransferTool:
    """Test transfer_tool routing behavior"""

    @pytest.mark.asyncio
    async def test_transfer_routes_to_target_and_updates_path(self):
        transfer = special_transfer_tool(allowed_tasks=["target_task"])
        tool_node = ToolNode(tools=[transfer])

        result = await tool_node.ainvoke(make_state_with_transfer("target_task"))

        assert isinstance(result[0], Command)
        assert result[0].goto == "target_task"
        assert result[0].update['last_active_task']['path'][0] == "target_task"

    @pytest.mark.asyncio
    async def test_parent_transfer_pops_path(self):
        transfer = special_transfer_tool(allowed_tasks=[])
        tool_node = ToolNode(tools=[transfer])

        state = make_state_with_transfer("<PARENT>")
        state['last_active_task'] = {'path': ['outer', 'inner', 'deep'], 'depth': 2}

        result = await tool_node.ainvoke(state)

        assert result[0].goto == PARENT_ROUTER_NODE
        assert result[0].update['last_active_task']['path'] == ['outer']

    @pytest.mark.asyncio
    async def test_manual_transfer_routes_to_parent_router(self):
        transfer = special_transfer_tool(allowed_tasks=[])
        tool_node = ToolNode(tools=[transfer])

        state = make_state_with_transfer("<MANUAL_TRANSFER>")
        state['last_active_task'] = {'path': ['current_task'], 'depth': 0}

        result = await tool_node.ainvoke(state)

        assert result[0].goto == PARENT_ROUTER_NODE

    @pytest.mark.asyncio
    async def test_invalid_task_returns_error(self):
        transfer = special_transfer_tool(allowed_tasks=["allowed"])
        tool_node = ToolNode(tools=[transfer])

        result = await tool_node.ainvoke(make_state_with_transfer("not_allowed"))

        error_msg = result['messages'][0]
        assert error_msg.status == 'error'
        assert "Invalid task" in error_msg.content


class TestOpenAPITools:
    """Test that OpenAPI tools make correct HTTP requests (async ``aiohttp``)."""

    @pytest.mark.asyncio
    async def test_generated_tool_makes_correct_http_call(self, openapi_spec):
        """Tool should substitute path params and make HTTP request."""
        tools = openapi_spec_to_tools(openapi_spec)
        get_user = next(t for t in tools if t.name == "getUser")

        body = '{"id": "123"}'
        session_patch, mock_session = _aiohttp_client_session_patch(body)
        with session_patch:
            result = await get_user.ainvoke({"userId": "123"})

        mock_session.request.assert_called_once()
        call_kw = mock_session.request.call_args
        assert call_kw[0][0] == "GET"
        assert call_kw[0][1] == "https://api.example.com/users/123"
        assert result == {"id": "123"}

    @pytest.mark.asyncio
    async def test_post_tool_sends_body(self, openapi_spec):
        """POST tool should send body as JSON."""
        tools = openapi_spec_to_tools(openapi_spec)
        create_user = next(t for t in tools if t.name == "createUser")

        session_patch, mock_session = _aiohttp_client_session_patch('{"id": "new"}')
        with session_patch:
            await create_user.ainvoke({"name": "Test", "email": "test@example.com"})

        mock_session.request.assert_called_once()
        call_kw = mock_session.request.call_args
        assert call_kw[0][0] == "POST"
        assert call_kw[1]["json"] == {"name": "Test", "email": "test@example.com"}


class TestYAMLParsing:
    """Test YAML to tools conversion"""

    @pytest.mark.asyncio
    async def test_parses_yaml_and_creates_working_tool(self):
        """Should parse YAML string and create callable tool"""
        yaml_content = """
openapi: "3.0.0"
info:
  title: Test
  version: "1.0"
servers:
  - url: https://test.com
paths:
  /items:
    get:
      operationId: listItems
      summary: List items
"""
        tools = openapi_yaml_list_to_tools([yaml_content])

        assert len(tools) == 1
        assert tools[0].name == "listItems"

        session_patch, mock_session = _aiohttp_client_session_patch(json.dumps([]))
        with session_patch:
            await tools[0].ainvoke({})

        call_kw = mock_session.request.call_args
        assert "test.com/items" in call_kw[0][1]


class TestMetadataRoundtrip:
    """Test spec → metadata → tool roundtrip"""

    @pytest.mark.asyncio
    async def test_metadata_to_tool_creates_working_tool(self, openapi_spec):
        """Metadata converted to tool should work correctly"""
        metadata_list = openapi_spec_to_metadata(openapi_spec)
        tool = openapi_metadata_to_tool(metadata_list[0])

        session_patch, mock_session = _aiohttp_client_session_patch(json.dumps({"result": "ok"}))
        with session_patch:
            result = await tool.ainvoke({"userId": "abc"})
        assert result == {"result": "ok"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
