import pytest

from agent_builder.base.tools_mock import (
    MockStructuredTool,
    create_mock_tool_from_metadata,
    make_mock_tools_from_openapi,
    openai_schema_to_prompt_tool_metadata,
    openapi_spec_to_prompt_tool_metadata,
)



class TestMockStructuredTool:
    """Test MockStructuredTool with real LLM calls"""
    
    @pytest.mark.asyncio
    async def test_tool_returns_valid_json(self, advance_llm_config, openai_schema):
        """Tool should return valid JSON from LLM.
        Uses gpt-4o for better accuracy.
        Note: MockStructuredTool._run already parses JSON, so result is a dict."""
        tool = MockStructuredTool.from_openai_schema(
            schema=openai_schema,
            behavior="Return a JSON with 'temperature' (number) and 'conditions' (string)",
            llm_config=advance_llm_config,
        )
        
        result = await tool.ainvoke({"city": "Paris", "units": "celsius"})
        
        # MockStructuredTool._run already parses JSON and returns a dict
        assert isinstance(result, dict)
        assert "temperature" in result
        assert "conditions" in result

    
    @pytest.mark.asyncio
    async def test_behavior_specification_is_followed(self, advance_llm_config):
        """LLM should follow the behavior specification exactly.
        Uses gpt-4o for better accuracy.
        Note: MockStructuredTool._run already parses JSON, so result is a dict."""
        schema = {
            "name": "get_status",
            "description": "Get status",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"]
            }
        }
        
        tool = MockStructuredTool.from_openai_schema(
            schema=schema,
            behavior='Always return exactly: {"status": "active", "code": 200}',
            llm_config=advance_llm_config,
        )
        
        result = await tool.ainvoke({"id": "123"})
        
        # MockStructuredTool._run already parses JSON and returns a dict
        assert isinstance(result, dict)
        assert result["status"] == "active"
        assert result["code"] == 200



class TestBulkBuilders:
    """Test bulk tool creation functions"""
    
    def test_make_mock_tools_from_openapi_creates_all_tools(self, base_llm_config, openapi_spec):
        """Should create one tool per operation"""
        tools = make_mock_tools_from_openapi(openapi_spec, llm_config=base_llm_config)
        
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "getUser" in names
        assert "createUser" in names
    
    def test_make_mock_tools_from_openapi_with_filter(self, base_llm_config, openapi_spec):
        """filter_name should return only matching tool"""
        tools = make_mock_tools_from_openapi(
            openapi_spec, 
            llm_config=base_llm_config,
            filter_name="getUser"
        )
        
        assert len(tools) == 1
        assert tools[0].name == "getUser"



class TestMetadataRoundtrip:
    """Test metadata creation and tool reconstruction"""
    
    def test_openapi_metadata_roundtrip(self, openapi_spec):
        """openapi spec → metadata → tool should work"""
        llm_config_dict = {"model": "gpt-4", "provider": "OPEN_AI", "partner_id": 0}
        
        metadata_list = openapi_spec_to_prompt_tool_metadata(
            openapi_spec,
            llm_config=llm_config_dict,
            default_behavior="Mock response"
        )
        
        assert len(metadata_list) == 2
        
        # Reconstruct tool from metadata
        tool = create_mock_tool_from_metadata(metadata_list[0])
        assert isinstance(tool, MockStructuredTool)
    
    def test_openai_schema_metadata_roundtrip(self, openai_schema):
        """openai schema → metadata → tool should work"""
        llm_config_dict = {"model": "gpt-4", "provider": "OPEN_AI", "partner_id": 0}
        
        metadata = openai_schema_to_prompt_tool_metadata(
            openai_schema,
            llm_config=llm_config_dict
        )
        
        assert metadata["name"] == "get_weather"
        assert metadata["toolType"] == "prompt_tool"
        
        # Reconstruct tool
        tool = create_mock_tool_from_metadata(metadata)
        assert tool.name == "get_weather"
    
    def test_create_mock_tool_from_metadata_rejects_wrong_type(self):
        """Should raise on non-prompt_tool metadata"""
        metadata = {"toolType": "regular_tool", "name": "test"}
        
        with pytest.raises(ValueError) as exc_info:
            create_mock_tool_from_metadata(metadata)
        
        assert "not for a prompt tool" in str(exc_info.value)
    
    def test_create_mock_tool_from_metadata_rejects_missing_data(self):
        """Should raise when required data is missing"""
        metadata = {
            "name": "test",
            "toolType": "prompt_tool",
            "mockConfig": {"source_type": "openapi", "llm_config": {"model": "gpt-4", "provider": "OPEN_AI", "partner_id": 0}}
            # Missing op_dict
        }
        
        with pytest.raises(ValueError) as exc_info:
            create_mock_tool_from_metadata(metadata)
        
        assert "missing op_dict" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
