import pytest
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from agent_builder.base.configs import LLMConfig
from agent_builder.llm_client.sprinklr_chat_model import SprinklrChatModel
from agent_builder.llm_client.utils.message_converters import (
    _convert_message_to_dict,
    _convert_dict_to_message,
)
from llm_router.sdk.client import LLMClient


def test_message_dict_conversion_consistency():
    """
    Tests bidirectional conversion consistency using a complex AIMessage with tool calls.
    """
    # Create a complex AIMessage with tool calls
    complex_msg = AIMessage(
        content="Let me check the weather in multiple cities.",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston", "unit": "celsius"}'
                    }
                }
            ]
        }
    )
    
    # Test: Message -> Dict -> Message
    msg_dict = _convert_message_to_dict(complex_msg)
    msg_back = _convert_dict_to_message(msg_dict, disable_streaming=False)
    
    assert isinstance(msg_back, AIMessage)
    assert msg_back.content == complex_msg.content
    assert "tool_calls" in msg_back.additional_kwargs
    assert msg_back.additional_kwargs["tool_calls"][0]["id"] == "call_123"
    assert msg_back.additional_kwargs["tool_calls"][0]["function"]["name"] == "get_weather"
    print("✓ Message -> Dict -> Message: Consistent")
    
    # Test: Dict -> Message -> Dict
    dict_back = _convert_message_to_dict(msg_back)
    
    assert dict_back["role"] == "assistant"
    assert dict_back["content"] == complex_msg.content
    assert "tool_calls" in dict_back
    assert dict_back["tool_calls"][0]["id"] == "call_123"
    assert dict_back["tool_calls"][0]["function"]["arguments"] == '{"location": "Boston", "unit": "celsius"}'
    print("✓ Dict -> Message -> Dict: Consistent")
    
    print("✅ Bidirectional conversion consistency verified!")


def test_default_params_llm_configuration_id_router_payload():
    """Non-LOCAL provider with ``llm_configuration_id`` sends only id + tracking/partner/kwargs."""
    cfg = LLMConfig(
        model="gpt-4.1-mini-2025-04-14",
        provider="AZURE_OPEN_AI",
        temperature=0.1,
        max_tokens=1024,
        partner_id=66000000,
        llm_router_url="intuitionx-llm-router-v2.qa6-k8singress-intuition-gke.sprinklr.com",
        tracking_params={"release": "ca_research", "feature": "AGENT_BUILDER"},
        timeout=60,
        llm_configuration_id="cfg-presets-001",
        kwargs={"extra_flag": True},
    )
    model = SprinklrChatModel(llm_config=cfg)
    params = model._default_params
    assert params["llm_config_id"] == "cfg-presets-001"
    assert params["partner_id"] == 66000000
    assert params["extra_flag"] is True
    # assert "model" not in params
    assert "temperature" not in params
    assert "max_tokens" not in params


class TestSprinklrChatModelE2E:

    def test_initialization_provider_llmclient(self, base_llm_config):
        """
        Tests that the default provider initializes the `LLMClient` without error.
        """
        model = None
        try:
            model = SprinklrChatModel(llm_config=base_llm_config)
            assert model.client is not None
            # Check for the correct class, assuming llm_router is importable
            assert isinstance(model.client, LLMClient)
        except Exception as e:
            pytest.fail(f"LLMClient initialization failed: {e}")

    @pytest.mark.asyncio
    async def test_agenerate_standard_provider(self, base_llm_config):
        """
        Tests a real, non-streaming call to the standard provider.
        """
        model = SprinklrChatModel(llm_config=base_llm_config)
        messages = [HumanMessage(content="Hello there!")]
        
        result = await model._agenerate(messages)

        assert len(result.generations[0].message.content) > 0
        assert result.llm_output["token_usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_astream_standard_provider(self, base_llm_config):
        """
        Tests a real, streaming call to the standard provider.
        """
        model = SprinklrChatModel(llm_config=base_llm_config)
        messages = [HumanMessage(content="Hey there!!")]
        
        chunks = []
        async for chunk in model._astream(messages):
            assert isinstance(chunk, ChatGenerationChunk)
            assert isinstance(chunk.message, AIMessageChunk)
            chunks.append(chunk)

        assert len(chunks) > 1

    @pytest.mark.asyncio
    async def test_astream_with_tools(self, base_llm_config):
        """
        Tests that streaming with tool binding produces valid chunks.
        Uses bind_tools with tool_choice to force the model to make a tool call.
        """
        class GetWeather(BaseModel):
            """Gets the weather for a location."""
            location: str = Field(description="The city and state")

        model = SprinklrChatModel(llm_config=base_llm_config).with_structured_output(GetWeather)

        messages = [HumanMessage(content="What's the weather in Boston?")]

        chunks = []
        async for chunk in model.astream(messages):
            assert isinstance(chunk, AIMessageChunk), "Message should be AIMessageChunk"
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive streaming chunks"
        assert chunks[0].tool_call_chunks[0]['name'] == 'GetWeather'

    @pytest.mark.asyncio
    async def test_with_structured_output(self, base_llm_config):
        """
        Ensures the LangChain-standard `.with_structured_output` method
        actually calls the tool by making a real API call.
        """
        
        class GetWeather(BaseModel):
            """Gets the weather."""
            location: str = Field(description="The city and state")

        model = SprinklrChatModel(llm_config=base_llm_config)
        bound_structured = model.with_structured_output(GetWeather)
        
        messages = [HumanMessage(content="What's the weather in Boston?")]
        
        result = await bound_structured.ainvoke(messages)
        
        # Verify that the tool was actually called
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0]['name'] == 'GetWeather'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--asyncio-mode=auto"])