"""
Tests for Tavily Integration and Agentic Chat - Janus 3.5

Tests cover:
- TavilyService (search, extract, mock mode)
- ToolExecutor (tool schemas, execution, error handling)
- AgenticChatHandler (agentic loop, iteration limits)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import sys
import os

# Add janus3 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTavilyService:
    """Test cases for TavilyService."""

    @pytest.mark.asyncio
    async def test_search_returns_mock_when_no_api_key(self):
        """Verify mock response when API key not configured."""
        with patch("janus3.config.settings.TAVILY_API_KEY", ""):
            # Need to reimport to get fresh instance
            from janus3.services.tavily_service import TavilyService

            # Reset singleton
            TavilyService._instance = None
            TavilyService._client = None

            service = TavilyService()
            result = await service.search("test query")

            assert "Mock" in result.get("answer", "")
            assert result.get("results") == []

    @pytest.mark.asyncio
    async def test_search_has_api_property(self):
        """Verify has_api property reflects API availability."""
        with patch("janus3.config.settings.TAVILY_API_KEY", ""):
            from janus3.services.tavily_service import TavilyService

            TavilyService._instance = None
            TavilyService._client = None

            service = TavilyService()
            assert service.has_api is False

    @pytest.mark.asyncio
    async def test_extract_returns_mock_when_no_api_key(self):
        """Verify mock extract response when API key not configured."""
        with patch("janus3.config.settings.TAVILY_API_KEY", ""):
            from janus3.services.tavily_service import TavilyService

            TavilyService._instance = None
            TavilyService._client = None

            service = TavilyService()
            result = await service.extract(["https://example.com"])

            assert len(result.get("results", [])) == 1
            assert "Mock" in result["results"][0].get("raw_content", "")


class TestToolExecutor:
    """Test cases for ToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        """Verify unknown tools return error result."""
        from janus3.core.tool_executor import ToolExecutor
        from janus3.services.tavily_service import TavilyService

        mock_service = MagicMock(spec=TavilyService)
        executor = ToolExecutor(mock_service)

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "unknown_tool"
        mock_tool_call.function.arguments = "{}"

        result = await executor.execute(mock_tool_call)

        assert result.success is False
        assert "Unknown tool" in result.content
        assert result.name == "unknown_tool"

    @pytest.mark.asyncio
    async def test_execute_invalid_json_arguments(self):
        """Verify invalid JSON arguments handled gracefully."""
        from janus3.core.tool_executor import ToolExecutor
        from janus3.services.tavily_service import TavilyService

        mock_service = MagicMock(spec=TavilyService)
        executor = ToolExecutor(mock_service)

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "tavily_search"
        mock_tool_call.function.arguments = "not valid json"

        result = await executor.execute(mock_tool_call)

        assert result.success is False
        assert "Invalid arguments" in result.content

    @pytest.mark.asyncio
    async def test_execute_search_formats_results(self):
        """Verify search results are properly formatted."""
        from janus3.core.tool_executor import ToolExecutor
        from janus3.services.tavily_service import TavilyService

        mock_service = MagicMock(spec=TavilyService)
        mock_service.search = AsyncMock(return_value={
            "answer": "Test answer from Tavily",
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "Content snippet 1"
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "content": "Content snippet 2"
                }
            ]
        })

        executor = ToolExecutor(mock_service)

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "tavily_search"
        mock_tool_call.function.arguments = '{"query": "test query"}'

        result = await executor.execute(mock_tool_call)

        assert result.success is True
        assert "Test answer from Tavily" in result.content
        assert "Result 1" in result.content
        assert "https://example.com/1" in result.content

    @pytest.mark.asyncio
    async def test_execute_extract_formats_results(self):
        """Verify extract results are properly formatted."""
        from janus3.core.tool_executor import ToolExecutor
        from janus3.services.tavily_service import TavilyService

        mock_service = MagicMock(spec=TavilyService)
        mock_service.extract = AsyncMock(return_value={
            "results": [
                {
                    "url": "https://example.com",
                    "raw_content": "Full page content here"
                }
            ]
        })

        executor = ToolExecutor(mock_service)

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "tavily_extract"
        mock_tool_call.function.arguments = '{"urls": ["https://example.com"]}'

        result = await executor.execute(mock_tool_call)

        assert result.success is True
        assert "https://example.com" in result.content
        assert "Full page content" in result.content

    @pytest.mark.asyncio
    async def test_execute_all_processes_multiple_calls(self):
        """Verify execute_all handles multiple tool calls."""
        from janus3.core.tool_executor import ToolExecutor
        from janus3.services.tavily_service import TavilyService

        mock_service = MagicMock(spec=TavilyService)
        mock_service.search = AsyncMock(return_value={
            "answer": "Answer",
            "results": []
        })

        executor = ToolExecutor(mock_service)

        # Create two tool calls
        mock_call_1 = MagicMock()
        mock_call_1.id = "call_1"
        mock_call_1.function.name = "tavily_search"
        mock_call_1.function.arguments = '{"query": "query 1"}'

        mock_call_2 = MagicMock()
        mock_call_2.id = "call_2"
        mock_call_2.function.name = "tavily_search"
        mock_call_2.function.arguments = '{"query": "query 2"}'

        results = await executor.execute_all([mock_call_1, mock_call_2])

        assert len(results) == 2
        assert results[0].tool_call_id == "call_1"
        assert results[1].tool_call_id == "call_2"


class TestAgenticChatHandler:
    """Test cases for AgenticChatHandler."""

    @pytest.mark.asyncio
    async def test_chat_returns_direct_response_when_no_tool_calls(self):
        """Verify direct response when LLM doesn't use tools."""
        from janus3.core.agentic_chat import AgenticChatHandler
        from janus3.services.tavily_service import TavilyService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Direct response without tools"
        mock_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_tavily = MagicMock(spec=TavilyService)
        handler = AgenticChatHandler(mock_client, mock_tavily)

        result = await handler.chat("System prompt", "User message")

        assert result.content == "Direct response without tools"
        assert result.tool_calls_made == 0
        assert result.tools_used == []

    @pytest.mark.asyncio
    async def test_chat_executes_tool_and_continues(self):
        """Verify tool execution and continuation."""
        from janus3.core.agentic_chat import AgenticChatHandler
        from janus3.services.tavily_service import TavilyService

        mock_client = MagicMock()

        # First call returns tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "tavily_search"
        mock_tool_call.function.arguments = '{"query": "test"}'

        tool_response = MagicMock()
        tool_response.choices = [MagicMock()]
        tool_response.choices[0].message.content = None
        tool_response.choices[0].message.tool_calls = [mock_tool_call]

        # Second call returns final response
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "Final response with search results"
        final_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[tool_response, final_response]
        )

        mock_tavily = MagicMock(spec=TavilyService)
        mock_tavily.search = AsyncMock(return_value={
            "answer": "Search answer",
            "results": []
        })

        handler = AgenticChatHandler(mock_client, mock_tavily)

        result = await handler.chat("System prompt", "User message")

        assert result.content == "Final response with search results"
        assert result.tool_calls_made == 1
        assert "tavily_search" in result.tools_used

    @pytest.mark.asyncio
    async def test_chat_respects_max_iterations(self):
        """Verify max iteration limit is enforced."""
        from janus3.core.agentic_chat import AgenticChatHandler
        from janus3.services.tavily_service import TavilyService

        mock_client = MagicMock()

        # Create tool call that always triggers more tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_loop"
        mock_tool_call.function.name = "tavily_search"
        mock_tool_call.function.arguments = '{"query": "endless loop"}'

        tool_response = MagicMock()
        tool_response.choices = [MagicMock()]
        tool_response.choices[0].message.content = None
        tool_response.choices[0].message.tool_calls = [mock_tool_call]

        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "Final response after max iterations"
        final_response.choices[0].message.tool_calls = None

        # Return tool_response for first 3 calls (max iterations), then final
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                tool_response,  # Iteration 1
                tool_response,  # Iteration 2
                tool_response,  # Iteration 3
                final_response  # Final call without tools
            ]
        )

        mock_tavily = MagicMock(spec=TavilyService)
        mock_tavily.search = AsyncMock(return_value={
            "answer": "Answer",
            "results": []
        })

        handler = AgenticChatHandler(mock_client, mock_tavily)

        with patch("janus3.config.settings.TAVILY_MAX_TOOL_ITERATIONS", 3):
            result = await handler.chat("System prompt", "User message")

        assert result.content == "Final response after max iterations"
        assert result.tool_calls_made == 3  # 3 iterations

    @pytest.mark.asyncio
    async def test_chat_with_tools_disabled(self):
        """Verify behavior when tools are disabled."""
        from janus3.core.agentic_chat import AgenticChatHandler
        from janus3.services.tavily_service import TavilyService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response without tools"
        mock_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_tavily = MagicMock(spec=TavilyService)
        handler = AgenticChatHandler(mock_client, mock_tavily)

        result = await handler.chat(
            "System prompt",
            "User message",
            enable_tools=False
        )

        assert result.content == "Response without tools"
        assert result.tool_calls_made == 0

        # Verify tools were not passed to LLM
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs or call_kwargs.get("tools") is None


class TestToolSchemas:
    """Test cases for tool schema definitions."""

    def test_tool_schemas_have_required_fields(self):
        """Verify tool schemas have all required fields."""
        from janus3.core.tool_executor import TOOL_SCHEMAS

        for schema in TOOL_SCHEMAS:
            assert schema.get("type") == "function"
            assert "function" in schema

            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

            params = func["parameters"]
            assert params.get("type") == "object"
            assert "properties" in params
            assert "required" in params

    def test_tavily_search_schema_structure(self):
        """Verify tavily_search schema structure."""
        from janus3.core.tool_executor import TOOL_SCHEMAS

        search_schema = next(
            s for s in TOOL_SCHEMAS
            if s["function"]["name"] == "tavily_search"
        )

        props = search_schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert "search_depth" in props
        assert "max_results" in props

        assert props["query"]["type"] == "string"
        assert props["search_depth"]["enum"] == ["basic", "advanced"]
        assert props["max_results"]["type"] == "integer"

    def test_tavily_extract_schema_structure(self):
        """Verify tavily_extract schema structure."""
        from janus3.core.tool_executor import TOOL_SCHEMAS

        extract_schema = next(
            s for s in TOOL_SCHEMAS
            if s["function"]["name"] == "tavily_extract"
        )

        props = extract_schema["function"]["parameters"]["properties"]
        assert "urls" in props
        assert props["urls"]["type"] == "array"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
