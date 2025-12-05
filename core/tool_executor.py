"""
Tool Executor - Janus 3.5

Handles execution of tools during agentic chat loop.
Defines tool schemas and executes tool calls from OpenAI function calling.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from janus3.services.tavily_service import TavilyService, TavilyAPIError

logger = logging.getLogger(__name__)


# OpenAI Function Calling Tool Schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "Search the web for current information. Use this when the user asks about "
                "recent events, news, real-time data, or information you don't have in your "
                "knowledge base. Returns search results with titles, URLs, and content snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant web content."
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": (
                            "Search depth. 'basic' is faster and suitable for simple queries. "
                            "'advanced' is more thorough for complex topics. Default: basic"
                        )
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-10). Default: 5",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_extract",
            "description": (
                "Extract the full content from specific URLs. Use this when you need detailed "
                "information from a particular webpage that was mentioned or found in search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to extract content from (max 5).",
                        "maxItems": 5
                    }
                },
                "required": ["urls"]
            }
        }
    }
]


@dataclass
class ToolResult:
    """Result of tool execution."""
    tool_call_id: str
    name: str
    content: str
    success: bool
    error: Optional[str] = None


class ToolExecutor:
    """
    Executes tools based on LLM tool calls.

    Supports:
    - tavily_search: Web search via Tavily API
    - tavily_extract: URL content extraction via Tavily API
    """

    def __init__(self, tavily_service: TavilyService):
        """
        Initialize tool executor.

        Args:
            tavily_service: TavilyService instance for web search operations.
        """
        self.tavily = tavily_service

    async def execute(self, tool_call: Any) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: OpenAI tool call object with:
                - id: Tool call ID
                - function.name: Tool name
                - function.arguments: JSON string of arguments

        Returns:
            ToolResult with execution outcome.
        """
        tool_name = tool_call.function.name
        tool_call_id = tool_call.id

        # Parse arguments
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON arguments for {tool_name}: {e}")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=f"Error: Invalid arguments format - {str(e)}",
                success=False,
                error=str(e)
            )

        # Execute the appropriate tool
        try:
            if tool_name == "tavily_search":
                result = await self._execute_search(arguments)
            elif tool_name == "tavily_extract":
                result = await self._execute_extract(arguments)
            else:
                logger.warning(f"Unknown tool requested: {tool_name}")
                return ToolResult(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=f"Error: Unknown tool '{tool_name}'",
                    success=False,
                    error=f"Unknown tool: {tool_name}"
                )

            logger.info(f"Tool {tool_name} executed successfully")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=result,
                success=True
            )

        except TavilyAPIError as e:
            logger.error(f"Tavily API error in {tool_name}: {e}")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=f"Search API error: {str(e)}",
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.exception(f"Unexpected error in tool {tool_name}")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=f"Unexpected error: {str(e)}",
                success=False,
                error=str(e)
            )

    async def execute_all(self, tool_calls: List[Any]) -> List[ToolResult]:
        """
        Execute multiple tool calls sequentially.

        Args:
            tool_calls: List of OpenAI tool call objects.

        Returns:
            List of ToolResult objects in the same order.
        """
        results = []
        for tool_call in tool_calls:
            result = await self.execute(tool_call)
            results.append(result)
        return results

    async def _execute_search(self, arguments: Dict[str, Any]) -> str:
        """
        Execute tavily_search tool.

        Args:
            arguments: Dict with 'query', optional 'search_depth', 'max_results'

        Returns:
            Formatted search results as string for LLM consumption.
        """
        query = arguments.get("query", "")
        search_depth = arguments.get("search_depth", "basic")
        max_results = arguments.get("max_results", 5)

        logger.debug(f"Executing search: query='{query}', depth={search_depth}, max={max_results}")

        response = await self.tavily.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results
        )

        # Format results for LLM consumption
        output_parts = []

        # Include summary answer if available
        if response.get("answer"):
            output_parts.append(f"Summary: {response['answer']}\n")

        # Format individual results
        results = response.get("results", [])
        if results:
            output_parts.append("Search Results:")
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "N/A")
                content = result.get("content", "")

                output_parts.append(f"\n[{i}] {title}")
                output_parts.append(f"    URL: {url}")

                if content:
                    # Truncate long content to avoid token limits
                    if len(content) > 500:
                        content = content[:500] + "..."
                    output_parts.append(f"    Content: {content}")
        else:
            output_parts.append("No results found for this search query.")

        return "\n".join(output_parts)

    async def _execute_extract(self, arguments: Dict[str, Any]) -> str:
        """
        Execute tavily_extract tool.

        Args:
            arguments: Dict with 'urls' list

        Returns:
            Formatted extracted content as string for LLM consumption.
        """
        urls = arguments.get("urls", [])

        if not urls:
            return "Error: No URLs provided for extraction."

        # Limit to 5 URLs
        urls = urls[:5]

        logger.debug(f"Executing extract for {len(urls)} URLs")

        response = await self.tavily.extract(urls=urls)

        # Format results
        output_parts = ["Extracted Content:"]

        for result in response.get("results", []):
            url = result.get("url", "Unknown URL")
            # Try raw_content first, then content
            content = result.get("raw_content", result.get("content", "No content extracted"))

            # Truncate very long content to avoid token limits
            if len(content) > 2000:
                content = content[:2000] + "\n... [content truncated]"

            output_parts.append(f"\n--- {url} ---")
            output_parts.append(content)

        if not response.get("results"):
            output_parts.append("\nNo content could be extracted from the provided URLs.")

        return "\n".join(output_parts)
