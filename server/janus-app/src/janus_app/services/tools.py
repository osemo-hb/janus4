"""
Tool definitions and types for agentic chat.
"""

from enum import Enum
from typing import List, Optional, TypedDict


class ToolName(str, Enum):
    """Available tool names for function calling."""
    TAVILY_SEARCH = "tavily_search"
    TAVILY_EXTRACT = "tavily_extract"


class SearchArguments(TypedDict, total=False):
    """Arguments for tavily_search tool."""
    query: str
    search_depth: str
    max_results: int


class ExtractArguments(TypedDict, total=False):
    """Arguments for tavily_extract tool."""
    urls: List[str]


class ToolCallFunction(TypedDict):
    """OpenAI tool call function structure."""
    name: str
    arguments: str


class ToolCall(TypedDict):
    """OpenAI tool call structure."""
    id: str
    function: ToolCallFunction


class ChatMessage(TypedDict, total=False):
    """Chat message structure for OpenAI API."""
    role: str
    content: Optional[str]
    tool_call_id: Optional[str]
    tool_calls: Optional[List[ToolCall]]


# OpenAI Function Calling Tool Schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": ToolName.TAVILY_SEARCH.value,
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
            "name": ToolName.TAVILY_EXTRACT.value,
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
