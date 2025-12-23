"""
Agentic Chat Handler - Janus 3.5

Tool-calling loop for chat with web search capabilities.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from janus_app.config import app_settings as settings
from janus_app.services.tool_executor import ToolExecutor
from janus_app.services.tools import TOOL_SCHEMAS, ChatMessage
from janus_app.services.tavily import TavilyService

logger = logging.getLogger(__name__)


@dataclass
class AgenticResponse:
    """Response from agentic chat."""
    content: str
    tool_calls_made: int
    tools_used: List[str] = field(default_factory=list)


class AgenticChatHandler:
    """Chat handler with agentic tool-calling loop."""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        tavily_service: TavilyService
    ):
        """
        Initialize agentic chat handler.

        Args:
            openai_client: AsyncOpenAI client instance.
            tavily_service: TavilyService for web search operations.
        """
        self.client = openai_client
        self.tool_executor = ToolExecutor(tavily_service)

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        enable_tools: bool = True
    ) -> AgenticResponse:
        """
        Execute agentic chat with optional tool calling.

        Args:
            system_prompt: System prompt with context from retrieval.
            user_message: User's message.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum response tokens.
            enable_tools: Whether to enable tool calling.

        Returns:
            AgenticResponse with final content and tool usage statistics.
        """
        messages: List[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        tools_used: List[str] = []
        max_iterations = settings.TAVILY_MAX_TOOL_ITERATIONS
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM with or without tools
            response = await self._call_llm(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=TOOL_SCHEMAS if enable_tools else None
            )

            message = response.choices[0].message

            # Check for tool calls
            if not message.tool_calls:
                # No tool calls - return final response
                logger.debug(
                    f"Agentic chat complete after {iteration} iteration(s), "
                    f"{len(tools_used)} tool call(s)"
                )
                return AgenticResponse(
                    content=message.content or "",
                    tool_calls_made=len(tools_used),
                    tools_used=tools_used
                )

            # Process tool calls
            logger.info(
                f"Iteration {iteration}: Processing {len(message.tool_calls)} tool call(s)"
            )

            # Append assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute all tool calls
            results = await self.tool_executor.execute_all(message.tool_calls)

            # Append tool results to messages
            for result in results:
                tools_used.append(result.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content
                })

                if not result.success:
                    logger.warning(f"Tool {result.name} failed: {result.error}")

        # Max iterations reached - make one final call without tools
        logger.warning(
            f"Max tool iterations ({max_iterations}) reached, "
            f"making final call without tools"
        )

        final_response = await self._call_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None  # Disable tools for final response
        )

        return AgenticResponse(
            content=final_response.choices[0].message.content or "",
            tool_calls_made=len(tools_used),
            tools_used=tools_used
        )

    async def _call_llm(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ):
        """Make OpenAI API call."""
        kwargs: Dict[str, Any] = {
            "model": settings.GENERATION_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return await self.client.chat.completions.create(**kwargs)
