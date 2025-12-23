"""
LLM Service

Wrapper around OpenAI API for chat completions.
"""

import json
from typing import List, Dict, Optional, Any

from openai import AsyncOpenAI

from janus_core.config import settings
from janus_core.patterns import singleton


@singleton
class LLMService:
    """Async OpenAI service for chat completions."""

    _client: Optional[AsyncOpenAI] = None

    def __init__(self):
        if LLMService._client is None and settings.OPENAI_API_KEY:
            LLMService._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        """Get the OpenAI client."""
        return LLMService._client

    @property
    def has_api(self) -> bool:
        """Check if API is available."""
        return LLMService._client is not None

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a chat completion.

        Args:
            system_prompt: System prompt with context.
            user_message: User's message.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Generated response text.
        """
        if not self.has_api:
            return self._mock_response(user_message)

        response = await self.client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    async def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a chat completion with full message history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Generated response text.
        """
        if not self.has_api:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "test"
            )
            return self._mock_response(last_user)

        response = await self.client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Generate a chat completion with optional tool calling.

        Args:
            messages: List of message dicts (system, user, assistant, tool).
            tools: Optional list of tool schemas for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Full ChatCompletion response object (not just content).
            Access response.choices[0].message.tool_calls for tool invocations.
        """
        if not self.has_api:
            return self._mock_tool_response(messages)

        kwargs = {
            "model": settings.GENERATION_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return await self.client.chat.completions.create(**kwargs)

    def _mock_tool_response(self, messages: List[Dict]) -> Any:
        """Generate a mock response for tool-enabled chat."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[Mock Response] API key not configured."
        mock_response.choices[0].message.tool_calls = None
        return mock_response

    async def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a JSON response.

        Uses json_object response format when available.

        Args:
            system_prompt: System prompt (should mention JSON output).
            user_content: Content to analyze.
            temperature: Sampling temperature (lower for JSON).

        Returns:
            JSON string response.
        """
        if not self.has_api:
            return self._mock_json_response()

        response = await self.client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=2000
        )

        return response.choices[0].message.content

    async def extract_facts(self, conversation_text: str) -> Dict[str, Any]:
        """
        Extract facts from conversation using LLM.

        Args:
            conversation_text: Formatted conversation text.

        Returns:
            Dict with 'summary' and 'facts' keys.
        """
        system_prompt = """Analyze this conversation and extract key information.

Return JSON with:
1. "summary": A concise narrative summary (1-2 sentences).
2. "facts": List of permanent facts learned. Each fact should have:
   - "subject": The entity the fact is about
   - "predicate": The relationship/property
   - "object": The value
   - "confidence": How certain (0.0-1.0)

Focus on facts useful for future conversations:
- User preferences and interests
- Names, relationships, locations
- Technical details mentioned
- Commitments or plans"""

        json_str = await self.generate_json(system_prompt, conversation_text)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "summary": "Conversation summary unavailable.",
                "facts": []
            }

    def _mock_response(self, user_message: str) -> str:
        """Generate a mock response when API is unavailable."""
        return f"[Mock Response] I received your message: '{user_message[:50]}...'. API key not configured."

    def _mock_json_response(self) -> str:
        """Generate a mock JSON response when API is unavailable."""
        return json.dumps({
            "summary": "Mock conversation summary.",
            "facts": [
                {
                    "subject": "User",
                    "predicate": "tested",
                    "object": "mock mode",
                    "confidence": 0.5
                }
            ]
        })


# Dependency injection helper
def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    return LLMService()
