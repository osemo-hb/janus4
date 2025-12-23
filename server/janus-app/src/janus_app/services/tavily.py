"""
Tavily Search Service - Janus 3.5

Async wrapper around Tavily API for web search and content extraction.
Follows singleton pattern consistent with LLMService.
"""

import logging
from typing import Optional, List, Dict, Any

import httpx

from janus_app.config import app_settings as settings

logger = logging.getLogger(__name__)


class TavilyAPIError(Exception):
    """Custom exception for Tavily API errors."""
    pass


class TavilyService:
    """
    Async Tavily API client.

    Uses singleton pattern consistent with LLMService.
    Provides web search and URL content extraction capabilities.
    """

    _instance: Optional["TavilyService"] = None
    _client: Optional[httpx.AsyncClient] = None

    TAVILY_API_URL = "https://api.tavily.com"

    def __new__(cls) -> "TavilyService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            if settings.TAVILY_API_KEY:
                TavilyService._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    headers={"Content-Type": "application/json"}
                )
                logger.info("TavilyService initialized with API key")
            else:
                logger.warning("TavilyService: No API key configured, mock mode enabled")

    @property
    def has_api(self) -> bool:
        """Check if API is available."""
        return bool(settings.TAVILY_API_KEY) and TavilyService._client is not None

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform web search via Tavily API.

        Args:
            query: Search query string.
            search_depth: "basic" (faster) or "advanced" (more thorough).
            max_results: Maximum number of results to return (1-10).
            include_domains: Optional list of domains to include.
            exclude_domains: Optional list of domains to exclude.

        Returns:
            Dict with 'answer' (summary) and 'results' list containing
            title, url, content, score for each result.

        Raises:
            TavilyAPIError: If the API call fails.
        """
        if not self.has_api:
            logger.debug(f"Tavily mock search: {query}")
            return self._mock_search_response(query)

        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": True,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            response = await self._client.post(
                f"{self.TAVILY_API_URL}/search",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Tavily search returned {len(result.get('results', []))} results")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily search failed: {e.response.status_code} - {e.response.text}")
            raise TavilyAPIError(f"Search failed: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Tavily request error: {e}")
            raise TavilyAPIError(f"Request failed: {str(e)}")

    async def extract(
        self,
        urls: List[str],
    ) -> Dict[str, Any]:
        """
        Extract content from URLs via Tavily Extract API.

        Args:
            urls: List of URLs to extract content from (max 5).

        Returns:
            Dict with 'results' containing extracted content per URL.

        Raises:
            TavilyAPIError: If the API call fails.
        """
        if not self.has_api:
            logger.debug(f"Tavily mock extract: {urls}")
            return self._mock_extract_response(urls)

        # Limit to 5 URLs as per Tavily API limits
        urls = urls[:5]

        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "urls": urls,
        }

        try:
            response = await self._client.post(
                f"{self.TAVILY_API_URL}/extract",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Tavily extract returned {len(result.get('results', []))} results")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily extract failed: {e.response.status_code} - {e.response.text}")
            raise TavilyAPIError(f"Extract failed: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Tavily request error: {e}")
            raise TavilyAPIError(f"Request failed: {str(e)}")

    def _mock_search_response(self, query: str) -> Dict[str, Any]:
        """Mock response when API unavailable."""
        return {
            "answer": f"[Mock] Web search is not configured. Query was: '{query}'",
            "results": [],
        }

    def _mock_extract_response(self, urls: List[str]) -> Dict[str, Any]:
        """Mock response when API unavailable."""
        return {
            "results": [
                {"url": url, "raw_content": "[Mock] Content extraction is not configured."}
                for url in urls
            ],
        }

    async def close(self):
        """Close HTTP client gracefully."""
        if TavilyService._client:
            await TavilyService._client.aclose()
            TavilyService._client = None
            logger.info("TavilyService HTTP client closed")


def get_tavily_service() -> TavilyService:
    """Get Tavily service instance (dependency injection helper)."""
    return TavilyService()
