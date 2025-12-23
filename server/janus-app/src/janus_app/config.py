"""
Janus App Configuration

Application-specific settings (server, external services, observability).
Core settings are inherited from janus-core.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional


class AppSettings(BaseSettings):
    """Application layer settings loaded from environment variables."""

    # ============================================================
    # Server Configuration
    # ============================================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # ============================================================
    # Tavily (Web Search)
    # ============================================================
    TAVILY_API_KEY: str = ""
    TAVILY_ENABLED: bool = True
    TAVILY_DEFAULT_SEARCH_DEPTH: str = "basic"
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_MAX_TOOL_ITERATIONS: int = 3

    # ============================================================
    # OpenTelemetry (SDK Configuration)
    # ============================================================
    OTEL_ENABLED: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    OTEL_SERVICE_NAME: str = "janus-app"
    OTEL_CONSOLE_EXPORT: bool = False

    # ============================================================
    # Prometheus
    # ============================================================
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_MULTIPROC_DIR: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
app_settings = AppSettings()
