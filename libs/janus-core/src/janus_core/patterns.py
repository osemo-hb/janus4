"""
Common patterns for Janus Core.

Provides reusable decorators and utilities to reduce boilerplate.
"""

import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def singleton(cls: Type[T]) -> Type[T]:
    """
    Singleton decorator for classes.

    Ensures only one instance of the class exists. Thread-safe via class attribute.

    Usage:
        @singleton
        class MyService:
            def __init__(self):
                self.client = SomeClient()
    """
    _instances: dict = {}

    @wraps(cls, updated=[])
    class SingletonWrapper(cls):
        def __new__(wrapper_cls, *args, **kwargs):
            if cls not in _instances:
                instance = super().__new__(wrapper_cls)
                _instances[cls] = instance
            return _instances[cls]

    return SingletonWrapper


def parse_json_response(
    response: str,
    fallback: T,
    context: str = "JSON parse"
) -> T:
    """
    Parse JSON response with error handling.

    Args:
        response: JSON string to parse.
        fallback: Value to return on parse failure.
        context: Description for logging on error.

    Returns:
        Parsed JSON or fallback value.
    """
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"{context} failed: {e}")
        return fallback
