"""
Unified LLM Extractor for Janus 3.5

Single LLM call that extracts:
- Entities (typed: person, organization, location, concept, product, event)
- Facts (subject, predicate, object triples)
- Summary

Replaces: GLiNER EntityLinker + spaCy SpaCyExtractor + LLMFactEnricher
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from janus3.services.llm_service import LLMService
from janus3.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    name: str
    entity_type: str  # person, organization, location, concept, product, event


@dataclass
class ExtractedFact:
    """Fact extracted from text."""
    subject: str
    predicate: str
    object_value: str


@dataclass
class ExtractionResult:
    """Result of unified extraction."""
    entities: List[ExtractedEntity]
    facts: List[ExtractedFact]
    summary: str


EXTRACTION_SYSTEM_PROMPT = """You are an information extraction assistant. Analyze the conversation and extract:

1. ENTITIES: Named entities with their types
   Types must be one of: person, organization, location, concept, product, event

2. FACTS: Subject-predicate-object triples representing key information
   Focus on:
   - User preferences and interests
   - Names, relationships, locations
   - Technical details mentioned
   - Commitments or plans

3. SUMMARY: A concise 1-2 sentence summary of the conversation

Return JSON in this exact format:
{
    "entities": [
        {"name": "John Smith", "type": "person"},
        {"name": "Acme Corp", "type": "organization"}
    ],
    "facts": [
        {"subject": "John Smith", "predicate": "works_at", "object": "Acme Corp"},
        {"subject": "User", "predicate": "prefers", "object": "vegetarian food"}
    ],
    "summary": "User discussed their work at Acme Corp and dietary preferences."
}

Guidelines:
- Extract only meaningful, persistent information useful for future conversations
- Use consistent entity names (prefer full names over pronouns)
- Use clear, lowercase predicates with underscores (e.g., works_at, prefers, lives_in)
- The subject of a fact should match an extracted entity name when possible"""


ENTITY_TYPES = ["person", "organization", "location", "concept", "product", "event"]


class UnifiedExtractor:
    """
    Unified LLM-based extraction for entities, facts, and summary.

    Singleton pattern for LLM service reuse.
    """

    _instance: Optional["UnifiedExtractor"] = None

    def __new__(cls) -> "UnifiedExtractor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.llm = LLMService()
            self._initialized = True

    async def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities, facts, and summary from text.

        Args:
            text: Conversation text to analyze.

        Returns:
            ExtractionResult with entities, facts, and summary.
        """
        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty extraction")
            return ExtractionResult(entities=[], facts=[], summary="")

        if not self.llm.has_api:
            logger.debug("LLM API unavailable, returning mock extraction")
            return self._mock_extraction()

        try:
            # Truncate to avoid token limits (approximately 4000 chars = ~1000 tokens)
            truncated_text = text[:4000] if len(text) > 4000 else text

            response = await self.llm.generate_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_content=f"Conversation:\n{truncated_text}",
                temperature=0.1,  # Low temperature for consistent extraction
            )

            return self._parse_response(response)

        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            return ExtractionResult(entities=[], facts=[], summary="")

    def _parse_response(self, response: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        try:
            data = json.loads(response)

            # Parse entities with validation
            entities = []
            for e in data.get("entities", []):
                name = e.get("name", "").strip()
                entity_type = e.get("type", "concept").lower()

                if not name:
                    continue

                # Validate entity type
                if entity_type not in ENTITY_TYPES:
                    entity_type = "concept"

                entities.append(ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                ))

            # Parse facts with validation
            facts = []
            for f in data.get("facts", []):
                subject = f.get("subject", "").strip()
                predicate = f.get("predicate", "").strip()
                obj = f.get("object", "").strip()

                # All three fields required
                if not subject or not predicate or not obj:
                    continue

                # Normalize predicate (lowercase, underscores)
                predicate = predicate.lower().replace(" ", "_")

                facts.append(ExtractedFact(
                    subject=subject,
                    predicate=predicate,
                    object_value=obj,
                ))

            summary = data.get("summary", "").strip()

            logger.debug(
                f"Extracted {len(entities)} entities, {len(facts)} facts"
            )

            return ExtractionResult(
                entities=entities,
                facts=facts,
                summary=summary,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return ExtractionResult(entities=[], facts=[], summary="")

    def _mock_extraction(self) -> ExtractionResult:
        """Return mock extraction when API is unavailable."""
        return ExtractionResult(
            entities=[
                ExtractedEntity(name="User", entity_type="person"),
            ],
            facts=[
                ExtractedFact(
                    subject="User",
                    predicate="tested",
                    object_value="mock mode",
                ),
            ],
            summary="Mock conversation summary (API unavailable).",
        )

    @staticmethod
    def get_entity_types() -> List[str]:
        """Get supported entity types."""
        return ENTITY_TYPES.copy()


def get_unified_extractor() -> UnifiedExtractor:
    """Get unified extractor instance (dependency injection helper)."""
    return UnifiedExtractor()
