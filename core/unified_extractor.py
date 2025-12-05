"""
Unified LLM Extractor for Janus 3.5

Single LLM call that extracts:
- Entities (typed: person, organization, location, concept, product, event)
- Facts (subject, predicate, object triples with canonical predicates)
- Summary + Compressed State
- Topic Label

Replaces: GLiNER EntityLinker + spaCy SpaCyExtractor + LLMFactEnricher
"""

import json
import logging
from dataclasses import dataclass, field
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
    """Fact extracted from text with canonical predicate."""
    subject: str
    predicate: str
    canonical_predicate: str  # schema.org-style normalized predicate
    object_value: str
    source_span: Optional[str] = None  # Original text for provenance


@dataclass
class EpisodeState:
    """Compressed episode state for context efficiency."""
    compressed: str  # 20-40 word abstraction
    participants: List[str] = field(default_factory=list)  # Canonical entity names
    topic_label: str = ""  # 2-3 word topic


@dataclass
class ExtractionResult:
    """Result of unified extraction."""
    entities: List[ExtractedEntity]
    facts: List[ExtractedFact]
    summary: str
    episode_state: Optional[EpisodeState] = None


# Canonical predicate mapping (raw -> schema.org style)
CANONICAL_PREDICATES = {
    # Employment
    "works_at": "worksFor",
    "works_for": "worksFor",
    "employed_by": "worksFor",
    "employed_at": "worksFor",
    "job_at": "worksFor",
    "role_at": "hasRole",
    "position_at": "hasRole",
    # Location
    "lives_in": "residesIn",
    "lives_at": "residesIn",
    "located_in": "locatedIn",
    "based_in": "locatedIn",
    "from": "birthPlace",
    "born_in": "birthPlace",
    # Relationships
    "knows": "knows",
    "friend_of": "knows",
    "colleague_of": "colleagueOf",
    "married_to": "spouse",
    "parent_of": "parentOf",
    "child_of": "childOf",
    # Skills & Abilities
    "can_do": "hasSkill",
    "knows_how": "hasSkill",
    "skilled_in": "hasSkill",
    "expert_in": "hasSkill",
    "uses": "uses",
    "uses_tool": "uses",
    # Preferences
    "likes": "hasPreference",
    "prefers": "hasPreference",
    "enjoys": "hasPreference",
    "dislikes": "hasNegativePreference",
    "avoids": "hasNegativePreference",
    # Ownership
    "owns": "owns",
    "has": "owns",
    "possesses": "owns",
    # Education
    "studied_at": "alumniOf",
    "graduated_from": "alumniOf",
    "degree_in": "hasCredential",
    # Plans & Intent
    "wants_to": "intends",
    "plans_to": "intends",
    "will": "intends",
    "interested_in": "hasInterest",
    # Communication
    "email": "email",
    "phone": "telephone",
    "contact": "contactPoint",
    # Membership
    "member_of": "memberOf",
    "belongs_to": "memberOf",
    "part_of": "partOf",
}


EXTRACTION_SYSTEM_PROMPT = """You are an information extraction assistant. Analyze the conversation and extract:

1. ENTITIES: Named entities with their types
   Types: person, organization, location, concept, product, event

2. FACTS: Subject-predicate-object triples representing key information
   Use CANONICAL predicates from this list when possible:
   - worksFor, hasRole (employment)
   - residesIn, locatedIn, birthPlace (location)
   - knows, colleagueOf, spouse, parentOf, childOf (relationships)
   - hasSkill, uses (abilities)
   - hasPreference, hasNegativePreference (likes/dislikes)
   - owns, alumniOf, hasCredential (ownership/education)
   - intends, hasInterest (plans)
   - memberOf, partOf (membership)

3. COMPRESSED STATE: A 20-40 word abstraction capturing the key state of this conversation segment.
   This should be a highly compressed summary that captures the essential information.

4. TOPIC LABEL: A 2-3 word label for the topic being discussed.

5. SUMMARY: A concise 1-2 sentence summary.

Return JSON in this exact format:
{
    "entities": [
        {"name": "John Smith", "type": "person"},
        {"name": "Acme Corp", "type": "organization"}
    ],
    "facts": [
        {
            "subject": "John Smith",
            "predicate": "worksFor",
            "object": "Acme Corp",
            "source_span": "I work at Acme Corp"
        }
    ],
    "episode_state": {
        "compressed": "John discussed his role at Acme Corp as a senior developer. He prefers Python and works remotely from Seattle.",
        "participants": ["John Smith", "Acme Corp"],
        "topic_label": "work discussion"
    },
    "summary": "User discussed their work at Acme Corp and development preferences."
}

Guidelines:
- Use canonical predicate names from the list above
- Extract source_span: the original text that supports each fact
- Compressed state should be 20-40 words capturing key information
- participants should list canonical entity names involved
- topic_label should be 2-3 words describing the conversation topic"""


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
            ExtractionResult with entities, facts, summary, and episode_state.
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

            # Parse facts with canonical predicates
            facts = []
            for f in data.get("facts", []):
                subject = f.get("subject", "").strip()
                predicate = f.get("predicate", "").strip()
                obj = f.get("object", "").strip()
                source_span = f.get("source_span", "").strip() or None

                # All three core fields required
                if not subject or not predicate or not obj:
                    continue

                # Canonicalize predicate
                canonical = self._canonicalize_predicate(predicate)

                facts.append(ExtractedFact(
                    subject=subject,
                    predicate=predicate,
                    canonical_predicate=canonical,
                    object_value=obj,
                    source_span=source_span,
                ))

            summary = data.get("summary", "").strip()

            # Parse episode state
            episode_state = None
            state_data = data.get("episode_state", {})
            if state_data:
                episode_state = EpisodeState(
                    compressed=state_data.get("compressed", "").strip(),
                    participants=state_data.get("participants", []),
                    topic_label=state_data.get("topic_label", "").strip(),
                )

            logger.debug(
                f"Extracted {len(entities)} entities, {len(facts)} facts"
            )

            return ExtractionResult(
                entities=entities,
                facts=facts,
                summary=summary,
                episode_state=episode_state,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return ExtractionResult(entities=[], facts=[], summary="")

    def _canonicalize_predicate(self, raw_predicate: str) -> str:
        """
        Map raw predicate to canonical schema.org-style form.

        Args:
            raw_predicate: The raw predicate from extraction.

        Returns:
            Canonical predicate name.
        """
        # Normalize: lowercase, replace spaces with underscores
        normalized = raw_predicate.lower().replace(" ", "_").strip("_")

        # Look up in mapping
        if normalized in CANONICAL_PREDICATES:
            return CANONICAL_PREDICATES[normalized]

        # If not found, convert to camelCase
        return self._to_camel_case(normalized)

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split('_')
        # First component stays lowercase, rest get capitalized
        return components[0] + ''.join(x.title() for x in components[1:])

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
                    canonical_predicate="tested",
                    object_value="mock mode",
                    source_span="API unavailable",
                ),
            ],
            summary="Mock conversation summary (API unavailable).",
            episode_state=EpisodeState(
                compressed="Mock extraction in test mode without API access.",
                participants=["User"],
                topic_label="test mode",
            ),
        )

    @staticmethod
    def get_entity_types() -> List[str]:
        """Get supported entity types."""
        return ENTITY_TYPES.copy()

    @staticmethod
    def get_canonical_predicates() -> dict:
        """Get canonical predicate mappings."""
        return CANONICAL_PREDICATES.copy()


def get_unified_extractor() -> UnifiedExtractor:
    """Get unified extractor instance (dependency injection helper)."""
    return UnifiedExtractor()
