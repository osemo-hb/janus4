"""
Tests for Unified Extractor (GPT-5.1 Upgrades 1 & 3)

Tests for:
- Compressed episodic state model (Upgrade 1)
- Canonicalized facts via LLM (Upgrade 3)
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_extractor import (
    UnifiedExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    EpisodeState,
    CANONICAL_PREDICATES,
    ENTITY_TYPES,
    get_unified_extractor,
)


class TestExtractedEntity:
    """Test ExtractedEntity dataclass."""

    def test_creation(self):
        """Test basic creation."""
        entity = ExtractedEntity(
            name="John Smith",
            entity_type="person"
        )

        assert entity.name == "John Smith"
        assert entity.entity_type == "person"


class TestExtractedFact:
    """Test ExtractedFact dataclass."""

    def test_creation_with_canonical(self):
        """Test fact with canonical predicate."""
        fact = ExtractedFact(
            subject="John",
            predicate="works_at",
            canonical_predicate="worksFor",
            object_value="Acme Corp"
        )

        assert fact.subject == "John"
        assert fact.predicate == "works_at"
        assert fact.canonical_predicate == "worksFor"
        assert fact.object_value == "Acme Corp"
        assert fact.source_span is None

    def test_creation_with_source_span(self):
        """Test fact with source span for provenance."""
        fact = ExtractedFact(
            subject="John",
            predicate="lives_in",
            canonical_predicate="residesIn",
            object_value="Seattle",
            source_span="I live in Seattle"
        )

        assert fact.source_span == "I live in Seattle"


class TestEpisodeState:
    """Test EpisodeState dataclass."""

    def test_creation(self):
        """Test basic creation."""
        state = EpisodeState(
            compressed="User discussed Python development and debugging techniques.",
            participants=["User", "Assistant"],
            topic_label="python debugging"
        )

        assert "Python" in state.compressed
        assert len(state.participants) == 2
        assert state.topic_label == "python debugging"

    def test_default_fields(self):
        """Test default factory fields."""
        state = EpisodeState(compressed="Test")

        assert state.participants == []
        assert state.topic_label == ""


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = ExtractionResult(
            entities=[ExtractedEntity(name="John", entity_type="person")],
            facts=[ExtractedFact(
                subject="John",
                predicate="worksFor",
                canonical_predicate="worksFor",
                object_value="Acme"
            )],
            summary="Test summary"
        )

        assert len(result.entities) == 1
        assert len(result.facts) == 1
        assert result.summary == "Test summary"
        assert result.episode_state is None

    def test_creation_with_episode_state(self):
        """Test with episode state."""
        result = ExtractionResult(
            entities=[],
            facts=[],
            summary="Test",
            episode_state=EpisodeState(
                compressed="Compressed test state",
                topic_label="test"
            )
        )

        assert result.episode_state is not None
        assert result.episode_state.compressed == "Compressed test state"


class TestCanonicalPredicates:
    """Test canonical predicate mapping."""

    def test_employment_predicates(self):
        """Test employment-related predicates."""
        assert CANONICAL_PREDICATES["works_at"] == "worksFor"
        assert CANONICAL_PREDICATES["works_for"] == "worksFor"
        assert CANONICAL_PREDICATES["employed_by"] == "worksFor"
        assert CANONICAL_PREDICATES["role_at"] == "hasRole"

    def test_location_predicates(self):
        """Test location-related predicates."""
        assert CANONICAL_PREDICATES["lives_in"] == "residesIn"
        assert CANONICAL_PREDICATES["lives_at"] == "residesIn"
        assert CANONICAL_PREDICATES["located_in"] == "locatedIn"
        assert CANONICAL_PREDICATES["born_in"] == "birthPlace"

    def test_relationship_predicates(self):
        """Test relationship predicates."""
        assert CANONICAL_PREDICATES["knows"] == "knows"
        assert CANONICAL_PREDICATES["friend_of"] == "knows"
        assert CANONICAL_PREDICATES["married_to"] == "spouse"

    def test_skill_predicates(self):
        """Test skill-related predicates."""
        assert CANONICAL_PREDICATES["can_do"] == "hasSkill"
        assert CANONICAL_PREDICATES["skilled_in"] == "hasSkill"
        assert CANONICAL_PREDICATES["expert_in"] == "hasSkill"

    def test_preference_predicates(self):
        """Test preference predicates."""
        assert CANONICAL_PREDICATES["likes"] == "hasPreference"
        assert CANONICAL_PREDICATES["prefers"] == "hasPreference"
        assert CANONICAL_PREDICATES["dislikes"] == "hasNegativePreference"

    def test_education_predicates(self):
        """Test education predicates."""
        assert CANONICAL_PREDICATES["studied_at"] == "alumniOf"
        assert CANONICAL_PREDICATES["graduated_from"] == "alumniOf"


class TestEntityTypes:
    """Test entity types."""

    def test_all_types_present(self):
        """Verify all expected entity types."""
        assert "person" in ENTITY_TYPES
        assert "organization" in ENTITY_TYPES
        assert "location" in ENTITY_TYPES
        assert "concept" in ENTITY_TYPES
        assert "product" in ENTITY_TYPES
        assert "event" in ENTITY_TYPES


class TestUnifiedExtractor:
    """Test UnifiedExtractor class."""

    def test_singleton_pattern(self):
        """UnifiedExtractor should be a singleton."""
        extractor1 = UnifiedExtractor()
        extractor2 = UnifiedExtractor()

        assert extractor1 is extractor2

    def test_get_unified_extractor_helper(self):
        """get_unified_extractor should return the singleton instance."""
        extractor = get_unified_extractor()

        assert isinstance(extractor, UnifiedExtractor)
        assert extractor is UnifiedExtractor()

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        """Empty text should return empty extraction."""
        extractor = UnifiedExtractor()

        result = await extractor.extract("")

        assert result.entities == []
        assert result.facts == []
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_whitespace_text_returns_empty(self):
        """Whitespace-only text should return empty extraction."""
        extractor = UnifiedExtractor()

        result = await extractor.extract("   \n\t  ")

        assert result.entities == []
        assert result.facts == []

    def test_canonicalize_predicate_known(self):
        """Known predicates should be canonicalized."""
        extractor = UnifiedExtractor()

        assert extractor._canonicalize_predicate("works_at") == "worksFor"
        assert extractor._canonicalize_predicate("works_for") == "worksFor"
        assert extractor._canonicalize_predicate("lives_in") == "residesIn"

    def test_canonicalize_predicate_with_spaces(self):
        """Predicates with spaces should be normalized."""
        extractor = UnifiedExtractor()

        assert extractor._canonicalize_predicate("works at") == "worksFor"
        assert extractor._canonicalize_predicate("lives in") == "residesIn"

    def test_canonicalize_predicate_unknown(self):
        """Unknown predicates should be converted to camelCase."""
        extractor = UnifiedExtractor()

        result = extractor._canonicalize_predicate("custom_predicate")
        assert result == "customPredicate"

        result = extractor._canonicalize_predicate("some_long_predicate_name")
        assert result == "someLongPredicateName"

    def test_to_camel_case(self):
        """Test snake_case to camelCase conversion."""
        extractor = UnifiedExtractor()

        assert extractor._to_camel_case("hello_world") == "helloWorld"
        assert extractor._to_camel_case("test") == "test"
        assert extractor._to_camel_case("one_two_three") == "oneTwoThree"

    def test_parse_response_valid(self):
        """Test parsing valid JSON response."""
        extractor = UnifiedExtractor()

        response = '''
        {
            "entities": [
                {"name": "John Smith", "type": "person"},
                {"name": "Acme Corp", "type": "organization"}
            ],
            "facts": [
                {
                    "subject": "John Smith",
                    "predicate": "works_at",
                    "object": "Acme Corp",
                    "source_span": "I work at Acme"
                }
            ],
            "episode_state": {
                "compressed": "John discussed his work at Acme Corp.",
                "participants": ["John Smith", "Acme Corp"],
                "topic_label": "employment"
            },
            "summary": "User discussed their employment."
        }
        '''

        result = extractor._parse_response(response)

        assert len(result.entities) == 2
        assert result.entities[0].name == "John Smith"
        assert result.entities[0].entity_type == "person"

        assert len(result.facts) == 1
        assert result.facts[0].subject == "John Smith"
        assert result.facts[0].canonical_predicate == "worksFor"  # Canonicalized
        assert result.facts[0].source_span == "I work at Acme"

        assert result.episode_state is not None
        assert "Acme" in result.episode_state.compressed
        assert result.episode_state.topic_label == "employment"

        assert result.summary == "User discussed their employment."

    def test_parse_response_validates_entity_type(self):
        """Invalid entity types should default to 'concept'."""
        extractor = UnifiedExtractor()

        response = '''
        {
            "entities": [
                {"name": "Test", "type": "invalid_type"}
            ],
            "facts": [],
            "summary": "Test"
        }
        '''

        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "concept"  # Defaulted

    def test_parse_response_skips_incomplete_facts(self):
        """Facts missing required fields should be skipped."""
        extractor = UnifiedExtractor()

        response = '''
        {
            "entities": [],
            "facts": [
                {"subject": "John"},
                {"subject": "John", "predicate": "knows"},
                {"subject": "John", "predicate": "knows", "object": "Jane"}
            ],
            "summary": "Test"
        }
        '''

        result = extractor._parse_response(response)

        # Only the complete fact should be included
        assert len(result.facts) == 1
        assert result.facts[0].object_value == "Jane"

    def test_parse_response_skips_empty_entity_names(self):
        """Entities with empty names should be skipped."""
        extractor = UnifiedExtractor()

        response = '''
        {
            "entities": [
                {"name": "", "type": "person"},
                {"name": "  ", "type": "person"},
                {"name": "Valid Name", "type": "person"}
            ],
            "facts": [],
            "summary": "Test"
        }
        '''

        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].name == "Valid Name"

    def test_parse_response_invalid_json(self):
        """Invalid JSON should return empty result."""
        extractor = UnifiedExtractor()

        result = extractor._parse_response("not valid json {{{")

        assert result.entities == []
        assert result.facts == []
        assert result.summary == ""

    def test_mock_extraction(self):
        """Test mock extraction when API unavailable."""
        extractor = UnifiedExtractor()

        result = extractor._mock_extraction()

        assert len(result.entities) > 0
        assert len(result.facts) > 0
        assert result.summary != ""
        assert result.episode_state is not None
        assert result.episode_state.topic_label == "test mode"

    def test_get_entity_types_returns_copy(self):
        """get_entity_types should return a copy."""
        types = UnifiedExtractor.get_entity_types()

        assert types == ENTITY_TYPES
        types.append("new_type")
        assert "new_type" not in ENTITY_TYPES

    def test_get_canonical_predicates_returns_copy(self):
        """get_canonical_predicates should return a copy."""
        predicates = UnifiedExtractor.get_canonical_predicates()

        assert predicates == CANONICAL_PREDICATES
        predicates["new_predicate"] = "newPredicate"
        assert "new_predicate" not in CANONICAL_PREDICATES


class TestCanonicalPredicateMapping:
    """Test comprehensive canonical predicate scenarios."""

    def test_all_predicates_are_camel_case(self):
        """All canonical predicates should be camelCase."""
        for raw, canonical in CANONICAL_PREDICATES.items():
            # Canonical should not contain underscores
            assert "_" not in canonical, f"{canonical} should be camelCase"

            # First char should be lowercase
            assert canonical[0].islower(), f"{canonical} should start lowercase"

    def test_predicate_count(self):
        """Verify we have substantial predicate coverage."""
        # Should have at least 30 predicates for good coverage
        assert len(CANONICAL_PREDICATES) >= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
