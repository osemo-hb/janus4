"""
Tests for Adaptive Threshold (Fix 4: Cold Start)
"""

import pytest

from janus_core.config import settings
from janus_core.threshold import AdaptiveThreshold


class TestAdaptiveThreshold:
    """Test cases for AdaptiveThreshold class."""

    def test_cold_start_initialization(self):
        """
        Verify threshold uses pre-calculated velocities on cold start (Fix 4).
        """
        threshold = AdaptiveThreshold("test-session")

        # Should be initialized with global averages
        assert len(threshold.velocities) == len(settings.COLD_START_VELOCITIES)
        assert threshold.velocities == list(settings.COLD_START_VELOCITIES)
        assert threshold.last_vector is None

    def test_first_input_no_boundary(self):
        """First input should not trigger boundary (no previous vector)."""
        threshold = AdaptiveThreshold("test-session")

        # First call - no velocity to calculate
        vector = [0.1] * 384
        velocity, is_boundary = threshold.update(vector)

        assert velocity == 0.0
        assert is_boundary is False
        assert threshold.last_vector == vector

    def test_similar_vectors_no_boundary(self):
        """Similar consecutive vectors should not trigger boundary."""
        threshold = AdaptiveThreshold("test-session")

        # First input
        vec1 = [0.1] * 384
        threshold.update(vec1)

        # Very similar second input
        vec2 = [0.11] * 384
        velocity, is_boundary = threshold.update(vec2)

        assert velocity < 0.1  # Very small distance
        assert is_boundary is False

    def test_dissimilar_vectors_trigger_boundary(self):
        """Large semantic jump should trigger boundary."""
        threshold = AdaptiveThreshold("test-session")

        # First input
        vec1 = [0.1] * 384
        threshold.update(vec1)

        # Very different second input
        vec2 = [0.9] * 384
        velocity, is_boundary = threshold.update(vec2)

        assert velocity > 0.3  # Large distance
        assert is_boundary is True

    def test_rolling_window(self):
        """Velocities should be trimmed to window size."""
        threshold = AdaptiveThreshold("test-session")
        threshold.velocities = []  # Clear for this test

        # Add more than window size
        for i in range(100):
            threshold.velocities.append(0.3)

        # Should be trimmed
        assert len(threshold.velocities) <= settings.ADAPTIVE_THRESHOLD_WINDOW

    def test_stats_output(self):
        """get_stats should return meaningful statistics."""
        threshold = AdaptiveThreshold("test-session")

        stats = threshold.get_stats()

        assert "velocity_count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "threshold" in stats
        assert stats["velocity_count"] == len(settings.COLD_START_VELOCITIES)

    def test_reset(self):
        """Reset should restore cold start state."""
        threshold = AdaptiveThreshold("test-session")

        # Modify state
        threshold.velocities = [0.5, 0.6, 0.7]
        threshold.last_vector = [0.1] * 384

        # Reset
        threshold.reset()

        assert threshold.velocities == list(settings.COLD_START_VELOCITIES)
        assert threshold.last_vector is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
