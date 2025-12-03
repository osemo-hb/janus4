"""
Adaptive Threshold for Topic Boundary Detection

Implements Fix 4: Cold start initialization with pre-calculated velocities.
Uses running statistics (mean + std) instead of magic numbers.
"""

from typing import List, Tuple, Optional
import numpy as np

from config import settings


class AdaptiveThreshold:
    """
    Adaptive semantic boundary detection.

    Detects topic shifts by monitoring "semantic velocity" - the cosine
    distance between consecutive user inputs. A high velocity indicates
    a potential topic change.

    Fix 4: Initialized with pre-calculated global average velocities
    to avoid cold start issues where the first few turns have unstable
    threshold calculations.
    """

    def __init__(self, session_id: str):
        """
        Initialize threshold detector for a session.

        Args:
            session_id: Unique identifier for the session.
        """
        self.session_id = session_id

        # Fix 4: Pre-populate with global average velocities
        # This prevents the cold start problem where we have no data
        # to calculate meaningful statistics
        self.velocities: List[float] = list(settings.COLD_START_VELOCITIES)

        # Track the last user input vector
        self.last_vector: Optional[List[float]] = None

        # Window size for rolling statistics
        self.window_size = settings.ADAPTIVE_THRESHOLD_WINDOW

        # Multiplier for standard deviation (how many std above mean = boundary)
        self.std_multiplier = settings.ADAPTIVE_THRESHOLD_MULTIPLIER

    def update(self, current_vector: List[float]) -> Tuple[float, bool]:
        """
        Calculate semantic velocity and determine if a boundary is crossed.

        Args:
            current_vector: Embedding of the current user input.

        Returns:
            Tuple of (velocity, is_boundary):
            - velocity: Cosine distance from last vector (0.0 if first input)
            - is_boundary: True if topic shift detected
        """
        if self.last_vector is None:
            # First input in session - no velocity to calculate
            self.last_vector = current_vector
            return 0.0, False

        # Calculate cosine distance (velocity)
        velocity = self._cosine_distance(current_vector, self.last_vector)

        # Update last vector
        self.last_vector = current_vector

        # Add to velocity history
        self.velocities.append(velocity)

        # Trim to window size
        if len(self.velocities) > self.window_size:
            self.velocities = self.velocities[-self.window_size:]

        # Calculate adaptive threshold
        is_boundary = self._is_boundary(velocity)

        return velocity, is_boundary

    def _is_boundary(self, velocity: float) -> bool:
        """
        Determine if velocity indicates a topic boundary.

        Uses adaptive threshold: mean + (std_multiplier * std)
        Clamped to reasonable range [0.25, 0.65] to prevent
        extreme thresholds from unstable data.
        """
        if len(self.velocities) < 3:
            # Not enough data - use fallback threshold
            return velocity > 0.45

        # Calculate rolling statistics
        recent = self.velocities[-self.window_size:]
        mean_v = np.mean(recent)
        std_v = np.std(recent)

        # Adaptive threshold with clamping
        threshold = mean_v + (self.std_multiplier * std_v)
        threshold = max(0.25, min(0.65, threshold))

        return velocity > threshold

    def _cosine_distance(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine distance between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0

        similarity = dot_product / (norm_a * norm_b)
        return 1.0 - similarity

    def get_stats(self) -> dict:
        """Get current threshold statistics for debugging."""
        if len(self.velocities) < 3:
            return {
                "velocity_count": len(self.velocities),
                "mean": None,
                "std": None,
                "threshold": 0.45,
            }

        recent = self.velocities[-self.window_size:]
        mean_v = float(np.mean(recent))
        std_v = float(np.std(recent))
        threshold = mean_v + (self.std_multiplier * std_v)
        threshold = max(0.25, min(0.65, threshold))

        return {
            "velocity_count": len(self.velocities),
            "mean": mean_v,
            "std": std_v,
            "threshold": threshold,
        }

    def reset(self):
        """Reset to cold start state (useful for testing)."""
        self.velocities = list(settings.COLD_START_VELOCITIES)
        self.last_vector = None
