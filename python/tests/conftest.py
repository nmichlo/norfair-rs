# =============================================================================
# COPIED FROM: norfair/tests/conftest.py
# Original source: https://github.com/tryolabs/norfair
# Modified imports to use norfair_rs instead of norfair
# =============================================================================

import numpy as np
import pytest

# NOTE: validate_points is not available in norfair_rs
# Using a simple inline implementation instead
def validate_points(points):
    """Ensure points is a 2D numpy array."""
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


@pytest.fixture
def mock_det():
    class FakeDetection:
        def __init__(self, points, scores=None, label=None) -> None:
            if not isinstance(points, np.ndarray):
                points = np.array(points)
            self.points = points

            if scores is not None and not isinstance(scores, np.ndarray):
                scores = np.array(scores)
                if scores.ndim == 0 and points.shape[0] > 1:
                    scores = np.full(points.shape[0], scores)
            self.scores = scores
            self.label = label

    return FakeDetection


@pytest.fixture
def mock_obj(mock_det):
    class FakeTrackedObject:
        def __init__(self, points, scores=None, label=None):
            if not isinstance(points, np.ndarray):
                points = np.array(points)

            self.estimate = points
            self.last_detection = mock_det(points, scores=scores)
            self.label = label

    return FakeTrackedObject


@pytest.fixture
def mock_coordinate_transformation():

    # simple mock to return abs or relative positions
    class TransformMock:
        def __init__(self, relative_points, absolute_points) -> None:
            self.absolute_points = validate_points(absolute_points)
            self.relative_points = validate_points(relative_points)

        def abs_to_rel(self, points):
            np.testing.assert_equal(points, self.absolute_points)
            return self.relative_points

        def rel_to_abs(self, points):
            np.testing.assert_equal(points, self.relative_points)
            return self.absolute_points

    return TransformMock
