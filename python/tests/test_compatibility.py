"""
Dual-library compatibility tests.

This module tests that norfair_rs behaves identically to the original norfair library.
Tests are parameterized to run against both libraries simultaneously.
"""

import numpy as np
import pytest


@pytest.fixture(params=["norfair", "norfair_rs"])
def nf(request):
    """Fixture that returns either norfair or norfair_rs module."""
    if request.param == "norfair":
        try:
            import norfair
            return norfair
        except ImportError:
            pytest.skip("norfair not installed")
    else:
        import norfair_rs
        return norfair_rs


class TestDetectionAttributes:
    """Test Detection class attributes."""

    def test_detection_data_attribute(self, nf):
        """Test that Detection.data stores arbitrary user data."""
        data = {"custom": "data", "id": 123}
        det = nf.Detection(points=np.array([[1.0, 2.0]]), data=data)
        assert det.data == data

    def test_detection_data_none(self, nf):
        """Test that Detection.data defaults to None."""
        det = nf.Detection(points=np.array([[1.0, 2.0]]))
        assert det.data is None

    def test_detection_points(self, nf):
        """Test that Detection.points returns the correct shape."""
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        det = nf.Detection(points=points)
        np.testing.assert_array_almost_equal(det.points, points)

    def test_detection_scores(self, nf):
        """Test that Detection.scores works correctly."""
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        scores = np.array([0.9, 0.8])
        det = nf.Detection(points=points, scores=scores)
        np.testing.assert_array_almost_equal(det.scores, scores)

    def test_detection_label(self, nf):
        """Test that Detection.label works correctly."""
        det = nf.Detection(points=np.array([[1.0, 2.0]]), label="person")
        assert det.label == "person"

    def test_detection_1d_points(self, nf):
        """Test that 1D points are automatically reshaped to 2D."""
        points_1d = np.array([1.0, 2.0])
        det = nf.Detection(points=points_1d)
        assert det.points.shape == (1, 2)


class TestTrackedObjectAttributes:
    """Test TrackedObject class attributes."""

    def test_tracked_object_live_points(self, nf):
        """Test that live_points is a boolean array."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=0,
        )
        det = nf.Detection(points=np.array([[1.0, 1.0], [2.0, 2.0]]))
        objs = tracker.update([det])
        assert len(objs) == 1
        live_points = objs[0].live_points
        assert len(live_points) == 2
        assert all(isinstance(x, (bool, np.bool_)) for x in live_points)

    def test_tracked_object_detected_at_least_once_points(self, nf):
        """Test detected_at_least_once_points attribute."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=0,
        )
        det = nf.Detection(points=np.array([[1.0, 1.0], [2.0, 2.0]]))
        objs = tracker.update([det])
        assert len(objs) == 1
        mask = objs[0].detected_at_least_once_points
        assert len(mask) == 2
        assert all(mask)  # All points detected on first frame

    def test_tracked_object_estimate_shape(self, nf):
        """Test that estimate has the correct shape."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=0,
        )
        det = nf.Detection(points=np.array([[1.0, 1.0], [2.0, 2.0]]))
        objs = tracker.update([det])
        assert len(objs) == 1
        estimate = objs[0].estimate
        assert estimate.shape == (2, 2)

    def test_tracked_object_id(self, nf):
        """Test that initialized objects have an ID."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=0,
        )
        det = nf.Detection(points=np.array([[1.0, 1.0]]))
        objs = tracker.update([det])
        assert len(objs) == 1
        assert objs[0].id is not None

    def test_tracked_object_initializing_id(self, nf):
        """Test that objects have an initializing_id."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=5,
        )
        det = nf.Detection(points=np.array([[1.0, 1.0]]))
        tracker.update([det])  # Object is still initializing
        # Check internal tracked_objects
        assert len(tracker.tracked_objects) == 1
        assert tracker.tracked_objects[0].initializing_id is not None


class TestDistanceFunctions:
    """Test distance functions."""

    def test_iou_available(self, nf):
        """Test that iou is available."""
        assert hasattr(nf, 'iou')

    def test_iou_opt_available(self, nf):
        """Test that iou_opt is available."""
        assert hasattr(nf, 'iou_opt')

    def test_frobenius_available(self, nf):
        """Test that frobenius is available."""
        assert hasattr(nf, 'frobenius')

    def test_mean_euclidean_available(self, nf):
        """Test that mean_euclidean is available."""
        assert hasattr(nf, 'mean_euclidean')

    def test_mean_manhattan_available(self, nf):
        """Test that mean_manhattan is available."""
        assert hasattr(nf, 'mean_manhattan')

    def test_get_distance_by_name(self, nf):
        """Test get_distance_by_name function."""
        assert hasattr(nf, 'get_distance_by_name')
        dist = nf.get_distance_by_name("euclidean")
        assert dist is not None

    def test_available_vectorized_distances(self, nf):
        """Test AVAILABLE_VECTORIZED_DISTANCES constant (norfair_rs extension)."""
        # Note: norfair doesn't export this constant, it's a norfair_rs extension
        if nf.__name__ == "norfair":
            pytest.skip("AVAILABLE_VECTORIZED_DISTANCES is a norfair_rs extension")
        assert hasattr(nf, 'AVAILABLE_VECTORIZED_DISTANCES')
        assert 'iou' in nf.AVAILABLE_VECTORIZED_DISTANCES
        assert 'euclidean' in nf.AVAILABLE_VECTORIZED_DISTANCES


class TestDistanceFactoryFunctions:
    """Test distance factory functions."""

    def test_create_keypoints_voting_distance(self, nf):
        """Test create_keypoints_voting_distance factory function."""
        assert hasattr(nf, 'create_keypoints_voting_distance')
        dist_fn = nf.create_keypoints_voting_distance(
            keypoint_distance_threshold=10.0,
            detection_threshold=0.5
        )
        assert callable(dist_fn)

    def test_create_normalized_mean_euclidean_distance(self, nf):
        """Test create_normalized_mean_euclidean_distance factory function."""
        assert hasattr(nf, 'create_normalized_mean_euclidean_distance')
        dist_fn = nf.create_normalized_mean_euclidean_distance(height=100, width=200)
        assert callable(dist_fn)


class TestFilterFactories:
    """Test filter factory classes."""

    def test_optimized_kalman_filter_factory(self, nf):
        """Test OptimizedKalmanFilterFactory."""
        assert hasattr(nf, 'OptimizedKalmanFilterFactory')
        factory = nf.OptimizedKalmanFilterFactory()
        assert factory is not None

    def test_filterpy_kalman_filter_factory(self, nf):
        """Test FilterPyKalmanFilterFactory."""
        assert hasattr(nf, 'FilterPyKalmanFilterFactory')
        factory = nf.FilterPyKalmanFilterFactory()
        assert factory is not None

    def test_no_filter_factory(self, nf):
        """Test NoFilterFactory."""
        assert hasattr(nf, 'NoFilterFactory')
        factory = nf.NoFilterFactory()
        assert factory is not None


class TestTrackerBasics:
    """Test basic Tracker functionality."""

    def test_tracker_creation(self, nf):
        """Test basic tracker creation."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=50.0,
        )
        assert tracker is not None

    def test_tracker_with_string_distance(self, nf):
        """Test tracker with string distance function."""
        tracker = nf.Tracker(
            distance_function="iou",
            distance_threshold=0.5,
        )
        # Create bbox detections for IoU
        det = nf.Detection(points=np.array([[0.0, 0.0], [10.0, 10.0]]))
        tracker.update([det])

    def test_tracker_update_empty(self, nf):
        """Test update with empty detections."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=50.0,
        )
        objs = tracker.update([])
        assert len(objs) == 0

    def test_tracker_object_counts(self, nf):
        """Test tracker object count methods."""
        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=0,
        )
        assert tracker.total_object_count == 0
        assert tracker.current_object_count == 0

        det = nf.Detection(points=np.array([[1.0, 1.0]]))
        tracker.update([det])

        assert tracker.total_object_count == 1
        assert tracker.current_object_count == 1


class TestModuleExports:
    """Test that all expected items are exported."""

    def test_core_exports(self, nf):
        """Test that core items are exported."""
        # These are exported by both norfair and norfair_rs
        expected = [
            'Detection', 'Tracker',
            'OptimizedKalmanFilterFactory', 'FilterPyKalmanFilterFactory', 'NoFilterFactory',
            'frobenius', 'mean_euclidean', 'mean_manhattan', 'iou', 'iou_opt',
            'get_distance_by_name',
            'create_keypoints_voting_distance', 'create_normalized_mean_euclidean_distance',
        ]
        for name in expected:
            assert hasattr(nf, name), f"Missing export: {name}"

    def test_norfair_rs_extensions(self, nf):
        """Test norfair_rs-specific extensions."""
        if nf.__name__ == "norfair":
            pytest.skip("Testing norfair_rs extensions")
        # norfair_rs exports TrackedObject directly (norfair doesn't)
        assert hasattr(nf, 'TrackedObject')
        assert hasattr(nf, 'AVAILABLE_VECTORIZED_DISTANCES')


class TestTrackerWithFilters:
    """Test Tracker with different filter types."""

    @pytest.mark.parametrize("filter_factory_name", [
        "OptimizedKalmanFilterFactory",
        "FilterPyKalmanFilterFactory",
        "NoFilterFactory",
    ])
    def test_tracker_with_filter(self, nf, filter_factory_name):
        """Test tracker with different filter factories."""
        factory_cls = getattr(nf, filter_factory_name)
        factory = factory_cls()

        tracker = nf.Tracker(
            distance_function="euclidean",
            distance_threshold=100.0,
            initialization_delay=0,
            filter_factory=factory,
        )

        det = nf.Detection(points=np.array([[1.0, 1.0]]))
        objs = tracker.update([det])
        assert len(objs) == 1
