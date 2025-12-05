# =============================================================================
# COPIED FROM: norfair/tests/test_distances.py
# Original source: https://github.com/tryolabs/norfair
#
# CHANGES FOR norfair_rs:
# - Changed imports from norfair.distances to norfair_rs
# - Overrode mock_det/mock_obj fixtures to create real norfair_rs objects
#   (norfair_rs distance functions require typed objects, not duck-typed)
#
# AVAILABLE FEATURES (fully working):
#   - create_keypoints_voting_distance (Python implementation in __init__.py)
#   - create_normalized_mean_euclidean_distance (Python implementation in __init__.py)
#   - Python callable distance functions (works via PyCallableDistance)
#
# SKIPPED (not applicable to norfair_rs):
#   - ScipyDistance class (norfair_rs uses get_distance_by_name instead)
#   - ScalarDistance/VectorizedDistance with custom Python callables (norfair_rs
#     handles callables differently - pass directly to Tracker instead)
# =============================================================================

import numpy as np
import pytest
from norfair_rs import (
    Detection,
    NoFilterFactory,
    ScalarDistance,
    Tracker,
    VectorizedDistance,
    create_keypoints_voting_distance,
    create_normalized_mean_euclidean_distance,
    frobenius,
    get_distance_by_name,
)

# NOTE: ScipyDistance is not available in norfair_rs
# Use get_distance_by_name("euclidean") etc instead


# =============================================================================
# OVERRIDE FIXTURES: norfair_rs requires real Detection/TrackedObject objects
# =============================================================================


@pytest.fixture
def mock_det():
    """Override conftest mock_det to create real norfair_rs Detection objects."""

    def _make_det(points, scores=None, label=None):
        pts = np.array(points, dtype=np.float64)
        if scores is not None:
            scores = np.array(scores, dtype=np.float64)
            if scores.ndim == 0:
                scores = np.full(pts.shape[0], float(scores))
        return Detection(pts, scores=scores, label=label)

    return _make_det


@pytest.fixture
def mock_obj():
    """Override conftest mock_obj to create real norfair_rs TrackedObject.

    Since TrackedObject can only be created by Tracker, we run a detection
    through a tracker with initialization_delay=0.

    We use NoFilterFactory to ensure the estimate matches the input points
    exactly (no Kalman filtering), which is needed for tests that compare
    distances against exact point values.
    """

    def _make_obj(points, scores=None, label=None):
        pts = np.array(points, dtype=np.float64)
        if scores is not None:
            scores_arr = np.array(scores, dtype=np.float64)
            if scores_arr.ndim == 0:
                scores_arr = np.full(pts.shape[0], float(scores_arr))
        else:
            scores_arr = None
        det = Detection(pts, scores=scores_arr, label=label)
        tracker = Tracker(
            "euclidean",
            distance_threshold=1000,
            initialization_delay=0,
            hit_counter_max=2,
            filter_factory=NoFilterFactory(),  # Use NoFilter to preserve exact points
        )
        tracked = tracker.update([det])
        assert len(tracked) == 1
        return tracked[0]

    return _make_obj


def test_frobenius(mock_obj, mock_det):
    fro = get_distance_by_name("frobenius")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), np.sqrt(1))

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 2)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), np.sqrt(4))

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 2)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 0)


def test_mean_manhattan(mock_det, mock_obj):
    man = get_distance_by_name("mean_manhattan")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 1 / 2)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 1)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 2)

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 0)


def test_mean_euclidean(mock_det, mock_obj):
    euc = get_distance_by_name("mean_euclidean")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 1 / 2)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 1)

    # distance of 2 in 1 dimension of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [5, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 2)

    # distance of 2 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 4], [5, 6]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), np.sqrt(8))

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), np.sqrt(2))

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 0)


def test_iou():
    iou = get_distance_by_name("iou")

    # perfect match
    det = np.array([[0, 0, 1, 1]])
    obj = np.array([[0, 0, 1, 1]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 0)

    # float type
    det = np.array([[0.0, 0.0, 1.1, 1.1]])
    obj = np.array([[0.0, 0.0, 1.1, 1.1]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 0)

    # det contained in obj
    det = np.array([[0, 0, 1, 1]])
    obj = np.array([[0, 0, 2, 2]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1 - 1 / 4)

    # no overlap
    det = np.array([[0, 0, 1, 1]])
    obj = np.array([[1, 1, 2, 2]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1)

    # obj fully contained on det
    det = np.array([[0, 0, 4, 4]])
    obj = np.array([[1, 1, 2, 2]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1 - 1 / 16)

    # partial overlap
    det = np.array([[0, 0, 2, 2]])
    obj = np.array([[1, 1, 3, 3]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1 - 1 / (8 - 1))

    # invalid bbox
    det = np.array([[0, 0]])
    obj = np.array([[0, 0]])
    with pytest.raises(AssertionError):
        iou.distance_function(det, obj)

    # invalid bbox
    det = np.array([[0, 0, 1, 1, 2, 2]])
    obj = np.array([[0, 0, 2, 2]])
    with pytest.raises(AssertionError):
        iou.distance_function(det, obj)


def test_keypoint_vote(mock_obj, mock_det):
    vote_d = create_keypoints_voting_distance(
        keypoint_distance_threshold=np.sqrt(8), detection_threshold=0.5
    )

    # perfect match
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 4)  # 3 matches

    # just under distance threshold
    det = mock_det(points=[[0, 0], [1, 1], [2, 2.0]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [4, 3.9]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 4)  # 3 matches

    # just above distance threshold
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [4, 4]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # just under score threshold on detection
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.5])
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.6])
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # just under score threshold on tracked_object's last detection
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.6])
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.5])
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # no match because of scores
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.5)
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=0.5)
    np.testing.assert_almost_equal(vote_d(det, obj), 1)  # 0 matches

    # no match because of distances
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[2, 2], [3, 3], [4, 4]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1)  # 0 matches


def test_normalized_euclidean(mock_obj, mock_det):
    norm_e = create_normalized_mean_euclidean_distance(10, 10)

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.05)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.1)

    # distance of 2 in 1 dimension of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [5, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.2)

    # distance of 2 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 4], [5, 6]])
    np.testing.assert_almost_equal(norm_e(det, obj), np.sqrt(8) / 10)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(norm_e(det, obj), np.sqrt(2) / 10)

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)


# SKIP: ScalarDistance wrapper for Python callables is a different API in norfair_rs
# In norfair_rs, pass Python callables directly to Tracker(distance_function=...) instead
@pytest.mark.skip(reason="ScalarDistance wrapper API differs - pass callables directly to Tracker")
def test_scalar_distance(mock_obj, mock_det):
    fro = ScalarDistance(frobenius)

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])

    dist_matrix = fro.get_distances([obj], [det])

    assert type(dist_matrix) == np.ndarray
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 0


# SKIP: VectorizedDistance wrapper for Python callables is a different API in norfair_rs
# In norfair_rs, pass Python callables directly to Tracker(distance_function=...) instead
@pytest.mark.skip(
    reason="VectorizedDistance wrapper API differs - pass callables directly to Tracker"
)
def test_vectorized_distance(mock_obj, mock_det):
    def distance_function(cands, objs):
        distance_matrix = np.full(
            (len(cands), len(objs)),
            fill_value=np.inf,
            dtype=np.float32,
        )
        for c, cand in enumerate(cands):
            for o, obj in enumerate(objs):
                distance_matrix[c, o] = np.linalg.norm(cand - obj)
        return distance_matrix

    fro = VectorizedDistance(distance_function)

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])

    dist_matrix = fro.get_distances([obj], [det])

    assert type(dist_matrix) == np.ndarray
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 0


# SKIP: ScipyDistance class doesn't exist in norfair_rs
# Use get_distance_by_name("euclidean") etc. instead for the same functionality
@pytest.mark.skip(reason="ScipyDistance class N/A - use get_distance_by_name() instead")
def test_scipy_distance(mock_obj, mock_det):
    euc = ScipyDistance("euclidean")

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [4, 4]])

    dist_matrix = euc.get_distances([obj], [det])

    assert type(dist_matrix) == np.ndarray
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 1.0


def test_tracker_with_callable_distance():
    """Test that Tracker can use a Python callable as distance function."""
    import warnings

    # Define a simple euclidean distance function
    def my_euclidean_distance(detection, tracked_object):
        det_pts = detection.points
        obj_pts = tracked_object.estimate
        return np.linalg.norm(det_pts - obj_pts)

    # Create tracker with callable - should emit warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracker = Tracker(
            distance_function=my_euclidean_distance,
            distance_threshold=50.0,
            hit_counter_max=5,
            initialization_delay=0,
        )
        assert len(w) == 1
        assert "Python callable" in str(w[0].message)

    # Track a stationary object
    for frame in range(5):
        det = Detection(np.array([[100.0, 100.0]]))
        tracked = tracker.update([det])
        assert len(tracked) == 1, f"Expected 1 object, got {len(tracked)}"

    assert tracker.total_object_count == 1


def test_tracker_callable_distance_high_distance():
    """Test that callable returning high distances creates multiple objects."""
    import warnings

    def always_high_distance(det, obj):
        return 1000.0

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tracker = Tracker(
            distance_function=always_high_distance,
            distance_threshold=50.0,
            hit_counter_max=10,
            initialization_delay=0,
        )

    # Each detection should create a new object since nothing matches
    for _ in range(3):
        det = Detection(np.array([[100.0, 100.0]]))
        tracker.update([det])

    assert (
        tracker.total_object_count >= 2
    ), f"Should create multiple objects, got {tracker.total_object_count}"


def test_tracker_callable_distance_zero_distance():
    """Test that callable returning zero always matches."""
    import warnings

    def always_zero_distance(det, obj):
        return 0.0

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tracker = Tracker(
            distance_function=always_zero_distance,
            distance_threshold=50.0,
            hit_counter_max=10,
            initialization_delay=0,
        )

    # All detections should match the same object
    for frame in range(5):
        det = Detection(np.array([[frame * 100.0, frame * 100.0]]))
        tracked = tracker.update([det])
        assert len(tracked) == 1

    assert tracker.total_object_count == 1, "Should only have 1 object"
