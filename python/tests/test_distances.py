# =============================================================================
# COPIED FROM: norfair/tests/test_distances.py
# Original source: https://github.com/tryolabs/norfair
#
# CHANGES FOR norfair_rs:
# - Changed imports from norfair.distances to norfair_rs
# - Overrode mock_det/mock_obj fixtures to create real norfair_rs objects
#   (norfair_rs distance functions require typed objects, not duck-typed)
# - Skipped tests for unavailable features:
#   - create_keypoints_voting_distance (not available)
#   - create_normalized_mean_euclidean_distance (not available)
#   - ScipyDistance (not available - use get_distance_by_name instead)
#   - Python callable ScalarDistance/VectorizedDistance (not yet supported)
# =============================================================================

import numpy as np
import pytest
from norfair_rs import (
    Detection,
    ScalarDistance,
    Tracker,
    VectorizedDistance,
    frobenius,
    get_distance_by_name,
)

# NOTE: These are not available in norfair_rs:
# - ScipyDistance (use get_distance_by_name("euclidean") etc instead)
# - create_keypoints_voting_distance
# - create_normalized_mean_euclidean_distance


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


# SKIP: create_keypoints_voting_distance not available in norfair_rs
@pytest.mark.skip(reason="create_keypoints_voting_distance not available in norfair_rs")
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


# SKIP: create_normalized_mean_euclidean_distance not available in norfair_rs
@pytest.mark.skip(reason="create_normalized_mean_euclidean_distance not available in norfair_rs")
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


# SKIP: Python callable distance functions not yet supported in norfair_rs
@pytest.mark.skip(reason="Python callable distance functions not yet supported in norfair_rs")
def test_scalar_distance(mock_obj, mock_det):
    fro = ScalarDistance(frobenius)

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])

    dist_matrix = fro.get_distances([obj], [det])

    assert type(dist_matrix) == np.ndarray
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 0


# SKIP: Python callable distance functions not yet supported in norfair_rs
@pytest.mark.skip(reason="Python callable distance functions not yet supported in norfair_rs")
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


# SKIP: ScipyDistance not available in norfair_rs (use get_distance_by_name instead)
@pytest.mark.skip(
    reason="ScipyDistance not available in norfair_rs - use get_distance_by_name instead"
)
def test_scipy_distance(mock_obj, mock_det):
    euc = ScipyDistance("euclidean")

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [4, 4]])

    dist_matrix = euc.get_distances([obj], [det])

    assert type(dist_matrix) == np.ndarray
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 1.0
