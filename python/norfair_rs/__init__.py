"""
norfair_rs - Fast object tracking library.

A Rust implementation of norfair for high-performance multi-object tracking.
Compatible with norfair v2.3.0 API.

API COMPATIBILITY NOTES:
========================

This module aims to be a drop-in replacement for norfair's core tracking API.
The following are FULLY COMPATIBLE with norfair:

  - Detection: Same constructor signature and attributes
    (points, scores, data, label, embedding, age)
  - TrackedObject: Same attributes
    (id, estimate, live_points, detected_at_least_once_points, etc.)
  - Tracker: Same constructor signature and update() method
  - Filter factories: OptimizedKalmanFilterFactory, FilterPyKalmanFilterFactory,
    NoFilterFactory
  - Distance functions: frobenius, mean_euclidean, mean_manhattan, iou, iou_opt,
    get_distance_by_name
  - Distance factory functions: create_keypoints_voting_distance,
    create_normalized_mean_euclidean_distance

The following are NOT AVAILABLE in norfair_rs (requires OpenCV or complex dependencies):

  - Video: Requires OpenCV for video I/O
  - Drawing functions: draw_boxes, draw_points, draw_tracked_objects, etc.
    * Workaround: norfair_rs objects work with norfair.drawing via duck-typing
  - FixedCamera, camera_motion module: Motion estimation requires OpenCV
  - HomographyTransformation: Requires OpenCV
  - get_cutout, print_objects_as_table: Utility functions

The following have MINOR DIFFERENCES:

  - TranslationTransformation: Available (simple 2D translation)
  - ScalarDistance, VectorizedDistance: Wrappers for custom distance functions
  - Distance: Type alias for get_distance_by_name() return value
  - Python callable distance functions are supported

Example:
    >>> from norfair_rs import Tracker, Detection
    >>> import numpy as np
    >>>
    >>> tracker = Tracker(
    ...     distance_function="euclidean",
    ...     distance_threshold=50.0,
    ... )
    >>>
    >>> # Process detections frame by frame
    >>> points = np.array([[100, 100], [200, 200]], dtype=np.float64)
    >>> detections = [Detection(points)]
    >>> tracked_objects = tracker.update(detections)

Using with norfair.drawing:
    norfair.drawing uses isinstance() checks, so norfair_rs objects must be
    converted first using the to_norfair() method:

    >>> from norfair.drawing import draw_points
    >>> from norfair_rs import Tracker, Detection
    >>> # ... create tracker and get tracked_objects ...
    >>> # Convert norfair_rs objects to norfair objects for drawing
    >>> norfair_objects = [obj.to_norfair() for obj in tracked_objects]
    >>> draw_points(frame, norfair_objects)  # Works!

    Note: to_norfair() returns a norfair.Detection or dict (for TrackedObject).
"""

from collections.abc import Callable

import numpy as np

# Test utilities (internal use)
from norfair_rs._norfair_rs import (
    # Core classes - FULLY COMPATIBLE with norfair
    Detection,
    FilterPyKalmanFilterFactory,
    NoFilterFactory,
    # Filter factories - FULLY COMPATIBLE with norfair
    OptimizedKalmanFilterFactory,
    # Distance classes - norfair_rs specific wrappers
    ScalarDistance,
    TrackedObject,
    Tracker,
    # Transformations - PARTIALLY COMPATIBLE
    # Only TranslationTransformation is available.
    # HomographyTransformation requires OpenCV and is NOT AVAILABLE.
    TranslationTransformation,
    VectorizedDistance,
    __norfair_compat_version__,
    # Version info
    __version__,
    _reset_global_id_counter,  # noqa F401
    # helper? internal?
    frobenius,
    # Distance functions - FULLY COMPATIBLE with norfair
    get_distance_by_name,
    iou,
    mean_euclidean,
    mean_manhattan,
)

# Distance is the type returned by get_distance_by_name()
# In norfair, this is an internal type. We expose it for type checking.
Distance = type(get_distance_by_name("euclidean"))

# iou_opt is a deprecated alias for iou (for backwards compatibility)
iou_opt = iou

# List of available vectorized distance functions
AVAILABLE_VECTORIZED_DISTANCES = [
    "iou",
    "iou_opt",
    "euclidean",
    "cosine",
    "cityblock",
    "sqeuclidean",
    "chebyshev",
    "braycurtis",
    "canberra",
    "correlation",
]


def create_keypoints_voting_distance(
    keypoint_distance_threshold: float,
    detection_threshold: float,
) -> Callable[[Detection, TrackedObject], float]:
    """
    Construct a keypoint voting distance function configured with the thresholds.

    Count how many points in a detection match the with a tracked_object.
    A match is considered when distance between the points is < `keypoint_distance_threshold`
    and the score of the last_detection of the tracked_object is > `detection_threshold`.
    Notice the if multiple points are tracked, the ith point in detection can only match the ith
    point in the tracked object.

    Distance is 1 if no point matches and approximates 0 as more points are matched.

    Args:
        keypoint_distance_threshold: Points closer than this threshold are considered a match.
        detection_threshold: Detections/objects with score lower than this are ignored.

    Returns:
        A distance function that takes (detection, tracked_object) and returns
        a float distance value.
    """

    def keypoints_voting_distance(detection: Detection, tracked_object: TrackedObject) -> float:
        distances = np.linalg.norm(detection.points - tracked_object.estimate, axis=1)
        # Get scores from last_detection if available, otherwise use detection scores
        last_det_scores = getattr(tracked_object, "last_detection", None)
        if last_det_scores is not None:
            last_det_scores = last_det_scores.scores
        if last_det_scores is None:
            # Fall back to using detection scores for both sides
            last_det_scores = detection.scores

        match_num = np.count_nonzero(
            (distances < keypoint_distance_threshold)
            * (detection.scores > detection_threshold)
            * (last_det_scores > detection_threshold)
        )
        return 1 / (1 + match_num)

    return keypoints_voting_distance


def create_normalized_mean_euclidean_distance(
    height: int,
    width: int,
) -> Callable[[Detection, TrackedObject], float]:
    """
    Construct a normalized mean euclidean distance function.

    The result distance is bound to [0, 1] where 1 indicates opposite corners.

    Args:
        height: Height of the image.
        width: Width of the image.

    Returns:
        A distance function that takes (detection, tracked_object) and returns
        a float distance value.
    """

    def normalized_mean_euclidean_distance(
        detection: Detection, tracked_object: TrackedObject
    ) -> float:
        """Normalized mean euclidean distance"""
        # calculate distances and normalize by width and height
        difference = (detection.points - tracked_object.estimate).astype(float)
        difference[:, 0] /= width
        difference[:, 1] /= height

        # calculate euclidean distance and average
        return np.linalg.norm(difference, axis=1).mean()

    return normalized_mean_euclidean_distance


__all__ = [
    # Core classes
    "Detection",
    "TrackedObject",
    "Tracker",
    # Filter factories
    "OptimizedKalmanFilterFactory",
    "FilterPyKalmanFilterFactory",
    "NoFilterFactory",
    # Distance classes
    "Distance",
    "ScalarDistance",
    "VectorizedDistance",
    # Distance functions
    "get_distance_by_name",
    "frobenius",
    "mean_euclidean",
    "mean_manhattan",
    "iou",
    "iou_opt",  # Deprecated alias for iou
    # Distance factory functions
    "create_keypoints_voting_distance",
    "create_normalized_mean_euclidean_distance",
    # Constants
    "AVAILABLE_VECTORIZED_DISTANCES",
    # Transformations
    "TranslationTransformation",
    # Version info
    "__version__",
    "__norfair_compat_version__",
]
