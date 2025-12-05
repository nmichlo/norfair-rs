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
    detection_threshold: float = 0.0,
) -> Callable[[Detection, TrackedObject], float]:
    """
    Create a keypoint voting distance function.

    This distance function counts the number of keypoints that are within a
    threshold distance from their corresponding predicted positions, and
    returns 1 - (matched_keypoints / total_keypoints).

    Args:
        keypoint_distance_threshold: Maximum distance for a keypoint to be
            considered a match.
        detection_threshold: Minimum score for a keypoint to be considered
            valid. Default is 0.0.

    Returns:
        A distance function that takes (detection, tracked_object) and returns
        a float distance value in [0, 1].
    """

    def voting_distance(detection: Detection, tracked_object: TrackedObject) -> float:
        det_points = detection.points
        obj_estimate = tracked_object.estimate
        det_scores = detection.scores

        # Count valid and matching keypoints
        n_points = len(det_points)
        n_valid = 0
        n_matching = 0

        for i in range(n_points):
            # Check if this keypoint is valid (score > threshold)
            if det_scores is not None and det_scores[i] <= detection_threshold:
                continue

            n_valid += 1

            # Compute distance between detection and estimate
            dist = np.linalg.norm(det_points[i] - obj_estimate[i])
            if dist <= keypoint_distance_threshold:
                n_matching += 1

        if n_valid == 0:
            return 1.0  # No valid keypoints, maximum distance

        return 1.0 - (n_matching / n_valid)

    return voting_distance


def create_normalized_mean_euclidean_distance(
    height: int,
    width: int,
) -> Callable[[Detection, TrackedObject], float]:
    """
    Create a normalized mean euclidean distance function.

    This distance function computes the mean euclidean distance between
    detection points and tracked object estimate, normalized by the frame
    dimensions (diagonal length).

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        A distance function that takes (detection, tracked_object) and returns
        a float distance value normalized to approximately [0, 1].
    """
    diagonal = np.sqrt(height**2 + width**2)

    def normalized_distance(detection: Detection, tracked_object: TrackedObject) -> float:
        det_points = detection.points
        obj_estimate = tracked_object.estimate

        # Compute mean euclidean distance
        diff = det_points - obj_estimate
        distances = np.linalg.norm(diff, axis=1)
        mean_dist = np.mean(distances)

        # Normalize by diagonal
        return mean_dist / diagonal

    return normalized_distance


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
