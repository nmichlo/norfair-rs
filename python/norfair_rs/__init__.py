"""
norfair_rs - Fast object tracking library.

A Rust implementation of norfair for high-performance multi-object tracking.
Compatible with norfair v2.3.0 API.

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
"""

from norfair_rs._norfair_rs import (
    # Core classes
    Detection,
    TrackedObject,
    Tracker,
    # Filter factories
    OptimizedKalmanFilterFactory,
    FilterPyKalmanFilterFactory,
    NoFilterFactory,
    # Distance classes
    Distance,
    ScalarDistance,
    VectorizedDistance,
    # Distance functions
    get_distance_by_name,
    frobenius,
    mean_euclidean,
    mean_manhattan,
    iou,
    # Transformations
    TranslationTransformation,
    # Version info
    __version__,
    __norfair_compat_version__,
)

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
    # Transformations
    "TranslationTransformation",
    # Version info
    "__version__",
    "__norfair_compat_version__",
]
