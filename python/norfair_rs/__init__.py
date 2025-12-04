"""
norfair_rs - Fast object tracking library.

A Rust implementation of norfair for high-performance multi-object tracking.
Compatible with norfair v2.3.0 API.

API COMPATIBILITY NOTES:
========================

This module aims to be a drop-in replacement for norfair's core tracking API.
The following are FULLY COMPATIBLE with norfair:

  - Detection: Same constructor signature and attributes
  - TrackedObject: Same attributes (id, estimate, live_points, etc.)
  - Tracker: Same constructor signature and update() method
  - Filter factories: OptimizedKalmanFilterFactory, FilterPyKalmanFilterFactory, NoFilterFactory
  - Distance functions: frobenius, mean_euclidean, mean_manhattan, iou, get_distance_by_name

The following are NOT AVAILABLE in norfair_rs (requires OpenCV or complex dependencies):

  - Video: Requires OpenCV for video I/O
  - Drawing functions: draw_boxes, draw_points, draw_tracked_objects, etc.
    * Workaround: norfair_rs objects work with norfair.drawing via duck-typing
  - FixedCamera, camera_motion module: Motion estimation requires OpenCV
  - HomographyTransformation: Requires OpenCV
  - iou_opt: Optimized IoU variant (use iou instead)
  - create_keypoints_voting_distance: Custom distance factory
  - create_normalized_mean_euclidean_distance: Custom distance factory
  - get_cutout, print_objects_as_table: Utility functions

The following have MINOR DIFFERENCES:

  - TranslationTransformation: Available (simple 2D translation)
  - ScalarDistance, VectorizedDistance: Wrappers for custom distance functions
    * NOTE: Python callable distance functions are NOT YET SUPPORTED.
      Please use string names like "euclidean", "iou", etc.
  - Distance: Type alias for get_distance_by_name() return value

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

from norfair_rs._norfair_rs import (
    # Core classes - FULLY COMPATIBLE with norfair
    Detection,
    TrackedObject,
    Tracker,
    # Filter factories - FULLY COMPATIBLE with norfair
    OptimizedKalmanFilterFactory,
    FilterPyKalmanFilterFactory,
    NoFilterFactory,
    # Distance classes - norfair_rs specific wrappers
    # NOTE: ScalarDistance and VectorizedDistance wrap Python callables,
    # but Python callable distances are NOT YET SUPPORTED in Tracker.
    # Use string names like "euclidean", "iou", etc. instead.
    ScalarDistance,
    VectorizedDistance,
    # Distance functions - FULLY COMPATIBLE with norfair
    get_distance_by_name,
    frobenius,
    mean_euclidean,
    mean_manhattan,
    iou,
    # Transformations - PARTIALLY COMPATIBLE
    # Only TranslationTransformation is available.
    # HomographyTransformation requires OpenCV and is NOT AVAILABLE.
    TranslationTransformation,
    # Version info
    __version__,
    __norfair_compat_version__,
)

# Distance is the type returned by get_distance_by_name()
# In norfair, this is an internal type. We expose it for type checking.
Distance = type(get_distance_by_name("euclidean"))

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
