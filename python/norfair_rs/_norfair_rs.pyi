"""
Type stubs for norfair_rs native module.

These type hints provide IDE support and static type checking for the
norfair_rs Python bindings.

Compatible with norfair v2.3.0 API.
"""

from typing import Any, Callable, Hashable, List, Optional, Sequence, Union
import numpy as np
import numpy.typing as npt

# Version info
__version__: str
__norfair_compat_version__: str

# Type aliases
NDArrayFloat = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayInt = npt.NDArray[np.int32]

class Detection:
    """
    A detection to be tracked.

    Represents a detected object in a frame, with its position points
    and optional metadata like scores, labels, and embeddings.

    Compatible with norfair.drawing functions via duck-typing.

    Attributes:
        points: Detection points as a numpy array of shape (n_points, n_dims).
        scores: Optional per-point confidence scores of shape (n_points,).
        label: Optional class label for multi-class tracking.
        embedding: Optional embedding vector for re-identification.
        absolute_points: Points in absolute coordinates (world frame).
    """

    points: NDArrayFloat
    scores: Optional[NDArrayFloat]
    label: Optional[str]
    embedding: Optional[NDArrayFloat]
    absolute_points: NDArrayFloat

    def __init__(
        self,
        points: NDArrayFloat,
        scores: Optional[NDArrayFloat] = None,
        data: Any = None,
        label: Optional[str] = None,
        embedding: Optional[NDArrayFloat] = None,
    ) -> None:
        """
        Create a new Detection.

        Args:
            points: Detection points as a numpy array of shape (n_points, n_dims).
                    For keypoints: [[x1, y1], [x2, y2], ...]
                    For bounding boxes: [[x1, y1], [x2, y2]] (top-left, bottom-right)
            scores: Optional per-point confidence scores of shape (n_points,).
            data: Optional arbitrary user data (not currently supported).
            label: Optional class label for multi-class tracking.
            embedding: Optional embedding vector for re-identification.
        """
        ...

    def num_points(self) -> int:
        """Number of points in this detection."""
        ...

    def num_dims(self) -> int:
        """Dimensionality of points (typically 2 for 2D tracking)."""
        ...

    def to_norfair(self) -> Any:
        """
        Convert to a native norfair.Detection if norfair is installed.

        Returns:
            A norfair.Detection object with the same data.

        Raises:
            ImportError: If norfair is not installed.
        """
        ...


class TrackedObject:
    """
    A tracked object maintained by the tracker.

    Contains the object's state estimate, ID, age, and tracking metadata.
    Read-only - created and managed by Tracker.

    Compatible with norfair.drawing functions via duck-typing.

    Attributes:
        id: Permanent instance ID (None while initializing).
        global_id: Global ID unique across all trackers.
        initializing_id: Temporary ID during initialization phase.
        age: Frames since first detection.
        hit_counter: Remaining frames before object is considered dead.
        estimate: Current state estimate (position) from Kalman filter.
        estimate_velocity: Current velocity estimate from Kalman filter.
        last_detection: Most recent matched detection.
        last_distance: Distance to most recent match.
        live_points: Boolean mask indicating which points are actively tracked.
        is_initializing: Whether the object is still in initialization phase.
        label: Class label (for multi-class tracking).
        reid_hit_counter: Re-identification hit counter.
        past_detections: History of past detections.
        point_hit_counter: Per-point hit counters.
    """

    id: Optional[int]
    global_id: int
    initializing_id: Optional[int]
    age: int
    hit_counter: int
    estimate: NDArrayFloat
    estimate_velocity: NDArrayFloat
    last_detection: Optional[Detection]
    last_distance: Optional[float]
    live_points: NDArrayBool
    is_initializing: bool
    label: Optional[str]
    reid_hit_counter: Optional[int]
    past_detections: List[Detection]
    point_hit_counter: NDArrayInt
    hit_counter_is_positive: bool
    reid_hit_counter_is_positive: bool

    def get_estimate(self, absolute: bool = False) -> NDArrayFloat:
        """
        Get the current position estimate.

        Args:
            absolute: If True, return in absolute (world) coordinates.

        Returns:
            Position estimate matrix (n_points, n_dims).
        """
        ...

    def to_norfair(self) -> dict:
        """
        Convert to a dict with key attributes for norfair compatibility.

        Returns:
            A dict with id, estimate, live_points, etc.
        """
        ...


class Tracker:
    """
    Object tracker.

    Maintains a set of tracked objects across frames, matching new detections
    to existing objects and managing object lifecycles.

    Example:
        >>> from norfair_rs import Tracker, Detection
        >>> import numpy as np
        >>>
        >>> tracker = Tracker(
        ...     distance_function="euclidean",
        ...     distance_threshold=50.0,
        ... )
        >>>
        >>> detections = [Detection(np.array([[100, 100]]))]
        >>> tracked_objects = tracker.update(detections)
    """

    current_object_count: int
    total_object_count: int
    tracked_objects: List[TrackedObject]

    def __init__(
        self,
        distance_function: Union[str, Callable[[Detection, TrackedObject], float]],
        distance_threshold: float,
        hit_counter_max: int = 15,
        initialization_delay: Optional[int] = None,
        pointwise_hit_counter_max: int = 4,
        detection_threshold: float = 0.0,
        filter_factory: Optional[
            Union[
                "OptimizedKalmanFilterFactory",
                "FilterPyKalmanFilterFactory",
                "NoFilterFactory",
            ]
        ] = None,
        past_detections_length: int = 4,
        reid_distance_function: Optional[
            Callable[["TrackedObject", "TrackedObject"], float]
        ] = None,
        reid_distance_threshold: float = 0.0,
        reid_hit_counter_max: Optional[int] = None,
    ) -> None:
        """
        Create a new Tracker.

        Args:
            distance_function: Distance function for matching detections to objects.
                Can be a string name (e.g., "euclidean", "iou") or a callable
                that takes (Detection, TrackedObject) and returns a float.
            distance_threshold: Maximum distance for valid matches.
            hit_counter_max: Maximum hit counter value. Default: 15.
            initialization_delay: Frames before permanent ID. Default: hit_counter_max // 2.
            pointwise_hit_counter_max: Maximum hit counter for points. Default: 4.
            detection_threshold: Minimum score for detection points. Default: 0.0.
            filter_factory: Factory for creating Kalman filters.
            past_detections_length: Number of past detections to store. Default: 4.
            reid_distance_function: Optional distance function for re-identification.
            reid_distance_threshold: Distance threshold for re-identification.
            reid_hit_counter_max: Maximum hit counter for re-identification.
        """
        ...

    def update(
        self,
        detections: Optional[List[Detection]] = None,
        period: int = 1,
        coord_transformations: Optional["TranslationTransformation"] = None,
    ) -> List[TrackedObject]:
        """
        Update the tracker with new detections.

        Args:
            detections: List of Detection objects for this frame.
            period: Frame period for hit counter adjustment. Default: 1.
            coord_transformations: Optional coordinate transformation.

        Returns:
            List of active (non-initializing) TrackedObject instances.
        """
        ...

    def get_active_objects(self) -> List[TrackedObject]:
        """
        Get all currently active (non-initializing) objects.

        Returns:
            List of TrackedObject instances that have been initialized.
        """
        ...


# Filter Factories

class OptimizedKalmanFilterFactory:
    """
    Optimized Kalman filter factory with simplified covariance tracking.

    This is the default filter factory used by Tracker.
    """

    def __init__(
        self,
        R: float = 4.0,
        Q: float = 0.1,
        pos_variance: float = 10.0,
        pos_vel_covariance: float = 0.0,
        vel_variance: float = 1.0,
    ) -> None:
        """
        Create a new OptimizedKalmanFilterFactory.

        Args:
            R: Measurement noise variance. Default: 4.0.
            Q: Process noise variance. Default: 0.1.
            pos_variance: Initial position variance. Default: 10.0.
            pos_vel_covariance: Initial position-velocity covariance. Default: 0.0.
            vel_variance: Initial velocity variance. Default: 1.0.
        """
        ...


class FilterPyKalmanFilterFactory:
    """
    FilterPy-compatible Kalman filter factory.

    Maintains full covariance matrices for higher accuracy.
    """

    def __init__(
        self,
        R: float = 4.0,
        Q: float = 0.1,
        P: float = 10.0,
    ) -> None:
        """
        Create a new FilterPyKalmanFilterFactory.

        Args:
            R: Measurement noise variance. Default: 4.0.
            Q: Process noise variance. Default: 0.1.
            P: Initial state covariance. Default: 10.0.
        """
        ...


class NoFilterFactory:
    """
    No-filter factory (baseline without prediction).
    """

    def __init__(self) -> None:
        """Create a new NoFilterFactory."""
        ...


# Distance Classes

class Distance:
    """
    A built-in distance function.

    Created by get_distance_by_name().
    """

    ...


class ScalarDistance:
    """
    Wrapper for scalar distance functions.

    Scalar distance functions compute the distance between a single
    Detection and TrackedObject pair.
    """

    def __init__(
        self,
        distance_function: Callable[[Detection, TrackedObject], float],
    ) -> None:
        """
        Create a new ScalarDistance.

        Args:
            distance_function: A callable that takes (Detection, TrackedObject)
                               and returns a float distance.
        """
        ...


class VectorizedDistance:
    """
    Wrapper for vectorized distance functions.

    Vectorized distance functions compute distances between batches
    of detections and tracked objects efficiently.
    """

    def __init__(
        self,
        distance_function: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
    ) -> None:
        """
        Create a new VectorizedDistance.

        Args:
            distance_function: A callable that takes (candidates_array, objects_array)
                               and returns a distance matrix.
        """
        ...


# Distance Functions

def get_distance_by_name(name: str) -> Distance:
    """
    Get a built-in distance function by name.

    Supported names:
        - "euclidean", "manhattan", "cosine", "chebyshev" - scipy metrics
        - "frobenius" - Frobenius norm of difference
        - "mean_euclidean" - Mean L2 distance per point
        - "mean_manhattan" - Mean L1 distance per point
        - "iou" - Intersection over Union for bounding boxes

    Args:
        name: Name of the distance function.

    Returns:
        A Distance object.

    Raises:
        ValueError: If the distance name is not recognized.
    """
    ...


def frobenius(detection: Detection, tracked_object: TrackedObject) -> float:
    """
    Frobenius norm distance between detection and tracked object.

    Args:
        detection: The detection.
        tracked_object: The tracked object.

    Returns:
        Frobenius norm of the difference.
    """
    ...


def mean_euclidean(detection: Detection, tracked_object: TrackedObject) -> float:
    """
    Mean Euclidean distance between detection and tracked object.

    Args:
        detection: The detection.
        tracked_object: The tracked object.

    Returns:
        Mean L2 distance across all corresponding points.
    """
    ...


def mean_manhattan(detection: Detection, tracked_object: TrackedObject) -> float:
    """
    Mean Manhattan distance between detection and tracked object.

    Args:
        detection: The detection.
        tracked_object: The tracked object.

    Returns:
        Mean L1 distance across all corresponding points.
    """
    ...


def iou(candidates: NDArrayFloat, objects: NDArrayFloat) -> NDArrayFloat:
    """
    Intersection over Union (IoU) distance for bounding boxes.

    Args:
        candidates: Array of candidate bounding boxes, shape (n_candidates, 4).
                    Each row is [x1, y1, x2, y2].
        objects: Array of object bounding boxes, shape (n_objects, 4).

    Returns:
        Distance matrix of shape (n_candidates, n_objects).
        Distance is 1 - IoU, so 0 means perfect overlap.
    """
    ...


# Transformations

class TranslationTransformation:
    """
    Simple 2D translation transformation (camera pan/tilt).

    This transformation adds/subtracts a movement vector to convert between
    relative (camera frame) and absolute (world frame) coordinates.
    """

    movement_vector: tuple[float, float]

    def __init__(self, movement_vector: NDArrayFloat) -> None:
        """
        Create a new TranslationTransformation.

        Args:
            movement_vector: A 2-element array [dx, dy] representing camera movement.
        """
        ...

    def abs_to_rel(self, points: NDArrayFloat) -> NDArrayFloat:
        """
        Transform from absolute to relative coordinates.

        Args:
            points: Array of points, shape (n_points, 2).

        Returns:
            Transformed points in relative coordinates.
        """
        ...

    def rel_to_abs(self, points: NDArrayFloat) -> NDArrayFloat:
        """
        Transform from relative to absolute coordinates.

        Args:
            points: Array of points, shape (n_points, 2).

        Returns:
            Transformed points in absolute coordinates.
        """
        ...
