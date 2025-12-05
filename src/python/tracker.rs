//! Python wrapper for Tracker.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distances::{try_distance_function_by_name, DistanceFunction};
use crate::{Detection, Tracker, TrackerConfig};

use super::detection::PyDetection;
use super::distances::{extract_distance, PyBuiltinDistance, PyDistanceEnum};
use super::filters::extract_filter_factory;
use super::tracked_object::PyTrackedObject;
use super::transforms::extract_transform;

/// Object tracker.
///
/// Maintains a set of tracked objects across frames, matching new detections
/// to existing objects and managing object lifecycles.
///
/// Example:
///     >>> from norfair_rs import Tracker, Detection
///     >>> import numpy as np
///     >>>
///     >>> tracker = Tracker(
///     ...     distance_function="euclidean",
///     ...     distance_threshold=50.0,
///     ... )
///     >>>
///     >>> # Process detections frame by frame
///     >>> detections = [Detection(np.array([[100, 100]]))]
///     >>> tracked_objects = tracker.update(detections)
#[pyclass(name = "Tracker")]
pub struct PyTracker {
    inner: Tracker,
    /// Store the Python distance function if using a callable
    py_distance: Option<PyDistanceEnum>,
}

#[pymethods]
impl PyTracker {
    /// Create a new Tracker.
    ///
    /// Args:
    ///     distance_function: Distance function for matching detections to objects.
    ///         Can be a string name (e.g., "euclidean", "iou") or a callable
    ///         that takes (Detection, TrackedObject) and returns a float.
    ///     distance_threshold: Maximum distance for valid matches.
    ///     hit_counter_max: Maximum hit counter value (frames to keep object alive
    ///         without detections). Default: 15.
    ///     initialization_delay: Frames before an object gets a permanent ID.
    ///         Default: hit_counter_max // 2.
    ///     pointwise_hit_counter_max: Maximum hit counter for individual points.
    ///         Default: 4.
    ///     detection_threshold: Minimum score for a detection point to be considered.
    ///         Default: 0.0.
    ///     filter_factory: Factory for creating Kalman filters.
    ///         Default: OptimizedKalmanFilterFactory().
    ///     past_detections_length: Number of past detections to store.
    ///         Default: 4.
    ///     reid_distance_function: Optional distance function for re-identification.
    ///     reid_distance_threshold: Distance threshold for re-identification.
    ///         Default: 0.0.
    ///     reid_hit_counter_max: Maximum hit counter for re-identification phase.
    #[new]
    #[pyo3(signature = (
        distance_function,
        distance_threshold,
        hit_counter_max=15,
        initialization_delay=None,
        pointwise_hit_counter_max=4,
        detection_threshold=0.0,
        filter_factory=None,
        past_detections_length=4,
        reid_distance_function=None,
        reid_distance_threshold=0.0,
        reid_hit_counter_max=None
    ))]
    fn new(
        py: Python<'_>,
        distance_function: &Bound<'_, PyAny>,
        distance_threshold: f64,
        hit_counter_max: i32,
        initialization_delay: Option<i32>,
        pointwise_hit_counter_max: i32,
        detection_threshold: f64,
        filter_factory: Option<&Bound<'_, PyAny>>,
        past_detections_length: usize,
        reid_distance_function: Option<&Bound<'_, PyAny>>,
        reid_distance_threshold: f64,
        reid_hit_counter_max: Option<i32>,
    ) -> PyResult<Self> {
        // Extract distance function - convert to DistanceFunction enum
        let py_distance = extract_distance(py, distance_function)?;

        // Get the DistanceFunction enum for built-in distances
        // For Python callables, we currently don't support them (would need custom tracker path)
        let rust_distance: DistanceFunction =
            if let Ok(name) = distance_function.extract::<String>() {
                // String name - use distance_function_by_name
                try_distance_function_by_name(&name).map_err(|e| PyValueError::new_err(e))?
            } else if let Ok(d) = distance_function.extract::<PyRef<PyBuiltinDistance>>() {
                // PyBuiltinDistance - use its name
                try_distance_function_by_name(&d.name).map_err(|e| PyValueError::new_err(e))?
            } else {
                // Python callable - not yet supported with enum-based tracker
                return Err(PyValueError::new_err(
                    "Python callable distance functions are not yet supported. \
                 Please use a string name like 'euclidean', 'iou', 'frobenius', etc.",
                ));
            };

        // Validate parameters before creating config
        if let Some(delay) = initialization_delay {
            if delay < 0 {
                return Err(PyValueError::new_err(
                    "initialization_delay must be non-negative",
                ));
            }
        }

        if hit_counter_max <= 0 {
            return Err(PyValueError::new_err("hit_counter_max must be positive"));
        }

        // Extract filter factory and convert to FilterFactoryEnum
        let filter_enum = extract_filter_factory(filter_factory)?;

        // Build config with enum-based types
        let mut config = TrackerConfig::new(rust_distance, distance_threshold);
        config.hit_counter_max = hit_counter_max;
        // Use -1 as sentinel for "use default" (hit_counter_max / 2)
        config.initialization_delay = initialization_delay.unwrap_or(-1);
        config.pointwise_hit_counter_max = pointwise_hit_counter_max;
        config.detection_threshold = detection_threshold;
        config.filter_factory = filter_enum.to_filter_factory_enum();
        config.past_detections_length = past_detections_length;
        config.reid_distance_threshold = reid_distance_threshold;
        config.reid_hit_counter_max = reid_hit_counter_max;

        // TODO: Handle reid_distance_function if provided

        // Create tracker
        let tracker = Tracker::new(config).map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Store Python distance if it's a callable (for future use)
        let py_distance_opt = match &py_distance {
            PyDistanceEnum::Builtin(_) => None,
            _ => Some(py_distance),
        };

        Ok(Self {
            inner: tracker,
            py_distance: py_distance_opt,
        })
    }

    /// Update the tracker with new detections.
    ///
    /// Args:
    ///     detections: List of Detection objects for this frame. Can be None or empty.
    ///     period: Frame period for hit counter adjustment. Default: 1.
    ///     coord_transformations: Optional coordinate transformation for camera motion.
    ///
    /// Returns:
    ///     List of active (non-initializing) TrackedObject instances.
    #[pyo3(signature = (detections=None, period=1, coord_transformations=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        detections: Option<Vec<PyDetection>>,
        period: i32,
        coord_transformations: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyTrackedObject>> {
        // Keep reference to original PyDetections so we can update them in-place
        let py_detections = detections.unwrap_or_default();

        // Convert detections
        let rust_detections: Vec<Detection> =
            py_detections.iter().map(|d| d.get_detection()).collect();

        // Extract coordinate transformation
        let transform = extract_transform(coord_transformations)?;
        let transform_ref: Option<&dyn crate::camera_motion::CoordinateTransformation> =
            transform.as_ref().map(|t| t.as_transform());

        // If we have a coordinate transformation, update the original PyDetection objects
        // with their absolute_points (like Python norfair does)
        if let Some(ref t) = transform {
            for py_det in &py_detections {
                let det = py_det.inner.read().unwrap();
                let abs_points = t.as_transform().rel_to_abs(&det.points);
                drop(det); // Release read lock
                py_det
                    .inner
                    .write()
                    .unwrap()
                    .set_absolute_points(abs_points);
            }
        }

        // If we have a Python callable distance, we need to handle it specially
        // For now, we use the built-in distance in the Rust tracker
        // TODO: Implement proper Python callable support by wrapping in Distance trait

        // Update tracker
        let tracked = self.inner.update(rust_detections, period, transform_ref);

        // Get Python coordinate transformation reference if provided
        let py_coord_transform: Option<Py<PyAny>> = coord_transformations
            .filter(|obj| !obj.is_none())
            .map(|obj| obj.clone().unbind());

        // Convert to Python objects, passing the coordinate transformation
        let py_tracked: Vec<PyTrackedObject> = tracked
            .into_iter()
            .map(|obj| {
                PyTrackedObject::from_tracked_object_with_transform(
                    obj,
                    py_coord_transform.as_ref().map(|t| t.clone_ref(py)),
                )
            })
            .collect();

        Ok(py_tracked)
    }

    /// Get all currently active (non-initializing) objects.
    ///
    /// Returns:
    ///     List of TrackedObject instances that have been initialized.
    fn get_active_objects(&self) -> Vec<PyTrackedObject> {
        self.inner
            .tracked_objects
            .iter()
            .filter(|obj| !obj.is_initializing)
            .map(|obj| PyTrackedObject::from_tracked_object(obj))
            .collect()
    }

    /// Number of currently active tracked objects.
    #[getter]
    fn current_object_count(&self) -> usize {
        // Count objects that are initialized AND have positive hit_counter
        // (matches Rust tracker's active_objects() logic)
        self.inner
            .tracked_objects
            .iter()
            .filter(|obj| !obj.is_initializing && obj.hit_counter >= 0)
            .count()
    }

    /// Total number of objects that have been assigned permanent IDs.
    #[getter]
    fn total_object_count(&self) -> i32 {
        self.inner.total_object_count()
    }

    /// All currently tracked objects (including initializing).
    #[getter]
    fn tracked_objects(&self) -> Vec<PyTrackedObject> {
        self.inner
            .tracked_objects
            .iter()
            .map(|obj| PyTrackedObject::from_tracked_object(obj))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tracker(tracked_objects={}, total_initialized={})",
            self.inner.tracked_objects.len(),
            self.inner.total_object_count()
        )
    }
}
