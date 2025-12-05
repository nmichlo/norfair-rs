//! Python wrapper for TrackedObject.

use nalgebra::DMatrix;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use super::detection::{dmatrix_to_numpy, numpy_to_dmatrix, PyDetection};
use crate::TrackedObject;

/// A tracked object maintained by the tracker.
///
/// Contains the object's state estimate, ID, age, and tracking metadata.
/// Read-only - created and managed by Tracker.
///
/// Compatible with norfair.drawing functions via duck-typing.
#[pyclass(name = "TrackedObject")]
pub struct PyTrackedObject {
    /// Internal data snapshot (we store a copy since TrackedObject is owned by Tracker)
    pub(crate) id: Option<i32>,
    pub(crate) global_id: i32,
    pub(crate) initializing_id: Option<i32>,
    pub(crate) age: i32,
    pub(crate) hit_counter: i32,
    pub(crate) point_hit_counter: Vec<i32>,
    pub(crate) last_detection: Option<PyDetection>,
    pub(crate) last_distance: Option<f64>,
    pub(crate) current_min_distance: Option<f64>,
    pub(crate) label: Option<String>,
    pub(crate) reid_hit_counter: Option<i32>,
    pub(crate) estimate: DMatrix<f64>,
    pub(crate) estimate_velocity: DMatrix<f64>,
    pub(crate) is_initializing: bool,
    pub(crate) detected_at_least_once_points: Vec<bool>,
    pub(crate) past_detections: Vec<PyDetection>,
    /// Optional coordinate transformation for converting between absolute/relative
    pub(crate) coord_transform: Option<Py<PyAny>>,
}

impl PyTrackedObject {
    /// Create a PyTrackedObject from a reference to a Rust TrackedObject.
    pub fn from_tracked_object(obj: &TrackedObject) -> Self {
        Self::from_tracked_object_with_transform(obj, None)
    }

    /// Create a PyTrackedObject with an optional coordinate transformation.
    pub fn from_tracked_object_with_transform(
        obj: &TrackedObject,
        coord_transform: Option<Py<PyAny>>,
    ) -> Self {
        let last_detection = obj
            .last_detection
            .as_ref()
            .map(|d| PyDetection::from_detection(d.clone()));
        let past_detections = obj
            .past_detections
            .iter()
            .map(|d| PyDetection::from_detection(d.clone()))
            .collect();

        Self {
            id: obj.id,
            global_id: obj.global_id,
            initializing_id: obj.initializing_id,
            age: obj.age,
            hit_counter: obj.hit_counter,
            point_hit_counter: obj.point_hit_counter.clone(),
            last_detection,
            last_distance: obj.last_distance,
            current_min_distance: obj.current_min_distance,
            label: obj.label.clone(),
            reid_hit_counter: obj.reid_hit_counter,
            estimate: obj.estimate.clone(),
            estimate_velocity: obj.estimate_velocity.clone(),
            is_initializing: obj.is_initializing,
            detected_at_least_once_points: obj.detected_at_least_once_points.clone(),
            past_detections,
            coord_transform,
        }
    }
}

#[pymethods]
impl PyTrackedObject {
    /// Permanent instance ID (None while initializing).
    #[getter]
    fn id(&self) -> Option<i32> {
        self.id
    }

    /// Global ID unique across all trackers.
    #[getter]
    fn global_id(&self) -> i32 {
        self.global_id
    }

    /// Temporary ID during initialization phase.
    #[getter]
    fn initializing_id(&self) -> Option<i32> {
        self.initializing_id
    }

    /// Frames since first detection.
    #[getter]
    fn age(&self) -> i32 {
        self.age
    }

    /// Remaining frames before object is considered dead.
    #[getter]
    fn hit_counter(&self) -> i32 {
        self.hit_counter
    }

    /// Current state estimate (position) from Kalman filter.
    /// Shape: (n_points, n_dims)
    /// Returns coordinates in relative (camera frame) by default, like get_estimate().
    #[getter(estimate)]
    fn estimate_getter<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Like Python norfair, estimate property returns get_estimate() which defaults to relative
        self.get_estimate(py, false)
    }

    /// Current velocity estimate from Kalman filter.
    /// Shape: (n_points, n_dims)
    #[getter]
    fn estimate_velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.estimate_velocity)
    }

    /// Get the position estimate in absolute or relative coordinates.
    ///
    /// Args:
    ///     absolute: If True, returns coordinates in world frame.
    ///               If False (default), returns coordinates in camera frame.
    ///
    /// Returns:
    ///     Position estimate array of shape (n_points, n_dims).
    ///
    /// Note:
    ///     If no coordinate transformation was provided to the tracker,
    ///     both absolute=True and absolute=False return the same raw coordinates.
    #[pyo3(signature = (absolute=false))]
    fn get_estimate<'py>(
        &self,
        py: Python<'py>,
        absolute: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let estimate_np = dmatrix_to_numpy(py, &self.estimate);

        if let Some(ref transform) = self.coord_transform {
            if absolute {
                // estimate is already in absolute coordinates
                Ok(estimate_np)
            } else {
                // Convert from absolute to relative using abs_to_rel
                let result = transform.call_method1(py, "abs_to_rel", (estimate_np,))?;
                let result_arr = numpy_to_dmatrix(py, result.bind(py))?;
                Ok(dmatrix_to_numpy(py, &result_arr))
            }
        } else {
            // No coordinate transformation - estimate is in "raw" coordinates
            // Both absolute=True and absolute=False return the same value since
            // there's no transform to distinguish them
            Ok(estimate_np)
        }
    }

    /// Most recent matched detection.
    #[getter]
    fn last_detection(&self) -> Option<PyDetection> {
        self.last_detection.clone()
    }

    /// Distance to most recent match.
    #[getter]
    fn last_distance(&self) -> Option<f64> {
        self.last_distance
    }

    /// Minimum distance to any detection in the current frame (for debugging).
    /// This is set by the tracker during update regardless of whether a match occurs.
    #[getter]
    fn current_min_distance(&self) -> Option<f64> {
        self.current_min_distance
    }

    /// Boolean mask indicating which points have been detected at least once.
    /// Shape: (n_points,)
    #[getter]
    fn detected_at_least_once_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        Array1::from_vec(self.detected_at_least_once_points.clone()).into_pyarray(py)
    }

    /// Boolean mask indicating which points are actively tracked.
    /// Shape: (n_points,)
    #[getter]
    fn live_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let live: Vec<bool> = self.point_hit_counter.iter().map(|&c| c > 0).collect();
        Array1::from_vec(live).into_pyarray(py)
    }

    /// Whether the object is still in initialization phase.
    #[getter]
    fn is_initializing(&self) -> bool {
        self.is_initializing
    }

    /// Class label (for multi-class tracking).
    #[getter]
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Re-identification hit counter (separate from main hit counter).
    #[getter]
    fn reid_hit_counter(&self) -> Option<i32> {
        self.reid_hit_counter
    }

    /// History of past detections for re-identification.
    #[getter]
    fn past_detections(&self) -> Vec<PyDetection> {
        self.past_detections.clone()
    }

    /// Per-point hit counters for partial visibility tracking.
    #[getter]
    fn point_hit_counter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        Array1::from_vec(self.point_hit_counter.clone()).into_pyarray(py)
    }

    /// Check if hit_counter >= 0.
    #[getter]
    fn hit_counter_is_positive(&self) -> bool {
        self.hit_counter >= 0
    }

    /// Check if reid_hit_counter is None or >= 0.
    #[getter]
    fn reid_hit_counter_is_positive(&self) -> bool {
        self.reid_hit_counter.map_or(true, |c| c >= 0)
    }

    fn __repr__(&self) -> String {
        let id_str = match self.id {
            Some(id) => format!("id={}", id),
            None => format!("initializing_id={:?}", self.initializing_id),
        };
        format!(
            "TrackedObject({}, age={}, hit_counter={}, is_initializing={})",
            id_str, self.age, self.hit_counter, self.is_initializing
        )
    }

    /// Convert to a native norfair.TrackedObject if norfair is installed.
    ///
    /// Note: This creates a partial representation since norfair.TrackedObject
    /// requires internal state that cannot be fully replicated.
    fn to_norfair(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // norfair.TrackedObject is not directly constructable from Python,
        // so we return a dict with the key attributes instead
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("id", self.id)?;
        dict.set_item("global_id", self.global_id)?;
        dict.set_item("initializing_id", self.initializing_id)?;
        dict.set_item("age", self.age)?;
        dict.set_item("hit_counter", self.hit_counter)?;
        dict.set_item("estimate", dmatrix_to_numpy(py, &self.estimate))?;
        dict.set_item(
            "estimate_velocity",
            dmatrix_to_numpy(py, &self.estimate_velocity),
        )?;
        dict.set_item("live_points", self.live_points(py))?;
        dict.set_item("is_initializing", self.is_initializing)?;
        dict.set_item("label", self.label.clone())?;
        Ok(dict.unbind().into_any())
    }
}
