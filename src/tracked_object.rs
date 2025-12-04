//! TrackedObject struct for tracked objects maintained by the tracker.

use std::collections::VecDeque;
use std::fmt;
use nalgebra::DMatrix;
use crate::{Detection, Filter};
use crate::camera_motion::CoordinateTransformation;

/// A tracked object maintained by the tracker.
///
/// Contains the object's state estimate, ID, age, and tracking metadata.
pub struct TrackedObject {
    /// Permanent instance ID (None while initializing).
    pub id: Option<i32>,

    /// Global ID unique across all trackers.
    pub global_id: i32,

    /// Temporary ID during initialization phase.
    pub initializing_id: Option<i32>,

    /// Frames since first detection.
    pub age: i32,

    /// Remaining frames before object is considered dead.
    pub hit_counter: i32,

    /// Per-point hit counters for partial visibility tracking.
    pub point_hit_counter: Vec<i32>,

    /// Most recent matched detection.
    pub last_detection: Option<Detection>,

    /// Distance to most recent match.
    pub last_distance: Option<f64>,

    /// History of past detections for re-identification.
    pub past_detections: VecDeque<Detection>,

    /// Class label (for multi-class tracking).
    pub label: Option<String>,

    /// Re-identification hit counter (separate from main hit counter).
    pub reid_hit_counter: Option<i32>,

    /// Current state estimate (position, from filter).
    pub estimate: DMatrix<f64>,

    /// Current velocity estimate (from filter).
    pub estimate_velocity: DMatrix<f64>,

    /// Whether the object is still in initialization phase.
    pub is_initializing: bool,

    /// The Kalman filter maintaining this object's state.
    pub(crate) filter: Box<dyn Filter>,

    /// Number of points being tracked.
    pub(crate) num_points: usize,

    /// Dimensionality of each point.
    pub(crate) dim_points: usize,

    /// Last coordinate transformation (for absolute/relative conversion).
    pub(crate) last_coord_transform: Option<Box<dyn CoordinateTransformation>>,
}

impl fmt::Debug for TrackedObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrackedObject")
            .field("id", &self.id)
            .field("global_id", &self.global_id)
            .field("initializing_id", &self.initializing_id)
            .field("age", &self.age)
            .field("hit_counter", &self.hit_counter)
            .field("point_hit_counter", &self.point_hit_counter)
            .field("last_detection", &self.last_detection)
            .field("last_distance", &self.last_distance)
            .field("past_detections", &self.past_detections)
            .field("label", &self.label)
            .field("reid_hit_counter", &self.reid_hit_counter)
            .field("estimate", &self.estimate)
            .field("estimate_velocity", &self.estimate_velocity)
            .field("is_initializing", &self.is_initializing)
            .field("filter", &"<Filter>")
            .field("num_points", &self.num_points)
            .field("dim_points", &self.dim_points)
            .field("last_coord_transform", &self.last_coord_transform.as_ref().map(|_| "<CoordinateTransformation>"))
            .finish()
    }
}

impl TrackedObject {
    /// Get the current position estimate.
    ///
    /// # Arguments
    /// * `absolute` - If true, return in absolute (world) coordinates
    ///
    /// # Returns
    /// Position estimate matrix (n_points x n_dims)
    pub fn get_estimate(&self, absolute: bool) -> DMatrix<f64> {
        if absolute {
            if let Some(ref transform) = self.last_coord_transform {
                return transform.rel_to_abs(&self.estimate);
            }
        }
        self.estimate.clone()
    }

    /// Get the current velocity estimate.
    pub fn get_estimate_velocity(&self) -> &DMatrix<f64> {
        &self.estimate_velocity
    }

    /// Check which points are currently "live" (actively tracked).
    pub fn live_points(&self) -> Vec<bool> {
        self.point_hit_counter
            .iter()
            .map(|&c| c > 0)
            .collect()
    }
}

impl Default for TrackedObject {
    fn default() -> Self {
        Self {
            id: None,
            global_id: 0,
            initializing_id: None,
            age: 0,
            hit_counter: 0,
            point_hit_counter: Vec::new(),
            last_detection: None,
            last_distance: None,
            past_detections: VecDeque::new(),
            label: None,
            reid_hit_counter: None,
            estimate: DMatrix::zeros(1, 2),
            estimate_velocity: DMatrix::zeros(1, 2),
            is_initializing: true,
            filter: Box::new(crate::filter::NoFilter::new(&DMatrix::zeros(1, 2))),
            num_points: 1,
            dim_points: 2,
            last_coord_transform: None,
        }
    }
}
