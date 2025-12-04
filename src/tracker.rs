//! Main tracker implementation.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, Ordering};
use nalgebra::{DMatrix, DVector};

use crate::{Detection, TrackedObject, Distance, FilterFactory, Error, Result};
use crate::filter::OptimizedKalmanFilterFactory;
use crate::distances::distance_by_name;
use crate::matching::{match_detections_and_objects, get_unmatched};
use crate::camera_motion::CoordinateTransformation;

// Global ID counter for unique IDs across all trackers
static GLOBAL_ID_COUNTER: AtomicI32 = AtomicI32::new(0);

/// Configuration for the tracker.
#[derive(Clone)]
pub struct TrackerConfig {
    /// Distance function for matching detections to objects.
    pub distance_function: Box<dyn Distance>,

    /// Maximum distance threshold for valid matches.
    pub distance_threshold: f64,

    /// Maximum hit counter value (frames to keep object alive without detections).
    pub hit_counter_max: i32,

    /// Frames before an object becomes "initialized" (gets permanent ID).
    pub initialization_delay: i32,

    /// Maximum hit counter for individual points.
    pub pointwise_hit_counter_max: i32,

    /// Minimum score for a detection point to be considered.
    pub detection_threshold: f64,

    /// Factory for creating Kalman filters.
    pub filter_factory: Box<dyn FilterFactory>,

    /// Number of past detections to store for re-identification.
    pub past_detections_length: usize,

    /// Optional distance function for re-identification.
    pub reid_distance_function: Option<Box<dyn Distance>>,

    /// Distance threshold for re-identification.
    pub reid_distance_threshold: f64,

    /// Maximum hit counter for re-identification phase.
    pub reid_hit_counter_max: Option<i32>,
}

impl TrackerConfig {
    /// Create a new tracker configuration.
    ///
    /// # Arguments
    /// * `distance_function` - Distance function for matching
    /// * `distance_threshold` - Maximum match distance
    pub fn new(distance_function: Box<dyn Distance>, distance_threshold: f64) -> Self {
        Self {
            distance_function,
            distance_threshold,
            hit_counter_max: 15,
            initialization_delay: -1, // Will be set to hit_counter_max / 2
            pointwise_hit_counter_max: 4,
            detection_threshold: 0.0,
            filter_factory: Box::new(OptimizedKalmanFilterFactory::default()),
            past_detections_length: 4,
            reid_distance_function: None,
            reid_distance_threshold: 1.0,
            reid_hit_counter_max: None,
        }
    }

    /// Create configuration from a distance function name.
    pub fn from_distance_name(name: &str, distance_threshold: f64) -> Self {
        Self::new(distance_by_name(name), distance_threshold)
    }
}

impl Clone for Box<dyn Distance> {
    fn clone(&self) -> Self {
        // Distance functions are typically stateless, so we can safely create new ones
        // This is a simplified approach - in practice you might want a Clone trait bound
        distance_by_name("euclidean") // Default fallback
    }
}

impl Clone for Box<dyn FilterFactory> {
    fn clone(&self) -> Self {
        Box::new(OptimizedKalmanFilterFactory::default())
    }
}

/// Object tracker.
///
/// Maintains a set of tracked objects across frames, matching new detections
/// to existing objects and managing object lifecycles.
pub struct Tracker {
    /// Tracker configuration.
    pub config: TrackerConfig,

    /// Currently tracked objects.
    pub tracked_objects: Vec<TrackedObject>,

    /// Local instance ID counter.
    instance_id_counter: i32,

    /// Local initializing ID counter.
    initializing_id_counter: i32,
}

impl Tracker {
    /// Create a new tracker with the given configuration.
    pub fn new(mut config: TrackerConfig) -> Result<Self> {
        // Validate and set defaults
        if config.initialization_delay == -1 {
            config.initialization_delay = config.hit_counter_max / 2;
        }

        if config.initialization_delay < 0 {
            return Err(Error::InvalidConfig(
                "initialization_delay must be non-negative".to_string(),
            ));
        }

        if config.initialization_delay >= config.hit_counter_max {
            return Err(Error::InvalidConfig(
                "initialization_delay must be less than hit_counter_max".to_string(),
            ));
        }

        Ok(Self {
            config,
            tracked_objects: Vec::new(),
            instance_id_counter: 0,
            initializing_id_counter: 0,
        })
    }

    /// Update the tracker with new detections.
    ///
    /// # Arguments
    /// * `detections` - New detections for this frame
    /// * `period` - Frame period (for hit counter increment)
    /// * `coord_transform` - Optional coordinate transformation for camera motion
    ///
    /// # Returns
    /// Slice of active (non-initializing) tracked objects
    pub fn update(
        &mut self,
        mut detections: Vec<Detection>,
        period: i32,
        coord_transform: Option<&dyn CoordinateTransformation>,
    ) -> Vec<&TrackedObject> {
        // Apply coordinate transformation to detections
        if let Some(transform) = coord_transform {
            for det in &mut detections {
                let abs_points = transform.rel_to_abs(&det.points);
                det.set_absolute_points(abs_points);
            }
        }

        // Age all tracked objects (predict step) - inline to avoid borrow issues
        for obj in &mut self.tracked_objects {
            obj.age += 1;
            // Only decrement hit_counter for non-initializing objects
            // Initializing objects need to accumulate hits without decay
            if !obj.is_initializing {
                obj.hit_counter -= 1;
            }

            // Decrement point hit counters
            for counter in &mut obj.point_hit_counter {
                *counter = (*counter - 1).max(0);
            }

            // Kalman predict
            obj.filter.predict();

            // Update estimate from filter
            obj.estimate = obj.filter.get_state();

            // Update velocity estimate
            let state = obj.filter.get_state_vector();
            let dim_z = obj.filter.dim_z();
            if state.len() >= dim_z * 2 {
                let velocity_flat: Vec<f64> = state.iter().skip(dim_z).take(dim_z).cloned().collect();
                obj.estimate_velocity = DMatrix::from_vec(obj.num_points, obj.dim_points, velocity_flat);
            }

            // Store coordinate transform for later use
            if let Some(transform) = coord_transform {
                obj.last_coord_transform = Some(transform.clone_box());
            }
        }

        // Remove dead objects (hit_counter < 0)
        self.tracked_objects.retain(|obj| obj.hit_counter >= 0);

        // Separate initializing and initialized objects
        let (initialized_indices, initializing_indices): (Vec<_>, Vec<_>) = self
            .tracked_objects
            .iter()
            .enumerate()
            .partition(|(_, obj)| !obj.is_initializing);

        let initialized_indices: Vec<_> = initialized_indices.into_iter().map(|(i, _)| i).collect();
        let initializing_indices: Vec<_> = initializing_indices.into_iter().map(|(i, _)| i).collect();

        // Match initialized objects first
        let det_refs: Vec<&Detection> = detections.iter().collect();
        let init_obj_refs: Vec<&TrackedObject> = initialized_indices
            .iter()
            .map(|&i| &self.tracked_objects[i])
            .collect();

        let distance_matrix = if !init_obj_refs.is_empty() && !det_refs.is_empty() {
            self.config.distance_function.get_distances(&init_obj_refs, &det_refs)
        } else {
            DMatrix::zeros(det_refs.len(), init_obj_refs.len())
        };

        let (matched_dets, matched_objs) =
            match_detections_and_objects(&distance_matrix, self.config.distance_threshold);

        // Update matched initialized objects
        for (&det_idx, &obj_local_idx) in matched_dets.iter().zip(matched_objs.iter()) {
            let obj_idx = initialized_indices[obj_local_idx];
            self.hit_object(obj_idx, &detections[det_idx], period, distance_matrix[(det_idx, obj_local_idx)]);
        }

        // Get unmatched detections
        let unmatched_det_indices = get_unmatched(detections.len(), &matched_dets);

        // Match initializing objects with unmatched detections
        let unmatched_det_refs: Vec<&Detection> = unmatched_det_indices
            .iter()
            .map(|&i| &detections[i])
            .collect();
        let init_obj_refs: Vec<&TrackedObject> = initializing_indices
            .iter()
            .map(|&i| &self.tracked_objects[i])
            .collect();

        let init_distance_matrix = if !init_obj_refs.is_empty() && !unmatched_det_refs.is_empty() {
            self.config.distance_function.get_distances(&init_obj_refs, &unmatched_det_refs)
        } else {
            DMatrix::zeros(unmatched_det_refs.len(), init_obj_refs.len())
        };

        let (init_matched_dets, init_matched_objs) =
            match_detections_and_objects(&init_distance_matrix, self.config.distance_threshold);

        // Update matched initializing objects
        for (&local_det_idx, &obj_local_idx) in init_matched_dets.iter().zip(init_matched_objs.iter()) {
            let det_idx = unmatched_det_indices[local_det_idx];
            let obj_idx = initializing_indices[obj_local_idx];
            self.hit_object(obj_idx, &detections[det_idx], period, init_distance_matrix[(local_det_idx, obj_local_idx)]);
        }

        // Create new objects for remaining unmatched detections
        let still_unmatched: Vec<_> = get_unmatched(unmatched_det_indices.len(), &init_matched_dets)
            .into_iter()
            .map(|i| unmatched_det_indices[i])
            .collect();

        for det_idx in still_unmatched {
            self.create_object(&detections[det_idx], period, coord_transform);
        }

        // Return active (non-initializing, positive hit_counter) objects
        self.tracked_objects
            .iter()
            .filter(|obj| !obj.is_initializing && obj.hit_counter > 0)
            .collect()
    }

    /// Get the total number of objects that have been assigned permanent IDs.
    pub fn total_object_count(&self) -> i32 {
        self.instance_id_counter
    }

    /// Get the current number of active (non-initializing) objects.
    pub fn current_object_count(&self) -> usize {
        self.tracked_objects
            .iter()
            .filter(|obj| !obj.is_initializing && obj.hit_counter > 0)
            .count()
    }

    // Internal: update object with matched detection
    fn hit_object(&mut self, obj_idx: usize, detection: &Detection, period: i32, distance: f64) {
        // First, build observation matrix while we only need immutable access
        let h = {
            let obj = &self.tracked_objects[obj_idx];
            self.build_observation_matrix_impl(obj, detection)
        };

        // Now get mutable access for updates
        let obj = &mut self.tracked_objects[obj_idx];

        // Update hit counter
        obj.hit_counter = (obj.hit_counter + period).min(self.config.hit_counter_max);

        // Check for initialization transition
        if obj.is_initializing && obj.hit_counter >= self.config.initialization_delay {
            obj.is_initializing = false;
            obj.id = Some(self.instance_id_counter);
            self.instance_id_counter += 1;
            obj.initializing_id = None;

            // Reset reid_hit_counter if configured
            if self.config.reid_hit_counter_max.is_some() {
                obj.reid_hit_counter = None;
            }
        }

        // Update point hit counters
        for (i, counter) in obj.point_hit_counter.iter_mut().enumerate() {
            let score = detection.scores.as_ref().map(|s| s[i]).unwrap_or(1.0);
            if score > self.config.detection_threshold {
                *counter = (*counter + period).min(self.config.pointwise_hit_counter_max);
            }
        }

        // Kalman update
        let measurement: Vec<f64> = detection.get_absolute_points().iter().cloned().collect();
        let measurement = DVector::from_vec(measurement);
        obj.filter.update(&measurement, None, h.as_ref());

        // Update estimate
        obj.estimate = obj.filter.get_state();

        // Store detection
        obj.last_detection = Some(detection.clone());
        obj.last_distance = Some(distance);

        // Update past detections
        if self.config.past_detections_length > 0 {
            obj.past_detections.push_back(detection.clone());
            while obj.past_detections.len() > self.config.past_detections_length {
                obj.past_detections.pop_front();
            }
        }
    }

    // Internal: create new tracked object
    fn create_object(
        &mut self,
        detection: &Detection,
        period: i32,
        coord_transform: Option<&dyn CoordinateTransformation>,
    ) {
        let global_id = GLOBAL_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        let initializing_id = self.initializing_id_counter;
        self.initializing_id_counter += 1;

        let num_points = detection.num_points();
        let dim_points = detection.num_dims();

        // Create filter
        let filter = self.config.filter_factory.create_filter(detection.get_absolute_points());

        // Initialize point hit counters
        let point_hit_counter = vec![period.min(self.config.pointwise_hit_counter_max); num_points];

        let mut obj = TrackedObject {
            id: None,
            global_id,
            initializing_id: Some(initializing_id),
            age: 0,
            hit_counter: period,
            point_hit_counter,
            last_detection: Some(detection.clone()),
            last_distance: None,
            past_detections: VecDeque::new(),
            label: detection.label.clone(),
            reid_hit_counter: self.config.reid_hit_counter_max,
            estimate: filter.get_state(),
            estimate_velocity: DMatrix::zeros(num_points, dim_points),
            is_initializing: true,
            filter,
            num_points,
            dim_points,
            last_coord_transform: coord_transform.map(|t| t.clone_box()),
        };

        // Check for immediate initialization (delay = 0)
        if self.config.initialization_delay == 0 {
            obj.is_initializing = false;
            obj.id = Some(self.instance_id_counter);
            self.instance_id_counter += 1;
            obj.initializing_id = None;
        }

        self.tracked_objects.push(obj);
    }

    // Internal: build observation matrix for partial observations
    fn build_observation_matrix_impl(&self, obj: &TrackedObject, detection: &Detection) -> Option<DMatrix<f64>> {
        let dim_z = obj.filter.dim_z();
        let dim_x = obj.filter.dim_x();

        // Check if any points should be masked
        let scores = detection.scores.as_ref();
        let needs_mask = scores.map(|s| s.iter().any(|&score| score <= self.config.detection_threshold)).unwrap_or(false);

        if !needs_mask {
            return None;
        }

        // Build H matrix with zeros for masked points
        let mut h = DMatrix::zeros(dim_z, dim_x);
        for i in 0..dim_z {
            let point_idx = i / obj.dim_points;
            let score = scores.map(|s| s[point_idx]).unwrap_or(1.0);
            if score > self.config.detection_threshold {
                h[(i, i)] = 1.0;
            }
        }

        Some(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_new() {
        let config = TrackerConfig::from_distance_name("euclidean", 100.0);
        let tracker = Tracker::new(config).unwrap();

        assert_eq!(tracker.tracked_objects.len(), 0);
        assert_eq!(tracker.total_object_count(), 0);
    }

    #[test]
    fn test_tracker_invalid_config() {
        let mut config = TrackerConfig::from_distance_name("euclidean", 100.0);
        config.initialization_delay = -2;

        assert!(Tracker::new(config).is_err());
    }

    #[test]
    fn test_tracker_simple_update() {
        let mut config = TrackerConfig::from_distance_name("euclidean", 100.0);
        config.hit_counter_max = 5;
        config.initialization_delay = 2;

        let mut tracker = Tracker::new(config).unwrap();

        let det = Detection::from_slice(&[10.0, 20.0], 1, 2).unwrap();
        let active = tracker.update(vec![det], 1, None);

        // Should be initializing, not active yet
        assert_eq!(active.len(), 0);
        assert_eq!(tracker.tracked_objects.len(), 1);
    }

    #[test]
    fn test_tracker_initialization() {
        let mut config = TrackerConfig::from_distance_name("euclidean", 100.0);
        config.hit_counter_max = 5;
        config.initialization_delay = 2;

        let mut tracker = Tracker::new(config).unwrap();

        // First update - initializing (hit_counter = 1)
        let det = Detection::from_slice(&[10.0, 20.0], 1, 2).unwrap();
        let active = tracker.update(vec![det.clone()], 1, None);
        assert_eq!(active.len(), 0);

        // Second update - should be initialized now (hit_counter reaches 2)
        // Initializing objects don't decay hit_counter, so 1 + 1 = 2 >= initialization_delay
        let active = tracker.update(vec![det], 1, None);
        assert_eq!(active.len(), 1);
        assert!(active[0].id.is_some());
    }
}
