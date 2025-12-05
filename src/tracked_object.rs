//! TrackedObject struct for tracked objects maintained by the tracker.

use crate::camera_motion::CoordinateTransformation;
use crate::filter::FilterEnum;
use crate::Detection;
use nalgebra::DMatrix;
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicI32, Ordering};

/// Global ID counter for unique IDs across all factories.
static GLOBAL_ID_COUNTER: AtomicI32 = AtomicI32::new(0);

/// Get the next global ID (unique across all trackers/factories).
/// Uses Relaxed ordering since we only need uniqueness, not memory ordering.
#[inline]
pub fn get_next_global_id() -> i32 {
    GLOBAL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Factory for creating tracked objects with unique IDs.
///
/// This handles ID management for tracked objects, including:
/// - Global IDs that are unique across all factories/trackers
/// - Initializing IDs for objects in the initialization phase
/// - Permanent IDs for fully initialized objects
#[derive(Debug)]
pub struct TrackedObjectFactory {
    /// Counter for permanent (initialized) object IDs.
    permanent_id_counter: AtomicI32,
    /// Counter for initializing object IDs.
    initializing_id_counter: AtomicI32,
}

impl TrackedObjectFactory {
    /// Create a new TrackedObjectFactory.
    pub fn new() -> Self {
        Self {
            permanent_id_counter: AtomicI32::new(0),
            initializing_id_counter: AtomicI32::new(0),
        }
    }

    /// Get the next global ID (unique across all factories).
    #[inline]
    pub fn get_global_id(&self) -> i32 {
        get_next_global_id()
    }

    /// Get the next initializing ID (unique within this factory).
    #[inline]
    pub fn get_initializing_id(&self) -> i32 {
        self.initializing_id_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the next permanent ID (unique within this factory).
    #[inline]
    pub fn get_permanent_id(&self) -> i32 {
        self.permanent_id_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Get both a global ID and initializing ID for a new object.
    ///
    /// Returns (global_id, initializing_id).
    pub fn get_ids(&self) -> (i32, i32) {
        let global_id = self.get_global_id();
        let initializing_id = self.get_initializing_id();
        (global_id, initializing_id)
    }

    /// Get the current count of permanent IDs issued.
    pub fn permanent_id_count(&self) -> i32 {
        self.permanent_id_counter.load(Ordering::Relaxed)
    }

    /// Get the current count of initializing IDs issued.
    pub fn initializing_id_count(&self) -> i32 {
        self.initializing_id_counter.load(Ordering::Relaxed)
    }

    /// Reset the global ID counter (for testing only).
    #[cfg(test)]
    pub fn reset_global_counter() {
        GLOBAL_ID_COUNTER.store(0, Ordering::Relaxed);
    }
}

impl Default for TrackedObjectFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for TrackedObjectFactory {
    fn clone(&self) -> Self {
        // Create a new factory with current counter values
        Self {
            permanent_id_counter: AtomicI32::new(self.permanent_id_counter.load(Ordering::Relaxed)),
            initializing_id_counter: AtomicI32::new(
                self.initializing_id_counter.load(Ordering::Relaxed),
            ),
        }
    }
}

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

    /// Minimum distance to any detection in the current frame (for debugging).
    /// This is set by the tracker during update regardless of whether a match occurs.
    pub current_min_distance: Option<f64>,

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

    /// Boolean mask indicating which points have been detected at least once.
    /// This is used to track which points were initially detected vs inferred.
    pub detected_at_least_once_points: Vec<bool>,

    /// The Kalman filter maintaining this object's state (enum-based static dispatch).
    pub(crate) filter: FilterEnum,

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
            .field("current_min_distance", &self.current_min_distance)
            .field("past_detections", &self.past_detections)
            .field("label", &self.label)
            .field("reid_hit_counter", &self.reid_hit_counter)
            .field("estimate", &self.estimate)
            .field("estimate_velocity", &self.estimate_velocity)
            .field("is_initializing", &self.is_initializing)
            .field(
                "detected_at_least_once_points",
                &self.detected_at_least_once_points,
            )
            .field("filter", &"<Filter>")
            .field("num_points", &self.num_points)
            .field("dim_points", &self.dim_points)
            .field(
                "last_coord_transform",
                &self
                    .last_coord_transform
                    .as_ref()
                    .map(|_| "<CoordinateTransformation>"),
            )
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
        self.point_hit_counter.iter().map(|&c| c > 0).collect()
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
            current_min_distance: None,
            past_detections: VecDeque::new(),
            label: None,
            reid_hit_counter: None,
            estimate: DMatrix::zeros(1, 2),
            estimate_velocity: DMatrix::zeros(1, 2),
            is_initializing: true,
            detected_at_least_once_points: vec![true], // Default: 1 point, detected
            filter: FilterEnum::None(crate::filter::NoFilter::new(&DMatrix::zeros(1, 2))),
            num_points: 1,
            dim_points: 2,
            last_coord_transform: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    // ===== TrackedObjectFactory basic tests =====

    #[test]
    fn test_factory_get_initializing_id() {
        let factory = TrackedObjectFactory::new();

        // Should get sequential IDs
        assert_eq!(factory.get_initializing_id(), 0);
        assert_eq!(factory.get_initializing_id(), 1);
        assert_eq!(factory.get_initializing_id(), 2);
    }

    #[test]
    fn test_factory_get_permanent_id() {
        let factory = TrackedObjectFactory::new();

        // Should get sequential IDs
        assert_eq!(factory.get_permanent_id(), 0);
        assert_eq!(factory.get_permanent_id(), 1);
        assert_eq!(factory.get_permanent_id(), 2);
    }

    #[test]
    fn test_factory_get_ids() {
        // NOTE: Don't reset global counter - tests run in parallel and share it
        let factory = TrackedObjectFactory::new();

        let (global_id, init_id) = factory.get_ids();
        assert_eq!(init_id, 0); // Initializing ID starts at 0 per factory

        let (global_id2, init_id2) = factory.get_ids();
        assert_eq!(init_id2, 1);
        // Global IDs should be sequential (though not necessarily starting at 0)
        assert_eq!(global_id2, global_id + 1);
    }

    #[test]
    fn test_factory_global_id_uniqueness() {
        // NOTE: Don't reset global counter - tests run in parallel and share it
        let factory1 = TrackedObjectFactory::new();
        let factory2 = TrackedObjectFactory::new();

        // Global IDs from different factories should be unique
        let g1a = factory1.get_global_id();
        let g2a = factory2.get_global_id();
        let g1b = factory1.get_global_id();
        let g2b = factory2.get_global_id();

        let ids = vec![g1a, g2a, g1b, g2b];
        let unique_ids: HashSet<_> = ids.iter().cloned().collect();
        assert_eq!(
            ids.len(),
            unique_ids.len(),
            "All global IDs should be unique"
        );
    }

    #[test]
    fn test_factory_initializing_vs_permanent_ids() {
        let factory = TrackedObjectFactory::new();

        // Initializing and permanent IDs are independent
        assert_eq!(factory.get_initializing_id(), 0);
        assert_eq!(factory.get_permanent_id(), 0);
        assert_eq!(factory.get_initializing_id(), 1);
        assert_eq!(factory.get_permanent_id(), 1);

        // Check counters
        assert_eq!(factory.initializing_id_count(), 2);
        assert_eq!(factory.permanent_id_count(), 2);
    }

    #[test]
    fn test_factory_mixed_sequence() {
        let factory = TrackedObjectFactory::new();

        // Simulate: create 3 objects, then 2 get promoted, then 2 more created
        let init1 = factory.get_initializing_id(); // 0
        let init2 = factory.get_initializing_id(); // 1
        let init3 = factory.get_initializing_id(); // 2

        // Object 1 and 2 get promoted (get permanent IDs)
        let perm1 = factory.get_permanent_id(); // 0
        let perm2 = factory.get_permanent_id(); // 1

        // Object 3 dies (no permanent ID)
        // Two new objects
        let init4 = factory.get_initializing_id(); // 3
        let init5 = factory.get_initializing_id(); // 4

        assert_eq!(init1, 0);
        assert_eq!(init2, 1);
        assert_eq!(init3, 2);
        assert_eq!(perm1, 0);
        assert_eq!(perm2, 1);
        assert_eq!(init4, 3);
        assert_eq!(init5, 4);
    }

    // ===== Concurrent access tests =====

    #[test]
    fn test_factory_concurrent_initializing_ids() {
        let factory = Arc::new(TrackedObjectFactory::new());
        let num_threads = 10;
        let ids_per_thread = 100;

        let mut handles = vec![];

        for _ in 0..num_threads {
            let factory_clone = Arc::clone(&factory);
            let handle = thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..ids_per_thread {
                    ids.push(factory_clone.get_initializing_id());
                }
                ids
            });
            handles.push(handle);
        }

        let mut all_ids = Vec::new();
        for handle in handles {
            all_ids.extend(handle.join().unwrap());
        }

        // All IDs should be unique
        let unique_ids: HashSet<_> = all_ids.iter().cloned().collect();
        assert_eq!(
            all_ids.len(),
            unique_ids.len(),
            "All concurrent initializing IDs should be unique"
        );
        assert_eq!(all_ids.len(), num_threads * ids_per_thread);
    }

    #[test]
    fn test_factory_concurrent_permanent_ids() {
        let factory = Arc::new(TrackedObjectFactory::new());
        let num_threads = 10;
        let ids_per_thread = 100;

        let mut handles = vec![];

        for _ in 0..num_threads {
            let factory_clone = Arc::clone(&factory);
            let handle = thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..ids_per_thread {
                    ids.push(factory_clone.get_permanent_id());
                }
                ids
            });
            handles.push(handle);
        }

        let mut all_ids = Vec::new();
        for handle in handles {
            all_ids.extend(handle.join().unwrap());
        }

        // All IDs should be unique
        let unique_ids: HashSet<_> = all_ids.iter().cloned().collect();
        assert_eq!(
            all_ids.len(),
            unique_ids.len(),
            "All concurrent permanent IDs should be unique"
        );
        assert_eq!(all_ids.len(), num_threads * ids_per_thread);
    }

    #[test]
    fn test_factory_concurrent_multiple_factories() {
        // Note: Other tests may be running concurrently and also incrementing
        // the global counter. We use a barrier to synchronize our threads and
        // verify that the IDs we generate are unique within this test.

        use std::sync::Barrier;

        let num_factories = 4;
        let ids_per_factory = 100; // Reduced count for faster, more reliable test
        let expected_total = num_factories * ids_per_factory;

        let barrier = Arc::new(Barrier::new(num_factories));
        let mut handles = vec![];

        for _ in 0..num_factories {
            let barrier = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                let factory = TrackedObjectFactory::new();
                // Wait for all threads to be ready
                barrier.wait();

                let mut ids = Vec::new();
                for _ in 0..ids_per_factory {
                    ids.push(factory.get_global_id());
                }
                ids
            });
            handles.push(handle);
        }

        let mut all_ids = Vec::new();
        for handle in handles {
            all_ids.extend(handle.join().unwrap());
        }

        // Verify we collected the expected number of IDs from our threads
        assert_eq!(
            all_ids.len(),
            expected_total,
            "Should have collected {} IDs, got {}",
            expected_total,
            all_ids.len()
        );

        // All global IDs from this test should be unique among themselves
        let unique_ids: HashSet<_> = all_ids.iter().cloned().collect();
        assert_eq!(
            all_ids.len(),
            unique_ids.len(),
            "All {} IDs generated in this test should be unique, but only {} were unique",
            all_ids.len(),
            unique_ids.len()
        );
    }

    // ===== TrackedObject tests =====

    #[test]
    fn test_tracked_object_live_points() {
        let mut obj = TrackedObject::default();
        obj.point_hit_counter = vec![1, 0, 2, 0, 3];

        let live = obj.live_points();
        assert_eq!(live, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_tracked_object_default() {
        let obj = TrackedObject::default();

        assert_eq!(obj.id, None);
        assert_eq!(obj.global_id, 0);
        assert_eq!(obj.initializing_id, None);
        assert_eq!(obj.age, 0);
        assert_eq!(obj.hit_counter, 0);
        assert!(obj.is_initializing);
        assert_eq!(obj.num_points, 1);
        assert_eq!(obj.dim_points, 2);
    }

    #[test]
    fn test_tracked_object_get_estimate() {
        let mut obj = TrackedObject::default();
        obj.estimate = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        // Without transform, should return estimate directly
        let estimate = obj.get_estimate(false);
        assert_eq!(estimate[(0, 0)], 1.0);
        assert_eq!(estimate[(0, 1)], 2.0);
        assert_eq!(estimate[(1, 0)], 3.0);
        assert_eq!(estimate[(1, 1)], 4.0);
    }
}
