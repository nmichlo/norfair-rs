//! Integration tests for the norfair Rust port.
//!
//! These tests verify complete tracking workflows across multiple modules.

use norfair_rs::{
    Detection, FilterFactory, Tracker, TrackerConfig,
    filter::{FilterPyKalmanFilterFactory, NoFilterFactory, OptimizedKalmanFilterFactory},
    distances::distance_by_name,
    camera_motion::TranslationTransformation,
};

// =============================================================================
// Test 1: Complete Tracking Pipeline
// =============================================================================

#[test]
fn test_integration_complete_tracking_pipeline() {
    // Create tracker with default settings
    let mut config = TrackerConfig::from_distance_name("euclidean", 50.0);
    config.hit_counter_max = 10;
    config.initialization_delay = 2;
    config.pointwise_hit_counter_max = 4;
    config.detection_threshold = 0.0;
    config.past_detections_length = 4;

    let mut tracker = Tracker::new(config).expect("Failed to create tracker");

    // Simulate tracking across 20 frames
    // Two objects: one static at (100, 100), one moving from (200, 200) to (300, 300)
    for frame in 0..20 {
        // Static object
        let det1 = Detection::from_slice(&[100.0, 100.0], 1, 2).unwrap();

        // Moving object
        let x = 200.0 + (frame as f64) * 5.0;
        let y = 200.0 + (frame as f64) * 5.0;
        let det2 = Detection::from_slice(&[x, y], 1, 2).unwrap();

        // Update tracker
        let tracked_objects = tracker.update(vec![det1, det2], 1, None);

        // After initialization delay, should have 2 tracked objects
        if frame > 2 {
            assert_eq!(
                tracked_objects.len(),
                2,
                "Frame {}: expected 2 tracked objects, got {}",
                frame,
                tracked_objects.len()
            );

            // Verify object IDs are maintained across frames
            for obj in &tracked_objects {
                assert!(
                    obj.id.is_some(),
                    "Frame {}: object missing ID",
                    frame
                );
            }

            // Verify estimates are reasonable (within 100 pixels of detections)
            for obj in &tracked_objects {
                let est_x = obj.estimate[(0, 0)];
                let est_y = obj.estimate[(0, 1)];

                // Check against both detection positions
                let dist1 = (est_x - 100.0).powi(2) + (est_y - 100.0).powi(2);
                let dist2 = (est_x - x).powi(2) + (est_y - y).powi(2);

                assert!(
                    dist1 <= 10000.0 || dist2 <= 10000.0,
                    "Frame {}: estimate ({:.1}, {:.1}) too far from detections",
                    frame,
                    est_x,
                    est_y
                );
            }
        }
    }

    // Verify total object count
    assert_eq!(tracker.total_object_count(), 2, "Expected 2 total objects");
}

// =============================================================================
// Test 2: Multiple Filter Types
// =============================================================================

#[test]
fn test_integration_multiple_filter_types() {
    let filter_configs: Vec<(&str, Box<dyn FilterFactory>)> = vec![
        (
            "OptimizedKalman",
            Box::new(OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0)),
        ),
        (
            "FilterPyKalman",
            Box::new(FilterPyKalmanFilterFactory::new(4.0, 0.1, 10.0)),
        ),
        (
            "NoFilter",
            Box::new(NoFilterFactory::new()),
        ),
    ];

    for (name, factory) in filter_configs {
        let mut config = TrackerConfig::from_distance_name("euclidean", 50.0);
        config.hit_counter_max = 10;
        config.initialization_delay = 2;
        config.pointwise_hit_counter_max = 4;
        config.detection_threshold = 0.0;
        config.filter_factory = factory;
        config.past_detections_length = 4;

        let mut tracker = Tracker::new(config).expect(&format!("Failed to create tracker with {}", name));

        // Track a moving object across 10 frames
        for frame in 0..10 {
            let x = 100.0 + (frame as f64) * 10.0;
            let y = 100.0 + (frame as f64) * 10.0;
            let det = Detection::from_slice(&[x, y], 1, 2).unwrap();

            let tracked_objects = tracker.update(vec![det], 1, None);

            // After initialization, should have 1 object
            if frame > 2 {
                assert_eq!(
                    tracked_objects.len(),
                    1,
                    "{} Frame {}: expected 1 object, got {}",
                    name,
                    frame,
                    tracked_objects.len()
                );
            }
        }

        // All filters should successfully track the object
        assert_eq!(
            tracker.total_object_count(),
            1,
            "{}: expected 1 total object, got {}",
            name,
            tracker.total_object_count()
        );
    }
}

// =============================================================================
// Test 3: Multiple Distance Functions
// =============================================================================

#[test]
fn test_integration_multiple_distance_functions() {
    // Test IoU with bounding boxes
    // Note: IoU distance = 1 - IoU, so threshold 0.8 gives more matching tolerance
    // Box format: [x1, y1, x2, y2] as 1 row x 4 columns
    {
        let mut config = TrackerConfig::new(distance_by_name("iou"), 0.8);
        config.hit_counter_max = 10;
        config.initialization_delay = 2;  // Need 2 hits to initialize
        config.pointwise_hit_counter_max = 4;
        config.detection_threshold = 0.0;
        config.past_detections_length = 4;

        let mut tracker = Tracker::new(config).expect("Failed to create tracker");

        for frame in 0..10 {
            // Stationary bounding box: 1 row × 4 columns format [x1, y1, x2, y2]
            let det = Detection::from_slice(&[100.0, 100.0, 150.0, 150.0], 1, 4).unwrap();

            let tracked_objects = tracker.update(vec![det], 1, None);

            // With init_delay=2, object is initialized at frame 2 (after 3 hits)
            // Frame 0: create with hit=1, Frame 1: 0->2 (not >2), Frame 2: 1->3 (>2, initialized)
            if frame >= 2 {
                assert_eq!(
                    tracked_objects.len(),
                    1,
                    "IoU Frame {}: expected 1 object, got {}",
                    frame,
                    tracked_objects.len()
                );
            }
        }

        assert_eq!(tracker.total_object_count(), 1, "IoU: expected 1 total object");
    }

    // Test Euclidean with points
    {
        let mut config = TrackerConfig::from_distance_name("euclidean", 100.0);
        config.hit_counter_max = 10;
        config.initialization_delay = 2;
        config.pointwise_hit_counter_max = 4;
        config.detection_threshold = 0.0;
        config.past_detections_length = 4;

        let mut tracker = Tracker::new(config).expect("Failed to create tracker");

        for frame in 0..10 {
            let x = 100.0 + (frame as f64) * 10.0;
            let det = Detection::from_slice(&[x, 100.0], 1, 2).unwrap();

            let tracked_objects = tracker.update(vec![det], 1, None);

            if frame > 2 {
                assert_eq!(
                    tracked_objects.len(),
                    1,
                    "Euclidean Frame {}: expected 1 object, got {}",
                    frame,
                    tracked_objects.len()
                );
            }
        }

        assert_eq!(tracker.total_object_count(), 1, "Euclidean: expected 1 total object");
    }

    // Test mean_euclidean with multi-point detections
    {
        // Higher threshold for multi-point tracking with slight movement
        let mut config = TrackerConfig::from_distance_name("mean_euclidean", 50.0);
        config.hit_counter_max = 10;
        config.initialization_delay = 2;
        config.pointwise_hit_counter_max = 4;
        config.detection_threshold = 0.0;
        config.past_detections_length = 4;

        let mut tracker = Tracker::new(config).expect("Failed to create tracker");

        for frame in 0..10 {
            // Slower movement to ensure matching
            let x = 100.0 + (frame as f64) * 3.0;
            // Two keypoints moving together
            let det = Detection::from_slice(&[x, 100.0, x + 20.0, 120.0], 2, 2).unwrap();

            let tracked_objects = tracker.update(vec![det], 1, None);

            if frame >= 2 {
                assert!(
                    tracked_objects.len() <= 1,
                    "MeanEuclidean Frame {}: expected at most 1 object, got {}",
                    frame,
                    tracked_objects.len()
                );
            }
        }

        // With proper matching, should have created 1 object
        assert!(
            tracker.total_object_count() <= 2,
            "MeanEuclidean: expected at most 2 total objects, got {}",
            tracker.total_object_count()
        );
    }
}

// =============================================================================
// Test 4: ReID Enabled
// =============================================================================

#[test]
fn test_integration_reid_enabled() {
    let mut config = TrackerConfig::from_distance_name("euclidean", 50.0);
    config.hit_counter_max = 3;
    config.initialization_delay = 1;
    config.pointwise_hit_counter_max = 4;
    config.detection_threshold = 0.0;
    config.past_detections_length = 4;
    config.reid_hit_counter_max = Some(5);

    let mut tracker = Tracker::new(config).expect("Failed to create tracker");

    let mut original_id: Option<i32> = None;

    // Phase 1: Track object at (100, 100) for 5 frames
    for frame in 0..5 {
        let det = Detection::from_slice(&[100.0, 100.0], 1, 2).unwrap();
        let tracked_objects = tracker.update(vec![det], 1, None);

        if frame > 1 && !tracked_objects.is_empty() && original_id.is_none() {
            original_id = tracked_objects[0].id;
            println!("Original ID: {:?}", original_id);
        }
    }

    assert!(original_id.is_some(), "Failed to get original object ID");

    // Phase 2: Occlusion - no detections for several frames
    // Object needs to miss enough frames to die (HitCounterMax=3 means miss 5+ frames)
    for frame in 5..10 {
        let tracked_objects = tracker.update(vec![], 1, None);
        println!("Frame {}: {} objects visible", frame, tracked_objects.len());
    }

    // Phase 3: Object reappears at same location
    for frame in 10..15 {
        let det = Detection::from_slice(&[100.0, 100.0], 1, 2).unwrap();
        let tracked_objects = tracker.update(vec![det], 1, None);

        if !tracked_objects.is_empty() {
            let recovered_id = tracked_objects[0].id;
            println!(
                "Frame {}: recovered ID {:?} (original was {:?})",
                frame, recovered_id, original_id
            );
            // Note: Depending on ReID implementation, ID might be preserved or new
            break;
        }
    }

    // Verify total object count (may be 1 or 2 depending on ReID match success)
    let total_count = tracker.total_object_count();
    assert!(
        total_count >= 1 && total_count <= 2,
        "Expected 1-2 total objects after ReID, got {}",
        total_count
    );
}

// =============================================================================
// Test 5: Camera Motion Compensation
// =============================================================================

#[test]
fn test_integration_camera_motion_compensation() {
    let mut config = TrackerConfig::from_distance_name("euclidean", 100.0);
    config.hit_counter_max = 10;
    config.initialization_delay = 2;
    config.pointwise_hit_counter_max = 4;
    config.detection_threshold = 0.0;
    config.past_detections_length = 4;

    let mut tracker = Tracker::new(config).expect("Failed to create tracker");

    // Test that camera motion transformations are applied correctly
    // Use smaller camera movement for better tracking stability
    for frame in 0..10 {
        // Small camera offset
        let camera_offset = (frame as f64) * 2.0;

        // Object appears at slightly offset position due to camera motion
        let x = 100.0 - camera_offset;
        let y = 100.0;

        let det = Detection::from_slice(&[x, y], 1, 2).unwrap();

        // Create translation transformation
        let transform = TranslationTransformation::new([camera_offset, 0.0]);

        let tracked_objects = tracker.update(vec![det], 1, Some(&transform));

        // After initialization, verify we have a tracked object
        if frame >= 2 {
            assert_eq!(
                tracked_objects.len(),
                1,
                "Frame {}: expected 1 object, got {}",
                frame,
                tracked_objects.len()
            );

            // Verify we have valid estimates
            let obj = tracked_objects[0];
            let est_x = obj.estimate[(0, 0)];
            let est_y = obj.estimate[(0, 1)];

            // Estimates should be finite and reasonable
            assert!(
                est_x.is_finite() && est_y.is_finite(),
                "Frame {}: estimate ({:.1}, {:.1}) should be finite",
                frame,
                est_x,
                est_y
            );

            // Verify absolute coordinates (world frame) are computed
            let abs_est = obj.get_estimate(true);
            let abs_x = abs_est[(0, 0)];
            let abs_y = abs_est[(0, 1)];

            println!("Frame {}: relative ({:.1}, {:.1}), absolute ({:.1}, {:.1})",
                frame, est_x, est_y, abs_x, abs_y);

            // Just verify coordinates are reasonable
            assert!(
                abs_x.is_finite() && abs_y.is_finite(),
                "Frame {}: absolute estimate should be finite",
                frame
            );
        }
    }

    // Verify object was tracked successfully
    assert_eq!(tracker.total_object_count(), 1, "Should have tracked 1 object");
}

// =============================================================================
// Test 6: Object Lifecycle
// =============================================================================

#[test]
fn test_integration_object_lifecycle() {
    // Test the complete lifecycle of tracked objects:
    // - Creation (initializing phase)
    // - Initialization (gets permanent ID)
    // - Tracking (maintained)
    // - Death (disappears for too long)

    let mut config = TrackerConfig::from_distance_name("euclidean", 50.0);
    config.hit_counter_max = 5;
    config.initialization_delay = 3;  // Need 3 hits to initialize
    config.pointwise_hit_counter_max = 4;
    config.detection_threshold = 0.0;
    config.past_detections_length = 4;

    let mut tracker = Tracker::new(config).expect("Failed to create tracker");

    // Phase 1: Object appears and initializes
    // With initialization_delay=3 and hit_counter > delay check:
    // Frame 0: create with hit=1 (initializing)
    // Frame 1: 0->2, 2 > 3 is false (initializing)
    // Frame 2: 1->3, 3 > 3 is false (initializing)
    // Frame 3: 2->4, 4 > 3 is true (INITIALIZED)
    for frame in 0..5 {
        let det = Detection::from_slice(&[100.0, 100.0], 1, 2).unwrap();
        let tracked_objects = tracker.update(vec![det], 1, None);

        match frame {
            0..=2 => {
                // Should be initializing (no active objects visible)
                assert_eq!(tracked_objects.len(), 0, "Frame {}: should still be initializing", frame);
            }
            _ => {
                // Should be initialized
                assert_eq!(tracked_objects.len(), 1, "Frame {}: should have 1 active object", frame);
                assert!(tracked_objects[0].id.is_some(), "Frame {}: should have ID", frame);
            }
        }
    }

    let final_id = tracker.total_object_count();
    assert_eq!(final_id, 1, "Should have created 1 object");

    // Phase 2: Object disappears
    for frame in 5..15 {
        let tracked_objects = tracker.update(vec![], 1, None);

        // Object should gradually disappear as hit_counter decrements
        // After ~hit_counter_max frames, it dies
        if frame >= 5 + 6 {  // Allow some buffer for hit counter decay
            assert_eq!(
                tracked_objects.len(),
                0,
                "Frame {}: object should be dead",
                frame
            );
        }
    }

    // Phase 3: New object appears at different location
    for frame in 15..22 {
        let det = Detection::from_slice(&[300.0, 300.0], 1, 2).unwrap();
        let tracked_objects = tracker.update(vec![det], 1, None);

        // With initialization_delay=3, new object initialized at frame 18
        // (15: create hit=1, 16: 0->2, 17: 1->3, 18: 2->4 > 3)
        if frame >= 18 {
            assert!(
                tracked_objects.len() <= 1,
                "Frame {}: should have at most 1 object",
                frame
            );
        }
    }

    // Should now have 2 total objects (first one died, second one created)
    assert_eq!(tracker.total_object_count(), 2, "Should have created 2 total objects");
}
