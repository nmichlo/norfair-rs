//! End-to-end fixture tests for norfair-rust.
//!
//! These tests compare the Rust tracker output against Python reference fixtures
//! to ensure cross-implementation consistency.
//!
//! Run with: cargo test --release fixture

use nalgebra::DMatrix;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

use norfair_rs::distances::distance_by_name;
use norfair_rs::{Detection, Tracker, TrackerConfig};

// ============================================================================
// Fixture JSON Schema
// ============================================================================

#[derive(Debug, Deserialize)]
struct Fixture {
    tracker_config: TrackerConfigJson,
    steps: Vec<Step>,
}

#[derive(Debug, Deserialize)]
struct TrackerConfigJson {
    distance_function: String,
    distance_threshold: f64,
    hit_counter_max: i32,
    initialization_delay: i32,
}

#[derive(Debug, Deserialize)]
struct Step {
    frame_id: usize,
    inputs: Inputs,
    outputs: Outputs,
}

#[derive(Debug, Deserialize)]
struct Inputs {
    detections: Vec<DetectionJson>,
}

#[derive(Debug, Deserialize)]
struct DetectionJson {
    bbox: Vec<f64>,
    #[allow(dead_code)]
    ground_truth_id: i32,
}

#[derive(Debug, Deserialize)]
struct Outputs {
    tracked_objects: Vec<TrackedObjectJson>,
    all_objects: Vec<TrackedObjectJson>,
}

#[derive(Debug, Deserialize)]
struct TrackedObjectJson {
    id: Option<i32>,
    initializing_id: i32,
    estimate: Vec<Vec<f64>>,
    age: i32,
    hit_counter: i32,
    is_initializing: bool,
}

// ============================================================================
// Test Helpers
// ============================================================================

fn find_testdata_dir() -> PathBuf {
    // Try various locations relative to where tests run
    let candidates = [
        PathBuf::from("testdata/fixtures"),
        PathBuf::from("../testdata/fixtures"),
        PathBuf::from("../../testdata/fixtures"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }
    panic!("Could not find testdata/fixtures directory");
}

fn load_fixture(scenario: &str) -> Fixture {
    let testdata_dir = find_testdata_dir();
    let path = testdata_dir.join(format!("fixture_{}.json", scenario));

    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read fixture file {:?}: {}", path, e));

    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture file {:?}: {}", path, e))
}

fn create_tracker(config: &TrackerConfigJson) -> Tracker {
    let mut tracker_config =
        TrackerConfig::new(distance_by_name(&config.distance_function), config.distance_threshold);
    tracker_config.hit_counter_max = config.hit_counter_max;
    tracker_config.initialization_delay = config.initialization_delay;

    Tracker::new(tracker_config).expect("Failed to create tracker")
}

fn compare_tracked_objects(
    step_idx: usize,
    frame_id: usize,
    expected: &[TrackedObjectJson],
    actual: &[&norfair_rs::TrackedObject],
    label: &str,
    tolerance: f64,
) -> Result<(), String> {
    // First compare counts
    if expected.len() != actual.len() {
        let mut msg = format!(
            "FIRST DIVERGENCE at step {} (frame_id={}):\n",
            step_idx, frame_id
        );
        msg.push_str(&format!(
            "  Expected {} {}: {}\n",
            expected.len(),
            label,
            expected.len()
        ));
        msg.push_str(&format!(
            "  Actual {} {}: {}\n\n",
            actual.len(),
            label,
            actual.len()
        ));

        msg.push_str("Expected objects:\n");
        for obj in expected {
            msg.push_str(&format!(
                "  ID={:?}, initializing_id={}, estimate={:?}, age={}, hit_counter={}, is_initializing={}\n",
                obj.id, obj.initializing_id, obj.estimate, obj.age, obj.hit_counter, obj.is_initializing
            ));
        }
        msg.push_str("\nActual objects:\n");
        for obj in actual {
            msg.push_str(&format!(
                "  ID={:?}, initializing_id={:?}, estimate={:?}, age={}, hit_counter={}, is_initializing={}\n",
                obj.id, obj.initializing_id, obj.estimate, obj.age, obj.hit_counter, obj.is_initializing
            ));
        }
        return Err(msg);
    }

    // Compare each object (by position in list for now - might need ID matching later)
    for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
        // Compare ID (Python starts at 1, Rust starts at 0 - adjust for offset)
        let act_id = act.id.map(|id| id + 1);
        if exp.id != act_id {
            return Err(format!(
                "Step {} frame {}: Object {} ID mismatch: expected {:?}, got {:?} (Rust raw: {:?})",
                step_idx, frame_id, i, exp.id, act_id, act.id
            ));
        }

        // Compare initializing_id (Python starts at 1, Rust starts at 0 - adjust for offset)
        // Python uses counter++ (returns after increment), Rust uses fetch_add (returns before increment)
        let act_init_id = act.initializing_id.map(|id| id + 1).unwrap_or(-1);
        if exp.initializing_id != act_init_id {
            return Err(format!(
                "Step {} frame {}: Object {} initializing_id mismatch: expected {}, got {:?} (Rust raw: {:?})",
                step_idx, frame_id, i, exp.initializing_id, act_init_id, act.initializing_id
            ));
        }

        // Compare age
        if exp.age != act.age {
            return Err(format!(
                "Step {} frame {}: Object {} age mismatch: expected {}, got {}",
                step_idx, frame_id, i, exp.age, act.age
            ));
        }

        // Compare hit_counter
        if exp.hit_counter != act.hit_counter {
            return Err(format!(
                "Step {} frame {}: Object {} hit_counter mismatch: expected {}, got {}",
                step_idx, frame_id, i, exp.hit_counter, act.hit_counter
            ));
        }

        // Compare is_initializing
        if exp.is_initializing != act.is_initializing {
            return Err(format!(
                "Step {} frame {}: Object {} is_initializing mismatch: expected {}, got {}",
                step_idx, frame_id, i, exp.is_initializing, act.is_initializing
            ));
        }

        // Compare estimates (with tolerance)
        let exp_estimate: Vec<Vec<f64>> = exp.estimate.clone();
        let act_estimate = &act.estimate;

        for (row_idx, exp_row) in exp_estimate.iter().enumerate() {
            for (col_idx, &exp_val) in exp_row.iter().enumerate() {
                let act_val = act_estimate[(row_idx, col_idx)];
                let diff = (exp_val - act_val).abs();
                if diff > tolerance {
                    return Err(format!(
                        "Step {} frame {}: Object {} estimate[{}][{}] mismatch: expected {}, got {} (diff={})",
                        step_idx, frame_id, i, row_idx, col_idx, exp_val, act_val, diff
                    ));
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Fixture Test Runner
// ============================================================================

fn run_fixture_test(scenario: &str) {
    let fixture = load_fixture(scenario);
    let mut tracker = create_tracker(&fixture.tracker_config);

    // Tolerance for numerical comparisons
    // Start strict per CLAUDE.md guidance - loosen only if there are real precision differences
    let tolerance = 1e-6;

    for (step_idx, step) in fixture.steps.iter().enumerate() {
        // Convert inputs to detections
        let detections: Vec<Detection> = step
            .inputs
            .detections
            .iter()
            .filter_map(|det| {
                // Create 2x2 matrix for bounding box (top-left, bottom-right)
                let points = DMatrix::from_row_slice(
                    2,
                    2,
                    &[det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]],
                );
                Detection::new(points).ok()
            })
            .collect();

        // Update tracker
        let tracked_objects = tracker.update(detections, 1, None);

        // Compare tracked_objects (returned by update - initialized, active objects)
        // update() returns Vec<&TrackedObject>, so we can use it directly
        if let Err(msg) = compare_tracked_objects(
            step_idx,
            step.frame_id,
            &step.outputs.tracked_objects,
            &tracked_objects,
            "tracked_objects",
            tolerance,
        ) {
            panic!("{}", msg);
        }

        // Compare all_objects (all internal objects including initializing)
        // Sort by initializing_id to ensure consistent ordering for comparison
        let mut all_refs: Vec<&norfair_rs::TrackedObject> = tracker.tracked_objects.iter().collect();
        all_refs.sort_by_key(|obj| obj.initializing_id.unwrap_or(i32::MAX));
        if let Err(msg) = compare_tracked_objects(
            step_idx,
            step.frame_id,
            &step.outputs.all_objects,
            &all_refs,
            "all_objects",
            tolerance,
        ) {
            panic!("{}", msg);
        }
    }

    println!(
        "Fixture test '{}' passed: {} steps verified",
        scenario,
        fixture.steps.len()
    );
}

// ============================================================================
// Test Cases
// ============================================================================

#[test]
fn test_fixture_small() {
    run_fixture_test("small");
}

#[test]
fn test_fixture_medium() {
    run_fixture_test("medium");
}
