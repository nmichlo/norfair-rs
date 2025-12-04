//! Benchmark for norfair-rust tracking.
//!
//! Usage:
//!     cargo run --release --example benchmark_rust [scenario_name]
//!
//! Example:
//!     cargo run --release --example benchmark_rust medium

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

// Note: This would use the norfair crate in a real setup
// For now, we define a minimal inline implementation for benchmarking

#[derive(Debug, Deserialize)]
struct DetectionData {
    bbox: Vec<f64>,
    ground_truth_id: i32,
}

#[derive(Debug, Deserialize)]
struct FrameData {
    frame_id: i32,
    detections: Vec<DetectionData>,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    #[serde(default)]
    name: String,
    seed: i32,
    num_objects: i32,
    num_frames: i32,
    detection_prob: f64,
    noise_std: f64,
    frames: Vec<FrameData>,
}

#[derive(Debug, Serialize)]
struct Results {
    language: String,
    scenario: String,
    num_frames: i32,
    total_detections: i32,
    total_tracked: i32,
    elapsed_seconds: f64,
    fps: f64,
    detections_per_second: f64,
}

fn find_data_dir() -> Option<PathBuf> {
    // Try various locations
    let candidates = [
        PathBuf::from("data"),
        PathBuf::from("examples/benchmark/data"),
        PathBuf::from("../data"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Some(candidate.clone());
        }
    }
    None
}

fn load_scenario(name: &str) -> Result<Scenario, Box<dyn std::error::Error>> {
    let data_dir = find_data_dir().ok_or("Data directory not found")?;
    let path = data_dir.join(format!("{}.json", name));

    let content = fs::read_to_string(&path)?;
    let mut scenario: Scenario = serde_json::from_str(&content)?;
    scenario.name = name.to_string();
    Ok(scenario)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let scenario_name = args.get(1).map(|s| s.as_str()).unwrap_or("medium");

    // println!("Loading scenario: {}", scenario_name);
    let scenario = match load_scenario(scenario_name) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Run generate_data.py first to create test data.");
            std::process::exit(1);
        }
    };

    // println!("Running Rust benchmark...");
    // println!("  Objects: {}", scenario.num_objects);
    // println!("  Frames: {}", scenario.num_frames);

    // Import norfair components
    use norfair::{Detection, Tracker, TrackerConfig};
    use norfair::distances::distance_by_name;

    // Create tracker with standard settings
    let mut config = TrackerConfig::new(distance_by_name("iou"), 0.5);
    config.hit_counter_max = 15;
    config.initialization_delay = 3;

    let mut tracker = Tracker::new(config)?;

    // Warm up
    for _ in 0..10 {
        tracker.update(vec![], 1, None);
    }

    // Reset tracker
    let mut config = TrackerConfig::new(distance_by_name("iou"), 0.5);
    config.hit_counter_max = 15;
    config.initialization_delay = 3;
    let mut tracker = Tracker::new(config)?;

    // Run benchmark
    let start_time = Instant::now();
    let mut total_tracked = 0i32;
    let mut total_detections = 0i32;

    for frame_data in &scenario.frames {
        // Convert detections to norfair format
        let detections: Vec<Detection> = frame_data
            .detections
            .iter()
            .filter_map(|det| {
                // Create 2x2 matrix for bounding box (top-left, bottom-right)
                let points = DMatrix::from_row_slice(2, 2, &[
                    det.bbox[0], det.bbox[1],  // top-left
                    det.bbox[2], det.bbox[3],  // bottom-right
                ]);
                Detection::new(points).ok()
            })
            .collect();

        total_detections += detections.len() as i32;

        // Update tracker
        let tracked_objects = tracker.update(detections, 1, None);
        total_tracked += tracked_objects.len() as i32;
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let num_frames = scenario.frames.len() as i32;

    let results = Results {
        language: "rust".to_string(),
        scenario: scenario.name.clone(),
        num_frames,
        total_detections,
        total_tracked,
        elapsed_seconds: elapsed,
        fps: num_frames as f64 / elapsed,
        detections_per_second: total_detections as f64 / elapsed,
    };

    // println!("\nResults:");
    // println!("  Elapsed: {:.3}s", results.elapsed_seconds);
    // println!("  FPS: {:.1}", results.fps);
    // println!("  Detections/sec: {:.0}", results.detections_per_second);

    // Output JSON for comparison
    println!("\n{}", serde_json::to_string(&results)?);

    Ok(())
}
