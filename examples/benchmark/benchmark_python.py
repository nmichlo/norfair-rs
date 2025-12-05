# /// script
# dependencies = ["numpy", "norfair"]
# python-version = "3.13"
# ///
"""
Python benchmark for norfair tracking.

Usage:
    python benchmark_python.py [scenario_name]

Example:
    python benchmark_python.py medium
"""

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np


def load_scenario(name: str) -> dict:
    """Load a scenario from the data directory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, f"{name}.json")

    if not os.path.exists(path):
        print(f"Error: Scenario '{name}' not found at {path}")
        print("Run generate_data.py first to create test data.")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def run_benchmark(
    scenario: dict,
    use_norfair_rs: bool,
) -> dict:
    """Run the tracking benchmark and return timing results."""

    if not use_norfair_rs:
        # original library
        try:
            from norfair import Detection, Tracker
        except ImportError:
            print("Error: norfair not installed. Run: uv pip install norfair")
            sys.exit(1)
    else:
        # drop in replacement!
        try:
            from norfair_rs import Detection, Tracker
        except ImportError:
            print("Error: norfair_rs not installed. Run: uv run maturin develop --release")
            sys.exit(1)

    # Create tracker with standard settings
    tracker = Tracker(
        distance_function="iou",
        distance_threshold=0.5,
        hit_counter_max=15,
        initialization_delay=3,
    )

    # Warm up
    for _ in range(10):
        tracker.update([])

    # Reset tracker
    tracker = Tracker(
        distance_function="iou",
        distance_threshold=0.5,
        hit_counter_max=15,
        initialization_delay=3,
    )

    # Run benchmark
    start_time = time.perf_counter()
    total_tracked = 0

    for frame_data in scenario["frames"]:
        # Convert detections to norfair format
        detections: List[Detection] = []
        for det in frame_data["detections"]:
            bbox = det["bbox"]
            # norfair expects shape (N, 2) for N points
            # For bounding boxes, use top-left and bottom-right corners
            points = np.array([
                [bbox[0], bbox[1]],  # top-left
                [bbox[2], bbox[3]],  # bottom-right
            ])
            detections.append(Detection(points=points))

        # Update tracker
        tracked_objects = tracker.update(detections=detections)
        total_tracked += len(tracked_objects)

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    num_frames = len(scenario["frames"])
    total_detections = sum(len(f["detections"]) for f in scenario["frames"])

    return {
        "language": "python_rs" if use_norfair_rs else "python",
        "scenario": scenario.get("name", "unknown"),
        "num_frames": num_frames,
        "total_detections": total_detections,
        "total_tracked": total_tracked,
        "elapsed_seconds": elapsed,
        "fps": num_frames / elapsed,
        "detections_per_second": total_detections / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark norfair tracking")
    parser.add_argument("scenario", nargs="?", default="medium", help="Scenario name (default: medium)")
    parser.add_argument("--norfair-rs", action="store_true", help="Use norfair_rs instead of norfair")
    args = parser.parse_args()

    use_norfair_rs = args.norfair_rs
    scenario_name = args.scenario

    # print(f"Loading scenario: {scenario_name}")
    scenario = load_scenario(scenario_name)
    scenario["name"] = scenario_name

    # print(f"Running Python benchmark...")
    # print(f"  Objects: {scenario['num_objects']}")
    # print(f"  Frames: {scenario['num_frames']}")

    results = run_benchmark(scenario, use_norfair_rs=use_norfair_rs)

    # print(f"\nResults:")
    # print(f"  Elapsed: {results['elapsed_seconds']:.3f}s")
    # print(f"  FPS: {results['fps']:.1f}")
    # print(f"  Detections/sec: {results['detections_per_second']:.0f}")

    # Output JSON for comparison
    print(f"\n{json.dumps(results)}")


if __name__ == "__main__":
    main()
