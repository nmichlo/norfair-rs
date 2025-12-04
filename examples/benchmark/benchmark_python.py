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

import json
import os
import sys
import time
from typing import List

import numpy as np

# Import norfair
try:
    from norfair import Detection, Tracker
except ImportError:
    print("Error: norfair not installed. Run: pip install norfair")
    sys.exit(1)


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


def run_benchmark(scenario: dict) -> dict:
    """Run the tracking benchmark and return timing results."""

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
        "language": "python",
        "scenario": scenario.get("name", "unknown"),
        "num_frames": num_frames,
        "total_detections": total_detections,
        "total_tracked": total_tracked,
        "elapsed_seconds": elapsed,
        "fps": num_frames / elapsed,
        "detections_per_second": total_detections / elapsed,
    }


def main():
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "medium"

    # print(f"Loading scenario: {scenario_name}")
    scenario = load_scenario(scenario_name)
    scenario["name"] = scenario_name

    # print(f"Running Python benchmark...")
    # print(f"  Objects: {scenario['num_objects']}")
    # print(f"  Frames: {scenario['num_frames']}")

    results = run_benchmark(scenario)

    # print(f"\nResults:")
    # print(f"  Elapsed: {results['elapsed_seconds']:.3f}s")
    # print(f"  FPS: {results['fps']:.1f}")
    # print(f"  Detections/sec: {results['detections_per_second']:.0f}")

    # Output JSON for comparison
    print(f"\n{json.dumps(results)}")


if __name__ == "__main__":
    main()
