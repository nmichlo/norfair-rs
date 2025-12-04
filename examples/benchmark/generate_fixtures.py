#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["norfair>=2.0", "numpy"]
# ///
"""
Generate E2E fixture files for testing Go and Rust norfair implementations.

This script runs the Python norfair tracker on scenario data and captures
the full state at each step (inputs and outputs) for cross-implementation testing.

Usage:
    uv run generate_fixtures.py [scenario_name]
    uv run generate_fixtures.py --all

Examples:
    uv run generate_fixtures.py small
    uv run generate_fixtures.py medium
    uv run generate_fixtures.py --all
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from norfair import Detection, Tracker


# Tracker configuration - must match benchmark settings
TRACKER_CONFIG = {
    "distance_function": "iou",
    "distance_threshold": 0.5,
    "hit_counter_max": 15,
    "initialization_delay": 3,
}


def load_scenario(name: str) -> dict:
    """Load a scenario from the data directory."""
    data_dir = Path(__file__).parent / "data"
    path = data_dir / f"{name}.json"

    if not path.exists():
        print(f"Error: Scenario '{name}' not found at {path}")
        print("Run generate_data.py first to create test data.")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def tracked_object_to_dict(obj) -> dict[str, Any]:
    """Convert a TrackedObject to a dictionary with full state."""
    return {
        "id": obj.id,
        "initializing_id": obj.initializing_id,
        "estimate": obj.estimate.tolist(),
        "age": obj.age,
        "hit_counter": obj.hit_counter,
        "is_initializing": obj.is_initializing,
    }


def generate_fixture(scenario_name: str) -> dict:
    """Run tracker on scenario and capture all step inputs/outputs."""
    scenario = load_scenario(scenario_name)

    # Create tracker with standard settings
    tracker = Tracker(
        distance_function=TRACKER_CONFIG["distance_function"],
        distance_threshold=TRACKER_CONFIG["distance_threshold"],
        hit_counter_max=TRACKER_CONFIG["hit_counter_max"],
        initialization_delay=TRACKER_CONFIG["initialization_delay"],
    )

    steps = []

    for frame_data in scenario["frames"]:
        frame_id = frame_data["frame_id"]

        # Build inputs
        inputs = {
            "detections": frame_data["detections"],  # Keep original format: {bbox, ground_truth_id}
        }

        # Convert detections to norfair format
        detections: list[Detection] = []
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

        # Capture outputs
        outputs = {
            # What tracker.update() returns (initialized, active objects)
            "tracked_objects": [tracked_object_to_dict(obj) for obj in tracked_objects],
            # ALL internal objects including initializing ones (for debugging)
            "all_objects": [tracked_object_to_dict(obj) for obj in tracker.tracked_objects],
        }

        steps.append({
            "frame_id": frame_id,
            "inputs": inputs,
            "outputs": outputs,
        })

    return {
        "tracker_config": TRACKER_CONFIG,
        "steps": steps,
    }


def save_fixture(fixture: dict, scenario_name: str) -> Path:
    """Save fixture to fixtures directory."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    output_path = fixtures_dir / f"fixture_{scenario_name}.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    return output_path


def main():
    scenarios = ["small", "medium"]

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            pass  # Use default list
        elif arg in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)
        else:
            scenarios = [arg]

    for scenario_name in scenarios:
        print(f"Generating fixture for: {scenario_name}")
        fixture = generate_fixture(scenario_name)
        output_path = save_fixture(fixture, scenario_name)
        print(f"  Saved to: {output_path}")
        print(f"  Steps: {len(fixture['steps'])}")

    print("\nDone! Copy fixtures to test directories:")
    print("  cp fixtures/*.json ../../testdata/fixtures/")
    print("  cp fixtures/*.json ../../../norfair-go/testdata/fixtures/")


if __name__ == "__main__":
    main()
