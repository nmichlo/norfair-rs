#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["norfair>=2.0", "numpy"]
# ///
"""
Generate E2E fixture files for testing Go and Rust norfair implementations.

This script runs the Python norfair tracker on scenario data and captures
the full state at each step (inputs and outputs) for cross-implementation testing.

Supports multiple tracker configurations including:
- Different distance functions (iou, euclidean)
- ReID (re-identification) with distance functions
- Various hit_counter_max and initialization_delay settings

Usage:
    uv run generate_fixtures.py [scenario_name]
    uv run generate_fixtures.py --all
    uv run generate_fixtures.py --config CONFIG_NAME

Examples:
    uv run generate_fixtures.py small
    uv run generate_fixtures.py --all
    uv run generate_fixtures.py --config reid_euclidean
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from norfair import Detection, Tracker

# ============================================================================
# Tracker Configurations
# ============================================================================


# Simple euclidean distance for ReID (comparing estimates between TrackedObjects)
def reid_euclidean_distance(obj1, obj2):
    """Simple euclidean distance between TrackedObject estimates for ReID."""
    return np.linalg.norm(obj1.estimate - obj2.estimate)


# Configuration registry: name -> (config_dict, scenario_name)
FIXTURE_CONFIGS = {
    # Original fixtures - IoU with default settings
    "small": (
        {
            "distance_function": "iou",
            "distance_threshold": 0.5,
            "hit_counter_max": 15,
            "initialization_delay": 3,
        },
        "small",  # scenario to use
    ),
    "medium": (
        {
            "distance_function": "iou",
            "distance_threshold": 0.5,
            "hit_counter_max": 15,
            "initialization_delay": 3,
        },
        "medium",
    ),
    # Euclidean distance with bounding box centers
    "euclidean_small": (
        {
            "distance_function": "euclidean",
            "distance_threshold": 100.0,  # pixels
            "hit_counter_max": 10,
            "initialization_delay": 2,
        },
        "small",
    ),
    # ReID with euclidean distance - uses occlusion scenario
    "reid_euclidean": (
        {
            "distance_function": "euclidean",
            "distance_threshold": 50.0,
            "hit_counter_max": 3,  # Short - objects die quickly to test ReID
            "initialization_delay": 2,
            # ReID configuration
            "reid_distance_function": reid_euclidean_distance,
            "reid_distance_threshold": 100.0,
            "reid_hit_counter_max": 10,  # Long - survive in ReID phase
        },
        "occlusion",  # Uses occlusion scenario for ReID testing
    ),
    # Fast initialization (delay=0)
    "fast_init": (
        {
            "distance_function": "iou",
            "distance_threshold": 0.5,
            "hit_counter_max": 10,
            "initialization_delay": 0,
        },
        "small",
    ),
    # IoU on occlusion data (no ReID - to compare)
    "iou_occlusion": (
        {
            "distance_function": "iou",
            "distance_threshold": 0.5,
            "hit_counter_max": 5,
            "initialization_delay": 2,
        },
        "occlusion",
    ),
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
        "reid_hit_counter": obj.reid_hit_counter,  # Capture ReID state
    }


def config_to_json_serializable(config: dict) -> dict:
    """Convert config to JSON-serializable format (removing callables)."""
    result = {}
    for key, value in config.items():
        if callable(value):
            # Store function name instead of the callable
            result[key] = f"<callable:{value.__name__}>"
        else:
            result[key] = value
    return result


def generate_fixture(config_name: str) -> dict:
    """Run tracker on scenario and capture all step inputs/outputs."""
    if config_name not in FIXTURE_CONFIGS:
        print(f"Error: Unknown config '{config_name}'")
        print(f"Available configs: {list(FIXTURE_CONFIGS.keys())}")
        sys.exit(1)

    tracker_config, scenario_name = FIXTURE_CONFIGS[config_name]
    scenario = load_scenario(scenario_name)

    # Create tracker with configuration
    tracker = Tracker(**tracker_config)

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
            points = np.array(
                [
                    [bbox[0], bbox[1]],  # top-left
                    [bbox[2], bbox[3]],  # bottom-right
                ]
            )
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

        steps.append(
            {
                "frame_id": frame_id,
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    return {
        "config_name": config_name,
        "scenario_name": scenario_name,
        "tracker_config": config_to_json_serializable(tracker_config),
        "steps": steps,
    }


def save_fixture(fixture: dict, config_name: str) -> Path:
    """Save fixture to fixtures directory."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    output_path = fixtures_dir / f"fixture_{config_name}.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    return output_path


def main():
    configs_to_generate = []

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            configs_to_generate = list(FIXTURE_CONFIGS.keys())
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            print("\nAvailable configurations:")
            for name, (config, scenario) in FIXTURE_CONFIGS.items():
                reid = "reid_distance_function" in config
                print(
                    f"  {name}: {config.get('distance_function', 'unknown')} on {scenario} {'[ReID]' if reid else ''}"
                )
            sys.exit(0)
        elif arg == "--config":
            if len(sys.argv) < 3:
                print("Error: --config requires a config name")
                sys.exit(1)
            configs_to_generate = [sys.argv[2]]
        elif arg in FIXTURE_CONFIGS:
            configs_to_generate = [arg]
        else:
            # Legacy: treat as scenario name for backwards compatibility
            # Check if it matches any config that uses this scenario
            for name, (_, scenario) in FIXTURE_CONFIGS.items():
                if scenario == arg:
                    configs_to_generate.append(name)
                    break
            if not configs_to_generate:
                print(f"Error: Unknown config or scenario '{arg}'")
                print(f"Available configs: {list(FIXTURE_CONFIGS.keys())}")
                sys.exit(1)
    else:
        # Default: generate original fixtures for backwards compatibility
        configs_to_generate = ["small", "medium"]

    for config_name in configs_to_generate:
        print(f"Generating fixture for: {config_name}")
        fixture = generate_fixture(config_name)
        output_path = save_fixture(fixture, config_name)
        print(f"  Saved to: {output_path}")
        print(f"  Steps: {len(fixture['steps'])}")

        # Show ReID info if applicable
        _, scenario = FIXTURE_CONFIGS[config_name]
        config = fixture["tracker_config"]
        if "reid_distance_function" in config:
            print(
                f"  ReID enabled: threshold={config.get('reid_distance_threshold')}, max={config.get('reid_hit_counter_max')}"
            )

    print("\nDone! Copy fixtures to test directories:")
    print("  cp fixtures/*.json ../../tests/data/fixtures/")
    print("  cp fixtures/*.json ../../../norfair-go/tests/data/fixtures/")


if __name__ == "__main__":
    main()
