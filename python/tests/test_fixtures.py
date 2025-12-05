# =============================================================================
# Fixture-based tests for norfair_rs Python bindings.
#
# These tests compare norfair_rs output against Python reference fixtures
# to ensure cross-implementation consistency with Rust and Go.
#
# Run with: pytest tests/python/test_fixtures.py -v
# =============================================================================

import json
import numpy as np
import pytest
from pathlib import Path

from norfair_rs import Detection, Tracker


def find_testdata_dir() -> Path:
    """Find the tests/data/fixtures directory."""
    candidates = [
        Path("tests/data/fixtures"),
        Path("../tests/data/fixtures"),
        Path("../../tests/data/fixtures"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find tests/data/fixtures directory")


def load_fixture(scenario: str) -> dict:
    """Load a fixture JSON file."""
    testdata_dir = find_testdata_dir()
    path = testdata_dir / f"fixture_{scenario}.json"

    with open(path, "r") as f:
        return json.load(f)


def create_tracker(config: dict) -> Tracker:
    """Create a tracker from fixture config."""
    return Tracker(
        distance_function=config["distance_function"],
        distance_threshold=config["distance_threshold"],
        hit_counter_max=config["hit_counter_max"],
        initialization_delay=config["initialization_delay"],
    )


def compare_tracked_objects(
    step_idx: int,
    frame_id: int,
    expected: list,
    actual: list,
    label: str,
    tolerance: float = 1e-6,
):
    """Compare expected and actual tracked objects.

    Raises AssertionError with detailed message on mismatch.
    """
    # First compare counts
    if len(expected) != len(actual):
        msg = f"FIRST DIVERGENCE at step {step_idx} (frame_id={frame_id}):\n"
        msg += f"  Expected {len(expected)} {label}\n"
        msg += f"  Actual {len(actual)} {label}\n\n"

        msg += "Expected objects:\n"
        for obj in expected:
            msg += f"  ID={obj.get('id')}, initializing_id={obj.get('initializing_id')}, "
            msg += f"estimate={obj.get('estimate')}, age={obj.get('age')}, "
            msg += f"hit_counter={obj.get('hit_counter')}, is_initializing={obj.get('is_initializing')}\n"

        msg += "\nActual objects:\n"
        for obj in actual:
            msg += f"  ID={obj.id}, initializing_id={obj.initializing_id}, "
            msg += f"estimate={obj.get_estimate(absolute=True).tolist()}, age={obj.age}, "
            msg += f"hit_counter={obj.hit_counter}, is_initializing={obj.is_initializing}\n"

        raise AssertionError(msg)

    # Compare each object
    for i, (exp, act) in enumerate(zip(expected, actual)):
        # Compare ID (Python norfair starts at 1, Rust starts at 0 - adjust for offset)
        act_id = act.id + 1 if act.id is not None else None
        if exp.get("id") != act_id:
            raise AssertionError(
                f"Step {step_idx} frame {frame_id}: Object {i} ID mismatch: "
                f"expected {exp.get('id')}, got {act_id} (raw: {act.id})"
            )

        # Compare initializing_id (adjust for offset)
        act_init_id = act.initializing_id + 1 if act.initializing_id is not None else -1
        if exp.get("initializing_id") != act_init_id:
            raise AssertionError(
                f"Step {step_idx} frame {frame_id}: Object {i} initializing_id mismatch: "
                f"expected {exp.get('initializing_id')}, got {act_init_id} (raw: {act.initializing_id})"
            )

        # Compare age
        if exp.get("age") != act.age:
            raise AssertionError(
                f"Step {step_idx} frame {frame_id}: Object {i} age mismatch: "
                f"expected {exp.get('age')}, got {act.age}"
            )

        # Compare hit_counter
        if exp.get("hit_counter") != act.hit_counter:
            raise AssertionError(
                f"Step {step_idx} frame {frame_id}: Object {i} hit_counter mismatch: "
                f"expected {exp.get('hit_counter')}, got {act.hit_counter}"
            )

        # Compare is_initializing
        if exp.get("is_initializing") != act.is_initializing:
            raise AssertionError(
                f"Step {step_idx} frame {frame_id}: Object {i} is_initializing mismatch: "
                f"expected {exp.get('is_initializing')}, got {act.is_initializing}"
            )

        # Compare estimates (with tolerance)
        # Use absolute=True because fixtures store absolute coordinates
        act_estimate = act.get_estimate(absolute=True)
        exp_estimate = np.array(exp.get("estimate"))

        diff = np.abs(exp_estimate - act_estimate)
        max_diff = np.max(diff)
        if max_diff > tolerance:
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"Step {step_idx} frame {frame_id}: Object {i} estimate[{idx[0]}][{idx[1]}] mismatch: "
                f"expected {exp_estimate[idx]}, got {act_estimate[idx[0], idx[1]]} (diff={max_diff})"
            )


def run_fixture_test(scenario: str):
    """Run a fixture test for the given scenario."""
    fixture = load_fixture(scenario)
    tracker = create_tracker(fixture["tracker_config"])

    # Tolerance for numerical comparisons
    tolerance = 1e-6

    for step_idx, step in enumerate(fixture["steps"]):
        # Convert inputs to detections (bboxes as 2x2 matrices: [[x1,y1], [x2,y2]])
        detections = []
        for det in step["inputs"]["detections"]:
            bbox = det["bbox"]
            points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float64)
            detections.append(Detection(points))

        # Update tracker
        tracked_objects = tracker.update(detections)

        # Compare tracked_objects (returned by update - initialized, active objects)
        compare_tracked_objects(
            step_idx,
            step["frame_id"],
            step["outputs"]["tracked_objects"],
            tracked_objects,
            "tracked_objects",
            tolerance,
        )

        # Compare all_objects (all internal objects including initializing)
        # Sort by initializing_id to ensure consistent ordering
        all_objects = sorted(
            tracker.tracked_objects,
            key=lambda obj: obj.initializing_id if obj.initializing_id is not None else float('inf')
        )
        compare_tracked_objects(
            step_idx,
            step["frame_id"],
            step["outputs"]["all_objects"],
            all_objects,
            "all_objects",
            tolerance,
        )

    print(f"Fixture test '{scenario}' passed: {len(fixture['steps'])} steps verified")


# ============================================================================
# Test Cases
# ============================================================================

def test_fixture_small():
    """Test with small fixture (100 frames)."""
    run_fixture_test("small")


def test_fixture_medium():
    """Test with medium fixture (1000 frames)."""
    run_fixture_test("medium")
