# =============================================================================
# Fixture-based tests for norfair_rs Python bindings.
#
# These tests compare norfair_rs output against Python reference fixtures
# to ensure cross-implementation consistency with Rust and Go.
#
# IMPORTANT: These tests run with BOTH norfair (Python) and norfair_rs (Rust)
# to verify that both implementations produce identical results.
#
# Run with: pytest python/tests/test_fixtures.py -v
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pytest
from norfair_rs import Detection as RsDetection
from norfair_rs import Tracker as RsTracker
from norfair_rs import _reset_global_id_counter

# Import original norfair for comparison
try:
    from norfair import Detection as PyDetection
    from norfair import Tracker as PyTracker

    HAS_NORFAIR = True
except ImportError:
    HAS_NORFAIR = False


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

    with open(path) as f:
        return json.load(f)


def create_rs_tracker(config: dict) -> RsTracker:
    """Create a norfair_rs tracker from fixture config."""
    kwargs = {
        "distance_function": config["distance_function"],
        "distance_threshold": config["distance_threshold"],
        "hit_counter_max": config["hit_counter_max"],
        "initialization_delay": config["initialization_delay"],
    }

    # Add ReID config if present
    if config.get("reid_distance_threshold") is not None:
        # For norfair_rs, use the same distance function for ReID
        kwargs["reid_distance_function"] = config["distance_function"]
        kwargs["reid_distance_threshold"] = config["reid_distance_threshold"]
    if config.get("reid_hit_counter_max") is not None:
        kwargs["reid_hit_counter_max"] = config["reid_hit_counter_max"]

    return RsTracker(**kwargs)


def create_py_tracker(config: dict) -> "PyTracker":
    """Create a norfair (Python) tracker from fixture config."""
    kwargs = {
        "distance_function": config["distance_function"],
        "distance_threshold": config["distance_threshold"],
        "hit_counter_max": config["hit_counter_max"],
        "initialization_delay": config["initialization_delay"],
    }

    # Add ReID config if present
    if config.get("reid_distance_threshold") is not None:
        # For Python norfair, use a simple euclidean distance function for ReID

        def reid_distance(obj1, obj2):
            return np.linalg.norm(obj1.estimate - obj2.estimate)

        kwargs["reid_distance_function"] = reid_distance
        kwargs["reid_distance_threshold"] = config["reid_distance_threshold"]
    if config.get("reid_hit_counter_max") is not None:
        kwargs["reid_hit_counter_max"] = config["reid_hit_counter_max"]

    return PyTracker(**kwargs)


def compare_tracked_objects(
    step_idx: int,
    frame_id: int,
    expected: list,
    actual: list,
    label: str,
    tolerance: float = 1e-6,
    impl_name: str = "norfair_rs",
):
    """Compare expected and actual tracked objects.

    Raises AssertionError with detailed message on mismatch.
    """
    # First compare counts
    if len(expected) != len(actual):
        msg = f"[{impl_name}] FIRST DIVERGENCE at step {step_idx} (frame_id={frame_id}):\n"
        msg += f"  Expected {len(expected)} {label}\n"
        msg += f"  Actual {len(actual)} {label}\n\n"

        msg += "Expected objects:\n"
        for obj in expected:
            msg += f"  ID={obj.get('id')}, initializing_id={obj.get('initializing_id')}, "
            msg += f"estimate={obj.get('estimate')}, age={obj.get('age')}, "
            msg += f"hit_counter={obj.get('hit_counter')}, is_initializing={obj.get('is_initializing')}, "
            msg += f"reid_hit_counter={obj.get('reid_hit_counter')}\n"

        msg += "\nActual objects:\n"
        for obj in actual:
            est = (
                obj.get_estimate(absolute=True).tolist()
                if hasattr(obj, "get_estimate")
                else obj.estimate.tolist()
            )
            msg += f"  ID={obj.id}, initializing_id={getattr(obj, 'initializing_id', None)}, "
            msg += f"estimate={est}, age={obj.age}, "
            msg += f"hit_counter={obj.hit_counter}, is_initializing={obj.is_initializing}, "
            msg += f"reid_hit_counter={getattr(obj, 'reid_hit_counter', None)}\n"

        raise AssertionError(msg)

    # Compare each object
    for i, (exp, act) in enumerate(zip(expected, actual)):
        # Compare ID (both norfair and norfair_rs are 1-indexed)
        act_id = act.id
        if exp.get("id") != act_id:
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} ID mismatch: "
                f"expected {exp.get('id')}, got {act_id}"
            )

        # Compare initializing_id
        act_init_id = (
            getattr(act, "initializing_id", None)
            if getattr(act, "initializing_id", None) is not None
            else -1
        )
        if exp.get("initializing_id") != act_init_id:
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} initializing_id mismatch: "
                f"expected {exp.get('initializing_id')}, got {act_init_id}"
            )

        # Compare age
        if exp.get("age") != act.age:
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} age mismatch: "
                f"expected {exp.get('age')}, got {act.age}"
            )

        # Compare hit_counter
        if exp.get("hit_counter") != act.hit_counter:
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} hit_counter mismatch: "
                f"expected {exp.get('hit_counter')}, got {act.hit_counter}"
            )

        # Compare is_initializing
        if exp.get("is_initializing") != act.is_initializing:
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} is_initializing mismatch: "
                f"expected {exp.get('is_initializing')}, got {act.is_initializing}"
            )

        # Compare reid_hit_counter
        exp_reid = exp.get("reid_hit_counter")
        act_reid = getattr(act, "reid_hit_counter", None)
        if exp_reid != act_reid:
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} reid_hit_counter mismatch: "
                f"expected {exp_reid}, got {act_reid}"
            )

        # Compare estimates (with tolerance)
        # norfair_rs has get_estimate(), Python norfair uses .estimate directly
        # Python norfair's get_estimate(absolute=True) requires coord_transformations
        if hasattr(act, "get_estimate") and impl_name == "norfair_rs":
            act_estimate = act.get_estimate(absolute=True)
        else:
            act_estimate = act.estimate
        exp_estimate = np.array(exp.get("estimate"))

        diff = np.abs(exp_estimate - act_estimate)
        max_diff = np.max(diff)
        if max_diff > tolerance:
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"[{impl_name}] Step {step_idx} frame {frame_id}: Object {i} estimate[{idx[0]}][{idx[1]}] mismatch: "
                f"expected {exp_estimate[idx]}, got {act_estimate[idx[0], idx[1]]} (diff={max_diff})"
            )


def run_fixture_test_rs(scenario: str):
    """Run a fixture test for the given scenario using norfair_rs."""
    # Reset global ID counter to ensure consistent IDs across test runs
    _reset_global_id_counter()

    fixture = load_fixture(scenario)
    tracker = create_rs_tracker(fixture["tracker_config"])

    # Tolerance for numerical comparisons
    tolerance = 1e-6

    for step_idx, step in enumerate(fixture["steps"]):
        # Convert inputs to detections (bboxes as 2x2 matrices: [[x1,y1], [x2,y2]])
        detections = []
        for det in step["inputs"]["detections"]:
            bbox = det["bbox"]
            points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float64)
            detections.append(RsDetection(points))

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
            impl_name="norfair_rs",
        )

        # Compare all_objects (all internal objects including initializing)
        # Sort by initializing_id to ensure consistent ordering
        all_objects = sorted(
            tracker.tracked_objects,
            key=lambda obj: obj.initializing_id
            if obj.initializing_id is not None
            else float("inf"),
        )
        compare_tracked_objects(
            step_idx,
            step["frame_id"],
            step["outputs"]["all_objects"],
            all_objects,
            "all_objects",
            tolerance,
            impl_name="norfair_rs",
        )

    print(f"[norfair_rs] Fixture test '{scenario}' passed: {len(fixture['steps'])} steps verified")


def run_fixture_test_py(scenario: str):
    """Run a fixture test for the given scenario using original norfair (Python)."""
    if not HAS_NORFAIR:
        pytest.skip("norfair not installed")

    fixture = load_fixture(scenario)
    tracker = create_py_tracker(fixture["tracker_config"])

    # Tolerance for numerical comparisons
    tolerance = 1e-6

    for step_idx, step in enumerate(fixture["steps"]):
        # Convert inputs to detections (bboxes as 2x2 matrices: [[x1,y1], [x2,y2]])
        detections = []
        for det in step["inputs"]["detections"]:
            bbox = det["bbox"]
            points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float64)
            detections.append(PyDetection(points))

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
            impl_name="norfair",
        )

        # Compare all_objects (all internal objects including initializing)
        # Sort by initializing_id to ensure consistent ordering
        all_objects = sorted(
            tracker.tracked_objects,
            key=lambda obj: obj.initializing_id
            if obj.initializing_id is not None
            else float("inf"),
        )
        compare_tracked_objects(
            step_idx,
            step["frame_id"],
            step["outputs"]["all_objects"],
            all_objects,
            "all_objects",
            tolerance,
            impl_name="norfair",
        )

    print(f"[norfair] Fixture test '{scenario}' passed: {len(fixture['steps'])} steps verified")


# ============================================================================
# Test Cases - norfair_rs (Rust)
# ============================================================================


def test_fixture_small_rs():
    """Test small fixture with norfair_rs."""
    run_fixture_test_rs("small")


def test_fixture_medium_rs():
    """Test medium fixture with norfair_rs."""
    run_fixture_test_rs("medium")


def test_fixture_euclidean_small_rs():
    """Test euclidean_small fixture with norfair_rs."""
    run_fixture_test_rs("euclidean_small")


def test_fixture_fast_init_rs():
    """Test fast_init fixture with norfair_rs."""
    run_fixture_test_rs("fast_init")


def test_fixture_iou_occlusion_rs():
    """Test iou_occlusion fixture with norfair_rs."""
    run_fixture_test_rs("iou_occlusion")


def test_fixture_reid_euclidean_rs():
    """Test reid_euclidean fixture with norfair_rs."""
    run_fixture_test_rs("reid_euclidean")


# ============================================================================
# Test Cases - norfair (Python) - for comparison
# ============================================================================


@pytest.mark.skipif(not HAS_NORFAIR, reason="norfair not installed")
def test_fixture_small_py():
    """Test small fixture with original norfair (Python)."""
    run_fixture_test_py("small")


@pytest.mark.skipif(not HAS_NORFAIR, reason="norfair not installed")
def test_fixture_medium_py():
    """Test medium fixture with original norfair (Python)."""
    run_fixture_test_py("medium")


@pytest.mark.skipif(not HAS_NORFAIR, reason="norfair not installed")
def test_fixture_euclidean_small_py():
    """Test euclidean_small fixture with original norfair (Python)."""
    run_fixture_test_py("euclidean_small")


@pytest.mark.skipif(not HAS_NORFAIR, reason="norfair not installed")
def test_fixture_fast_init_py():
    """Test fast_init fixture with original norfair (Python)."""
    run_fixture_test_py("fast_init")


@pytest.mark.skipif(not HAS_NORFAIR, reason="norfair not installed")
def test_fixture_iou_occlusion_py():
    """Test iou_occlusion fixture with original norfair (Python)."""
    run_fixture_test_py("iou_occlusion")


@pytest.mark.skipif(not HAS_NORFAIR, reason="norfair not installed")
def test_fixture_reid_euclidean_py():
    """Test reid_euclidean fixture with original norfair (Python)."""
    run_fixture_test_py("reid_euclidean")
