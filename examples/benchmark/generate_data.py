#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Generate deterministic test data for cross-language benchmarks.

Uses a simple LCG (Linear Congruential Generator) PRNG for reproducibility
across Python, Go, and Rust implementations.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Simple LCG PRNG (same parameters for all languages)
# Using parameters from Numerical Recipes
LCG_A = 1664525
LCG_C = 1013904223
LCG_M = 2**32


class SimplePRNG:
    """Simple LCG PRNG for cross-language reproducibility."""

    def __init__(self, seed: int = 42):
        self.state = seed & (LCG_M - 1)

    def next_u32(self) -> int:
        self.state = (LCG_A * self.state + LCG_C) % LCG_M
        return self.state

    def next_float(self) -> float:
        """Returns float in [0, 1)."""
        return self.next_u32() / LCG_M

    def next_range(self, min_val: float, max_val: float) -> float:
        """Returns float in [min_val, max_val)."""
        return min_val + self.next_float() * (max_val - min_val)


@dataclass
class SimulatedObject:
    """A simulated moving object."""

    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float

    def step(self, dt: float = 1.0):
        """Move object by one time step."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_bbox(self) -> tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2)."""
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )


def generate_scenario(
    seed: int,
    num_objects: int,
    num_frames: int,
    detection_prob: float = 0.9,
    noise_std: float = 2.0,
) -> dict:
    """Generate a complete tracking scenario."""

    rng = SimplePRNG(seed)

    # Initialize objects with random positions and velocities
    objects: list[SimulatedObject] = []
    for _ in range(num_objects):
        obj = SimulatedObject(
            x=rng.next_range(100, 900),
            y=rng.next_range(100, 700),
            vx=rng.next_range(-5, 5),
            vy=rng.next_range(-5, 5),
            width=rng.next_range(30, 80),
            height=rng.next_range(30, 80),
        )
        objects.append(obj)

    # Generate frames
    frames = []
    for frame_idx in range(num_frames):
        detections = []

        for obj_idx, obj in enumerate(objects):
            # Simulate detection probability
            if rng.next_float() < detection_prob:
                # Add noise to bounding box
                x1, y1, x2, y2 = obj.get_bbox()
                noise_x1 = rng.next_range(-noise_std, noise_std)
                noise_y1 = rng.next_range(-noise_std, noise_std)
                noise_x2 = rng.next_range(-noise_std, noise_std)
                noise_y2 = rng.next_range(-noise_std, noise_std)

                detections.append(
                    {
                        "bbox": [
                            x1 + noise_x1,
                            y1 + noise_y1,
                            x2 + noise_x2,
                            y2 + noise_y2,
                        ],
                        "ground_truth_id": obj_idx,
                    }
                )

            # Move object
            obj.step()

        frames.append(
            {
                "frame_id": frame_idx,
                "detections": detections,
            }
        )

    return {
        "seed": seed,
        "num_objects": num_objects,
        "num_frames": num_frames,
        "detection_prob": detection_prob,
        "noise_std": noise_std,
        "frames": frames,
    }


def generate_occlusion_scenario(
    seed: int,
    num_objects: int = 5,
    num_frames: int = 50,
    occlusion_duration: int = 5,
    noise_std: float = 2.0,
) -> dict:
    """Generate scenario with deliberate occlusions for ReID testing.

    Objects disappear for `occlusion_duration` frames and then reappear.
    This tests the reid_hit_counter countdown and merge functionality.
    """
    rng = SimplePRNG(seed)

    # Initialize objects with random positions and slower velocities
    objects: list[SimulatedObject] = []
    for _ in range(num_objects):
        obj = SimulatedObject(
            x=rng.next_range(100, 900),
            y=rng.next_range(100, 700),
            vx=rng.next_range(-2, 2),  # Slower to ensure reappearance is nearby
            vy=rng.next_range(-2, 2),
            width=rng.next_range(40, 70),
            height=rng.next_range(40, 70),
        )
        objects.append(obj)

    # Define occlusion periods for each object
    # Object i is occluded from frame occlusion_start[i] to occlusion_start[i] + occlusion_duration
    occlusion_start = []
    for i in range(num_objects):
        # Stagger occlusions to test different phases
        start = 10 + i * 8  # Objects occlude at different times
        occlusion_start.append(start)

    # Generate frames
    frames = []
    for frame_idx in range(num_frames):
        detections = []

        for obj_idx, obj in enumerate(objects):
            # Check if object is in occlusion period
            occ_start = occlusion_start[obj_idx]
            occ_end = occ_start + occlusion_duration

            is_occluded = occ_start <= frame_idx < occ_end

            if not is_occluded:
                # Object is visible - add detection
                x1, y1, x2, y2 = obj.get_bbox()
                noise_x1 = rng.next_range(-noise_std, noise_std)
                noise_y1 = rng.next_range(-noise_std, noise_std)
                noise_x2 = rng.next_range(-noise_std, noise_std)
                noise_y2 = rng.next_range(-noise_std, noise_std)

                detections.append(
                    {
                        "bbox": [
                            x1 + noise_x1,
                            y1 + noise_y1,
                            x2 + noise_x2,
                            y2 + noise_y2,
                        ],
                        "ground_truth_id": obj_idx,
                    }
                )

            # Move object (even during occlusion)
            obj.step()

        frames.append(
            {
                "frame_id": frame_idx,
                "detections": detections,
            }
        )

    return {
        "seed": seed,
        "num_objects": num_objects,
        "num_frames": num_frames,
        "occlusion_duration": occlusion_duration,
        "noise_std": noise_std,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark test data")
    parser.add_argument("--force", action="store_true", help="Regenerate existing data files")
    args = parser.parse_args()

    force = args.force

    # Generate test scenarios of increasing size
    scenarios = [
        # (name, num_objects, num_frames)
        ("small", 5, 100),
        ("medium", 20, 500),
        ("large", 50, 1000),
        ("stress", 100, 2000),
    ]

    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    for name, num_objects, num_frames in scenarios:
        output_path = os.path.join(output_dir, f"{name}.json")

        if force or not Path(output_path).exists():
            print(f"Generating {name} scenario ({num_objects} objects, {num_frames} frames)...")
        else:
            print(
                f"Skipping {name} scenario ({num_objects} objects, {num_frames} frames)... Already exists!"
            )
            continue

        scenario = generate_scenario(
            seed=42,  # Same seed for reproducibility
            num_objects=num_objects,
            num_frames=num_frames,
        )

        with open(output_path, "w") as f:
            json.dump(scenario, f)

        # Calculate stats
        total_detections = sum(len(frame["detections"]) for frame in scenario["frames"])
        print(f"  -> {output_path} ({total_detections} total detections)")

    # Generate occlusion scenario for ReID testing
    occlusion_path = os.path.join(output_dir, "occlusion.json")
    if force or not Path(occlusion_path).exists():
        print("Generating occlusion scenario (5 objects, 50 frames, for ReID testing)...")
        scenario = generate_occlusion_scenario(
            seed=42,
            num_objects=5,
            num_frames=50,
            occlusion_duration=5,
        )
        with open(occlusion_path, "w") as f:
            json.dump(scenario, f)
        total_detections = sum(len(frame["detections"]) for frame in scenario["frames"])
        print(f"  -> {occlusion_path} ({total_detections} total detections)")
    else:
        print("Skipping occlusion scenario... Already exists!")

    print("\nDone! Data files generated in:", output_dir)


if __name__ == "__main__":
    main()
