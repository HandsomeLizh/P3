#!/usr/bin/env python3
"""Diagnose fixed body-frame conventions in one completed ROMA run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def rotation_matrix_to_rpy_deg(values: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-values[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(values[..., 2, 1], values[..., 2, 2])
    yaw = np.arctan2(values[..., 1, 0], values[..., 0, 0])
    return np.rad2deg(np.stack((roll, pitch, yaw), axis=-1))


def wrap_deg(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def rotation_vector_deg(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1.0e-10:
        return np.zeros(3)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return np.rad2deg(angle * axis)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import evaluate_all_metrics as evaluator

    output_dir = args.output_dir.resolve()
    estimate = evaluator.load_tum(output_dir / "trajectory_map.tum")
    reference = evaluator.load_tum(output_dir / "pq_reference.tum")
    summary = evaluator._elapsed_time_basis(output_dir / "input_processing_summary.json")
    scale = float(summary["wall_time_per_sensor_time_scale"])

    start_allowed = estimate.timestamps[0] + 30.0
    candidates = np.flatnonzero(
        (estimate.timestamps >= start_allowed)
        & (estimate.timestamps >= reference.timestamps[0])
        & (estimate.timestamps <= reference.timestamps[-1])
    )
    accepted, ref_positions, _, _ = evaluator.interpolate_reference(
        reference, estimate.timestamps[candidates], max_bracket_sec=0.5
    )
    indices = candidates[accepted]
    timestamps = estimate.timestamps[indices]
    est_positions = estimate.positions[indices]
    est_center = np.mean(est_positions, axis=0)
    ref_center = np.mean(ref_positions, axis=0)
    left, _, right_t = np.linalg.svd(
        (est_positions - est_center).T @ (ref_positions - ref_center)
    )
    world_alignment = right_t.T @ left.T
    if np.linalg.det(world_alignment) < 0:
        right_t[-1] *= -1
        world_alignment = right_t.T @ left.T

    duration = float((timestamps[-1] - timestamps[0]) * scale)
    boundary_elapsed = 500.0 * np.arange(int(np.floor(duration / 500.0)) + 1)
    if not np.isclose(boundary_elapsed[-1], duration, atol=1.0e-6):
        boundary_elapsed = np.append(boundary_elapsed, duration)
    boundary_times = timestamps[0] + boundary_elapsed / scale
    boundary_times[-1] = timestamps[-1]
    _, _, est_rotations, _ = evaluator.interpolate_reference(
        estimate, boundary_times, max_bracket_sec=0.5
    )
    _, _, ref_rotations, _ = evaluator.interpolate_reference(
        reference, boundary_times, max_bracket_sec=0.5
    )
    est_rotations = np.einsum("ij,njk->nik", world_alignment, est_rotations)

    print("world alignment matrix:\n", world_alignment)
    print("world alignment RPY deg:", rotation_matrix_to_rpy_deg(world_alignment))
    raw_estimate_rotations = np.einsum(
        "ij,njk->nik", world_alignment.T, est_rotations
    )
    print(
        "raw estimate start/end RPY:",
        rotation_matrix_to_rpy_deg(raw_estimate_rotations)[[0, -1]],
    )

    identity = np.eye(3)
    frd_to_flu = np.diag([1.0, -1.0, -1.0])
    candidates = {
        "identity": (identity, identity),
        "reference_FRD_to_FLU": (identity, frd_to_flu),
        "estimate_FRD_to_FLU": (frd_to_flu, identity),
        "both_FRD_to_FLU": (frd_to_flu, frd_to_flu),
    }
    print(f"duration={duration:.6f}s periods={len(boundary_times)-1}")
    for name, (est_common_from_source, ref_common_from_source) in candidates.items():
        est_common = est_rotations @ est_common_from_source.T
        ref_common = ref_rotations @ ref_common_from_source.T
        est_rpy = rotation_matrix_to_rpy_deg(est_common)
        ref_rpy = rotation_matrix_to_rpy_deg(ref_common)
        legacy = wrap_deg(
            wrap_deg(est_rpy[1:] - est_rpy[:-1])
            - wrap_deg(ref_rpy[1:] - ref_rpy[:-1])
        )
        errors = np.einsum("nji,njk->nik", ref_common, est_common)
        error_changes = np.einsum("nji,njk->nik", errors[:-1], errors[1:])
        rotvec = np.stack([rotation_vector_deg(value) for value in error_changes])
        print(f"\n{name}")
        print("  estimate start/end RPY:", est_rpy[0], est_rpy[-1])
        print("  reference start/end RPY:", ref_rpy[0], ref_rpy[-1])
        print("  legacy endpoint-RPY change error:", legacy)
        print("  SO(3) error-change rotvec XYZ:", rotvec)
        print("  SO(3) error-change angle:", np.linalg.norm(rotvec, axis=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
