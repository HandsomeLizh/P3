#!/usr/bin/env python3
"""Evaluate a RoMa TUM trajectory against batch_organized FastLIO poses.

FastLIO is used only as an offline reference.  The reported acceptance-style
position metrics use a fixed-scale SE(3) alignment; no Sim(3) scale correction
is applied.  The script also reports a scale diagnostic so stereo scale errors
are visible without changing the official metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t1_dataset_sim"))

from t1_dataset_sim.dataset_formats import load_organized_stereo_records


def _load_tum(path: Path) -> Tuple[np.ndarray, np.ndarray, Rotation]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 TUM fields, "
                    f"got {len(fields)}"
                )
            rows.append([float(value) for value in fields])
    if not rows:
        raise ValueError(f"TUM trajectory has no poses: {path}")

    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(values[:, 0])
    values = values[order]
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"TUM timestamps must be strictly increasing: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"TUM trajectory contains non-finite values: {path}")
    quaternion_norm = np.linalg.norm(values[:, 4:8], axis=1)
    if np.any(quaternion_norm < 1e-8):
        raise ValueError(f"TUM trajectory contains a zero quaternion: {path}")
    quaternions = values[:, 4:8] / quaternion_norm[:, None]
    return values[:, 0], values[:, 1:4], Rotation.from_quat(quaternions)


def _load_left_image_times(sequence_dir: Path) -> np.ndarray:
    path = sequence_dir / "images" / "timestamps.tsv"
    timestamps = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("side") == "left":
                timestamps.append(int(row["timestamp_ns"]) * 1e-9)
    if not timestamps:
        raise ValueError(f"No left-image timestamps found in {path}")
    result = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"Left-image timestamps are not increasing: {path}")
    return result


def _load_reference_camera_poses(
    sequence_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Rotation, Dict[str, object]]:
    trajectory_path = sequence_dir / "lidar_fastlio" / "trajectory.npz"
    with np.load(str(trajectory_path)) as archive:
        times = np.asarray(archive["header_time_ns"], dtype=np.int64) * 1e-9
        body_positions = np.asarray(archive["position"], dtype=np.float64)
        body_quaternions = np.asarray(
            archive["orientation"], dtype=np.float64
        )

    if len(times) < 2 or body_positions.shape != (len(times), 3):
        raise ValueError(f"Invalid FastLIO trajectory: {trajectory_path}")
    if body_quaternions.shape != (len(times), 4):
        raise ValueError(f"Invalid FastLIO orientations: {trajectory_path}")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"FastLIO timestamps are not increasing: {trajectory_path}")

    extrinsics_path = sequence_dir / "calibration" / "extrinsics.json"
    with extrinsics_path.open("r", encoding="utf-8") as stream:
        extrinsics = json.load(stream)
    transform_camera_lidar = np.asarray(
        extrinsics["LiDAR_to_Cam_L"]["RT"], dtype=np.float64
    )
    if transform_camera_lidar.shape != (4, 4):
        raise ValueError(
            f"LiDAR_to_Cam_L.RT must be 4x4: {extrinsics_path}"
        )

    rotation_camera_lidar = transform_camera_lidar[:3, :3]
    translation_camera_lidar = transform_camera_lidar[:3, 3]
    orthogonality_error = float(
        np.linalg.norm(
            rotation_camera_lidar.T @ rotation_camera_lidar - np.eye(3)
        )
    )
    if orthogonality_error > 1e-3 or np.linalg.det(rotation_camera_lidar) < 0.0:
        raise ValueError(
            f"LiDAR_to_Cam_L rotation is invalid: {extrinsics_path}"
        )

    # Metadata states that FastLIO poses transform body points into
    # camera_init.  The calibration maps LiDAR points into Cam_L, so this
    # assumes the FastLIO body frame is the calibrated LiDAR frame:
    # T_map_camera = T_map_body @ inverse(T_camera_lidar).
    rotation_lidar_camera = rotation_camera_lidar.T
    translation_lidar_camera = (
        -rotation_lidar_camera @ translation_camera_lidar
    )
    body_rotations = Rotation.from_quat(body_quaternions)
    camera_positions = body_positions + body_rotations.apply(
        np.repeat(translation_lidar_camera[None, :], len(times), axis=0)
    )
    camera_rotations = body_rotations * Rotation.from_matrix(
        rotation_lidar_camera
    )
    metadata = {
        "trajectory_path": str(trajectory_path),
        "extrinsics_path": str(extrinsics_path),
        "fastlio_pose_frame": "body",
        "reference_pose_frame": "Cam_L",
        "extrinsic_assumption": (
            "FastLIO body frame equals the LiDAR frame used by "
            "LiDAR_to_Cam_L; T_map_cam=T_map_body*inverse(T_cam_lidar)"
        ),
        "extrinsic_rotation_orthogonality_error": orthogonality_error,
    }
    return times, camera_positions, camera_rotations, metadata


def _interpolate_reference(
    reference_times: np.ndarray,
    reference_positions: np.ndarray,
    reference_rotations: Rotation,
    query_times: np.ndarray,
) -> Tuple[np.ndarray, Rotation]:
    relative_reference_times = reference_times - reference_times[0]
    relative_query_times = query_times - reference_times[0]
    positions = np.column_stack(
        [
            np.interp(relative_query_times, relative_reference_times, axis)
            for axis in reference_positions.T
        ]
    )
    rotations = Slerp(
        relative_reference_times, reference_rotations
    )(relative_query_times)
    return positions, rotations


def _rigid_alignment(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return R, t and diagnostic Sim(3) scale for target ~= R*source+t."""
    source_center = np.mean(source_positions, axis=0)
    target_center = np.mean(target_positions, axis=0)
    source_zero = source_positions - source_center
    target_zero = target_positions - target_center
    covariance = source_zero.T @ target_zero
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(right_transpose.T @ left.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = right_transpose.T @ correction @ left.T
    translation = target_center - rotation @ source_center
    source_energy = float(np.sum(source_zero * source_zero))
    sim3_scale = (
        float(np.sum(singular_values * np.diag(correction))) / source_energy
        if source_energy > 1e-15
        else float("nan")
    )
    return rotation, translation, sim3_scale


def _statistics(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(array)),
        "rmse": float(math.sqrt(np.mean(array * array))),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _reference_interval_path_length(
    reference_times: np.ndarray,
    reference_positions: np.ndarray,
    reference_rotations: Rotation,
    start_time: float,
    end_time: float,
) -> float:
    internal = (reference_times > start_time) & (reference_times < end_time)
    boundary_positions, _ = _interpolate_reference(
        reference_times,
        reference_positions,
        reference_rotations,
        np.asarray([start_time, end_time], dtype=np.float64),
    )
    positions = np.vstack(
        (boundary_positions[:1], reference_positions[internal], boundary_positions[1:])
    )
    return _path_length(positions)


def _relative_errors(
    positions: np.ndarray,
    rotations: Rotation,
    reference_positions: np.ndarray,
    reference_rotations: Rotation,
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    translation_errors = []
    rotation_errors = []
    rotation_matrices = rotations.as_matrix()
    reference_matrices = reference_rotations.as_matrix()
    for first, second in np.asarray(pairs, dtype=np.int64):
        estimate_relative_rotation = (
            rotation_matrices[first].T @ rotation_matrices[second]
        )
        estimate_relative_translation = (
            rotation_matrices[first].T
            @ (positions[second] - positions[first])
        )
        reference_relative_rotation = (
            reference_matrices[first].T @ reference_matrices[second]
        )
        reference_relative_translation = (
            reference_matrices[first].T
            @ (reference_positions[second] - reference_positions[first])
        )
        error_rotation = (
            reference_relative_rotation.T @ estimate_relative_rotation
        )
        error_translation = reference_relative_rotation.T @ (
            estimate_relative_translation - reference_relative_translation
        )
        translation_errors.append(float(np.linalg.norm(error_translation)))
        rotation_errors.append(
            float(np.degrees(Rotation.from_matrix(error_rotation).magnitude()))
        )
    return (
        np.asarray(translation_errors, dtype=np.float64),
        np.asarray(rotation_errors, dtype=np.float64),
    )


def _one_second_pairs(times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pairs = []
    actual_deltas = []
    for first, start_time in enumerate(times[:-1]):
        target = start_time + 1.0
        insertion = int(np.searchsorted(times, target))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if first < index < len(times)
        ]
        if not candidates:
            continue
        second = min(candidates, key=lambda index: abs(times[index] - target))
        if abs((times[second] - start_time) - 1.0) > 0.35:
            continue
        pairs.append((first, second))
        actual_deltas.append(float(times[second] - start_time))
    return (
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray(actual_deltas, dtype=np.float64),
    )


def _write_tum(
    path: Path,
    times: np.ndarray,
    positions: np.ndarray,
    rotations: Rotation,
) -> None:
    quaternions = rotations.as_quat()
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, position, quaternion in zip(
            times, positions, quaternions
        ):
            stream.write(
                f"{timestamp:.9f} "
                f"{position[0]:.9f} {position[1]:.9f} {position[2]:.9f} "
                f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
                f"{quaternion[2]:.9f} {quaternion[3]:.9f}\n"
            )


def evaluate(
    estimate_path: Path,
    sequence_dir: Path,
    output_path: Path,
    reference_start_offset_s: float = None,
    reference_end_offset_s: float = None,
    max_stereo_skew_ms: float = 20.0,
    max_frames: int = 0,
) -> Dict[str, object]:
    estimate_times, estimate_positions, estimate_rotations = _load_tum(
        estimate_path
    )
    image_times = _load_left_image_times(sequence_dir)
    selected_records, stereo_pairing = load_organized_stereo_records(
        sequence_dir,
        max_abs_skew_ns=int(round(float(max_stereo_skew_ms) * 1_000_000.0)),
        max_frames=int(max_frames),
    )
    selected_image_times = np.asarray(
        [record.left_timestamp_ns * 1e-9 for record in selected_records],
        dtype=np.float64,
    )
    if not len(selected_image_times):
        raise ValueError(
            "No stereo pairs passed the configured selection for evaluation"
        )
    (
        reference_times,
        reference_positions,
        reference_rotations,
        reference_metadata,
    ) = _load_reference_camera_poses(sequence_dir)

    evaluation_start = float(reference_times[0])
    evaluation_end = float(reference_times[-1])
    if reference_start_offset_s is not None:
        evaluation_start = max(
            evaluation_start,
            float(reference_times[0]) + float(reference_start_offset_s),
        )
    if reference_end_offset_s is not None:
        evaluation_end = min(
            evaluation_end,
            float(reference_times[0]) + float(reference_end_offset_s),
        )
    if evaluation_end <= evaluation_start:
        raise ValueError(
            "Reference evaluation interval is empty: "
            f"{evaluation_start} -> {evaluation_end}"
        )

    reference_range_mask = (
        (estimate_times >= evaluation_start)
        & (estimate_times <= evaluation_end)
    )
    dropped_outside_reference = int(np.count_nonzero(~reference_range_mask))
    estimate_times = estimate_times[reference_range_mask]
    estimate_positions = estimate_positions[reference_range_mask]
    estimate_rotations = estimate_rotations[reference_range_mask]

    compatible_image_mask = (
        (image_times >= evaluation_start) & (image_times <= evaluation_end)
    )
    compatible_image_count = int(np.count_nonzero(compatible_image_mask))
    compatible_selected_mask = (
        (selected_image_times >= evaluation_start)
        & (selected_image_times <= evaluation_end)
    )
    compatible_selected_count = int(
        np.count_nonzero(compatible_selected_mask)
    )
    selected_coverage_percent = (
        100.0 * len(estimate_times) / compatible_selected_count
        if compatible_selected_count
        else float("nan")
    )
    raw_coverage_percent = (
        100.0 * len(estimate_times) / compatible_image_count
        if compatible_image_count
        else float("nan")
    )
    stereo_eligibility_percent = (
        100.0 * stereo_pairing.eligible_rows / stereo_pairing.shared_rows
        if stereo_pairing.shared_rows
        else float("nan")
    )
    result: Dict[str, object] = {
        "status": "insufficient_poses" if len(estimate_times) < 3 else "ok",
        "metric_alignment": "SE(3), fixed scale",
        "sim3_used_for_reported_errors": False,
        "estimate_path": str(estimate_path),
        "sequence_dir": str(sequence_dir),
        "reference": reference_metadata,
        "evaluation_crop": {
            "reference_start_offset_s": float(
                evaluation_start - reference_times[0]
            ),
            "reference_end_offset_s": float(
                evaluation_end - reference_times[0]
            ),
        },
        "counts": {
            # Retained for compatibility: this is the raw left-image count,
            # not the number of pairs that passed the stereo-skew gate.
            "dataset_stereo_frames": int(len(image_times)),
            "reference_compatible_stereo_frames": compatible_image_count,
            "dataset_shared_stereo_candidates": int(
                stereo_pairing.shared_rows
            ),
            "dataset_stereo_skew_eligible_pairs": int(
                stereo_pairing.eligible_rows
            ),
            "dataset_selected_stereo_pairs": int(len(selected_records)),
            "dataset_pairs_excluded_by_stereo_skew": int(
                stereo_pairing.skipped_for_skew
            ),
            "dataset_pairs_excluded_by_max_frames": int(
                stereo_pairing.skipped_for_max_frames
            ),
            "reference_compatible_selected_stereo_pairs": (
                compatible_selected_count
            ),
            "estimate_poses_in_file": int(len(reference_range_mask)),
            "estimate_poses_evaluated": int(len(estimate_times)),
            "estimate_poses_outside_reference_range": (
                dropped_outside_reference
            ),
            "estimated_tracking_success_percent": selected_coverage_percent,
            "estimated_pose_coverage_of_selected_percent": (
                selected_coverage_percent
            ),
            "estimated_pose_coverage_of_raw_left_percent": (
                raw_coverage_percent
            ),
            "stereo_skew_pass_percent": stereo_eligibility_percent,
        },
        "dataset": {
            "image_duration_s": float(image_times[-1] - image_times[0]),
            "stereo_selection": stereo_pairing.selection_summary(
                pairing_method="frame_index",
                configured_max_abs_skew_ns=int(
                    round(float(max_stereo_skew_ms) * 1_000_000.0)
                ),
                max_frames=int(max_frames),
            ),
            "full_fastlio_duration_s": float(
                reference_times[-1] - reference_times[0]
            ),
            "full_fastlio_path_length_m": _path_length(reference_positions),
        },
    }
    if len(estimate_times) < 3:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=True)
            stream.write("\n")
        return result

    matched_reference_positions, matched_reference_rotations = (
        _interpolate_reference(
            reference_times,
            reference_positions,
            reference_rotations,
            estimate_times,
        )
    )
    alignment_rotation, alignment_translation, sim3_scale = _rigid_alignment(
        estimate_positions, matched_reference_positions
    )
    aligned_positions = (
        alignment_rotation @ estimate_positions.T
    ).T + alignment_translation
    aligned_rotations = (
        Rotation.from_matrix(alignment_rotation) * estimate_rotations
    )

    position_error = np.linalg.norm(
        aligned_positions - matched_reference_positions, axis=1
    )
    orientation_error = (
        matched_reference_rotations.inv() * aligned_rotations
    ).magnitude()
    orientation_error_deg = np.degrees(orientation_error)
    evaluated_reference_path = _reference_interval_path_length(
        reference_times,
        reference_positions,
        reference_rotations,
        float(estimate_times[0]),
        float(estimate_times[-1]),
    )
    sampled_reference_path = _path_length(matched_reference_positions)
    estimated_path = _path_length(estimate_positions)
    estimate_time_gaps = np.diff(estimate_times)
    position_stats = _statistics(position_error)
    sampled_reference_fraction = (
        100.0 * sampled_reference_path / evaluated_reference_path
        if evaluated_reference_path > 0.0
        else float("nan")
    )
    validity_notes = [
        "FastLIO is a proxy reference, not certified ground truth.",
        "Reported ATE/RPE use fixed-scale SE(3) alignment only.",
        "The Sim(3) scale is diagnostic and is not applied to errors.",
        (
            "Absolute orientation is not acceptance-valid until the "
            "FastLIO body-to-Cam_L extrinsic is independently calibrated."
        ),
        "Attitude-drift acceptance requires a sequence longer than 300 s.",
        "Positioning acceptance requires a continuous path D >= 1 km.",
    ]
    if (
        np.isfinite(sampled_reference_fraction)
        and sampled_reference_fraction < 90.0
    ):
        validity_notes.append(
            "Estimated poses sample only "
            f"{sampled_reference_fraction:.3f}% of the continuous reference "
            "path in the evaluated interval; the aggregate ATE is not a "
            "route-complete accuracy result."
        )

    consecutive_pairs = np.column_stack(
        (
            np.arange(len(estimate_times) - 1, dtype=np.int64),
            np.arange(1, len(estimate_times), dtype=np.int64),
        )
    )
    consecutive_translation, consecutive_rotation = _relative_errors(
        aligned_positions,
        aligned_rotations,
        matched_reference_positions,
        matched_reference_rotations,
        consecutive_pairs,
    )
    one_second_pairs, one_second_deltas = _one_second_pairs(estimate_times)
    one_second_translation, one_second_rotation = _relative_errors(
        aligned_positions,
        aligned_rotations,
        matched_reference_positions,
        matched_reference_rotations,
        one_second_pairs,
    )

    nearest_indices = np.searchsorted(reference_times, estimate_times)
    nearest_indices = np.clip(nearest_indices, 1, len(reference_times) - 1)
    before = nearest_indices - 1
    after = nearest_indices
    nearest_time_delta = np.minimum(
        estimate_times - reference_times[before],
        reference_times[after] - estimate_times,
    )
    bracket_span = reference_times[after] - reference_times[before]

    result.update(
        {
            "evaluated_interval": {
                "start_timestamp_s": float(estimate_times[0]),
                "end_timestamp_s": float(estimate_times[-1]),
                "duration_s": float(estimate_times[-1] - estimate_times[0]),
                "reference_path_length_m": evaluated_reference_path,
                "sampled_reference_path_length_m": sampled_reference_path,
                "sampled_reference_fraction_of_continuous_path_percent": (
                    sampled_reference_fraction
                ),
                "estimated_path_length_m": estimated_path,
                "aligned_estimated_path_length_m": _path_length(aligned_positions),
                "estimated_to_sampled_reference_path_percent": (
                    100.0 * estimated_path / sampled_reference_path
                    if sampled_reference_path > 0.0
                    else float("nan")
                ),
                "estimate_time_gap_sec": _statistics(estimate_time_gaps),
                "timestamp_nearest_delta_s": _statistics(nearest_time_delta),
                "timestamp_bracket_span_s": _statistics(bracket_span),
            },
            "alignment": {
                "rotation_matrix": alignment_rotation.tolist(),
                "translation_m": alignment_translation.tolist(),
                "diagnostic_best_sim3_scale": sim3_scale,
                "diagnostic_scale_error_percent": 100.0 * (sim3_scale - 1.0),
            },
            "ate_translation_m": position_stats,
            "ate_orientation_deg": _statistics(orientation_error_deg),
            "position_rmse_over_distance_percent": (
                100.0 * position_stats["rmse"] / evaluated_reference_path
                if evaluated_reference_path > 0.0
                else float("nan")
            ),
            "endpoint": {
                "translation_error_m": float(position_error[-1]),
                "orientation_error_deg": float(orientation_error_deg[-1]),
                "orientation_error_over_duration_deg_per_s": (
                    float(orientation_error_deg[-1])
                    / float(estimate_times[-1] - estimate_times[0])
                    if estimate_times[-1] > estimate_times[0]
                    else float("nan")
                ),
            },
            "rpe_consecutive": {
                "translation_m": _statistics(consecutive_translation),
                "rotation_deg": _statistics(consecutive_rotation),
            },
            "rpe_approximately_1s": {
                "actual_delta_s": _statistics(one_second_deltas),
                "translation_m": _statistics(one_second_translation),
                "rotation_deg": _statistics(one_second_rotation),
            },
            "validity_notes": validity_notes,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_stem = output_path.stem
    report_suffix = (
        report_stem[len("accuracy_report") :]
        if report_stem.startswith("accuracy_report")
        else "_" + report_stem
    )
    aligned_path = output_path.with_name(
        "trajectory_aligned_to_fastlio" + report_suffix + ".tum"
    )
    _write_tum(
        aligned_path, estimate_times, aligned_positions, aligned_rotations
    )
    result["aligned_trajectory_path"] = str(aligned_path)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=True)
        stream.write("\n")
    return result


def _print_summary(result: Dict[str, object]) -> None:
    counts = result["counts"]
    print(f"status: {result['status']}")
    print(
        "poses: "
        f"{counts['estimate_poses_evaluated']}/"
        f"{counts['reference_compatible_selected_stereo_pairs']} "
        f"({counts['estimated_tracking_success_percent']:.2f}%)"
    )
    print(
        "stereo eligibility: "
        f"{counts['dataset_stereo_skew_eligible_pairs']}/"
        f"{counts['dataset_shared_stereo_candidates']} "
        f"({counts['stereo_skew_pass_percent']:.2f}%)"
    )
    print(
        "full reference: "
        f"{result['dataset']['full_fastlio_path_length_m']:.3f} m"
    )
    if result["status"] != "ok":
        return
    interval = result["evaluated_interval"]
    ate = result["ate_translation_m"]
    print(
        f"evaluated: {interval['duration_s']:.3f} s, "
        f"D={interval['reference_path_length_m']:.3f} m"
    )
    print(
        f"ATE RMSE={ate['rmse']:.3f} m, median={ate['median']:.3f} m, "
        f"P95={ate['p95']:.3f} m, max={ate['max']:.3f} m"
    )
    print(
        "RMSE/D="
        f"{result['position_rmse_over_distance_percent']:.4f}%, "
        "diagnostic Sim(3) scale="
        f"{result['alignment']['diagnostic_best_sim3_scale']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimate",
        required=True,
        type=Path,
        help="RoMa trajectory_map.tum (or another TUM world-from-camera pose file)",
    )
    parser.add_argument(
        "--sequence-dir",
        required=True,
        type=Path,
        help="batch_organized sequence directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: next to estimate)",
    )
    parser.add_argument(
        "--reference-start-offset-s",
        type=float,
        help="Evaluate no earlier than this offset from the first reference pose",
    )
    parser.add_argument(
        "--reference-end-offset-s",
        type=float,
        help="Evaluate no later than this offset from the first reference pose",
    )
    parser.add_argument(
        "--max-stereo-skew-ms",
        type=float,
        default=20.0,
        help="Stereo eligibility gate used by the evaluated run (default: 20)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Selected eligible-frame limit used by the run; 0 means all",
    )
    arguments = parser.parse_args()
    if arguments.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    output = arguments.output or arguments.estimate.with_name(
        "accuracy_report.json"
    )
    report = evaluate(
        arguments.estimate.resolve(),
        arguments.sequence_dir.resolve(),
        output.resolve(),
        reference_start_offset_s=arguments.reference_start_offset_s,
        reference_end_offset_s=arguments.reference_end_offset_s,
        max_stereo_skew_ms=arguments.max_stereo_skew_ms,
        max_frames=arguments.max_frames,
    )
    _print_summary(report)
    print(f"report: {output.resolve()}")


if __name__ == "__main__":
    main()
