#!/usr/bin/env python3
"""Evaluate yaw orientation stability against a reference trajectory.

The formal metric is the time-weighted mean of the absolute change in yaw
error.  A legacy static self-drift diagnostic is retained under an explicitly
non-acceptance mode so that historical data can still be inspected without
being confused with the formal estimate-versus-truth test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FORMAL_EVALUATION_TYPE = "formal_yaw_error_stability"
DIAGNOSTIC_EVALUATION_TYPE = "diagnostic_static_self_drift"


@dataclass(frozen=True)
class TumTrajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Trajectory contains an invalid quaternion")
    return value / norm


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_quaternion_xyzw(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion_xyzw(first)
    q1 = normalize_quaternion_xyzw(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(q0 + alpha * (q1 - q0))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - alpha) * angle) / sine * q0
        + math.sin(alpha * angle) / sine * q1
    )


def load_tum(path: Path, *, time_offset_sec: float = 0.0) -> TumTrajectory:
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(
                f"{path}:{line_number}: expected 8 TUM fields, got {len(fields)}"
            )
        values = np.asarray([float(field) for field in fields], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}:{line_number}: non-finite value")
        timestamps.append(float(values[0] + time_offset_sec))
        positions.append(values[1:4])
        quaternions.append(normalize_quaternion_xyzw(values[4:8]))
    if len(timestamps) < 2:
        raise ValueError(f"{path}: trajectory must contain at least two valid poses")
    time_array = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return TumTrajectory(
        timestamps=time_array,
        positions=np.asarray(positions, dtype=np.float64),
        quaternions_xyzw=np.asarray(quaternions, dtype=np.float64),
    )


def validate_rotation_matrix(values: Iterable[float], name: str) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64)
    if matrix.size != 9:
        raise ValueError(f"{name} must contain exactly 9 row-major values")
    matrix = matrix.reshape(3, 3)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{name} must have determinant +1")
    return matrix


def interpolate_reference(
    reference: TumTrajectory,
    query_times: np.ndarray,
    *,
    max_bracket_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return accepted query indices, positions, rotations and bracket spans."""
    accepted_indices: list[int] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    bracket_spans: list[float] = []
    ref_times = reference.timestamps

    for query_index, query_time in enumerate(np.asarray(query_times, dtype=np.float64)):
        right = int(np.searchsorted(ref_times, query_time, side="left"))
        if right < len(ref_times) and math.isclose(
            float(ref_times[right]), float(query_time), rel_tol=0.0, abs_tol=1e-9
        ):
            left = right
            alpha = 0.0
            span = 0.0
        elif right == 0 or right >= len(ref_times):
            continue
        else:
            left = right - 1
            span = float(ref_times[right] - ref_times[left])
            if span > max_bracket_sec:
                continue
            alpha = float((query_time - ref_times[left]) / span)

        if left == right:
            position = reference.positions[left]
            quaternion = reference.quaternions_xyzw[left]
        else:
            position = (
                (1.0 - alpha) * reference.positions[left]
                + alpha * reference.positions[right]
            )
            quaternion = slerp_quaternion_xyzw(
                reference.quaternions_xyzw[left],
                reference.quaternions_xyzw[right],
                alpha,
            )
        accepted_indices.append(query_index)
        positions.append(position)
        rotations.append(quaternion_xyzw_to_matrix(quaternion))
        bracket_spans.append(span)

    if not accepted_indices:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray(accepted_indices, dtype=np.int64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(rotations, dtype=np.float64),
        np.asarray(bracket_spans, dtype=np.float64),
    )


def yaw_deg(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    return np.rad2deg(np.arctan2(values[..., 1, 0], values[..., 0, 0]))


def wrap_deg(angles: np.ndarray | float) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    cosines = np.clip((np.trace(values, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosines))


def series_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def reference_path_length(
    reference: TumTrajectory, start_time: float, end_time: float
) -> float:
    boundary_indices, boundary_positions, _, _ = interpolate_reference(
        reference,
        np.asarray([start_time, end_time], dtype=np.float64),
        max_bracket_sec=float("inf"),
    )
    if len(boundary_indices) != 2:
        raise ValueError("Valid interval extends beyond the reference trajectory")
    internal = reference.positions[
        (reference.timestamps > start_time) & (reference.timestamps < end_time)
    ]
    points = np.vstack([boundary_positions[0], internal, boundary_positions[1]])
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def evaluate_formal(
    estimate: TumTrajectory,
    reference: TumTrajectory,
    *,
    estimate_path: str,
    reference_path: str,
    estimate_frame: str,
    reference_frame: str,
    world_frame: str,
    estimate_body_to_common_body: np.ndarray,
    reference_body_to_common_body: np.ndarray,
    warmup_sec: float,
    max_reference_bracket_sec: float,
    threshold_deg_per_sec: float,
    minimum_duration_sec: float,
    minimum_distance_m: float,
    minimum_samples: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    if threshold_deg_per_sec <= 0.0:
        raise ValueError("threshold_deg_per_sec must be positive")
    if max_reference_bracket_sec <= 0.0:
        raise ValueError("max_reference_bracket_sec must be positive")

    first_allowed_time = float(estimate.timestamps[0] + warmup_sec)
    candidate_mask = (
        (estimate.timestamps >= first_allowed_time)
        & (estimate.timestamps >= reference.timestamps[0])
        & (estimate.timestamps <= reference.timestamps[-1])
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) < 2:
        raise ValueError("Warm-up and time overlap leave fewer than two estimate poses")

    candidate_times = estimate.timestamps[candidate_indices]
    relative_indices, reference_positions, reference_rotations, bracket_spans = (
        interpolate_reference(
            reference,
            candidate_times,
            max_bracket_sec=max_reference_bracket_sec,
        )
    )
    valid_indices = candidate_indices[relative_indices]
    if len(valid_indices) < 2:
        raise ValueError("Reference interpolation leaves fewer than two valid pose pairs")

    times = estimate.timestamps[valid_indices]
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("Paired estimate timestamps must be strictly increasing")

    estimate_rotations = np.stack(
        [quaternion_xyzw_to_matrix(q) for q in estimate.quaternions_xyzw[valid_indices]]
    )
    # A TUM orientation is world_R_source.  The supplied fixed rotation is
    # common_R_source, hence world_R_common = world_R_source * source_R_common.
    estimate_common = estimate_rotations @ estimate_body_to_common_body.T
    reference_common = reference_rotations @ reference_body_to_common_body.T

    estimate_yaw = yaw_deg(estimate_common)
    reference_yaw = yaw_deg(reference_common)
    yaw_error = wrap_deg(estimate_yaw - reference_yaw)
    yaw_error_change = np.abs(wrap_deg(np.diff(yaw_error)))
    instantaneous_rate = yaw_error_change / dt
    duration = float(times[-1] - times[0])
    cumulative_change = float(np.sum(yaw_error_change))
    mean_rate = cumulative_change / duration

    orientation_error_rotations = np.einsum(
        "nij,njk->nik", np.transpose(reference_common, (0, 2, 1)), estimate_common
    )
    orientation_error_angle = rotation_angle_deg(orientation_error_rotations)
    orientation_error_steps = np.einsum(
        "nij,njk->nik",
        np.transpose(orientation_error_rotations[:-1], (0, 2, 1)),
        orientation_error_rotations[1:],
    )
    orientation_change_angle = rotation_angle_deg(orientation_error_steps)
    three_dimensional_change_rate = float(np.sum(orientation_change_angle) / duration)
    distance = reference_path_length(reference, float(times[0]), float(times[-1]))

    criteria = {
        "enough_samples": bool(len(times) >= minimum_samples),
        "enough_duration": bool(duration >= minimum_duration_sec),
        "enough_distance": bool(distance >= minimum_distance_m),
        "within_yaw_stability_threshold": bool(mean_rate < threshold_deg_per_sec),
    }
    acceptance_valid = bool(
        criteria["enough_samples"]
        and criteria["enough_duration"]
        and criteria["enough_distance"]
    )
    passed = bool(acceptance_valid and criteria["within_yaw_stability_threshold"])
    status = "pass" if passed else ("fail" if acceptance_valid else "insufficient_data")

    report = {
        "evaluation_type": FORMAL_EVALUATION_TYPE,
        "metric_definition": "sum(abs(wrap(delta yaw_error))) / elapsed_time",
        "acceptance_axis": "yaw",
        "angle_unit": "degree",
        "rate_unit": "degree/second",
        "estimate_path": estimate_path,
        "reference_path": reference_path,
        "frames": {
            "world_frame": world_frame,
            "estimate_source_body_frame": estimate_frame,
            "reference_source_body_frame": reference_frame,
            "common_body_frame": "body",
            "estimate_source_body_to_common_body_rotation_row_major": (
                estimate_body_to_common_body.reshape(-1).tolist()
            ),
            "reference_source_body_to_common_body_rotation_row_major": (
                reference_body_to_common_body.reshape(-1).tolist()
            ),
        },
        "samples": int(len(times)),
        "candidate_estimate_samples": int(len(candidate_indices)),
        "rejected_reference_pair_samples": int(len(candidate_indices) - len(times)),
        "start_time_sec": float(times[0]),
        "end_time_sec": float(times[-1]),
        "duration_sec": duration,
        "reference_path_length_m": distance,
        "warmup_sec": float(warmup_sec),
        "max_reference_bracket_sec": float(max_reference_bracket_sec),
        "maximum_used_reference_bracket_sec": float(np.max(bracket_spans)),
        "threshold_deg_per_sec": float(threshold_deg_per_sec),
        "minimum_duration_sec": float(minimum_duration_sec),
        "minimum_distance_m": float(minimum_distance_m),
        "minimum_samples": int(minimum_samples),
        "cumulative_absolute_yaw_error_change_deg": cumulative_change,
        "mean_yaw_error_change_rate_deg_per_sec": mean_rate,
        "yaw_error_deg": series_statistics(np.abs(yaw_error)),
        "instantaneous_yaw_error_change_rate_deg_per_sec": series_statistics(
            instantaneous_rate
        ),
        "auxiliary_three_dimensional_orientation_error_deg": series_statistics(
            orientation_error_angle
        ),
        "auxiliary_three_dimensional_error_change_rate_deg_per_sec": (
            three_dimensional_change_rate
        ),
        "criteria": criteria,
        "acceptance_valid": acceptance_valid,
        "pass": passed,
        "status": status,
    }
    samples = {
        "timestamp_sec": times,
        "reference_bracket_sec": bracket_spans,
        "estimate_yaw_deg": estimate_yaw,
        "reference_yaw_deg": reference_yaw,
        "yaw_error_deg": yaw_error,
        "delta_time_sec": np.concatenate([[np.nan], dt]),
        "absolute_yaw_error_change_deg": np.concatenate([[np.nan], yaw_error_change]),
        "yaw_error_change_rate_deg_per_sec": np.concatenate(
            [[np.nan], instantaneous_rate]
        ),
        "orientation_error_angle_deg": orientation_error_angle,
        "reference_x_m": reference_positions[:, 0],
        "reference_y_m": reference_positions[:, 1],
        "reference_z_m": reference_positions[:, 2],
    }
    return report, samples


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        raise ValueError("Orientation change is too close to 180 degrees")
    axis = np.array(
        [
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return axis * angle


def evaluate_diagnostic_static(
    trajectory: TumTrajectory,
    *,
    warmup_sec: float,
    threshold_deg_per_sec: float,
    minimum_duration_sec: float,
) -> dict:
    start_time = float(trajectory.timestamps[0] + warmup_sec)
    start_index = int(np.searchsorted(trajectory.timestamps, start_time, side="left"))
    if start_index >= len(trajectory.timestamps) - 1:
        raise ValueError("Warm-up removes all but one trajectory sample")
    timestamps = trajectory.timestamps[start_index:]
    times = timestamps - timestamps[0]
    duration = float(times[-1])
    if duration < minimum_duration_sec:
        raise ValueError(
            f"Static interval is only {duration:.3f}s; require at least "
            f"{minimum_duration_sec:.3f}s"
        )
    rotations = np.stack(
        [quaternion_xyzw_to_matrix(q) for q in trajectory.quaternions_xyzw[start_index:]]
    )
    reference_rotation = rotations[0]
    rotation_vectors_deg = np.rad2deg(
        np.stack([rotation_vector(reference_rotation.T @ value) for value in rotations])
    )
    design = np.column_stack([times, np.ones_like(times)])
    slopes = np.linalg.lstsq(design, rotation_vectors_deg, rcond=None)[0][0]
    drift_angles = np.linalg.norm(rotation_vectors_deg, axis=1)
    fitted_rate = float(np.linalg.norm(slopes))
    endpoint_rate = float(drift_angles[-1] / duration)
    translation_drift = np.linalg.norm(
        trajectory.positions[start_index:] - trajectory.positions[start_index], axis=1
    )
    return {
        "evaluation_type": DIAGNOSTIC_EVALUATION_TYPE,
        "warning": (
            "Diagnostic only: no reference truth is used, so this result cannot "
            "be used for formal orientation-stability acceptance."
        ),
        "acceptance_valid": False,
        "pass": None,
        "status": "diagnostic_only",
        "samples": int(len(times)),
        "warmup_sec": float(warmup_sec),
        "duration_sec": duration,
        "historical_threshold_deg_per_sec": float(threshold_deg_per_sec),
        "fitted_axis_rates_deg_per_sec": slopes.tolist(),
        "fitted_self_drift_rate_deg_per_sec": fitted_rate,
        "endpoint_self_drift_deg": float(drift_angles[-1]),
        "endpoint_self_drift_rate_deg_per_sec": endpoint_rate,
        "maximum_orientation_change_from_start_deg": float(np.max(drift_angles)),
        "maximum_translation_change_from_start_m": float(np.max(translation_drift)),
    }


def aggregate_formal_reports(
    paths: list[Path], *, required_runs: int, output_threshold: float | None = None
) -> dict:
    if required_runs <= 0:
        raise ValueError("required_runs must be positive")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, report in zip(paths, reports):
        if report.get("evaluation_type") != FORMAL_EVALUATION_TYPE:
            raise ValueError(f"{path}: not a formal yaw-stability report")
    thresholds = [float(report["threshold_deg_per_sec"]) for report in reports]
    threshold = float(output_threshold) if output_threshold is not None else thresholds[0]
    if any(not math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12) for value in thresholds):
        raise ValueError("Input reports do not use one common acceptance threshold")
    rates = [float(report["mean_yaw_error_change_rate_deg_per_sec"]) for report in reports]
    enough_runs = len(reports) >= required_runs
    all_valid = bool(reports) and all(bool(report.get("acceptance_valid")) for report in reports)
    maximum_rate = float(max(rates)) if rates else None
    passed = bool(
        enough_runs
        and all_valid
        and maximum_rate is not None
        and maximum_rate < threshold
    )
    status = "pass" if passed else ("fail" if enough_runs and all_valid else "insufficient_data")
    return {
        "evaluation_type": "formal_yaw_error_stability_aggregate",
        "aggregation_rule": "maximum of independent valid run metrics",
        "required_runs": int(required_runs),
        "provided_runs": int(len(reports)),
        "threshold_deg_per_sec": threshold,
        "run_metrics_deg_per_sec": rates,
        "maximum_run_metric_deg_per_sec": maximum_rate,
        "all_runs_acceptance_valid": all_valid,
        "acceptance_valid": bool(enough_runs and all_valid),
        "pass": passed,
        "status": status,
        "reports": [str(path) for path in paths],
    }


def write_json(path: Path | None, result: dict) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def write_samples_csv(path: Path, samples: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(samples)
    row_count = len(samples[keys[0]])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row_index in range(row_count):
            writer.writerow([samples[key][row_index] for key in keys])


def add_rotation_argument(parser: argparse.ArgumentParser, option: str, help_text: str) -> None:
    parser.add_argument(
        option,
        nargs=9,
        type=float,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        default=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate formal yaw stability or historical static self drift."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    formal = subparsers.add_parser(
        "formal", help="Evaluate estimate yaw error against reference truth."
    )
    formal.add_argument("--estimate", type=Path, required=True)
    formal.add_argument("--reference", type=Path, required=True)
    formal.add_argument("--estimate-frame", required=True)
    formal.add_argument("--reference-frame", required=True)
    formal.add_argument("--world-frame", required=True)
    formal.add_argument("--estimate-time-offset-sec", type=float, default=0.0)
    formal.add_argument("--reference-time-offset-sec", type=float, default=0.0)
    formal.add_argument("--warmup-sec", type=float, default=30.0)
    formal.add_argument("--max-reference-bracket-sec", type=float, default=0.5)
    formal.add_argument("--threshold-deg-per-sec", type=float, default=0.005)
    formal.add_argument("--minimum-duration-sec", type=float, default=300.0)
    formal.add_argument("--minimum-distance-m", type=float, default=1000.0)
    formal.add_argument("--minimum-samples", type=int, default=100)
    add_rotation_argument(
        formal,
        "--estimate-body-to-common-body",
        "Fixed row-major rotation from the estimate body frame to the common body frame.",
    )
    add_rotation_argument(
        formal,
        "--reference-body-to-common-body",
        "Fixed row-major rotation from the reference body frame to the common body frame.",
    )
    formal.add_argument("--samples-csv", type=Path)
    formal.add_argument("--output-json", type=Path)

    aggregate = subparsers.add_parser(
        "aggregate", help="Judge the maximum metric from multiple formal runs."
    )
    aggregate.add_argument("reports", type=Path, nargs="+")
    aggregate.add_argument("--required-runs", type=int, default=5)
    aggregate.add_argument("--threshold-deg-per-sec", type=float)
    aggregate.add_argument("--output-json", type=Path)

    diagnostic = subparsers.add_parser(
        "diagnostic-static",
        help="Run the historical no-truth self-drift diagnostic (not acceptance).",
    )
    diagnostic.add_argument("trajectory", type=Path)
    diagnostic.add_argument("--warmup-sec", type=float, default=30.0)
    diagnostic.add_argument("--minimum-duration-sec", type=float, default=300.0)
    diagnostic.add_argument("--threshold-deg-per-sec", type=float, default=0.005)
    diagnostic.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "formal":
        estimate = load_tum(args.estimate, time_offset_sec=args.estimate_time_offset_sec)
        reference = load_tum(args.reference, time_offset_sec=args.reference_time_offset_sec)
        report, samples = evaluate_formal(
            estimate,
            reference,
            estimate_path=str(args.estimate),
            reference_path=str(args.reference),
            estimate_frame=args.estimate_frame,
            reference_frame=args.reference_frame,
            world_frame=args.world_frame,
            estimate_body_to_common_body=validate_rotation_matrix(
                args.estimate_body_to_common_body, "estimate_body_to_common_body"
            ),
            reference_body_to_common_body=validate_rotation_matrix(
                args.reference_body_to_common_body, "reference_body_to_common_body"
            ),
            warmup_sec=args.warmup_sec,
            max_reference_bracket_sec=args.max_reference_bracket_sec,
            threshold_deg_per_sec=args.threshold_deg_per_sec,
            minimum_duration_sec=args.minimum_duration_sec,
            minimum_distance_m=args.minimum_distance_m,
            minimum_samples=args.minimum_samples,
        )
        if args.samples_csv is not None:
            write_samples_csv(args.samples_csv, samples)
        write_json(args.output_json, report)
        return 0 if report["pass"] else 2

    if args.mode == "aggregate":
        report = aggregate_formal_reports(
            args.reports,
            required_runs=args.required_runs,
            output_threshold=args.threshold_deg_per_sec,
        )
        write_json(args.output_json, report)
        return 0 if report["pass"] else 2

    trajectory = load_tum(args.trajectory)
    report = evaluate_diagnostic_static(
        trajectory,
        warmup_sec=args.warmup_sec,
        threshold_deg_per_sec=args.threshold_deg_per_sec,
        minimum_duration_sec=args.minimum_duration_sec,
    )
    write_json(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
