#!/usr/bin/env python3
"""Evaluate constant-bias IMU preintegration on data1223.

This ROS-independent evaluator reports both the full SO(3) attitude error and
the yaw-only error.  It compares raw gyro integration, constant-bias-corrected
integration, and the discrete truth-rate integration floor over identical
image-time windows.

Unless a fixed bias is supplied, the bias is estimated with RealState truth.
That default is an offline upper-bound experiment, not a deployable online
calibration method.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def excel_column_index(reference: str) -> int:
    value = 0
    for character in "".join(
        item for item in reference if item.isalpha()
    ).upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def load_xlsx_numeric_columns(
    path: Path,
    column_names: Sequence[str],
) -> np.ndarray:
    """Read selected numeric columns without requiring openpyxl."""

    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.parse(archive.open("xl/sharedStrings.xml")).getroot()
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{NS}t"))
                for item in root.findall(f"{NS}si")
            ]
        worksheet = ET.parse(
            archive.open("xl/worksheets/sheet1.xml")
        ).getroot()
        rows = worksheet.findall(f".//{NS}row")
        if not rows:
            raise ValueError(f"No rows found in {path}")

        def values(row: ET.Element) -> dict[int, str]:
            output: dict[int, str] = {}
            for cell in row.findall(f"{NS}c"):
                value_node = cell.find(f"{NS}v")
                value = "" if value_node is None else (value_node.text or "")
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{NS}t")
                    )
                output[
                    excel_column_index(cell.attrib.get("r", "A1"))
                ] = value
            return output

        header = {text: index for index, text in values(rows[0]).items()}
        missing = [name for name in column_names if name not in header]
        if missing:
            raise KeyError(f"Missing columns in {path}: {missing}")
        indices = [header[name] for name in column_names]
        output = []
        for row in rows[1:]:
            row_values = values(row)
            try:
                output.append([float(row_values[index]) for index in indices])
            except (KeyError, ValueError) as exception:
                raise ValueError(
                    f"Invalid row {row.attrib.get('r')} in {path}"
                ) from exception
    return np.asarray(output, dtype=np.float64)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def exp_so3(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if angle < 1e-10:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    return (
        np.eye(3)
        + math.sin(angle) / angle * matrix
        + (1.0 - math.cos(angle)) / (angle * angle) * matrix @ matrix
    )


def log_so3(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    if angle < 1e-10:
        return 0.5 * vee
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (matrix + np.eye(3))
        )
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(axis @ vee) < 0.0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * vee


def rpy_xyz_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation_z @ rotation_y @ rotation_x


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion has zero or invalid norm")
    return value / norm


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized-or-raw [w, x, y, z] quaternion to SO(3)."""

    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert SO(3) to a normalized [w, x, y, z] quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(trace + 1.0, 0.0))
        value = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * math.sqrt(
                max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)
            )
            value = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * math.sqrt(
                max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)
            )
            value = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(
                max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)
            )
            value = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_quaternion_wxyz(value)


def slerp_quaternion_wxyz(
    start: np.ndarray,
    end: np.ndarray,
    ratio: float,
) -> np.ndarray:
    first = normalize_quaternion_wxyz(start)
    second = normalize_quaternion_wxyz(end)
    dot = float(first @ second)
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_wxyz(first + ratio * (second - first))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion_wxyz(
        math.sin((1.0 - ratio) * angle) / sine * first
        + math.sin(ratio * angle) / sine * second
    )


def rotation_trajectory_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    return np.asarray(
        [matrix_to_quaternion_wxyz(rotation) for rotation in rotations]
    )


def quaternion_trajectory_to_rotations(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [quaternion_wxyz_to_matrix(value) for value in quaternions]
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(log_so3(rotation))))


def yaw_deg(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation).reshape(3, 3)
    return math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))


def wrapped_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def interpolate_rotation(
    timestamps: np.ndarray,
    rotations: np.ndarray,
    timestamp: float,
) -> np.ndarray:
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise ValueError(f"Rotation timestamp outside data: {timestamp}")
    right = int(np.searchsorted(timestamps, timestamp, side="left"))
    if right < len(timestamps) and abs(timestamps[right] - timestamp) < 1e-7:
        return rotations[right].copy()
    if right == 0 or right == len(timestamps):
        raise ValueError(f"Cannot bracket timestamp: {timestamp}")
    left = right - 1
    ratio = float(
        (timestamp - timestamps[left])
        / (timestamps[right] - timestamps[left])
    )
    relative = rotations[left].T @ rotations[right]
    return rotations[left] @ exp_so3(ratio * log_so3(relative))


def interpolate_quaternion(
    timestamps: np.ndarray,
    quaternions: np.ndarray,
    timestamp: float,
    *,
    max_bracket_seconds: float | None = None,
) -> np.ndarray:
    """SLERP a quaternion and reject an excessive reference-data gap."""

    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise ValueError(f"Quaternion timestamp outside data: {timestamp}")
    right = int(np.searchsorted(timestamps, timestamp, side="left"))
    if right < len(timestamps) and abs(timestamps[right] - timestamp) < 1e-7:
        return normalize_quaternion_wxyz(quaternions[right])
    if right == 0 or right == len(timestamps):
        raise ValueError(f"Cannot bracket quaternion timestamp: {timestamp}")
    left = right - 1
    bracket = float(timestamps[right] - timestamps[left])
    if max_bracket_seconds is not None and bracket > max_bracket_seconds:
        raise ValueError(
            "Reference quaternion bracket exceeds the frozen limit: "
            f"{bracket:.9f}s > {max_bracket_seconds:.9f}s"
        )
    ratio = float((timestamp - timestamps[left]) / bracket)
    return slerp_quaternion_wxyz(
        quaternions[left], quaternions[right], ratio
    )


def preintegrate_gyro(
    timestamps: np.ndarray,
    angular_rates_deg_per_sec: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """Return R_start_end from trapezoidal body-rate integration."""

    if end_time <= start_time:
        raise ValueError("Preintegration duration must be positive")
    if start_time < timestamps[0] or end_time > timestamps[-1]:
        raise ValueError("Preintegration interval is outside IMU coverage")
    internal = timestamps[(timestamps > start_time) & (timestamps < end_time)]
    nodes = np.concatenate(([start_time], internal, [end_time]))
    node_rates = np.column_stack(
        [
            np.interp(nodes, timestamps, angular_rates_deg_per_sec[:, axis])
            for axis in range(3)
        ]
    )
    delta_rotation = np.eye(3)
    for index in range(len(nodes) - 1):
        delta_time = float(nodes[index + 1] - nodes[index])
        average_rate = 0.5 * (node_rates[index] + node_rates[index + 1])
        delta_rotation = delta_rotation @ exp_so3(
            np.deg2rad(average_rate) * delta_time
        )
    return delta_rotation


def distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(array * array))),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def estimate_truth_assisted_constant_bias(
    measured_rates: np.ndarray,
    truth_rates: np.ndarray,
    mad_multiplier: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(measured_rates) - np.asarray(truth_rates)
    centre = np.median(residual, axis=0)
    mad = np.median(np.abs(residual - centre), axis=0)
    robust_sigma = np.maximum(1.4826 * mad, 1e-4)
    inliers = np.all(
        np.abs(residual - centre) <= mad_multiplier * robust_sigma,
        axis=1,
    )
    if int(np.sum(inliers)) < 10:
        raise ValueError("Too few bias-estimation inliers")
    bias = np.median(residual[inliers], axis=0)
    return bias, inliers, residual[inliers] - bias


def continuous_orientation_trajectory(
    *,
    imu_timestamps: np.ndarray,
    angular_rates_deg_per_sec: np.ndarray,
    image_timestamps: np.ndarray,
    initial_rotation: np.ndarray,
) -> np.ndarray:
    """Continuously integrate one orientation at every image timestamp."""

    rotations = [np.asarray(initial_rotation, dtype=np.float64).copy()]
    for index in range(len(image_timestamps) - 1):
        delta = preintegrate_gyro(
            imu_timestamps,
            angular_rates_deg_per_sec,
            float(image_timestamps[index]),
            float(image_timestamps[index + 1]),
        )
        rotations.append(rotations[-1] @ delta)
    return np.asarray(rotations, dtype=np.float64)


def orientation_error_change(
    reference_start: np.ndarray,
    reference_end: np.ndarray,
    estimated_start: np.ndarray,
    estimated_end: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return full/yaw errors at the endpoints and their change.

    Rotations use the body-to-world convention.  The attitude error is
    ``E(t) = R_ref(t)^T R_est(t)`` and stability uses
    ``angle(E(t0)^T E(t1))`` exactly as specified by the test procedure.
    Yaw-only stability is the wrapped change of the yaw component of E(t).
    """

    error_start = reference_start.T @ estimated_start
    error_end = reference_end.T @ estimated_end
    full_error_start = rotation_angle_deg(error_start)
    full_error_end = rotation_angle_deg(error_end)
    full_error_change = rotation_angle_deg(error_start.T @ error_end)

    yaw_error_start = wrapped_angle_deg(yaw_deg(error_start))
    yaw_error_end = wrapped_angle_deg(yaw_deg(error_end))
    yaw_error_change = abs(
        wrapped_angle_deg(yaw_error_end - yaw_error_start)
    )
    return (
        full_error_start,
        full_error_end,
        full_error_change,
        yaw_error_change,
    )


def evaluate_windows(
    *,
    reference_rotations: np.ndarray,
    estimated_trajectories: Mapping[str, np.ndarray],
    image_timestamps: np.ndarray,
    reference_mode: str,
    gap: int,
    threshold_deg_per_sec: float,
) -> dict:
    if gap <= 0 or gap >= len(image_timestamps):
        raise ValueError(f"Invalid image gap: {gap}")
    metrics = {
        name: {
            "full_error_change_deg": [],
            "yaw_error_change_deg": [],
            "full_error_change_rate_deg_per_sec": [],
            "yaw_error_change_rate_deg_per_sec": [],
        }
        for name in estimated_trajectories
    }
    window_records = {name: [] for name in estimated_trajectories}
    durations = []
    for start_index in range(len(image_timestamps) - gap):
        end_index = start_index + gap
        start_time = float(image_timestamps[start_index])
        end_time = float(image_timestamps[end_index])
        duration = end_time - start_time
        durations.append(duration)
        for name, estimated in estimated_trajectories.items():
            (
                full_error_start,
                full_error_end,
                full_error_change,
                yaw_error_change,
            ) = orientation_error_change(
                reference_rotations[start_index],
                reference_rotations[end_index],
                estimated[start_index],
                estimated[end_index],
            )
            full_rate = full_error_change / duration
            yaw_rate = yaw_error_change / duration
            values = metrics[name]
            values["full_error_change_deg"].append(full_error_change)
            values["yaw_error_change_deg"].append(yaw_error_change)
            values["full_error_change_rate_deg_per_sec"].append(full_rate)
            values["yaw_error_change_rate_deg_per_sec"].append(yaw_rate)
            window_records[name].append(
                {
                    "start_image_index": start_index,
                    "end_image_index": end_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration,
                    "full_error_at_start_deg": full_error_start,
                    "full_error_at_end_deg": full_error_end,
                    "full_error_change_rate_deg_per_sec": full_rate,
                    "yaw_error_change_rate_deg_per_sec": yaw_rate,
                }
            )

    cases = {}
    for name, values in metrics.items():
        cases[name] = {
            key: distribution(value) for key, value in values.items()
        }
        full_rates = np.asarray(
            values["full_error_change_rate_deg_per_sec"]
        )
        yaw_rates = np.asarray(
            values["yaw_error_change_rate_deg_per_sec"]
        )
        for label, rates in (("full", full_rates), ("yaw", yaw_rates)):
            cases[name][f"{label}_rate_threshold"] = {
                "threshold_deg_per_sec": threshold_deg_per_sec,
                "passing_windows": int(np.sum(rates < threshold_deg_per_sec)),
                "total_windows": int(len(rates)),
                "passing_fraction": float(
                    np.mean(rates < threshold_deg_per_sec)
                ),
            }
        cases[name]["worst_full_rate_windows"] = sorted(
            window_records[name],
            key=lambda item: item[
                "full_error_change_rate_deg_per_sec"
            ],
            reverse=True,
        )[:10]
        cases[name]["worst_yaw_rate_windows"] = sorted(
            window_records[name],
            key=lambda item: item[
                "yaw_error_change_rate_deg_per_sec"
            ],
            reverse=True,
        )[:10]
    return {
        "reference_mode": reference_mode,
        "definition": (
            "continuous error change: E(t)=R_ref(t)^T R_est(t); "
            "full=angle(E(t0)^T E(t1))/dt; "
            "yaw=abs(wrap(e_yaw(t1)-e_yaw(t0)))/dt"
        ),
        "image_gap": gap,
        "window_count": len(durations),
        "duration_seconds": distribution(durations),
        "cases": cases,
    }


def evaluate_stability_from_quaternions(
    *,
    reference_quaternions: np.ndarray,
    estimated_quaternion_trajectories: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    reference_mode: str,
    threshold_deg_per_sec: float,
    minimum_duration_seconds: float,
) -> dict:
    """Apply the formal adjacent-error total-variation stability metric."""

    time_values = np.asarray(timestamps, dtype=np.float64)
    if len(time_values) < 2 or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("Stability timestamps must be strictly increasing")
    normalized_reference = np.asarray(
        [normalize_quaternion_wxyz(q) for q in reference_quaternions]
    )
    if len(normalized_reference) != len(time_values):
        raise ValueError("Reference quaternion/timestamp length mismatch")
    reference_rotations = quaternion_trajectory_to_rotations(
        normalized_reference
    )
    durations = np.diff(time_values)
    effective_duration = float(np.sum(durations))
    cases = {}
    for name, quaternion_values in estimated_quaternion_trajectories.items():
        normalized_estimated = np.asarray(
            [normalize_quaternion_wxyz(q) for q in quaternion_values]
        )
        if len(normalized_estimated) != len(time_values):
            raise ValueError(
                f"Estimated quaternion/timestamp length mismatch: {name}"
            )
        estimated_rotations = quaternion_trajectory_to_rotations(
            normalized_estimated
        )
        full_errors = []
        yaw_errors = []
        for reference, estimated in zip(
            reference_rotations, estimated_rotations
        ):
            error = reference.T @ estimated
            full_errors.append(rotation_angle_deg(error))
            yaw_errors.append(abs(wrapped_angle_deg(yaw_deg(error))))

        full_changes = []
        yaw_changes = []
        interval_records = []
        for index, duration in enumerate(durations, start=1):
            result = orientation_error_change(
                reference_rotations[index - 1],
                reference_rotations[index],
                estimated_rotations[index - 1],
                estimated_rotations[index],
            )
            full_change = result[2]
            yaw_change = result[3]
            full_changes.append(full_change)
            yaw_changes.append(yaw_change)
            interval_records.append(
                {
                    "start_index": index - 1,
                    "end_index": index,
                    "start_time": float(time_values[index - 1]),
                    "end_time": float(time_values[index]),
                    "duration_seconds": float(duration),
                    "full_error_change_deg": full_change,
                    "yaw_error_change_deg": yaw_change,
                    "full_drift_rate_deg_per_sec": (
                        full_change / float(duration)
                    ),
                    "yaw_drift_rate_deg_per_sec": (
                        yaw_change / float(duration)
                    ),
                }
            )

        full_total = float(np.sum(full_changes))
        yaw_total = float(np.sum(yaw_changes))
        full_rate = full_total / effective_duration
        yaw_rate = yaw_total / effective_duration
        duration_pass = effective_duration > minimum_duration_seconds
        cases[name] = {
            "valid_sample_count": int(len(time_values)),
            "valid_interval_count": int(len(durations)),
            "effective_duration_seconds": effective_duration,
            "duration_requirement": {
                "minimum_exclusive_seconds": minimum_duration_seconds,
                "passes": duration_pass,
            },
            "full_orientation": {
                "cumulative_error_change_deg": full_total,
                "time_weighted_mean_drift_rate_deg_per_sec": full_rate,
                "rate_passes": full_rate < threshold_deg_per_sec,
                "overall_passes": (
                    duration_pass and full_rate < threshold_deg_per_sec
                ),
            },
            "yaw_only": {
                "cumulative_error_change_deg": yaw_total,
                "time_weighted_mean_drift_rate_deg_per_sec": yaw_rate,
                "rate_passes": yaw_rate < threshold_deg_per_sec,
                "overall_passes": (
                    duration_pass and yaw_rate < threshold_deg_per_sec
                ),
            },
            "auxiliary_absolute_attitude_error_deg": distribution(
                full_errors
            ),
            "auxiliary_absolute_yaw_error_deg": distribution(yaw_errors),
            "auxiliary_interval_duration_seconds": distribution(durations),
            "worst_full_drift_intervals": sorted(
                interval_records,
                key=lambda item: item["full_drift_rate_deg_per_sec"],
                reverse=True,
            )[:10],
            "worst_yaw_drift_intervals": sorted(
                interval_records,
                key=lambda item: item["yaw_drift_rate_deg_per_sec"],
                reverse=True,
            )[:10],
        }
    return {
        "reference_mode": reference_mode,
        "definition": (
            "q_wxyz -> normalize -> R; E_i=R_ref_i^T R_est_i; "
            "delta_i=E_(i-1)^T E_i; mean=sum(angle(delta_i))/sum(dt_i)"
        ),
        "threshold_deg_per_sec": threshold_deg_per_sec,
        "cases": cases,
    }


def endpoint_change_interval(
    *,
    reference_rotations: np.ndarray,
    estimated_trajectories: Mapping[str, np.ndarray],
    image_timestamps: np.ndarray,
    reference_mode: str,
) -> dict:
    duration = float(image_timestamps[-1] - image_timestamps[0])
    cases = {}
    for name, estimated in estimated_trajectories.items():
        (
            full_error_start,
            full_error_end,
            full_error_change,
            yaw_error_change,
        ) = orientation_error_change(
            reference_rotations[0],
            reference_rotations[-1],
            estimated[0],
            estimated[-1],
        )
        cases[name] = {
            "full_error_at_start_deg": full_error_start,
            "full_error_at_end_deg": full_error_end,
            "full_error_change_deg": full_error_change,
            "yaw_error_change_deg": yaw_error_change,
            "full_error_change_rate_deg_per_sec": (
                full_error_change / duration
            ),
            "yaw_error_change_rate_deg_per_sec": (
                yaw_error_change / duration
            ),
        }
    return {
        "reference_mode": reference_mode,
        "duration_seconds": duration,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-gaps", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--threshold-deg-per-sec", type=float, default=0.005
    )
    parser.add_argument(
        "--minimum-duration-seconds", type=float, default=300.0
    )
    parser.add_argument(
        "--max-reference-bracket-seconds", type=float, default=0.25
    )
    parser.add_argument(
        "--fixed-bias-deg-per-sec",
        type=float,
        nargs=3,
        metavar=("BX", "BY", "BZ"),
    )
    arguments = parser.parse_args()
    data = arguments.data.expanduser().resolve()

    measured = load_xlsx_numeric_columns(
        data / "Measure.xlsx",
        ["%星时(s)", "wbix(deg/s)", "wbiy(deg/s)", "wbiz(deg/s)"],
    )
    truth = load_xlsx_numeric_columns(
        data / "RealState.xlsx",
        [
            "%星时(s)",
            "滚动角真值(deg)",
            "俯仰角真值(deg)",
            "偏航角真值(deg)",
            "X角速度真值(deg/s)",
            "Y角速度真值(deg/s)",
            "Z角速度真值(deg/s)",
        ],
    )
    if len(measured) != len(truth) or not np.allclose(
        measured[:, 0], truth[:, 0], atol=1e-7, rtol=0.0
    ):
        raise ValueError("Measure and RealState timestamps do not match")
    imu_timestamps = measured[:, 0]
    measured_rates = measured[:, 1:4]
    truth_rates = truth[:, 4:7]
    truth_rotations = np.asarray(
        [rpy_xyz_matrix(*angles) for angles in truth[:, 1:4]]
    )
    image_timestamps = np.atleast_1d(
        np.loadtxt(data / "timestamps.txt", skiprows=1, usecols=0)
    )
    input_image_timestamp_count = int(len(image_timestamps))
    if len(image_timestamps) < 2 or np.any(np.diff(image_timestamps) <= 0.0):
        raise ValueError("Image timestamps are invalid")

    if arguments.fixed_bias_deg_per_sec is None:
        bias, inliers, corrected_residuals = (
            estimate_truth_assisted_constant_bias(
                measured_rates, truth_rates
            )
        )
        bias_source = "offline_truth_assisted_robust_median"
    else:
        bias = np.asarray(arguments.fixed_bias_deg_per_sec)
        inliers = np.ones(len(measured_rates), dtype=bool)
        corrected_residuals = measured_rates - truth_rates - bias
        bias_source = "user_supplied_independent_calibration"

    rate_cases = {
        "truth_rate_discrete_baseline": truth_rates,
        "imu_raw": measured_rates,
        "imu_constant_bias_corrected": measured_rates - bias,
    }
    truth_quaternions_at_imu = rotation_trajectory_to_quaternions(
        truth_rotations
    )
    valid_image_timestamps = []
    sampled_reference_quaternions = []
    rejected_time_sync_samples = []
    for index, timestamp in enumerate(image_timestamps):
        try:
            reference_quaternion = interpolate_quaternion(
                imu_timestamps,
                truth_quaternions_at_imu,
                float(timestamp),
                max_bracket_seconds=(
                    arguments.max_reference_bracket_seconds
                ),
            )
        except ValueError as exception:
            rejected_time_sync_samples.append(
                {
                    "input_index": index,
                    "timestamp": float(timestamp),
                    "reason": str(exception),
                }
            )
            continue
        valid_image_timestamps.append(float(timestamp))
        sampled_reference_quaternions.append(reference_quaternion)
    image_timestamps = np.asarray(valid_image_timestamps, dtype=np.float64)
    sampled_attitude_truth_quaternions = np.asarray(
        sampled_reference_quaternions, dtype=np.float64
    )
    if len(image_timestamps) < 2:
        raise ValueError("Fewer than two time-synchronized attitude samples")
    sampled_attitude_truth = quaternion_trajectory_to_rotations(
        sampled_attitude_truth_quaternions
    )
    initial_rotation = sampled_attitude_truth[0]
    estimated_trajectories = {
        name: continuous_orientation_trajectory(
            imu_timestamps=imu_timestamps,
            angular_rates_deg_per_sec=rates,
            image_timestamps=image_timestamps,
            initial_rotation=initial_rotation,
        )
        for name, rates in rate_cases.items()
    }
    estimated_quaternion_trajectories = {
        name: rotation_trajectory_to_quaternions(rotations)
        for name, rotations in estimated_trajectories.items()
    }
    reference_trajectories = {
        "attitude_truth": sampled_attitude_truth,
        "truth_rate_preintegration": estimated_trajectories[
            "truth_rate_discrete_baseline"
        ],
    }
    reference_quaternion_trajectories = {
        "attitude_truth": sampled_attitude_truth_quaternions,
        "truth_rate_preintegration": estimated_quaternion_trajectories[
            "truth_rate_discrete_baseline"
        ],
    }
    window_evaluations = {
        reference_mode: {
            f"gap_{gap}": evaluate_windows(
                reference_rotations=reference_rotations,
                estimated_trajectories=estimated_trajectories,
                image_timestamps=image_timestamps,
                reference_mode=reference_mode,
                gap=gap,
                threshold_deg_per_sec=arguments.threshold_deg_per_sec,
            )
            for gap in arguments.window_gaps
        }
        for reference_mode, reference_rotations in (
            reference_trajectories.items()
        )
    }
    endpoint_change_evaluations = {
        reference_mode: endpoint_change_interval(
            reference_rotations=reference_rotations,
            estimated_trajectories=estimated_trajectories,
            image_timestamps=image_timestamps,
            reference_mode=reference_mode,
        )
        for reference_mode, reference_rotations in (
            reference_trajectories.items()
        )
    }
    stability_evaluations = {
        reference_mode: evaluate_stability_from_quaternions(
            reference_quaternions=reference_quaternions,
            estimated_quaternion_trajectories=(
                estimated_quaternion_trajectories
            ),
            timestamps=image_timestamps,
            reference_mode=reference_mode,
            threshold_deg_per_sec=arguments.threshold_deg_per_sec,
            minimum_duration_seconds=arguments.minimum_duration_seconds,
        )
        for reference_mode, reference_quaternions in (
            reference_quaternion_trajectories.items()
        )
    }
    payload = {
        "schema_version": 4,
        "dataset": str(data),
        "formal_test_readiness": {
            "ready": False,
            "reasons": [
                "RealState.xlsx stores reference Euler angles, not raw reference quaternions.",
            "The fixed sensor-to-body and world-frame transforms are not supplied.",
            "The STSA/STSB quaternion stream meaning and ordering are not documented.",
            "Algorithm output quaternions were not recorded; the estimate is reconstructed from sparse image-time IMU preintegration.",
            ],
            "unused_raw_measurement_quaternion_columns": [
                "STSAQ1..4",
                "STSBQ1..4",
                "STSA2Q1..4",
                "STSB2Q1..4",
            ],
        },
        "attitude_input_pipeline": {
            "quaternion_order": "w,x,y,z",
            "estimated": (
                "IMU gyro preintegration encoded as normalized quaternions "
                "before stability evaluation"
            ),
            "reference": (
                "RealState roll/pitch/yaw converted to quaternions because "
                "the dataset has no raw reference quaternion"
            ),
            "time_alignment": (
                "reference quaternion SLERP at estimate/image timestamps"
            ),
            "max_reference_bracket_seconds": (
                arguments.max_reference_bracket_seconds
            ),
            "fixed_extrinsic": (
                "identity assumed; no dynamic alignment or result-based fit"
            ),
        },
        "definition": {
            "continuous_full_orientation_error": (
                "E(t) = R_ref(t)^T * R_est(t), for body-to-world rotations"
            ),
            "full_error_change": (
                "angle(E(t0)^T * E(t1))"
            ),
            "continuous_yaw_error": (
                "e_yaw(t) = wrap(yaw(E(t)))"
            ),
            "yaw_error_change": (
                "abs(wrap(e_yaw(t1) - e_yaw(t0)))"
            ),
            "rate_metric": "error change divided by window duration",
            "primary_stability_aggregation": (
                "sum of adjacent error-change angles divided by sum of "
                "valid adjacent durations (time-weighted mean rate)"
            ),
            "threshold_deg_per_sec": arguments.threshold_deg_per_sec,
            "minimum_duration_seconds_exclusive": (
                arguments.minimum_duration_seconds
            ),
            "truth_rotation": "R_world_body = Rz(yaw)*Ry(pitch)*Rx(roll)",
        },
        "samples": {
            "imu": int(len(imu_timestamps)),
            "input_estimate_timestamps": input_image_timestamp_count,
            "valid_time_synchronized_samples": int(len(image_timestamps)),
            "rejected_time_sync_samples": rejected_time_sync_samples,
            "imu_median_period_seconds": float(
                np.median(np.diff(imu_timestamps))
            ),
            "estimate_interval_seconds": distribution(
                np.diff(image_timestamps)
            ),
            "image_start_time": float(image_timestamps[0]),
            "image_end_time": float(image_timestamps[-1]),
        },
        "constant_bias": {
            "source": bias_source,
            "value_deg_per_sec": bias.tolist(),
            "truth_assisted_inlier_fraction": float(np.mean(inliers)),
            "corrected_residual_mean_xyz_deg_per_sec": np.mean(
                corrected_residuals, axis=0
            ).tolist(),
            "corrected_residual_std_xyz_deg_per_sec": np.std(
                corrected_residuals, axis=0
            ).tolist(),
            "warning": (
                "Truth-assisted bias is an offline upper bound. Runtime must "
                "use a separately verified stationary calibration interval."
            ),
        },
        "stability_evaluation": stability_evaluations,
        "window_evaluations": window_evaluations,
        "endpoint_change_no_visual_reset_auxiliary": (
            endpoint_change_evaluations
        ),
    }
    if arguments.output:
        output = arguments.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "bias_source": bias_source,
        "bias_deg_per_sec": bias.tolist(),
        "formal_test_ready": payload["formal_test_readiness"]["ready"],
        "stability_evaluation": {
            reference_name: {
                case_name: {
                    "duration_seconds": case["effective_duration_seconds"],
                    "full_rate": case["full_orientation"][
                        "time_weighted_mean_drift_rate_deg_per_sec"
                    ],
                    "full_pass": case["full_orientation"][
                        "overall_passes"
                    ],
                    "yaw_rate": case["yaw_only"][
                        "time_weighted_mean_drift_rate_deg_per_sec"
                    ],
                    "yaw_pass": case["yaw_only"]["overall_passes"],
                }
                for case_name, case in evaluation["cases"].items()
            }
            for reference_name, evaluation in stability_evaluations.items()
        },
        "window_evaluations": {},
        "endpoint_change_no_visual_reset_auxiliary": {
            name: value["cases"]
            for name, value in endpoint_change_evaluations.items()
        },
    }
    for reference_name, windows in window_evaluations.items():
        summary["window_evaluations"][reference_name] = {}
        for window_name, window in windows.items():
            summary["window_evaluations"][reference_name][window_name] = {}
            for case_name, case in window["cases"].items():
                summary["window_evaluations"][reference_name][window_name][
                    case_name
                ] = {
                    "full_rate_unweighted_mean": case[
                        "full_error_change_rate_deg_per_sec"
                    ]["mean"],
                    "full_rate_rmse": case[
                        "full_error_change_rate_deg_per_sec"
                    ]["rmse"],
                    "full_rate_p95": case[
                        "full_error_change_rate_deg_per_sec"
                    ]["p95"],
                    "yaw_rate_rmse": case[
                        "yaw_error_change_rate_deg_per_sec"
                    ]["rmse"],
                    "yaw_rate_unweighted_mean": case[
                        "yaw_error_change_rate_deg_per_sec"
                    ]["mean"],
                    "yaw_rate_p95": case[
                        "yaw_error_change_rate_deg_per_sec"
                    ]["p95"],
                    "full_pass_fraction": case[
                        "full_rate_threshold"
                    ]["passing_fraction"],
                    "yaw_pass_fraction": case[
                        "yaw_rate_threshold"
                    ]["passing_fraction"],
                }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
