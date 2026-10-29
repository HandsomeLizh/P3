#!/usr/bin/env python3
"""Evaluate the complete Task-3 acceptance metric set from one run directory.

The evaluator never substitutes estimator self-consistency for ground truth.
When a bag has no PQ messages, reference-dependent metrics are reported as
``unavailable``.  PQ decoding defaults to the observed T5 seven-value layout
``counter, x, y, z, roll, pitch, yaw`` and records that assumption in every
report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_orientation_stability import (
    TumTrajectory,
    interpolate_reference,
    load_tum,
    quaternion_xyzw_to_matrix,
    reference_path_length,
    series_statistics,
    wrap_deg,
    yaw_deg,
)


DEFAULT_PQ_TOPIC = "/Car/T5/pq_data"
DEFAULT_STAMPED_PQ_TOPIC = "/t3/reference/pq_stamped"
DEFAULT_PQ_RPY_SIGNS = (-1.0, 1.0, 1.0)
REFERENCE_SAMPLE_RELATIVE_PATH = Path("_internal/reference_pq_samples.jsonl")
REFERENCE_RECORDING_METADATA_RELATIVE_PATH = Path(
    "_internal/reference_pq_recording.json"
)
POSITION_THRESHOLD_FRACTION = 0.01
TIME_THRESHOLD_SEC = 2.0
STABILITY_THRESHOLD_DEG_PER_SEC = 0.005
STABILITY_PERIOD_SEC = 500.0
DOCUMENT_SAMPLE_SIZE = 10
DOCUMENT_SAMPLE_SEED = 20260821
MAP_RANGE_THRESHOLD_KM2 = 0.5
STORAGE_THRESHOLD_GB_PER_KM2 = 3.0


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def record_reference_data(
    output_dir: Path,
    *,
    pq_topic: str,
    stamped_pq_topic: str,
    parent_pid: int,
) -> int:
    """Passively save PQ samples without participating in the ROMA pipeline."""

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import Float64MultiArray, String
    except ImportError as error:
        raise RuntimeError(
            f"ROS 2 Python dependencies unavailable for reference recorder: {error}"
        ) from error

    sample_path = output_dir / REFERENCE_SAMPLE_RELATIVE_PATH
    metadata_path = output_dir / REFERENCE_RECORDING_METADATA_RELATIVE_PATH
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall_time_sec = time.time()

    class ReferenceDataRecorder(Node):
        def __init__(self) -> None:
            super().__init__("t3_reference_trajectory_recorder")
            self.raw_count = 0
            self.stamped_count = 0
            self.invalid_stamped_count = 0
            self._closed = False
            self._stream = sample_path.open(
                "w", encoding="utf-8", buffering=1
            )
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._pq_subscription = self.create_subscription(
                Float64MultiArray,
                pq_topic,
                self._on_raw_pq,
                qos,
            )
            self._stamped_pq_subscription = self.create_subscription(
                String,
                stamped_pq_topic,
                self._on_stamped_pq,
                qos,
            )
            self._parent_timer = (
                self.create_timer(1.0, self._check_parent)
                if parent_pid > 0
                else None
            )
            self.get_logger().info(
                f"Saving reference data to {sample_path}; "
                f"raw={pq_topic}, stamped={stamped_pq_topic}"
            )

        def _write(self, row: dict[str, Any]) -> None:
            self._stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

        def _on_raw_pq(self, message: Float64MultiArray) -> None:
            received_wall_time_ns = time.time_ns()
            self._write(
                {
                    "schema_version": 1,
                    "timestamp_ns": received_wall_time_ns,
                    "timestamp_source": "subscriber_reception_system_time",
                    "received_wall_time_ns": received_wall_time_ns,
                    "received_monotonic_ns": time.monotonic_ns(),
                    "values": [float(value) for value in message.data],
                }
            )
            self.raw_count += 1

        def _on_stamped_pq(self, message: String) -> None:
            received_wall_time_ns = time.time_ns()
            try:
                payload = json.loads(message.data)
                timestamp_ns = int(payload["timestamp_ns"])
                values = [float(value) for value in payload["values"]]
                if timestamp_ns <= 0 or not values:
                    raise ValueError("timestamp and values must be non-empty")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.invalid_stamped_count += 1
                return
            self._write(
                {
                    "schema_version": 1,
                    "timestamp_ns": timestamp_ns,
                    "timestamp_source": "publisher_record_time",
                    "received_wall_time_ns": received_wall_time_ns,
                    "received_monotonic_ns": time.monotonic_ns(),
                    "values": values,
                }
            )
            self.stamped_count += 1

        def _check_parent(self) -> None:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                self.get_logger().info("ROMA launcher exited; stopping recorder")
                rclpy.shutdown()
            except PermissionError:
                pass

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            self._stream.close()
            metadata = {
                "schema_version": 1,
                "status": "available",
                "raw_topic": pq_topic,
                "stamped_topic": stamped_pq_topic,
                "sample_file": str(sample_path),
                "raw_sample_count": self.raw_count,
                "publisher_stamped_sample_count": self.stamped_count,
                "invalid_publisher_stamped_sample_count": (
                    self.invalid_stamped_count
                ),
                "started_wall_time_sec": started_wall_time_sec,
                "ended_wall_time_sec": time.time(),
                "note": (
                    "publisher_record_time is authoritative during custom bag "
                    "replay; subscriber reception time is the live-source "
                    "fallback because Float64MultiArray has no Header"
                ),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    rclpy.init()
    node = ReferenceDataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _elapsed_time_basis(summary_path: Path) -> dict[str, Any]:
    fallback = {
        "status": "fallback_to_sensor_time",
        "source": "trajectory/bag timestamps",
        "wall_time_per_sensor_time_scale": 1.0,
        "synchronized_sensor_span_sec": None,
        "synchronized_wall_span_sec": None,
        "effective_consumption_rate_x": None,
    }
    if not summary_path.exists():
        fallback["reason"] = f"missing {summary_path.name}"
        return fallback
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sensor_span = float(summary.get("synchronized_sensor_span_sec"))
        wall_span = float(summary.get("synchronized_wall_span_sec"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        fallback["reason"] = f"invalid runtime span: {error}"
        return fallback
    if (
        not math.isfinite(sensor_span)
        or not math.isfinite(wall_span)
        or sensor_span <= 0.0
        or wall_span <= 0.0
    ):
        fallback["reason"] = "runtime spans must both be finite and positive"
        return fallback
    scale = wall_span / sensor_span
    return {
        "status": "available",
        "source": "input_processing_summary synchronized wall-clock span",
        "wall_time_per_sensor_time_scale": scale,
        "synchronized_sensor_span_sec": sensor_span,
        "synchronized_wall_span_sec": wall_span,
        "effective_consumption_rate_x": sensor_span / wall_span,
        "definition": (
            "trajectory matching uses sensor timestamps; elapsed durations, "
            "average speed and 500 s stability periods use measured wall time"
        ),
    }


def _status(value: float | None, threshold: float, *, valid: bool = True) -> str:
    if value is None:
        return "unavailable"
    if not valid:
        return "insufficient_data"
    return "pass" if value <= threshold else "fail"


def _status_at_least(
    value: float | None, threshold: float, *, valid: bool = True
) -> str:
    if value is None:
        return "unavailable"
    if not valid:
        return "insufficient_data"
    return "pass" if value >= threshold else "fail"


def _euler_rpy_deg_to_quaternion_xyzw(
    roll_deg: float, pitch_deg: float, yaw_value_deg: float
) -> np.ndarray:
    roll, pitch, yaw_value = np.deg2rad([roll_deg, pitch_deg, yaw_value_deg])
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw_value / 2.0), math.sin(yaw_value / 2.0)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _apply_rpy_signs_to_trajectory(
    trajectory: TumTrajectory,
    signs: tuple[float, float, float],
) -> TumTrajectory:
    """Apply an explicit Euler-field polarity conversion to a trajectory."""

    sign_values = np.asarray(signs, dtype=np.float64)
    if sign_values.shape != (3,) or not np.all(np.isin(sign_values, (-1.0, 1.0))):
        raise ValueError("RPY signs must contain exactly three values from {-1, +1}")
    rotations = np.stack(
        [quaternion_xyzw_to_matrix(value) for value in trajectory.quaternions_xyzw]
    )
    corrected_rpy = _rotation_matrix_to_euler_xyz_deg(rotations) * sign_values
    corrected_quaternions = np.stack(
        [_euler_rpy_deg_to_quaternion_xyzw(*value) for value in corrected_rpy]
    )
    return TumTrajectory(
        trajectory.timestamps.copy(),
        trajectory.positions.copy(),
        corrected_quaternions,
    )


def _rotation_matrix_to_euler_xyz_deg(rotations: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must have shape (..., 3, 3)")

    sin_pitch = np.clip(-values[..., 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    nonsingular = np.abs(np.cos(pitch)) > 1e-8

    regular_roll = np.arctan2(values[..., 2, 1], values[..., 2, 2])
    regular_yaw = np.arctan2(values[..., 1, 0], values[..., 0, 0])

    # At gimbal lock choose yaw = 0 and retain an equivalent roll.  This gives
    # a deterministic XYZ representation instead of amplifying round-off.
    singular_roll = np.where(
        sin_pitch >= 0.0,
        np.arctan2(values[..., 0, 1], values[..., 1, 1]),
        np.arctan2(-values[..., 0, 1], values[..., 1, 1]),
    )
    roll = np.where(nonsingular, regular_roll, singular_roll)
    yaw_value = np.where(nonsingular, regular_yaw, 0.0)
    return np.rad2deg(np.stack([roll, pitch, yaw_value], axis=-1))


def _orientation_error_rpy_deg(
    estimate_rotations: np.ndarray, reference_rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate_rpy = _rotation_matrix_to_euler_xyz_deg(estimate_rotations)
    reference_rpy = _rotation_matrix_to_euler_xyz_deg(reference_rotations)
    error_rpy = wrap_deg(estimate_rpy - reference_rpy)
    return estimate_rpy, reference_rpy, error_rpy


def extract_pq_reference(
    bag: Path,
    topic: str,
    output_tum: Path,
    *,
    position_indices: tuple[int, int, int],
    rpy_indices: tuple[int, int, int],
    rpy_signs: tuple[float, float, float] = DEFAULT_PQ_RPY_SIGNS,
) -> tuple[TumTrajectory | None, dict[str, Any]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import Float64MultiArray
    except ImportError as error:
        return None, {
            "status": "unavailable",
            "reason": f"ROS 2 Python bag dependencies unavailable: {error}",
        }

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {
        value.name: value.type for value in reader.get_all_topics_and_types()
    }
    if topic not in topic_types:
        return None, {
            "status": "unavailable",
            "reason": f"PQ topic is not registered: {topic}",
            "topic": topic,
            "message_count": 0,
        }
    expected_type = "std_msgs/msg/Float64MultiArray"
    if topic_types[topic] != expected_type:
        return None, {
            "status": "unavailable",
            "reason": (
                f"PQ topic type is {topic_types[topic]}, expected {expected_type}"
            ),
            "topic": topic,
            "message_count": 0,
        }

    bag_filter = rosbag2_py.StorageFilter()
    bag_filter.topics = [topic]
    reader.set_filter(bag_filter)
    maximum_index = max(*position_indices, *rpy_indices)
    rows: list[tuple[float, np.ndarray, np.ndarray, list[float]]] = []
    invalid_count = 0
    while reader.has_next():
        _, serialized, bag_ns = reader.read_next()
        message = deserialize_message(serialized, Float64MultiArray)
        values = np.asarray(message.data, dtype=np.float64)
        if len(values) <= maximum_index or not np.isfinite(values).all():
            invalid_count += 1
            continue
        position = values[list(position_indices)]
        decoded_rpy = values[list(rpy_indices)] * np.asarray(rpy_signs)
        quaternion = _euler_rpy_deg_to_quaternion_xyzw(*decoded_rpy)
        rows.append((bag_ns * 1e-9, position, quaternion, values.tolist()))

    if not rows:
        return None, {
            "status": "unavailable",
            "reason": f"PQ topic contains zero valid messages: {topic}",
            "topic": topic,
            "message_count": 0,
            "invalid_message_count": invalid_count,
        }

    rows.sort(key=lambda row: row[0])
    unique_rows = []
    duplicate_timestamp_count = 0
    for row in rows:
        if unique_rows and row[0] <= unique_rows[-1][0]:
            duplicate_timestamp_count += 1
            continue
        unique_rows.append(row)
    timestamps = np.asarray([row[0] for row in unique_rows], dtype=np.float64)
    positions = np.asarray([row[1] for row in unique_rows], dtype=np.float64)
    quaternions = np.asarray([row[2] for row in unique_rows], dtype=np.float64)
    if len(timestamps) < 2:
        return None, {
            "status": "unavailable",
            "reason": "PQ topic contains fewer than two unique valid timestamps",
            "topic": topic,
            "message_count": len(timestamps),
        }

    output_tum.parent.mkdir(parents=True, exist_ok=True)
    with output_tum.open("w", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, position, quaternion in zip(
            timestamps, positions, quaternions
        ):
            values = [timestamp, *position.tolist(), *quaternion.tolist()]
            stream.write(" ".join(f"{value:.9f}" for value in values) + "\n")

    dt = np.diff(timestamps)
    distance_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = distance_steps / dt
    rotations = np.stack(
        [quaternion_xyzw_to_matrix(value) for value in quaternions]
    )
    yaw_steps = np.abs(wrap_deg(np.diff(yaw_deg(rotations))))
    yaw_rate = yaw_steps / dt
    metadata = {
        "status": "available",
        "topic": topic,
        "message_type": expected_type,
        "message_count": int(len(timestamps)),
        "invalid_message_count": int(invalid_count),
        "duplicate_or_nonmonotonic_timestamp_count": int(
            duplicate_timestamp_count
        ),
        "timestamp_source": "rosbag SQLite messages.timestamp",
        "payload_layout_assumption": (
            "counter, x, y, z, roll_deg, pitch_deg, yaw_deg"
        ),
        "position_indices": list(position_indices),
        "rpy_indices": list(rpy_indices),
        "rpy_signs_applied": list(rpy_signs),
        "rpy_sign_evidence": (
            "PQ roll polarity is opposite to ROMA and independent WIT roll "
            "over the 2026-08-24 validation bag; configurable by --pq-rpy-signs"
        ),
        "euler_convention": "world_R_body = Rz(yaw) * Ry(pitch) * Rx(roll)",
        "output_tum": str(output_tum),
        "duration_sec": float(timestamps[-1] - timestamps[0]),
        "record_interval_sec": series_statistics(dt),
        "diagnostic_speed_mps": series_statistics(speed),
        "diagnostic_absolute_yaw_rate_deg_per_sec": series_statistics(yaw_rate),
        "warning": (
            "The seven PQ fields are not self-describing. Confirm their schema, "
            "units, body frame and reference-system accuracy before formal use."
        ),
    }
    return TumTrajectory(timestamps, positions, quaternions), metadata


def extract_saved_pq_reference(
    sample_path: Path,
    output_tum: Path,
    *,
    position_indices: tuple[int, int, int],
    rpy_indices: tuple[int, int, int],
    rpy_signs: tuple[float, float, float] = DEFAULT_PQ_RPY_SIGNS,
) -> tuple[TumTrajectory | None, dict[str, Any]]:
    """Create a TUM reference from passively recorded PQ topic samples."""

    if not sample_path.exists():
        return None, {
            "status": "unavailable",
            "reason": f"missing {sample_path}",
        }

    maximum_index = max(*position_indices, *rpy_indices)
    exact_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    reception_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    invalid_count = 0
    raw_line_count = 0
    for raw_line in sample_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        raw_line_count += 1
        try:
            record = json.loads(line)
            timestamp_ns = int(record["timestamp_ns"])
            values = np.asarray(record["values"], dtype=np.float64)
            timestamp_source = str(record["timestamp_source"])
            if (
                timestamp_ns <= 0
                or len(values) <= maximum_index
                or not np.isfinite(values).all()
            ):
                raise ValueError("invalid timestamp or PQ payload")
            position = values[list(position_indices)]
            decoded_rpy = values[list(rpy_indices)] * np.asarray(rpy_signs)
            quaternion = _euler_rpy_deg_to_quaternion_xyzw(*decoded_rpy)
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            invalid_count += 1
            continue
        row = (timestamp_ns * 1.0e-9, position, quaternion)
        if timestamp_source == "publisher_record_time":
            exact_rows.append(row)
        elif timestamp_source == "subscriber_reception_system_time":
            reception_rows.append(row)
        else:
            invalid_count += 1

    if len(exact_rows) >= 2:
        rows = exact_rows
        timestamp_source = "publisher_record_time"
        timestamp_quality = "exact_bag_record_time"
    else:
        rows = reception_rows
        timestamp_source = "subscriber_reception_system_time"
        timestamp_quality = "live_reception_time_fallback"
    if len(rows) < 2:
        return None, {
            "status": "unavailable",
            "reason": "recording contains fewer than two usable PQ samples",
            "sample_file": str(sample_path),
            "raw_line_count": raw_line_count,
            "invalid_sample_count": invalid_count,
            "publisher_stamped_sample_count": len(exact_rows),
            "reception_timestamp_sample_count": len(reception_rows),
        }

    rows.sort(key=lambda row: row[0])
    unique_rows = []
    duplicate_timestamp_count = 0
    for row in rows:
        if unique_rows and row[0] <= unique_rows[-1][0]:
            duplicate_timestamp_count += 1
            continue
        unique_rows.append(row)
    if len(unique_rows) < 2:
        return None, {
            "status": "unavailable",
            "reason": "recording contains fewer than two unique timestamps",
            "sample_file": str(sample_path),
        }

    timestamps = np.asarray(
        [row[0] for row in unique_rows], dtype=np.float64
    )
    positions = np.asarray(
        [row[1] for row in unique_rows], dtype=np.float64
    )
    quaternions = np.asarray(
        [row[2] for row in unique_rows], dtype=np.float64
    )
    output_tum.parent.mkdir(parents=True, exist_ok=True)
    with output_tum.open("w", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, position, quaternion in zip(
            timestamps, positions, quaternions
        ):
            values = [timestamp, *position.tolist(), *quaternion.tolist()]
            stream.write(" ".join(f"{value:.9f}" for value in values) + "\n")

    dt = np.diff(timestamps)
    distance_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = distance_steps / dt
    metadata = {
        "status": "available",
        "source": "saved ROS PQ topic observations",
        "topic": DEFAULT_PQ_TOPIC,
        "message_type": "std_msgs/msg/Float64MultiArray",
        "sample_file": str(sample_path),
        "output_tum": str(output_tum),
        "message_count": int(len(timestamps)),
        "raw_line_count": raw_line_count,
        "invalid_sample_count": invalid_count,
        "publisher_stamped_sample_count": len(exact_rows),
        "reception_timestamp_sample_count": len(reception_rows),
        "duplicate_or_nonmonotonic_timestamp_count": int(
            duplicate_timestamp_count
        ),
        "timestamp_source": timestamp_source,
        "timestamp_quality": timestamp_quality,
        "payload_layout_assumption": (
            "counter, x, y, z, roll_deg, pitch_deg, yaw_deg"
        ),
        "position_indices": list(position_indices),
        "rpy_indices": list(rpy_indices),
        "rpy_signs_applied": list(rpy_signs),
        "rpy_sign_evidence": (
            "PQ roll polarity is opposite to ROMA and independent WIT roll "
            "over the 2026-08-24 validation bag; configurable by --pq-rpy-signs"
        ),
        "euler_convention": "world_R_body = Rz(yaw) * Ry(pitch) * Rx(roll)",
        "duration_sec": float(timestamps[-1] - timestamps[0]),
        "record_interval_sec": series_statistics(dt),
        "diagnostic_speed_mps": series_statistics(speed),
        "warning": (
            "Live Float64MultiArray has no Header. Reception timestamps are "
            "valid for formal pairing only when the image Header.stamp uses "
            "the same synchronized host clock. Custom bag replay supplies "
            "the exact original bag record timestamp instead."
        ),
    }
    return TumTrajectory(timestamps, positions, quaternions), metadata


def _relative_trajectory(
    positions: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    origin_position = positions[0]
    origin_rotation = rotations[0]
    relative_positions = (origin_rotation.T @ (positions - origin_position).T).T
    relative_rotations = np.einsum("ij,njk->nik", origin_rotation.T, rotations)
    return relative_positions, relative_rotations


def evaluate_reference_metrics(
    estimate: TumTrajectory,
    reference: TumTrajectory,
    *,
    warmup_sec: float,
    maximum_reference_bracket_sec: float,
    elapsed_time_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not math.isfinite(elapsed_time_scale) or elapsed_time_scale <= 0.0:
        raise ValueError("elapsed_time_scale must be finite and positive")

    start_allowed = estimate.timestamps[0] + warmup_sec
    candidate = np.flatnonzero(
        (estimate.timestamps >= start_allowed)
        & (estimate.timestamps >= reference.timestamps[0])
        & (estimate.timestamps <= reference.timestamps[-1])
    )
    if len(candidate) < 2:
        raise ValueError("Estimate and reference have fewer than two overlapping poses")
    accepted_relative, reference_positions, reference_rotations, brackets = (
        interpolate_reference(
            reference,
            estimate.timestamps[candidate],
            max_bracket_sec=maximum_reference_bracket_sec,
        )
    )
    estimate_indices = candidate[accepted_relative]
    if len(estimate_indices) < 2:
        raise ValueError("Reference interpolation produced fewer than two pose pairs")
    timestamps = estimate.timestamps[estimate_indices]
    elapsed_runtime_sec = (
        timestamps - timestamps[0]
    ) * elapsed_time_scale
    estimate_positions = estimate.positions[estimate_indices]
    estimate_rotations = np.stack(
        [
            quaternion_xyzw_to_matrix(value)
            for value in estimate.quaternions_xyzw[estimate_indices]
        ]
    )

    # The Word procedure explicitly requires coordinate conversion and SE(3)
    # alignment.  This is a rigid alignment only; scale is never fitted.
    estimate_center = np.mean(estimate_positions, axis=0)
    reference_center = np.mean(reference_positions, axis=0)
    covariance = (
        (estimate_positions - estimate_center).T
        @ (reference_positions - reference_center)
    )
    left, _, right_transpose = np.linalg.svd(covariance)
    alignment_rotation = right_transpose.T @ left.T
    if np.linalg.det(alignment_rotation) < 0.0:
        right_transpose[-1, :] *= -1.0
        alignment_rotation = right_transpose.T @ left.T
    alignment_translation = (
        reference_center - alignment_rotation @ estimate_center
    )
    aligned_estimate_positions = (
        alignment_rotation @ estimate_positions.T
    ).T + alignment_translation
    aligned_estimate_rotations = np.einsum(
        "ij,njk->nik", alignment_rotation, estimate_rotations
    )

    position_error = np.linalg.norm(
        aligned_estimate_positions - reference_positions, axis=1
    )
    terminal_error = float(position_error[-1])
    position_rmse = float(np.sqrt(np.mean(np.square(position_error))))
    path_length = reference_path_length(
        reference, float(timestamps[0]), float(timestamps[-1])
    )
    motion_duration_sec = max(0.0, float(elapsed_runtime_sec[-1]))
    average_motion_speed_mps = (
        path_length / motion_duration_sec if motion_duration_sec > 0.0 else None
    )

    terminal_fraction = terminal_error / path_length if path_length > 0.0 else None
    rmse_fraction = position_rmse / path_length if path_length > 0.0 else None

    stability_duration_sec = motion_duration_sec
    boundary_times = np.empty(0, dtype=np.float64)
    boundary_elapsed_runtime_sec = np.empty(0, dtype=np.float64)
    period_durations = np.empty(0, dtype=np.float64)
    if stability_duration_sec > 0.0:
        complete_period_count = int(
            math.floor(stability_duration_sec / STABILITY_PERIOD_SEC)
        )
        boundary_elapsed_runtime_sec = STABILITY_PERIOD_SEC * np.arange(
            complete_period_count + 1, dtype=np.float64
        )
        if math.isclose(
            float(boundary_elapsed_runtime_sec[-1]),
            stability_duration_sec,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            boundary_elapsed_runtime_sec[-1] = stability_duration_sec
        else:
            boundary_elapsed_runtime_sec = np.append(
                boundary_elapsed_runtime_sec, stability_duration_sec
            )
        boundary_times = (
            timestamps[0] + boundary_elapsed_runtime_sec / elapsed_time_scale
        )
        boundary_times[-1] = timestamps[-1]
        period_durations = np.diff(boundary_elapsed_runtime_sec)
    period_count = int(len(period_durations))
    complete_period_count = int(
        np.count_nonzero(
            np.isclose(period_durations, STABILITY_PERIOD_SEC, atol=1e-6)
        )
    )
    partial_period_count = period_count - complete_period_count
    period_rows: list[dict[str, Any]] = []
    if period_count:
        estimate_boundary_indices, _, estimate_boundary_rotations, _ = (
            interpolate_reference(
                estimate,
                boundary_times,
                max_bracket_sec=maximum_reference_bracket_sec,
            )
        )
        reference_boundary_indices, _, reference_boundary_rotations, _ = (
            interpolate_reference(
                reference,
                boundary_times,
                max_bracket_sec=maximum_reference_bracket_sec,
            )
        )
        expected_indices = np.arange(len(boundary_times), dtype=np.int64)
        complete_boundaries = (
            np.array_equal(estimate_boundary_indices, expected_indices)
            and np.array_equal(reference_boundary_indices, expected_indices)
        )
        if complete_boundaries:
            aligned_boundary_rotations = np.einsum(
                "ij,njk->nik",
                alignment_rotation,
                estimate_boundary_rotations,
            )
            (
                estimate_boundary_rpy,
                reference_boundary_rpy,
                boundary_error_rpy,
            ) = _orientation_error_rpy_deg(
                aligned_boundary_rotations,
                reference_boundary_rotations,
            )
            estimate_orientation_change_rpy = wrap_deg(
                estimate_boundary_rpy[1:] - estimate_boundary_rpy[:-1]
            )
            reference_orientation_change_rpy = wrap_deg(
                reference_boundary_rpy[1:] - reference_boundary_rpy[:-1]
            )
            error_change_rpy = wrap_deg(
                estimate_orientation_change_rpy - reference_orientation_change_rpy
            )
            error_change_inf_norm = np.max(np.abs(error_change_rpy), axis=1)
            is_complete_period = np.isclose(
                period_durations,
                STABILITY_PERIOD_SEC,
                rtol=0.0,
                atol=1e-6,
            )
            normalization_durations = np.where(
                is_complete_period,
                STABILITY_PERIOD_SEC,
                period_durations,
            )
            period_rates = error_change_inf_norm / normalization_durations
            if period_count <= DOCUMENT_SAMPLE_SIZE:
                selected_indices = list(range(period_count))
                selection_method = "all_periods_when_period_count_le_10"
            else:
                selected_indices = sorted(
                    random.Random(DOCUMENT_SAMPLE_SEED).sample(
                        range(period_count), DOCUMENT_SAMPLE_SIZE
                    )
                )
                selection_method = "deterministic_random_10_periods"
            selected_set = set(selected_indices)
            selection_order = {
                index: order + 1 for order, index in enumerate(selected_indices)
            }
            period_rows = [
                {
                    "period_index": int(index + 1),
                    "start_timestamp_sec": float(boundary_times[index]),
                    "end_timestamp_sec": float(boundary_times[index + 1]),
                    "start_elapsed_runtime_sec": float(
                        boundary_elapsed_runtime_sec[index]
                    ),
                    "end_elapsed_runtime_sec": float(
                        boundary_elapsed_runtime_sec[index + 1]
                    ),
                    "duration_sec": float(period_durations[index]),
                    "motion_time_sec": float(period_durations[index]),
                    "is_partial_period": not math.isclose(
                        float(period_durations[index]),
                        STABILITY_PERIOD_SEC,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ),
                    "normalization_duration_sec": float(
                        normalization_durations[index]
                    ),
                    "estimate_start_rpy_deg": estimate_boundary_rpy[index].tolist(),
                    "estimate_end_rpy_deg": estimate_boundary_rpy[index + 1].tolist(),
                    "reference_start_rpy_deg": reference_boundary_rpy[index].tolist(),
                    "reference_end_rpy_deg": reference_boundary_rpy[index + 1].tolist(),
                    "orientation_error_start_rpy_deg": (
                        boundary_error_rpy[index].tolist()
                    ),
                    "orientation_error_end_rpy_deg": (
                        boundary_error_rpy[index + 1].tolist()
                    ),
                    "reference_orientation_change_rpy_deg": (
                        reference_orientation_change_rpy[index].tolist()
                    ),
                    "estimate_orientation_change_rpy_deg": (
                        estimate_orientation_change_rpy[index].tolist()
                    ),
                    "reference_to_estimate_orientation_change_error_rpy_deg": (
                        error_change_rpy[index].tolist()
                    ),
                    "reference_to_estimate_orientation_change_inf_norm_deg": float(
                        error_change_inf_norm[index]
                    ),
                    "orientation_error_change_rpy_deg": (
                        error_change_rpy[index].tolist()
                    ),
                    "orientation_error_change_inf_norm_deg": float(
                        error_change_inf_norm[index]
                    ),
                    "orientation_error_change_rate_deg_per_sec": float(rate),
                    "selected_for_acceptance": index in selected_set,
                    "selection_order": selection_order.get(index),
                }
                for index, rate in enumerate(period_rates)
            ]
        else:
            selection_method = "unavailable_due_to_boundary_interpolation"
            selected_indices = []
    else:
        selection_method = "no_positive_stability_duration"
        selected_indices = []

    selected_periods = (
        [period_rows[index] for index in selected_indices] if period_rows else []
    )
    selected_rates = np.asarray(
        [
            row["orientation_error_change_rate_deg_per_sec"]
            for row in period_rows
            if row["selected_for_acceptance"]
        ],
        dtype=np.float64,
    )
    stability_value = (
        float(np.max(selected_rates)) if len(selected_rates) else None
    )
    stability_status = _status(
        stability_value,
        STABILITY_THRESHOLD_DEG_PER_SEC,
        valid=len(selected_rates) > 0,
    )

    metrics = {
        "pairing": {
            "candidate_estimate_samples": int(len(candidate)),
            "paired_samples": int(len(timestamps)),
            "rejected_samples": int(len(candidate) - len(timestamps)),
            "maximum_reference_bracket_sec": float(np.max(brackets)),
            "configured_maximum_reference_bracket_sec": float(
                maximum_reference_bracket_sec
            ),
            "timestamp_method": (
                "reference SLERP/linear interpolation at estimate timestamps"
            ),
            "elapsed_time_method": (
                "measured synchronized wall time scaled onto sensor timestamps"
            ),
            "wall_time_per_sensor_time_scale": elapsed_time_scale,
            "alignment": "rigid SE(3), no scale",
        },
        "position_accuracy": {
            "primary_definition": "terminal position error / reference distance D",
            "terminal_error_m": terminal_error,
            "terminal_reference_position_xyz_m": reference_positions[-1].tolist(),
            "terminal_estimate_position_xyz_m": (
                aligned_estimate_positions[-1].tolist()
            ),
            "motion_distance_m": path_length,
            "motion_duration_sec": motion_duration_sec,
            "average_motion_speed_m_per_sec": average_motion_speed_mps,
            "localization_accuracy_fraction": terminal_fraction,
            "localization_accuracy_percent": (
                100.0 * terminal_fraction if terminal_fraction is not None else None
            ),
            "reference_distance_m": path_length,
            "terminal_error_over_distance_fraction": terminal_fraction,
            "terminal_error_over_distance_percent": (
                100.0 * terminal_fraction
                if terminal_fraction is not None
                else None
            ),
            "threshold_fraction": POSITION_THRESHOLD_FRACTION,
            "threshold_percent": 100.0 * POSITION_THRESHOLD_FRACTION,
            "supplementary_ate_rmse_m": position_rmse,
            "supplementary_rmse_over_distance_percent": (
                100.0 * rmse_fraction if rmse_fraction is not None else None
            ),
            "error_m": series_statistics(position_error),
            "document_note": (
                "The procedure paragraph mentions ATE RMSE/D, while the explicit "
                "formula defines terminal error e_T/D. The explicit formula is "
                "primary and RMSE/D is retained as a supplementary result."
            ),
            "status": _status(
                terminal_fraction,
                POSITION_THRESHOLD_FRACTION,
                valid=path_length > 0.0,
            ),
        },
        "orientation_stability": {
            "definition": (
                "max selected infinity norm of the XYZ (roll, pitch, yaw) "
                "difference between estimated and reference orientation-change "
                "vectors over each interval; complete periods use 500 s and an "
                "incomplete final period uses its actual runtime as denominator"
            ),
            "euler_convention": "R = Rz(yaw) * Ry(pitch) * Rx(roll); vector order XYZ",
            "period_sec": STABILITY_PERIOD_SEC,
            "period_count": period_count,
            "complete_period_count": complete_period_count,
            "partial_period_count": partial_period_count,
            "evaluated_duration_sec": stability_duration_sec,
            "selection_method": selection_method,
            "selection_seed": DOCUMENT_SAMPLE_SEED,
            "selected_period_count": int(len(selected_rates)),
            "selected_max_deg_per_sec": stability_value,
            "threshold_deg_per_sec": STABILITY_THRESHOLD_DEG_PER_SEC,
            "periods": period_rows,
            "selected_periods": selected_periods,
            "status": stability_status,
        },
    }
    samples = {
        "timestamp_sec": timestamps,
        "elapsed_runtime_sec": elapsed_runtime_sec,
        "estimate_x_m": aligned_estimate_positions[:, 0],
        "estimate_y_m": aligned_estimate_positions[:, 1],
        "estimate_z_m": aligned_estimate_positions[:, 2],
        "reference_x_m": reference_positions[:, 0],
        "reference_y_m": reference_positions[:, 1],
        "reference_z_m": reference_positions[:, 2],
        "position_error_m": position_error,
        "reference_bracket_sec": brackets,
    }
    return metrics, samples


def evaluate_runtime_metrics(
    summary_path: Path, *, time_statistic: str
) -> dict[str, Any]:
    if not summary_path.exists():
        unavailable = {
            "status": "unavailable",
            "reason": f"missing {summary_path.name}",
        }
        return {
            "position_processing_time": unavailable,
            "map_update_time": unavailable,
            "tracking": unavailable,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    measurements = summary.get("test_metric_measurements")

    def formal_time_metric(
        key: str,
        definition: str,
        start_timestamp_key: str,
        completion_timestamp_key: str,
    ) -> dict[str, Any]:
        if not isinstance(measurements, dict):
            return {
                "definition": definition,
                "status": "insufficient_data",
                "reason": (
                    "legacy summary has aggregate stage statistics but no exact "
                    "per-valid-frame measurement or random-10 sample"
                ),
            }
        sampling = measurements.get("document_random_sample", {})
        samples = sampling.get("samples", [])
        start_timestamp_sec_key = start_timestamp_key.removesuffix(
            "_ns"
        ) + "_sec"
        completion_timestamp_sec_key = completion_timestamp_key.removesuffix(
            "_ns"
        ) + "_sec"

        def nanoseconds_to_seconds(value: Any) -> float | None:
            return None if value is None else float(value) * 1.0e-9

        selected_samples = []
        for sample_order, row in enumerate(samples, start=1):
            if row.get(key) is None:
                continue
            selected_samples.append(
                {
                    "sample_order": sample_order,
                    "valid_frame_index": row.get("valid_frame_index"),
                    "sensor_stamp_sec": nanoseconds_to_seconds(
                        row.get("sensor_stamp_ns")
                    ),
                    start_timestamp_sec_key: nanoseconds_to_seconds(
                        row.get(start_timestamp_key)
                    ),
                    completion_timestamp_sec_key: nanoseconds_to_seconds(
                        row.get(completion_timestamp_key)
                    ),
                    key: float(row[key]),
                }
            )
        values = [row[key] for row in selected_samples]
        selected_max = max(values) if values else None
        endpoints_available = bool(selected_samples) and all(
            row[start_timestamp_sec_key] is not None
            and row[completion_timestamp_sec_key] is not None
            for row in selected_samples
        )
        excluded_cold_start_frames = int(
            sampling.get("excluded_initial_valid_frame_count", 0)
        )
        sampling_scope = (
            " after excluding the first cold-start valid frame"
            if excluded_cold_start_frames
            else ""
        )
        enough_samples = len(values) == DOCUMENT_SAMPLE_SIZE
        status = _status(
            selected_max,
            TIME_THRESHOLD_SEC,
            valid=enough_samples,
        )
        return {
            "definition": definition,
            "acceptance_method": (
                "Word procedure: randomly select 10 valid-frame values"
                f"{sampling_scope}; all selected values must be <= 2 s"
            ),
            "excluded_initial_valid_frame_count": excluded_cold_start_frames,
            "threshold_sec": TIME_THRESHOLD_SEC,
            "population_count": sampling.get("population_count"),
            "sample_method": sampling.get("method"),
            "sample_seed": sampling.get("seed"),
            "sample_count": len(values),
            "selected_values_sec": values,
            "selected_samples": selected_samples,
            "start_timestamp_field": start_timestamp_sec_key,
            "completion_timestamp_field": completion_timestamp_sec_key,
            "endpoint_clock": "process monotonic clock",
            "endpoint_timestamps_available": endpoints_available,
            "endpoint_unavailable_reason": (
                None
                if endpoints_available
                else (
                    "input_processing_summary.json does not store both process-"
                    "monotonic endpoint timestamps; sensor_stamp_sec belongs "
                    "to the sensor/bag clock and is not substituted"
                )
            ),
            "selected_max_sec": selected_max,
            "all_selected_at_or_below_threshold": (
                all(value <= TIME_THRESHOLD_SEC for value in values)
                if values
                else None
            ),
            "all_valid_frame_statistics": measurements.get(key),
            "status": status,
            "reason": (
                None
                if enough_samples
                else f"requires exactly {DOCUMENT_SAMPLE_SIZE} valid samples"
            ),
        }

    position = formal_time_metric(
        "pose_publish_latency_sec",
        "synchronized mapper callback entry to return from _publish_pose",
        "localization_solution_start_monotonic_ns",
        "pose_publish_completion_monotonic_ns",
    )
    map_update = formal_time_metric(
        "map_update_latency_sec",
        "map-update stage entry to corresponding map update/publish return",
        "map_update_start_monotonic_ns",
        "map_publish_completion_monotonic_ns",
    )
    mapper_results = summary.get("mapper_results", {})
    callbacks = int(summary.get("synchronized_callbacks", 0))
    valid_poses = int(mapper_results.get("valid_pose_count", 0))
    tracking = {
        "synchronized_callbacks": callbacks,
        "valid_pose_count": valid_poses,
        "failure_count": int(
            mapper_results.get("tracking_or_processing_failure_count", 0)
        ),
        "success_percent": 100.0 * valid_poses / callbacks if callbacks else None,
        "sensor_span_sec": summary.get("synchronized_sensor_span_sec"),
        "status": "available",
    }
    return {
        "position_processing_time": position,
        "map_update_time": map_update,
        "tracking": tracking,
        "legacy_time_statistic_argument": time_statistic,
    }


def evaluate_map_metrics(
    output_dir: Path, *, motion_distance_m: float | None = None
) -> dict[str, Any]:
    npz_path = output_dir / "global_grid_map.npz"
    sqlite_path = output_dir / "global_grid_map.sqlite3"
    if not npz_path.exists() and not sqlite_path.exists():
        return {
            "map_range": {"status": "unavailable", "reason": "no global map file"},
            "storage": {"status": "unavailable", "reason": "no global map file"},
        }

    resolution = None
    width = height = None
    dense_area_km2 = None
    known_cells = None
    coverage_bbox_km2 = None
    map_revision = None
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as values:
            resolution = float(values["resolution"])
            width = int(values["width"])
            height = int(values["height"])
            dense_area_km2 = (
                float(values["length_x"]) * float(values["length_y"]) / 1e6
            )
            map_revision = int(values["map_revision"])
            known_layers = []
            if "occupancy" in values:
                known_layers.append(np.asarray(values["occupancy"]) >= 0)
            if "observation_count" in values:
                known_layers.append(
                    np.asarray(values["observation_count"]) > 0
                )
            if "elevation" in values:
                known_layers.append(
                    np.isfinite(np.asarray(values["elevation"]))
                )
            known = (
                np.logical_or.reduce(known_layers)
                if known_layers
                else None
            )
            if known is not None:
                known_cells = int(np.count_nonzero(known))
                indices = np.argwhere(known)
                if len(indices):
                    bbox_height = int(indices[:, 0].max() - indices[:, 0].min() + 1)
                    bbox_width = int(indices[:, 1].max() - indices[:, 1].min() + 1)
                    coverage_bbox_km2 = (
                        bbox_width * bbox_height * resolution * resolution / 1e6
                    )
    elif sqlite_path.exists():
        with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        resolution = float(metadata["resolution_m"])
        map_revision = int(metadata.get("map_revision", 0))
        known_cells = int(metadata.get("known_cell_count", 0))

    known_area_km2 = (
        known_cells * resolution * resolution / 1e6
        if known_cells is not None and resolution is not None
        else None
    )
    primary_area = known_area_km2
    range_status = _status_at_least(
        primary_area,
        MAP_RANGE_THRESHOLD_KM2,
        valid=primary_area is not None,
    )
    map_roots = (
        npz_path,
        sqlite_path,
        output_dir / "_internal" / "grid_map_snapshot",
        output_dir / "_internal" / "map_observation_archive",
    )
    map_files: list[Path] = []
    for root in map_roots:
        if root.is_file():
            map_files.append(root)
        elif root.is_dir():
            map_files.extend(path for path in root.rglob("*") if path.is_file())
    file_sizes = {
        str(path.relative_to(output_dir)): path.stat().st_size
        for path in sorted(set(map_files))
    }
    all_map_bytes = int(sum(file_sizes.values()))
    published_map_bytes = int(
        sum(
            path.stat().st_size
            for path in (npz_path, sqlite_path)
            if path.exists()
        )
    )
    compact_delivery_bytes = (
        int(sqlite_path.stat().st_size) if sqlite_path.exists() else None
    )
    storage_rate = (
        all_map_bytes / 1e9 / primary_area
        if primary_area is not None and primary_area > 0.0
        else None
    )
    published_storage_rate = (
        published_map_bytes / 1e9 / primary_area
        if primary_area is not None and primary_area > 0.0
        else None
    )
    compact_storage_rate = (
        compact_delivery_bytes / 1e9 / primary_area
        if compact_delivery_bytes is not None
        and primary_area is not None
        and primary_area > 0.0
        else None
    )
    storage_valid = primary_area is not None and primary_area > 0.0
    return {
        "map_range": {
            "primary_definition": "valid observed cell union: N_cell * r^2",
            "primary_area_km2": primary_area,
            "observed_cell_area_km2": known_area_km2,
            "observed_cell_bbox_area_km2": coverage_bbox_km2,
            "dense_matrix_extent_km2": dense_area_km2,
            "threshold_km2": MAP_RANGE_THRESHOLD_KM2,
            "resolution_m": resolution,
            "width_cells": width,
            "height_cells": height,
            "motion_distance_m": motion_distance_m,
            "known_cells": known_cells,
            "map_revision": map_revision,
            "status": range_status,
        },
        "storage": {
            "primary_definition": (
                "sum of every retained map-data file in the output directory, "
                "divided by valid observed-cell area"
            ),
            "all_map_data_files_bytes": all_map_bytes,
            "published_map_artifacts_bytes": published_map_bytes,
            "compact_sqlite_delivery_bytes": compact_delivery_bytes,
            "map_data_file_bytes": file_sizes,
            "normalized_gb_per_km2": storage_rate,
            "published_map_artifacts_gb_per_km2": published_storage_rate,
            "compact_sqlite_delivery_gb_per_km2": compact_storage_rate,
            "threshold_gb_per_km2": STORAGE_THRESHOLD_GB_PER_KM2,
            "normalization_area_km2": primary_area,
            "status": _status(
                storage_rate, STORAGE_THRESHOLD_GB_PER_KM2, valid=storage_valid
            ),
        },
    }


def _write_samples(path: Path, samples: dict[str, np.ndarray]) -> None:
    keys = list(samples)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(samples[key] for key in keys)))


def _metric_line(name: str, metric: dict[str, Any], value: str) -> str:
    return f"| {name} | {value} | {metric.get('status', 'unavailable')} |"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    reference = report["reference"]
    elapsed_time_basis = report["elapsed_time_basis"]
    metrics = report["metrics"]
    position = metrics["position_accuracy"]
    stability = metrics["orientation_stability"]
    runtime = metrics["runtime"]
    map_metrics = metrics["map"]
    lines = [
        "# Task-3 all-metrics report",
        "",
        f"- Output directory: `{report['output_directory']}`",
        f"- Reference status: `{reference.get('status')}`",
        f"- Reference note: {reference.get('reason', reference.get('warning', 'available'))}",
        f"- Elapsed-time source: {elapsed_time_basis.get('source')}",
        (
            f"- Synchronized sensor span: "
            f"{elapsed_time_basis.get('synchronized_sensor_span_sec')} s"
        ),
        (
            f"- Synchronized wall-clock span: "
            f"{elapsed_time_basis.get('synchronized_wall_span_sec')} s"
        ),
        f"- Effective consumption rate: {elapsed_time_basis.get('effective_consumption_rate_x')}x",
        "",
        "| Metric | Result | Status |",
        "|---|---:|---|",
        _metric_line(
            "Position RMSE / D",
            position,
            (
                f"{position['terminal_error_over_distance_percent']:.6f}%"
                if position.get("terminal_error_over_distance_percent") is not None
                else "N/A"
            ),
        ),
        _metric_line(
            "3D orientation stability (500 s periods)",
            stability,
            (
                f"{stability['selected_max_deg_per_sec']:.6f} deg/s"
                if stability.get("selected_max_deg_per_sec") is not None
                else "N/A"
            ),
        ),
        _metric_line(
            "Position processing time",
            runtime["position_processing_time"],
            (
                f"{runtime['position_processing_time']['selected_max_sec']:.6f} s"
                if runtime["position_processing_time"].get("selected_max_sec") is not None
                else "N/A"
            ),
        ),
        _metric_line(
            "Map update time",
            runtime["map_update_time"],
            (
                f"{runtime['map_update_time']['selected_max_sec']:.6f} s"
                if runtime["map_update_time"].get("selected_max_sec") is not None
                else "N/A"
            ),
        ),
        _metric_line(
            "Map range",
            map_metrics["map_range"],
            (
                f"{map_metrics['map_range']['primary_area_km2']:.6f} km^2"
                if map_metrics["map_range"].get("primary_area_km2") is not None
                else "N/A"
            ),
        ),
        _metric_line(
            "Map storage",
            map_metrics["storage"],
            (
                f"{map_metrics['storage']['normalized_gb_per_km2']:.6f} GB/km^2"
                if map_metrics["storage"].get("normalized_gb_per_km2") is not None
                else "N/A"
            ),
        ),
        "",
        "`insufficient_data` means a numeric pre-test result exists but the formal duration/range requirement was not met.",
        "`unavailable` means the required source data does not exist and no substitute was used.",
    ]
    selected_periods = stability.get("selected_periods", [])
    lines.extend(
        [
            "",
            "## Orientation stability sampled periods",
            "",
            (
                "XYZ vectors are `[roll, pitch, yaw]` in degrees. Complete periods "
                "use a 500 s denominator; a partial period uses its actual runtime."
            ),
            "",
        ]
    )
    if selected_periods:
        lines.extend(
            [
                "| Sample | Period | Reference change XYZ (deg) | "
                "Estimate change XYZ (deg) | Difference XYZ (deg) | "
                "Infinity norm (deg) | Motion time (s) | Rate (deg/s) |",
                "|---:|---:|---|---|---|---:|---:|---:|",
            ]
        )
        for row in selected_periods:
            vector = lambda values: "[" + ", ".join(  # noqa: E731
                f"{float(value):.6f}" for value in values
            ) + "]"
            lines.append(
                f"| {row['selection_order']} | {row['period_index']} | "
                f"{vector(row['reference_orientation_change_rpy_deg'])} | "
                f"{vector(row['estimate_orientation_change_rpy_deg'])} | "
                f"{vector(row['reference_to_estimate_orientation_change_error_rpy_deg'])} | "
                f"{row['reference_to_estimate_orientation_change_inf_norm_deg']:.6f} | "
                f"{row['motion_time_sec']:.6f} | "
                f"{row['orientation_error_change_rate_deg_per_sec']:.9f} |"
            )
    else:
        lines.append("No orientation-stability period was available for sampling.")

    def vector_or_na(values: Any) -> str:
        if not isinstance(values, list):
            return "N/A"
        return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"

    def number_or_na(value: Any, suffix: str = "") -> str:
        return f"{float(value):.6f}{suffix}" if value is not None else "N/A"

    lines.extend(
        [
            "",
            "## Position accuracy details",
            "",
            f"- Terminal reference position XYZ (m): {vector_or_na(position.get('terminal_reference_position_xyz_m'))}",
            f"- Terminal estimate position XYZ (m): {vector_or_na(position.get('terminal_estimate_position_xyz_m'))}",
            f"- Motion distance: {number_or_na(position.get('motion_distance_m'), ' m')}",
            f"- Motion duration: {number_or_na(position.get('motion_duration_sec'), ' s')}",
            f"- Average motion speed: {number_or_na(position.get('average_motion_speed_m_per_sec'), ' m/s')}",
            f"- Localization accuracy: {number_or_na(position.get('localization_accuracy_percent'), '%')}",
        ]
    )

    def append_time_samples(title: str, metric: dict[str, Any], key: str) -> None:
        start_key = metric.get("start_timestamp_field", "start_timestamp")
        completion_key = metric.get(
            "completion_timestamp_field", "completion_timestamp"
        )
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "Endpoint timestamps use the process monotonic clock. `N/A` means "
                "the run summary did not record that endpoint; sensor time is not substituted.",
                "",
                "| Sample | Valid frame | Sensor stamp (s) | Start monotonic (s) | "
                "Publish completion monotonic (s) | Processing time (s) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected_samples = metric.get("selected_samples", [])
        if not selected_samples:
            lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
            return
        for row in selected_samples:
            sensor_stamp = row.get("sensor_stamp_sec")
            start_timestamp = row.get(start_key)
            completion_timestamp = row.get(completion_key)
            lines.append(
                f"| {row['sample_order']} | {row.get('valid_frame_index', 'N/A')} | "
                f"{f'{sensor_stamp:.9f}' if sensor_stamp is not None else 'N/A'} | "
                f"{f'{start_timestamp:.9f}' if start_timestamp is not None else 'N/A'} | "
                f"{f'{completion_timestamp:.9f}' if completion_timestamp is not None else 'N/A'} | "
                f"{float(row[key]):.6f} |"
            )

    append_time_samples(
        "Position processing sampled frames",
        runtime["position_processing_time"],
        "pose_publish_latency_sec",
    )
    map_range = map_metrics["map_range"]
    lines.extend(
        [
            "",
            "## Map construction details",
            "",
            f"- Grid resolution: {number_or_na(map_range.get('resolution_m'), ' m')}",
            f"- Valid cell count: {map_range.get('known_cells', 'N/A')}",
            f"- Valid coverage area: {number_or_na(map_range.get('primary_area_km2'), ' km^2')}",
            f"- Motion distance: {number_or_na(map_range.get('motion_distance_m'), ' m')}",
        ]
    )
    append_time_samples(
        "Map update sampled frames",
        runtime["map_update_time"],
        "map_update_latency_sec",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_output_artifacts(output_dir: Path) -> dict[str, Any]:
    def trajectory_audit(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"status": "missing", "path": str(path)}
        timestamps = []
        invalid_rows = 0
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 8:
                invalid_rows += 1
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                invalid_rows += 1
                continue
            if not all(math.isfinite(value) for value in values):
                invalid_rows += 1
                continue
            timestamps.append(values[0])
        timestamp_values = np.asarray(timestamps, dtype=np.float64)
        strictly_increasing = bool(
            len(timestamp_values) >= 2
            and np.all(np.diff(timestamp_values) > 0.0)
        )
        return {
            "status": "available",
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "valid_rows": len(timestamps),
            "invalid_rows": invalid_rows,
            "first_timestamp_sec": (
                float(timestamp_values[0]) if len(timestamp_values) else None
            ),
            "last_timestamp_sec": (
                float(timestamp_values[-1]) if len(timestamp_values) else None
            ),
            "unique_timestamp_count": int(len(np.unique(timestamp_values))),
            "strictly_increasing_timestamps": strictly_increasing,
            "usable_for_time_pairing": strictly_increasing,
        }

    summary_path = output_dir / "input_processing_summary.json"
    summary_audit: dict[str, Any]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_audit = {
            "status": "available",
            "path": str(summary_path),
            "schema_version": summary.get("schema_version"),
            "synchronized_callbacks": summary.get("synchronized_callbacks"),
            "valid_pose_count": summary.get("mapper_results", {}).get(
                "valid_pose_count"
            ),
            "has_exact_valid_frame_metrics": isinstance(
                summary.get("test_metric_measurements"), dict
            ),
        }
    else:
        summary_audit = {"status": "missing", "path": str(summary_path)}

    formal_trajectory = trajectory_audit(output_dir / "trajectory_map.tum")
    internal_trajectory = trajectory_audit(
        output_dir / "_internal" / "trajectory_opencv.tum"
    )
    return {
        "formal_map_trajectory": formal_trajectory,
        "internal_opencv_trajectory": internal_trajectory,
        "input_processing_summary": summary_audit,
        "formal_trajectory_row_count_matches_valid_pose_count": (
            formal_trajectory.get("valid_rows")
            == summary_audit.get("valid_pose_count")
            if formal_trajectory.get("valid_rows") is not None
            and summary_audit.get("valid_pose_count") is not None
            else None
        ),
        "global_grid_map_npz": {
            "exists": (output_dir / "global_grid_map.npz").exists(),
            "path": str(output_dir / "global_grid_map.npz"),
        },
        "global_grid_map_sqlite3": {
            "exists": (output_dir / "global_grid_map.sqlite3").exists(),
            "path": str(output_dir / "global_grid_map.sqlite3"),
        },
        "grid_map_snapshot": {
            "exists": (output_dir / "_internal" / "grid_map_snapshot").is_dir(),
            "path": str(output_dir / "_internal" / "grid_map_snapshot"),
        },
        "map_observation_archive": {
            "exists": (
                output_dir / "_internal" / "map_observation_archive"
            ).is_dir(),
            "path": str(
                output_dir / "_internal" / "map_observation_archive"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Completed mapper output directory")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--bag", type=Path, help="Bag containing PQ reference data")
    reference.add_argument("--reference-tum", type=Path)
    parser.add_argument("--pq-topic", default=DEFAULT_PQ_TOPIC)
    parser.add_argument(
        "--stamped-pq-topic",
        default=DEFAULT_STAMPED_PQ_TOPIC,
        help=(
            "Internal stamped PQ topic emitted by the custom bag player; "
            "live sensors may publish only --pq-topic"
        ),
    )
    parser.add_argument(
        "--record-reference-data",
        action="store_true",
        help=(
            "Run as a passive PQ recorder and save raw reference observations "
            "inside output_dir instead of evaluating metrics"
        ),
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=0,
        help="In recorder mode, stop when this launcher PID exits",
    )
    parser.add_argument("--pq-position-indices", nargs=3, type=int, default=(1, 2, 3))
    parser.add_argument("--pq-rpy-indices", nargs=3, type=int, default=(4, 5, 6))
    parser.add_argument(
        "--pq-rpy-signs",
        nargs=3,
        type=float,
        choices=(-1.0, 1.0),
        default=DEFAULT_PQ_RPY_SIGNS,
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Polarity applied to raw PQ roll/pitch/yaw fields before Euler "
            "conversion. Default: -1 +1 +1, validated against WIT roll."
        ),
    )
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--max-reference-bracket-sec", type=float, default=0.5)
    parser.add_argument(
        "--time-statistic",
        choices=("mean", "recent_p95", "max"),
        default="mean",
        help="Statistic used for the two <=2 s acceptance decisions.",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.record_reference_data:
        return record_reference_data(
            output_dir,
            pq_topic=args.pq_topic,
            stamped_pq_topic=args.stamped_pq_topic,
            parent_pid=args.parent_pid,
        )

    estimate_path = output_dir / "trajectory_map.tum"
    summary_path = output_dir / "input_processing_summary.json"
    elapsed_time_basis = _elapsed_time_basis(summary_path)
    elapsed_time_scale = float(
        elapsed_time_basis["wall_time_per_sensor_time_scale"]
    )
    reference_trajectory = None
    requested_pq_rpy_signs = tuple(float(value) for value in args.pq_rpy_signs)
    if args.reference_tum is not None:
        reference_trajectory = load_tum(args.reference_tum.resolve())
        reference_metadata: dict[str, Any] = {
            "status": "available",
            "source": "TUM trajectory",
            "path": str(args.reference_tum.resolve()),
        }
    elif args.bag is not None:
        pq_tum = output_dir / "pq_reference.tum"
        reference_trajectory, reference_metadata = extract_pq_reference(
            args.bag.resolve(),
            args.pq_topic,
            pq_tum,
            position_indices=tuple(args.pq_position_indices),
            rpy_indices=tuple(args.pq_rpy_indices),
            rpy_signs=requested_pq_rpy_signs,
        )
        reference_metadata["bag"] = str(args.bag.resolve())
        (output_dir / "pq_reference_metadata.json").write_text(
            json.dumps(_json_value(reference_metadata), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    elif (output_dir / "pq_reference.tum").exists():
        # A reference extracted directly from the rosbag carries the bag's
        # authoritative record timestamps.  Prefer it over passively recorded
        # Float64MultiArray samples, whose fallback timestamps are replay-time
        # wall-clock values when no stamped PQ topic was published.
        pq_tum = output_dir / "pq_reference.tum"
        reference_trajectory = load_tum(pq_tum)
        pq_metadata = output_dir / "pq_reference_metadata.json"
        if pq_metadata.exists():
            loaded_metadata = json.loads(
                pq_metadata.read_text(encoding="utf-8")
            )
            reference_metadata = (
                loaded_metadata
                if isinstance(loaded_metadata, dict)
                else {}
            )
        else:
            reference_metadata = {}
        reference_metadata.setdefault("status", "available")
        reference_metadata.setdefault(
            "source", "pre-extracted rosbag PQ trajectory"
        )
        reference_metadata["output_tum"] = str(pq_tum.resolve())
        stored_pq_rpy_signs = tuple(
            float(value)
            for value in reference_metadata.get(
                "rpy_signs_applied", (1.0, 1.0, 1.0)
            )
        )
        if stored_pq_rpy_signs != requested_pq_rpy_signs:
            correction_signs = tuple(
                requested / stored
                for requested, stored in zip(
                    requested_pq_rpy_signs, stored_pq_rpy_signs
                )
            )
            reference_trajectory = _apply_rpy_signs_to_trajectory(
                reference_trajectory, correction_signs
            )
            reference_metadata["stored_rpy_signs_applied"] = list(
                stored_pq_rpy_signs
            )
            reference_metadata["in_memory_rpy_sign_correction"] = list(
                correction_signs
            )
        reference_metadata["effective_rpy_signs_applied"] = list(
            requested_pq_rpy_signs
        )
        reference_metadata["rpy_sign_evidence"] = (
            "PQ roll polarity is opposite to ROMA and independent WIT roll "
            "over the 2026-08-24 validation bag; configurable by --pq-rpy-signs"
        )
    elif (output_dir / REFERENCE_SAMPLE_RELATIVE_PATH).exists():
        reference_tum = output_dir / "reference.tum"
        reference_trajectory, reference_metadata = extract_saved_pq_reference(
            output_dir / REFERENCE_SAMPLE_RELATIVE_PATH,
            reference_tum,
            position_indices=tuple(args.pq_position_indices),
            rpy_indices=tuple(args.pq_rpy_indices),
            rpy_signs=requested_pq_rpy_signs,
        )
        (output_dir / "reference_metadata.json").write_text(
            json.dumps(_json_value(reference_metadata), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    else:
        reference_metadata = {
            "status": "unavailable",
            "reason": (
                "no --reference-tum/--bag and no saved reference PQ data "
                f"at {output_dir / REFERENCE_SAMPLE_RELATIVE_PATH}"
            ),
        }

    unavailable_reference_metric = {
        "status": "unavailable",
        "reason": reference_metadata.get("reason", "reference unavailable"),
    }
    position_metric = dict(unavailable_reference_metric)
    stability_metric = dict(unavailable_reference_metric)
    if reference_trajectory is not None and estimate_path.exists():
        try:
            reference_metrics, samples = evaluate_reference_metrics(
                load_tum(estimate_path),
                reference_trajectory,
                warmup_sec=args.warmup_sec,
                maximum_reference_bracket_sec=args.max_reference_bracket_sec,
                elapsed_time_scale=elapsed_time_scale,
            )
            position_metric = reference_metrics["position_accuracy"]
            stability_metric = reference_metrics["orientation_stability"]
            reference_metadata["pairing"] = reference_metrics["pairing"]
            _write_samples(output_dir / "trajectory_comparison.csv", samples)
        except ValueError as error:
            position_metric = {"status": "insufficient_data", "reason": str(error)}
            stability_metric = {"status": "insufficient_data", "reason": str(error)}
    elif not estimate_path.exists():
        reason = f"missing {estimate_path.name}"
        position_metric = {"status": "unavailable", "reason": reason}
        stability_metric = {"status": "unavailable", "reason": reason}

    report = {
        "schema_version": 2,
        "output_directory": str(output_dir),
        "estimate_trajectory": str(estimate_path),
        "output_artifact_audit": audit_output_artifacts(output_dir),
        "elapsed_time_basis": elapsed_time_basis,
        "reference": reference_metadata,
        "acceptance_thresholds": {
            "terminal_position_error_over_distance_percent": 1.0,
            "orientation_stability_deg_per_sec": STABILITY_THRESHOLD_DEG_PER_SEC,
            "orientation_stability_period_sec": STABILITY_PERIOD_SEC,
            "position_processing_time_sec": TIME_THRESHOLD_SEC,
            "map_update_time_sec": TIME_THRESHOLD_SEC,
            "map_range_km2": MAP_RANGE_THRESHOLD_KM2,
            "map_storage_gb_per_km2": STORAGE_THRESHOLD_GB_PER_KM2,
        },
        "metrics": {
            "position_accuracy": position_metric,
            "orientation_stability": stability_metric,
            "runtime": evaluate_runtime_metrics(
                summary_path, time_statistic=args.time_statistic,
            ),
            "map": evaluate_map_metrics(
                output_dir,
                motion_distance_m=position_metric.get("motion_distance_m"),
            ),
        },
        "notes": [
            "Reference-dependent metrics are never replaced by estimator self-drift.",
            (
                "Saved PQ uses exact publisher record time when available; "
                "live Float64MultiArray otherwise uses subscriber reception time."
            ),
            (
                "Trajectory pairing stays in sensor time, while reported elapsed "
                "durations and stability periods use measured wall-clock time."
            ),
            "Position and map update times use the Word random-10 rule.",
            "Map area uses valid observed cells; storage includes every retained map-data file.",
            (
                "Absolute processing endpoint timestamps are reported only when "
                "present in the run summary and are never reconstructed from sensor time."
            ),
        ],
    }
    report = _json_value(report)
    report_json = args.report_json or output_dir / "all_metrics_report.json"
    report_md = args.report_md or output_dir / "all_metrics_report.md"
    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(report_md, report)
    print(f"JSON report: {report_json}")
    print(f"Markdown report: {report_md}")
    for key in (
        "position_accuracy",
        "orientation_stability",
    ):
        print(f"{key}: {report['metrics'][key]['status']}")
    print(
        "position_processing_time: "
        f"{report['metrics']['runtime']['position_processing_time']['status']}"
    )
    print(
        "map_update_time: "
        f"{report['metrics']['runtime']['map_update_time']['status']}"
    )
    print(f"map_range: {report['metrics']['map']['map_range']['status']}")
    print(f"storage: {report['metrics']['map']['storage']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
