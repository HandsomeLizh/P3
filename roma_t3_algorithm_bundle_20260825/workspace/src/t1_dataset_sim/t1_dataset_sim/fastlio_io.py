from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class OrganizedFastLioOdometryRecord:
    timestamp_ns: int
    position: np.ndarray
    orientation_xyzw: np.ndarray
    pose_covariance: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    twist_covariance: np.ndarray


def _optional_array(
    archive,
    key: str,
    *,
    count: int,
    tail_shape: tuple[int, ...],
) -> np.ndarray:
    if key not in archive:
        return np.zeros((count,) + tail_shape, dtype=np.float64)
    values = np.asarray(archive[key], dtype=np.float64)
    expected = (count,) + tail_shape
    if values.shape != expected:
        raise ValueError(
            f"FAST-LIO trajectory field {key!r} has shape "
            f"{values.shape}, expected {expected}"
        )
    return values


def load_organized_fastlio_odometry(
    data_dir: Path,
    *,
    trajectory_path: Optional[Path] = None,
    first_timestamp_ns: Optional[int] = None,
    last_timestamp_ns: Optional[int] = None,
    stride: int = 1,
) -> List[OrganizedFastLioOdometryRecord]:
    """Load saved FAST-LIO output for deterministic ROS interface testing.

    These records are estimator output, not ground truth.  Production and
    rosbag operation should run an open-source FAST-LIO2 node from raw LiDAR
    and IMU instead of feeding this archive back into accuracy evaluation.
    """

    if int(stride) <= 0:
        raise ValueError("FAST-LIO odometry stride must be positive")
    path = (
        Path(trajectory_path).expanduser()
        if trajectory_path is not None
        else Path(data_dir) / "lidar_fastlio" / "trajectory.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"FAST-LIO trajectory does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        time_key = (
            "header_time_ns"
            if "header_time_ns" in archive
            else "record_time_ns"
        )
        required = {time_key, "position", "orientation"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"FAST-LIO trajectory is missing fields: {sorted(missing)}"
            )
        timestamps = np.asarray(archive[time_key], dtype=np.int64).reshape(-1)
        count = len(timestamps)
        position = _optional_array(
            archive, "position", count=count, tail_shape=(3,)
        )
        orientation = _optional_array(
            archive, "orientation", count=count, tail_shape=(4,)
        )
        pose_covariance = _optional_array(
            archive, "pose_covariance", count=count, tail_shape=(6, 6)
        )
        linear_velocity = _optional_array(
            archive, "linear_velocity", count=count, tail_shape=(3,)
        )
        angular_velocity = _optional_array(
            archive, "angular_velocity", count=count, tail_shape=(3,)
        )
        twist_covariance = _optional_array(
            archive, "twist_covariance", count=count, tail_shape=(6, 6)
        )

    if count == 0:
        return []
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("FAST-LIO trajectory timestamps are not increasing")
    finite = (
        np.isfinite(position).all(axis=1)
        & np.isfinite(orientation).all(axis=1)
        & np.isfinite(pose_covariance).all(axis=(1, 2))
        & np.isfinite(linear_velocity).all(axis=1)
        & np.isfinite(angular_velocity).all(axis=1)
        & np.isfinite(twist_covariance).all(axis=(1, 2))
    )
    quaternion_norm = np.linalg.norm(orientation, axis=1)
    finite &= quaternion_norm > 1e-8
    selected = np.flatnonzero(finite)
    if first_timestamp_ns is not None:
        selected = selected[timestamps[selected] >= int(first_timestamp_ns)]
    if last_timestamp_ns is not None:
        selected = selected[timestamps[selected] <= int(last_timestamp_ns)]
    selected = selected[:: int(stride)]

    records = []
    for index in selected:
        quaternion = orientation[index] / quaternion_norm[index]
        records.append(
            OrganizedFastLioOdometryRecord(
                timestamp_ns=int(timestamps[index]),
                position=position[index].copy(),
                orientation_xyzw=quaternion.copy(),
                pose_covariance=pose_covariance[index].copy(),
                linear_velocity=linear_velocity[index].copy(),
                angular_velocity=angular_velocity[index].copy(),
                twist_covariance=twist_covariance[index].copy(),
            )
        )
    return records
