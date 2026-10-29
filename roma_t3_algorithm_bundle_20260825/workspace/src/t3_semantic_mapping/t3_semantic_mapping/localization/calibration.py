from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .pose_math import inverse_pose, validated_pose_matrix


def load_rectified_mounting_to_camera_transform(
    calibration: Mapping,
    *,
    sensor_key: str,
) -> Optional[np.ndarray]:
    """Derive a calibrated sensor-to-rectified-left-camera transform.

    Mounting matrices use ``p_vehicle = T_vehicle_sensor p_sensor`` and the
    measured vehicle frame X-forward, Y-right, Z-down.  RoMa consumes the
    rectified left image, therefore the raw left-camera transform must be
    followed by the generated ``rectification.left_r`` rotation.

    Missing mounting entries return ``None`` so legacy calibration files keep
    their existing explicit-extrinsics or pitch-only fallback behaviour.
    """

    mounting = calibration.get("mounting", {})
    if not isinstance(mounting, Mapping):
        return None
    vehicle_left_values = mounting.get("transform_vehicle_left")
    vehicle_sensor_values = mounting.get(sensor_key)
    if vehicle_left_values is None or vehicle_sensor_values is None:
        return None

    vehicle_axes = str(mounting.get("vehicle_axes", "")).strip().lower()
    if vehicle_axes != "x_forward_y_right_z_down":
        raise ValueError(
            "Mounting-derived calibration requires vehicle_axes="
            "x_forward_y_right_z_down"
        )

    vehicle_left = validated_pose_matrix(
        np.asarray(vehicle_left_values, dtype=np.float64),
        name="mounting.transform_vehicle_left",
    )
    vehicle_sensor = validated_pose_matrix(
        np.asarray(vehicle_sensor_values, dtype=np.float64),
        name=f"mounting.{sensor_key}",
    )
    raw_camera_sensor = inverse_pose(vehicle_left) @ vehicle_sensor

    rectification = calibration.get("rectification", {})
    left_rotation_values = (
        rectification.get("left_r", np.eye(3, dtype=np.float64))
        if isinstance(rectification, Mapping)
        else np.eye(3, dtype=np.float64)
    )
    rectification_pose = np.eye(4, dtype=np.float64)
    rectification_pose[:3, :3] = np.asarray(
        left_rotation_values, dtype=np.float64
    ).reshape(3, 3)
    rectification_pose = validated_pose_matrix(
        rectification_pose,
        name="rectification.left_r",
    )
    return validated_pose_matrix(
        rectification_pose @ raw_camera_sensor,
        name=f"rectified {sensor_key}-to-left-camera transform",
    )


def resolve_extrinsics_path(
    calibration: Mapping,
    configured_path: Optional[Path] = None,
) -> Path:
    source = calibration.get("source", {})
    path_text = (
        str(Path(configured_path).expanduser())
        if configured_path is not None
        else str(source.get("extrinsics_path", "")).strip()
    )
    if not path_text:
        raise FileNotFoundError(
            "LiDAR localisation/mapping requires either calibrated mounting "
            "transforms, calibration/extrinsics.json, or an explicit "
            "extrinsics path"
        )
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Extrinsics file does not exist: {path}")
    return path


def load_rectified_lidar_to_camera_transform(
    calibration: Mapping,
    *,
    extrinsics_path: Optional[Path] = None,
) -> np.ndarray:
    """Load or derive ``T_Cam_L_Rect_LiDAR`` from calibration.

    With measured vehicle mounting matrices, the transform is derived through
    the common vehicle frame.  An explicitly configured JSON path takes
    precedence for backwards compatibility.  JSON transforms map LiDAR points
    into the original left camera.  RoMa consumes rectified stereo images, so
    the left rectification rotation is composed here once and shared by both
    LIO pose adaptation and LiDAR map geometry.
    """

    if extrinsics_path is None:
        mounted = load_rectified_mounting_to_camera_transform(
            calibration,
            sensor_key="transform_vehicle_lidar",
        )
        if mounted is not None:
            return mounted

    path = resolve_extrinsics_path(calibration, extrinsics_path)
    extrinsics = json.loads(path.read_text(encoding="utf-8"))
    entry = extrinsics.get("LiDAR_to_Cam_L")
    if not isinstance(entry, dict) or "RT" not in entry:
        raise KeyError(f"LiDAR_to_Cam_L.RT is missing from {path}")

    rectification = calibration.get("rectification", {})
    if "left_r" not in rectification:
        raise KeyError(
            "LiDAR localisation/mapping requires the left-camera "
            "rectification rotation"
        )
    raw = validated_pose_matrix(
        np.asarray(entry["RT"], dtype=np.float64),
        name="LiDAR_to_Cam_L.RT",
    )
    left_rotation = np.asarray(
        rectification["left_r"], dtype=np.float64
    ).reshape(3, 3)
    rectified = np.eye(4, dtype=np.float64)
    rectified[:3, :3] = left_rotation
    return validated_pose_matrix(
        rectified @ raw,
        name="rectified LiDAR-to-camera transform",
    )


def row_major_transform(
    values,
    *,
    name: str,
) -> np.ndarray:
    """Parse an explicitly named row-major homogeneous transform.

    Keeping this conversion in one place makes the ROS parameter convention
    testable and prevents an unnoticed transpose at an external odometry
    boundary.
    """

    raw = np.asarray(values, dtype=np.float64)
    if raw.size != 16:
        raise ValueError(
            f"{name} must contain 16 row-major values, got {raw.size}"
        )
    return validated_pose_matrix(raw.reshape(4, 4), name=name)


def compose_lio_child_to_camera_transform(
    transform_camera_lidar: np.ndarray,
    transform_lidar_child: np.ndarray,
) -> np.ndarray:
    """Return the calibration mapping a LIO child frame into the camera.

    ``transform_camera_lidar`` uses ``p_camera = T_camera_lidar p_lidar``.
    ``transform_lidar_child`` similarly maps the Odometry child frame into the
    LiDAR frame.  The latter is identity only when the LIO output pose is the
    LiDAR pose.  Stock FAST-LIO2 publishes the IMU/body pose, in which case it
    must be the inverse of FAST-LIO2's LiDAR-in-IMU extrinsic.
    """

    camera_lidar = validated_pose_matrix(
        transform_camera_lidar,
        name="rectified LiDAR-to-camera transform",
    )
    lidar_child = validated_pose_matrix(
        transform_lidar_child,
        name="LIO child-to-LiDAR transform",
    )
    return validated_pose_matrix(
        camera_lidar @ lidar_child,
        name="LIO child-to-camera transform",
    )
