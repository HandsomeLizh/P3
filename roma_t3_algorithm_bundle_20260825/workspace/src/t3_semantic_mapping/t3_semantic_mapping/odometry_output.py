from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .localization.pose_math import inverse_pose, validated_pose_matrix


_ROS_COVARIANCE_ORDER = np.array([3, 4, 5, 0, 1, 2], dtype=np.int64)


def _normalised_frame(value: str, *, name: str) -> str:
    frame = str(value).strip().lstrip("/")
    if not frame:
        raise ValueError(f"{name} cannot be empty")
    return frame


def _adjoint_rotation_first(transform_target_source: np.ndarray) -> np.ndarray:
    """Return the SE(3) adjoint for ``[angular, linear]`` vectors."""

    transform = validated_pose_matrix(
        transform_target_source, name="adjoint transform"
    )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    skew = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ],
        dtype=np.float64,
    )
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[3:, :3] = skew @ rotation
    result[3:, 3:] = rotation
    return result


def _validated_covariance(value: np.ndarray, *, name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (6, 6):
        raise ValueError(f"{name} must be 6x6, got {covariance.shape}")
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return covariance.copy()


@dataclass(frozen=True)
class OdometryOutputSample:
    """ROS-independent values for one ``nav_msgs/Odometry`` message.

    ``twist_angular_linear`` is expressed in ``child_frame`` and follows the
    internal ordering ``[wx, wy, wz, vx, vy, vz]``.  Both covariance matrices
    already use the ROS ordering ``[x, y, z, rx, ry, rz]``.
    """

    pose_map_child: np.ndarray
    twist_angular_linear: np.ndarray
    pose_covariance_ros: np.ndarray
    twist_covariance_ros: np.ndarray
    velocity_method: str


class OdometryOutputModel:
    """Make the published pose/twist frame contract explicit and stateful.

    The localization and mapping pipeline estimates ``T_map_camera``.  A
    different Odometry child is permitted only with an explicitly confirmed
    calibration ``T_camera_child``.  Missing source velocity is filled by a
    backward pose difference; no filtering, look-ahead, or frame buffering is
    introduced.
    """

    def __init__(
        self,
        *,
        camera_frame: str,
        child_frame: str,
        transform_camera_child: np.ndarray,
        child_transform_confirmed: bool,
        velocity_max_interval_sec: float = 1.0,
        unknown_covariance_variance: float = 1.0e6,
    ) -> None:
        self.camera_frame = _normalised_frame(
            camera_frame, name="odometry camera_frame"
        )
        self.child_frame = _normalised_frame(
            child_frame, name="odometry child_frame"
        )
        self.transform_camera_child = validated_pose_matrix(
            transform_camera_child,
            name="Odometry child-to-camera transform",
        )
        identity_transform = np.allclose(
            self.transform_camera_child, np.eye(4), atol=1e-12
        )
        if self.child_frame == self.camera_frame:
            if not identity_transform:
                raise ValueError(
                    "Odometry child_frame equals camera_frame, so "
                    "T_camera_child must be identity"
                )
        elif not bool(child_transform_confirmed):
            raise ValueError(
                "Odometry child_frame differs from the estimated camera "
                "frame. Supply the calibrated T_camera_child and set "
                "odometry_child_transform_confirmed=true; changing only the "
                "frame name would mislabel the camera pose."
            )

        self.velocity_max_interval_sec = float(velocity_max_interval_sec)
        if (
            not np.isfinite(self.velocity_max_interval_sec)
            or self.velocity_max_interval_sec <= 0.0
        ):
            raise ValueError(
                "odometry_velocity_max_interval_sec must be finite and positive"
            )
        self.unknown_covariance_variance = float(
            unknown_covariance_variance
        )
        if (
            not np.isfinite(self.unknown_covariance_variance)
            or self.unknown_covariance_variance <= 0.0
        ):
            raise ValueError(
                "odometry_unknown_covariance_variance must be finite and positive"
            )

        self._transform_child_camera = inverse_pose(
            self.transform_camera_child
        )
        self._adjoint_child_camera = _adjoint_rotation_first(
            self._transform_child_camera
        )
        self._previous_timestamp: Optional[float] = None
        self._previous_pose_map_child: Optional[np.ndarray] = None

    def transform_pose(self, pose_map_camera: np.ndarray) -> np.ndarray:
        camera_pose = validated_pose_matrix(
            pose_map_camera, name="map-to-camera pose"
        )
        return validated_pose_matrix(
            camera_pose @ self.transform_camera_child,
            name="map-to-Odometry-child pose",
        )

    def _unknown_covariance(self) -> np.ndarray:
        return np.eye(6, dtype=np.float64) * self.unknown_covariance_variance

    def _pose_covariance_ros(
        self, covariance_camera_rotation_first: Optional[np.ndarray]
    ) -> np.ndarray:
        if covariance_camera_rotation_first is None:
            return self._unknown_covariance()
        covariance_camera = _validated_covariance(
            covariance_camera_rotation_first,
            name="camera pose covariance",
        )
        covariance_child = (
            self._adjoint_child_camera
            @ covariance_camera
            @ self._adjoint_child_camera.T
        )
        order = _ROS_COVARIANCE_ORDER
        return covariance_child[np.ix_(order, order)]

    def _transform_source_twist(self, velocity_camera: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity_camera, dtype=np.float64)
        if velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError(
                "source velocity must be a finite [angular, linear] 6-vector"
            )
        return self._adjoint_child_camera @ velocity

    @staticmethod
    def _backward_difference(
        previous_pose: np.ndarray,
        current_pose: np.ndarray,
        interval_sec: float,
    ) -> np.ndarray:
        previous = validated_pose_matrix(previous_pose, name="previous pose")
        current = validated_pose_matrix(current_pose, name="current pose")
        dt = float(interval_sec)
        displacement_map = current[:3, 3] - previous[:3, 3]
        linear_current = current[:3, :3].T @ displacement_map / dt
        relative_rotation = previous[:3, :3].T @ current[:3, :3]
        angular = Rotation.from_matrix(relative_rotation).as_rotvec() / dt
        return np.concatenate([angular, linear_current])

    def update(
        self,
        *,
        timestamp: float,
        pose_map_camera: np.ndarray,
        velocity_camera: Optional[np.ndarray] = None,
        pose_covariance_camera: Optional[np.ndarray] = None,
        reset_velocity_history: bool = False,
    ) -> OdometryOutputSample:
        sensor_time = float(timestamp)
        if not np.isfinite(sensor_time):
            raise ValueError("Odometry timestamp must be finite")
        pose_map_child = self.transform_pose(pose_map_camera)

        if reset_velocity_history:
            self._previous_timestamp = None
            self._previous_pose_map_child = None

        if velocity_camera is not None:
            twist = self._transform_source_twist(velocity_camera)
            method = "source"
        elif self._previous_timestamp is None:
            twist = np.zeros(6, dtype=np.float64)
            method = "unavailable_first_sample"
        else:
            interval = sensor_time - self._previous_timestamp
            if interval <= 0.0:
                twist = np.zeros(6, dtype=np.float64)
                method = "unavailable_non_monotonic_time"
            elif interval > self.velocity_max_interval_sec:
                twist = np.zeros(6, dtype=np.float64)
                method = "unavailable_time_gap"
            else:
                twist = self._backward_difference(
                    self._previous_pose_map_child,
                    pose_map_child,
                    interval,
                )
                method = "backward_difference"

        self._previous_timestamp = sensor_time
        self._previous_pose_map_child = pose_map_child.copy()
        return OdometryOutputSample(
            pose_map_child=pose_map_child,
            twist_angular_linear=np.asarray(twist, dtype=np.float64),
            pose_covariance_ros=self._pose_covariance_ros(
                pose_covariance_camera
            ),
            # Neither direct VO/LIO nor the current GP wrapper exposes a
            # calibrated twist posterior. A large finite covariance is an
            # explicit unknown value; all-zero covariance would claim perfect
            # certainty and is unsafe for downstream fusion.
            twist_covariance_ros=self._unknown_covariance(),
            velocity_method=method,
        )

