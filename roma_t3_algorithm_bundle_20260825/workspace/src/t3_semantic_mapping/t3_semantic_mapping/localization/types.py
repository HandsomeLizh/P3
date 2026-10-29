from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from .pose_math import validated_pose_matrix


def _readonly_array(value: np.ndarray, shape, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TimedPose:
    timestamp: float
    pose: np.ndarray
    covariance: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("TimedPose timestamp must be finite")
        pose = validated_pose_matrix(self.pose)
        pose.setflags(write=False)
        covariance = self.covariance
        if covariance is not None:
            covariance = _readonly_array(
                covariance, (6, 6), name="pose covariance"
            )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class PoseEstimate:
    """One camera-state estimate in the localization world frame.

    ``velocity_body`` follows the GTSAM/right-tangent ordering
    ``[wx, wy, wz, vx, vy, vz]`` and is expressed at the camera origin in the
    camera frame.  When present, ``covariance`` uses the same rotation-first
    right-tangent convention.  Conversion to the ROS covariance ordering is
    owned exclusively by :mod:`t3_semantic_mapping.odometry_output`.
    """

    timestamp: float
    pose: np.ndarray
    source: str
    velocity_body: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("PoseEstimate timestamp must be finite")
        pose = validated_pose_matrix(self.pose)
        pose.setflags(write=False)
        velocity = self.velocity_body
        if velocity is not None:
            velocity = _readonly_array(
                velocity, (6,), name="body-frame velocity"
            )
        covariance = self.covariance
        if covariance is not None:
            covariance = _readonly_array(
                covariance, (6, 6), name="pose covariance"
            )
        diagnostics = MappingProxyType(dict(self.diagnostics))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "velocity_body", velocity)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class FusionUpdate:
    estimate: PoseEstimate
    measurement_index: int
    active_poses: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        active = {}
        for index, pose in self.active_poses.items():
            value = validated_pose_matrix(
                pose, name=f"active pose {int(index)}"
            )
            value.setflags(write=False)
            active[int(index)] = value
        object.__setattr__(self, "measurement_index", int(self.measurement_index))
        object.__setattr__(self, "active_poses", MappingProxyType(active))
