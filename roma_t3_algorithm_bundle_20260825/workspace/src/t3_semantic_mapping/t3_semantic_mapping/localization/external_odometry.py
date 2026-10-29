from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose_buffer import PoseBufferQuery, TimedPoseBuffer
from .pose_math import (
    child_pose_to_camera_pose,
    pose_matrix,
    relative_pose,
    validated_pose_matrix,
)
from .types import TimedPose


class FrameMismatchError(ValueError):
    pass


def _normalised_frame(value: str) -> str:
    return str(value).strip().lstrip("/")


@dataclass(frozen=True)
class ExternalPoseQuery:
    pose: np.ndarray
    covariance: Optional[np.ndarray]
    method: str
    nearest_offset_sec: float
    bracket_span_sec: float


class ExternalOdometryAdapter:
    """Validate and time-align an external child-frame odometry stream."""

    def __init__(
        self,
        *,
        transform_camera_child: np.ndarray,
        expected_parent_frame: str = "camera_init",
        expected_child_frame: str = "body",
        max_buffer_size: int = 2000,
        max_buffer_age_sec: float = 300.0,
    ) -> None:
        self.transform_camera_child = validated_pose_matrix(
            transform_camera_child,
            name="external child-to-camera calibration",
        )
        self.expected_parent_frame = _normalised_frame(expected_parent_frame)
        self.expected_child_frame = _normalised_frame(expected_child_frame)
        self.buffer = TimedPoseBuffer(
            max_size=max_buffer_size,
            max_age_sec=max_buffer_age_sec,
        )
        self._alignment_reference: Optional[np.ndarray] = None

    def add_components(
        self,
        *,
        timestamp: float,
        parent_frame: str,
        child_frame: str,
        position_xyz: np.ndarray,
        quaternion_xyzw: np.ndarray,
        covariance: Optional[np.ndarray] = None,
    ) -> None:
        parent = _normalised_frame(parent_frame)
        child = _normalised_frame(child_frame)
        if self.expected_parent_frame and parent != self.expected_parent_frame:
            raise FrameMismatchError(
                "External odometry parent frame mismatch: "
                f"expected={self.expected_parent_frame!r}, received={parent!r}"
            )
        if self.expected_child_frame and child != self.expected_child_frame:
            raise FrameMismatchError(
                "External odometry child frame mismatch: "
                f"expected={self.expected_child_frame!r}, received={child!r}"
            )
        parent_child = pose_matrix(position_xyz, quaternion_xyzw)
        parent_camera = child_pose_to_camera_pose(
            parent_child, self.transform_camera_child
        )
        self.buffer.add(TimedPose(timestamp, parent_camera, covariance))

    def query(
        self,
        timestamp: float,
        *,
        max_time_offset_sec: float,
        align_to_first_query: bool = True,
        allow_nearest: bool = True,
    ) -> Optional[ExternalPoseQuery]:
        result: Optional[PoseBufferQuery] = self.buffer.query(
            timestamp,
            max_time_offset_sec=max_time_offset_sec,
            allow_nearest=allow_nearest,
        )
        if result is None:
            return None
        pose = result.sample.pose
        if align_to_first_query:
            if self._alignment_reference is None:
                self._alignment_reference = pose.copy()
            pose = relative_pose(self._alignment_reference, pose)
        return ExternalPoseQuery(
            pose=pose,
            covariance=result.sample.covariance,
            method=result.method,
            nearest_offset_sec=result.nearest_offset_sec,
            bracket_span_sec=result.bracket_span_sec,
        )
