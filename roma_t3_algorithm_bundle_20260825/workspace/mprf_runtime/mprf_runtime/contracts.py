"""Versioned and strictly validated service data contracts."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np

from .config import PROTOCOL_VERSION


_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


def validated_session_id(value: str) -> str:
    session = str(value).strip()
    if not _SESSION_PATTERN.fullmatch(session) or ".." in session:
        raise ValueError(
            "session_id must contain 1-96 letters, digits, '.', '_' or '-' "
            "and cannot contain '..'"
        )
    return session


def _readonly_array(value, *, dtype, shape, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _validated_transform(value, *, name: str) -> np.ndarray:
    result = _readonly_array(
        value, dtype=np.float64, shape=(4, 4), name=name
    ).copy()
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class KeyframePacket:
    session_id: str
    node_id: int
    timestamp_ns: int
    image_bgr: np.ndarray
    points_sensor: np.ndarray
    camera_matrix: np.ndarray
    transform_camera_sensor: np.ndarray
    calibration_width: int
    calibration_height: int

    def __post_init__(self) -> None:
        session = validated_session_id(self.session_id)
        node = int(self.node_id)
        timestamp = int(self.timestamp_ns)
        if node < 0:
            raise ValueError("node_id must be non-negative")
        if timestamp <= 0:
            raise ValueError("timestamp_ns must be positive")
        image = np.asarray(self.image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must be an HxWx3 uint8 array")
        if min(image.shape[:2]) < 14 or max(image.shape[:2]) > 8192:
            raise ValueError("image dimensions are outside [14, 8192]")
        image = np.ascontiguousarray(image.copy())
        image.setflags(write=False)
        points = np.asarray(self.points_sensor, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("points_sensor contains non-finite values")
        points = np.ascontiguousarray(points.copy())
        points.setflags(write=False)
        camera = _readonly_array(
            self.camera_matrix,
            dtype=np.float64,
            shape=(3, 3),
            name="camera_matrix",
        )
        if camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
            raise ValueError("camera_matrix focal lengths must be positive")
        width = int(self.calibration_width)
        height = int(self.calibration_height)
        if width <= 0 or height <= 0:
            raise ValueError("calibration dimensions must be positive")
        object.__setattr__(self, "session_id", session)
        object.__setattr__(self, "node_id", node)
        object.__setattr__(self, "timestamp_ns", timestamp)
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "points_sensor", points)
        object.__setattr__(self, "camera_matrix", camera)
        object.__setattr__(
            self,
            "transform_camera_sensor",
            _validated_transform(
                self.transform_camera_sensor,
                name="transform_camera_sensor",
            ),
        )
        object.__setattr__(self, "calibration_width", width)
        object.__setattr__(self, "calibration_height", height)


def decode_keyframe_packet(
    body: bytes, *, maximum_points: int
) -> KeyframePacket:
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as payload:
            version = int(np.asarray(payload["protocol_version"]).reshape(()))
            if version != PROTOCOL_VERSION:
                raise ValueError(
                    f"unsupported protocol_version={version}; expected {PROTOCOL_VERSION}"
                )
            points = np.asarray(payload["points_sensor"])
            if points.size % 3 or points.size // 3 > int(maximum_points):
                raise ValueError(
                    f"points_sensor exceeds maximum {int(maximum_points)}"
                )
            calibration_size = np.asarray(
                payload["calibration_size"], dtype=np.int64
            ).reshape(2)
            packet = KeyframePacket(
                session_id=str(np.asarray(payload["session_id"]).reshape(())),
                node_id=int(np.asarray(payload["node_id"]).reshape(())),
                timestamp_ns=int(
                    np.asarray(payload["timestamp_ns"]).reshape(())
                ),
                image_bgr=payload["image_bgr"],
                points_sensor=points,
                camera_matrix=payload["camera_matrix"],
                transform_camera_sensor=payload["transform_camera_sensor"],
                calibration_width=int(calibration_size[0]),
                calibration_height=int(calibration_size[1]),
            )
    except (KeyError, OSError, TypeError, ValueError) as exception:
        raise ValueError(f"invalid MPRF keyframe payload: {exception}") from exception
    return packet


@dataclass(frozen=True)
class PoseFeatureSet:
    points_sensor: np.ndarray
    descriptors: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_sensor, dtype=np.float32).reshape(-1, 3)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if descriptors.ndim != 2 or len(descriptors) != len(points):
            raise ValueError("pose feature points and descriptors do not align")
        if not np.isfinite(points).all() or not np.isfinite(descriptors).all():
            raise ValueError("pose feature set contains non-finite values")
        points = np.ascontiguousarray(points.copy())
        descriptors = np.ascontiguousarray(descriptors.copy())
        points.setflags(write=False)
        descriptors.setflags(write=False)
        object.__setattr__(self, "points_sensor", points)
        object.__setattr__(self, "descriptors", descriptors)
