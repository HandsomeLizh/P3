from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Optional

import numpy as np
from sensor_msgs.msg import Imu


@dataclass
class IntegrationDiagnostic:
    attempted: bool = False
    success: bool = False
    interval_sec: float = 0.0
    raw_samples: int = 0
    integration_nodes: int = 0
    message: str = "No integration attempted yet."


class RosImuBuffer:
    """Timestamp-based gyro integration adapter for StereoSLAMv2.

    Only angular velocity is currently integrated.  A full calibrated
    IMU-to-rectified-camera rotation can replace the historical pitch-only
    axis approximation; acceleration, position and message orientation remain
    unused.

    Unlike the original file-based helper, integration uses each message's
    actual Header.stamp and therefore supports both high- and low-rate IMU.
    """

    def __init__(
        self,
        camera_pitch_deg: float = 30.0,
        max_age_sec: float = 300.0,
        max_boundary_extrapolation_sec: float = 1.0,
    ) -> None:
        self.max_age_sec = float(max_age_sec)
        self.max_boundary_extrapolation_sec = float(
            max_boundary_extrapolation_sec
        )
        self.samples: Deque[tuple[float, np.ndarray]] = deque()
        self.image_times: list[float] = []
        self.lock = Lock()
        self.R_cam_imu = self._build_R_cam_imu(camera_pitch_deg)
        self.last_diagnostic = IntegrationDiagnostic()

    def set_rotation_camera_imu(self, rotation_camera_imu: np.ndarray) -> None:
        """Install the exact rotation mapping IMU vectors into the camera."""

        rotation = np.asarray(rotation_camera_imu, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(
                "rotation_camera_imu must be 3x3, "
                f"got {rotation.shape}"
            )
        orthogonality_error = np.linalg.norm(
            rotation.T @ rotation - np.eye(3), ord="fro"
        )
        determinant = float(np.linalg.det(rotation))
        if (
            not np.isfinite(rotation).all()
            or orthogonality_error > 1e-3
            or determinant <= 0.0
        ):
            raise ValueError(
                "rotation_camera_imu is invalid: "
                f"error={orthogonality_error:.3e}, det={determinant:.6f}"
            )
        self.R_cam_imu = rotation.copy()

    @staticmethod
    def _build_R_cam_imu(pitch_deg: float) -> np.ndarray:
        # Legacy fallback kept identical to the original IMUHelper in
        # roma_vo.py. New T5 calibration replaces it with the measured matrix.
        align = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        theta = np.deg2rad(float(pitch_deg))
        pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )
        return pitch @ align

    @staticmethod
    def _time_seconds(message: Imu) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )

    @staticmethod
    def _exp_so3(rotation_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
        angle = np.linalg.norm(vector)
        if angle < 1e-12:
            return np.eye(3, dtype=np.float64)
        axis = vector / angle
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        return (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    def add_message(self, message: Imu) -> None:
        timestamp = self._time_seconds(message)
        gyro = np.array(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(timestamp) or not np.isfinite(gyro).all():
            return

        with self.lock:
            if self.samples and timestamp < self.samples[-1][0]:
                # Insert and sort rare out-of-order messages.
                values = list(self.samples)
                values.append((timestamp, gyro))
                values.sort(key=lambda item: item[0])
                self.samples = deque(values)
            elif self.samples and abs(timestamp - self.samples[-1][0]) < 1e-12:
                self.samples[-1] = (timestamp, gyro)
            else:
                self.samples.append((timestamp, gyro))

            newest = self.samples[-1][0]
            while self.samples and newest - self.samples[0][0] > self.max_age_sec:
                self.samples.popleft()

    def register_image_timestamp(self, timestamp_seconds: float) -> None:
        self.image_times.append(float(timestamp_seconds))

    def get_timestamp(self, frame_index: int) -> float:
        # StereoSLAMv2 frame indices are one-based.
        if 1 <= int(frame_index) <= len(self.image_times):
            return self.image_times[int(frame_index) - 1]
        return -1.0

    def get_delta_R_cam(
        self,
        frame_from: int,
        frame_to: int,
    ) -> Optional[np.ndarray]:
        t0 = self.get_timestamp(frame_from)
        t1 = self.get_timestamp(frame_to)

        diagnostic = IntegrationDiagnostic(
            attempted=True,
            interval_sec=max(0.0, t1 - t0),
        )

        if t0 < 0.0 or t1 <= t0:
            diagnostic.message = "Invalid image timestamp interval."
            self.last_diagnostic = diagnostic
            return None

        with self.lock:
            sample_list = list(self.samples)

        if not sample_list:
            diagnostic.message = "No IMU samples in buffer."
            self.last_diagnostic = diagnostic
            return None

        times = np.array([item[0] for item in sample_list], dtype=np.float64)
        gyros = np.stack([item[1] for item in sample_list], axis=0)

        if (
            t0 < times[0] - self.max_boundary_extrapolation_sec
            or t1 > times[-1] + self.max_boundary_extrapolation_sec
        ):
            diagnostic.message = (
                f"IMU time coverage insufficient: buffer=[{times[0]:.6f}, "
                f"{times[-1]:.6f}], request=[{t0:.6f}, {t1:.6f}]."
            )
            self.last_diagnostic = diagnostic
            return None

        internal_mask = (times > t0) & (times < t1)
        internal_times = times[internal_mask]

        nodes = np.concatenate(([t0], internal_times, [t1]))
        nodes = np.unique(nodes)

        if len(times) == 1:
            node_gyros = np.repeat(gyros[:1], len(nodes), axis=0)
        else:
            node_gyros = np.column_stack(
                [
                    np.interp(nodes, times, gyros[:, axis])
                    for axis in range(3)
                ]
            )

        dR_body = np.eye(3, dtype=np.float64)
        for index in range(len(nodes) - 1):
            dt = float(nodes[index + 1] - nodes[index])
            if dt <= 0.0:
                continue
            average_gyro = 0.5 * (
                node_gyros[index] + node_gyros[index + 1]
            )
            dR_body = dR_body @ self._exp_so3(average_gyro * dt)

        raw_samples = int(np.count_nonzero((times > t0) & (times <= t1)))
        diagnostic.success = True
        diagnostic.raw_samples = raw_samples
        diagnostic.integration_nodes = len(nodes)
        diagnostic.message = (
            f"Integrated gyro interval={t1 - t0:.6f}s, "
            f"raw_samples={raw_samples}, integration_nodes={len(nodes)}"
        )
        self.last_diagnostic = diagnostic

        return self.R_cam_imu @ dR_body @ self.R_cam_imu.T
