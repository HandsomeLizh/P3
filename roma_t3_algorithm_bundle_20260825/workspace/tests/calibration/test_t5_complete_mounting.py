from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src" / "t1_dataset_sim"))
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t1_dataset_sim.dataset_formats import load_calibration
from t3_semantic_mapping.localization.calibration import (
    load_rectified_lidar_to_camera_transform,
    load_rectified_mounting_to_camera_transform,
)
from t3_semantic_mapping.localization.pose_math import inverse_pose


class T5CompleteMountingCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = load_calibration(
            WORKSPACE
            / "src"
            / "t3_semantic_mapping"
            / "config"
            / "t5_rgb10_rgb11.yaml"
        )

    def test_stereo_relative_transform_closes_through_vehicle(self) -> None:
        mounting = self.calibration["mounting"]
        vehicle_left = np.asarray(
            mounting["transform_vehicle_left"], dtype=np.float64
        )
        vehicle_right = np.asarray(
            mounting["transform_vehicle_right"], dtype=np.float64
        )
        expected = np.asarray(
            self.calibration["raw_stereo"]["transform_right_to_left"],
            dtype=np.float64,
        )
        actual = inverse_pose(vehicle_left) @ vehicle_right
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.0)

    def test_map_alignment_uses_frd_mounting_and_rectification(self) -> None:
        mounting = self.calibration["mounting"]
        self.assertEqual(
            mounting["vehicle_axes"], "x_forward_y_right_z_down"
        )
        vehicle_left = np.asarray(
            mounting["transform_vehicle_left"], dtype=np.float64
        )
        left_rectification = np.asarray(
            self.calibration["rectification"]["left_r"], dtype=np.float64
        )
        expected = (
            np.diag([1.0, -1.0, -1.0])
            @ vehicle_left[:3, :3]
            @ left_rectification.T
        )
        actual = np.asarray(
            self.calibration["map_alignment"]["rotation_camera_to_map"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_imu_and_lidar_include_left_rectification(self) -> None:
        mounting = self.calibration["mounting"]
        vehicle_left = np.asarray(
            mounting["transform_vehicle_left"], dtype=np.float64
        )
        left_rectification = np.eye(4, dtype=np.float64)
        left_rectification[:3, :3] = np.asarray(
            self.calibration["rectification"]["left_r"], dtype=np.float64
        )

        for sensor_key in (
            "transform_vehicle_imu",
            "transform_vehicle_lidar",
        ):
            vehicle_sensor = np.asarray(
                mounting[sensor_key], dtype=np.float64
            )
            expected = (
                left_rectification
                @ inverse_pose(vehicle_left)
                @ vehicle_sensor
            )
            actual = load_rectified_mounting_to_camera_transform(
                self.calibration,
                sensor_key=sensor_key,
            )
            self.assertIsNotNone(actual)
            np.testing.assert_allclose(
                actual, expected, atol=1e-12, rtol=0.0
            )

        lidar = load_rectified_lidar_to_camera_transform(self.calibration)
        direct = load_rectified_mounting_to_camera_transform(
            self.calibration,
            sensor_key="transform_vehicle_lidar",
        )
        np.testing.assert_allclose(lidar, direct, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
