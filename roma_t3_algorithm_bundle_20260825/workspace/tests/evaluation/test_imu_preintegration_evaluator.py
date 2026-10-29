from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


LOCAL_SCRIPT = Path(__file__).with_name("evaluate_imu_preintegration.py")
WORKSPACE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_imu_preintegration.py"
)
SCRIPT = LOCAL_SCRIPT if LOCAL_SCRIPT.is_file() else WORKSPACE_SCRIPT
SPEC = importlib.util.spec_from_file_location("imu_evaluator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ImuPreintegrationEvaluatorTest(unittest.TestCase):
    def test_quaternion_round_trip_normalizes_input(self) -> None:
        expected = MODULE.exp_so3(np.deg2rad([5.0, -10.0, 35.0]))
        quaternion = MODULE.matrix_to_quaternion_wxyz(expected) * 3.0
        actual = MODULE.quaternion_wxyz_to_matrix(quaternion)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_primary_stability_is_time_weighted_mean(self) -> None:
        timestamps = np.array([0.0, 1.0, 3.0])
        reference = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        estimated = MODULE.rotation_trajectory_to_quaternions(
            np.asarray(
                [
                    MODULE.exp_so3([0.0, 0.0, np.deg2rad(angle)])
                    for angle in (0.0, 1.0, 5.0)
                ]
            )
        )
        result = MODULE.evaluate_stability_from_quaternions(
            reference_quaternions=reference,
            estimated_quaternion_trajectories={"estimate": estimated},
            timestamps=timestamps,
            reference_mode="test",
            threshold_deg_per_sec=2.0,
            minimum_duration_seconds=2.0,
        )
        case = result["cases"]["estimate"]
        self.assertAlmostEqual(
            case["full_orientation"]["cumulative_error_change_deg"],
            5.0,
            places=9,
        )
        self.assertAlmostEqual(
            case["full_orientation"][
                "time_weighted_mean_drift_rate_deg_per_sec"
            ],
            5.0 / 3.0,
            places=9,
        )

    def test_constant_bias_is_removed(self) -> None:
        timestamps = np.linspace(0.0, 10.0, 51)
        truth_rate = np.tile([0.0, 0.0, 2.0], (len(timestamps), 1))
        bias = np.array([0.01, -0.02, 0.05])
        raw = MODULE.preintegrate_gyro(
            timestamps, truth_rate + bias, 0.0, 10.0
        )
        corrected = MODULE.preintegrate_gyro(
            timestamps, truth_rate, 0.0, 10.0
        )
        truth = MODULE.preintegrate_gyro(
            timestamps, truth_rate, 0.0, 10.0
        )
        self.assertGreater(MODULE.rotation_angle_deg(truth.T @ raw), 0.4)
        self.assertLess(
            MODULE.rotation_angle_deg(truth.T @ corrected), 1e-9
        )

    def test_robust_bias_rejects_outliers(self) -> None:
        truth = np.zeros((100, 3))
        expected = np.array([0.001, -0.002, 0.006])
        measured = np.tile(expected, (100, 1))
        measured[10] = [10.0, -20.0, 30.0]
        measured[70] = [-5.0, 6.0, -7.0]
        actual, inliers, _ = (
            MODULE.estimate_truth_assisted_constant_bias(measured, truth)
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
        self.assertEqual(int(np.sum(~inliers)), 2)

    def test_so3_interpolation(self) -> None:
        timestamps = np.array([0.0, 2.0])
        rotations = np.asarray(
            [
                np.eye(3),
                MODULE.exp_so3([0.0, 0.0, np.deg2rad(20.0)]),
            ]
        )
        middle = MODULE.interpolate_rotation(timestamps, rotations, 1.0)
        self.assertAlmostEqual(MODULE.yaw_deg(middle), 10.0, places=9)

    def test_quaternion_sync_rejects_excessive_reference_gap(self) -> None:
        timestamps = np.array([0.0, 1.0])
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        with self.assertRaisesRegex(ValueError, "bracket exceeds"):
            MODULE.interpolate_quaternion(
                timestamps,
                quaternions,
                0.5,
                max_bracket_seconds=0.25,
            )

    def test_truth_rate_reference_has_zero_integration_error(self) -> None:
        timestamps = np.linspace(0.0, 2.0, 11)
        truth_rates = np.tile([0.0, 0.0, 1.0], (len(timestamps), 1))
        image_timestamps = np.array([0.0, 1.0, 2.0])
        truth_rotations = MODULE.continuous_orientation_trajectory(
            imu_timestamps=timestamps,
            angular_rates_deg_per_sec=truth_rates,
            image_timestamps=image_timestamps,
            initial_rotation=np.eye(3),
        )
        result = MODULE.evaluate_windows(
            reference_rotations=truth_rotations,
            estimated_trajectories={"truth": truth_rotations},
            image_timestamps=image_timestamps,
            reference_mode="truth_rate_preintegration",
            gap=1,
            threshold_deg_per_sec=0.005,
        )
        case = result["cases"]["truth"]
        self.assertLess(case["full_error_change_deg"]["max"], 1e-9)
        self.assertLess(case["yaw_error_change_deg"]["max"], 1e-9)
        self.assertEqual(
            case["worst_full_rate_windows"][0]["start_image_index"], 0
        )
        self.assertEqual(
            case["worst_full_rate_windows"][0]["end_image_index"], 1
        )

    def test_stability_uses_error_change_not_terminal_error(self) -> None:
        reference_start = MODULE.exp_so3(
            [0.0, 0.0, np.deg2rad(10.0)]
        )
        reference_end = MODULE.exp_so3(
            [0.0, 0.0, np.deg2rad(35.0)]
        )
        constant_error = MODULE.exp_so3(
            np.deg2rad([10.0, 0.0, 0.0])
        )
        result = MODULE.orientation_error_change(
            reference_start,
            reference_end,
            reference_start @ constant_error,
            reference_end @ constant_error,
        )
        full_start, full_end, full_change, yaw_change = result
        self.assertAlmostEqual(full_start, 10.0, places=9)
        self.assertAlmostEqual(full_end, 10.0, places=9)
        self.assertAlmostEqual(full_change, 0.0, places=9)
        self.assertAlmostEqual(yaw_change, 0.0, places=9)

        changed_error = MODULE.exp_so3(
            np.deg2rad([11.0, 0.0, 0.0])
        )
        result = MODULE.orientation_error_change(
            reference_start,
            reference_end,
            reference_start @ constant_error,
            reference_end @ changed_error,
        )
        self.assertAlmostEqual(result[2], 1.0, places=9)
        self.assertAlmostEqual(result[3], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
