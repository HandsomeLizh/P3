#!/usr/bin/env python3
"""Regression tests for evaluate_orientation_stability.py."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("evaluate_orientation_stability.py")
SPEC = importlib.util.spec_from_file_location("orientation_stability", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def yaw_quaternion(degrees: float) -> np.ndarray:
    half = math.radians(degrees) * 0.5
    return np.asarray([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def yaw_matrix(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def trajectory(times: np.ndarray, yaws: np.ndarray, speed: float = 1.0):
    return MODULE.TumTrajectory(
        timestamps=np.asarray(times, dtype=np.float64),
        positions=np.column_stack(
            [np.asarray(times, dtype=np.float64) * speed, np.zeros(len(times)), np.zeros(len(times))]
        ),
        quaternions_xyzw=np.stack([yaw_quaternion(value) for value in yaws]),
    )


def evaluate(estimate, reference, threshold: float = 0.005):
    return MODULE.evaluate_formal(
        estimate,
        reference,
        estimate_path="estimate.tum",
        reference_path="reference.tum",
        estimate_frame="base_link",
        reference_frame="base_link",
        world_frame="map",
        estimate_body_to_common_body=np.eye(3),
        reference_body_to_common_body=np.eye(3),
        warmup_sec=0.0,
        max_reference_bracket_sec=10.0,
        threshold_deg_per_sec=threshold,
        minimum_duration_sec=0.0,
        minimum_distance_m=0.0,
        minimum_samples=2,
    )


class OrientationStabilityTest(unittest.TestCase):
    def test_identical_irregular_trajectories_have_zero_metric(self):
        times = np.asarray([0.0, 0.4, 1.7, 3.0, 6.2])
        yaws = np.asarray([0.0, 1.0, 4.0, 7.0, 12.0])
        report, _ = evaluate(trajectory(times, yaws), trajectory(times, yaws))
        self.assertAlmostEqual(report["mean_yaw_error_change_rate_deg_per_sec"], 0.0)
        self.assertTrue(report["pass"])

    def test_constant_yaw_offset_does_not_count_as_instability(self):
        times = np.asarray([0.0, 1.0, 2.0, 5.0])
        reference_yaw = np.asarray([0.0, 2.0, 5.0, 9.0])
        report, _ = evaluate(
            trajectory(times, reference_yaw + 10.0),
            trajectory(times, reference_yaw),
        )
        self.assertAlmostEqual(report["mean_yaw_error_change_rate_deg_per_sec"], 0.0)
        self.assertAlmostEqual(report["yaw_error_deg"]["mean"], 10.0)

    def test_linear_yaw_error_gives_expected_time_weighted_rate(self):
        times = np.asarray([0.0, 0.7, 2.5, 8.0, 20.0])
        drift_rate = 0.004
        report, _ = evaluate(
            trajectory(times, times * drift_rate),
            trajectory(times, np.zeros(len(times))),
        )
        self.assertAlmostEqual(
            report["mean_yaw_error_change_rate_deg_per_sec"], drift_rate, places=10
        )
        self.assertTrue(report["pass"])

    def test_wrap_crossing_is_two_degrees_not_358(self):
        times = np.asarray([0.0, 1.0])
        report, samples = evaluate(
            trajectory(times, np.asarray([179.0, -179.0])),
            trajectory(times, np.zeros(2)),
            threshold=3.0,
        )
        self.assertAlmostEqual(report["cumulative_absolute_yaw_error_change_deg"], 2.0)
        self.assertAlmostEqual(samples["absolute_yaw_error_change_deg"][1], 2.0)

    def test_reference_is_slerped_to_estimate_timestamps(self):
        reference_times = np.asarray([0.0, 2.0, 4.0, 6.0])
        estimate_times = np.asarray([0.5, 1.5, 3.0, 5.5])
        report, _ = evaluate(
            trajectory(estimate_times, estimate_times * 10.0),
            trajectory(reference_times, reference_times * 10.0),
        )
        self.assertAlmostEqual(report["mean_yaw_error_change_rate_deg_per_sec"], 0.0)
        self.assertEqual(report["rejected_reference_pair_samples"], 0)

    def test_fixed_source_to_common_rotation_is_applied_in_correct_direction(self):
        times = np.asarray([0.0, 1.0, 2.0])
        estimate = trajectory(times, np.asarray([30.0, 31.0, 32.0]))
        reference = trajectory(times, np.asarray([20.0, 21.0, 22.0]))
        report, _ = MODULE.evaluate_formal(
            estimate,
            reference,
            estimate_path="estimate.tum",
            reference_path="reference.tum",
            estimate_frame="camera",
            reference_frame="base_link",
            world_frame="map",
            estimate_body_to_common_body=yaw_matrix(10.0),
            reference_body_to_common_body=np.eye(3),
            warmup_sec=0.0,
            max_reference_bracket_sec=10.0,
            threshold_deg_per_sec=0.005,
            minimum_duration_sec=0.0,
            minimum_distance_m=0.0,
            minimum_samples=2,
        )
        self.assertAlmostEqual(report["mean_yaw_error_change_rate_deg_per_sec"], 0.0)

    def test_large_reference_gap_is_rejected(self):
        estimate_times = np.asarray([0.0, 1.0, 5.0, 10.0])
        reference_times = np.asarray([0.0, 2.0, 10.0])
        estimate = trajectory(estimate_times, np.zeros(4))
        reference = trajectory(reference_times, np.zeros(3))
        report, _ = MODULE.evaluate_formal(
            estimate,
            reference,
            estimate_path="estimate.tum",
            reference_path="reference.tum",
            estimate_frame="base_link",
            reference_frame="base_link",
            world_frame="map",
            estimate_body_to_common_body=np.eye(3),
            reference_body_to_common_body=np.eye(3),
            warmup_sec=0.0,
            max_reference_bracket_sec=3.0,
            threshold_deg_per_sec=0.005,
            minimum_duration_sec=0.0,
            minimum_distance_m=0.0,
            minimum_samples=2,
        )
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["rejected_reference_pair_samples"], 1)

    def test_aggregate_uses_worst_of_five_valid_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, rate in enumerate([0.001, 0.002, 0.003, 0.004, 0.0049]):
                path = Path(directory) / f"run_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "evaluation_type": MODULE.FORMAL_EVALUATION_TYPE,
                            "threshold_deg_per_sec": 0.005,
                            "mean_yaw_error_change_rate_deg_per_sec": rate,
                            "acceptance_valid": True,
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            report = MODULE.aggregate_formal_reports(paths, required_runs=5)
            self.assertAlmostEqual(report["maximum_run_metric_deg_per_sec"], 0.0049)
            self.assertTrue(report["pass"])

    def test_static_diagnostic_never_claims_acceptance(self):
        times = np.asarray([0.0, 1.0, 2.0])
        report = MODULE.evaluate_diagnostic_static(
            trajectory(times, np.zeros(3)),
            warmup_sec=0.0,
            threshold_deg_per_sec=0.005,
            minimum_duration_sec=0.0,
        )
        self.assertFalse(report["acceptance_valid"])
        self.assertIsNone(report["pass"])
        self.assertEqual(report["status"], "diagnostic_only")


if __name__ == "__main__":
    unittest.main()
