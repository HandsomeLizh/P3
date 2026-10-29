#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from evaluate_all_metrics import (
    _apply_rpy_signs_to_trajectory,
    _euler_rpy_deg_to_quaternion_xyzw,
    _rotation_matrix_to_euler_xyz_deg,
    evaluate_map_metrics,
    evaluate_reference_metrics,
    evaluate_runtime_metrics,
)
from evaluate_orientation_stability import (
    TumTrajectory,
    quaternion_xyzw_to_matrix,
)


class EvaluateAllMetricsTest(unittest.TestCase):
    def test_pq_roll_polarity_conversion_preserves_other_euler_fields(self):
        timestamps = np.asarray([0.0, 1.0])
        positions = np.zeros((2, 3))
        source_rpy = np.asarray([[3.0, -2.0, 25.0], [8.0, 4.0, -40.0]])
        trajectory = TumTrajectory(
            timestamps,
            positions,
            np.stack(
                [_euler_rpy_deg_to_quaternion_xyzw(*rpy) for rpy in source_rpy]
            ),
        )

        corrected = _apply_rpy_signs_to_trajectory(
            trajectory, (-1.0, 1.0, 1.0)
        )
        corrected_rpy = _rotation_matrix_to_euler_xyz_deg(
            np.stack(
                [
                    quaternion_xyzw_to_matrix(quaternion)
                    for quaternion in corrected.quaternions_xyzw
                ]
            )
        )

        np.testing.assert_allclose(
            corrected_rpy,
            source_rpy * np.asarray([-1.0, 1.0, 1.0]),
            atol=1e-10,
        )

    def test_reference_metrics_match_identical_long_trajectories(self):
        timestamps = np.arange(0.0, 1001.0, 20.0)
        positions = np.column_stack(
            [timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)]
        )
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(timestamps), 1))
        trajectory = TumTrajectory(timestamps, positions, quaternions)

        metrics, samples = evaluate_reference_metrics(
            trajectory,
            trajectory,
            warmup_sec=0.0,
            maximum_reference_bracket_sec=20.1,
        )

        self.assertEqual(metrics["position_accuracy"]["status"], "pass")
        self.assertEqual(metrics["orientation_stability"]["status"], "pass")
        self.assertAlmostEqual(
            metrics["position_accuracy"]["terminal_error_m"], 0.0
        )
        self.assertAlmostEqual(
            metrics["position_accuracy"]["supplementary_ate_rmse_m"], 0.0
        )
        self.assertEqual(metrics["orientation_stability"]["period_count"], 2)
        self.assertAlmostEqual(
            metrics["orientation_stability"]["selected_max_deg_per_sec"],
            0.0,
        )
        self.assertEqual(len(samples["timestamp_sec"]), len(timestamps))

    def test_reference_metrics_include_short_partial_stability_period(self):
        timestamps = np.arange(0.0, 401.0, 100.0)
        positions = np.column_stack(
            [timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)]
        )
        reference_quaternions = np.tile(
            [0.0, 0.0, 0.0, 1.0], (len(timestamps), 1)
        )
        yaw_rad = np.deg2rad(timestamps * 0.01)
        estimate_quaternions = np.column_stack(
            [
                np.zeros_like(yaw_rad),
                np.zeros_like(yaw_rad),
                np.sin(0.5 * yaw_rad),
                np.cos(0.5 * yaw_rad),
            ]
        )
        estimate = TumTrajectory(
            timestamps, positions, estimate_quaternions
        )
        reference = TumTrajectory(
            timestamps, positions, reference_quaternions
        )

        metrics, _ = evaluate_reference_metrics(
            estimate,
            reference,
            warmup_sec=0.0,
            maximum_reference_bracket_sec=100.1,
        )

        stability = metrics["orientation_stability"]
        self.assertEqual(stability["period_count"], 1)
        self.assertEqual(stability["complete_period_count"], 0)
        self.assertEqual(stability["partial_period_count"], 1)
        self.assertAlmostEqual(stability["periods"][0]["duration_sec"], 400.0)
        self.assertTrue(stability["periods"][0]["is_partial_period"])
        self.assertEqual(stability["periods"][0]["normalization_duration_sec"], 400.0)
        self.assertAlmostEqual(
            stability["periods"][0]["orientation_error_change_inf_norm_deg"],
            4.0,
        )
        self.assertAlmostEqual(stability["selected_max_deg_per_sec"], 0.01)
        self.assertEqual(stability["status"], "fail")
        self.assertEqual(len(stability["selected_periods"]), 1)
        self.assertEqual(stability["selected_periods"][0]["selection_order"], 1)

    def test_runtime_metrics_use_exact_word_random_ten_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input_processing_summary.json"
            samples = [
                {
                    "valid_frame_index": index + 1,
                    "sensor_stamp_ns": index,
                    "pose_publish_latency_sec": 1.4,
                    "map_update_latency_sec": 0.8,
                    "valid_end_to_end_latency_sec": 2.3,
                }
                for index in range(10)
            ]
            path.write_text(
                json.dumps(
                    {
                        "synchronized_callbacks": 10,
                        "synchronized_sensor_span_sec": 20.0,
                        "mapper_results": {
                            "valid_pose_count": 9,
                            "tracking_or_processing_failure_count": 1,
                        },
                        "test_metric_measurements": {
                            "document_random_sample": {
                                "population_count": 10,
                                "method": "reservoir_sampling_over_all_valid_frames",
                                "seed": 20260821,
                                "samples": samples,
                            },
                            "pose_publish_latency_sec": {
                                "count": 10,
                                "mean": 1.4,
                                "max": 1.4,
                            },
                            "map_update_latency_sec": {
                                "count": 10,
                                "mean": 0.8,
                                "max": 0.8,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = evaluate_runtime_metrics(path, time_statistic="mean")
            self.assertAlmostEqual(
                result["position_processing_time"]["selected_max_sec"], 1.4
            )
            self.assertEqual(result["position_processing_time"]["status"], "pass")
            self.assertEqual(result["map_update_time"]["status"], "pass")
            self.assertAlmostEqual(result["tracking"]["success_percent"], 90.0)

    def test_map_range_uses_at_least_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            observed = np.ones((4, 4), dtype=np.uint32)
            internal = output / "_internal" / "map_observation_archive"
            internal.mkdir(parents=True)
            (internal / "evidence.bin").write_bytes(b"map evidence")
            np.savez_compressed(
                output / "global_grid_map.npz",
                resolution=np.asarray(200.0),
                width=np.asarray(4),
                height=np.asarray(4),
                length_x=np.asarray(800.0),
                length_y=np.asarray(800.0),
                map_revision=np.asarray(1),
                observation_count=observed,
            )
            result = evaluate_map_metrics(output)
            self.assertAlmostEqual(
                result["map_range"]["primary_area_km2"], 0.64
            )
            self.assertEqual(result["map_range"]["status"], "pass")
            self.assertEqual(result["storage"]["status"], "pass")
            self.assertIn(
                "_internal/map_observation_archive/evidence.bin",
                result["storage"]["map_data_file_bytes"],
            )


if __name__ == "__main__":
    unittest.main()
