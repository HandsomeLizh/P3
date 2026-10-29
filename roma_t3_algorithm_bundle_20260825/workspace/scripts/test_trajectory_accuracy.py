#!/usr/bin/env python3
"""Regression tests for trajectory-evaluation input denominators."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_trajectory_accuracy import evaluate


class TrajectoryInputCoverageTest(unittest.TestCase):
    def test_tracking_coverage_uses_selected_stereo_pairs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "images"
            (image_root / "left").mkdir(parents=True)
            (image_root / "right").mkdir()

            lines = ["side\tframe_index\ttimestamp_ns\tfilename\n"]
            left_times = [
                1_100_000_000,
                2_100_000_000,
                3_100_000_000,
                4_100_000_000,
            ]
            right_times = [
                1_100_000_000,
                2_100_000_000,
                3_150_000_000,
                4_100_000_000,
            ]
            for side, times in (("left", left_times), ("right", right_times)):
                for index, timestamp_ns in enumerate(times):
                    frame_index = f"{index:06d}"
                    relative = f"images/{side}/{frame_index}.png"
                    (root / relative).touch()
                    lines.append(
                        f"{side}\t{frame_index}\t{timestamp_ns}\t{relative}\n"
                    )
            (image_root / "timestamps.tsv").write_text(
                "".join(lines), encoding="utf-8"
            )

            fastlio = root / "lidar_fastlio"
            fastlio.mkdir()
            np.savez(
                fastlio / "trajectory.npz",
                header_time_ns=np.asarray(
                    [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_200_000_000],
                    dtype=np.int64,
                ),
                position=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                orientation=np.asarray(
                    [[0.0, 0.0, 0.0, 1.0]] * 4, dtype=np.float64
                ),
            )
            calibration = root / "calibration"
            calibration.mkdir()
            (calibration / "extrinsics.json").write_text(
                json.dumps(
                    {"LiDAR_to_Cam_L": {"RT": np.eye(4).tolist()}}
                ),
                encoding="utf-8",
            )

            estimate = root / "trajectory_map.tum"
            estimate.write_text(
                "# timestamp tx ty tz qx qy qz qw\n"
                "1.100000000 0 0 0 0 0 0 1\n"
                "2.100000000 1 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            output = root / "accuracy.json"
            result = evaluate(estimate, root, output)

            counts = result["counts"]
            self.assertEqual(counts["dataset_shared_stereo_candidates"], 4)
            self.assertEqual(
                counts["dataset_stereo_skew_eligible_pairs"], 3
            )
            self.assertEqual(counts["dataset_selected_stereo_pairs"], 3)
            self.assertEqual(
                counts["dataset_pairs_excluded_by_stereo_skew"], 1
            )
            self.assertEqual(
                counts["reference_compatible_selected_stereo_pairs"], 3
            )
            self.assertAlmostEqual(
                counts["estimated_tracking_success_percent"], 200.0 / 3.0
            )
            self.assertAlmostEqual(
                counts["estimated_pose_coverage_of_selected_percent"],
                200.0 / 3.0,
            )
            self.assertAlmostEqual(
                counts["estimated_pose_coverage_of_raw_left_percent"], 50.0
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
