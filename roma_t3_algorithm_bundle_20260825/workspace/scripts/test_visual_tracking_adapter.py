#!/usr/bin/env python3
"""Tests for the ROS-independent visual tracking adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.visual_tracking_adapter import (
    process_visual_tracking,
    visual_frontend_required,
)


class _Slam:
    def __init__(self, *, pose=None, status=None):
        self.pose = np.eye(4) if pose is None else pose
        self.status = (
            SimpleNamespace(success=True, reason="ok")
            if status is None
            else status
        )
        self.process_exception = None
        self.status_exception = None
        self.calls = []

    def process_frame(self, left, right):
        self.calls.append((left, right))
        if self.process_exception is not None:
            raise self.process_exception
        return self.pose

    def get_tracking_status(self):
        if self.status_exception is not None:
            raise self.status_exception
        return self.status


def _run(slam, *, required, enabled=True):
    left = np.zeros((2, 3, 3), dtype=np.uint8)
    right = np.ones((2, 3, 3), dtype=np.uint8)
    result = process_visual_tracking(
        slam,
        left,
        right,
        visual_required=required,
        frame_index=7,
        geometry_revision=11,
        enabled=enabled,
    )
    return result, left, right


class VisualTrackingAdapterTest(unittest.TestCase):
    def test_only_direct_lio_without_stereo_fallback_bypasses_frontend(self):
        cases = {
            ("vo", False, "direct", False): True,
            ("vo_lio", True, "direct", False): True,
            ("lio", False, "direct", False): True,
            ("lio", True, "assist", False): True,
            ("lio", True, "direct", True): True,
            ("lio", True, "direct", False): False,
        }
        for configuration, expected in cases.items():
            with self.subTest(configuration=configuration):
                self.assertEqual(
                    visual_frontend_required(
                        localization_mode=configuration[0],
                        enable_lidar_mapping=configuration[1],
                        lidar_mapping_mode=configuration[2],
                        lidar_fallback_to_stereo=configuration[3],
                    ),
                    expected,
                )

    def test_disabled_optional_frontend_never_calls_slam(self):
        slam = _Slam()
        (pose, status), _, _ = _run(
            slam,
            required=False,
            enabled=False,
        )

        self.assertEqual(slam.calls, [])
        self.assertEqual(pose.shape, (0, 0))
        self.assertFalse(status.success)
        self.assertFalse(status.attempted)
        self.assertEqual(status.frame_index, 7)
        self.assertEqual(status.reason, "visual frontend intentionally bypassed")

    def test_required_frontend_cannot_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "cannot be bypassed"):
            _run(_Slam(), required=True, enabled=False)

    def test_success_returns_pose_status_and_forwards_images_once(self):
        pose = np.eye(4, dtype=np.float64)
        status = SimpleNamespace(success=True, reason="tracked")
        slam = _Slam(pose=pose, status=status)

        (actual_pose, actual_status), left, right = _run(
            slam, required=True
        )

        self.assertIs(actual_pose, pose)
        self.assertIs(actual_status, status)
        self.assertEqual(len(slam.calls), 1)
        self.assertIs(slam.calls[0][0], left)
        self.assertIs(slam.calls[0][1], right)

    def test_frontend_tracking_failure_status_is_not_rewritten(self):
        status = SimpleNamespace(success=False, reason="pnp_failed")
        slam = _Slam(status=status)
        (pose, actual_status), _, _ = _run(slam, required=True)
        np.testing.assert_array_equal(pose, np.eye(4))
        self.assertIs(actual_status, status)

    def test_required_visual_mode_preserves_process_exception(self):
        slam = _Slam()
        expected = RuntimeError("dark image")
        slam.process_exception = expected
        with self.assertRaises(RuntimeError) as context:
            _run(slam, required=True)
        self.assertIs(context.exception, expected)

    def test_optional_visual_mode_normalises_process_exception(self):
        slam = _Slam()
        slam.process_exception = RuntimeError("dark image")
        (pose, status), _, _ = _run(slam, required=False)

        self.assertEqual(pose.shape, (0, 0))
        self.assertFalse(status.success)
        self.assertEqual(status.frame_index, 7)
        self.assertEqual(status.pose_index, -1)
        self.assertFalse(status.keyframe_created)
        self.assertIsNone(status.keyframe_id)
        self.assertFalse(status.geometry_corrected)
        self.assertEqual(status.geometry_revision, 11)
        self.assertEqual(status.inlier_count, 0)
        self.assertEqual(status.correspondence_count, 0)
        self.assertEqual(
            status.reason,
            "visual processing exception: RuntimeError: dark image",
        )

    def test_required_visual_mode_rejects_invalid_pose(self):
        slam = _Slam(pose=np.eye(3))
        with self.assertRaisesRegex(ValueError, "Invalid pose returned"):
            _run(slam, required=True)

    def test_optional_visual_mode_preserves_invalid_pose_for_diagnostics(self):
        slam = _Slam(pose=np.eye(3))
        (pose, status), _, _ = _run(slam, required=False)
        self.assertEqual(pose.shape, (3, 3))
        self.assertEqual(
            status.reason,
            "visual processing exception: ValueError: visual algorithm "
            "returned an invalid 4x4 pose",
        )

    def test_missing_status_contract_fails_or_normalises_by_mode(self):
        slam = SimpleNamespace(
            process_frame=lambda _left, _right: np.eye(4)
        )
        with self.assertRaisesRegex(RuntimeError, "does not expose"):
            _run(slam, required=True)

        (pose, status), _, _ = _run(slam, required=False)
        self.assertEqual(pose.shape, (4, 4))
        self.assertIn("does not expose", status.reason)

    def test_status_exception_fails_or_normalises_by_mode(self):
        slam = _Slam()
        expected = LookupError("status unavailable")
        slam.status_exception = expected
        with self.assertRaises(LookupError) as context:
            _run(slam, required=True)
        self.assertIs(context.exception, expected)

        (_, status), _, _ = _run(slam, required=False)
        self.assertEqual(
            status.reason,
            "visual processing exception: LookupError: status unavailable",
        )


if __name__ == "__main__":
    unittest.main()
