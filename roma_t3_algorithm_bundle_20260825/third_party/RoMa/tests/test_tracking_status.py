import unittest
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


ROMA_PATH = Path(__file__).resolve().parents[1] / "roma_vo.py"


def _load_front_door_contract():
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    tracking = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrackingStatus"
    )
    slam = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StereoSLAMv2"
    )
    wanted = {
        "_set_tracking_status",
        "_mark_geometry_corrected",
        "get_tracking_status",
        "_handle_tracking_failure",
        "process_frame",
        "_track_keyframe_with_klt",
        "_mark_feature_match_tracking_success",
    }
    miniature = ast.ClassDef(
        name="StereoSLAMv2",
        bases=[],
        keywords=[],
        body=[
            node
            for node in slam.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[tracking, miniature], type_ignores=[])
    )
    namespace = {
        "np": np,
        "dataclass": dataclass,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return namespace["StereoSLAMv2"]


StereoSLAMv2 = _load_front_door_contract()


class _Matcher:
    def __init__(self, count):
        self.count = int(count)

    def match(self, *_args, **_kwargs):
        points = np.zeros((self.count, 2), dtype=np.float32)
        confidence = np.ones(self.count, dtype=np.float32)
        return points, points.copy(), confidence


class _Triangulator:
    def __init__(self, count):
        self.count = int(count)

    def triangulate(self, _left, _right):
        points = np.ones((self.count, 3), dtype=np.float64)
        indices = np.arange(self.count, dtype=np.int64)
        return points, indices


def _minimal_slam(point_count):
    """Build a model-free StereoSLAMv2 instance for front-door tests."""
    slam = StereoSLAMv2.__new__(StereoSLAMv2)
    slam.frame_count = 0
    slam.poses = [np.eye(4, dtype=np.float64)]
    slam.matcher = _Matcher(point_count)
    slam.triangulator = _Triangulator(point_count)
    slam.stereo_matches = 32
    slam.match_resize = None
    # Exercise the original front-door contract without model feature reuse
    # or KLT; these production branches were added after this test harness.
    slam.feature_reuse_active = False
    slam.enable_klt_temporal_tracking = False
    slam.stereo_upsample_refinement = False
    slam.symmetric_matching = False
    slam._kf_img = None
    slam._apply_pending_loop_results = lambda: False
    slam.tracking_state = "NOT_INITIALIZED"
    return slam


class TrackingStatusTest(unittest.TestCase):
    def test_insufficient_stereo_points_marks_failure(self):
        slam = _minimal_slam(point_count=5)

        returned_pose = slam.process_frame(None, None)
        status = slam.get_tracking_status()

        self.assertFalse(status.success)
        self.assertFalse(status.pose_updated)
        self.assertEqual(status.frame_index, 1)
        self.assertEqual(status.reason, "insufficient_stereo_points")
        self.assertEqual(status.correspondence_count, 5)
        self.assertEqual(status.pose_index, -1)
        self.assertFalse(status.keyframe_created)
        np.testing.assert_array_equal(returned_pose, slam.poses[-1])

    def test_first_frame_initialization_marks_success(self):
        slam = _minimal_slam(point_count=10)
        slam._init_first_kf = lambda *_args, **_kwargs: setattr(
            slam, "_kf_img", np.zeros((1, 1, 3), dtype=np.uint8)
        )

        slam.process_frame(None, None)
        status = slam.get_tracking_status()

        self.assertTrue(status.success)
        self.assertTrue(status.pose_updated)
        self.assertTrue(status.initialized)
        self.assertEqual(status.reason, "initialized")
        self.assertEqual(status.correspondence_count, 10)
        self.assertEqual(status.pose_index, 0)
        self.assertTrue(status.keyframe_created)

    def test_geometry_correction_survives_later_tracking_failure(self):
        slam = _minimal_slam(point_count=5)
        slam.geometry_revision = 0

        def apply_correction():
            slam._mark_geometry_corrected()
            return True

        slam._apply_pending_loop_results = apply_correction
        slam.process_frame(None, None)
        status = slam.get_tracking_status()

        self.assertFalse(status.success)
        self.assertTrue(status.geometry_corrected)
        self.assertEqual(status.geometry_revision, 1)


if __name__ == "__main__":
    unittest.main()
