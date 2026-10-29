import unittest
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROMA_PATH = Path(__file__).resolve().parents[1] / "roma_vo.py"


def _load_mapping_contract():
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    data_names = {
        "MapPoint",
        "KeyFrame",
        "TrackingStatus",
        "MappingKeyFrameSnapshot",
    }
    data_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in data_names
    ]
    slam = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StereoSLAMv2"
    )
    method_names = {
        "_readonly_copy",
        "_mark_geometry_corrected",
        "get_mapping_keyframe_snapshot",
        "get_keyframe_pose_snapshot",
        "get_pose_history_snapshot",
    }
    miniature = ast.ClassDef(
        name="StereoSLAMv2",
        bases=[],
        keywords=[],
        body=[
            node
            for node in slam.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=data_nodes + [miniature], type_ignores=[])
    )
    namespace = {
        "np": np,
        "dataclass": dataclass,
        "field": field,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return namespace["KeyFrame"], namespace["StereoSLAMv2"]


KeyFrame, StereoSLAMv2 = _load_mapping_contract()


class MappingContractTest(unittest.TestCase):
    def test_keyframe_snapshot_is_aligned_detached_and_read_only(self):
        slam = StereoSLAMv2.__new__(StereoSLAMv2)
        keyframe = KeyFrame(
            id=3,
            timestamp=4.5,
            pose=np.eye(4, dtype=np.float64),
            keypoints=[[10.0, 20.0], [30.0, 40.0]],
            camera_points=[[1.0, 2.0, 3.0], [4.0, 5.0, -1.0]],
        )
        slam.keyframes = [keyframe]

        snapshot = slam.get_mapping_keyframe_snapshot(3)

        self.assertEqual(snapshot.id, 3)
        self.assertEqual(snapshot.keypoints.shape, (1, 2))
        self.assertEqual(snapshot.camera_points.shape, (1, 3))
        keyframe.keypoints[0][0] = 99.0
        keyframe.camera_points[0][0] = 99.0
        self.assertEqual(float(snapshot.keypoints[0, 0]), 10.0)
        self.assertEqual(float(snapshot.camera_points[0, 0]), 1.0)
        with self.assertRaises(ValueError):
            snapshot.pose[0, 3] = 1.0

    def test_geometry_revision_only_changes_when_marked(self):
        slam = StereoSLAMv2.__new__(StereoSLAMv2)
        slam.geometry_revision = 5
        slam._geometry_corrected_in_process = False
        slam.keyframes = []
        slam.poses = [np.eye(4, dtype=np.float64)]

        revision, _ = slam.get_keyframe_pose_snapshot()
        self.assertEqual(revision, 5)
        slam._mark_geometry_corrected()
        revision, _ = slam.get_keyframe_pose_snapshot()
        self.assertEqual(revision, 6)
        self.assertTrue(slam._geometry_corrected_in_process)

    def test_pose_snapshots_do_not_alias_live_history(self):
        slam = StereoSLAMv2.__new__(StereoSLAMv2)
        slam.geometry_revision = 2
        slam.poses = [np.eye(4, dtype=np.float64)]

        revision, poses = slam.get_pose_history_snapshot()
        slam.poses[0][0, 3] = 8.0

        self.assertEqual(revision, 2)
        self.assertEqual(float(poses[0][0, 3]), 0.0)
        with self.assertRaises(ValueError):
            poses[0][0, 3] = 1.0


if __name__ == "__main__":
    unittest.main()
