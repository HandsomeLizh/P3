"""Numerical regression tests for SlidingWindowBA without loading RoMa/Torch."""

import ast
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares


ROMA_PATH = Path(__file__).resolve().parents[1] / "roma_vo.py"


def _load_ba_classes():
    """Execute only the data classes and BA class from roma_vo.py."""
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    wanted = {"MapPoint", "KeyFrame", "SlidingWindowBA"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "np": np,
        "cv2": cv2,
        "least_squares": least_squares,
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "field": field,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return (
        namespace["MapPoint"],
        namespace["KeyFrame"],
        namespace["SlidingWindowBA"],
    )


MapPoint, KeyFrame, SlidingWindowBA = _load_ba_classes()


def _project(point_world, pose, K, baseline):
    point_camera = pose[:3, :3].T @ (point_world - pose[:3, 3])
    x, y, z = point_camera
    left = np.array([
        K[0, 0] * x / z + K[0, 2],
        K[1, 1] * y / z + K[1, 2],
    ])
    right = np.array([
        K[0, 0] * (x - baseline) / z + K[0, 2],
        left[1],
    ])
    return left, right


def _scaled_scene(scale=1.7):
    K = np.array([
        [140.0, 0.0, 160.0],
        [0.0, 140.0, 120.0],
        [0.0, 0.0, 1.0],
    ])
    baseline = 0.2
    true_poses = []
    for x in (0.0, 0.3, 0.6):
        pose = np.eye(4)
        pose[0, 3] = x
        true_poses.append(pose)

    true_points = []
    for row in range(4):
        for col in range(5):
            true_points.append(np.array([
                -0.8 + 0.4 * col,
                -0.45 + 0.3 * row,
                4.0 + 0.15 * ((row + col) % 3),
            ]))

    mappoints = [
        MapPoint(id=index, position=point * scale)
        for index, point in enumerate(true_points)
    ]
    keyframes = []
    for kf_id, true_pose in enumerate(true_poses):
        initial_pose = true_pose.copy()
        initial_pose[:3, 3] *= scale
        left_points, right_points = [], []
        for mp, true_point in zip(mappoints, true_points):
            left, right = _project(true_point, true_pose, K, baseline)
            left_points.append(left)
            right_points.append(right)
            mp.add_observation(kf_id, left)
        keyframes.append(KeyFrame(
            id=kf_id,
            timestamp=float(kf_id),
            pose=initial_pose,
            keypoints=left_points,
            right_keypoints=right_points,
            mappoints=list(mappoints),
            K=K,
        ))
    return K, baseline, true_poses, true_points, mappoints, keyframes


class StereoBATest(unittest.TestCase):
    def test_stereo_residual_recovers_metric_scale_and_keeps_anchor(self):
        K, baseline, true_poses, _points, _mps, keyframes = _scaled_scene()
        anchor_before = keyframes[0].pose.copy()
        ba = SlidingWindowBA(window_size=5, K=K, baseline=baseline)
        for keyframe in keyframes:
            ba.add_keyframe(keyframe)

        error, inliers = ba.optimize(iterations=12)

        self.assertTrue(ba.last_accepted, ba.last_failure_reason)
        self.assertEqual(ba.last_mode, "stereo_joint")
        np.testing.assert_array_equal(keyframes[0].pose, anchor_before)
        self.assertGreaterEqual(inliers, 50)
        self.assertLess(error, 1e-3)
        self.assertAlmostEqual(
            keyframes[1].pose[0, 3], true_poses[1][0, 3], delta=0.03)
        self.assertAlmostEqual(
            keyframes[2].pose[0, 3], true_poses[2][0, 3], delta=0.05)

    def test_window_eviction_preserves_keyframe_and_observation_history(self):
        K = np.eye(3)
        mp = MapPoint(id=4, position=np.array([0.0, 0.0, 2.0]))
        keyframes = []
        for kf_id in range(3):
            point = np.array([0.0, 0.0])
            mp.add_observation(kf_id, point)
            keyframes.append(KeyFrame(
                id=kf_id,
                timestamp=float(kf_id),
                pose=np.eye(4),
                keypoints=[point],
                right_keypoints=[point],
                mappoints=[mp],
                K=K,
            ))
        ba = SlidingWindowBA(window_size=2, K=K, baseline=0.2)
        for keyframe in keyframes:
            ba.add_keyframe(keyframe)

        self.assertEqual([kf.id for kf in ba.keyframes], [1, 2])
        self.assertFalse(keyframes[0].is_bad)
        self.assertIn(0, mp.observations)
        self.assertIn(mp.id, ba.mappoints)

    def test_final_threshold_rejects_bad_observation_without_round_overcount(self):
        K, baseline, _poses, _points, mappoints, keyframes = _scaled_scene(
            scale=1.0)
        keyframes[2].keypoints[0] = keyframes[2].keypoints[0] + np.array(
            [80.0, -60.0])
        keyframes[2].right_keypoints[0] = (
            keyframes[2].right_keypoints[0] + np.array([80.0, -60.0]))
        ba = SlidingWindowBA(window_size=5, K=K, baseline=baseline)
        for keyframe in keyframes:
            ba.add_keyframe(keyframe)

        _error, inliers = ba.optimize(iterations=10)

        self.assertTrue(ba.last_accepted, ba.last_failure_reason)
        self.assertLess(inliers, 60)
        self.assertGreaterEqual(inliers, 58)
        # 该点的多数观测最终被拒绝，但一次 BA 只能累计一次，而不是按
        # 三轮优化累计三次；达到 5 次独立失败前也不会直接标坏。
        self.assertEqual(mappoints[0].ba_outlier_count, 1)
        self.assertFalse(mappoints[0].is_bad)

    def test_optimizer_exception_has_zero_writeback(self):
        K, baseline, _poses, _points, mappoints, keyframes = _scaled_scene()
        ba = SlidingWindowBA(window_size=5, K=K, baseline=baseline)
        for keyframe in keyframes:
            ba.add_keyframe(keyframe)
        poses_before = [kf.pose.copy() for kf in keyframes]
        points_before = [mp.position.copy() for mp in mappoints]

        globals_dict = SlidingWindowBA.optimize.__globals__
        original_solver = globals_dict["least_squares"]
        globals_dict["least_squares"] = lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(RuntimeError("synthetic failure")))
        try:
            error, inliers = ba.optimize(iterations=2)
        finally:
            globals_dict["least_squares"] = original_solver

        self.assertFalse(ba.last_accepted)
        self.assertEqual((error, inliers), (0.0, 0))
        self.assertIn("optimizer_failed", ba.last_failure_reason)
        for keyframe, before in zip(keyframes, poses_before):
            np.testing.assert_array_equal(keyframe.pose, before)
        for mappoint, before in zip(mappoints, points_before):
            np.testing.assert_array_equal(mappoint.position, before)


if __name__ == "__main__":
    unittest.main()
