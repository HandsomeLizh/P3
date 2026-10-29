"""Numerical tests for independent, thread-isolated loop closing."""

import ast
import queue
import threading
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares


ROMA_PATH = Path(__file__).resolve().parents[1] / "roma_vo.py"


def _load_loop_classes():
    wanted = {
        "MapPoint",
        "KeyFrame",
        "LoopKeyFrameSnapshot",
        "LoopConstraint",
        "PoseGraphEdge",
        "LoopClosureResult",
        "LoopClosing",
        "Relocalizer",
        "BackendOptimizer",
    }
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if ((isinstance(node, ast.ClassDef) and node.name in wanted)
            or (isinstance(node, ast.FunctionDef)
                and node.name == "make_keyframe_geometry_snapshot"))
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "np": np,
        "cv2": cv2,
        "least_squares": least_squares,
        "queue": queue,
        "threading": threading,
        "dataclass": dataclass,
        "field": field,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return namespace


CLASSES = _load_loop_classes()
MapPoint = CLASSES["MapPoint"]
KeyFrame = CLASSES["KeyFrame"]
LoopKeyFrameSnapshot = CLASSES["LoopKeyFrameSnapshot"]
LoopConstraint = CLASSES["LoopConstraint"]
LoopClosureResult = CLASSES["LoopClosureResult"]
PoseGraphEdge = CLASSES["PoseGraphEdge"]
LoopClosing = CLASSES["LoopClosing"]
Relocalizer = CLASSES["Relocalizer"]
BackendOptimizer = CLASSES["BackendOptimizer"]


def _load_loop_commit_methods():
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    slam = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StereoSLAMv2"
    )
    wanted = {
        "_mark_geometry_corrected",
        "_sync_pose_history_from_keyframes",
        "_refresh_last_kf_cache_from_map",
        "_apply_pending_loop_results",
    }
    methods = [
        node for node in slam.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    miniature = ast.ClassDef(
        name="MiniStereoSLAM",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[miniature], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return namespace["MiniStereoSLAM"]


MiniStereoSLAM = _load_loop_commit_methods()


def _transform_points(transform, points):
    return (
        transform[:3, :3] @ points.T
        + transform[:3, 3:4]
    ).T


def _project(K, points):
    return np.column_stack([
        K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
        K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
    ])


def _loop_pair(count=60):
    K = np.array([
        [300.0, 0.0, 320.0],
        [0.0, 300.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    rng = np.random.default_rng(12)
    loop_points = np.column_stack([
        rng.uniform(-1.6, 1.6, count),
        rng.uniform(-1.1, 1.1, count),
        rng.uniform(4.0, 7.0, count),
    ])
    angle = np.deg2rad(4.0)
    T_loop_current = np.eye(4)
    T_loop_current[:3, :3] = cv2.Rodrigues(
        np.array([0.0, angle, 0.0]))[0]
    T_loop_current[:3, 3] = np.array([0.25, -0.04, 0.08])
    T_current_loop = np.linalg.inv(T_loop_current)
    current_points = _transform_points(T_current_loop, loop_points)

    descriptors = rng.integers(1, 256, size=(count, 32), dtype=np.uint8)
    loop_pose_estimate = np.eye(4)
    # Deliberately absurd odometry drift: candidate search/measurement must not
    # depend on this global distance.
    current_pose_estimate = np.eye(4)
    current_pose_estimate[0, 3] = 40.0
    loop = LoopKeyFrameSnapshot(
        id=0,
        pose=loop_pose_estimate,
        keypoints=_project(K, loop_points),
        descriptors=descriptors.copy(),
        camera_points=loop_points,
        image_size=(480, 640),
    )
    current = LoopKeyFrameSnapshot(
        id=20,
        pose=current_pose_estimate,
        keypoints=_project(K, current_points),
        descriptors=descriptors.copy(),
        camera_points=current_points,
        image_size=(480, 640),
    )
    return K, T_loop_current, loop, current


class LoopClosingTest(unittest.TestCase):
    def test_independent_geometry_recovers_loop_despite_large_pose_drift(self):
        K, expected, loop, current = _loop_pair()
        closing = LoopClosing(
            queue.Queue(), K,
            min_dist_thresh=3.0,
            min_interval=15,
            min_descriptor_matches=20,
            min_pnp_inliers=15,
        )
        closing.kfs[loop.id] = loop

        constraint = closing._detect(current)

        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.loop_id, 0)
        self.assertGreaterEqual(constraint.inliers, 55)
        self.assertEqual(
            len(constraint.loop_feature_indices), constraint.inliers)
        self.assertEqual(
            len(constraint.current_feature_indices), constraint.inliers)
        self.assertFalse(constraint.loop_feature_indices.flags.writeable)
        self.assertFalse(constraint.current_feature_indices.flags.writeable)
        np.testing.assert_allclose(
            constraint.T_loop_current, expected, atol=2e-3)

    def test_relocalizer_accepts_recent_keyframe_without_backend_thread(self):
        K, expected, loop, current = _loop_pair()
        recent_current = LoopKeyFrameSnapshot(
            id=1,
            pose=current.pose,
            keypoints=current.keypoints,
            descriptors=current.descriptors,
            camera_points=current.camera_points,
            image_size=current.image_size,
        )
        relocalizer = Relocalizer(K, max_relative_translation=3.0)
        relocalizer.add_snapshot(loop)

        constraint = relocalizer.query(recent_current, allowed_ids={0})

        self.assertIsNotNone(constraint)
        self.assertIsNone(relocalizer.geometry.thread)
        self.assertFalse(relocalizer.geometry.running)
        np.testing.assert_allclose(
            constraint.T_loop_current, expected, atol=2e-3)

    def test_relocalizer_rejects_appearance_only_candidate(self):
        K, _expected, loop, current = _loop_pair()
        rng = np.random.default_rng(123)
        bad_current = LoopKeyFrameSnapshot(
            id=1,
            pose=current.pose,
            keypoints=np.column_stack([
                rng.uniform(0.0, 639.0, len(current.keypoints)),
                rng.uniform(0.0, 479.0, len(current.keypoints)),
            ]),
            descriptors=current.descriptors,
            camera_points=current.camera_points,
            image_size=current.image_size,
        )
        relocalizer = Relocalizer(K)
        relocalizer.add_snapshot(loop)

        self.assertIsNone(relocalizer.query(bad_current))

    def test_appearance_match_without_geometry_is_rejected(self):
        K, _expected, loop, current = _loop_pair()
        rng = np.random.default_rng(99)
        bad_current = LoopKeyFrameSnapshot(
            id=current.id,
            pose=current.pose,
            keypoints=np.column_stack([
                rng.uniform(0.0, 639.0, len(current.keypoints)),
                rng.uniform(0.0, 479.0, len(current.keypoints)),
            ]),
            descriptors=current.descriptors,
            camera_points=current.camera_points,
            image_size=current.image_size,
        )
        closing = LoopClosing(
            queue.Queue(), K,
            min_descriptor_matches=20,
            min_pnp_inliers=15,
        )

        self.assertIsNone(closing._verify(loop, bad_current))

    def test_background_worker_emits_result_without_live_state_access(self):
        K, expected, loop, current = _loop_pair()
        task_queue = queue.Queue()
        closing = LoopClosing(
            task_queue, K,
            min_dist_thresh=3.0,
            min_interval=15,
            min_descriptor_matches=20,
            min_pnp_inliers=15,
        )
        results = []
        closing.on_loop_result = results.append
        closing.start()
        task_queue.put(loop)
        task_queue.put(current)
        closing.stop()

        self.assertFalse(closing.running)
        self.assertEqual(len(results), 1)
        np.testing.assert_allclose(
            results[0].constraint.T_loop_current, expected, atol=2e-3)

    def test_pose_graph_uses_frozen_edges_and_keeps_anchor(self):
        K = np.eye(3)
        closing = LoopClosing(queue.Queue(), K, min_pnp_inliers=8)
        poses = {}
        for kid, x in enumerate((0.0, 1.0, 2.0)):
            pose = np.eye(4)
            pose[0, 3] = x
            poses[kid] = pose
        closing.pose_estimates = {kid: pose.copy() for kid, pose in poses.items()}
        for first, second in ((0, 1), (1, 2)):
            measurement = np.eye(4)
            measurement[0, 3] = 1.0
            closing.odometry_edges.append(PoseGraphEdge(
                first, second, measurement, 1.0, "odometry"))
        loop_measurement = np.eye(4)
        loop_measurement[0, 3] = 0.2
        closing.loop_edges.append(PoseGraphEdge(
            0, 2, loop_measurement, 5.0, "loop"))

        result = closing._pgo_scipy()

        self.assertIsNotNone(result)
        corrections, optimized = result
        np.testing.assert_array_equal(optimized[0], poses[0])
        self.assertLess(optimized[2][0, 3], 0.5)
        self.assertIn(2, corrections)
        np.testing.assert_allclose(
            closing.odometry_edges[0].measurement,
            np.array([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        )

    def test_backend_snapshot_is_detached_from_live_keyframe(self):
        K = np.array([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ])
        mp = MapPoint(id=1, position=np.array([0.0, 0.0, 4.0]))
        mp.add_observation(2, np.array([50.0, 40.0]))
        keyframe = KeyFrame(
            id=2,
            timestamp=0.0,
            pose=np.eye(4),
            image=np.zeros((80, 100, 3), dtype=np.uint8),
            keypoints=[np.array([50.0, 40.0])],
            camera_points=[np.array([0.0, 0.0, 3.5])],
            descriptors=np.ones((1, 32), dtype=np.uint8),
            mappoints=[mp],
            K=K,
        )
        backend = BackendOptimizer(K, enable_loop_closing=False)
        snapshot = backend._make_loop_snapshot(keyframe)
        keyframe.pose[0, 3] = 9.0
        mp.position[2] = 10.0

        self.assertEqual(snapshot.pose[0, 3], 0.0)
        self.assertEqual(snapshot.camera_points[0, 2], 3.5)
        self.assertEqual(snapshot.image_size, (80, 100))
        self.assertEqual(int(snapshot.source_indices[0]), 0)
        self.assertFalse(snapshot.source_indices.flags.writeable)
        with self.assertRaises(ValueError):
            snapshot.pose[0, 3] = 1.0

    def test_odometry_edge_is_frozen_in_one_frontend_coordinate_state(self):
        K = np.eye(3)
        backend = BackendOptimizer(K, enable_loop_closing=True)
        first = KeyFrame(id=0, timestamp=0.0, pose=np.eye(4), K=K)
        backend.add_keyframe(first)
        backend.lc_queue.get_nowait()

        # Simulate an already-applied world correction before the next KF.
        first.pose[0, 3] = 10.0
        second_pose = np.eye(4)
        second_pose[0, 3] = 11.0
        second = KeyFrame(id=1, timestamp=1.0, pose=second_pose, K=K)
        backend.add_keyframe(second)
        snapshot = backend.lc_queue.get_nowait()
        first.pose[0, 3] = 100.0
        second.pose[0, 3] = 200.0

        self.assertEqual(snapshot.previous_id, 0)
        self.assertAlmostEqual(snapshot.T_previous_current[0, 3], 1.0)
        with self.assertRaises(ValueError):
            snapshot.T_previous_current[0, 3] = 2.0

    def test_loop_result_is_committed_once_on_main_state(self):
        K = np.eye(3)
        kf0 = KeyFrame(id=0, timestamp=0.0, pose=np.eye(4), K=K)
        pose20 = np.eye(4)
        pose20[0, 3] = 2.0
        mp20 = MapPoint(id=20, position=np.array([2.0, 0.0, 5.0]))
        mp20.add_observation(20, np.array([0.0, 0.0]))
        kf20 = KeyFrame(
            id=20,
            timestamp=20.0,
            pose=pose20,
            keypoints=[np.array([0.0, 0.0])],
            mappoints=[mp20],
            K=K,
        )
        # A tail keyframe was created while the worker optimized through KF20.
        pose21 = np.eye(4)
        pose21[0, 3] = 3.0
        mp21 = MapPoint(id=21, position=np.array([3.0, 0.0, 5.0]))
        mp21.add_observation(21, np.array([0.0, 0.0]))
        kf21 = KeyFrame(
            id=21,
            timestamp=21.0,
            pose=pose21,
            keypoints=[np.array([0.0, 0.0])],
            mappoints=[mp21],
            K=K,
        )

        correction20 = np.eye(4)
        correction20[0, 3] = -1.0
        constraint = LoopConstraint(
            current_id=20,
            loop_id=0,
            T_loop_current=np.eye(4),
            descriptor_matches=50,
            inliers=45,
            inlier_ratio=0.9,
            reprojection_rmse=0.3,
            appearance_score=0.8,
        )
        result = LoopClosureResult(
            constraint=constraint,
            corrections={0: np.eye(4), 20: correction20},
            optimized_poses={0: np.eye(4), 20: correction20 @ pose20},
        )

        class Backend:
            def __init__(self):
                self.results = [result]

            def consume_loop_results(self):
                values, self.results = self.results, []
                return values

        slam = MiniStereoSLAM()
        slam.backend = Backend()
        slam.keyframes = [kf0, kf20, kf21]
        slam.global_mps = {20: mp20, 21: mp21}
        slam.poses = [np.eye(4), pose20.copy(), pose21.copy()]
        slam._pose_reference_kf_ids = [0, 20, 21]
        slam._pose_relative_to_kf = [np.eye(4), np.eye(4), np.eye(4)]
        slam._last_kf = kf21
        slam._kf_pts3d = np.array([[0.0, 0.0, 5.0]])
        slam.loop_count = 0
        slam.geometry_revision = 0

        slam._apply_pending_loop_results()

        self.assertEqual(slam.loop_count, 1)
        self.assertAlmostEqual(kf20.pose[0, 3], 1.0)
        self.assertAlmostEqual(kf21.pose[0, 3], 2.0)
        self.assertAlmostEqual(mp20.position[0], 1.0)
        self.assertAlmostEqual(mp21.position[0], 2.0)
        self.assertAlmostEqual(slam.poses[1][0, 3], 1.0)
        self.assertAlmostEqual(slam.poses[2][0, 3], 2.0)
        self.assertEqual(slam.geometry_revision, 1)
        # Polling again cannot apply the same correction twice.
        slam._apply_pending_loop_results()
        self.assertAlmostEqual(kf20.pose[0, 3], 1.0)
        self.assertEqual(slam.geometry_revision, 1)


if __name__ == "__main__":
    unittest.main()
