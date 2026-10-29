"""Model-free regression tests for tracking-loss relocalization commits."""

import ast
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy.spatial import cKDTree


ROMA_PATH = Path(__file__).resolve().parents[1] / "roma_vo.py"


def _load_relocalization_contract():
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    data_names = {
        "MapPoint",
        "KeyFrame",
        "TrackingStatus",
        "LoopKeyFrameSnapshot",
        "LoopConstraint",
    }
    data_nodes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in data_names
    ]
    snapshot_builder = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "make_keyframe_geometry_snapshot"
    )
    slam = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StereoSLAMv2"
    )
    method_names = {
        "_set_tracking_status",
        "get_tracking_status",
        "_snapshot_source_index",
        "_relocalization_associations",
        "_commit_relocalized_frame",
        "_try_relocalize",
        "_handle_tracking_failure",
        "_create_keyframe",
        "trim_keyframe_images",
        "_update_kf_cache",
        "_build_klt_indices",
    }
    miniature = ast.ClassDef(
        name="StereoSLAMv2",
        bases=[],
        keywords=[],
        body=[
            node for node in slam.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(
        body=data_nodes + [snapshot_builder, miniature],
        type_ignores=[],
    ))
    namespace = {
        "np": np,
        "time": time,
        "dataclass": dataclass,
        "field": field,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "cv2": cv2,
        "cKDTree": cKDTree,
    }
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return namespace


CONTRACT = _load_relocalization_contract()
MapPoint = CONTRACT["MapPoint"]
KeyFrame = CONTRACT["KeyFrame"]
TrackingStatus = CONTRACT["TrackingStatus"]
LoopKeyFrameSnapshot = CONTRACT["LoopKeyFrameSnapshot"]
LoopConstraint = CONTRACT["LoopConstraint"]
StereoSLAMv2 = CONTRACT["StereoSLAMv2"]


class _FakeRelocalizer:
    def __init__(self, snapshot, constraint):
        self.kfs = {snapshot.id: snapshot}
        self.snapshot = snapshot
        self.constraint = constraint
        self.allowed_ids = None

    def query(self, _current, allowed_ids=None):
        self.allowed_ids = set(allowed_ids or ())
        return self.constraint

    def get_snapshot(self, keyframe_id):
        if keyframe_id == self.snapshot.id:
            return self.snapshot
        return None


def _snapshot(keyframe_id, source_indices):
    count = len(source_indices)
    return LoopKeyFrameSnapshot(
        id=keyframe_id,
        pose=np.eye(4, dtype=np.float64),
        keypoints=np.zeros((count, 2), dtype=np.float64),
        descriptors=np.ones((count, 32), dtype=np.uint8),
        camera_points=np.column_stack([
            np.zeros(count), np.zeros(count), np.full(count, 4.0)
        ]),
        image_size=(48, 64),
        source_indices=np.asarray(source_indices, dtype=np.int32),
    )


def _base_slam(candidate, map_point, relocalizer):
    slam = StereoSLAMv2.__new__(StereoSLAMv2)
    slam.K = np.array([
        [100.0, 0.0, 32.0],
        [0.0, 100.0, 24.0],
        [0.0, 0.0, 1.0],
    ])
    slam.poses = [np.eye(4, dtype=np.float64)]
    slam._pose_reference_kf_ids = [candidate.id]
    slam._pose_relative_to_kf = [np.eye(4, dtype=np.float64)]
    slam.keyframes = [candidate]
    slam.global_mps = {map_point.id: map_point}
    slam.next_mp_id = map_point.id + 1
    slam.next_kf_id = candidate.id + 1
    slam.kf_count = 1
    slam.frame_count = 8
    slam.geometry_revision = 0
    slam._geometry_corrected_in_process = False
    slam.last_tracking_status = TrackingStatus()
    slam._kf_id_to_pose_idx = {candidate.id: 0}
    slam._kf_img = np.full((2, 2), 7, dtype=np.uint8)
    slam._kf_pts3d = np.full((1, 3), 7.0)
    slam._kf_kpts = np.full((1, 2), 7.0)
    slam._kf_mp_ids = [map_point.id]
    slam._last_kf = candidate
    slam._last_kf_frame_idx = 1
    slam.relocalizer = relocalizer
    slam.relocalization_count = 0
    slam.tracking_state = "LOST"
    slam.enable_backend = False
    slam.klt_image_scale = 0.5
    slam.klt_max_points = 1200
    class ModelFreeMatcher:
        @staticmethod
        def freeze_feature_cache(cache):
            return cache
    slam.matcher = ModelFreeMatcher()
    return slam


class RelocalizationContractTest(unittest.TestCase):
    def _scene(self, constraint):
        map_point = MapPoint(
            id=7, position=np.array([10.0, 0.0, 4.0]))
        map_point.add_observation(0, np.array([20.0, 20.0]))
        candidate_pose = np.eye(4, dtype=np.float64)
        candidate_pose[0, 3] = 10.0
        candidate = KeyFrame(
            id=0,
            timestamp=0.0,
            pose=candidate_pose,
            keypoints=[[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
            camera_points=[
                [0.0, 0.0, 4.0],
                [0.1, 0.0, 4.0],
                [0.2, 0.0, 4.0],
            ],
            mappoints=[None, None, map_point],
        )
        historical_snapshot = _snapshot(0, [2])
        relocalizer = _FakeRelocalizer(historical_snapshot, constraint)
        return _base_slam(candidate, map_point, relocalizer)

    def test_verified_recovery_uses_live_pose_and_forces_new_anchor(self):
        relative = np.eye(4, dtype=np.float64)
        relative[0, 3] = 0.5
        constraint = LoopConstraint(
            current_id=1,
            loop_id=0,
            T_loop_current=relative,
            descriptor_matches=42,
            inliers=36,
            inlier_ratio=36 / 42,
            reprojection_rmse=0.4,
            appearance_score=0.9,
            loop_feature_indices=np.array([0], dtype=np.int32),
            current_feature_indices=np.array([0], dtype=np.int32),
        )
        slam = self._scene(constraint)
        current_snapshot = _snapshot(1, [1])
        descriptors = np.ones((3, 32), dtype=np.uint8)
        slam._make_relocalization_query = lambda *_args: (
            current_snapshot, descriptors)
        captured = {}

        def create_keyframe(
                image, points_camera, _points_world, keypoints,
                right_keypoints, pose, associations, descriptors=None):
            captured["associations"] = dict(associations)
            captured["descriptors"] = descriptors.copy()
            keyframe = KeyFrame(
                id=slam.next_kf_id,
                timestamp=1.0,
                pose=pose.copy(),
                image=image.copy(),
                keypoints=keypoints.tolist(),
                right_keypoints=right_keypoints.tolist(),
                camera_points=points_camera.tolist(),
                descriptors=descriptors.copy(),
            )
            slam.keyframes.append(keyframe)
            slam.next_kf_id += 1
            slam.kf_count += 1
            slam._last_kf = keyframe
            slam._last_kf_frame_idx = slam.frame_count
            return keyframe

        slam._create_keyframe = create_keyframe
        slam._submit_keyframe_to_backend = lambda keyframe: keyframe.pose.copy()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        points = np.array([
            [0.0, 0.0, 4.0],
            [0.1, 0.0, 4.0],
            [0.2, 0.0, 4.0],
        ])
        keypoints = np.array([
            [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]
        ])

        recovered = slam._handle_tracking_failure(
            reason="pnp_failed",
            left=image,
            pts3d_cam=points,
            kp_l_valid=keypoints,
            kp_r_valid=keypoints - np.array([1.0, 0.0]),
            correspondence_count=12,
        )

        self.assertAlmostEqual(recovered[0, 3], 10.5)
        self.assertAlmostEqual(slam.poses[-1][0, 3], 10.5)
        self.assertEqual(captured["associations"], {1: 7})
        np.testing.assert_array_equal(captured["descriptors"], descriptors)
        self.assertEqual(slam.relocalizer.allowed_ids, {0})
        self.assertEqual(slam._last_kf_frame_idx, slam.frame_count)
        self.assertEqual(slam._pose_reference_kf_ids[-1], 1)
        self.assertEqual(slam.tracking_state, "TRACKING")
        self.assertEqual(slam.relocalization_count, 1)
        np.testing.assert_array_equal(slam._kf_img, image)
        np.testing.assert_array_equal(slam._kf_pts3d, points)
        self.assertEqual(slam._kf_gray.shape, (24, 32))
        status = slam.get_tracking_status()
        self.assertTrue(status.success)
        self.assertEqual(status.reason, "relocalized")
        self.assertTrue(status.keyframe_created)
        self.assertEqual(status.keyframe_id, 1)
        self.assertEqual(status.inlier_count, 36)

    def test_rejected_recovery_does_not_change_pose_or_map(self):
        constraint = None
        slam = self._scene(constraint)
        current_snapshot = _snapshot(1, [1])
        slam._make_relocalization_query = lambda *_args: (
            current_snapshot, np.ones((3, 32), dtype=np.uint8))
        old_pose = slam.poses[-1].copy()
        old_keyframes = tuple(slam.keyframes)
        old_map_positions = {
            key: value.position.copy() for key, value in slam.global_mps.items()
        }
        old_observations = {
            key: dict(value.observations) for key, value in slam.global_mps.items()
        }
        old_cache = (
            slam._kf_img.copy(), slam._kf_pts3d.copy(), slam._kf_kpts.copy())

        returned = slam._handle_tracking_failure(
            reason="pnp_failed",
            left=np.zeros((48, 64, 3), dtype=np.uint8),
            pts3d_cam=np.ones((3, 3), dtype=np.float64),
            kp_l_valid=np.ones((3, 2), dtype=np.float64),
            kp_r_valid=np.ones((3, 2), dtype=np.float64),
        )

        np.testing.assert_array_equal(returned, old_pose)
        self.assertEqual(tuple(slam.keyframes), old_keyframes)
        self.assertEqual(len(slam.poses), 1)
        for key, mp in slam.global_mps.items():
            np.testing.assert_array_equal(mp.position, old_map_positions[key])
            self.assertEqual(mp.observations.keys(), old_observations[key].keys())
            for frame_id, point in mp.observations.items():
                np.testing.assert_array_equal(
                    point, old_observations[key][frame_id])
        np.testing.assert_array_equal(slam._kf_img, old_cache[0])
        np.testing.assert_array_equal(slam._kf_pts3d, old_cache[1])
        np.testing.assert_array_equal(slam._kf_kpts, old_cache[2])
        self.assertEqual(slam.tracking_state, "LOST")
        self.assertFalse(slam.get_tracking_status().success)

    def _throttle_scene(self):
        slam = self._scene(None)
        current_snapshot = _snapshot(1, [1])
        slam._make_relocalization_query = lambda *_args: (
            current_snapshot, np.ones((3, 32), dtype=np.uint8))
        attempts = []
        original_query = slam.relocalizer.query

        def counting_query(current, allowed_ids=None):
            attempts.append(slam.frame_count)
            return original_query(current, allowed_ids=allowed_ids)

        slam.relocalizer.query = counting_query
        return slam, attempts

    def _fail_frame(self, slam, frame):
        slam.frame_count = frame
        slam._handle_tracking_failure(
            reason="pnp_failed",
            left=np.zeros((48, 64, 3), dtype=np.uint8),
            pts3d_cam=np.ones((3, 3), dtype=np.float64),
            kp_l_valid=np.ones((3, 2), dtype=np.float64),
            kp_r_valid=np.ones((3, 2), dtype=np.float64),
        )

    def test_relocalization_query_is_throttled_between_attempts(self):
        slam, attempts = self._throttle_scene()
        slam.relocalization_min_interval = 3
        for frame in range(8, 14):
            self._fail_frame(slam, frame)
        # 间隔不足 3 帧的失败帧不得触发全库检索。
        self.assertEqual(attempts, [8, 11])

    def test_relocalization_default_interval_keeps_every_frame(self):
        slam, attempts = self._throttle_scene()
        for frame in range(8, 11):
            self._fail_frame(slam, frame)
        # 未配置节流（默认 1）时保持原行为：每个失败帧都尝试。
        self.assertEqual(attempts, [8, 9, 10])

    def test_keyframe_preparation_failure_is_zero_write(self):
        constraint = None
        slam = self._scene(constraint)
        store = {}

        class SnapshotStore:
            def add_snapshot(self, snapshot):
                store[snapshot.id] = snapshot

        slam.relocalizer = SnapshotStore()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
        keypoints = np.array([[20.0, 20.0], [30.0, 30.0]])
        before = (
            len(slam.keyframes), len(slam.global_mps),
            slam.next_kf_id, slam.next_mp_id)

        with self.assertRaises(ValueError):
            slam._create_keyframe(
                image,
                points,
                points.copy(),
                keypoints,
                keypoints.copy(),
                np.eye(4),
                {},
                descriptors=np.ones((1, 32), dtype=np.uint8),
            )

        self.assertEqual(
            (len(slam.keyframes), len(slam.global_mps),
             slam.next_kf_id, slam.next_mp_id),
            before)
        self.assertEqual(store, {})

    def test_committed_keyframe_reuses_point_and_registers_snapshot(self):
        slam = self._scene(constraint=None)
        snapshots = {}

        class SnapshotStore:
            def add_snapshot(self, snapshot):
                snapshots[snapshot.id] = snapshot

        slam.relocalizer = SnapshotStore()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        points = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]])
        keypoints = np.array([[20.0, 20.0], [30.0, 30.0]])
        right_keypoints = keypoints - np.array([1.0, 0.0])
        descriptors = np.ones((2, 32), dtype=np.uint8)

        keyframe = slam._create_keyframe(
            image,
            points,
            points.copy(),
            keypoints,
            right_keypoints,
            np.eye(4),
            {0: 7},
            descriptors=descriptors,
        )

        self.assertEqual(keyframe.id, 1)
        self.assertIs(keyframe.mappoints[0], slam.global_mps[7])
        self.assertIn(1, slam.global_mps[7].observations)
        self.assertEqual(len(slam.global_mps), 2)
        self.assertIn(1, snapshots)
        np.testing.assert_array_equal(
            snapshots[1].source_indices, np.array([0, 1], dtype=np.int32))
        self.assertFalse(snapshots[1].source_indices.flags.writeable)
        self.assertEqual(slam._last_kf_frame_idx, slam.frame_count)


if __name__ == "__main__":
    unittest.main()
