#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.localization.global_pose_graph import (
    GtsamGlobalPoseGraph,
)
from t3_semantic_mapping.localization.lidar_loop_closure import (
    LidarLoopClosureDetector,
    LidarLoopConfiguration,
    scan_context_descriptor,
    scan_context_similarity,
    trimmed_point_to_point_icp,
)
from t3_semantic_mapping.localization.loop_constraints import (
    LoopCandidate,
    LoopConstraint,
)


def _pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    result[:3, 3] = [x, y, z]
    return result


def _structured_cloud(seed=4, count=800):
    generator = np.random.default_rng(seed)
    points = generator.uniform([-5.0, -4.0, -0.4], [5.0, 4.0, 1.2], (count, 3))
    points[:, 2] += 0.25 * np.sin(points[:, 0]) + 0.15 * points[:, 1]
    return points


class LoopConstraintContractTest(unittest.TestCase):
    def test_candidate_and_constraint_are_immutable_snapshots(self):
        coarse = _pose(1.0)
        candidate = LoopCandidate(0, 5, "test", 0.8, coarse)
        coarse[0, 3] = 99.0
        self.assertAlmostEqual(candidate.coarse_relative_pose[0, 3], 1.0)

        constraint = LoopConstraint(
            0,
            5,
            _pose(0.1),
            np.full(6, 0.1),
            "lidar_icp",
            0.9,
        )
        self.assertFalse(constraint.relative_pose_reference_query.flags.writeable)
        self.assertFalse(constraint.sigmas.flags.writeable)

    def test_rejects_self_loop_and_invalid_noise(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            LoopCandidate(2, 2, "bad", 0.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            LoopConstraint(
                0, 1, np.eye(4), np.zeros(6), "bad", 0.5
            )


class LidarLoopGeometryTest(unittest.TestCase):
    def test_scan_context_is_invariant_to_sector_rotation(self):
        cloud = _structured_cloud()
        rotated = cloud @ Rotation.from_euler("z", 0.7).as_matrix().T
        first = scan_context_descriptor(
            cloud, rings=20, sectors=60, max_radius_m=10.0
        )
        second = scan_context_descriptor(
            rotated, rings=20, sectors=60, max_radius_m=10.0
        )
        similarity, _ = scan_context_similarity(first, second)
        # A non-integer sector rotation incurs polar-bin quantisation, but the
        # cyclic search must still preserve a strong retrieval score.
        self.assertGreater(similarity, 0.80)

    def test_trimmed_icp_recovers_metric_transform(self):
        source = _structured_cloud(count=1000)
        truth = _pose(0.35, -0.20, 0.08, yaw=0.06)
        target = source @ truth[:3, :3].T + truth[:3, 3]
        initial = _pose(0.30, -0.15, 0.05, yaw=0.04)
        result = trimmed_point_to_point_icp(
            source_points=source,
            target_points=target,
            initial_transform_target_source=initial,
            max_correspondence_m=0.5,
            max_iterations=40,
            trim_fraction=0.8,
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(
            result.transform_target_source, truth, atol=2.0e-3
        )
        self.assertGreater(result.overlap, 0.70)
        self.assertLess(result.rmse_m, 2.0e-3)

    def test_detector_requires_metric_icp_before_emitting_factor(self):
        configuration = LidarLoopConfiguration(
            enabled=True,
            submap_keyframes=1,
            voxel_size_m=0.05,
            max_points=1200,
            descriptor_min_similarity=0.1,
            descriptor_shortlist=2,
            min_keyframe_separation=1,
            min_time_separation_sec=0.0,
            max_candidates=1,
            icp_max_correspondence_m=0.5,
            icp_min_inliers=100,
            icp_min_overlap=0.5,
            icp_min_unique_target_ratio=0.8,
            icp_max_rmse_m=0.05,
            max_correction_translation_m=0.5,
            confirmation_count=1,
        )
        detector = LidarLoopClosureDetector(configuration)
        cloud = _structured_cloud(count=900)
        first = detector.add_keyframe(
            node_id=0,
            keyframe_id=10,
            timestamp=0.0,
            local_camera_pose=_pose(),
            points_sensor=cloud,
            transform_camera_sensor=np.eye(4),
        )
        second = detector.add_keyframe(
            node_id=1,
            keyframe_id=11,
            timestamp=1.0,
            # Local odometry drifted even though the scan revisits node 0.
            local_camera_pose=_pose(0.2),
            points_sensor=cloud,
            transform_camera_sensor=np.eye(4),
        )
        self.assertEqual(first.status, "no_geometrically_verified_candidate")
        self.assertEqual(second.status, "accepted")
        self.assertEqual(len(second.accepted_constraints), 1)
        np.testing.assert_allclose(
            second.accepted_constraints[0].relative_pose_reference_query,
            np.eye(4),
            atol=2.0e-3,
        )

    def test_external_candidate_verification_loads_disk_backed_submaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            configuration = LidarLoopConfiguration(
                enabled=False,
                archive_directory=Path(temporary),
                geometry_cache_records=1,
                submap_keyframes=1,
                voxel_size_m=0.05,
                max_points=1200,
                min_keyframe_separation=1,
                min_time_separation_sec=0.0,
                icp_max_correspondence_m=0.5,
                icp_min_inliers=100,
                icp_min_overlap=0.5,
                icp_min_unique_target_ratio=0.8,
                icp_max_rmse_m=0.05,
                max_correction_translation_m=0.5,
                confirmation_count=1,
            )
            detector = LidarLoopClosureDetector(configuration)
            cloud = _structured_cloud(count=900)
            for node_id in range(3):
                detector.add_keyframe(
                    node_id=node_id,
                    keyframe_id=node_id,
                    timestamp=float(node_id),
                    local_camera_pose=_pose(0.1 * node_id),
                    points_sensor=cloud,
                    transform_camera_sensor=np.eye(4),
                )
            self.assertEqual(len(list(Path(temporary).glob("loop_*.npz"))), 3)
            update = detector.submit_candidates(
                (
                    LoopCandidate(
                        reference_node=1,
                        query_node=2,
                        source="mprf_bad_alternative",
                        score=1.0,
                        coarse_relative_pose=_pose(10.0),
                    ),
                    LoopCandidate(
                        reference_node=0,
                        query_node=2,
                        source="mprf_test",
                        score=0.9,
                        coarse_relative_pose=np.eye(4),
                    ),
                )
            )
            self.assertEqual(update.status, "accepted")
            self.assertEqual(update.candidates_considered, 2)
            self.assertEqual(update.geometrically_verified, 1)
            self.assertEqual(len(update.accepted_constraints), 1)


class GlobalPoseGraphTest(unittest.TestCase):
    def test_verified_loop_corrects_global_graph_without_feedback(self):
        graph = GtsamGlobalPoseGraph(
            odometry_rotation_sigma_deg=(1.0, 1.0, 1.0),
            odometry_translation_sigma_m=(0.10, 0.10, 0.10),
        )
        local = {
            0: _pose(0.0),
            1: _pose(1.0),
            2: _pose(2.0),
            # The robot returned to the origin with 0.4 m local drift.
            3: _pose(0.4),
        }
        for node_id, pose in local.items():
            graph.add_keyframe(
                node_id=node_id,
                timestamp=float(node_id),
                local_pose=pose,
            )
        update = graph.add_loop_constraints(
            [
                LoopConstraint(
                    0,
                    3,
                    np.eye(4),
                    np.r_[np.full(3, 0.01), np.full(3, 0.02)],
                    "test_lidar_icp",
                    0.95,
                )
            ]
        )
        self.assertEqual(update.loop_constraints_added, 1)
        self.assertTrue(update.changed_nodes)
        self.assertLess(abs(update.node_poses[3][0, 3]), 0.05)
        # Raw local input remains unchanged and is corrected only on output.
        self.assertAlmostEqual(local[3][0, 3], 0.4)
        np.testing.assert_allclose(
            graph.correct_pose(node_id=3, local_pose=local[3]),
            update.node_poses[3],
        )

    def test_duplicate_loop_factor_is_idempotent(self):
        graph = GtsamGlobalPoseGraph()
        graph.add_keyframe(node_id=0, timestamp=0.0, local_pose=_pose())
        graph.add_keyframe(node_id=1, timestamp=1.0, local_pose=_pose(0.1))
        constraint = LoopConstraint(
            0,
            1,
            _pose(0.1),
            np.full(6, 0.1),
            "test",
            0.9,
        )
        self.assertEqual(
            graph.add_loop_constraints([constraint]).loop_constraints_added,
            1,
        )
        self.assertEqual(
            graph.add_loop_constraints([constraint]).loop_constraints_added,
            0,
        )
        self.assertEqual(graph.loop_count, 1)


if __name__ == "__main__":
    unittest.main()
