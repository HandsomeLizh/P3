import tempfile
import unittest
from pathlib import Path

import numpy as np

from mprf_runtime.config import MprfRuntimeConfiguration
from mprf_runtime.contracts import KeyframePacket, PoseFeatureSet
from mprf_runtime.engine import MprfOnlineEngine
from mprf_runtime.geometry import (
    estimate_pose_ransac,
    match_local_features,
    rigid_alignment,
    transform_points,
)


def pose(x=0.0, y=0.0, yaw=0.0):
    result = np.eye(4)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    result[:2, :2] = [[cosine, -sine], [sine, cosine]]
    result[:3, 3] = [x, y, 0.0]
    return result


class GeometryTest(unittest.TestCase):
    def test_rigid_alignment_and_ransac_recover_outlier_contaminated_pose(self):
        generator = np.random.default_rng(7)
        query = generator.uniform([-3.0, -2.0, -0.5], [3.0, 2.0, 1.0], (120, 3))
        truth = pose(0.45, -0.22, 0.08)
        reference = transform_points(truth, query)
        descriptors = generator.normal(size=(120, 64)).astype(np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        query_descriptors = descriptors + generator.normal(0.0, 0.002, descriptors.shape)
        # Deliberately corrupt a fifth of the coordinates while retaining
        # descriptor matches; RANSAC must reject these pairs.
        reference[:24] += generator.uniform(2.0, 4.0, (24, 3))
        matches = match_local_features(
            reference_descriptors=descriptors,
            query_descriptors=query_descriptors,
            mode="mutual",
            minimum_similarity=0.8,
            minimum_margin=0.05,
            maximum_matches=200,
        )
        result = estimate_pose_ransac(
            reference_points=reference,
            query_points=query,
            matches=matches,
            maximum_iterations=1000,
            residual_threshold_m=0.05,
            minimum_inliers=50,
            minimum_inlier_ratio=0.5,
            maximum_rmse_m=0.02,
            minimum_spread_m=0.2,
            maximum_translation_m=2.0,
            maximum_rotation_deg=30.0,
            random_seed=9,
        )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(
            result.transform_reference_query, truth, atol=1.0e-5
        )
        self.assertEqual(result.inlier_count, 96)

    def test_alignment_rejects_too_few_pairs(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            rigid_alignment(np.zeros((2, 3)), np.zeros((2, 3)))


class FakeFeatureBackend:
    def __init__(self, descriptors):
        self.descriptors = descriptors

    def extract_retrieval(self, image_bgr):
        marker = int(image_bgr[0, 0, 0])
        if marker in {0, 2}:
            return np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0, 0.0])
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0, 0.0])

    def extract_pose_features(self, packet):
        return PoseFeatureSet(packet.points_sensor, self.descriptors)

    def metadata(self):
        return {"backend": "fake", "device": "cpu"}


def engine_packet(node, points):
    image = np.full((32, 48, 3), node, dtype=np.uint8)
    return KeyframePacket(
        session_id="engine-test",
        node_id=node,
        timestamp_ns=(node + 1) * 1_000_000_000,
        image_bgr=image,
        points_sensor=points.astype(np.float32),
        camera_matrix=np.asarray(
            [[40.0, 0.0, 24.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]
        ),
        transform_camera_sensor=np.eye(4),
        calibration_width=48,
        calibration_height=32,
    )


class EngineTest(unittest.TestCase):
    def test_two_stage_candidate_has_explicit_reference_query_pose(self):
        generator = np.random.default_rng(22)
        reference = generator.uniform([-3, -2, 1], [3, 2, 4], (100, 3))
        truth = pose(0.35, -0.1, 0.05)
        query = transform_points(np.linalg.inv(truth), reference)
        descriptors = generator.normal(size=(100, 48)).astype(np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            configuration = MprfRuntimeConfiguration(
                state_root=Path(temporary),
                min_keyframe_separation=2,
                min_time_separation_sec=0.0,
                retrieval_shortlist=2,
                refined_shortlist=2,
                min_salad_similarity=0.5,
                min_dino_similarity=0.5,
                match_min_similarity=0.9,
                match_min_margin=0.05,
                ransac_residual_m=0.03,
                ransac_min_inliers=50,
                ransac_min_inlier_ratio=0.5,
                ransac_max_rmse_m=0.02,
                ransac_min_spread_m=0.2,
            )
            engine = MprfOnlineEngine(
                configuration,
                feature_backend=FakeFeatureBackend(descriptors),
            )
            self.assertEqual(engine.process(engine_packet(0, reference))["status"], "stored")
            # Middle frame is intentionally a different place.
            self.assertEqual(engine.process(engine_packet(1, reference + 8.0))["status"], "stored")
            response = engine.process(engine_packet(2, query))
            self.assertEqual(response["status"], "candidate")
            candidate = response["candidate"]
            self.assertEqual(candidate["reference_node"], 0)
            self.assertEqual(candidate["query_node"], 2)
            np.testing.assert_allclose(
                np.asarray(candidate["relative_pose_reference_query"]).reshape(4, 4),
                truth,
                atol=1.0e-5,
            )
            self.assertEqual(response["database_records"], 3)
            self.assertGreater(candidate["ransac_inliers"], 90)
            # A client timeout followed by the same POST must replay the
            # original candidate rather than silently downgrade to duplicate.
            replay = engine.process(engine_packet(2, query))
            self.assertEqual(replay, response)

    def test_retrieval_shortlist_survives_local_pose_rejection(self):
        generator = np.random.default_rng(31)
        points = generator.uniform([-3, -2, 1], [3, 2, 4], (100, 3))
        # Indistinguishable local descriptors cannot support a metric pose,
        # but the two-stage place descriptor still supplies a shortlist for
        # the independent LiDAR verifier.
        descriptors = np.ones((100, 16), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            configuration = MprfRuntimeConfiguration(
                state_root=Path(temporary),
                min_keyframe_separation=2,
                min_time_separation_sec=0.0,
                retrieval_shortlist=1,
                refined_shortlist=1,
                min_salad_similarity=0.5,
                min_dino_similarity=0.5,
            )
            engine = MprfOnlineEngine(
                configuration,
                feature_backend=FakeFeatureBackend(descriptors),
            )
            engine.process(engine_packet(0, points))
            response = engine.process(engine_packet(2, points))
            self.assertEqual(response["status"], "candidate")
            self.assertEqual(len(response["candidates"]), 1)
            candidate = response["candidates"][0]
            self.assertEqual(candidate["candidate_mode"], "retrieval_only")
            self.assertNotIn("relative_pose_reference_query", candidate)
            self.assertEqual(response["pose_solved_records"], 0)


if __name__ == "__main__":
    unittest.main()
