#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.localization import (
    ExternalOdometryAdapter,
    FrameMismatchError,
    LocalizationConfiguration,
    TimedPose,
    TimedPoseBuffer,
)
from t3_semantic_mapping.localization.calibration import (
    compose_lio_child_to_camera_transform,
    row_major_transform,
)


def _pose(x=0.0, y=0.0, yaw=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:3, 3] = [x, y, 0.0]
    return pose


class LocalizationConfigurationTest(unittest.TestCase):
    def test_supported_mode_matrix(self):
        for mode, backend in (
            ("vo", "none"),
            ("lio", "none"),
            ("vo_lio", "gtsam_gp"),
        ):
            configuration = LocalizationConfiguration(
                mode=mode, fusion_backend=backend
            )
            self.assertEqual(configuration.mode, mode)
            self.assertEqual(configuration.fusion_backend, backend)

    def test_rejects_ambiguous_or_unowned_backend(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            LocalizationConfiguration(mode="vo_lio", fusion_backend="none")
        with self.assertRaisesRegex(ValueError, "only valid"):
            LocalizationConfiguration(mode="vo", fusion_backend="gtsam_gp")

    def test_default_import_does_not_load_gtsam(self):
        # GTSAM stays an optional runtime dependency for vo/lio direct modes.
        environment = dict(os.environ)
        source = str(WORKSPACE / "src" / "t3_semantic_mapping")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import t3_semantic_mapping.localization; "
                    "raise SystemExit(int('gtsam' in sys.modules))"
                ),
            ],
            env=environment,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


class TimedPoseBufferTest(unittest.TestCase):
    def test_interpolates_translation_and_rotation(self):
        buffer = TimedPoseBuffer(max_size=10, max_age_sec=10.0)
        buffer.add(TimedPose(1.0, _pose(0.0, yaw=0.0)))
        buffer.add(TimedPose(2.0, _pose(2.0, yaw=np.pi / 2.0)))

        query = buffer.query(1.5, max_time_offset_sec=0.6)

        self.assertEqual(query.method, "interpolated")
        np.testing.assert_allclose(query.sample.pose[:3, 3], [1.0, 0.0, 0.0])
        yaw = Rotation.from_matrix(
            query.sample.pose[:3, :3].copy()
        ).as_euler("zyx")[0]
        self.assertAlmostEqual(yaw, np.pi / 4.0, places=7)

    def test_does_not_interpolate_across_lio_outage(self):
        buffer = TimedPoseBuffer(max_size=10, max_age_sec=10.0)
        buffer.add(TimedPose(1.0, _pose(0.0)))
        buffer.add(TimedPose(5.0, _pose(4.0)))

        self.assertIsNone(
            buffer.query(3.0, max_time_offset_sec=0.2)
        )

    def test_duplicate_timestamp_replaces_sample(self):
        buffer = TimedPoseBuffer(max_size=10, max_age_sec=10.0)
        buffer.add(TimedPose(1.0, _pose(0.0)))
        buffer.add(TimedPose(1.0, _pose(3.0)))

        self.assertEqual(len(buffer), 1)
        query = buffer.query(1.0, max_time_offset_sec=0.0)
        self.assertAlmostEqual(query.sample.pose[0, 3], 3.0)


class ExternalOdometryAdapterTest(unittest.TestCase):
    def test_composes_fastlio_body_and_camera_extrinsics(self):
        camera_lidar = np.eye(4, dtype=np.float64)
        camera_lidar[0, 3] = 0.2
        # FAST-LIO has T_body_lidar=[+1 m]; its body-to-LiDAR transform is
        # therefore the inverse, T_lidar_body=[-1 m].
        lidar_body = np.eye(4, dtype=np.float64)
        lidar_body[1, 3] = -1.0
        parsed = row_major_transform(
            lidar_body.reshape(-1).tolist(),
            name="test child transform",
        )

        camera_body = compose_lio_child_to_camera_transform(
            camera_lidar, parsed
        )

        np.testing.assert_allclose(
            camera_body[:3, 3], [0.2, -1.0, 0.0]
        )

    def test_rejects_malformed_child_transform(self):
        with self.assertRaisesRegex(ValueError, "16 row-major"):
            row_major_transform([1.0, 2.0], name="bad transform")

    def test_applies_child_to_camera_extrinsic_and_first_pose_alignment(self):
        transform_camera_child = np.eye(4, dtype=np.float64)
        transform_camera_child[0, 3] = 0.2
        adapter = ExternalOdometryAdapter(
            transform_camera_child=transform_camera_child,
            expected_parent_frame="camera_init",
            expected_child_frame="body",
        )
        for timestamp, x in ((1.0, 10.0), (2.0, 12.0)):
            adapter.add_components(
                timestamp=timestamp,
                parent_frame="/camera_init",
                child_frame="/body",
                position_xyz=[x, 0.0, 0.0],
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            )

        first = adapter.query(1.0, max_time_offset_sec=0.0)
        second = adapter.query(2.0, max_time_offset_sec=0.0)

        np.testing.assert_allclose(first.pose, np.eye(4), atol=1e-12)
        # The fixed lever arm cancels after alignment; relative travel is 2 m.
        np.testing.assert_allclose(second.pose[:3, 3], [2.0, 0.0, 0.0])

    def test_rejects_wrong_frame_before_buffering(self):
        adapter = ExternalOdometryAdapter(
            transform_camera_child=np.eye(4),
            expected_parent_frame="camera_init",
            expected_child_frame="body",
        )
        with self.assertRaises(FrameMismatchError):
            adapter.add_components(
                timestamp=1.0,
                parent_frame="map",
                child_frame="body",
                position_xyz=[0.0, 0.0, 0.0],
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        self.assertEqual(len(adapter.buffer), 0)


class GtsamGpFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            __import__("gtsam")
            __import__("gtsam_unstable")
        except ImportError as exception:
            raise unittest.SkipTest(str(exception))

    def test_fixed_lag_gp_stays_finite_through_vo_dropout(self):
        from t3_semantic_mapping.localization.gtsam_gp import GtsamGpWnoaFusion

        fusion = GtsamGpWnoaFusion(lag_sec=1.0, enable_gp_prior=True)
        update = None
        for index in range(30):
            truth = _pose(0.1 * index, 0.0, yaw=0.01 * index)
            lio = _pose(0.1 * index, 0.001 * index, yaw=0.01 * index)
            vo = None if 5 <= index < 20 else truth
            update = fusion.add_measurement(
                timestamp=0.1 * index,
                vo_pose=vo,
                lio_pose=lio,
            )

        self.assertIsNotNone(update)
        self.assertTrue(np.isfinite(update.estimate.pose).all())
        self.assertLessEqual(len(update.active_poses), 12)
        self.assertAlmostEqual(update.estimate.pose[0, 3], 2.9, delta=0.05)

    def test_gp_can_be_disabled_for_ablation(self):
        from t3_semantic_mapping.localization.gtsam_gp import GtsamGpWnoaFusion

        fusion = GtsamGpWnoaFusion(lag_sec=1.0, enable_gp_prior=False)
        first = fusion.add_measurement(
            timestamp=0.0,
            vo_pose=_pose(),
            lio_pose=_pose(),
        )
        second = fusion.add_measurement(
            timestamp=0.1,
            vo_pose=_pose(0.1),
            lio_pose=_pose(0.1),
        )

        self.assertIsNone(first.estimate.velocity_body)
        self.assertIsNone(second.estimate.velocity_body)
        self.assertFalse(second.estimate.diagnostics["gp_prior_enabled"])

    def test_late_lio_origin_aligns_to_current_fused_state(self):
        from t3_semantic_mapping.localization.gtsam_gp import GtsamGpWnoaFusion

        fusion = GtsamGpWnoaFusion(lag_sec=1.0, enable_gp_prior=True)
        fusion.add_measurement(
            timestamp=0.0,
            vo_pose=_pose(5.0),
            lio_pose=None,
        )
        update = fusion.add_measurement(
            timestamp=0.1,
            vo_pose=_pose(5.1),
            # A separately initialized LIO origin must not pull x back to 0.
            lio_pose=_pose(100.0),
        )

        self.assertAlmostEqual(update.estimate.pose[0, 3], 0.1, delta=0.03)

    def test_active_state_reporting_uses_timestamps_for_irregular_rates(self):
        from t3_semantic_mapping.localization.gtsam_gp import GtsamGpWnoaFusion

        fusion = GtsamGpWnoaFusion(lag_sec=1.0, enable_gp_prior=False)
        update = None
        times = [0.01 * index for index in range(50)] + [0.99]
        for timestamp in times:
            update = fusion.add_measurement(
                timestamp=timestamp,
                vo_pose=_pose(timestamp),
                lio_pose=_pose(timestamp),
            )

        self.assertIsNotNone(update)
        self.assertGreaterEqual(len(update.active_poses), 45)


if __name__ == "__main__":
    unittest.main()
