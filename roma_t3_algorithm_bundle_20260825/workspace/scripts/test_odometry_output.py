#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.odometry_output import OdometryOutputModel


def _pose(x=0.0, y=0.0, yaw=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:3, 3] = [x, y, 0.0]
    return pose


def _model(**overrides):
    options = {
        "camera_frame": "camera_optical",
        "child_frame": "camera_optical",
        "transform_camera_child": np.eye(4),
        "child_transform_confirmed": False,
        "velocity_max_interval_sec": 1.0,
        "unknown_covariance_variance": 1.0e6,
    }
    options.update(overrides)
    return OdometryOutputModel(**options)


class OdometryFrameContractTest(unittest.TestCase):
    def test_default_camera_pose_needs_no_extra_calibration(self):
        model = _model()
        pose = _pose(1.0, 2.0, 0.3)
        np.testing.assert_allclose(model.transform_pose(pose), pose)

    def test_different_child_frame_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(ValueError, "changing only the frame name"):
            _model(child_frame="base_link")

    def test_same_named_frame_rejects_nonidentity_transform(self):
        transform = np.eye(4)
        transform[0, 3] = 0.2
        with self.assertRaisesRegex(ValueError, "must be identity"):
            _model(transform_camera_child=transform)

    def test_confirmed_child_pose_uses_camera_to_child_extrinsic(self):
        transform = np.eye(4)
        transform[0, 3] = 0.2
        model = _model(
            child_frame="base_link",
            transform_camera_child=transform,
            child_transform_confirmed=True,
        )
        output = model.transform_pose(_pose(1.0, yaw=np.pi / 2.0))
        np.testing.assert_allclose(output[:3, 3], [1.0, 0.2, 0.0], atol=1e-12)


class OdometryVelocityTest(unittest.TestCase):
    def test_first_sample_is_explicitly_unavailable(self):
        sample = _model().update(timestamp=1.0, pose_map_camera=_pose())
        self.assertEqual(sample.velocity_method, "unavailable_first_sample")
        np.testing.assert_array_equal(sample.twist_angular_linear, np.zeros(6))
        np.testing.assert_array_equal(
            np.diag(sample.twist_covariance_ros), np.full(6, 1.0e6)
        )

    def test_backward_difference_reports_twist_in_current_child_frame(self):
        model = _model()
        model.update(timestamp=1.0, pose_map_camera=_pose())
        sample = model.update(
            timestamp=1.5,
            pose_map_camera=_pose(x=1.0, yaw=np.pi / 2.0),
        )
        self.assertEqual(sample.velocity_method, "backward_difference")
        np.testing.assert_allclose(
            sample.twist_angular_linear[:3], [0.0, 0.0, np.pi], atol=1e-12
        )
        # +X map displacement is -Y in the current 90-degree-yawed child.
        np.testing.assert_allclose(
            sample.twist_angular_linear[3:], [0.0, -2.0, 0.0], atol=1e-12
        )

    def test_confirmed_lever_arm_transforms_source_twist(self):
        transform = np.eye(4)
        # Child origin lies +1 m along camera X.
        transform[0, 3] = 1.0
        model = _model(
            child_frame="base_link",
            transform_camera_child=transform,
            child_transform_confirmed=True,
        )
        sample = model.update(
            timestamp=1.0,
            pose_map_camera=_pose(),
            velocity_camera=np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        )
        self.assertEqual(sample.velocity_method, "source")
        np.testing.assert_allclose(
            sample.twist_angular_linear,
            [0.0, 0.0, 2.0, 0.0, 2.0, 0.0],
            atol=1e-12,
        )

    def test_time_gap_and_geometry_correction_reset_difference(self):
        model = _model(velocity_max_interval_sec=0.2)
        model.update(timestamp=1.0, pose_map_camera=_pose())
        gap = model.update(timestamp=1.3, pose_map_camera=_pose(1.0))
        self.assertEqual(gap.velocity_method, "unavailable_time_gap")
        reset = model.update(
            timestamp=1.4,
            pose_map_camera=_pose(4.0),
            reset_velocity_history=True,
        )
        self.assertEqual(reset.velocity_method, "unavailable_first_sample")


class OdometryCovarianceTest(unittest.TestCase):
    def test_missing_pose_covariance_is_not_false_zero(self):
        sample = _model().update(timestamp=1.0, pose_map_camera=_pose())
        np.testing.assert_array_equal(
            np.diag(sample.pose_covariance_ros), np.full(6, 1.0e6)
        )

    def test_rotation_first_covariance_is_reordered_for_ros(self):
        covariance = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sample = _model().update(
            timestamp=1.0,
            pose_map_camera=_pose(),
            pose_covariance_camera=covariance,
        )
        np.testing.assert_allclose(
            np.diag(sample.pose_covariance_ros), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        )

    def test_lever_arm_propagates_pose_covariance(self):
        transform = np.eye(4)
        transform[0, 3] = 1.0
        model = _model(
            child_frame="base_link",
            transform_camera_child=transform,
            child_transform_confirmed=True,
        )
        covariance = np.zeros((6, 6), dtype=np.float64)
        covariance[2, 2] = 0.25
        sample = model.update(
            timestamp=1.0,
            pose_map_camera=_pose(),
            pose_covariance_camera=covariance,
        )
        # Z rotation uncertainty at a one-metre lever arm creates Y position
        # uncertainty for the output child origin.
        self.assertAlmostEqual(sample.pose_covariance_ros[1, 1], 0.25)


if __name__ == "__main__":
    unittest.main()
