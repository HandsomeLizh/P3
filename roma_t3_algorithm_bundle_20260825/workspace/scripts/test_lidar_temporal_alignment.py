#!/usr/bin/env python3
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.lidar_mapping import (
    LidarMappingProcessor,
    pointcloud2_xyz_time_arrays,
)
from t3_semantic_mapping.lidar_temporal_alignment import (
    LidarTemporalConfiguration,
    align_lidar_points_to_reference,
    lidar_required_pose_time_range,
)
from t3_semantic_mapping.localization import ExternalOdometryAdapter


def _pose(x=0.0, y=0.0, yaw=0.0):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    value[:3, 3] = [x, y, 0.0]
    return value


def _field(name, offset, datatype):
    value = PointField()
    value.name = name
    value.offset = int(offset)
    value.datatype = int(datatype)
    value.count = 1
    return value


def _timed_cloud(*, time_datatype=PointField.FLOAT32, payload=None):
    message = PointCloud2()
    message.header.stamp.sec = 10
    message.header.frame_id = "velodyne"
    message.height = 1
    message.width = 2
    message.is_bigendian = False
    if time_datatype == PointField.FLOAT32:
        point_step = 16
        default_payload = b"".join(
            struct.pack("<ffff", x, 0.0, 1.0, offset)
            for x, offset in ((1.0, -0.1), (2.0, 0.0))
        )
    else:
        point_step = 16
        default_payload = b"".join(
            struct.pack("<fffI", x, 0.0, 1.0, offset)
            for x, offset in ((1.0, 100_000_000), (2.0, 0))
        )
    message.fields = [
        _field("x", 0, PointField.FLOAT32),
        _field("y", 4, PointField.FLOAT32),
        _field("z", 8, PointField.FLOAT32),
        _field("time", 12, time_datatype),
    ]
    message.point_step = point_step
    message.row_step = point_step * message.width
    message.data = default_payload if payload is None else payload
    return message


class PointTimeDecoderTest(unittest.TestCase):
    def test_float_seconds_preserve_signed_header_relative_offsets(self):
        points, offsets = pointcloud2_xyz_time_arrays(_timed_cloud())
        np.testing.assert_allclose(points[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(offsets, [-0.1, 0.0], atol=1e-8)

    def test_integer_nanoseconds_use_explicit_unit(self):
        points, offsets = pointcloud2_xyz_time_arrays(
            _timed_cloud(time_datatype=PointField.UINT32),
            time_unit="nsec",
        )
        np.testing.assert_allclose(points[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(offsets, [0.1, 0.0])

    def test_missing_or_ambiguous_time_contract_is_rejected(self):
        message = _timed_cloud()
        message.fields[-1].name = "curvature"
        with self.assertRaisesRegex(ValueError, "missing configured"):
            pointcloud2_xyz_time_arrays(message, time_field="time")
        with self.assertRaisesRegex(ValueError, "unit"):
            pointcloud2_xyz_time_arrays(
                _timed_cloud(), time_unit="ticks"
            )


class TemporalAlignmentTest(unittest.TestCase):
    def test_none_is_bit_exact_and_never_queries_pose(self):
        points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]])

        def forbidden(_):
            raise AssertionError("compatibility mode queried a pose")

        result = align_lidar_points_to_reference(
            points,
            cloud_timestamp=1.0,
            reference_timestamp=1.1,
            transform_camera_sensor=np.eye(4),
            configuration=LidarTemporalConfiguration(mode="none"),
            pose_query=forbidden,
        )
        np.testing.assert_array_equal(result.points_sensor, points)
        np.testing.assert_array_equal(result.sensor_origin_sensor, [0, 0, 0])
        self.assertEqual(
            result.statistics["temporal_compensation_applied"], 0.0
        )

    def test_upstream_deskewed_only_aligns_rigid_cloud_to_image_time(self):
        # A static world point is x=10. At cloud time 1.0 the already
        # deskewed sensor coordinate is 9.0; at image time 1.2 it is 8.8.
        result = align_lidar_points_to_reference(
            np.array([[9.0, 0.0, 0.0]]),
            cloud_timestamp=1.0,
            reference_timestamp=1.2,
            transform_camera_sensor=np.eye(4),
            configuration=LidarTemporalConfiguration(
                mode="upstream_deskewed"
            ),
            pose_query=lambda stamp: _pose(x=stamp),
            # A time array is deliberately supplied: upstream mode must
            # ignore it instead of deskewing the cloud twice.
            point_time_offsets_sec=np.array([-0.1]),
        )
        np.testing.assert_allclose(result.points_sensor, [[8.8, 0.0, 0.0]])
        np.testing.assert_allclose(
            result.sensor_origin_sensor, [-0.2, 0.0, 0.0]
        )
        self.assertEqual(result.statistics["point_deskew_applied"], 0.0)

    def test_per_point_compensation_collapses_constant_velocity_scan(self):
        offsets = np.array([-1.0, -0.5, 0.0])
        # Measurements of one static x=10 world point by a sensor moving at
        # 1 m/s. All points should become x=9 in the t=1 reference frame.
        measurements = np.column_stack(
            [10.0 - (1.0 + offsets), np.zeros(3), np.zeros(3)]
        )
        result = align_lidar_points_to_reference(
            measurements,
            cloud_timestamp=1.0,
            reference_timestamp=1.0,
            transform_camera_sensor=np.eye(4),
            configuration=LidarTemporalConfiguration(
                mode="per_point",
                max_point_time_offset_sec=1.1,
                pose_time_bin_sec=0.0,
            ),
            pose_query=lambda stamp: _pose(x=stamp),
            point_time_offsets_sec=offsets,
        )
        np.testing.assert_allclose(result.points_sensor[:, 0], [9.0, 9.0, 9.0])
        self.assertEqual(len(result.points_sensor), len(measurements))
        self.assertEqual(result.statistics["point_deskew_applied"], 1.0)

    def test_nonidentity_extrinsic_matches_full_se3_formula(self):
        camera_sensor = _pose(x=0.3, y=-0.2, yaw=0.4)
        point = np.array([[2.0, -1.0, 0.5]])
        world_camera_cloud = _pose(x=1.0, y=0.2, yaw=0.1)
        world_camera_reference = _pose(x=1.4, y=-0.1, yaw=0.3)

        def query(stamp):
            return world_camera_cloud if stamp == 2.0 else world_camera_reference

        result = align_lidar_points_to_reference(
            point,
            cloud_timestamp=2.0,
            reference_timestamp=3.0,
            transform_camera_sensor=camera_sensor,
            configuration=LidarTemporalConfiguration(
                mode="upstream_deskewed"
            ),
            pose_query=query,
        )
        expected_transform = (
            np.linalg.inv(camera_sensor)
            @ np.linalg.inv(world_camera_reference)
            @ world_camera_cloud
            @ camera_sensor
        )
        expected = (
            point @ expected_transform[:3, :3].T
            + expected_transform[:3, 3]
        )
        np.testing.assert_allclose(result.points_sensor, expected, atol=1e-12)

    def test_missing_pose_and_bad_point_time_fail_closed(self):
        configuration = LidarTemporalConfiguration(mode="per_point")
        with self.assertRaisesRegex(ValueError, "safety limit"):
            align_lidar_points_to_reference(
                [[1.0, 0.0, 0.0]],
                cloud_timestamp=1.0,
                reference_timestamp=1.0,
                transform_camera_sensor=np.eye(4),
                configuration=configuration,
                pose_query=lambda _: np.eye(4),
                point_time_offsets_sec=[2.0],
            )
        with self.assertRaisesRegex(RuntimeError, "does not bracket"):
            align_lidar_points_to_reference(
                [[1.0, 0.0, 0.0]],
                cloud_timestamp=1.0,
                reference_timestamp=1.1,
                transform_camera_sensor=np.eye(4),
                configuration=LidarTemporalConfiguration(
                    mode="upstream_deskewed"
                ),
                pose_query=lambda _: None,
            )

    def test_binning_bounds_query_error_without_dropping_points(self):
        offsets = np.linspace(-0.0994, 0.0004, 1000)
        configuration = LidarTemporalConfiguration(
            mode="per_point", pose_time_bin_sec=0.001
        )
        required = lidar_required_pose_time_range(
            cloud_timestamp=5.0,
            reference_timestamp=5.02,
            point_count=len(offsets),
            configuration=configuration,
            point_time_offsets_sec=offsets,
        )
        result = align_lidar_points_to_reference(
            np.ones((len(offsets), 3)),
            cloud_timestamp=5.0,
            reference_timestamp=5.02,
            transform_camera_sensor=np.eye(4),
            configuration=configuration,
            pose_query=lambda _: np.eye(4),
            point_time_offsets_sec=offsets,
        )
        self.assertEqual(len(result.points_sensor), len(offsets))
        self.assertLessEqual(
            result.statistics["pose_time_bin_max_error_sec"], 0.0005 + 1e-12
        )
        self.assertLess(result.statistics["pose_query_count"], 110)
        self.assertLessEqual(required[0], 4.901)
        self.assertGreaterEqual(required[1], 5.02)


class ProcessorMovingOriginTest(unittest.TestCase):
    def test_range_gate_and_ray_origin_follow_aligned_sensor_origin(self):
        processor = LidarMappingProcessor(
            transform_camera_lidar=np.eye(4),
            rectification_rotation=np.eye(3),
            camera_matrix=np.eye(3),
            calibration_width=10,
            calibration_height=10,
            class_count=5,
            min_range_m=0.1,
            max_range_m=10.0,
            voxel_size_m=0.0,
            max_points=0,
        )
        result = processor.make_observation(
            [[5.05, 0.0, 1.0], [6.0, 0.0, 1.0]],
            np.zeros((10, 10), dtype=np.uint8),
            sensor_origin_lidar=[5.0, 0.0, 1.0],
        )
        self.assertEqual(result.statistics["selected_points"], 1)
        np.testing.assert_allclose(result.sensor_origin_camera, [5, 0, 1])
        np.testing.assert_allclose(result.camera_points, [[6, 0, 1]])


class StrictExternalPoseQueryTest(unittest.TestCase):
    def test_adapter_can_disable_nearest_pose_fallback(self):
        adapter = ExternalOdometryAdapter(transform_camera_child=np.eye(4))
        adapter.add_components(
            timestamp=1.0,
            parent_frame="camera_init",
            child_frame="body",
            position_xyz=[0, 0, 0],
            quaternion_xyzw=[0, 0, 0, 1],
        )
        self.assertIsNotNone(
            adapter.query(
                1.05,
                max_time_offset_sec=0.1,
                align_to_first_query=False,
            )
        )
        self.assertIsNone(
            adapter.query(
                1.05,
                max_time_offset_sec=0.1,
                align_to_first_query=False,
                allow_nearest=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
