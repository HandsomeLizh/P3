#!/usr/bin/env python3
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(WORKSPACE / "src" / "t3_semantic_mapping")
)
sys.path.insert(0, str(WORKSPACE / "src" / "t1_dataset_sim"))

from t1_dataset_sim.lidar_io import read_binary_pcd_message
from t3_semantic_mapping.lidar_mapping import (
    LidarMappingProcessor,
    assist_stereo_depth_from_pointcloud,
    pointcloud2_xyz_array,
    stamp_nanoseconds,
)
from t3_semantic_mapping.lidar_occupancy import (
    ENDPOINT_FREE,
    ENDPOINT_OCCUPIED,
    ENDPOINT_UNKNOWN,
    LidarGroundSegmentationConfig,
    classify_lidar_endpoint_occupancy,
)
from t3_semantic_mapping.semantic_grid import LayeredSemanticGridMap
from t3_semantic_mapping.semantic_map_rebuilder import (
    KeyFrameObservationStore,
    SemanticKeyFrameObservation,
)


def _field(name: str, offset: int, datatype: int) -> PointField:
    field = PointField()
    field.name = name
    field.offset = int(offset)
    field.datatype = int(datatype)
    field.count = 1
    return field


def _cloud_with_row_padding() -> PointCloud2:
    message = PointCloud2()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 345
    message.header.frame_id = "velodyne"
    message.height = 2
    message.width = 2
    message.fields = [
        _field("x", 0, PointField.FLOAT32),
        _field("y", 4, PointField.FLOAT32),
        _field("z", 8, PointField.FLOAT32),
        _field("intensity", 12, PointField.FLOAT32),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = 36
    rows = []
    for points in (
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        ((7.0, 8.0, 9.0), (10.0, 11.0, 12.0)),
    ):
        rows.append(
            b"".join(struct.pack("<ffff", *point, 0.5) for point in points)
            + b"PAD!"
        )
    message.data = b"".join(rows)
    message.is_dense = True
    return message


def _processor(*, voxel_size_m: float = 0.0, max_points: int = 0):
    return LidarMappingProcessor(
        transform_camera_lidar=np.eye(4, dtype=np.float64),
        rectification_rotation=np.eye(3, dtype=np.float64),
        camera_matrix=np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        calibration_width=100,
        calibration_height=80,
        class_count=5,
        min_range_m=0.1,
        max_range_m=10.0,
        voxel_size_m=voxel_size_m,
        max_points=max_points,
    )


class PointCloudInputTest(unittest.TestCase):
    def test_xyz_decoder_honours_offsets_and_row_padding(self):
        message = _cloud_with_row_padding()
        points = pointcloud2_xyz_array(message)

        np.testing.assert_allclose(
            points,
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ),
        )
        self.assertEqual(stamp_nanoseconds(message), 12_000_000_345)

    def test_binary_pcd_is_forwarded_without_changing_point_layout(self):
        header = (
            "# .PCD v0.7\n"
            "VERSION 0.7\n"
            "FIELDS x y z intensity ring time\n"
            "SIZE 4 4 4 4 2 4\n"
            "TYPE F F F F U F\n"
            "COUNT 1 1 1 1 1 1\n"
            "WIDTH 2\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS 2\n"
            "DATA binary\n"
        ).encode("ascii")
        payload = b"".join(
            (
                struct.pack("<ffffHf", 1.0, 2.0, 3.0, 7.0, 4, 0.01),
                struct.pack("<ffffHf", 4.0, 5.0, 6.0, 8.0, 5, 0.02),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.pcd"
            path.write_bytes(header + payload)
            stamp = Time(sec=8, nanosec=9)
            message = read_binary_pcd_message(
                path, stamp=stamp, frame_id="velodyne"
            )

        self.assertEqual(message.point_step, 22)
        self.assertEqual(message.row_step, 44)
        self.assertEqual(message.header.frame_id, "velodyne")
        self.assertEqual([field.name for field in message.fields], [
            "x", "y", "z", "intensity", "ring", "time"
        ])
        np.testing.assert_allclose(
            pointcloud2_xyz_array(message),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

    def test_binary_pcd_sampling_retains_full_records(self):
        header = (
            "VERSION 0.7\n"
            "FIELDS x y z intensity\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            "WIDTH 3\n"
            "HEIGHT 1\n"
            "POINTS 3\n"
            "DATA binary\n"
        ).encode("ascii")
        payload = b"".join(
            struct.pack("<ffff", value, 0.0, 1.0, value + 10.0)
            for value in (1.0, 2.0, 3.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.pcd"
            path.write_bytes(header + payload)
            message = read_binary_pcd_message(
                path,
                stamp=Time(),
                frame_id="velodyne",
                max_points=2,
            )

        self.assertEqual(message.width, 2)
        self.assertEqual(message.height, 1)
        self.assertEqual(message.point_step, 16)
        self.assertEqual(message.row_step, 32)
        np.testing.assert_allclose(
            pointcloud2_xyz_array(message),
            [[1.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
        )


class LidarMappingProcessorTest(unittest.TestCase):
    def test_keeps_360_degree_geometry_and_projects_visible_semantics(self):
        semantic = np.zeros((80, 100), dtype=np.uint8)
        semantic[40, 50] = 4
        semantic[40, 60] = 1
        points = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.2, 0.0, 2.0],
                [0.0, 0.0, -2.0],
                [20.0, 0.0, 0.0],
                [np.nan, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        observation = _processor().make_observation(points, semantic)

        self.assertEqual(observation.camera_points.shape, (3, 3))
        np.testing.assert_array_equal(
            observation.semantic_labels, np.array([4, 1, 0], dtype=np.uint8)
        )
        np.testing.assert_allclose(observation.sensor_origin_camera, [0, 0, 0])
        self.assertEqual(observation.statistics["input_points"], 5)
        self.assertEqual(observation.statistics["selected_points"], 3)
        self.assertEqual(observation.statistics["semantic_projected_points"], 2)

    def test_voxel_filter_and_cap_are_deterministic(self):
        points = np.array(
            [
                [0.10, 0.10, 1.0],
                [0.11, 0.11, 1.0],
                [1.00, 0.00, 1.0],
                [2.00, 0.00, 1.0],
                [3.00, 0.00, 1.0],
            ],
            dtype=np.float64,
        )
        semantic = np.zeros((80, 100), dtype=np.uint8)
        processor = _processor(voxel_size_m=0.5, max_points=2)

        first = processor.make_observation(points, semantic)
        second = processor.make_observation(points, semantic)

        self.assertEqual(first.statistics["voxel_filtered_points"], 4)
        self.assertEqual(first.statistics["selected_points"], 2)
        np.testing.assert_array_equal(first.camera_points, second.camera_points)

    def test_inverse_calibrated_rotation_aligns_map_with_lidar_axes(self):
        angle = np.deg2rad(20.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.2, -0.1, 0.3]
        processor = LidarMappingProcessor(
            transform_camera_lidar=transform,
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
        points_lidar = np.array([[1.0, 2.0, 3.0], [2.0, -1.0, 1.0]])
        observation = processor.make_observation(
            points_lidar, np.zeros((10, 10), dtype=np.uint8)
        )

        rotation_lidar_camera = processor.rotation_camera_lidar.T
        points_map = observation.camera_points @ rotation_lidar_camera.T
        sensor_origin_map = (
            observation.sensor_origin_camera @ rotation_lidar_camera.T
        )
        np.testing.assert_allclose(
            points_map - sensor_origin_map, points_lidar, atol=1e-12
        )


class PointCloudAssistedStereoTest(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_pointcloud_depth_corrects_ray_and_far_unmatched_is_dropped(self):
        keypoints = np.array(
            [[50.0, 40.0], [60.0, 40.0], [70.0, 40.0]],
            dtype=np.float64,
        )
        stereo_points = np.array(
            [[0.0, 0.0, 20.0], [0.4, 0.0, 4.0], [1.4, 0.0, 7.0]],
            dtype=np.float64,
        )
        cloud_points = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)

        result = assist_stereo_depth_from_pointcloud(
            keypoints=keypoints,
            stereo_camera_points=stereo_points,
            pointcloud_camera_points=cloud_points,
            camera_matrix=self.camera_matrix,
            image_width=100,
            image_height=80,
            pixel_radius=1.0,
            unmatched_stereo_max_depth_m=5.0,
        )

        np.testing.assert_array_equal(result.source_indices, [0, 1])
        np.testing.assert_array_equal(
            result.pointcloud_supported, [True, False]
        )
        np.testing.assert_allclose(
            result.camera_points,
            [[0.0, 0.0, 2.0], [0.4, 0.0, 4.0]],
        )
        self.assertEqual(result.statistics["pointcloud_supported_points"], 1)
        self.assertEqual(result.statistics["unmatched_stereo_dropped"], 1)

    def test_nearest_z_buffer_surface_wins_deterministically(self):
        keypoints = np.array([[50.0, 40.0]], dtype=np.float64)
        stereo_points = np.array([[0.0, 0.0, 9.0]], dtype=np.float64)
        cloud_points = np.array(
            [[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]], dtype=np.float64
        )

        first = assist_stereo_depth_from_pointcloud(
            keypoints=keypoints,
            stereo_camera_points=stereo_points,
            pointcloud_camera_points=cloud_points,
            camera_matrix=self.camera_matrix,
            image_width=100,
            image_height=80,
            pixel_radius=0.0,
            unmatched_stereo_max_depth_m=0.0,
        )
        second = assist_stereo_depth_from_pointcloud(
            keypoints=keypoints,
            stereo_camera_points=stereo_points,
            pointcloud_camera_points=cloud_points,
            camera_matrix=self.camera_matrix,
            image_width=100,
            image_height=80,
            pixel_radius=0.0,
            unmatched_stereo_max_depth_m=0.0,
        )

        np.testing.assert_allclose(first.camera_points, [[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(first.camera_points, second.camera_points)
        self.assertEqual(first.statistics["projected_pointcloud_pixels"], 1)

    def test_near_stereo_is_not_replaced_by_default(self):
        result = assist_stereo_depth_from_pointcloud(
            keypoints=np.array([[50.0, 40.0]], dtype=np.float64),
            stereo_camera_points=np.array(
                [[0.0, 0.0, 3.0]], dtype=np.float64
            ),
            pointcloud_camera_points=np.array(
                [[0.0, 0.0, 2.0]], dtype=np.float64
            ),
            camera_matrix=self.camera_matrix,
            image_width=100,
            image_height=80,
            pixel_radius=0.0,
            unmatched_stereo_max_depth_m=5.0,
            replace_stereo_min_depth_m=5.0,
        )

        np.testing.assert_allclose(result.camera_points, [[0.0, 0.0, 3.0]])
        np.testing.assert_array_equal(
            result.pointcloud_supported, [False]
        )
        self.assertEqual(result.statistics["pointcloud_matched_points"], 1)
        self.assertEqual(result.statistics["pointcloud_supported_points"], 0)

    def test_empty_cloud_retains_only_configured_near_stereo(self):
        result = assist_stereo_depth_from_pointcloud(
            keypoints=np.array([[50.0, 40.0], [60.0, 40.0]]),
            stereo_camera_points=np.array(
                [[0.0, 0.0, 3.0], [1.0, 0.0, 10.0]]
            ),
            pointcloud_camera_points=np.empty((0, 3)),
            camera_matrix=self.camera_matrix,
            image_width=100,
            image_height=80,
            pixel_radius=2.0,
            unmatched_stereo_max_depth_m=5.0,
        )

        np.testing.assert_array_equal(result.source_indices, [0])
        np.testing.assert_allclose(result.camera_points, [[0.0, 0.0, 3.0]])
        self.assertEqual(result.statistics["pointcloud_supported_points"], 0)
        self.assertEqual(result.statistics["unmatched_stereo_dropped"], 1)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            assist_stereo_depth_from_pointcloud(
                keypoints=np.zeros((1, 2)),
                stereo_camera_points=np.zeros((2, 3)),
                pointcloud_camera_points=np.empty((0, 3)),
                camera_matrix=self.camera_matrix,
                image_width=100,
                image_height=80,
            )
        with self.assertRaises(ValueError):
            assist_stereo_depth_from_pointcloud(
                keypoints=np.zeros((0, 2)),
                stereo_camera_points=np.zeros((0, 3)),
                pointcloud_camera_points=np.empty((0, 3)),
                camera_matrix=self.camera_matrix,
                image_width=100,
                image_height=80,
                pixel_radius=-1.0,
            )


class LidarEndpointOccupancyTest(unittest.TestCase):
    @staticmethod
    def _surface_coordinates():
        values = np.linspace(-1.0, 1.0, 9)
        first, second = np.meshgrid(values, values, indexing="ij")
        return first.reshape(-1), second.reshape(-1)

    def test_floor_wall_and_ceiling_have_distinct_endpoint_states(self):
        first, second = self._surface_coordinates()
        floor = np.column_stack(
            (first + 2.0, second, np.full(len(first), -1.0))
        )
        wall = np.column_stack(
            (np.full(len(first), 3.0), first, second)
        )
        ceiling = np.column_stack(
            (first + 2.0, second, np.full(len(first), 2.0))
        )

        floor_states = classify_lidar_endpoint_occupancy(
            floor, np.zeros(3)
        )
        wall_states = classify_lidar_endpoint_occupancy(
            wall, np.zeros(3)
        )
        ceiling_states = classify_lidar_endpoint_occupancy(
            ceiling, np.zeros(3)
        )

        self.assertGreater(
            np.count_nonzero(floor_states == ENDPOINT_FREE),
            0.9 * len(floor),
        )
        self.assertFalse(np.any(floor_states == ENDPOINT_OCCUPIED))
        for states in (wall_states, ceiling_states):
            self.assertGreater(
                np.count_nonzero(states == ENDPOINT_OCCUPIED),
                0.9 * len(states),
            )
            self.assertFalse(np.any(states == ENDPOINT_FREE))

    def test_linear_scan_ring_and_sparse_cloud_remain_unknown(self):
        line = np.column_stack(
            (
                np.linspace(1.0, 4.0, 40),
                np.zeros(40),
                np.zeros(40),
            )
        )
        sparse = np.eye(3, dtype=np.float64)

        np.testing.assert_array_equal(
            classify_lidar_endpoint_occupancy(line, np.zeros(3)),
            np.full(len(line), ENDPOINT_UNKNOWN, dtype=np.int8),
        )
        np.testing.assert_array_equal(
            classify_lidar_endpoint_occupancy(sparse, np.zeros(3)),
            np.full(len(sparse), ENDPOINT_UNKNOWN, dtype=np.int8),
        )

    def test_invalid_ground_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            LidarGroundSegmentationConfig(max_slope_deg=90.0)
        with self.assertRaises(ValueError):
            LidarGroundSegmentationConfig(neighbor_count=2)
        with self.assertRaises(ValueError):
            LidarGroundSegmentationConfig(neighbor_count=12.5)
        with self.assertRaises(ValueError):
            LidarGroundSegmentationConfig(max_neighbor_distance_m=0.0)


class LidarProvenanceTest(unittest.TestCase):
    def test_lidar_observation_keeps_points_behind_camera(self):
        points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
        labels = np.array([1, 0], dtype=np.uint8)
        stereo = SemanticKeyFrameObservation(
            keyframe_id=1,
            sensor_timestamp=1.0,
            camera_points=points,
            semantic_labels=labels,
        )
        lidar = SemanticKeyFrameObservation(
            keyframe_id=1,
            sensor_timestamp=1.0,
            camera_points=points,
            semantic_labels=labels,
            sensor_origin_camera=np.array([0.2, -0.1, 0.3]),
            require_positive_depth=False,
            source="lidar",
        )

        self.assertEqual(len(stereo.camera_points), 1)
        self.assertEqual(len(lidar.camera_points), 2)
        np.testing.assert_allclose(lidar.sensor_origin_camera, [0.2, -0.1, 0.3])
        with self.assertRaises(ValueError):
            lidar.sensor_origin_camera[0] = 9.0

    def test_source_and_sensor_origin_are_part_of_conflict_detection(self):
        store = KeyFrameObservationStore()
        stereo = SemanticKeyFrameObservation(
            keyframe_id=2,
            sensor_timestamp=2.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([1], dtype=np.uint8),
        )
        lidar = SemanticKeyFrameObservation(
            keyframe_id=2,
            sensor_timestamp=2.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([1], dtype=np.uint8),
            sensor_origin_camera=np.array([0.1, 0.0, 0.0]),
            require_positive_depth=False,
            source="lidar",
        )
        self.assertTrue(store.add(stereo))
        with self.assertRaises(ValueError):
            store.add(lidar)

    def test_fusion_uses_lidar_origin_for_ray_tracing(self):
        class CapturingGrid(LayeredSemanticGridMap):
            def update(self, **kwargs):
                self.captured_origin = kwargs["camera_position_map"].copy()
                self.captured_endpoint_states = kwargs[
                    "geometric_endpoint_states"
                ].copy()
                return {"accepted_points": len(kwargs["points_map"])}

        grid = CapturingGrid(
            resolution=1.0,
            length_x=10.0,
            length_y=10.0,
        )
        observation = SemanticKeyFrameObservation(
            keyframe_id=3,
            sensor_timestamp=3.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([1], dtype=np.uint8),
            sensor_origin_camera=np.array([0.2, -0.3, 0.4]),
            require_positive_depth=False,
            source="lidar",
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [1.0, 2.0, 3.0]

        KeyFrameObservationStore.fuse_observation(
            grid=grid,
            observation=observation,
            pose_world_camera=pose,
            rotation_map_world=np.eye(3),
            map_translation_offset=np.zeros(3),
        )

        np.testing.assert_allclose(grid.captured_origin, [1.2, 1.7, 3.4])
        np.testing.assert_array_equal(
            grid.captured_endpoint_states,
            [ENDPOINT_UNKNOWN],
        )

    def test_stereo_observation_does_not_enable_lidar_endpoint_states(self):
        class CapturingGrid(LayeredSemanticGridMap):
            def update(self, **kwargs):
                self.endpoint_states = kwargs["geometric_endpoint_states"]
                return {"accepted_points": len(kwargs["points_map"])}

        grid = CapturingGrid(
            resolution=1.0,
            length_x=10.0,
            length_y=10.0,
        )
        observation = SemanticKeyFrameObservation(
            keyframe_id=4,
            sensor_timestamp=4.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([0], dtype=np.uint8),
            source="stereo",
        )

        KeyFrameObservationStore.fuse_observation(
            grid=grid,
            observation=observation,
            pose_world_camera=np.eye(4),
            rotation_map_world=np.eye(3),
            map_translation_offset=np.zeros(3),
        )

        self.assertIsNone(grid.endpoint_states)


if __name__ == "__main__":
    unittest.main()
