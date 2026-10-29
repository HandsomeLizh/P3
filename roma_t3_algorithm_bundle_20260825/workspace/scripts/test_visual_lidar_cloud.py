"""Regression checks for visual registration and durable full 3D geometry."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/t3_semantic_mapping"))

from t3_semantic_mapping.voxel_cloud_store import VoxelCloudStore


class VoxelStoreTests(unittest.TestCase):
    def test_envx_flu_input_mounting_converts_both_coordinate_ends(self):
        import yaml
        root = Path(__file__).resolve().parents[2]
        original = yaml.safe_load((root / 'workspace/src/t3_semantic_mapping/config/t5_rgb10_rgb11.yaml').read_text())
        measured = np.asarray(original['mounting']['transform_vehicle_lidar'])
        profile = yaml.safe_load((root / 'config/pointcloud_extrinsic_os1_mapping_only.yaml').read_text())
        sensor = profile['sensors']['os1']
        actual = np.asarray(sensor['transform_base_sensor'])
        conversion = np.diag([1., -1., -1., 1.])
        np.testing.assert_allclose(actual, conversion @ measured @ conversion, atol=1e-12)
        self.assertEqual(sensor['input_axes'], 'x_forward_y_left_z_up')
        # An observed floor below an upright FLU sensor must not become a ceiling.
        floor = actual @ np.array([0., 0., -1., 1.])
        self.assertLess(floor[2], 0.)
        self.assertGreater((actual @ np.array([0., 1., 0., 0.]))[1], .99)

    def test_vertical_structure_deduplication_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.sqlite3"
            store = VoxelCloudStore(path, 0.2)
            points = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02],
                               [0.01, 0.01, 1.01], [-0.01, 0.01, 0.01],
                               [np.nan, 0, 0]])
            self.assertEqual(store.append(points), 3)
            self.assertEqual(store.append(points), 0)
            self.assertEqual(store.count, 3)
            self.assertEqual(len(np.unique(store.preview()[:, 2])), 2)
            store.close()
            store = VoxelCloudStore(path, 0.2)
            self.assertEqual(store.count, 3)
            self.assertEqual(store.append(points), 0)
            store.close()
            with self.assertRaises(ValueError):
                VoxelCloudStore(path, 0.3)

    def test_full_binary_pcd_is_not_preview_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cloud.sqlite3"
            destination = Path(directory) / "cloud.pcd"
            store = VoxelCloudStore(database, 0.2)
            points = np.arange(300, dtype=np.float32).reshape(-1, 3)
            store.append(points)
            self.assertLessEqual(len(store.preview(7)), 7)
            self.assertEqual(store.export_pcd(database, destination), 100)
            header, payload = destination.read_bytes().split(b"DATA binary\n", 1)
            self.assertIn(b"POINTS 100", header)
            np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4").reshape(-1, 3), points)
            store.close()

    def test_registered_cloud_retains_wall_heights_and_body_transform(self):
        from std_msgs.msg import Header
        from t3_semantic_mapping.lidar_mapping import pointcloud2_xyz_array
        from t3_semantic_mapping.pointcloud_utils import make_xyz_point_cloud
        from t3_semantic_mapping.terrain_pointcloud_mapper import TerrainPointCloudMapper
        import yaml

        class Logger:
            def warning(self, message):
                raise AssertionError(message)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "sensor.yaml"
            sensor_transform = np.eye(4)
            sensor_transform[0, 3] = 1.
            config.write_text(yaml.safe_dump({
                "version": 1, "base_frame": "base_link", "map_frame": "map",
                "sensors": {"lidar": {
                    "topic": "/lidar", "expected_frame_id": "sensor", "queue_size": 5,
                    "max_time_offset_sec": 0.1, "min_range_m": 0., "max_range_m": 80.,
                    "voxel_size_m": 0.0, "max_points": 100,
                    "transform_base_sensor": sensor_transform.tolist(),
                }},
            }))
            mapper = TerrainPointCloudMapper(config, logger=Logger(), retain_registered_points=True)
            header = Header(frame_id="sensor")
            header.stamp.sec = 10
            points = np.array([[1.01, 0.01, 1.], [1.01, 0.01, 2.]])
            cloud = make_xyz_point_cloud(header=header, points_xyz=points)
            np.testing.assert_allclose(pointcloud2_xyz_array(cloud), points)
            mapper.record_message("lidar", cloud)
            pose = np.eye(4)
            pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            pose[:3, 3] = [10, 20, 3]
            result = mapper.prepare_update(target_stamp_ns=10_000_000_000,
                                           transform_map_base=pose, map_resolution=0.2)
            self.assertEqual(len(result.points_map), 1)
            self.assertEqual(len(result.registered_points_map), 2)
            np.testing.assert_allclose(result.registered_points_map,
                                       [[9.99, 22.01, 4.], [9.99, 22.01, 5.]], rtol=1e-6)
            mapper.mark_committed(result.selected_stamps_ns)
            repeated = mapper.prepare_update(target_stamp_ns=10_000_000_000,
                                             transform_map_base=pose, map_resolution=0.2)
            self.assertEqual(len(repeated.points_map), 0)
            mapper.close()


if __name__ == "__main__":
    unittest.main()
