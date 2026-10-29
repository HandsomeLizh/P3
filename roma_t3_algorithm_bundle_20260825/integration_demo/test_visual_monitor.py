"""Native ROS message/GUI math checks; no car commands or synthetic live estimates."""
import os
os.environ.pop('T3_VISUAL_RUNTIME', None)
import threading
import unittest
from types import SimpleNamespace
import numpy as np
import rclpy
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from t3_semantic_mapping.grid_map_message import make_grid_map_message
from t3_semantic_mapping.semantic_grid import GridGeometry
from t3_semantic_mapping.lidar_mapping import pointcloud2_xyz_array
from visual_monitor import colored_cloud, Monitor, Window
from visual_style import height_colors


class VisualMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_packed_rgb_cloud_preserves_coordinates(self):
        points = np.array([[1, 2, -1], [3, 4, 0], [5, 6, 1]], dtype=np.float32)
        message = colored_cloud(Header(frame_id='map'), points, (-1., 1.))
        np.testing.assert_array_equal(pointcloud2_xyz_array(message), points)
        self.assertEqual(len(message.data), 48)
        rgb = np.frombuffer(message.data, dtype='<u4').reshape(-1, 4)[:, 3]
        actual = np.column_stack([(rgb>>16)&255, (rgb>>8)&255, rgb&255])
        np.testing.assert_array_equal(actual, height_colors(points[:, 2], -1., 1.))

    def test_non_square_grid_decode_preserves_resolution_origin_and_cells(self):
        node = Monitor()
        try:
            elevation = np.array([[1, 2, 3], [4, np.nan, 6]], dtype=np.float32)
            geometry = GridGeometry(.2, 3, 2, 12., -4.)
            message = make_grid_map_message(header=Header(frame_id='map'), geometry=geometry,
                                            layers={'elevation': elevation})
            node.on_grid(message)
            np.testing.assert_array_equal(node.elevation_data[0], elevation)
            np.testing.assert_allclose(node.grid[1], (12., -4., .2, 3, 2))
            expected = np.flip(height_colors(elevation, *node.height_limits), axis=0)
            np.testing.assert_array_equal(np.asarray(node.grid[0]), expected)
            self.assertAlmostEqual(message.info.length_x, .6)
            self.assertAlmostEqual(message.info.length_y, .4)
            node.contrast_requested = True
            node.apply_color_mode()
            self.assertTrue(node.grid[5])
            np.testing.assert_array_equal(node.elevation_data[0], elevation)
        finally:
            node.destroy_node()

    def test_zoom_pan_reset_do_not_modify_map_or_path(self):
        path = Path(header=Header(frame_id='map'))
        for x, y in [(0., 0.), (10., 20.)]:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.w = 1.
            path.poses.append(pose)
        fake = SimpleNamespace(lock=threading.Lock(), path=path, grid=None)
        stop = threading.Event()
        window = Window(fake, stop)
        window.root.withdraw()
        try:
            window.root.update_idletasks()
            window.redraw()
            center, before = window.current_view
            window.zoom(SimpleNamespace(x=50, y=50), 2.)
            self.assertAlmostEqual(window.current_view[1], before*2)
            window.drag_at = (50, 50)
            window.pan(SimpleNamespace(x=75, y=70))
            self.assertIsNotNone(window.manual_view)
            window.reset_view()
            self.assertIsNone(window.manual_view)
            self.assertEqual([(p.pose.position.x, p.pose.position.y) for p in path.poses], [(0., 0.), (10., 20.)])
        finally:
            stop.set()
            window.root.destroy()


if __name__ == '__main__':
    unittest.main()
