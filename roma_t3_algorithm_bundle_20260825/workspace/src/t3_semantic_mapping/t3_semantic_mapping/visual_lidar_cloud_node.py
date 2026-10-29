"""Accumulate LiDAR keyframes registered by accepted RoMa visual poses."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .lidar_mapping import pointcloud2_xyz_array
from .pointcloud_utils import make_xyz_point_cloud
from .voxel_cloud_store import VoxelCloudStore


class VisualLidarCloudNode(Node):
    def __init__(self):
        super().__init__("visual_lidar_cloud_mapper")
        defaults = {
            "input_topic": "/T3/mapping/lidar_keyframe",
            "global_topic": "/T3/mapping/lidar_map",
            "incremental_topic": "/T3/mapping/lidar_incremental",
            "map_frame": "map",
            "output_dir": "",
            "voxel_size": 0.20,
            "publish_period": 2.0,
            "save_period": 20.0,
            "preview_max_points": 300000,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.values = {name: self.get_parameter(name).value for name in defaults}
        if not self.values["output_dir"]:
            raise ValueError("output_dir is required")
        if min(self.values["publish_period"], self.values["save_period"],
               self.values["preview_max_points"]) <= 0:
            raise ValueError("Publication/save periods and preview limit must be positive")
        output = Path(self.values["output_dir"])
        self.store = VoxelCloudStore(output / "lidar_map.sqlite3", self.values["voxel_size"])
        self.pcd_path = output / "lidar_map.pcd"
        self.status_path = output / "lidar_map_status.json"
        self.frames = 0
        self.rejected_frames = 0
        self.last_header = None
        self.last_published_count = -1
        self.last_save = time.monotonic()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lidar_pcd")
        self.save_future = None
        self.closed = False
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.global_pub = self.create_publisher(PointCloud2, self.values["global_topic"], retained)
        self.incremental_pub = self.create_publisher(
            PointCloud2, self.values["incremental_topic"], reliable)
        self.status_pub = self.create_publisher(String, "/T3/mapping/lidar_status", retained)
        self.subscription = self.create_subscription(
            PointCloud2, self.values["input_topic"], self.on_cloud, reliable)
        self.timer = self.create_timer(self.values["publish_period"], self.publish_map)
        self.get_logger().info(
            "LiDAR cloud mapper ready; geometry uses accepted visual poses; "
            f"voxel={self.store.voxel_size}m, database={self.store.path}")

    def on_cloud(self, message):
        if message.header.frame_id != self.values["map_frame"]:
            self.rejected_frames += 1
            self.get_logger().error(f"Rejected LiDAR keyframe in {message.header.frame_id!r}")
            return
        try:
            points = pointcloud2_xyz_array(message)
            self.store.append(points)
        except (ValueError, TypeError) as error:
            self.rejected_frames += 1
            self.get_logger().error(f"Rejected malformed LiDAR keyframe: {error}")
            return
        self.last_header = message.header
        self.frames += 1
        # This topic is a registered observation, so repeated voxels are valid.
        self.incremental_pub.publish(message)

    def publish_map(self):
        if self.last_header is None:
            return
        if self.last_published_count != self.store.count:
            points = self.store.preview(self.values["preview_max_points"])
            self.global_pub.publish(make_xyz_point_cloud(
                header=self.last_header, points_xyz=points))
            self.last_published_count = self.store.count
        status = {
            "registered_keyframes": self.frames,
            "rejected_frames": self.rejected_frames,
            "voxel_count": self.store.count,
            "voxel_size_m": self.store.voxel_size,
            "pose_source": "accepted_RoMa_visual_map_pose",
            "frame_id": self.values["map_frame"],
            "preview_max_points": self.values["preview_max_points"],
            "database": str(self.store.path),
            "pcd": str(self.pcd_path),
        }
        encoded = json.dumps(status, indent=2)
        self.status_pub.publish(String(data=encoded))
        self.status_path.write_text(encoded + "\n", encoding="utf-8")
        if self.save_future is not None and self.save_future.done():
            try:
                self.save_future.result()
            except Exception as error:
                self.get_logger().error(f"PCD export failed; SQLite is retained: {error}")
            self.save_future = None
        if self.save_future is None and time.monotonic() - self.last_save >= self.values["save_period"]:
            self.save_future = self.writer.submit(
                VoxelCloudStore.export_pcd, self.store.path, self.pcd_path)
            self.last_save = time.monotonic()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.writer.shutdown(wait=True)
        try:
            if self.store.count:
                count = VoxelCloudStore.export_pcd(self.store.path, self.pcd_path)
                message = f"Saved complete LiDAR map: {count} voxels -> {self.pcd_path}"
                if rclpy.ok():
                    self.get_logger().info(message)
                else:
                    print(message, flush=True)
        finally:
            self.store.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisualLidarCloudNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
