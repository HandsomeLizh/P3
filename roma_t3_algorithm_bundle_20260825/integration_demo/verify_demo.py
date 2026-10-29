#!/usr/bin/env python3
"""Passive interface check, or an explicitly selected recorded Env_X smoke test.

Recorded mode publishes ONLY measured images/clouds from files. It does not
publish a pose, command the car, or count as live simulation acceptance.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path as NavPath
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header, String
from t3_interfaces.msg import IncrementalSemanticMap

from t3_semantic_mapping.pointcloud_utils import make_xyz_point_cloud


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class Probe(Node):
    def __init__(self):
        super().__init__("t3_demo_verification")
        self.counts = {}
        self.latest = {}
        self.completed = set()
        self.timings = []
        self.pose_positions = []
        self.live = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        types = {
            "/T3/semantic/current_pose": Odometry,
            "/T3/semantic/trajectory": NavPath,
            "/T3/semantic/incremental_map": IncrementalSemanticMap,
            "/T3/mapping/global_grid_map": GridMap,
            "/T3/mapping/lidar_map": PointCloud2,
            "/T3/mapping/lidar_incremental": PointCloud2,
        }
        self.subscriptions_kept = [self.create_subscription(
            kind, topic, lambda message, name=topic: self.record(name, message),
            self.live if kind == Odometry or topic.endswith("lidar_incremental") else self.retained)
            for topic, kind in types.items()]
        self.create_subscription(Header, "/Car/T3/debug/input_processed",
                                 lambda header: self.completed.add(stamp_ns(header)), self.live)
        self.create_subscription(String, "/Car/T3/metrics/frame_timing",
                                 lambda message: self.timings.append(json.loads(message.data)), self.live)

    def record(self, topic, message):
        self.counts[topic] = self.counts.get(topic, 0) + 1
        self.latest[topic] = message
        if isinstance(message, Odometry):
            p = message.pose.pose.position
            self.pose_positions.append([p.x, p.y, p.z])

    def spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def replay(self, dataset, pairs, start_index=0):
        topics = ["/Car/T5/Cam_Left/image_raw/color", "/Car/T5/Cam_Right/image_raw/color",
                  "/Car/T5/OS1/points"]
        publishers = [self.create_publisher(kind, topic, self.live)
                      for kind, topic in zip([Image, Image, PointCloud2], topics)]
        if not self.spin_until(lambda: all(p.get_subscription_count() for p in publishers), 20):
            raise RuntimeError("Mapper input subscriptions are missing")
        if any(self.count_publishers(topic) != 1 for topic in topics):
            raise RuntimeError("Another sensor publisher exists; refusing to mix recorded and live data")
        batches = [path for path in sorted(Path(dataset).glob("batch_*"))
                   if all((path / filename).is_file() for filename in
                          ("RGB_L.png", "RGB_R.png", "cpulidar_0.pcd", "meta.json"))][start_index:start_index + pairs]
        if len(batches) < pairs:
            raise RuntimeError("Not enough complete recorded sensor batches")
        bridge = CvBridge()
        for index, batch in enumerate(batches):
            metadata = json.loads((batch / "meta.json").read_text())
            stamp = int(datetime.strptime(metadata["timestamp"], "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1e9)
            header = Header(frame_id="sensors")
            header.stamp.sec, header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            content = (batch / "cpulidar_0.pcd").read_bytes()
            pcd_header, payload = content.split(b"DATA binary\n", 1)
            if b"FIELDS x y z\n" not in pcd_header or b"TYPE F F F\n" not in pcd_header:
                raise ValueError("Recorded PCD is not binary float32 XYZ")
            points = np.frombuffer(payload, dtype="<f4").reshape(-1, 3)
            publishers[2].publish(make_xyz_point_cloud(header=header, points_xyz=points))
            self.spin_until(lambda: False, 0.5)
            for pub, filename, frame in zip(publishers[:2], ["RGB_L.png", "RGB_R.png"], ["rgb_0", "rgb_1"]):
                image = cv2.imread(str(batch / filename))
                if image is None:
                    raise ValueError(f"Cannot decode {batch / filename}")
                message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
                message.header.stamp = header.stamp
                message.header.frame_id = frame
                pub.publish(message)
            if not self.spin_until(lambda: stamp in self.completed, 90):
                raise RuntimeError(f"No processing receipt for recorded frame {index + 1}")
            print(f"Recorded batch {batch.name}: {index + 1}/{pairs} processed; "
                  f"pose_messages={len(self.pose_positions)}", flush=True)

    def report(self, source):
        result = {"source": source, "topic_message_counts": self.counts,
                  "pose_count": len(self.pose_positions), "processing_receipts": len(self.completed)}
        cloud = self.latest.get("/T3/mapping/lidar_map")
        result["lidar_preview_points"] = 0 if cloud is None else cloud.width * cloud.height
        grid = self.latest.get("/T3/mapping/global_grid_map")
        result["elevation_known_cells"] = 0
        if grid is not None and "elevation" in grid.layers:
            values = np.asarray(grid.data[list(grid.layers).index("elevation")].data)
            result["elevation_known_cells"] = int(np.isfinite(values).sum())
        result["pose_frames"] = list({
            (self.latest["/T3/semantic/current_pose"].header.frame_id,
             self.latest["/T3/semantic/current_pose"].child_frame_id)
        }) if self.pose_positions else []
        result["all_required_topics_received"] = all(self.counts.get(topic, 0) > 0 for topic in (
            "/T3/semantic/current_pose", "/T3/semantic/trajectory", "/T3/semantic/incremental_map",
            "/T3/mapping/global_grid_map", "/T3/mapping/lidar_map", "/T3/mapping/lidar_incremental"))
        result["poses_finite"] = bool(np.isfinite(self.pose_positions).all())
        if self.pose_positions:
            positions = np.asarray(self.pose_positions)
            result["visual_travel_m"] = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        result["timing_messages"] = self.timings
        result["passed"] = bool(result["all_required_topics_received"] and result["poses_finite"]
                                 and result["lidar_preview_points"] and result["elevation_known_cells"])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    probe = Probe()
    try:
        if args.dataset:
            probe.replay(args.dataset, args.pairs, args.start_index)
        probe.spin_until(lambda: False, args.seconds)
        result = probe.report("recorded_envx_smoke_test" if args.dataset else "live_ros_topics")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "timing_messages"}, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
