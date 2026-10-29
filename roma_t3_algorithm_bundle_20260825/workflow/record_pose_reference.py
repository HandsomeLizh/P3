#!/usr/bin/env python3
"""Record a PoseStamped ground-truth topic as a TUM trajectory."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--topic", default="/car/pose")
    parser.add_argument("--parent-pid", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    internal = output_dir / "_internal"
    internal.mkdir(parents=True, exist_ok=True)
    trajectory_path = internal / "reference_pose.tum"
    metadata_path = internal / "reference_pose_recording.json"
    started = time.time()

    class PoseRecorder(Node):
        def __init__(self) -> None:
            super().__init__("t3_pose_reference_recorder")
            self.count = 0
            self.first_stamp_ns = None
            self.last_stamp_ns = None
            self.stream = trajectory_path.open("w", encoding="utf-8", buffering=1)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1000,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.subscription = self.create_subscription(
                PoseStamped, args.topic, self.on_pose, qos
            )
            self.timer = (
                self.create_timer(1.0, self.check_parent)
                if args.parent_pid > 0
                else None
            )
            self.get_logger().info(
                f"Saving PoseStamped reference {args.topic} to {trajectory_path}"
            )

        def on_pose(self, message: PoseStamped) -> None:
            stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            pose = message.pose
            self.stream.write(
                f"{stamp_ns * 1e-9:.9f} "
                f"{pose.position.x:.12g} {pose.position.y:.12g} "
                f"{pose.position.z:.12g} {pose.orientation.x:.12g} "
                f"{pose.orientation.y:.12g} {pose.orientation.z:.12g} "
                f"{pose.orientation.w:.12g}\n"
            )
            self.count += 1
            if self.first_stamp_ns is None:
                self.first_stamp_ns = stamp_ns
            self.last_stamp_ns = stamp_ns

        def check_parent(self) -> None:
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                rclpy.shutdown()
            except PermissionError:
                pass

        def close(self) -> None:
            if self.stream.closed:
                return
            self.stream.close()
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "geometry_msgs/msg/PoseStamped",
                        "topic": args.topic,
                        "trajectory_file": str(trajectory_path),
                        "sample_count": self.count,
                        "first_stamp_ns": self.first_stamp_ns,
                        "last_stamp_ns": self.last_stamp_ns,
                        "started_wall_time_sec": started,
                        "ended_wall_time_sec": time.time(),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    rclpy.init()
    node = PoseRecorder()

    def stop(_signum, _frame) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(node)
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
