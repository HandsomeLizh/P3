#!/usr/bin/env python3
"""Replay selected bag topics using the SQLite record time.

Raw stereo image headers in several T5 bags are zero or inconsistent.  This
player replaces every available ``header.stamp`` with the rosbag record time.
PQ and IMU topics can be replayed at the same time for evaluation.  Messages
without a Header (PQ and WIT angle arrays) keep their payload unchanged.  For
PQ, an additional internal topic carries the exact bag record timestamp so a
passive run recorder can build a self-contained reference trajectory later.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String


DEFAULT_LEFT = "/Car/T5/Cam_Left/image_raw/color"
DEFAULT_RIGHT = "/Car/T5/Cam_Right/image_raw/color"
DEFAULT_PQ = "/Car/T5/pq_data"
DEFAULT_STAMPED_PQ = "/t3/reference/pq_stamped"
DEFAULT_WIT_IMU = "/Car/T5/wit/imu"
DEFAULT_WIT_ANGLE = "/Car/T5/wit/angle"
DEFAULT_OUSTER_IMU = "/Car/T5/OS1/imu"
DEFAULT_LEFT_CAMERA_INFO = "/Car/T5/Cam_Left/camera_info"
DEFAULT_RIGHT_CAMERA_INFO = "/Car/T5/Cam_Right/camera_info"
DEFAULT_TOF_LEFT_CAMERA_INFO = "/Car/T5/TOF_Left/camera_info"
DEFAULT_TOF_RIGHT_CAMERA_INFO = "/Car/T5/TOF_Right/camera_info"
DEFAULT_TOF_LEFT_POINTS = "/Car/T5/TOF_Left/points"
DEFAULT_TOF_RIGHT_POINTS = "/Car/T5/TOF_Right/points"
DEFAULT_TOF_LEFT_DEPTH = "/Car/T5/TOF_Left/depth"
DEFAULT_TOF_RIGHT_DEPTH = "/Car/T5/TOF_Right/depth"
DEFAULT_LIDAR_POINTS = "/Car/T5/OS1/points"
DEFAULT_HW2_IMAGE = "/Car/T5/HW2/image_raw"
DEFAULT_EVENT_LEFT = "/Car/T5/event_camera_left/cd_events"
DEFAULT_EVENT_RIGHT = "/Car/T5/event_camera_right/cd_events"
DEFAULT_RADAR_POINTS = "/Car/T5/radar/points"
DEFAULT_CLOCK = "/clock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay raw stereo Image topics at their original rate, replacing "
            "invalid message header stamps with rosbag record timestamps."
        )
    )
    parser.add_argument("bag", help="ROS 2 bag directory")
    parser.add_argument("--left-topic", default=DEFAULT_LEFT)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT)
    parser.add_argument("--pq-topic", default=DEFAULT_PQ)
    parser.add_argument(
        "--stamped-pq-topic",
        default=DEFAULT_STAMPED_PQ,
    )
    parser.add_argument("--wit-imu-topic", default=DEFAULT_WIT_IMU)
    parser.add_argument("--wit-angle-topic", default=DEFAULT_WIT_ANGLE)
    parser.add_argument("--ouster-imu-topic", default=DEFAULT_OUSTER_IMU)
    parser.add_argument("--clock-topic", default=DEFAULT_CLOCK)
    parser.add_argument(
        "--with-pq",
        action="store_true",
        help="Also replay the PQ reference topic when the bag contains data.",
    )
    parser.add_argument(
        "--with-imu",
        action="store_true",
        help="Also replay WIT IMU, WIT angle and Ouster IMU topics.",
    )
    parser.add_argument(
        "--with-clock",
        action="store_true",
        help="Also replay the recorded /clock topic when the bag contains it.",
    )
    parser.add_argument(
        "--all-metrics-topics",
        action="store_true",
        help="Equivalent to --with-pq --with-imu.",
    )
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Stop after this much bag time; zero plays the complete bag.",
    )
    parser.add_argument("--subscriber-wait-sec", type=float, default=30.0)
    return parser.parse_args()


def selected_topics(args: argparse.Namespace) -> list[str]:
    topics = [args.left_topic, args.right_topic]
    topics.extend([
        DEFAULT_LEFT_CAMERA_INFO, DEFAULT_RIGHT_CAMERA_INFO,
        DEFAULT_TOF_LEFT_CAMERA_INFO, DEFAULT_TOF_RIGHT_CAMERA_INFO,
        DEFAULT_TOF_LEFT_POINTS, DEFAULT_TOF_RIGHT_POINTS,
        DEFAULT_TOF_LEFT_DEPTH, DEFAULT_TOF_RIGHT_DEPTH,
        DEFAULT_LIDAR_POINTS, DEFAULT_HW2_IMAGE,
        DEFAULT_EVENT_LEFT, DEFAULT_EVENT_RIGHT, DEFAULT_RADAR_POINTS,
        DEFAULT_OUSTER_IMU, DEFAULT_WIT_IMU, DEFAULT_WIT_ANGLE,
    ])
    if args.with_pq or args.all_metrics_topics:
        topics.append(args.pq_topic)
    if args.with_imu or args.all_metrics_topics:
        topics.extend(
            [args.wit_imu_topic, args.wit_angle_topic, args.ouster_imu_topic]
        )
    if args.with_clock:
        topics.append(args.clock_topic)
    return list(OrderedDict.fromkeys(topics))


class SensorReplay(Node):
    def __init__(
        self,
        topic_types: dict[str, str],
        required_topics: list[str],
        pq_topic: str,
        stamped_pq_topic: str,
    ) -> None:
        super().__init__("bag_record_stamp_stereo_replay")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.topic_message_types = {
            topic: get_message(type_name)
            for topic, type_name in topic_types.items()
        }
        self.topic_publishers = {
            topic: self.create_publisher(self.topic_message_types[topic], topic, qos)
            for topic in topic_types
        }
        self.pq_topic = pq_topic
        self.stamped_pq_publisher = (
            self.create_publisher(String, stamped_pq_topic, qos)
            if pq_topic in topic_types
            else None
        )
        self.required_topics = required_topics

    def subscribers_ready(self) -> bool:
        return all(
            self.topic_publishers[topic].get_subscription_count() > 0
            for topic in self.required_topics
        )


def wait_for_subscribers(node: SensorReplay, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.subscribers_ready():
            return
    counts = {
        topic: node.topic_publishers[topic].get_subscription_count()
        for topic in node.required_topics
    }
    raise RuntimeError(f"stereo subscribers not ready; subscription counts={counts}")


def sleep_until(node: SensorReplay, deadline: float) -> None:
    while rclpy.ok():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        rclpy.spin_once(node, timeout_sec=min(remaining, 0.05))


def main() -> int:
    args = parse_args()
    if args.rate <= 0.0:
        raise ValueError("--rate must be positive")

    rclpy.init()
    requested_topics = selected_topics(args)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    available_types = {
        value.name: value.type for value in reader.get_all_topics_and_types()
    }
    missing_topics = [topic for topic in requested_topics if topic not in available_types]
    required_topics = [args.left_topic, args.right_topic]
    missing_required = [topic for topic in required_topics if topic not in available_types]
    if missing_required:
        raise RuntimeError(f"required stereo topics missing from bag: {missing_required}")
    replay_topics = [topic for topic in requested_topics if topic in available_types]
    for topic in missing_topics:
        print(f"Warning: requested topic is not registered in bag: {topic}", flush=True)

    node = SensorReplay(
        {topic: available_types[topic] for topic in replay_topics},
        required_topics,
        args.pq_topic,
        args.stamped_pq_topic,
    )
    try:
        storage_filter = rosbag2_py.StorageFilter()
        storage_filter.topics = replay_topics
        reader.set_filter(storage_filter)

        wait_for_subscribers(node, args.subscriber_wait_sec)
        first_bag_ns = None
        wall_start = None
        published = {topic: 0 for topic in replay_topics}

        while rclpy.ok() and reader.has_next():
            topic, serialized, bag_ns = reader.read_next()
            if first_bag_ns is None:
                first_bag_ns = bag_ns
                wall_start = time.monotonic()
            elapsed_bag_sec = (bag_ns - first_bag_ns) * 1e-9
            if args.duration_sec > 0.0 and elapsed_bag_sec > args.duration_sec:
                break

            deadline = wall_start + elapsed_bag_sec / args.rate
            sleep_until(node, deadline)
            message = deserialize_message(
                serialized, node.topic_message_types[topic]
            )
            if hasattr(message, "header") and hasattr(message.header, "stamp"):
                message.header.stamp.sec = int(bag_ns // 1_000_000_000)
                message.header.stamp.nanosec = int(bag_ns % 1_000_000_000)
            if (
                topic == node.pq_topic
                and node.stamped_pq_publisher is not None
            ):
                stamped_message = String()
                stamped_message.data = json.dumps(
                    {
                        "timestamp_ns": int(bag_ns),
                        "values": [float(value) for value in message.data],
                    },
                    separators=(",", ":"),
                )
                node.stamped_pq_publisher.publish(stamped_message)
            node.topic_publishers[topic].publish(message)
            published[topic] += 1

        counts = ", ".join(
            f"{topic}={count}" for topic, count in published.items()
        )
        print(
            f"Replay complete: {counts}, rate={args.rate:.3f}x, "
            f"duration={args.duration_sec:.3f}s",
            flush=True,
        )
        rclpy.spin_once(node, timeout_sec=1.0)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
