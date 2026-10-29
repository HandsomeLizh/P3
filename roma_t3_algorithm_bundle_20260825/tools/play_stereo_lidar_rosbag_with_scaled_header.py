#!/usr/bin/env python3
"""Replay paired ROS 2 stereo images with a scaled common Header.stamp.

The physical-bag path continues to use native ``ros2 bag play``.  This helper
is only selected when ``T3_HEADER_TIME_SCALE`` differs from 1.  It pairs the
left and right streams by ordinal position, scales elapsed sensor time around
the first pair, assigns the resulting common timestamp to both images, and
uses that corrected sensor timeline for rate control.
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from collections import deque
from pathlib import Path

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--header-time-scale", type=float, required=True)
    parser.add_argument(
        "--left-topic", default="/Car/T5/Cam_Left/image_raw/color"
    )
    parser.add_argument(
        "--right-topic", default="/Car/T5/Cam_Right/image_raw/color"
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--reference-type", choices=("none", "pose_stamped"), default="none"
    )
    parser.add_argument("--reference-topic", default="/car/pose")
    parser.add_argument("--lidar-topic", default="/Car/T5/OS1/points")
    parser.add_argument("--lidar-lead-sec", type=float, default=0.2)
    parser.add_argument("--discovery-timeout-sec", type=float, default=15.0)
    return parser.parse_args()


def stamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def set_stamp_ns(message: Image, value: int) -> None:
    message.header.stamp.sec = int(value // 1_000_000_000)
    message.header.stamp.nanosec = int(value % 1_000_000_000)


def sleep_until(node: Node, deadline: float) -> None:
    while rclpy.ok():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        rclpy.spin_once(node, timeout_sec=min(remaining, 0.05))


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0.0:
        raise ValueError("--rate must be a positive finite number")
    if not math.isfinite(args.header_time_scale) or args.header_time_scale <= 0.0:
        raise ValueError("--header-time-scale must be a positive finite number")
    if not math.isfinite(args.lidar_lead_sec) or args.lidar_lead_sec < 0.0:
        raise ValueError("--lidar-lead-sec must be a non-negative finite number")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")

    bag = args.bag.expanduser().resolve()
    if not bag.exists():
        raise FileNotFoundError(bag)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    selected_topics = [args.left_topic, args.right_topic, args.lidar_topic]
    if args.reference_type == "pose_stamped":
        selected_topics.append(args.reference_topic)
    reader.set_filter(rosbag2_py.StorageFilter(topics=selected_topics))

    rclpy.init()
    node = Node("t3_scaled_stereo_bag_player")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    left_publisher = node.create_publisher(Image, args.left_topic, qos)
    right_publisher = node.create_publisher(Image, args.right_topic, qos)
    lidar_publisher = node.create_publisher(PointCloud2, args.lidar_topic, qos)
    reference_publisher = None
    if args.reference_type == "pose_stamped":
        reference_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        reference_publisher = node.create_publisher(
            PoseStamped, args.reference_topic, reference_qos
        )

    interrupted = False

    def request_stop(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    discovery_deadline = time.monotonic() + args.discovery_timeout_sec
    while (
        rclpy.ok()
        and not interrupted
        and (
            left_publisher.get_subscription_count() == 0
            or right_publisher.get_subscription_count() == 0
            or lidar_publisher.get_subscription_count() == 0
            or (
                reference_publisher is not None
                and reference_publisher.get_subscription_count() == 0
            )
        )
        and time.monotonic() < discovery_deadline
    ):
        rclpy.spin_once(node, timeout_sec=0.1)
    print(
        "[SCALED-BAG] subscribers(left/right/lidar)="
        f"({left_publisher.get_subscription_count()}, "
        f"{right_publisher.get_subscription_count()}, "
        f"{lidar_publisher.get_subscription_count()})",
        flush=True,
    )
    if reference_publisher is not None:
        print(
            f"[SCALED-BAG] reference={args.reference_topic} "
            f"subscribers={reference_publisher.get_subscription_count()}",
            flush=True,
        )

    left_queue: deque[tuple[Image, int]] = deque()
    right_queue: deque[tuple[Image, int]] = deque()
    lidar_queue: deque[tuple[PointCloud2, int]] = deque()
    pending_references: deque[PoseStamped] = deque()
    first_raw_pair_ns: int | None = None
    first_corrected_ns: int | None = None
    start_wall: float | None = None
    published = 0
    reference_published = 0
    lidar_published = 0
    last_corrected_ns: int | None = None

    try:
        while rclpy.ok() and not interrupted and reader.has_next():
            topic, serialized, record_ns = reader.read_next()
            if topic == args.left_topic:
                message = deserialize_message(serialized, Image)
                left_queue.append((message, int(record_ns)))
            elif topic == args.right_topic:
                message = deserialize_message(serialized, Image)
                right_queue.append((message, int(record_ns)))
            elif topic == args.lidar_topic:
                message = deserialize_message(serialized, PointCloud2)
                lidar_queue.append((message, int(record_ns)))
            elif topic == args.reference_topic and reference_publisher is not None:
                reference_message = deserialize_message(serialized, PoseStamped)
                if first_corrected_ns is None or start_wall is None:
                    pending_references.append(reference_message)
                else:
                    reference_ns = stamp_ns(reference_message)
                    sleep_until(
                        node,
                        start_wall
                        + (reference_ns - first_corrected_ns) * 1e-9 / args.rate,
                    )
                    reference_publisher.publish(reference_message)
                    reference_published += 1

            while left_queue and right_queue and lidar_queue and not interrupted:
                left_message, left_record_ns = left_queue.popleft()
                right_message, right_record_ns = right_queue.popleft()
                lidar_message, _lidar_record_ns = lidar_queue.popleft()
                raw_pair_ns = int(
                    round((stamp_ns(left_message) + stamp_ns(right_message)) * 0.5)
                )
                if first_raw_pair_ns is None:
                    first_raw_pair_ns = raw_pair_ns
                    first_corrected_ns = int(
                        round((left_record_ns + right_record_ns) * 0.5)
                    )
                    start_wall = time.monotonic()
                    while pending_references:
                        reference_message = pending_references.popleft()
                        reference_publisher.publish(reference_message)
                        reference_published += 1
                assert first_corrected_ns is not None and start_wall is not None
                corrected_ns = first_corrected_ns + int(
                    round((raw_pair_ns - first_raw_pair_ns) * args.header_time_scale)
                )
                target_wall = start_wall + (
                    (corrected_ns - first_corrected_ns) * 1e-9 / args.rate
                )
                sleep_until(node, max(start_wall, target_wall - args.lidar_lead_sec))
                if interrupted or not rclpy.ok():
                    break
                set_stamp_ns(lidar_message, corrected_ns)
                lidar_publisher.publish(lidar_message)
                lidar_published += 1
                rclpy.spin_once(node, timeout_sec=0.0)
                sleep_until(node, target_wall)
                if interrupted or not rclpy.ok():
                    break
                set_stamp_ns(left_message, corrected_ns)
                set_stamp_ns(right_message, corrected_ns)
                left_publisher.publish(left_message)
                right_publisher.publish(right_message)
                rclpy.spin_once(node, timeout_sec=0.0)
                published += 1
                last_corrected_ns = corrected_ns
                if published == 1 or published % 20 == 0:
                    print(
                        f"[SCALED-BAG] published_pairs={published} "
                        f"stamp={corrected_ns // 1_000_000_000}."
                        f"{corrected_ns % 1_000_000_000:09d}",
                        flush=True,
                    )
                if args.max_pairs is not None and published >= args.max_pairs:
                    break
            if args.max_pairs is not None and published >= args.max_pairs:
                break
    finally:
        corrected_span = (
            0.0
            if first_corrected_ns is None or last_corrected_ns is None
            else (last_corrected_ns - first_corrected_ns) * 1e-9
        )
        print(
            f"[SCALED-BAG] complete published_pairs={published} "
            f"corrected_sensor_span_sec={corrected_span:.6f} "
            f"lidar_messages={lidar_published} "
            f"reference_messages={reference_published} "
            f"scale={args.header_time_scale:g} rate={args.rate:g} "
            f"lidar_lead_sec={args.lidar_lead_sec:g}",
            flush=True,
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
