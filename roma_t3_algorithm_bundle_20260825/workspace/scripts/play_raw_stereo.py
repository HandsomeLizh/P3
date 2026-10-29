#!/usr/bin/env python3
"""Publish timestamp-paired raw stereo images with optional lossless ACKs."""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header


_STAMP_PATTERN = re.compile(r"_(\d{8})_(\d{6})_(\d{3})\.[^.]+$")
_CST = timezone(timedelta(hours=8))


def _timestamp_ns(path: Path) -> int:
    match = _STAMP_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Image filename has no timestamp: {path.name}")
    value = datetime.strptime(
        "".join(match.groups()), "%Y%m%d%H%M%S%f"
    ).replace(tzinfo=_CST)
    return int(round(value.timestamp() * 1e9))


def _images(directory: Path) -> list[tuple[int, Path]]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    values = [
        (_timestamp_ns(path), path)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(values)


def _pair(
    left: list[tuple[int, Path]],
    right: list[tuple[int, Path]],
    maximum_skew_ns: int,
) -> list[tuple[tuple[int, Path], tuple[int, Path]]]:
    pairs = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        delta = left[left_index][0] - right[right_index][0]
        if abs(delta) <= maximum_skew_ns:
            pairs.append((left[left_index], right[right_index]))
            left_index += 1
            right_index += 1
        elif delta < 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


class RawStereoPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("raw_stereo_publisher")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        ack_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.left_pub = self.create_publisher(Image, args.left_topic, qos)
        self.right_pub = self.create_publisher(Image, args.right_topic, qos)
        self.completed_stamps: set[int] = set()
        self.ack_sub = self.create_subscription(
            Header, args.ack_topic, self._on_ack, ack_qos
        )
        self.bridge = CvBridge()

    def _on_ack(self, message: Header) -> None:
        self.completed_stamps.add(
            int(message.stamp.sec) * 1_000_000_000
            + int(message.stamp.nanosec)
        )

    def wait_for_consumers(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if (
                self.left_pub.get_subscription_count() > 0
                and self.right_pub.get_subscription_count() > 0
            ):
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError("Timed out waiting for both stereo subscribers")

    def publish_pair(
        self,
        left: tuple[int, Path],
        right: tuple[int, Path],
        ack_timeout_sec: float,
    ) -> float:
        left_stamp, left_path = left
        right_stamp, right_path = right
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            raise RuntimeError(f"Could not read {left_path} or {right_path}")
        left_message = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
        right_message = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")
        left_message.header.frame_id = "galaxy_Cam_Left_frame"
        right_message.header.frame_id = "galaxy_Cam_Right_frame"
        left_message.header.stamp.sec = left_stamp // 1_000_000_000
        left_message.header.stamp.nanosec = left_stamp % 1_000_000_000
        right_message.header.stamp.sec = right_stamp // 1_000_000_000
        right_message.header.stamp.nanosec = right_stamp % 1_000_000_000
        started = time.perf_counter()
        self.left_pub.publish(left_message)
        self.right_pub.publish(right_message)
        deadline = time.monotonic() + ack_timeout_sec
        while left_stamp not in self.completed_stamps:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No processing ACK for {left_path.name} within "
                    f"{ack_timeout_sec:.1f}s"
                )
            rclpy.spin_once(self, timeout_sec=0.1)
        self.completed_stamps.discard(left_stamp)
        return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--start", default="")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--max-skew-ms", type=float, default=30.0)
    parser.add_argument("--ack-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--left-topic", default="/Car/T5/Cam_Left/image_raw/color"
    )
    parser.add_argument(
        "--right-topic", default="/Car/T5/Cam_Right/image_raw/color"
    )
    parser.add_argument(
        "--ack-topic", default="/Car/T3/debug/input_processed"
    )
    args = parser.parse_args()
    if args.max_pairs <= 0 or args.max_skew_ms <= 0.0:
        raise ValueError("max-pairs and max-skew-ms must be positive")

    sequence = args.sequence.expanduser().resolve()
    left = _images(sequence / "Cam_Left")
    right = _images(sequence / "Cam_Right")
    if args.start:
        start_value = datetime.strptime(args.start, "%Y%m%d_%H%M%S").replace(
            tzinfo=_CST
        )
        start_ns = int(round(start_value.timestamp() * 1e9))
        left = [value for value in left if value[0] >= start_ns]
        right = [value for value in right if value[0] >= start_ns]
    pairs = _pair(left, right, int(round(args.max_skew_ms * 1e6)))
    pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("No stereo pairs satisfy the requested time gate")

    rclpy.init()
    node = RawStereoPublisher(args)
    try:
        node.wait_for_consumers(60.0)
        print(f"paired={len(pairs)} lossless=true", flush=True)
        for index, (left_item, right_item) in enumerate(pairs, start=1):
            elapsed = node.publish_pair(
                left_item, right_item, args.ack_timeout_sec
            )
            skew_ms = abs(left_item[0] - right_item[0]) / 1e6
            print(
                f"pair={index}/{len(pairs)} skew_ms={skew_ms:.3f} "
                f"ack_sec={elapsed:.3f} left={left_item[1].name} "
                f"right={right_item[1].name}",
                flush=True,
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
