#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


TOPIC = "/Car/T3/mapping/grid_map"


class Inspector(Node):
    def __init__(self):
        super().__init__("t3_grid_map_inspector")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.received = False
        self.started = time.monotonic()
        self.create_subscription(GridMap, TOPIC, self.callback, qos)
        self.create_timer(1.0, self.timer)
        print(f"Waiting for {TOPIC} ...")

    def timer(self):
        if time.monotonic() - self.started > 60.0 and not self.received:
            print("Timed out. Start the mapper or check QoS/topic.")
            rclpy.shutdown()

    def callback(self, message: GridMap):
        if self.received:
            return
        self.received = True

        width = int(round(message.info.length_x / message.info.resolution))
        height = int(round(message.info.length_y / message.info.resolution))
        cells = width * height
        layer_bytes = cells * 4
        total_bytes = layer_bytes * len(message.layers)
        area_m2 = message.info.length_x * message.info.length_y
        density = (
            (total_bytes / 1e9) / (area_m2 / 1e6)
            if area_m2 > 0
            else float("nan")
        )

        print("=" * 70)
        print("GridMap snapshot")
        print("=" * 70)
        print("frame_id:", message.header.frame_id)
        print("timestamp:", f"{message.header.stamp.sec}.{message.header.stamp.nanosec:09d}")
        print("resolution:", message.info.resolution, "m/cell")
        print("length:", message.info.length_x, "x", message.info.length_y, "m")
        print("estimated cells:", width, "x", height, "=", cells)
        print("layers:", list(message.layers))
        print("basic_layers:", list(message.basic_layers))
        print("raw float32 layer bytes:", total_bytes)
        print("allocated-area density:", f"{density:.6f} GB/km^2")
        print(
            "Note: this excludes ROS metadata, compression, indexes, trajectory, "
            "point cloud and diagnostic previews."
        )
        print("=" * 70)
        rclpy.shutdown()


def main():
    rclpy.init()
    node = Inspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
