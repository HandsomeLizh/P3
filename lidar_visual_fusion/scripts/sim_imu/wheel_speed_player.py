#!/usr/bin/env python3
"""Bag-replay companion: publish wheel-speed upper bounds from recorded /car/telemetry.

Run with use_sim_time next to a replay. Reads telemetry straight from the bag's sqlite files
(no index needed) and publishes geometry_msgs/TwistStamped on --topic once sim time reaches
each stamp minus --lead.

Stamp mapping: telemetry stamps are receive times on the same clock as the raw sensor headers.
  scripts/replay_bag.py re-bases headers; pass --timeline <run>/replay_frames.csv.
  ros2 bag play keeps original stamps; use the default --shift-ns 0.
--receive-offset subtracts the truth-stream latency so the stamps land near the sensor time
axis; the consumer's window makes the exact value uncritical.
"""
import argparse
import glob
import sqlite3
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import TwistStamped
from lunar_car_ctrl.msg import VehicleTelemetry

from sim_imu_player import timeline_shift


def load(bag, radius):
    rows = []
    for name in sorted(glob.glob(str(Path(bag) / '*.db3'))):
        con = sqlite3.connect(f'file:{Path(name).resolve()}?mode=ro', uri=True)
        found = con.execute("SELECT id FROM topics WHERE name='/car/telemetry'").fetchone()
        if found:
            for (data,) in con.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', found):
                m = deserialize_message(data, VehicleTelemetry)
                wheels = [m.wheel_speed_fl, m.wheel_speed_fr, m.wheel_speed_rl, m.wheel_speed_rr]
                if np.isfinite(wheels).all():
                    rows.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
                                 radius * float(np.abs(wheels).max())))
    rows.sort()
    return rows


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('bag', help='bag directory containing /car/telemetry')
    a.add_argument('--topic', default='/fusion/wheel_speed')
    a.add_argument('--radius', type=float, default=.1281)
    a.add_argument('--lead', type=float, default=.05)
    a.add_argument('--shift-ns', type=int, default=0)
    a.add_argument('--timeline', default='')
    a.add_argument('--timeline-timeout', type=float, default=120.)
    a.add_argument('--receive-offset', type=float, default=.6, help='seconds subtracted from receive stamps')
    args = a.parse_args(rclpy.utilities.remove_ros_args()[1:])
    rows = load(args.bag, args.radius)
    if not rows:
        raise SystemExit('bag has no usable /car/telemetry: ' + args.bag)
    rclpy.init()
    node = Node('wheel_speed_player', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    publisher = node.create_publisher(TwistStamped, args.topic,
                                      QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
    shift = (timeline_shift(args.timeline, args.timeline_timeout) if args.timeline else args.shift_ns)
    shift -= int(args.receive_offset * 1e9)
    stamps = np.array([r[0] for r in rows], dtype=np.int64) + shift
    speeds = [r[1] for r in rows]
    lead_ns = int(args.lead * 1e9)
    index = [0]
    node.get_logger().info(f'Wheel speed player: {len(stamps)} samples, shift {shift} ns, topic {args.topic}')

    def publish_due():
        now = node.get_clock().now().nanoseconds
        if now <= 0:
            return
        if index[0] == 0:
            index[0] = int(np.searchsorted(stamps, now - 2 * 10**9))
        while index[0] < len(stamps) and stamps[index[0]] <= now + lead_ns:
            k = index[0]
            m = TwistStamped()
            m.header.stamp.sec, m.header.stamp.nanosec = divmod(int(stamps[k]), 10**9)
            m.header.frame_id = 'base_link'
            m.twist.linear.x = speeds[k]
            publisher.publish(m)
            index[0] += 1

    node.create_timer(.01, publish_due)
    try:
        while rclpy.ok() and index[0] < len(stamps):
            rclpy.spin_once(node, timeout_sec=.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f'published {index[0]} / {len(stamps)} samples')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
