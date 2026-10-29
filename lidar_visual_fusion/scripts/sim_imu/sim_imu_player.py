#!/usr/bin/env python3
"""Bag-replay companion: publish a prebuilt synthetic IMU (npz) against the replay /clock.

Run with use_sim_time. Each sample is published once sim time reaches its stamp minus
--lead, so it is buffered before the LiDAR scan that needs it while staying inside
VoxelMap's future tolerance (0.1 s).

Stamp mapping:
  ros2 bag play of a live_input_bag: original headers, use --shift-ns 0 (default).
  scripts/replay_bag.py:             headers are re-based; pass --timeline <replay_frames.csv>
                                     and the shift is read from its first row.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def timeline_shift(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            rows = list(csv.reader(Path(path).open()))
            if len(rows) >= 2:
                raw_ns = int(rows[1][1])
                sensor_ns = int(round(float(rows[1][2]) * 1e9))
                return sensor_ns - raw_ns
        time.sleep(.05)
    raise SystemExit('replay timeline did not appear: ' + str(path))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('npz')
    a.add_argument('--topic', default='/sim/imu_from_truth')
    a.add_argument('--frame-id', default='imu_link')
    a.add_argument('--lead', type=float, default=.05)
    a.add_argument('--shift-ns', type=int, default=0)
    a.add_argument('--timeline', default='')
    a.add_argument('--timeline-timeout', type=float, default=120.)
    a.add_argument('--clean', action='store_true', help='publish the noise-free channels (ablation)')
    args = a.parse_args()
    if not 0. <= args.lead < .1:
        a.error('--lead must stay below the 0.1 s VoxelMap future tolerance')
    data = np.load(args.npz)
    stamps = data['stamp_ns'].astype(np.int64)
    gyro = data['gyro_clean' if args.clean else 'gyro']
    accel = data['accel_clean' if args.clean else 'accel']
    gyro_cov, accel_cov = data['gyro_cov'], data['accel_cov']
    rclpy.init()
    node = Node('sim_imu_player', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    publisher = node.create_publisher(Imu, args.topic, qos_profile_sensor_data)
    shift = timeline_shift(args.timeline, args.timeline_timeout) if args.timeline else args.shift_ns
    stamps = stamps + shift
    lead_ns = int(args.lead * 1e9)
    index = [0]
    node.get_logger().info(f'Synthetic IMU player: {len(stamps)} samples, shift {shift} ns, topic {args.topic}')

    def publish_due():
        now = node.get_clock().now().nanoseconds
        if now <= 0:
            return
        if index[0] == 0:
            # Do not flood samples far older than the replay start.
            index[0] = int(np.searchsorted(stamps, now - 2 * 10**9))
        while index[0] < len(stamps) and stamps[index[0]] <= now + lead_ns:
            k = index[0]
            m = Imu()
            m.header.stamp.sec, m.header.stamp.nanosec = divmod(int(stamps[k]), 10**9)
            m.header.frame_id = args.frame_id
            m.orientation_covariance[0] = -1.
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyro[k])
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, accel[k])
            for i in (0, 4, 8):
                m.angular_velocity_covariance[i] = float(gyro_cov[k])
                m.linear_acceleration_covariance[i] = float(accel_cov[k])
            publisher.publish(m)
            index[0] += 1

    node.create_timer(.005, publish_due)
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
