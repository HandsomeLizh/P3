#!/usr/bin/env python3
"""Live synthetic IMU from UE truth, stamped on the sensor (LiDAR/camera) time axis.

Runs in the capture/control domain (10) next to the operator stack. Subscribes only to
/car/pose, /sensor/meta and /sensor/batch_status; publishes sensor_msgs/Imu on --topic
(recorded with the truth). Never publishes poses, offsets or truth.

--direct-domain additionally publishes each sample, reliable and with VoxelMap's frame id,
straight to /fusion/imu in the mapping domain. The relay and the Python sensor adapter take
IMU through 5-deep best-effort queues and drop ~40% of a 100 Hz stream in bursts (measured),
which VoxelMap rejects as imu_gap. The direct copy arrives first; the later relayed copy of
the same stamp is rejected by VoxelMap as non-increasing.

Each sample for sensor time s is emitted once truth covering s + offset(s) has arrived:
  latency ~ offset + edge margin + truth jitter, typically below the LiDAR arrival latency.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_imu  # noqa: E402


class SimImuLive(Node):
    def __init__(self, args, direct_publisher=None):
        super().__init__('sim_imu_from_truth')
        self.args = args
        self.direct_publisher = direct_publisher
        self.dt = 1. / args.rate
        self.noise = gt_imu.NoiseModel(args.noise, args.noise_scale, args.seed)
        self.gyro_var, self.accel_var = self.noise.covariances(self.dt)
        self.offset = gt_imu.OffsetModel(prior=args.prior_offset, window=args.offset_window)
        self.truth = deque()                 # (t_sec, position, quat) on the truth receive axis
        self.meta, self.stamps = {}, {}      # batch_id -> (position, quat) / sensor stamp
        self.pending_pairs = deque()         # (sensor_s, position, quat, first_try_wall)
        self.next_s = None
        self.counts = dict(truth=0, meta=0, meta_invalid=0, pairs=0, pair_misses=0, published=0, skipped=0)
        self.status_path = Path(args.output) / 'sim_imu_status.jsonl' if args.output else None
        if self.status_path:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PoseStamped, args.pose_topic, self.on_pose, reliable)
        self.create_subscription(String, '/sensor/meta', self.on_meta, 10)
        self.create_subscription(String, '/sensor/batch_status', self.on_status, 10)
        self.publisher = self.create_publisher(Imu, args.topic, qos_profile_sensor_data)
        self.create_timer(.02, self.emit)
        self.create_timer(1., self.report)
        self.get_logger().info(f'Synthetic IMU (simulation truth) -> {args.topic} at {args.rate:g} Hz, '
                               f'noise {args.noise} x{args.noise_scale:g}; not a physical sensor')

    def on_pose(self, m):
        s = m.header.stamp
        t = s.sec + s.nanosec * 1e-9
        if self.truth and t <= self.truth[-1][0]:
            return
        p, q = m.pose.position, m.pose.orientation
        self.truth.append((t, np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])))
        while self.truth and t - self.truth[0][0] > self.args.history:
            self.truth.popleft()
        self.counts['truth'] += 1

    def on_meta(self, m):
        parsed = gt_imu.parse_meta(m.data)
        if parsed is None:
            self.counts['meta_invalid'] += 1
            return
        self.counts['meta'] += 1
        self.meta[parsed[0]] = (parsed[2], parsed[3])
        self.pair(parsed[0])

    def on_status(self, m):
        parsed = gt_imu.parse_batch_status(m.data)
        if parsed:
            self.stamps[parsed[0]] = parsed[1] * 1e-9
            self.pair(parsed[0])

    def pair(self, batch_id):
        if batch_id in self.meta and batch_id in self.stamps:
            position, quat = self.meta.pop(batch_id)
            self.pending_pairs.append((self.stamps.pop(batch_id), position, quat, time.monotonic()))
        for table in (self.meta, self.stamps):
            for key in sorted(table)[:-20]:
                del table[key]

    def arrays(self):
        t = np.array([x[0] for x in self.truth])
        p = np.array([x[1] for x in self.truth])
        q = np.array([x[2] for x in self.truth])
        keep = gt_imu.drop_stale_repeats(t, p, q, live=True)
        return t[keep], p[keep], q[keep]

    def match_pending(self, t, p, rotations):
        while self.pending_pairs:
            s, position, quat, first = self.pending_pairs[0]
            tau = gt_imu.match_capture(t, p, rotations, position, gt_imu.base_rotation(quat), s, s + 2.5)
            if tau is not None:
                self.offset.add(s, tau - s)
                self.counts['pairs'] += 1
            elif time.monotonic() - first < 3. and t[-1] < s + 2.5:
                return                       # truth around this capture may still arrive
            else:
                self.counts['pair_misses'] += 1   # stationary capture or no match
            self.pending_pairs.popleft()

    def emit(self):
        if len(self.truth) < 4:
            return
        t, p, q = self.arrays()
        if len(t) < 4:
            return
        rotations = gt_imu.base_rotation(q)
        self.match_pending(t, p, rotations)
        newest = t[-1] - self.args.edge_margin
        if self.next_s is None:
            first = t[0] - self.offset.target(t[0]) + self.args.history_warmup
            self.next_s = np.ceil(first / self.dt) * self.dt
        budget = 200
        while budget:
            s = self.next_s
            o = self.offset.at(s)
            if s + o > newest:
                break
            budget -= 1
            self.next_s = round((s + self.dt) / self.dt) * self.dt
            state = gt_imu.kinematics(t, p, rotations, s + o)
            if state is None:
                self.counts['skipped'] += 1
                continue
            gyro, accel = self.noise.apply(*gt_imu.imu_values(state), self.dt)
            m = Imu()
            stamp_ns = int(round(s * 1e9))
            m.header.stamp.sec, m.header.stamp.nanosec = divmod(stamp_ns, 10**9)
            m.header.frame_id = self.args.frame_id
            m.orientation_covariance[0] = -1.
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyro)
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, accel)
            for i in (0, 4, 8):
                m.angular_velocity_covariance[i] = self.gyro_var
                m.linear_acceleration_covariance[i] = self.accel_var
            self.publisher.publish(m)
            if self.direct_publisher is not None:
                m.header.frame_id = self.args.direct_frame_id
                self.direct_publisher.publish(m)
            self.counts['published'] += 1

    def report(self):
        line = dict(wall=time.time(), next_stamp=self.next_s, counts=self.counts, offset=self.offset.status(),
                    latency_sec=(time.time() - self.next_s) if self.next_s else None)
        if self.status_path:
            with self.status_path.open('a') as f:
                f.write(json.dumps(line, default=float) + '\n')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--topic', default='/sim/imu_from_truth')
    a.add_argument('--frame-id', default='imu_link')
    a.add_argument('--pose-topic', default='/car/pose')
    a.add_argument('--rate', type=float, default=100.)
    a.add_argument('--noise', default='mems', choices=sorted(gt_imu.NoiseModel.PRESETS))
    a.add_argument('--noise-scale', type=float, default=1.)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--prior-offset', type=float, default=.65)
    a.add_argument('--offset-window', type=float, default=60.)
    a.add_argument('--edge-margin', type=float, default=.05, help='truth needed beyond s+offset before emitting')
    a.add_argument('--history', type=float, default=30.)
    a.add_argument('--history-warmup', type=float, default=1.)
    a.add_argument('--output', default='', help='directory for sim_imu_status.jsonl')
    a.add_argument('--direct-domain', type=int, default=None, help='also publish to this mapping domain')
    a.add_argument('--direct-topic', default='/fusion/imu')
    a.add_argument('--direct-frame-id', default='imu', help='VoxelMap imu_frame')
    args = a.parse_args(rclpy.utilities.remove_ros_args()[1:])
    rclpy.init()
    direct_context = direct_node = direct_publisher = None
    if args.direct_domain is not None:
        direct_context = rclpy.Context()
        rclpy.init(context=direct_context, domain_id=args.direct_domain)
        direct_node = rclpy.create_node('sim_imu_direct', context=direct_context)
        direct_publisher = direct_node.create_publisher(
            Imu, args.direct_topic, QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE))
    node = SimImuLive(args, direct_publisher)
    if direct_node is not None:
        node.get_logger().info(f'Also publishing to domain {args.direct_domain} {args.direct_topic} '
                               f'(frame {args.direct_frame_id})')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.report()
        node.destroy_node()
        if direct_node is not None:
            direct_node.destroy_node()
            rclpy.try_shutdown(context=direct_context)
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
