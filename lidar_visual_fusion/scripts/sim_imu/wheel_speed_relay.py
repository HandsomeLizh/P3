#!/usr/bin/env python3
"""Relay simulator wheel speeds as an upper bound on body speed (live).

Subscribes /car/telemetry (lunar_car_ctrl/VehicleTelemetry) in the capture/control domain and
publishes geometry_msgs/TwistStamped on --topic in the mapping domain (--target-domain):
  twist.linear.x = wheel_radius * max(|wheel rad/s| of the four wheels)
This is an upper bound on body speed: wheel slip only spins wheels faster than the body
moves. Stamps keep the telemetry receive time; the consumer gates with a +-window that
absorbs the ~0.5-0.8 s latency of the truth stream. Never publishes poses or the truth.

Wheel radius 0.1281 m was fitted against simulation truth (bag 20260918_142301, 2026-10-29,
residual median 3 mm/s). The exact value barely matters for gating: it scales an upper
bound that already includes a slip factor on the consumer side.
"""
import argparse

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from lunar_car_ctrl.msg import VehicleTelemetry


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--telemetry-topic', default='/car/telemetry')
    a.add_argument('--topic', default='/fusion/wheel_speed')
    a.add_argument('--radius', type=float, default=.1281, help='effective wheel radius, m')
    a.add_argument('--target-domain', type=int, default=57)
    args = a.parse_args(rclpy.utilities.remove_ros_args()[1:])
    if not np.isfinite(args.radius) or args.radius <= 0:
        raise SystemExit('--radius must be finite and positive')
    rclpy.init()
    target_context = rclpy.Context()
    rclpy.init(context=target_context, domain_id=args.target_domain)
    target = rclpy.create_node('wheel_speed_target', context=target_context)
    publisher = target.create_publisher(TwistStamped, args.topic,
                                        QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
    node = Node('wheel_speed_relay')
    counts = dict(received=0, forwarded=0)

    def on_telemetry(m):
        counts['received'] += 1
        wheels = [m.wheel_speed_fl, m.wheel_speed_fr, m.wheel_speed_rl, m.wheel_speed_rr]
        if not np.isfinite(wheels).all():
            return
        out = TwistStamped()
        out.header = m.header
        out.header.frame_id = 'base_link'
        out.twist.linear.x = args.radius * float(np.abs(wheels).max())
        publisher.publish(out)
        counts['forwarded'] += 1

    node.create_subscription(VehicleTelemetry, args.telemetry_topic, on_telemetry,
                             QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
    node.get_logger().info(f'Wheel speed upper bound -> domain {args.target_domain} {args.topic} '
                           f'(radius {args.radius:g} m)')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f"wheel relay: {counts}")
        node.destroy_node()
        target.destroy_node()
        rclpy.try_shutdown(context=target_context)
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
