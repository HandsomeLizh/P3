"""Inspect a finished P3 map without rerunning the algorithm or publishing controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import democtl


def publish(output):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Header, String
    from nav_msgs.msg import Path as RosPath
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from sensor_msgs.msg import PointCloud2
    from grid_map_msgs.msg import GridMap
    from t3_semantic_mapping.grid_map_message import make_grid_map_message
    from t3_semantic_mapping.semantic_grid import GridGeometry
    from t3_semantic_mapping.pointcloud_utils import make_xyz_point_cloud

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('t3_saved_map_view')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publishers = []
    messages = []
    def add(kind, topic, message):
        publishers.append(node.create_publisher(kind, topic, qos))
        messages.append(message)

    trajectory = np.loadtxt(output / 'trajectory_map.tum').reshape(-1, 8)
    header = Header(frame_id='map')
    header.stamp.sec = int(trajectory[-1, 0])
    header.stamp.nanosec = int((trajectory[-1, 0]-header.stamp.sec)*1e9)
    path = RosPath(header=header)
    for row in trajectory:
        pose = PoseStamped(header=header)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, row[1:4])
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, row[4:8])
        path.poses.append(pose)
    add(RosPath, '/T3/semantic/trajectory', path)
    # A static snapshot frame lets RViz's vehicle-view camera operate on the last
    # saved pose. It stays in isolated domain 12 and is never a live estimate.
    broadcaster = StaticTransformBroadcaster(node)
    transform = TransformStamped(header=header, child_frame_id='base_link')
    transform.transform.translation.x = path.poses[-1].pose.position.x
    transform.transform.translation.y = path.poses[-1].pose.position.y
    transform.transform.translation.z = path.poses[-1].pose.position.z
    transform.transform.rotation = path.poses[-1].pose.orientation
    broadcaster.sendTransform(transform)
    with np.load(output / 'global_grid_map.npz', allow_pickle=False) as archive:
        elevation = archive['elevation']
        geometry = GridGeometry(float(archive['resolution']), elevation.shape[1], elevation.shape[0],
                                float(archive['origin_x']), float(archive['origin_y']))
        grid = make_grid_map_message(header=header, geometry=geometry, layers={'elevation': elevation})
    add(GridMap, '/T3/mapping/global_grid_map', grid)
    with (output / 'lidar_map.pcd').open('rb') as stream:
        count = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Invalid PCD header')
            if line.startswith(b'POINTS '):
                count = int(line.split()[1])
            if line == b'DATA binary\n':
                offset = stream.tell()
                break
    cloud = np.memmap(output / 'lidar_map.pcd', mode='r', dtype='<f4', offset=offset, shape=(count, 3))
    sample = cloud[::max(1, int(np.ceil(count/200000)))]
    add(PointCloud2, '/T3/mapping/lidar_map', make_xyz_point_cloud(header=header, points_xyz=sample))
    add(String, '/T3/mapping/lidar_status', String(data=json.dumps({'voxel_count': count})))
    with (output / 'pose_processing_time.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    add(String, '/Car/T3/metrics/frame_timing', String(data=json.dumps({
        'pose_processing_sec': float(rows[-1]['processing_time_sec']), 'outcome': 'completed'})))
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    deadline = 0.
    try:
        while not stopping:
            if time.monotonic() >= deadline:
                for publisher, message in zip(publishers, messages):
                    publisher.publish(message)
                deadline = time.monotonic()+10
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'publish', 'stop'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    democtl.STATE = democtl.ROOT / 'integration_demo/preview_state.json'
    state = democtl.read_state()
    if args.action == 'stop':
        for name in ('rviz', 'visuals', 'preview'):
            democtl.stop_service(state, name)
        return
    output = args.output.resolve()
    if not output.is_relative_to(Path('/home/yanfa/P3')):
        raise ValueError('Only inspect saved P3 outputs')
    if args.action == 'publish':
        publish(output)
        return
    if any(democtl.alive(service) for service in state['services'].values()):
        raise RuntimeError('Saved map view already running; stop it first')
    state = {'output': str(output), 'domain': 12, 'replay': False, 'services': {}}
    democtl.save_state(state)
    env = democtl.environment(output, domain=12)
    democtl.start_service(state, 'preview', ['/usr/bin/python3', __file__, 'publish', '--output', str(output)], env)
    democtl.start_rviz(state, env)
    print('Saved-map inspection only, isolated domain 12; no odometry or vehicle command publisher.')


if __name__ == '__main__':
    main()
