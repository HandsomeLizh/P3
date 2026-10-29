"""Explicit live-simulation entry point. Never runs a bag or an automatic route."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys

import democtl

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'integration_demo/run_state.json'


def commands(args):
    controller = ['/usr/bin/python3', str(ROOT/'integration_demo/democtl.py')]
    sensors = [*controller, 'sensors', '--host', args.host,
               '--capture-port', str(args.capture_port), '--car-port', str(args.car_port),
               '--capture-hz', str(args.capture_hz)]
    if args.action == 'start':
        return [[*controller, 'start', '--domain', '10'], [*sensors, '--capture-only']]
    if args.action == 'control':
        return [sensors, [*controller, 'gui']]
    return [[*controller, args.action]]


def local_checks():
    required = ['workspace/install_native/setup.bash', 'integration_demo/rviz_window/build/t3_visual_window',
                'envx_runtime/start_capture.sh', 'envx_runtime/start_car.sh', 'envx_runtime/start_gui.sh',
                'config/calibration_spreadsheets_parallel_24p9.yaml', 'config/pointcloud_extrinsic_os1_mapping_only.yaml']
    missing = [name for name in required if not (ROOT/name).is_file()]
    if missing:
        raise RuntimeError('Missing P3 runtime files: '+', '.join(missing))
    # A preparation check deliberately does not load a model, open UE sockets,
    # start ROS publishers or interfere with someone else's algorithm test.
    print('P3 runtime files present. UE sockets and model execution were not tested.', flush=True)


def guard_history():
    for name in ('history_job.json', 'history_demo_state.json'):
        file = ROOT/'integration_demo'/name
        state = json.loads(file.read_text()) if file.exists() else {}
        active = democtl.alive(state) if name == 'history_job.json' else democtl.alive(state.get('services', {}).get('mapper'))
        if active:
            raise RuntimeError('Historical mapper is still running. Stop it before the live workflow.')


def guard_vehicle_publishers(env):
    # A stale external GUI may continuously publish a nonzero speed. Do not
    # connect the vehicle bridge until the operator closes such publishers.
    probe = """
import time, rclpy
rclpy.init()
n=rclpy.create_node('t3_live_control_preflight')
end=time.monotonic()+3
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
publishers=n.get_publishers_info_by_topic('/car/cmd_vel')
names=[p.node_namespace+'/'+p.node_name for p in publishers]
n.destroy_node();rclpy.shutdown()
if names:raise SystemExit('Vehicle-control publishers already exist. Zero and close the old GUI/controller first: '+', '.join(names))
"""
    subprocess.run(['/usr/bin/python3', '-c', probe], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'control', 'status', 'stop', 'check'))
    parser.add_argument('--host', default='192.168.10.22')
    parser.add_argument('--capture-port', type=int, default=6665)
    parser.add_argument('--car-port', type=int, default=6668)
    parser.add_argument('--capture-hz', type=float, default=.25)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ipaddress.ip_address(args.host)
    if not 0 < args.capture_hz <= .25 or not all(1 <= port <= 65535 for port in (args.capture_port, args.car_port)):
        parser.error('Use valid ports and a capture rate in (0, 0.25] Hz for this tested profile')
    if args.action == 'check':
        local_checks()
        print(f'Prepared UE endpoint: {args.host}, acquisition {args.capture_port}, control {args.car_port}')
        return
    plan = commands(args)
    if args.dry_run:
        print(json.dumps({'commands': plan, 'runs_bag': False,
                          'connects_vehicle_control': args.action == 'control',
                          'starts_automatic_motion': False}, indent=2))
        return
    env = dict(os.environ, T3_DEMO_STATE=str(STATE), ROS_DOMAIN_ID='10', PYTHONDONTWRITEBYTECODE='1')
    if args.action in ('start', 'control'):
        local_checks()
        guard_history()
    if args.action == 'control':
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        if not democtl.alive(state.get('services', {}).get('mapper')):
            raise RuntimeError('Start live mapping first')
        if democtl.alive(state.get('services', {}).get('gui')):
            raise RuntimeError('Managed vehicle GUI is already open; do not start another controller')
        guard_vehicle_publishers(democtl.environment(state.get('output'), 10))
    for command in plan:
        subprocess.run(command, env=env, check=True)
    if args.action == 'start':
        print('Live mapping ready. Acquisition only: no vehicle bridge, automatic motion or bag recording was started.')
        print('When ready to drive, zero and close any old vehicle GUI, then run start_live_control.sh.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
