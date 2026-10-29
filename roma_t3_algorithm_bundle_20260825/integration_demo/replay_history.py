#!/usr/bin/env python3
"""Read-only historical bag replay, sensor whitelist, domain 11, no car commands."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from democtl import ROOT, alive, environment, identity
from history_groups import BatchGroups, INPUTS

STATE = ROOT / "integration_demo/history_job.json"
DEMO_STATE = ROOT / "integration_demo/history_demo_state.json"
DEFAULT_BAG = Path("/home/yanfa/Env_X/InterFace/bags/20260824_235238")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def replay(args, env):
    import rclpy
    from rclpy.serialization import deserialize_message
    from rclpy.signals import SignalHandlerOptions
    from rosgraph_msgs.msg import Clock
    import rosbag2_py
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import Header
    from verify_demo import Probe
    from resource_monitor import ResourceMonitor

    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = Probe()
    done = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.__setitem__(0, True))
    publishers = {topic: node.create_publisher(kind, topic, node.live)
                  for topic, kind in zip(INPUTS, (Image, Image, PointCloud2))}
    clock = node.create_publisher(Clock, "/clock", node.live)
    state = json.loads(DEMO_STATE.read_text())
    output = Path(state["output"])
    processed = 0
    groups = BatchGroups()
    report = {}
    monitor = ResourceMonitor(state)
    monitor.start()
    wall_started = time.monotonic()
    try:
        if not node.spin_until(lambda: all(pub.get_subscription_count() for pub in publishers.values()), 20):
            raise RuntimeError("Mapper input subscriptions missing")
        if any(node.count_publishers(topic) != 1 for topic in INPUTS):
            raise RuntimeError("Another sensor publisher exists in offline domain 11")
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
                    rosbag2_py.ConverterOptions("cdr", "cdr"))
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(INPUTS)))
        if args.start_seconds:
            # SQLite's timestamp index jumps directly to the requested segment.
            # Do not deserialize/scan hours of discarded image payloads first.
            reader.seek(args.start_ns)
        while reader.has_next() and not done[0]:
            if monitor.resource_error:
                raise RuntimeError(monitor.resource_error)
            topic, data, recorded_ns = reader.read_next()
            if args.start_seconds and recorded_ns < args.start_ns:
                continue
            batch = groups.add(topic, recorded_ns, data)
            if batch is None:
                continue
            # A single stamp per sensor batch removes inter-topic transport
            # delay and the legacy 1e-6 header clock error. Original bag is untouched.
            stamp_ns = batch[INPUTS[0]][0]
            header = Header()
            header.stamp.sec, header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
            clock.publish(Clock(clock=header.stamp))
            messages = {name: deserialize_message(batch[name][1], PointCloud2 if name == INPUTS[2] else Image)
                        for name in INPUTS}
            # Refuse a suspiciously wide legacy header spread rather than
            # combining sensor packets from demonstrably different captures.
            original_stamps = [m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
                               for m in messages.values()]
            if max(original_stamps) - min(original_stamps) > 2_100:
                raise RuntimeError("Historical sensor headers disagree; batch requires manual inspection")
            for message in messages.values():
                message.header.stamp = header.stamp
            publishers[INPUTS[2]].publish(messages[INPUTS[2]])
            node.spin_until(lambda: done[0], .25)
            publishers[INPUTS[0]].publish(messages[INPUTS[0]])
            publishers[INPUTS[1]].publish(messages[INPUTS[1]])
            if not node.spin_until(lambda: done[0] or stamp_ns in node.completed, 90):
                raise RuntimeError("No mapper processing receipt")
            if done[0]:
                break
            processed += 1
            print(f"Historical batch {processed} processed", flush=True)
            save(output / "history_progress.json", {"processed_batches": processed, "input_bag": str(args.bag)})
            if args.frames and processed >= args.frames:
                break
        node.spin_until(lambda: done[0], 2)
        report = node.report("historical_rosbag_sensor_whitelist")
        report.update(input_bag=str(args.bag), processed_batches=processed, unmatched_sensor_messages=groups.dropped,
                      interrupted=done[0], timestamp_policy="batch-left-record-time; equal stereo/lidar stamp",
                      original_bag_modified=False, ros_domain_id=11,
                      wall_seconds=time.monotonic() - wall_started, resources=monitor.finish())
        save(output / "history_verification.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "timing_messages"}, indent=2), flush=True)
        return 0 if report["passed"] and not done[0] else 1
    except Exception as error:
        report = node.report("historical_rosbag_sensor_whitelist")
        report.update(passed=False, error=str(error), input_bag=str(args.bag),
                      processed_batches=processed, interrupted=done[0],
                      unmatched_sensor_messages=groups.dropped,
                      original_bag_modified=False, ros_domain_id=11,
                      wall_seconds=time.monotonic() - wall_started, resources=monitor.finish())
        save(output / "history_verification.json", report)
        raise
    finally:
        monitor.finish()
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "run", "status", "stop"))
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--frames", type=int, default=0, help="0 = complete bag")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--keep-view", action="store_true", help="Keep the final display open after saving/stopping the mapper")
    args = parser.parse_args()
    job = json.loads(STATE.read_text()) if STATE.exists() else {}
    if args.action == "status":
        print(json.dumps(job, indent=2))
        if DEMO_STATE.exists():
            state = json.loads(DEMO_STATE.read_text())
            print("Output:", state["output"])
            progress = Path(state["output"]) / "history_progress.json"
            if progress.exists():
                print(progress.read_text())
        return 0
    if args.action == "stop":
        if alive(job):
            os.kill(job["pid"], signal.SIGINT)
            print("Historical replay stopping; maps will be saved")
        return 0
    if not (args.bag / "metadata.yaml").is_file():
        raise RuntimeError("Bag metadata missing")
    if args.frames < 0 or args.start_seconds < 0:
        raise RuntimeError("Frame count and start offset must be nonnegative")
    env = environment(domain=11)
    env["T3_DEMO_STATE"] = str(DEMO_STATE)
    if args.action == "start":
        if alive(job):
            raise RuntimeError("Historical replay is already active")
        log = ROOT / "integration_demo" / time.strftime("history_%Y%m%d_%H%M%S.log")
        with log.open("ab", buffering=0) as stream:
            process = subprocess.Popen(["nice", "-n", "5", "/usr/bin/python3", str(Path(__file__).resolve()),
                                        "run", "--bag", str(args.bag), "--frames", str(args.frames),
                                        "--start-seconds", str(args.start_seconds), *(["--keep-view"] if args.keep_view else [])], env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(STATE, {"pid": process.pid, "identity": identity(process.pid), "log": str(log), "bag": str(args.bag)})
        print("Historical replay started:", process.pid, log)
        return 0
    import yaml
    metadata = yaml.safe_load((args.bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    args.start_ns = metadata["starting_time"]["nanoseconds_since_epoch"] + int(args.start_seconds * 1e9)
    controller = ["/usr/bin/python3", str(ROOT / "integration_demo/democtl.py")]
    started = False
    try:
        subprocess.run([*controller, "start", "--domain", "11", "--replay", "--dashboard-port", "8766"],
                       env=env, check=True)
        started = True
        return replay(args, env)
    finally:
        if started:
            if args.keep_view:
                import democtl
                democtl.STATE = DEMO_STATE
                state = democtl.read_state()
                for name in ('mapper', 'dashboard'):
                    democtl.stop_service(state, name)
                print('Mapper stopped and maps saved; final visual window remains open.', flush=True)
            else:
                subprocess.run([*controller, "stop"], env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
