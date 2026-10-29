#!/usr/bin/env python3
"""Background UE motion + selected-topic rosbag, with an explicit GUI lease."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from democtl import ROOT, alive, environment, identity, read_state
from motion_plan import describe, route

STATE_PATH = ROOT / "integration_demo/motion_state.json"
TOPICS = ["/Car/T5/Cam_Left/image_raw/color", "/Car/T5/Cam_Right/image_raw/color",
          "/Car/T5/OS1/points", "/car/pose", "/car/odom",
          "/T3/semantic/current_pose", "/T3/semantic/trajectory",
          "/T3/semantic/incremental_map", "/Car/T3/metrics/frame_timing",
          "/T3/mapping/lidar_status", "/tf_static"]


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def worker(args):
    import fcntl
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, String

    case = args.case.resolve()
    plan = describe(args.profile)
    result = dict(plan, status="preflight", bag_path=str(case / "bag"), topic_whitelist=TOPICS,
                  actual_reference_travel_m=0., reference_max_radius_m=0.)
    status_file = case / "status.json"
    save(status_file, result)
    lock = (ROOT / "integration_demo/motion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read_state()
    for service in ("mapper", "capture", "car", "gui"):
        if not alive(state["services"].get(service)):
            raise RuntimeError(f"{service} must be running before a motion recording")
    if state.get("verification_mode") or state.get("ue_host") != "192.168.10.22":
        raise RuntimeError("This controller is restricted to the configured live UE demo")
    if state["domain"] != 10:
        raise RuntimeError("Motion is restricted to the live ROS domain 10")
    required = 10 * 1024**3 + (plan["duration_seconds"] + 45) * 30_000_000
    result["initial_free_bytes"] = shutil.disk_usage(case).free
    result["conservative_required_free_bytes"] = int(required)
    if result["initial_free_bytes"] < required:
        raise RuntimeError("Insufficient free space, including the 10 GiB reserve")
    # Verify the GUI installed on this machine implements the cooperative lease.
    check = subprocess.run(["ros2", "param", "get", "/lunar_car_gui_node",
                            "external_control_lease_supported"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    if check.returncode or "True" not in check.stdout:
        raise RuntimeError("GUI lease support missing; refusing competing cmd_vel publishers")
    for filename in ("calibration_spreadsheets_parallel_24p9.yaml", "pointcloud_extrinsic_os1_mapping_only.yaml"):
        shutil.copy2(ROOT / "config" / filename, case / filename)
    save(case / "demo_state_snapshot.json", state)
    for name in ("roma_semantic_mapper_node", "sensor_capture_node", "lunar_car_node"):
        with (case / (name + "_parameters.yaml")).open("w") as output:
            subprocess.run(["ros2", "param", "dump", "/" + name], stdout=output,
                           stderr=subprocess.STDOUT, timeout=20, check=True)

    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("t3_fixed_motion")
    command = node.create_publisher(Twist, "/car/cmd_vel", 10)
    mode = node.create_publisher(String, "/car/set_mode", 10)
    lease = node.create_publisher(Bool, "/car/external_control_lease", 10)
    stop_requested = [False]
    expected_mode = ["park"]
    latest = {"reference_at": 0., "visual_at": 0., "reference": None, "origin": None}
    history = []

    def on_reference(message):
        now = time.monotonic()
        p = message.pose.pose.position
        position = (p.x, p.y, p.z)
        if not all(math.isfinite(x) for x in position):
            stop_requested[0] = True
            return
        previous = latest["reference"]
        if previous is not None:
            step = math.dist(position, previous)
            if step > 2:
                stop_requested[0] = True
            result["actual_reference_travel_m"] += step
        latest.update(reference_at=now, reference=position)
        if latest["origin"] is None:
            latest["origin"] = position
        radius = math.dist(position, latest["origin"])
        result["reference_max_radius_m"] = max(result["reference_max_radius_m"], radius)
        if radius > 15:
            stop_requested[0] = True
        history.append((now, position))
        while history and history[0][0] < now - 12:
            history.pop(0)

    def on_mode(message):
        if expected_mode[0] == "auto" and message.data == "park":
            stop_requested[0] = True  # GUI stop / emergency stop always wins.

    node.create_subscription(Odometry, "/car/odom", on_reference, 10)
    node.create_subscription(Odometry, "/T3/semantic/current_pose",
                             lambda _: latest.update(visual_at=time.monotonic()), 10)
    node.create_subscription(String, "/car/set_mode", on_mode, 10)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))
    recorder = None
    recorder_log = None
    start = time.monotonic()
    last_report = 0.

    def tick(speed=0., curvature=0., guard=False):
        nonlocal last_report
        if stop_requested[0]:
            raise RuntimeError("Stop requested, reference jump, or 15 m operating boundary reached")
        lease.publish(Bool(data=True))
        twist = Twist()
        twist.linear.x, twist.angular.z = speed, speed * curvature
        command.publish(twist)
        until = time.monotonic() + .1
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.02)
        now = time.monotonic()
        if guard:
            if now - latest["reference_at"] > 2 or now - latest["visual_at"] > 20:
                raise RuntimeError("Telemetry or visual odometry timed out")
            if len(history) > 2 and now - history[0][0] > 10 and abs(speed) > .01:
                if math.dist(history[-1][1], history[0][1]) < .025:
                    raise RuntimeError("Vehicle appears stuck; stopped without forcing past obstacle")
        if recorder and recorder.poll() is not None:
            raise RuntimeError("rosbag recorder exited")
        if now - last_report > 2:
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    memory[key] = int(value.split()[0]) * 1024
            result["system_memory_bytes"] = memory
            if memory.get("MemAvailable", 0) < 8 * 1024**3:
                raise RuntimeError("System available memory below 8 GiB reserve")
            if shutil.disk_usage(case).free < 10 * 1024**3:
                raise RuntimeError("Disk reserve reached")
            unexpected = [info.node_name for info in node.get_publishers_info_by_topic("/car/cmd_vel")
                          if info.node_name not in ("t3_fixed_motion", "lunar_car_gui_node")]
            if unexpected:
                raise RuntimeError("Competing controller: " + ", ".join(unexpected))
            for service in ("mapper", "capture", "car"):
                if not alive(state["services"].get(service)):
                    raise RuntimeError(service + " stopped")
            result["elapsed_seconds"] = now - start
            with (case / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps({"elapsed_seconds": now-start,
                                         "free_disk_bytes": shutil.disk_usage(case).free,
                                         "system_memory_bytes": memory}) + "\n")
            save(status_file, result)
            last_report = now

    try:
        # Establish the GUI lease before issuing any automatic mode request.
        for _ in range(20):
            mode.publish(String(data="park"))
            tick()
        recorder_log = (case / "record.log").open("ab", buffering=0)
        recorder = subprocess.Popen(["nice", "-n", "10", "ionice", "-c", "2", "-n", "7",
                                     "ros2", "bag", "record", "-s", "sqlite3", "-o", str(case / "bag"),
                                     "--max-bag-size", str(2 * 1024**3), "--max-cache-size", str(128 * 1024**2),
                                     *TOPICS], stdout=recorder_log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        result.update(status="recording_warmup", recorder_pid=recorder.pid)
        # Give the recorder time to discover inputs and allow static map warmup.
        for _ in range(150):
            tick()
        if not latest["reference_at"] or not latest["visual_at"]:
            raise RuntimeError("No live telemetry or visual pose; motion cancelled")
        print("Recording ready. Motion begins in 5 seconds; GUI emergency stop remains active.", flush=True)
        for _ in range(50):
            tick()
        result["motion_started_at"] = time.time()
        result["status"] = "moving"
        history.clear()  # The stationary warmup must not trigger the motion stall guard.
        latest["origin"] = latest["reference"]
        result["actual_reference_travel_m"] = 0.0
        result["reference_max_radius_m"] = 0.0
        for segment in route(args.profile):
            result["segment"] = segment.name
            print(f"{segment.name}: {segment.seconds:.1f}s, v={segment.speed}, curvature={segment.curvature}", flush=True)
            expected_mode[0] = "auto"
            mode.publish(String(data="auto"))
            deadline = time.monotonic() + segment.seconds
            while time.monotonic() < deadline:
                tick(segment.speed, segment.curvature, guard=True)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="stopped", reason=str(error))
        print("Stopped:", error, flush=True)
    finally:
        expected_mode[0] = "park"
        for _ in range(20):
            lease.publish(Bool(data=True))
            command.publish(Twist())
            mode.publish(String(data="park"))
            rclpy.spin_once(node, timeout_sec=.05)
            time.sleep(.05)
        lease.publish(Bool(data=False))
        time.sleep(.2)
        node.destroy_node()
        rclpy.shutdown()
        if recorder and recorder.poll() is None:
            os.killpg(recorder.pid, signal.SIGINT)
            try:
                recorder.wait(timeout=45)
            except subprocess.TimeoutExpired:
                result["recorder_warning"] = "Recorder still flushing; inspect PID before another recording"
        if recorder_log:
            recorder_log.close()
        result["finished_at"] = time.time()
        result["bag_bytes"] = sum(p.stat().st_size for p in (case / "bag").glob("*") if p.is_file())
        result["bag_metadata_present"] = (case / "bag/metadata.yaml").is_file()
        save(status_file, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        lock.close()
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "run", "status", "stop", "plan"))
    parser.add_argument("--profile", choices=("short", "long"), default="short")
    parser.add_argument("--case", type=Path)
    args = parser.parse_args()
    previous = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    if args.action == "plan":
        print(json.dumps(describe(args.profile), indent=2))
    elif args.action == "status":
        print(json.dumps(previous, indent=2))
        if previous.get("case"):
            path = Path(previous["case"]) / "status.json"
            print(path.read_text() if path.exists() else "Starting")
    elif args.action == "stop":
        if alive(previous):
            os.kill(previous["pid"], signal.SIGINT)
            print("Stop requested; vehicle parking and recorder flushing")
        else:
            print("No active motion controller")
    elif args.action == "start":
        if alive(previous):
            raise RuntimeError("A motion recording is already active")
        case = ROOT / "recordings" / (args.profile + "_" + time.strftime("%Y%m%d_%H%M%S"))
        case.mkdir(parents=True, exist_ok=False)
        state = read_state()
        with (case / "motion.log").open("ab", buffering=0) as output:
            process = subprocess.Popen(["/usr/bin/python3", str(Path(__file__).resolve()), "run",
                                        "--profile", args.profile, "--case", str(case)],
                                       env=environment(state.get("output"), state.get("domain", 10)),
                                       stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        save(STATE_PATH, {"pid": process.pid, "identity": identity(process.pid), "case": str(case)})
        print("Background controller:", process.pid, "Log:", case / "motion.log")
    else:
        if args.case is None:
            parser.error("run requires --case")
        try:
            return worker(args)
        except Exception as error:
            save(args.case / "status.json", {"status": "preflight_failed", "reason": str(error)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
