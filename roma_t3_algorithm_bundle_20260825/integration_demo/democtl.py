#!/usr/bin/env python3
"""Start/status/stop the native demo and the existing Env_X programs."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("T3_DEMO_STATE", str(ROOT / "integration_demo/run_state.json")))
ENVX = Path(os.environ.get("T3_ENVX", str(ROOT / "envx_runtime")))


def environment(output=None, domain=None):
    setup = ROOT / "workspace/install_native/setup.bash"
    command = f"source /opt/ros/humble/setup.bash; source {shlex.quote(str(setup))}; env -0"
    result = subprocess.run(["bash", "-c", command], check=True, stdout=subprocess.PIPE)
    env = dict(entry.split("=", 1) for entry in result.stdout.decode().split("\0") if "=" in entry)
    env.update({"ROS_DOMAIN_ID": str(domain or os.environ.get("ROS_DOMAIN_ID", "10")),
                "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "ROS_LOCALHOST_ONLY": "0",
                "T3_BUNDLE_ROOT": str(ROOT), "TORCH_HOME": str(ROOT / "models/torch"),
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "ROS_LOG_DIR": str(ROOT / "integration_demo/logs/ros"),
                "XDG_CACHE_HOME": str(ROOT / ".cache"),
                "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib")})
    env["PYTHONPATH"] = str(ROOT / "third_party/RoMa") + ":" + env.get("PYTHONPATH", "")
    if output is not None:
        env["T3_OUTPUT_DIR"] = str(output)
        env["ROS_LOG_DIR"] = str(Path(output) / "logs/ros")
    env["CUDA_CACHE_PATH"] = str(ROOT / ".cache/cuda")
    env["TMPDIR"] = str(ROOT / "integration_demo/tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    plugin_prefix = ROOT / "integration_demo/rviz_plugins/opt/ros/humble"
    if plugin_prefix.exists():
        env["AMENT_PREFIX_PATH"] = str(plugin_prefix) + ":" + env.get("AMENT_PREFIX_PATH", "")
        env["LD_LIBRARY_PATH"] = str(plugin_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def display_environment(env):
    env = dict(env)
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
    authority = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
    env["XAUTHORITY"] = os.environ.get("XAUTHORITY", str(
        authority if authority.exists() else Path.home() / ".Xauthority"))
    env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    return env


def start_rviz(state, env):
    visual_env = display_environment(env)
    visual_env['T3_VISUAL_RUNTIME'] = str(Path(state['output']) / 'visual_runtime')
    if not alive(state["services"].get("visuals")):
        start_service(state, "visuals", ["/usr/bin/python3", str(ROOT / "integration_demo/visual_monitor.py")],
                      visual_env)
    if alive(state["services"].get("rviz")):
        print("Unified localization/map window is already open on the Orin display", flush=True)
        return
    viewer = ROOT / 'integration_demo/rviz_window/build/t3_visual_window'
    if not viewer.is_file():
        raise RuntimeError('Build integration_demo/rviz_window inside P3 before starting the visual window')
    start_service(state, "rviz", [str(viewer), str(ROOT / "integration_demo/t3_envx_demo.rviz"),
                                  visual_env['T3_VISUAL_RUNTIME'],
                                  "--ros-args", "-p", f"use_sim_time:={str(state.get('replay', False)).lower()}"],
                  visual_env)


def read_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"services": {}}


def save_state(state):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


def identity(pid):
    try:
        # Start time distinguishes a recycled PID from a process we started.
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return fields[19] if fields[0] != "Z" else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(service):
    return bool(service and service.get("identity")
                and identity(service["pid"]) == service["identity"])


def start_service(state, name, command, env):
    if alive(state["services"].get(name)):
        raise RuntimeError(f"{name} is already running")
    log = Path(state["output"]) / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    state["services"][name] = {
        "pid": process.pid, "identity": identity(process.pid),
        "command": command, "log": str(log),
    }
    save_state(state)
    print(f"Started {name}: PID {process.pid}; {log}", flush=True)


def wait_ready(state, name, text, timeout=150):
    service = state["services"][name]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        log = Path(service["log"]).read_text(errors="replace")
        if text in log and alive(service):
            return
        if not alive(service):
            raise RuntimeError(f"{name} exited:\n{log[-7000:]}")
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {name}; see {service['log']}")


def stop_service(state, name):
    service = state["services"].get(name)
    if not alive(service):
        return
    if name == "mapper":
        # ros2 launch forwards the signal and allows map persistence to finish.
        os.kill(service["pid"], signal.SIGINT)
    else:
        # Env_X shell wrappers wait on children and do not forward SIGINT.
        os.killpg(service["pid"], signal.SIGINT)
    deadline = time.monotonic() + 180
    while alive(service) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(service):
        raise RuntimeError(f"{name} is still saving; retained PID {service['pid']}")
    print(f"Stopped {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "start", "sensors", "gui", "rviz", "status", "stop"])
    parser.add_argument("--host", default="192.168.10.22")
    parser.add_argument("--capture-port", type=int, default=6665)
    parser.add_argument("--car-port", type=int, default=6668)
    parser.add_argument("--capture-hz", type=float, default=0.25)
    parser.add_argument("--capture-only", action="store_true", help="Start acquisition without any vehicle-control connection")
    parser.add_argument("--domain", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "10")))
    parser.add_argument("--receipts", action="store_true", help="Enable per-frame receipts for verification")
    parser.add_argument("--replay", action="store_true", help="Isolated bag replay with /clock and per-frame receipts")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--enable-semantics", action="store_true")
    parser.add_argument("--semantic-topic", default="/task2/seg_mask")
    args = parser.parse_args()
    state = read_state()
    if args.action == "check":
        env = environment(domain=args.domain)
        subprocess.run(["/usr/bin/python3", "-c",
                        "import rclpy, torch; from roma_vo import StereoSLAMv2; "
                        "print('torch:',torch.__version__,'CUDA:',torch.cuda.is_available()); "
                        "assert torch.cuda.is_available()"], env=env, check=True)
        for name, port in (("capture", args.capture_port), ("car", args.car_port)):
            try:
                with socket.create_connection((args.host, port), timeout=3):
                    print(f"{name}: reachable {args.host}:{port}")
            except OSError as error:
                print(f"{name}: unavailable {args.host}:{port}: {error}")
        return
    if args.action == "status":
        print("Output:", state.get("output", "not started"))
        for name, service in state["services"].items():
            print(name, "RUNNING" if alive(service) else "STOPPED", service["pid"], service["log"])
        return
    if args.action == "stop":
        motion_path = ROOT / "integration_demo/motion_state.json"
        if state.get("domain") == 10 and motion_path.exists():
            motion = json.loads(motion_path.read_text())
            if alive(motion):
                os.kill(motion["pid"], signal.SIGINT)
                deadline = time.monotonic() + 60
                while alive(motion) and time.monotonic() < deadline:
                    time.sleep(.2)
                if alive(motion):
                    raise RuntimeError("Motion recorder is still stopping; keep car communication running")
        for name in ("gui", "capture", "car", "mapper", "dashboard", "rviz", "visuals"):
            stop_service(state, name)
        return
    if args.action == "start":
        active_names = {name for name, service in state['services'].items() if alive(service)}
        if active_names and active_names <= {'rviz', 'visuals'}:
            for name in ('rviz', 'visuals'):
                stop_service(state, name)
        if any(alive(service) for service in state["services"].values()):
            raise RuntimeError("A demo is already running; use status or stop")
        # A saved-map inspection must not look like a newly growing live/replay map.
        # Stop only services recorded by our own isolated preview entry point.
        preview_file = ROOT / 'integration_demo/preview_state.json'
        if preview_file.exists() and STATE != preview_file:
            preview = json.loads(preview_file.read_text())
            for name in ('rviz', 'visuals', 'preview'):
                stop_service(preview, name)
        # Prevent competing odometry/TF publishers from older entry points.
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                command = path.read_bytes()
            except (OSError, PermissionError):
                continue
            if b"roma_semantic_mapper\0" in command or b"t5_roma_vo_live_parallel" in command:
                try:
                    entries = (path.parent / "environ").read_bytes().split(b"\0")
                    domain = next((entry.split(b"=", 1)[1].decode() for entry in entries
                                   if entry.startswith(b"ROS_DOMAIN_ID=")), "0")
                except OSError:
                    domain = str(args.domain)
                if domain == str(args.domain):
                    raise RuntimeError("An existing RoMa process is active in this ROS domain")
        output = ROOT / "output_demo" / time.strftime(("replay_" if args.replay else "envx_") + "%Y%m%d_%H%M%S")
        output.mkdir(parents=True, exist_ok=False)
        state = {"output": str(output), "domain": args.domain, "services": {},
                 "verification_mode": args.receipts or args.replay, "replay": args.replay,
                 "dashboard_port": args.dashboard_port}
        save_state(state)
        env = environment(output, args.domain)
        command = ["ros2", "launch", "t3_semantic_mapping", "t3_envx_demo.launch.py",
                   f"output_dir:={output}", f"publish_input_processed:={str(args.receipts or args.replay).lower()}",
                   f"use_sim_time:={str(args.replay).lower()}",
                   f"enable_semantics:={str(args.enable_semantics).lower()}",
                   f"semantic_topic:={args.semantic_topic}"]
        start_service(state, "mapper", command, env)
        wait_ready(state, "mapper", "Mapper ready. Waiting for synchronized image input.")
        start_service(state, "dashboard", [
            "/usr/bin/python3", str(ROOT / "realtime_dashboard/t3_realtime_dashboard.py"),
            "--output-dir", str(output), "--host", "0.0.0.0", "--port", str(args.dashboard_port),
            "--estimate-topic", "/T3/semantic/current_pose", "--reference-topic", "/car/pose",
            "--global-map-topic", "/T3/mapping/global_grid_map",
        ], env)
        wait_ready(state, "dashboard", "Dashboard telemetry ready", timeout=40)
        start_rviz(state, env)
        print(f"READY. Dashboard: http://192.168.100.216:{args.dashboard_port}", flush=True)
        print("Start Env_X with: democtl.py sensors --host <UE_IP>", flush=True)
        return
    if not alive(state["services"].get("mapper")):
        raise RuntimeError("Start the demo mapper first")
    env = environment(state["output"], state["domain"])
    if args.action == "sensors":
        if state.get("verification_mode"):
            raise RuntimeError("Stop the recorded-data verification run, then start a fresh live demo")
        if state.get("ue_host") not in (None, args.host):
            raise RuntimeError("Stop the current sensor session before changing UE host")
        targets = [("capture", args.capture_port)]
        if not args.capture_only:
            targets.append(("car", args.car_port))
        for name, port in targets:
            if not alive(state["services"].get(name)):
                with socket.create_connection((args.host, port), timeout=4):
                    pass
        state["ue_host"] = args.host
        env["SENSOR_TIMESTAMP_MODE"] = "meta_relative"
        if not args.capture_only and not alive(state["services"].get("car")):
            start_service(state, "car", ["bash", str(ENVX / "start_car.sh"), args.host,
                                         str(args.car_port), "false"], env)
        if not alive(state["services"].get("capture")):
            start_service(state, "capture", ["bash", str(ENVX / "start_capture.sh"),
                                              "--host", args.host, "--port", str(args.capture_port),
                                              "--capture-freq", str(args.capture_hz), "--no-save"], env)
    else:
        if args.action == "rviz":
            start_rviz(state, env)
        else:
            start_service(state, "gui", ["bash", str(ENVX / "start_gui.sh")], display_environment(env))


if __name__ == "__main__":
    main()
