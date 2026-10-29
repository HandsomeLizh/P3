#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

from democtl import ROOT, alive, read_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="short")
    args = parser.parse_args()
    controller = ["/usr/bin/python3", str(ROOT / "integration_demo/democtl.py")]
    state = read_state()
    if not alive(state["services"].get("mapper")):
        subprocess.run([*controller, "start"], check=True)
    state = read_state()
    if state.get("verification_mode") or state.get("domain") != 10:
        raise RuntimeError("Live recording requires the non-replay demo in domain 10")
    subprocess.run([*controller, "sensors", "--host", "192.168.10.22"], check=True)
    state = read_state()
    if not alive(state["services"].get("gui")):
        subprocess.run([*controller, "gui"], check=True)
    subprocess.run(["/usr/bin/python3", str(ROOT / "integration_demo/fixed_motion.py"),
                    "start", "--profile", args.profile], check=True)


if __name__ == "__main__":
    main()
