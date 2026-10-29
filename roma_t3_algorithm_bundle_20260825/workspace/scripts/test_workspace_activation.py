#!/usr/bin/env python3
"""Runtime contracts for repeatable workspace activation."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
ACTIVATION_SCRIPT = WORKSPACE / "scripts" / "enter_workspace.sh"


def _source_command(*, times: int) -> str:
    source = f"source {shlex.quote(str(ACTIVATION_SCRIPT))}"
    return "; ".join(["set -e", *([source] * times)])


class WorkspaceActivationTest(unittest.TestCase):
    def test_activation_is_repeatable(self):
        environment = dict(os.environ)
        environment.pop("ROS_VENV", None)

        result = subprocess.run(
            ["bash", "-c", _source_command(times=2)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )

    def test_explicit_missing_virtual_environment_still_fails(self):
        environment = dict(os.environ)
        missing = "/__t3_explicit_missing_virtual_environment__"
        environment["ROS_VENV"] = missing

        result = subprocess.run(
            ["bash", "-c", _source_command(times=1)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            f"Explicit ROS_VENV does not exist: {missing}",
            result.stderr,
        )

    def test_wrapper_runs_after_prior_activation(self):
        environment = dict(os.environ)
        environment.pop("ROS_VENV", None)

        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            binary_directory = directory / "bin"
            binary_directory.mkdir()
            (binary_directory / "python3").symlink_to("/usr/bin/python3")
            (binary_directory / "ros2").symlink_to("/bin/echo")
            dataset = directory / "sequence"
            (dataset / "images").mkdir(parents=True)
            (dataset / "images" / "timestamps.tsv").touch()
            environment["ROS_PYTHON_EXECUTABLE"] = str(
                binary_directory / "python3"
            )
            command = (
                _source_command(times=1)
                + "; bash "
                + shlex.quote(
                    str(WORKSPACE / "scripts" / "run_batch_organized.sh")
                )
                + " "
                + shlex.quote(str(dataset))
                + " max_frames:=1"
            )
            result = subprocess.run(
                ["bash", "-c", command],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )
        self.assertIn(
            "launch t3_semantic_mapping "
            "practice_live_online_semantics.launch.py",
            result.stdout,
        )
        self.assertIn("max_frames:=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
