"""Orin integration regression: SIGINT while a local TCP peer sends no data.

Only the private P3 binary is launched; no request is sent to the UE server.
Run after sourcing envx_runtime/install/setup.bash.
"""
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CaptureShutdownTests(unittest.TestCase):
    def test_idle_tcp_peer_does_not_block_shutdown(self):
        binary = ROOT / "envx_runtime/install/sensor_capture_ros2/lib/sensor_capture_ros2/sensor_capture_node"
        self.assertTrue(binary.is_file())
        test_dir = ROOT / "integration_demo/logs/capture_shutdown"
        test_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, ROS_DOMAIN_ID="19", ROS_LOG_DIR=str(test_dir / "ros"))
        for attempt in range(3):
            with self.subTest(attempt=attempt), socket.socket() as server:
                server.bind(("127.0.0.1", 0))
                server.listen(1)
                server.settimeout(15)
                with (test_dir / f"idle_{attempt}.log").open("wb") as log:
                    process = subprocess.Popen([
                        str(binary), "--ros-args", "-p", "tcp_host:=127.0.0.1",
                        "-p", f"tcp_port:={server.getsockname()[1]}",
                        "-p", "auto_start:=true", "-p", "save_data:=false",
                        "-p", f"save_dir:={test_dir / 'scratch'}"],
                        env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    try:
                        peer, _ = server.accept()
                        with peer:
                            peer.settimeout(10)
                            self.assertTrue(peer.recv(4096), "capture request missing")
                            time.sleep(0.3)
                            started = time.monotonic()
                            process.send_signal(signal.SIGINT)
                            code = process.wait(timeout=8)
                            self.assertEqual(code, 0)
                            self.assertLess(time.monotonic() - started, 5)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
