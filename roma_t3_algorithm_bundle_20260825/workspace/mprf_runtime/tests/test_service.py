import http.client
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from mprf_runtime.config import MprfRuntimeConfiguration, PROTOCOL_VERSION
from mprf_runtime.service import MprfServiceServer


class FakeEngine:
    def process(self, packet):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "status": "stored",
            "session_id": packet.session_id,
            "query_node": packet.node_id,
            "query_timestamp_ns": packet.timestamp_ns,
        }

    def health(self):
        return {"protocol_version": PROTOCOL_VERSION, "status": "ready"}

    def reset(self, session_id):
        return {"protocol_version": PROTOCOL_VERSION, "status": "ready", "session_id": session_id}


def payload():
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        session_id=np.asarray("http-test"),
        node_id=np.asarray(3),
        timestamp_ns=np.asarray(4_000_000_000),
        image_bgr=np.zeros((32, 48, 3), dtype=np.uint8),
        points_sensor=np.ones((40, 3), dtype=np.float32),
        camera_matrix=np.asarray(
            [[50.0, 0.0, 24.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]
        ),
        transform_camera_sensor=np.eye(4),
        calibration_size=np.asarray([48, 32]),
    )
    return stream.getvalue()


class HttpServiceTest(unittest.TestCase):
    def make_server(self, token=None):
        temporary = tempfile.TemporaryDirectory()
        configuration = MprfRuntimeConfiguration(state_root=Path(temporary.name))
        server = MprfServiceServer(
            ("127.0.0.1", 0),
            engine=FakeEngine(),
            configuration=configuration,
            auth_token=token,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(thread.join, 1.0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_npz_endpoint_round_trip(self):
        server = self.make_server()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2.0)
        body = payload()
        connection.request(
            "POST",
            "/v1/keyframes",
            body=body,
            headers={"Content-Type": "application/x-npz", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        decoded = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(decoded["query_node"], 3)
        self.assertEqual(decoded["status"], "stored")

    def test_non_loopback_style_token_gate_rejects_missing_bearer(self):
        server = self.make_server(token="secret")
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2.0)
        connection.request("GET", "/health")
        response = connection.getresponse()
        decoded = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 401)
        self.assertEqual(decoded["error"], "unauthorized")


if __name__ == "__main__":
    unittest.main()

