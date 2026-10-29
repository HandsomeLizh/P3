import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mprf_runtime.config import PROTOCOL_VERSION
from mprf_runtime.contracts import KeyframePacket, decode_keyframe_packet
from mprf_runtime.storage import SessionKeyframeStore


def make_packet(
    node_id=0, timestamp_ns=1_000_000_000, session_id="test-session"
):
    return KeyframePacket(
        session_id=session_id,
        node_id=node_id,
        timestamp_ns=timestamp_ns,
        image_bgr=np.zeros((32, 48, 3), dtype=np.uint8),
        points_sensor=np.arange(120, dtype=np.float32).reshape(-1, 3),
        camera_matrix=np.asarray(
            [[50.0, 0.0, 24.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]
        ),
        transform_camera_sensor=np.eye(4),
        calibration_width=48,
        calibration_height=32,
    )


def encode_packet(packet):
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int32),
        session_id=np.asarray(packet.session_id),
        node_id=np.asarray(packet.node_id, dtype=np.int64),
        timestamp_ns=np.asarray(packet.timestamp_ns, dtype=np.int64),
        image_bgr=packet.image_bgr,
        points_sensor=packet.points_sensor,
        camera_matrix=packet.camera_matrix,
        transform_camera_sensor=packet.transform_camera_sensor,
        calibration_size=np.asarray(
            [packet.calibration_width, packet.calibration_height], dtype=np.int32
        ),
    )
    return stream.getvalue()


class PacketContractTest(unittest.TestCase):
    def test_round_trip_is_read_only_and_pickle_free(self):
        packet = make_packet()
        decoded = decode_keyframe_packet(encode_packet(packet), maximum_points=100)
        self.assertEqual(decoded.session_id, packet.session_id)
        self.assertFalse(decoded.image_bgr.flags.writeable)
        self.assertFalse(decoded.points_sensor.flags.writeable)
        np.testing.assert_allclose(decoded.camera_matrix, packet.camera_matrix)

    def test_rejects_bad_pose_and_unsafe_session(self):
        bad_pose = np.eye(4)
        bad_pose[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            KeyframePacket(
                session_id="safe",
                node_id=0,
                timestamp_ns=1,
                image_bgr=np.zeros((20, 20, 3), np.uint8),
                points_sensor=np.zeros((3, 3)),
                camera_matrix=np.eye(3),
                transform_camera_sensor=bad_pose,
                calibration_width=20,
                calibration_height=20,
            )
        with self.assertRaisesRegex(ValueError, "session_id"):
            KeyframePacket(
                session_id="../escape",
                node_id=0,
                timestamp_ns=1,
                image_bgr=np.zeros((20, 20, 3), np.uint8),
                points_sensor=np.zeros((3, 3)),
                camera_matrix=np.diag([1.0, 1.0, 1.0]),
                transform_camera_sensor=np.eye(4),
                calibration_width=20,
                calibration_height=20,
            )

    def test_request_point_limit_is_enforced_before_copy(self):
        with self.assertRaisesRegex(ValueError, "exceeds maximum"):
            decode_keyframe_packet(encode_packet(make_packet()), maximum_points=10)


class PersistentStoreTest(unittest.TestCase):
    def test_atomic_store_survives_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = make_packet()
            store = SessionKeyframeStore(root, packet.session_id)
            store.add(
                packet,
                salad_descriptor=np.arange(8, dtype=np.float32) + 1.0,
                dino_descriptor=np.arange(5, dtype=np.float32) + 1.0,
            )
            self.assertEqual(len(store), 1)
            restarted = SessionKeyframeStore(root, packet.session_id)
            self.assertEqual(len(restarted), 1)
            restored = restarted.load_packet(restarted.records()[0])
            np.testing.assert_array_equal(restored.image_bgr, packet.image_bgr)
            np.testing.assert_allclose(restored.points_sensor, packet.points_sensor)
            self.assertFalse(list((root / packet.session_id).glob("*.tmp")))

    def test_rejects_non_monotonic_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionKeyframeStore(Path(temporary), "monotonic")
            first = make_packet(2, 2_000_000_000, "monotonic")
            store.add(
                first,
                salad_descriptor=np.ones(3),
                dino_descriptor=np.ones(4),
            )
            older = make_packet(1, 3_000_000_000, "monotonic")
            with self.assertRaisesRegex(ValueError, "increasing"):
                store.add(
                    older,
                    salad_descriptor=np.ones(3),
                    dino_descriptor=np.ones(4),
                )


if __name__ == "__main__":
    unittest.main()
