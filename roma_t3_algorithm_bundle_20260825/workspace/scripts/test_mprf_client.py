#!/usr/bin/env python3

import io
import sys
import time
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.localization.mprf_client import (
    MPRF_PROTOCOL_VERSION,
    MprfAsyncClient,
    MprfClientConfiguration,
    MprfKeyframeRequest,
    encode_mprf_request,
)


def request(node=4):
    return MprfKeyframeRequest(
        session_id="client-test",
        node_id=node,
        timestamp_ns=(node + 1) * 1_000_000_000,
        image_bgr=np.zeros((32, 48, 3), dtype=np.uint8),
        points_sensor=np.ones((40, 3), dtype=np.float32),
        camera_matrix=np.asarray(
            [[50.0, 0.0, 24.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]
        ),
        transform_camera_sensor=np.eye(4),
        calibration_width=48,
        calibration_height=32,
    )


def response(value, *, candidate=False):
    result = {
        "protocol_version": MPRF_PROTOCOL_VERSION,
        "status": "candidate" if candidate else "stored",
        "session_id": value.session_id,
        "query_node": value.node_id,
        "query_timestamp_ns": value.timestamp_ns,
    }
    if candidate:
        result["candidate"] = {
            "reference_node": 1,
            "query_node": value.node_id,
            "score": 0.75,
            "relative_pose_reference_query": np.eye(4).reshape(-1).tolist(),
            "ransac_inliers": 35,
        }
    return result


class MprfClientContractTest(unittest.TestCase):
    def test_npz_payload_has_versioned_pickle_free_fields(self):
        value = request()
        with np.load(io.BytesIO(encode_mprf_request(value)), allow_pickle=False) as data:
            self.assertEqual(int(data["protocol_version"]), 1)
            self.assertEqual(str(data["session_id"]), value.session_id)
            np.testing.assert_array_equal(data["image_bgr"], value.image_bgr)
            np.testing.assert_array_equal(data["calibration_size"], [48, 32])

    def test_candidate_response_becomes_untrusted_loop_candidate(self):
        client = MprfAsyncClient(
            MprfClientConfiguration(enabled=False), session_id="client-test"
        )
        value = request()
        parsed = client._parse_response(value, response(value, candidate=True))
        self.assertEqual(parsed.candidate.reference_node, 1)
        self.assertEqual(parsed.candidate.query_node, value.node_id)
        self.assertEqual(parsed.candidate.source, "mprf_dinov2_salad_sonata")
        self.assertFalse(parsed.candidate.coarse_relative_pose.flags.writeable)

    def test_retrieval_only_shortlist_keeps_pose_unset(self):
        client = MprfAsyncClient(
            MprfClientConfiguration(enabled=False), session_id="client-test"
        )
        value = request()
        first = {
            "reference_node": 1,
            "query_node": value.node_id,
            "score": 0.72,
            "candidate_mode": "retrieval_only",
        }
        second = {
            "reference_node": 2,
            "query_node": value.node_id,
            "score": 0.68,
            "candidate_mode": "retrieval_only",
        }
        payload = response(value, candidate=True)
        payload["candidate"] = first
        payload["candidates"] = [first, second]
        parsed = client._parse_response(value, payload)
        self.assertEqual(len(parsed.candidates), 2)
        self.assertIsNone(parsed.candidate.coarse_relative_pose)
        self.assertEqual(
            parsed.candidate.source, "mprf_salad_dinov2_retrieval"
        )

    def test_response_identity_mismatch_is_rejected(self):
        client = MprfAsyncClient(MprfClientConfiguration(enabled=False))
        value = request()
        bad = response(value)
        bad["query_node"] += 1
        with self.assertRaisesRegex(ValueError, "identity"):
            client._parse_response(value, bad)

    def test_invalid_candidate_score_is_rejected(self):
        client = MprfAsyncClient(MprfClientConfiguration(enabled=False))
        value = request()
        bad = response(value, candidate=True)
        bad["candidate"]["score"] = 1.5
        with self.assertRaisesRegex(ValueError, "outside"):
            client._parse_response(value, bad)

    def test_background_failure_is_delivered_without_raising_to_submitter(self):
        client = MprfAsyncClient(
            MprfClientConfiguration(
                enabled=True, request_timeout_sec=0.1, transient_retries=0
            ),
            session_id="client-test",
        )

        def fail(_request):
            raise RuntimeError("synthetic service failure")

        client._post = fail
        value = request()
        self.assertTrue(
            client.submit(
                node_id=value.node_id,
                timestamp_ns=value.timestamp_ns,
                image_bgr=value.image_bgr,
                points_sensor=value.points_sensor,
                camera_matrix=value.camera_matrix,
                transform_camera_sensor=value.transform_camera_sensor,
                calibration_width=value.calibration_width,
                calibration_height=value.calibration_height,
            )
        )
        deadline = time.monotonic() + 1.0
        results = ()
        while time.monotonic() < deadline and not results:
            results = client.poll()
            time.sleep(0.005)
        client.close()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "error")
        self.assertIn("synthetic service failure", results[0].error)


if __name__ == "__main__":
    unittest.main()
