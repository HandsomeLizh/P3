#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.map_admission import (
    MapAdmissionConfig,
    MapAdmissionGate,
    MapAdmissionState,
)


def _pose(x=0.0, y=0.0, yaw_deg=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    pose[:3, 3] = [x, y, 0.0]
    return pose


def _gate(**overrides):
    values = {
        "enabled": True,
        "min_recovery_inliers": 25,
        "min_recovery_inlier_ratio": 0.25,
        "stable_frames_required": 3,
        "max_translation_step_m": 1.0,
        "max_rotation_step_deg": 20.0,
    }
    values.update(overrides)
    return MapAdmissionGate(MapAdmissionConfig(**values))


def _observe(
    gate,
    pose,
    *,
    reason="tracking_success",
    inliers=30,
    correspondences=40,
):
    return gate.observe_tracking_success(
        pose,
        reason=reason,
        inlier_count=inliers,
        correspondence_count=correspondences,
    )


class MapAdmissionGateTest(unittest.TestCase):
    def test_initial_and_uninterrupted_tracking_preserve_old_behavior(self):
        gate = _gate()

        initial = _observe(
            gate,
            _pose(),
            reason="initialized",
            inliers=0,
            correspondences=0,
        )
        weak_but_uninterrupted = _observe(
            gate,
            _pose(0.1),
            inliers=6,
            correspondences=8,
        )

        self.assertTrue(initial.allow_new_observation)
        self.assertTrue(weak_but_uninterrupted.allow_new_observation)
        self.assertEqual(gate.state, MapAdmissionState.TRACKING)

    def test_low_inlier_success_after_failure_is_not_admitted(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        decision = _observe(
            gate,
            _pose(0.1),
            inliers=7,
            correspondences=9,
        )

        self.assertFalse(decision.allow_new_observation)
        self.assertEqual(decision.reason, "recovery_low_confidence")
        self.assertEqual(decision.stable_frames, 0)
        self.assertEqual(gate.state, MapAdmissionState.RECOVERING)

    def test_three_consecutive_confident_frames_reopen_map(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        first = _observe(gate, _pose(0.1))
        second = _observe(gate, _pose(0.2))
        third = _observe(gate, _pose(0.3))

        self.assertFalse(first.allow_new_observation)
        self.assertFalse(second.allow_new_observation)
        self.assertTrue(third.allow_new_observation)
        self.assertEqual(third.reason, "recovery_confirmed")
        self.assertEqual(gate.state, MapAdmissionState.TRACKING)

        ordinary = _observe(gate, _pose(0.4))
        self.assertEqual(ordinary.reason, "tracking")
        self.assertEqual(ordinary.stable_frames, 0)

    def test_low_confidence_frame_resets_consecutive_count(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        self.assertEqual(_observe(gate, _pose(0.1)).stable_frames, 1)
        weak = _observe(
            gate, _pose(0.2), inliers=24, correspondences=40
        )
        restarted = _observe(gate, _pose(0.3))

        self.assertFalse(weak.allow_new_observation)
        self.assertEqual(weak.stable_frames, 0)
        self.assertEqual(restarted.stable_frames, 1)

    def test_recovery_inlier_ratio_is_checked_independently(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        decision = _observe(
            gate,
            _pose(0.1),
            inliers=30,
            correspondences=200,
        )

        self.assertFalse(decision.allow_new_observation)
        self.assertEqual(decision.reason, "recovery_low_confidence")
        self.assertAlmostEqual(decision.inlier_ratio, 0.15)

    def test_recovery_rotation_jump_is_checked_independently(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        decision = _observe(gate, _pose(0.1, yaw_deg=45.0))

        self.assertFalse(decision.allow_new_observation)
        self.assertEqual(decision.reason, "recovery_motion_discontinuity")
        self.assertAlmostEqual(decision.rotation_step_deg, 45.0)

    def test_unverified_large_pose_jump_stays_blocked(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        jumped = _observe(gate, _pose(6.15))

        self.assertFalse(jumped.allow_new_observation)
        self.assertEqual(jumped.reason, "recovery_motion_discontinuity")
        self.assertAlmostEqual(jumped.translation_step_m, 6.15)
        self.assertEqual(jumped.stable_frames, 0)

        # A later pose returning to the trusted neighbourhood can start a new
        # clean consecutive sequence; the bad jump never becomes a reference.
        self.assertEqual(_observe(gate, _pose(0.1)).stable_frames, 1)
        self.assertEqual(_observe(gate, _pose(0.2)).stable_frames, 2)
        self.assertTrue(_observe(gate, _pose(0.3)).allow_new_observation)

    def test_verified_relocalization_can_seed_a_distant_reference(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        relocalized = _observe(
            gate,
            _pose(6.0),
            reason="relocalized",
            inliers=30,
            correspondences=60,
        )
        second = _observe(gate, _pose(6.1))
        third = _observe(gate, _pose(6.2))

        self.assertFalse(relocalized.allow_new_observation)
        self.assertTrue(relocalized.verified_relocalization)
        self.assertEqual(relocalized.stable_frames, 1)
        self.assertFalse(second.allow_new_observation)
        self.assertTrue(third.allow_new_observation)

    def test_low_inlier_relocalization_seeds_but_does_not_count(self):
        gate = _gate()
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        relocalized = _observe(
            gate,
            _pose(5.0),
            reason="relocalized",
            inliers=20,
            correspondences=50,
        )

        self.assertFalse(relocalized.allow_new_observation)
        self.assertEqual(relocalized.reason, "recovery_low_confidence")
        self.assertEqual(relocalized.stable_frames, 0)
        self.assertEqual(_observe(gate, _pose(5.1)).stable_frames, 1)
        self.assertEqual(_observe(gate, _pose(5.2)).stable_frames, 2)
        self.assertTrue(_observe(gate, _pose(5.3)).allow_new_observation)

    def test_disabled_gate_is_a_true_behavioral_bypass(self):
        gate = _gate(enabled=False)
        _observe(gate, _pose(), reason="initialized", inliers=0,
                 correspondences=0)
        gate.record_tracking_failure()

        decision = _observe(
            gate, _pose(100.0), inliers=0, correspondences=0
        )

        self.assertTrue(decision.allow_new_observation)
        self.assertEqual(decision.reason, "disabled")
        self.assertEqual(gate.state, MapAdmissionState.TRACKING)

    def test_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            _gate(min_recovery_inliers=0)
        with self.assertRaises(ValueError):
            _gate(min_recovery_inlier_ratio=1.1)
        with self.assertRaises(ValueError):
            _gate(stable_frames_required=0)
        with self.assertRaises(ValueError):
            _gate(max_translation_step_m=float("nan"))


if __name__ == "__main__":
    unittest.main()
