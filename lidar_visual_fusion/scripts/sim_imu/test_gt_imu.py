#!/usr/bin/env python3
"""Analytic checks for the synthetic IMU: kinematics, gravity, offset recovery, parsing."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_imu  # noqa: E402

G = gt_imu.GRAVITY
R_CIRCLE, W, SLOPE = 6., .04, np.radians(4.)   # rover-like: 0.24 m/s, 2.3 deg/s, 4 deg ramp


def truth_at(ue_t):
    """Circle on a tilted plane; body x tangent, body z the plane normal (base_link FLU)."""
    ue_t = np.atleast_1d(ue_t)
    tilt = Rot.from_euler('x', SLOPE)
    planar = np.column_stack([R_CIRCLE * np.cos(W * ue_t), R_CIRCLE * np.sin(W * ue_t), np.zeros_like(ue_t)])
    position = tilt.apply(planar)
    yaw = Rot.from_euler('z', W * ue_t + np.pi / 2)
    base = tilt * yaw
    ue_quat = (base * gt_imu.UE_CHILD_FROM_BASE.inv()).as_quat()   # what /car/pose carries
    return position, ue_quat, base


def expected_imu(ue_t):
    _, _, base = truth_at(ue_t)
    omega_world = Rot.from_euler('x', SLOPE).apply([0., 0., W])
    accel_world = Rot.from_euler('x', SLOPE).apply(-W ** 2 * R_CIRCLE * np.array([np.cos(W * ue_t), np.sin(W * ue_t), 0.]))
    gyro = base.inv().apply(omega_world)[0]
    force = base.inv().apply(accel_world + [0., 0., G])[0]
    return gyro, force


def stream(duration=120., offset=.6, drift=1e-4, jitter=.02, repeat=.05, seed=1):
    """/car/pose as received: 20 Hz UE samples, receive = UE time + offset(t) + jitter, stale repeats."""
    rng = np.random.default_rng(seed)
    ue = np.arange(0., duration, .05)
    receive = ue + offset + drift * ue + np.abs(rng.normal(0., jitter, len(ue)))
    receive = np.maximum.accumulate(receive + 1e-6 * np.arange(len(ue)))
    p, q, _ = truth_at(ue)
    p, q = p.copy(), q.copy()
    for k in np.flatnonzero(rng.random(len(ue)) < repeat):
        if k:
            p[k], q[k] = p[k - 1], q[k - 1]          # stale resend of the previous payload
    return receive, p, q


class SyntheticImu(unittest.TestCase):
    def test_kinematics_match_analytic_motion(self):
        ue = np.arange(0., 60., .05)
        p, q, _ = truth_at(ue)
        rotations = gt_imu.base_rotation(q)
        for tau in (10., 25.3, 40.7):
            state = gt_imu.kinematics(ue, p, rotations, tau)
            gyro, force = gt_imu.imu_values(state)
            want_g, want_f = expected_imu(tau)
            self.assertLess(np.abs(gyro - want_g).max(), 2e-5)
            self.assertLess(np.abs(force - want_f).max(), 2e-3)

    def test_specific_force_at_rest(self):
        t = np.arange(0., 3., .05)
        p = np.zeros((len(t), 3))
        level = np.tile((Rot.identity() * gt_imu.UE_CHILD_FROM_BASE.inv()).as_quat(), (len(t), 1))
        state = gt_imu.kinematics(t, p, gt_imu.base_rotation(level), 1.5)
        gyro, force = gt_imu.imu_values(state)
        np.testing.assert_allclose(force, [0., 0., G], atol=1e-9)
        np.testing.assert_allclose(gyro, 0., atol=1e-12)
        tilted = np.tile((Rot.from_euler('y', 10, degrees=True) * gt_imu.UE_CHILD_FROM_BASE.inv()).as_quat(), (len(t), 1))
        _, force = gt_imu.imu_values(gt_imu.kinematics(t, p, gt_imu.base_rotation(tilted), 1.5))
        self.assertAlmostEqual(np.linalg.norm(force), G, places=9)
        self.assertAlmostEqual(np.degrees(np.arctan2(-force[0], force[2])), 10., places=6)

    def test_offset_recovered_with_jitter_and_stale_repeats(self):
        offset, drift = .6, 1e-4
        receive, p, q = stream(offset=offset, drift=drift)
        keep = gt_imu.drop_stale_repeats(receive, p, q)
        t, p, q = receive[keep], p[keep], q[keep]
        rotations = gt_imu.base_rotation(q)
        model = gt_imu.OffsetModel(prior=.65)
        for s in np.arange(5., 115., 1.):            # captures: sensor stamp == UE time here
            position, quat, _ = truth_at(s)
            tau = gt_imu.match_capture(t, p, rotations, position[0], gt_imu.base_rotation(quat[0]), s, s + 2.5)
            self.assertIsNotNone(tau)
            model.add(s, tau - s)
        self.assertTrue(model.locked)
        for s in (60., 110.):
            self.assertLess(abs(model.target(s) - (offset + drift * s)), .02)

    def test_imu_on_sensor_axis_from_received_stream(self):
        """End to end: integrated gyro between captures equals the true relative rotation."""
        receive, p, q = stream(offset=.6, drift=0., jitter=.01)
        keep = gt_imu.drop_stale_repeats(receive, p, q)
        t, p, q = receive[keep], p[keep], q[keep]
        rotations = gt_imu.base_rotation(q)
        errors = {}
        for name, offset in (('aligned', .6), ('naive', 0.)):
            gyro_s = []
            grid = np.arange(20., 30.001, .01)
            for s in grid:
                gyro_s.append(gt_imu.kinematics(t, p, rotations, s + offset)['omega'])
            rotation = Rot.identity()
            for k in range(len(grid) - 1):
                rotation = rotation * Rot.from_rotvec(.5 * (gyro_s[k] + gyro_s[k + 1]) * .01)
            truth = truth_at(20.)[2].inv() * truth_at(30.)[2]
            errors[name] = np.degrees((rotation.inv() * truth).magnitude())
        self.assertLess(errors['aligned'], .05)

    def test_stop_is_kept_and_stale_repeat_dropped(self):
        t = np.arange(0., 2., .05)
        p = np.column_stack([np.minimum(t, 1.) * .1, np.zeros(len(t)), np.zeros(len(t))])
        q = np.tile([0., 0., 0., 1.], (len(t), 1))
        p[5] = p[4]                                    # stale resend while moving
        keep = gt_imu.drop_stale_repeats(t, p, q)
        self.assertFalse(keep[5])
        self.assertTrue(keep[25:].all())               # genuine stop after t=1 s
        live = gt_imu.drop_stale_repeats(t, p, q, live=True)
        self.assertTrue(live[25:-2].all())
        self.assertFalse(live[-1])                     # newest repeat unresolved

    def test_metadata_conversion_matches_pose_stream(self):
        # Recorded 2026-09-18: /sensor/meta and /car/pose of the same instant.
        text = json.dumps(dict(batch_id=1275, timestamp='2026-09-18 14:21:16.159',
            position=dict(x=135277.421875, y=101470.265625, z=614.69866943359375),
            rotation=dict(qx=-0.093100978823569214, qy=-0.73882643403563453,
                          qz=0.65964578285134612, qw=0.10166193623336194)))
        batch, stamp_ns, position, quat = gt_imu.parse_meta(text)
        np.testing.assert_allclose(position, [1352.774169921875, -1014.70263671875, 6.146986484527588], atol=1e-5)
        np.testing.assert_allclose(quat, [0.09310097992420197, -0.7388264536857605, -0.6596457958221436,
                                          0.10166193544864655], atol=1e-6)
        self.assertEqual(batch, 1275)
        self.assertEqual(stamp_ns % 10**9, 159000000)
        self.assertIsNone(gt_imu.parse_meta('{B'))
        self.assertEqual(gt_imu.parse_batch_status('{"batch_id":7,"stamp_ns":12}'), (7, 12))
        self.assertIsNone(gt_imu.parse_batch_status('{"batch_id":7}'))

    def test_noise_model(self):
        clean = gt_imu.NoiseModel('none')
        g, a = clean.apply(np.ones(3), np.ones(3), .01)
        np.testing.assert_array_equal(g, 1.); np.testing.assert_array_equal(a, 1.)
        noisy = gt_imu.NoiseModel('mems', seed=3)
        samples = np.array([noisy.apply(np.zeros(3), np.zeros(3), .01)[0] for _ in range(4000)])
        white = np.diff(samples, axis=0).std() / np.sqrt(2)
        self.assertAlmostEqual(white / (2.6e-4 / np.sqrt(.01)), 1., delta=.1)
        self.assertGreater(min(noisy.covariances(.01)), 0.)

    def test_offset_model_slew_and_restart(self):
        model = gt_imu.OffsetModel(prior=.5, slew=.02)
        self.assertEqual(model.at(0.), .5)
        for s in np.arange(1., 31.):
            model.add(s, .8)
        self.assertAlmostEqual(model.target(31.), .8, places=6)
        self.assertAlmostEqual(model.at(1.), .5 + .02 * 1., places=9)
        model.add(5., .8)                              # sensor stamps restarted
        self.assertFalse(model.locked)
        self.assertFalse(model.add(40., 9.))           # implausible offset rejected


if __name__ == '__main__':
    unittest.main()
