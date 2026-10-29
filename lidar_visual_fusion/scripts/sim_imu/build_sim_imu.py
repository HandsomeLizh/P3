#!/usr/bin/env python3
"""Offline: build a synthetic IMU (npz) on the sensor time axis from recorded truth.

Reads /car/pose, /sensor/meta and /sensor/batch_status from --truth. When batch_status has
no stamp_ns (older capture builds), sensor stamps come from LiDAR headers in --sensor-bag,
paired with meta by receive time. Writes <out>.npz and <out>.json (offset fit + self-checks).
The npz also stores the truth used, for evaluation only; it is never published.
"""
import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_imu  # noqa: E402


def db_files(directory):
    files = glob.glob(str(Path(directory) / '*.db3'))
    return sorted(files, key=lambda p: int(Path(p).stem.rsplit('_', 1)[1]) if Path(p).stem.rsplit('_', 1)[1].isdigit() else 0)


def read_topic(directory, topic, decode):
    """[(receive_ns, decoded)] in receive order across all bag files."""
    rows = []
    for name in db_files(directory):
        con = sqlite3.connect(f'file:{Path(name).resolve()}?mode=ro', uri=True)
        found = con.execute('SELECT id FROM topics WHERE name=?', (topic,)).fetchone()
        if found:
            rows += [(ts, decode(data)) for ts, data in
                     con.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp', found)]
        con.close()
    rows.sort(key=lambda r: r[0])
    return rows


def header_ns(data):
    """CDR std_msgs/Header stamp without deserialising the (large) payload."""
    return int.from_bytes(data[4:8], 'little', signed=True) * 10**9 + int.from_bytes(data[8:12], 'little')


def load(args):
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    def pose(data):
        m = deserialize_message(data, PoseStamped)
        s, p, q = m.header.stamp, m.pose.position, m.pose.orientation
        return s.sec * 10**9 + s.nanosec, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
    text = lambda data: deserialize_message(data, String).data
    truth = read_topic(args.truth, args.pose_topic, pose)
    meta = read_topic(args.truth, '/sensor/meta', text)
    status = read_topic(args.truth, '/sensor/batch_status', text)
    t = np.array([r[1][0] for r in truth], dtype=np.int64)
    p = np.array([r[1][1] for r in truth], float)
    q = np.array([r[1][2] for r in truth], float)
    order = np.argsort(t, kind='stable')
    t, p, q = t[order], p[order], q[order]
    stamps = {}
    for _, value in status:
        parsed = gt_imu.parse_batch_status(value)
        if parsed:
            stamps[parsed[0]] = parsed[1]
    captures, invalid, unstamped = [], 0, []
    for receive_ns, value in meta:
        parsed = gt_imu.parse_meta(value)
        if parsed is None:
            invalid += 1
            continue
        batch_id, _, position, quat = parsed
        if batch_id in stamps:
            captures.append((stamps[batch_id], position, quat))
        else:
            unstamped.append((receive_ns, position, quat))
    source = 'batch_status.stamp_ns'
    if unstamped and args.sensor_bag:
        lidar = read_topic(args.sensor_bag, args.sensor_topic, header_ns)
        receive = np.array([r[0] for r in lidar], dtype=np.int64)
        header = np.array([r[1] for r in lidar], dtype=np.int64)
        for receive_ns, position, quat in unstamped:
            k = int(np.argmin(np.abs(receive - receive_ns)))
            if abs(receive[k] - receive_ns) <= 5e8:
                captures.append((int(header[k]), position, quat))
        source = 'sensor header nearest by receive time' if not stamps else 'mixed'
    captures.sort(key=lambda c: c[0])
    return t, p, q, captures, dict(meta_invalid=invalid, meta_without_stamp=len(unstamped), stamp_source=source)


def robust_line(s, o, reference, max_slope=2e-3):
    slope = 0.
    if len(s) >= 5 and np.ptp(s) >= 15.:
        i, j = np.triu_indices(len(s), 1)
        usable = s[j] - s[i] > 1e-6
        slope = float(np.clip(np.median((o[j] - o[i])[usable] / (s[j] - s[i])[usable]), -max_slope, max_slope))
    return float(np.median(o - slope * (s - reference))), slope


def offline_offset(pair_s, pair_o, grid_s, half=30., prior=.65):
    """Centred robust line in +-half seconds at 1 s knots, two-pass 3-MAD rejection."""
    if len(pair_s) == 0:
        return np.full(len(grid_s), prior), np.zeros(0, bool)
    knots = np.arange(grid_s[0], grid_s[-1] + 1., 1.)
    values = np.full(len(knots), np.nan)
    inlier = np.ones(len(pair_s), bool)
    for _ in range(2):
        for k, c in enumerate(knots):
            use = (np.abs(pair_s - c) <= half) & inlier
            if use.sum() >= 3:
                values[k] = sum(robust_line(pair_s[use], pair_o[use], c)[:1])
        good = np.isfinite(values)
        if not good.any():
            return np.full(len(grid_s), float(np.median(pair_o))), inlier
        fitted = np.interp(pair_s, knots[good], values[good])
        residual = pair_o - fitted
        mad = 1.4826 * np.median(np.abs(residual[inlier]))
        inlier = np.abs(residual) <= max(3. * mad, .005)
    return np.interp(grid_s, knots[good], values[good]), inlier


def integrate_gyro(stamps, gyro, start, end):
    use = (stamps >= start) & (stamps <= end)
    ts, gs = stamps[use], gyro[use]
    rotation = Rot.identity()
    for k in range(len(ts) - 1):
        rotation = rotation * Rot.from_rotvec(.5 * (gs[k] + gs[k + 1]) * (ts[k + 1] - ts[k]))
    return rotation


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--truth', required=True, help='bag directory with /car/pose and /sensor/meta')
    a.add_argument('--sensor-bag', default='', help='bag with LiDAR headers, only if batch_status lacks stamp_ns')
    a.add_argument('--sensor-topic', default='/Car/T5/OS1/points')
    a.add_argument('--pose-topic', default='/car/pose')
    a.add_argument('--out', required=True, help='output path without extension')
    a.add_argument('--rate', type=float, default=100.)
    a.add_argument('--noise', default='mems', choices=sorted(gt_imu.NoiseModel.PRESETS))
    a.add_argument('--noise-scale', type=float, default=1.)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--fixed-offset', type=float, default=None, help='ablation: use this offset instead of the estimate')
    args = a.parse_args()
    out = Path(args.out)
    if out.with_suffix('.npz').exists():
        a.error('output exists; choose a new --out')

    t_ns, p, q, captures, load_report = load(args)
    if len(t_ns) < 50 or len(captures) < 5:
        raise SystemExit(f'not enough data: {len(t_ns)} truth poses, {len(captures)} stamped captures')
    origin_ns = int(captures[0][0])           # keep float seconds small and exact
    t = (t_ns - origin_ns) * 1e-9
    keep = gt_imu.drop_stale_repeats(t, p, q)
    t, p, q = t[keep], p[keep], q[keep]
    rotations = gt_imu.base_rotation(q)

    pair_s, pair_o = [], []
    for stamp_ns, position, quat in captures:
        s = (stamp_ns - origin_ns) * 1e-9
        tau = gt_imu.match_capture(t, p, rotations, position, gt_imu.base_rotation(quat), s, s + 2.5)
        if tau is not None:
            pair_s.append(s); pair_o.append(tau - s)
    pair_s, pair_o = np.array(pair_s), np.array(pair_o)

    s0 = max(0., t[0] - (np.median(pair_o) if len(pair_o) else .65))
    s_end = (captures[-1][0] - origin_ns) * 1e-9 + 1.
    grid = np.arange(np.ceil(s0 * args.rate) / args.rate, s_end, 1. / args.rate)
    if args.fixed_offset is None:
        offset, inlier = offline_offset(pair_s, pair_o, grid)
    else:
        offset, inlier = np.full(len(grid), args.fixed_offset), np.ones(len(pair_s), bool)

    noise = gt_imu.NoiseModel(args.noise, args.noise_scale, args.seed)
    dt = 1. / args.rate
    rows = []
    for s, o in zip(grid, offset):
        state = gt_imu.kinematics(t, p, rotations, s + o)
        if state is None:
            continue
        gyro, accel = gt_imu.imu_values(state)
        noisy_g, noisy_a = noise.apply(gyro, accel, dt)
        rows.append((s, o, gyro, accel, noisy_g, noisy_a, state['position'], state['rotation'].as_quat()))
    if not rows:
        raise SystemExit('no IMU samples could be generated')
    s = np.array([r[0] for r in rows])
    gyro_clean = np.array([r[2] for r in rows]); accel_clean = np.array([r[3] for r in rows])
    stamp_ns = origin_ns + np.round(s * 1e9).astype(np.int64)

    # Self-check 1: at rest the specific force is gravity.
    speed = np.linalg.norm(np.gradient(np.array([r[6] for r in rows]), s, axis=0), axis=1)
    rest = (speed < .005) & (np.linalg.norm(gyro_clean, axis=1) < .005)
    # Self-check 2: rotation between consecutive captures, gyro-integrated vs metadata truth.
    # The alternative constant offsets show what the time alignment is worth.
    def rotation_check(constant=None):
        errors = []
        for (sa, _, qa), (sb, _, qb) in zip(captures[:-1], captures[1:]):
            a_s, b_s = (sa - origin_ns) * 1e-9, (sb - origin_ns) * 1e-9
            truth = gt_imu.base_rotation(qa).inv() * gt_imu.base_rotation(qb)
            if not (0 < b_s - a_s < 2.) or truth.magnitude() < np.radians(.5):
                continue
            if constant is None:
                ts, gs = s, gyro_clean
            else:
                ts = np.arange(a_s, b_s + dt / 2, dt)
                states = [gt_imu.kinematics(t, p, rotations, x + constant) for x in ts]
                if any(v is None for v in states):
                    continue
                gs = np.array([v['omega'] for v in states])
            errors.append(np.degrees((integrate_gyro(ts, gs, a_s, b_s).inv() * truth).magnitude()))
        return dict(intervals=len(errors), median=float(np.median(errors)) if errors else None,
                    p95=float(np.percentile(errors, 95)) if errors else None)
    checks = {'estimated_offset': rotation_check(), 'zero_offset_naive': rotation_check(0.),
              'constant_prior_0.65s': rotation_check(.65)}
    gaps = np.diff(s)
    report = dict(load_report, truth_poses=int(keep.size), truth_kept=int(keep.sum()), captures=len(captures),
        offset_pairs=int(len(pair_s)), offset_inliers=int(inlier.sum()),
        offset_estimate=dict(median=float(np.median(offset)), min=float(offset.min()), max=float(offset.max()))
            if args.fixed_offset is None else dict(fixed=args.fixed_offset),
        offset_pair_residual_mad_ms=float(1e3 * 1.4826 * np.median(np.abs(pair_o - np.interp(pair_s, s, [r[1] for r in rows]))))
            if len(pair_s) else None,
        imu=dict(samples=len(s), rate_hz=args.rate, start_ns=int(stamp_ns[0]), end_ns=int(stamp_ns[-1]),
                 gaps_over_50ms=int((gaps > .05).sum()), largest_gap_sec=float(gaps.max()) if len(gaps) else 0.),
        noise=dict(preset=args.noise, scale=args.noise_scale, seed=args.seed, **noise.params),
        check_rest_specific_force=dict(samples=int(rest.sum()),
            norm_mean=float(np.linalg.norm(accel_clean[rest], axis=1).mean()) if rest.any() else None,
            norm_std=float(np.linalg.norm(accel_clean[rest], axis=1).std()) if rest.any() else None),
        check_capture_rotation_error_deg=checks,
        synthetic_notice='Synthetic IMU from simulation truth; not a physical sensor measurement.')
    np.savez_compressed(out.with_suffix('.npz'), stamp_ns=stamp_ns, gyro=np.array([r[4] for r in rows]),
        accel=np.array([r[5] for r in rows]), gyro_clean=gyro_clean, accel_clean=accel_clean,
        offset=np.array([r[1] for r in rows]), truth_position=np.array([r[6] for r in rows]),
        truth_quat_base=np.array([r[7] for r in rows]), pair_stamp_ns=origin_ns + np.round(pair_s * 1e9).astype(np.int64),
        pair_offset=pair_o, gyro_cov=np.full(len(s), noise.covariances(dt)[0]),
        accel_cov=np.full(len(s), noise.covariances(dt)[1]))
    out.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
