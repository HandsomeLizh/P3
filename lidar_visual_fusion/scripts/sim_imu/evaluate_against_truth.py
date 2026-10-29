#!/usr/bin/env python3
"""Score a run against simulation truth; separate real terrain height changes from estimator jumps.

Truth: a build_sim_imu.py npz (truth_position / truth_quat_base on the original sensor time
axis; identical for any noise preset). Runs replayed with scripts/replay_bag.py are re-based
in time; the shift is read from <run>/replay_frames.csv automatically.

Two alignments:
  origin  the estimate starts at identity in its own frame; map it with the truth pose at the
          first output stamp. No fitting, so accumulated drift (e.g. pitch -> height) is shown.
  se3     one rigid fit (no scale) over all common samples: ATE, absorbs part of the drift.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def timeline_shift_ns(run):
    path = Path(run) / 'replay_frames.csv'
    if not path.exists():
        return 0
    rows = list(csv.reader(path.open()))
    return int(round(float(rows[1][2]) * 1e9)) - int(rows[1][1]) if len(rows) > 1 else 0


def rigid_fit(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1., 1., d]) @ u.T
    return r, cd - r @ cs


def metrics(stamps, aligned, ref, jump_m):
    error = aligned - ref
    step_est, step_ref = np.diff(aligned[:, 2]), np.diff(ref[:, 2])
    bad = np.flatnonzero(np.abs(step_est - step_ref) > jump_m)
    return dict(
        ate_rmse_m=round(float(np.sqrt((error ** 2).sum(1).mean())), 4),
        horizontal_rmse_m=round(float(np.sqrt((error[:, :2] ** 2).sum(1).mean())), 4),
        z_rmse_m=round(float(np.sqrt((error[:, 2] ** 2).mean())), 4),
        z_error_max_abs_m=round(float(np.abs(error[:, 2]).max()), 4),
        final_error_xyz_m=error[-1].round(3).tolist(),
        estimator_z_jumps=dict(threshold_m=jump_m, count=int(len(bad)),
            first=[dict(t_rel_s=round(float(stamps[k + 1] - stamps[0]), 1), est_dz=round(float(step_est[k]), 3),
                        truth_dz=round(float(step_ref[k]), 3)) for k in bad[:10]]),
        z_error_every_60s=[[round(float(x - stamps[0])), round(float(np.interp(x, stamps, error[:, 2])), 3)]
                           for x in np.arange(stamps[0], stamps[-1], 60.)])


def score(name, stamps, est, truth, jump_m):
    tt, tp, slerp = truth
    inside = (stamps >= tt[0]) & (stamps <= tt[-1])
    stamps, est = stamps[inside], est[inside]
    if len(stamps) < 10:
        return dict(name=name, common_samples=int(len(stamps)))
    ref = np.column_stack([np.interp(stamps, tt, tp[:, i]) for i in range(3)])
    r0 = slerp([stamps[0]])[0]
    origin = r0.apply(est - est[0]) + ref[0]
    r, t = rigid_fit(est, ref)
    se3 = est @ r.T + t
    return dict(name=name, common_samples=int(len(stamps)), span_s=round(float(stamps[-1] - stamps[0]), 1),
                path_m=round(float(np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1).sum()), 2),
                truth_z_range_m=round(float(np.ptp(ref[:, 2])), 3),
                origin_aligned=metrics(stamps, origin, ref, jump_m), se3_aligned=metrics(stamps, se3, ref, jump_m))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', required=True)
    a.add_argument('--truth', required=True, help='npz from build_sim_imu.py')
    a.add_argument('--jump', type=float, default=.08, help='|est dz - truth dz| per output step counted as a jump')
    a.add_argument('--out', default='', help='write the report here as well')
    args = a.parse_args()
    run = Path(args.run)
    data = np.load(args.truth)
    shift = timeline_shift_ns(run)
    tt = (data['stamp_ns'] + shift) * 1e-9
    truth = (tt, data['truth_position'], Slerp(tt, Rot.from_quat(data['truth_quat_base'])))
    results = []
    tum = np.loadtxt(run / 'trajectory_map.tum', ndmin=2)
    results.append(score('formal output', tum[:, 0], tum[:, 1:4], truth, args.jump))
    lidar_rows = [json.loads(line) for line in (run / 'lidar_metrics.jsonl').open() if line.strip()]
    lidar = np.array([[m['stamp_sec'], *m['position']] for m in lidar_rows if m.get('position')])
    results.append(score('raw VoxelMap odometry', lidar[:, 0], lidar[:, 1:4], truth, args.jump))
    modes = {}
    for m in lidar_rows:
        key = (m.get('imu') or {}).get('mode', 'unknown')
        modes[key] = modes.get(key, 0) + 1
    stats = {}
    stats_path = run / 'map_statistics.json'
    if stats_path.exists():
        s = json.loads(stats_path.read_text())
        stats = {k: s.get(k) for k in ('mapped_scans', 'dropped_scans', 'height_conflict_dropped_scans',
                                       'backpressure_dropped_scans')}
    report = dict(run=str(run), truth=args.truth, replay_time_shift_ns=shift, voxelmap_imu_modes=modes,
                  lidar_tracking={k: sum(1 for m in lidar_rows if m.get('tracking_reason') == k)
                                  for k in sorted({m.get('tracking_reason') for m in lidar_rows})},
                  mapping=stats, results=results)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
