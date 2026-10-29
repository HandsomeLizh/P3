#!/usr/bin/env python3
"""Partial-truth check of one run: which height is real, LiDAR odometry or formal output?

Truth: build_sim_imu.py npz (sensor time axis) over the recorded window, plus the final
capture pose from the relay's latest_sensor_meta.json. Alignment odom<-world: chordal-mean
rotation of R_out R_truth^T and mean translation over the window, where the output was the
EKF (checked: output - LiDAR offset constant). Extrapolation to the end is reported with
the tilt uncertainty of that alignment (first vs second half of the window).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_imu  # noqa: E402

run, truth_npz = Path(sys.argv[1]), sys.argv[2]
truth = np.load(truth_npz)
tt = truth['stamp_ns'] * 1e-9
tp, tq = truth['truth_position'], truth['truth_quat_base']
tum = np.loadtxt(run / 'trajectory_map.tum', ndmin=2)
lidar = np.array([[m['stamp_sec'], *m['position']] for m in
                  (json.loads(x) for x in (run / 'lidar_metrics.jsonl').open() if x.strip())])
t0 = tum[0, 0]
interp = lambda t, x, v: np.column_stack([np.interp(t, x, v[:, i]) for i in range(v.shape[1])])


def alignment(mask):
    s = tum[mask, 0]
    r_out = Rot.from_quat(tum[mask, 4:8])
    r_truth = Rot.from_quat(interp(s, tt, tq))           # slerp-free: fine at 100 Hz truth
    m = sum((r_out[i] * r_truth[i].inv()).as_matrix() for i in range(len(s)))
    u, _, vt = np.linalg.svd(m)
    r = Rot.from_matrix(u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt)
    trans = (tum[mask, 1:4] - r.apply(interp(s, tt, tp))).mean(0)
    return r, trans


window = (tum[:, 0] >= tt[0] + 1) & (tum[:, 0] <= tt[-1] - 1)
half = tum[window, 0][len(tum[window]) // 2]
r, trans = alignment(window)
r1, _ = alignment(window & (tum[:, 0] < half))
r2, _ = alignment(window & (tum[:, 0] >= half))
tilt_spread = np.degrees(np.arccos(np.clip((r1.apply([0, 0, 1]) @ r2.apply([0, 0, 1])), -1, 1)))

def truth_in_odom(t):
    return r.apply(interp(np.atleast_1d(t), tt, tp))[0] + trans if np.ndim(t) == 0 else r.apply(interp(t, tt, tp)) + trans

print(f'run {run.name}: truth window t+{tt[0]-t0:.0f}..t+{tt[-1]-t0:.0f} s of {tum[-1,0]-t0:.0f} s; '
      f'alignment tilt spread (half vs half) {tilt_spread:.3f} deg')
s = tum[window, 0]
ref = truth_in_odom(s)
out = tum[window, 1:4]
lid = interp(s, lidar[:, 0], lidar[:, 1:4])
print(f'truth in window: z range {np.ptp(ref[:,2]):.3f} m, horizontal path {np.linalg.norm(np.diff(ref[:,:2],axis=0),axis=1).sum():.1f} m')
for name, est in (('formal output', out), ('raw LiDAR odometry', lid)):
    e = est - ref
    # Remove the constant part so LiDAR (different origin offset) is judged on drift only.
    e0 = e - e[:20].mean(0)
    print(f'  {name:20s} z drift over window: {e0[-1,2]:+.3f} m (max |.| {np.abs(e0[:,2]).max():.3f}); '
          f'horizontal drift {np.linalg.norm(e0[-1,:2]):.3f} m')
print('  truth z change across window %+.3f m | output %+.3f | LiDAR %+.3f' % (
    ref[-1, 2] - ref[0, 2], out[-1, 2] - out[0, 2], lid[-1, 2] - lid[0, 2]))

meta = json.loads((run / 'latest_sensor_meta.json').read_text())
parsed = gt_imu.parse_meta(json.dumps(meta))
end_truth = r.apply(parsed[2]) + trans
end_out, end_lid = tum[-1, 1:4], lidar[-1, 1:4]
start = tum[window, 0][0]
start_truth = truth_in_odom(start)
start_out = tum[window][0, 1:4]
start_lid = interp(np.array([start]), lidar[:, 0], lidar[:, 1:4])[0]
dist = np.linalg.norm(end_truth[:2] - start_truth[:2])
print(f'end of run (final capture pose; ~{dist:.1f} m horizontally from the window start, '
      f'tilt uncertainty -> +-{np.radians(tilt_spread)*dist:.3f} m in z):')
print(f'  truth  z change since window start {end_truth[2]-start_truth[2]:+.3f} m')
print(f'  output z change since window start {end_out[2]-start_out[2]:+.3f} m  (error {end_out[2]-start_out[2]-(end_truth[2]-start_truth[2]):+.3f})')
print(f'  LiDAR  z change since window start {end_lid[2]-start_lid[2]:+.3f} m  (error {end_lid[2]-start_lid[2]-(end_truth[2]-start_truth[2]):+.3f})')
print(f'  horizontal error at end: output {np.linalg.norm(end_out[:2]-end_truth[:2]):.2f} m, '
      f'LiDAR {np.linalg.norm(end_lid[:2]-end_truth[:2] - (start_lid[:2]-start_out[:2])):.2f} m')
# Which source does the output follow? output minus LiDAR, every 60 s and around switches
d = tum[:, 1:4] - interp(tum[:, 0], lidar[:, 0], lidar[:, 1:4])
print('output minus LiDAR z every 120 s:', ' '.join(f'{x-t0:.0f}:{np.interp(x, tum[:,0], d[:,2]):+.2f}'
                                                  for x in np.arange(t0, tum[-1, 0], 120)))
