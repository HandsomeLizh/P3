"""Synthetic IMU from UE simulation ground truth, on the sensor (LiDAR/camera) time axis.

Truth in:
  /car/pose            UE body pose, ~20 Hz, stamped with the Orin TCP receive time.
  /sensor/meta         per capture batch: exact UE pose at the capture instant (JSON).
  /sensor/batch_status per capture batch: batch_id and the sensor header stamp_ns.
IMU out: base_link gyro (rad/s) and specific force (m/s^2) at a fixed rate.

The /car/pose receive time lags the sensor header by a session-dependent, slowly drifting
offset (0.5-0.8 s measured). It is estimated online by finding when the pose stream passed
through each capture pose. No pose, offset or truth value is ever sent to the estimators.

Not a physical IMU model: no scale factor, misalignment, temperature or vibration.
Results obtained with it must be labelled as synthetic-IMU simulation experiments.
"""
import calendar
import datetime
import json
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# UE body axes -> ROS base_link (FLU); profile telemetry_motion.orientation_child_from_base_rotation.
UE_CHILD_FROM_BASE = Rot.from_matrix([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
GRAVITY = 9.81  # must equal t3_voxelmap OptionalImu gravity (G_m_s2)


def base_rotation(quat_xyzw):
    """World-from-base_link rotation(s) from right-handed UE pose quaternion(s)."""
    return Rot.from_quat(quat_xyzw) * UE_CHILD_FROM_BASE


def meta_time_ns(text):
    date, _, fraction = text.partition('.')
    seconds = calendar.timegm(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timetuple())
    return seconds * 10**9 + int((fraction or '0').ljust(9, '0')[:9])


def parse_meta(text):
    """(batch_id, meta_time_ns, position_m, quat_xyzw) or None for malformed metadata.

    UE metadata is left-handed centimetres; /car/telemetry documents the same mapping:
    x, -y, z and quaternion (-qx, qy, -qz, qw).
    """
    try:
        data = json.loads(text)
        position, rotation = data['position'], data['rotation']
        p = np.array([position['x'], -position['y'], position['z']], dtype=float) / 100.
        q = np.array([-rotation['qx'], rotation['qy'], -rotation['qz'], rotation['qw']], dtype=float)
        result = int(data['batch_id']), meta_time_ns(data['timestamp']), p, q / np.linalg.norm(q)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return None
    return result if np.isfinite(result[2]).all() and np.isfinite(result[3]).all() else None


def parse_batch_status(text):
    """(batch_id, stamp_ns) or None when the capture build does not report the batch stamp."""
    try:
        data = json.loads(text)
        return int(data['batch_id']), int(data['stamp_ns'])
    except (ValueError, KeyError, TypeError):
        return None


def drop_stale_repeats(t, p, q, hold=.1, live=False):
    """Keep-mask removing repeated payloads followed by a different pose within `hold` s.

    While moving such a repeat is a stale resend, so its receive time misplaces the pose.
    Long runs of identical poses are a genuine stop and are kept. live=True treats a
    trailing repeat as unresolved until it is `hold` older than the newest sample.
    """
    t = np.asarray(t, float)
    same = np.r_[False, np.all(np.diff(p, axis=0) == 0, axis=1) & np.all(np.diff(q, axis=0) == 0, axis=1)]
    distinct = np.flatnonzero(~same)
    following = np.searchsorted(distinct, np.arange(len(t)), side='right')
    has_next = following < len(distinct)
    next_time = np.where(has_next, t[distinct[np.minimum(following, len(distinct) - 1)]], np.inf)
    if live and len(t):
        next_time = np.where(has_next, next_time, np.where(t[-1] - t >= hold, np.inf, t))
    return ~same | (next_time - t >= hold)


def _poly(dt, w, y, degree):
    A = np.vander(dt, degree + 1, increasing=True)
    Aw = A * w[:, None]
    coefficients = np.linalg.lstsq(A.T @ Aw, Aw.T @ y, rcond=None)[0]
    return np.vstack([coefficients, np.zeros((2 - degree, y.shape[1]))])


def kinematics(t, p, rotations, tau, back=.5, fwd=.2, max_gap=.3):
    """Locally weighted quadratic fit of the truth around tau (truth time axis).

    Returns position, velocity, acceleration (world) and rotation, body rate, or None
    when tau is not bracketed by truth samples or the stream has a gap.
    """
    lo, hi = np.searchsorted(t, tau - back), np.searchsorted(t, tau + fwd, side='right')
    times = t[lo:hi]
    if len(times) == 0 or times[0] > tau or times[-1] < tau:
        return None
    edges = np.r_[tau - back, times, tau + fwd]
    if np.max(np.diff(edges)) > max_gap:
        return None
    dt = times - tau
    width = np.where(dt < 0, back, fwd)
    w = (1. - np.clip(np.abs(dt) / width, 0., 1.) ** 3) ** 3 + 1e-6
    distinct = len(np.unique(times))
    degree = 2 if distinct >= 4 else (1 if distinct >= 2 else 0)
    cp = _poly(dt, w, p[lo:hi], degree)
    window = rotations[lo:hi]
    reference = window[int(np.argmin(np.abs(dt)))]
    r = (reference.inv() * window).as_rotvec()
    cr = _poly(dt, w, r, degree)
    omega = cr[1] - .5 * np.cross(cr[0], cr[1])  # right Jacobian, first order
    return dict(position=cp[0], velocity=cp[1], acceleration=2. * cp[2],
                rotation=reference * Rot.from_rotvec(cr[0]), omega=omega)


def imu_values(state, gravity=GRAVITY):
    """Body gyro and specific force; at rest a level FLU body reads +g on z."""
    force = state['rotation'].inv().apply(state['acceleration'] + np.array([0., 0., gravity]))
    return state['omega'].copy(), force


def match_capture(t, p, rotations, position, rotation, lo, hi,
                  pos_tol=2e-3, rot_tol=3e-3, min_step_pos=1.5e-3, min_step_rot=2e-3):
    """Truth-stream time at which the capture pose was passed (sub-sample), or None.

    Only segments that translate or rotate measurably are used, so a stop (where the
    time is ambiguous) never produces a pair.
    """
    a0, a1 = np.searchsorted(t, lo), np.searchsorted(t, hi, side='right')
    best = None
    for a in range(max(a0, 0), min(a1, len(t)) - 1):
        segment = p[a + 1] - p[a]
        length2 = float(segment @ segment)
        step = (rotations[a].inv() * rotations[a + 1]).as_rotvec()
        angle2 = float(step @ step)
        if length2 >= min_step_pos ** 2:
            u = float((position - p[a]) @ segment / length2)
        elif angle2 >= min_step_rot ** 2:
            u = float((rotations[a].inv() * rotation).as_rotvec() @ step / angle2)
        else:
            continue
        if not 0. <= u <= 1.:
            continue
        error_pos = float(np.linalg.norm(p[a] + u * segment - position))
        error_rot = float(((rotations[a] * Rot.from_rotvec(u * step)).inv() * rotation).magnitude())
        if error_pos <= pos_tol and error_rot <= rot_tol:
            score = error_pos / pos_tol + error_rot / rot_tol
            if best is None or score < best[0]:
                best = (score, t[a] + u * (t[a + 1] - t[a]))
    return None if best is None else best[1]


class OffsetModel:
    """offset(s) = truth receive time - sensor stamp, robust linear in a sliding window.

    Before enough moving captures are matched the prior is used. at() is slew-limited so a
    re-estimate cannot jump the emitted time axis.
    """

    def __init__(self, prior=.65, window=60., min_pairs=5, slew=.02, max_slope=2e-3,
                 plausible=(0., 2.5)):
        self.prior, self.window, self.min_pairs = float(prior), float(window), int(min_pairs)
        self.slew, self.max_slope, self.plausible = float(slew), float(max_slope), plausible
        self.pairs = deque()
        self.fit = None
        self.current = None
        self.current_s = None
        self.rejected = 0

    def add(self, s, offset):
        if not (self.plausible[0] <= offset <= self.plausible[1]) or not np.isfinite([s, offset]).all():
            self.rejected += 1
            return False
        if self.pairs and s <= self.pairs[-1][0]:
            self.pairs.clear()  # capture restart re-anchors sensor stamps
            self.fit = None
        self.pairs.append((float(s), float(offset)))
        while self.pairs and self.pairs[-1][0] - self.pairs[0][0] > self.window:
            self.pairs.popleft()
        self.refit()
        return True

    def refit(self):
        if len(self.pairs) < self.min_pairs:
            self.fit = None
            return
        s, o = np.array(self.pairs).T
        reference = s[-1]
        inliers = np.ones(len(s), bool)
        slope = 0.
        for _ in range(2):
            ss, oo = s[inliers], o[inliers]
            if np.ptp(ss) >= 15. and len(ss) >= 5:
                i, j = np.triu_indices(len(ss), 1)
                usable = ss[j] - ss[i] > 1e-6
                slope = float(np.median((oo[j] - oo[i])[usable] / (ss[j] - ss[i])[usable]))
                slope = float(np.clip(slope, -self.max_slope, self.max_slope))
            intercept = float(np.median(oo - slope * (ss - reference)))
            residual = o - intercept - slope * (s - reference)
            mad = 1.4826 * float(np.median(np.abs(residual[inliers])))
            inliers = np.abs(residual) <= max(3. * mad, .005)
        if inliers.sum() >= self.min_pairs:
            self.fit = dict(intercept=intercept, slope=slope, reference=reference, mad=mad,
                            pairs=len(s), inliers=int(inliers.sum()))

    def target(self, s):
        if self.fit is None:
            return self.prior
        return self.fit['intercept'] + self.fit['slope'] * (s - self.fit['reference'])

    def at(self, s):
        """Slew-limited offset for monotonically increasing emission times s."""
        target = self.target(s)
        if self.current is None:
            self.current, self.current_s = target, s
            return target
        step = self.slew * max(0., s - self.current_s)
        self.current += float(np.clip(target - self.current, -step, step))
        self.current_s = s
        return self.current

    @property
    def locked(self):
        return self.fit is not None

    def status(self):
        return dict(locked=self.locked, offset=self.current, prior=self.prior,
                    fit=self.fit, rejected=self.rejected)


class NoiseModel:
    """White noise plus bias random walk; densities are continuous-time SI values.

    'mems' is a consumer-MEMS order of magnitude (ICM-20948-class datasheet noise
    densities; random walks and initial biases are engineering assumptions). Verify
    against the actual sensor datasheet before quoting it as a specific device.
    """
    PRESETS = {
        'none': dict(gyro_density=0., accel_density=0., gyro_walk=0., accel_walk=0.,
                     gyro_bias=0., accel_bias=0.),
        'mems': dict(gyro_density=2.6e-4, accel_density=2.3e-3, gyro_walk=2e-5, accel_walk=4e-4,
                     gyro_bias=2e-3, accel_bias=2e-2),
    }

    def __init__(self, preset='mems', scale=1., seed=0):
        if preset not in self.PRESETS or not np.isfinite(scale) or scale < 0:
            raise ValueError('Unknown noise preset or invalid scale')
        self.params = {k: v * scale for k, v in self.PRESETS[preset].items()}
        self.preset, self.scale = preset, float(scale)
        self.random = np.random.default_rng(seed)
        self.bias_g = self.random.normal(0., self.params['gyro_bias'], 3)
        self.bias_a = self.random.normal(0., self.params['accel_bias'], 3)

    def apply(self, gyro, accel, dt):
        p = self.params
        self.bias_g += self.random.normal(0., p['gyro_walk'] * np.sqrt(dt), 3)
        self.bias_a += self.random.normal(0., p['accel_walk'] * np.sqrt(dt), 3)
        gyro = gyro + self.bias_g + self.random.normal(0., p['gyro_density'] / np.sqrt(dt), 3)
        accel = accel + self.bias_a + self.random.normal(0., p['accel_density'] / np.sqrt(dt), 3)
        return gyro, accel

    def covariances(self, dt):
        """Per-sample white-noise variances for the Imu message (never zero)."""
        return (max((self.params['gyro_density'] / np.sqrt(dt)) ** 2, 1e-8),
                max((self.params['accel_density'] / np.sqrt(dt)) ** 2, 1e-6))
