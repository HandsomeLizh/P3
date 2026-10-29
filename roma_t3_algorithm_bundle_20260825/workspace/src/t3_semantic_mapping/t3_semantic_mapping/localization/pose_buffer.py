from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose_math import interpolate_pose
from .types import TimedPose


@dataclass(frozen=True)
class PoseBufferQuery:
    sample: TimedPose
    method: str
    nearest_offset_sec: float
    bracket_span_sec: float


class TimedPoseBuffer:
    """Thread-safe, timestamp-ordered bounded pose buffer.

    Interpolation is allowed only when *both* bracket samples are within the
    configured time gate.  This prevents a smooth-looking interpolation from
    silently bridging a long LIO outage.
    """

    def __init__(self, *, max_size: int = 2000, max_age_sec: float = 300.0):
        if int(max_size) <= 0:
            raise ValueError("max_size must be positive")
        if float(max_age_sec) <= 0.0:
            raise ValueError("max_age_sec must be positive")
        self.max_size = int(max_size)
        self.max_age_sec = float(max_age_sec)
        self._timestamps: list[float] = []
        self._samples: list[TimedPose] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def time_range(self) -> Optional[tuple[float, float]]:
        with self._lock:
            if not self._timestamps:
                return None
            return self._timestamps[0], self._timestamps[-1]

    def add(self, sample: TimedPose) -> None:
        if not isinstance(sample, TimedPose):
            raise TypeError("sample must be TimedPose")
        with self._lock:
            position = bisect.bisect_left(self._timestamps, sample.timestamp)
            if (
                position < len(self._timestamps)
                and self._timestamps[position] == sample.timestamp
            ):
                self._samples[position] = sample
            else:
                self._timestamps.insert(position, sample.timestamp)
                self._samples.insert(position, sample)
            self._prune_locked()

    def _prune_locked(self) -> None:
        if not self._timestamps:
            return
        oldest_allowed = self._timestamps[-1] - self.max_age_sec
        first_kept = bisect.bisect_left(self._timestamps, oldest_allowed)
        excess = max(0, len(self._timestamps) - self.max_size)
        remove_count = max(first_kept, excess)
        if remove_count:
            del self._timestamps[:remove_count]
            del self._samples[:remove_count]

    def query(
        self,
        timestamp: float,
        *,
        max_time_offset_sec: float,
        allow_nearest: bool = True,
    ) -> Optional[PoseBufferQuery]:
        target = float(timestamp)
        gate = float(max_time_offset_sec)
        if not np.isfinite(target):
            raise ValueError("query timestamp must be finite")
        if gate < 0.0:
            raise ValueError("max_time_offset_sec cannot be negative")

        with self._lock:
            if not self._samples:
                return None
            position = bisect.bisect_left(self._timestamps, target)

            if (
                position < len(self._timestamps)
                and self._timestamps[position] == target
            ):
                return PoseBufferQuery(
                    sample=self._samples[position],
                    method="exact",
                    nearest_offset_sec=0.0,
                    bracket_span_sec=0.0,
                )

            if 0 < position < len(self._samples):
                left = self._samples[position - 1]
                right = self._samples[position]
                left_offset = target - left.timestamp
                right_offset = right.timestamp - target
                span = right.timestamp - left.timestamp
                if (
                    span > 0.0
                    and left_offset <= gate
                    and right_offset <= gate
                ):
                    ratio = left_offset / span
                    pose = interpolate_pose(left.pose, right.pose, ratio)
                    covariance = None
                    if left.covariance is not None and right.covariance is not None:
                        covariance = (
                            (1.0 - ratio) * left.covariance
                            + ratio * right.covariance
                        )
                    return PoseBufferQuery(
                        sample=TimedPose(target, pose, covariance),
                        method="interpolated",
                        nearest_offset_sec=float(
                            min(left_offset, right_offset)
                        ),
                        bracket_span_sec=float(span),
                    )

            if not allow_nearest:
                return None
            candidates = []
            if position > 0:
                candidates.append(self._samples[position - 1])
            if position < len(self._samples):
                candidates.append(self._samples[position])
            nearest = min(
                candidates,
                key=lambda sample: abs(sample.timestamp - target),
            )
            offset = nearest.timestamp - target
            if abs(offset) > gate:
                return None
            return PoseBufferQuery(
                sample=nearest,
                method="nearest",
                nearest_offset_sec=float(offset),
                bracket_span_sec=0.0,
            )
