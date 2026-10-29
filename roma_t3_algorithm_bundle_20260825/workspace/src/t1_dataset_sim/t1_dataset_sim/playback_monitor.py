"""Lossless dataset-playback accounting independent of ROS."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class PlaybackMonitor:
    """Track publication and mapper completion by acquisition timestamp."""

    def __init__(
        self,
        *,
        expected_frame_count: int,
        expected_first_stamp_ns: int,
        expected_last_stamp_ns: int,
        requested_playback_rate: float,
        lossless_playback: bool,
    ) -> None:
        if expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be positive")
        if expected_last_stamp_ns < expected_first_stamp_ns:
            raise ValueError("expected frame timestamps are not monotonic")
        if (
            not math.isfinite(requested_playback_rate)
            or requested_playback_rate <= 0.0
        ):
            raise ValueError("requested_playback_rate must be positive")
        self.expected_frame_count = int(expected_frame_count)
        self.expected_first_stamp_ns = int(expected_first_stamp_ns)
        self.expected_last_stamp_ns = int(expected_last_stamp_ns)
        self.requested_playback_rate = float(requested_playback_rate)
        self.lossless_playback = bool(lossless_playback)
        self.frame_attempt_count = 0
        self.frame_publish_failure_count = 0
        self.published_by_stamp: Counter = Counter()
        self.acknowledged_by_stamp: Counter = Counter()
        self.published_frame_count = 0
        self.acknowledged_frame_count = 0
        self.unexpected_notification_count = 0
        self.publication_finished = False
        self.failure_reason: Optional[str] = None
        self._first_publish_wall: Optional[float] = None
        self._last_publish_wall: Optional[float] = None
        self._last_acknowledgement_wall: Optional[float] = None
        self._schedule_lateness_count = 0
        self._schedule_lateness_sum = 0.0
        self._schedule_lateness_max = 0.0

    def record_frame_attempt(
        self,
        stamp_ns: int,
        *,
        published: bool,
        wall_time: float,
        schedule_lateness_sec: float,
    ) -> None:
        lateness = float(schedule_lateness_sec)
        if not math.isfinite(lateness):
            raise ValueError("schedule_lateness_sec must be finite")
        self.frame_attempt_count += 1
        self._schedule_lateness_count += 1
        self._schedule_lateness_sum += max(0.0, lateness)
        self._schedule_lateness_max = max(
            self._schedule_lateness_max, max(0.0, lateness)
        )
        if not published:
            self.frame_publish_failure_count += 1
            return
        stamp_ns = int(stamp_ns)
        wall_time = float(wall_time)
        self.published_by_stamp[stamp_ns] += 1
        self.published_frame_count += 1
        if self._first_publish_wall is None:
            self._first_publish_wall = wall_time
        self._last_publish_wall = wall_time

    def acknowledge(self, stamp_ns: int, wall_time: float) -> bool:
        stamp_ns = int(stamp_ns)
        if (
            self.acknowledged_by_stamp[stamp_ns]
            >= self.published_by_stamp[stamp_ns]
        ):
            self.unexpected_notification_count += 1
            return False
        self.acknowledged_by_stamp[stamp_ns] += 1
        self.acknowledged_frame_count += 1
        self._last_acknowledgement_wall = float(wall_time)
        return True

    def mark_publication_finished(self) -> None:
        self.publication_finished = True

    def mark_failed(self, reason: str) -> None:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("failure reason cannot be empty")
        if self.failure_reason is None:
            self.failure_reason = reason

    @property
    def unacknowledged_frame_count(self) -> int:
        return self.published_frame_count - self.acknowledged_frame_count

    @property
    def is_complete(self) -> bool:
        return (
            self.failure_reason is None
            and self.publication_finished
            and self.frame_publish_failure_count == 0
            and self.published_frame_count == self.expected_frame_count
            and self.unacknowledged_frame_count == 0
        )

    def _unacknowledged_entries(self):
        entries = []
        for stamp_ns in sorted(self.published_by_stamp):
            missing = (
                self.published_by_stamp[stamp_ns]
                - self.acknowledged_by_stamp[stamp_ns]
            )
            if missing > 0:
                entries.append(
                    {"stamp_ns": int(stamp_ns), "count": int(missing)}
                )
        return entries

    def summary(self) -> dict:
        expected_sensor_span_sec = (
            self.expected_last_stamp_ns - self.expected_first_stamp_ns
        ) * 1e-9
        ideal_wall_span_sec = (
            expected_sensor_span_sec / self.requested_playback_rate
        )
        wall_publish_span_sec = None
        if (
            self._first_publish_wall is not None
            and self._last_publish_wall is not None
        ):
            wall_publish_span_sec = max(
                0.0, self._last_publish_wall - self._first_publish_wall
            )
        effective_rate = None
        if wall_publish_span_sec is not None and wall_publish_span_sec > 0.0:
            effective_rate = expected_sensor_span_sec / wall_publish_span_sec
        completion_wall_span_sec = None
        if (
            self._first_publish_wall is not None
            and self._last_acknowledgement_wall is not None
        ):
            completion_wall_span_sec = max(
                0.0,
                self._last_acknowledgement_wall - self._first_publish_wall,
            )
        if self.failure_reason is not None:
            status = "failed"
        elif self.is_complete:
            status = "complete"
        elif self.publication_finished:
            status = "incomplete"
        else:
            status = "running"
        lateness_mean = (
            self._schedule_lateness_sum / self._schedule_lateness_count
            if self._schedule_lateness_count
            else None
        )
        unacknowledged = self._unacknowledged_entries()
        return {
            "schema_version": 1,
            "status": status,
            "lossless_playback": self.lossless_playback,
            "requested_playback_rate_x": self.requested_playback_rate,
            "expected_frame_count": self.expected_frame_count,
            "frame_attempt_count": self.frame_attempt_count,
            "frame_publish_failure_count": self.frame_publish_failure_count,
            "published_frame_count": self.published_frame_count,
            "acknowledged_frame_count": self.acknowledged_frame_count,
            "unacknowledged_frame_count": self.unacknowledged_frame_count,
            "unexpected_completion_notifications": (
                self.unexpected_notification_count
            ),
            "publication_finished": self.publication_finished,
            "failure_reason": self.failure_reason,
            "expected_sensor_span_sec": expected_sensor_span_sec,
            "ideal_wall_publish_span_sec": ideal_wall_span_sec,
            "actual_wall_publish_span_sec": wall_publish_span_sec,
            "effective_playback_rate_x": effective_rate,
            "wall_first_publish_to_last_ack_sec": (
                completion_wall_span_sec
            ),
            "frame_schedule_lateness_sec": {
                "count": self._schedule_lateness_count,
                "mean": lateness_mean,
                "max": (
                    self._schedule_lateness_max
                    if self._schedule_lateness_count
                    else None
                ),
            },
            "unacknowledged": unacknowledged[:100],
            "unacknowledged_list_truncated": len(unacknowledged) > 100,
            "complete_input_consumption": self.is_complete,
        }

    @staticmethod
    def write_summary(path: Path, payload: dict) -> None:
        _atomic_json_write(Path(path), payload)
