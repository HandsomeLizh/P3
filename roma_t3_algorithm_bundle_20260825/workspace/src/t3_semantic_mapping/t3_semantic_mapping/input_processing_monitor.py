"""Bounded, ROS-independent accounting for synchronized mapper inputs."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Mapping, Optional


_SOURCES = ("left", "right", "semantic")
_TERMINAL_OUTCOMES = {
    "completed",
    "tracking_failure",
    "localization_unavailable",
    "processing_exception",
    "stereo_skew_rejected",
    "reentrant_rejected",
}
_EARLY_REJECTIONS = {"stereo_skew_rejected", "reentrant_rejected"}
_PROCESSING_STAGES = (
    "input_decode",
    "visual_tracking",
    "localization_and_pose",
    "map_update_and_publish",
    "callback_bookkeeping",
)

_FORMAL_COLD_START_VALID_FRAMES = 1

_FORMAL_SAMPLE_SIZE = 10
_FORMAL_SAMPLE_SEED = 20260821


def _percentile(
    values: Deque[float], percentile: float
) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        int(math.ceil(float(percentile) * len(ordered))) - 1,
    )
    return float(ordered[index])


def _recent_statistics(values: Deque[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "count": count,
        "median": float(median),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


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


def _finite_non_negative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class CallbackStageTimer:
    """Classify one accepted callback using explicit monotonic timestamps."""

    def __init__(
        self,
        wall_time: float,
        initial_stage: str = "input_decode",
    ) -> None:
        self._validate_stage(initial_stage)
        self._active_stage: Optional[str] = initial_stage
        self._last_wall_time = _finite_non_negative(
            wall_time, name="wall_time"
        )
        self._durations = {
            stage: 0.0 for stage in _PROCESSING_STAGES
        }

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in _PROCESSING_STAGES:
            raise ValueError(f"unknown processing stage: {stage}")

    def _close_active_interval(self, wall_time: float) -> None:
        if self._active_stage is None:
            raise RuntimeError("callback stage timer is already finished")
        current = _finite_non_negative(wall_time, name="wall_time")
        if current < self._last_wall_time:
            raise ValueError("callback stage timestamps must be monotonic")
        self._durations[self._active_stage] += (
            current - self._last_wall_time
        )
        self._last_wall_time = current

    def switch(self, stage: str, wall_time: float) -> None:
        """Close the active stage and begin ``stage`` at ``wall_time``."""

        self._validate_stage(stage)
        self._close_active_interval(wall_time)
        self._active_stage = stage

    def finish(self, wall_time: float) -> Dict[str, float]:
        """Close the timer and return only stages reached by this callback."""

        self._close_active_interval(wall_time)
        self._active_stage = None
        return {
            stage: float(self._durations[stage])
            for stage in _PROCESSING_STAGES
            if self._durations[stage] > 0.0
        }


class InputProcessingMonitor:
    """Account for every synchronized callback without retaining payloads.

    Only a bounded duration window is retained, so this monitor is safe for a
    long-running robot. Sensor timestamps and counters are plain integers and
    can be tested without ROS.
    """

    def __init__(self, duration_window_size: int = 2048) -> None:
        if duration_window_size <= 0:
            raise ValueError("duration_window_size must be positive")
        self.delivered_counts: Dict[str, int] = {
            source: 0 for source in _SOURCES
        }
        self.latest_stamp_ns: Dict[str, Optional[int]] = {
            source: None for source in _SOURCES
        }
        self.synchronized_count = 0
        self.processing_started_count = 0
        self.terminal_outcomes: Counter = Counter()
        self.completion_notifications_published = 0
        self.first_synchronized_stamp_ns: Optional[int] = None
        self.last_synchronized_stamp_ns: Optional[int] = None
        self._first_synchronized_wall: Optional[float] = None
        self._last_terminal_wall: Optional[float] = None
        self._duration_count = 0
        self._duration_sum = 0.0
        self._duration_max = 0.0
        self._recent_durations: Deque[float] = deque(
            maxlen=duration_window_size
        )
        self._classified_callback_count = 0
        self._stage_duration_count: Dict[str, int] = {
            stage: 0 for stage in _PROCESSING_STAGES
        }
        self._stage_duration_sum: Dict[str, float] = {
            stage: 0.0 for stage in _PROCESSING_STAGES
        }
        self._stage_duration_max: Dict[str, float] = {
            stage: 0.0 for stage in _PROCESSING_STAGES
        }
        self._recent_stage_durations: Dict[str, Deque[float]] = {
            stage: deque(maxlen=duration_window_size)
            for stage in _PROCESSING_STAGES
        }
        self._valid_frame_count = 0
        self._valid_pose_latency_sum = 0.0
        self._valid_pose_latency_max = 0.0
        self._valid_map_latency_sum = 0.0
        self._valid_map_latency_max = 0.0
        self._valid_end_to_end_latency_sum = 0.0
        self._valid_end_to_end_latency_max = 0.0
        self._recent_valid_pose_latencies: Deque[float] = deque(
            maxlen=duration_window_size
        )
        self._recent_valid_map_latencies: Deque[float] = deque(
            maxlen=duration_window_size
        )
        self._recent_valid_end_to_end_latencies: Deque[float] = deque(
            maxlen=duration_window_size
        )
        self._formal_sampling_population_count = 0
        self._formal_sample_random = random.Random(_FORMAL_SAMPLE_SEED)
        self._formal_sample_reservoir = []

    def record_delivered(self, source: str, stamp_ns: Optional[int]) -> None:
        if source not in self.delivered_counts:
            raise ValueError(f"unknown input source: {source}")
        self.delivered_counts[source] += 1
        self.latest_stamp_ns[source] = (
            None if stamp_ns is None else int(stamp_ns)
        )

    def record_synchronized(self, stamp_ns: int, wall_time: float) -> None:
        stamp_ns = int(stamp_ns)
        wall_time = float(wall_time)
        self.synchronized_count += 1
        if self.first_synchronized_stamp_ns is None:
            self.first_synchronized_stamp_ns = stamp_ns
            self._first_synchronized_wall = wall_time
        self.last_synchronized_stamp_ns = stamp_ns

    def record_processing_started(self) -> None:
        self.processing_started_count += 1

    def record_terminal(
        self,
        outcome: str,
        wall_time: float,
        duration_sec: Optional[float] = None,
        stage_durations_sec: Optional[Mapping[str, float]] = None,
        sensor_stamp_ns: Optional[int] = None,
        pose_publish_latency_sec: Optional[float] = None,
        map_update_latency_sec: Optional[float] = None,
        localization_solution_start_monotonic_ns: Optional[int] = None,
        pose_publish_completion_monotonic_ns: Optional[int] = None,
        map_update_start_monotonic_ns: Optional[int] = None,
        map_publish_completion_monotonic_ns: Optional[int] = None,
    ) -> None:
        if outcome not in _TERMINAL_OUTCOMES:
            raise ValueError(f"unknown terminal outcome: {outcome}")
        duration = None
        if duration_sec is not None:
            duration = _finite_non_negative(
                duration_sec, name="duration_sec"
            )
        stages = None
        if stage_durations_sec is not None:
            if duration is None:
                raise ValueError(
                    "stage durations require a total duration_sec"
                )
            stages = {}
            for stage, value in stage_durations_sec.items():
                if stage not in _PROCESSING_STAGES:
                    raise ValueError(
                        f"unknown processing stage: {stage}"
                    )
                stages[stage] = _finite_non_negative(
                    value, name=f"stage duration {stage}"
                )
            classified_duration = sum(stages.values())
            tolerance = max(1e-9, duration * 1e-9)
            if abs(classified_duration - duration) > tolerance:
                raise ValueError(
                    "stage durations must classify the complete callback: "
                    f"stages={classified_duration}, total={duration}"
                )

        pose_latency = None
        map_latency = None
        if (
            pose_publish_latency_sec is not None
            or map_update_latency_sec is not None
        ):
            if outcome != "completed" or duration is None:
                raise ValueError(
                    "valid-frame timing requires a completed callback with "
                    "duration_sec"
                )
            if (
                pose_publish_latency_sec is None
                or map_update_latency_sec is None
            ):
                raise ValueError(
                    "pose and map valid-frame timings must be recorded together"
                )
            pose_latency = _finite_non_negative(
                pose_publish_latency_sec,
                name="pose_publish_latency_sec",
            )
            map_latency = _finite_non_negative(
                map_update_latency_sec,
                name="map_update_latency_sec",
            )
            if pose_latency > duration:
                raise ValueError(
                    "pose publication cannot finish after callback termination"
                )
            if map_latency > duration:
                raise ValueError(
                    "map update cannot exceed complete callback duration"
                )

        self.terminal_outcomes[outcome] += 1
        self._last_terminal_wall = float(wall_time)
        if duration is None:
            return
        self._duration_count += 1
        self._duration_sum += duration
        self._duration_max = max(self._duration_max, duration)
        self._recent_durations.append(duration)
        if stages is not None:
            self._classified_callback_count += 1
            for stage, value in stages.items():
                self._stage_duration_count[stage] += 1
                self._stage_duration_sum[stage] += value
                self._stage_duration_max[stage] = max(
                    self._stage_duration_max[stage], value
                )
                self._recent_stage_durations[stage].append(value)
        if pose_latency is not None and map_latency is not None:
            self._valid_frame_count += 1
            self._valid_pose_latency_sum += pose_latency
            self._valid_pose_latency_max = max(
                self._valid_pose_latency_max, pose_latency
            )
            self._valid_map_latency_sum += map_latency
            self._valid_map_latency_max = max(
                self._valid_map_latency_max, map_latency
            )
            self._valid_end_to_end_latency_sum += duration
            self._valid_end_to_end_latency_max = max(
                self._valid_end_to_end_latency_max, duration
            )
            self._recent_valid_pose_latencies.append(pose_latency)
            self._recent_valid_map_latencies.append(map_latency)
            self._recent_valid_end_to_end_latencies.append(duration)
            if self._valid_frame_count > _FORMAL_COLD_START_VALID_FRAMES:
                self._formal_sampling_population_count += 1
                measurement = (
                    self._valid_frame_count,
                    None if sensor_stamp_ns is None else int(sensor_stamp_ns),
                    pose_latency,
                    map_latency,
                    duration,
                    localization_solution_start_monotonic_ns,
                    pose_publish_completion_monotonic_ns,
                    map_update_start_monotonic_ns,
                    map_publish_completion_monotonic_ns,
                )
                if len(self._formal_sample_reservoir) < _FORMAL_SAMPLE_SIZE:
                    self._formal_sample_reservoir.append(measurement)
                else:
                    slot = self._formal_sample_random.randrange(
                        self._formal_sampling_population_count
                    )
                    if slot < _FORMAL_SAMPLE_SIZE:
                        self._formal_sample_reservoir[slot] = measurement

    def record_completion_notification(self) -> None:
        self.completion_notifications_published += 1

    @property
    def terminal_count(self) -> int:
        return int(sum(self.terminal_outcomes.values()))

    @property
    def rejected_stereo_skew_count(self) -> int:
        return int(self.terminal_outcomes["stereo_skew_rejected"])

    def summary(
        self,
        *,
        processing_in_progress: bool,
        valid_pose_count: int,
        tracking_or_processing_failure_count: int,
        map_revision: int,
    ) -> dict:
        sensor_span_sec = None
        if (
            self.first_synchronized_stamp_ns is not None
            and self.last_synchronized_stamp_ns is not None
        ):
            sensor_span_sec = max(
                0.0,
                (
                    self.last_synchronized_stamp_ns
                    - self.first_synchronized_stamp_ns
                )
                * 1e-9,
            )
        wall_span_sec = None
        if (
            self._first_synchronized_wall is not None
            and self._last_terminal_wall is not None
        ):
            wall_span_sec = max(
                0.0,
                self._last_terminal_wall - self._first_synchronized_wall,
            )
        effective_rate = None
        if sensor_span_sec is not None and wall_span_sec is not None:
            if wall_span_sec > 0.0:
                effective_rate = sensor_span_sec / wall_span_sec

        early_rejections = sum(
            self.terminal_outcomes[outcome]
            for outcome in _EARLY_REJECTIONS
        )
        terminal_count = self.terminal_count
        callbacks_closed = (
            terminal_count == self.synchronized_count
            and not processing_in_progress
        )
        notifications_complete = (
            self.completion_notifications_published
            == self.synchronized_count
        )
        delivered_values = list(self.delivered_counts.values())
        delivered_streams_equal = len(set(delivered_values)) <= 1
        all_delivered_synchronized = all(
            count == self.synchronized_count
            for count in delivered_values
        )
        duration_mean = (
            self._duration_sum / self._duration_count
            if self._duration_count
            else None
        )
        classified_duration = sum(self._stage_duration_sum.values())
        stage_statistics = {}
        for stage in _PROCESSING_STAGES:
            count = self._stage_duration_count[stage]
            total = self._stage_duration_sum[stage]
            stage_statistics[stage] = {
                "count": count,
                "total": total,
                "mean": total / count if count else None,
                "max": self._stage_duration_max[stage] if count else None,
                "recent_p95": _percentile(
                    self._recent_stage_durations[stage], 0.95
                ),
                "recent_p99": _percentile(
                    self._recent_stage_durations[stage], 0.99
                ),
                "recent_window_count": len(
                    self._recent_stage_durations[stage]
                ),
                "recent_window_capacity": (
                    self._recent_stage_durations[stage].maxlen
                ),
                "fraction_of_classified_duration": (
                    total / classified_duration
                    if classified_duration > 0.0
                    else None
                ),
            }

        return {
            "schema_version": 2,
            "delivered_messages": dict(self.delivered_counts),
            "latest_delivered_stamp_ns": dict(self.latest_stamp_ns),
            "synchronized_callbacks": self.synchronized_count,
            "processing_started": self.processing_started_count,
            "terminal_callbacks": terminal_count,
            "terminal_outcomes": {
                outcome: int(self.terminal_outcomes[outcome])
                for outcome in sorted(_TERMINAL_OUTCOMES)
            },
            "completion_notifications_published": (
                self.completion_notifications_published
            ),
            "first_synchronized_stamp_ns": (
                self.first_synchronized_stamp_ns
            ),
            "last_synchronized_stamp_ns": self.last_synchronized_stamp_ns,
            "synchronized_sensor_span_sec": sensor_span_sec,
            "synchronized_wall_span_sec": wall_span_sec,
            "effective_consumption_rate_x": effective_rate,
            "processing_duration_sec": {
                "count": self._duration_count,
                "mean": duration_mean,
                "max": (
                    self._duration_max if self._duration_count else None
                ),
                "recent_p95": _percentile(
                    self._recent_durations, 0.95
                ),
                "recent_p99": _percentile(
                    self._recent_durations, 0.99
                ),
                "recent_window_count": len(self._recent_durations),
                "recent_window_capacity": self._recent_durations.maxlen,
            },
            "processing_stage_duration_sec": {
                "classified_callback_count": (
                    self._classified_callback_count
                ),
                "classified_total_sec": classified_duration,
                "stages": stage_statistics,
            },
            "test_metric_measurements": self._test_metric_measurements(),
            "mapper_results": {
                "valid_pose_count": int(valid_pose_count),
                "tracking_or_processing_failure_count": int(
                    tracking_or_processing_failure_count
                ),
                "map_revision": int(map_revision),
            },
            "invariants": {
                "all_synchronized_callbacks_terminated": callbacks_closed,
                "one_completion_notification_per_synchronized_callback": (
                    notifications_complete
                ),
                "processing_started_matches_non_rejected_callbacks": (
                    self.processing_started_count
                    == self.synchronized_count - early_rejections
                ),
                "delivered_stream_counts_equal": (
                    delivered_streams_equal
                ),
                "all_delivered_messages_synchronized": (
                    all_delivered_synchronized
                ),
                "processing_in_progress": bool(processing_in_progress),
            },
            "complete_input_consumption": (
                self.synchronized_count > 0
                and callbacks_closed
                and notifications_complete
                and delivered_streams_equal
                and all_delivered_synchronized
            ),
        }

    def _test_metric_measurements(self) -> dict:
        def latency_summary(
            total: float,
            maximum: float,
            recent: Deque[float],
        ) -> dict:
            recent_values = _recent_statistics(recent)
            return {
                "count": self._valid_frame_count,
                "mean": (
                    total / self._valid_frame_count
                    if self._valid_frame_count
                    else None
                ),
                "max": maximum if self._valid_frame_count else None,
                "recent_median": recent_values["median"],
                "recent_p95": recent_values["p95"],
                "recent_p99": recent_values["p99"],
                "recent_window_count": recent_values["count"],
                "recent_window_capacity": recent.maxlen,
                "recent_window_covers_all_valid_frames": (
                    len(recent) == self._valid_frame_count
                ),
            }

        samples = [
            {
                "valid_frame_index": int(value[0]),
                "sensor_stamp_ns": value[1],
                "pose_publish_latency_sec": float(value[2]),
                "map_update_latency_sec": float(value[3]),
                "valid_end_to_end_latency_sec": float(value[4]),
                "localization_solution_start_monotonic_ns": value[5],
                "pose_publish_completion_monotonic_ns": value[6],
                "map_update_start_monotonic_ns": value[7],
                "map_publish_completion_monotonic_ns": value[8],
            }
            for value in sorted(
                self._formal_sample_reservoir,
                key=lambda item: item[0],
            )
        ]
        return {
            "definitions": {
                "valid_frame": (
                    "synchronized input whose callback outcome is completed "
                    "and whose pose and map update both returned"
                ),
                "pose_publish_latency_sec": (
                    "monotonic time from synchronized mapper callback entry "
                    "to return from _publish_pose"
                ),
                "map_update_latency_sec": (
                    "monotonic time from map-update stage entry to return "
                    "from the corresponding map update/publish function"
                ),
                "valid_end_to_end_latency_sec": (
                    "monotonic time from synchronized mapper callback entry "
                    "to callback termination after bookkeeping"
                ),
                "localization_solution_start_monotonic_ns": (
                    "process-monotonic timestamp at synchronized mapper callback "
                    "entry, the start endpoint of pose_publish_latency_sec"
                ),
                "pose_publish_completion_monotonic_ns": (
                    "process-monotonic timestamp immediately after _publish_pose "
                    "returns"
                ),
                "map_update_start_monotonic_ns": (
                    "process-monotonic timestamp at map-update stage entry"
                ),
                "map_publish_completion_monotonic_ns": (
                    "process-monotonic timestamp after the corresponding map "
                    "update/publish function returns"
                ),
            },
            "valid_frame_count": self._valid_frame_count,
            "pose_publish_latency_sec": latency_summary(
                self._valid_pose_latency_sum,
                self._valid_pose_latency_max,
                self._recent_valid_pose_latencies,
            ),
            "map_update_latency_sec": latency_summary(
                self._valid_map_latency_sum,
                self._valid_map_latency_max,
                self._recent_valid_map_latencies,
            ),
            "valid_end_to_end_latency_sec": latency_summary(
                self._valid_end_to_end_latency_sum,
                self._valid_end_to_end_latency_max,
                self._recent_valid_end_to_end_latencies,
            ),
            "document_random_sample": {
                "method": (
                    "reservoir_sampling_after_initial_cold_start_exclusion"
                ),
                "seed": _FORMAL_SAMPLE_SEED,
                "requested_sample_count": _FORMAL_SAMPLE_SIZE,
                "source_valid_frame_count": self._valid_frame_count,
                "excluded_initial_valid_frame_count": min(
                    self._valid_frame_count,
                    _FORMAL_COLD_START_VALID_FRAMES,
                ),
                "exclusion_reason": (
                    "first completed valid frame is CUDA/model cold start"
                ),
                "population_count": self._formal_sampling_population_count,
                "sample_count": len(samples),
                "samples": samples,
            },
        }

    @staticmethod
    def write_summary(path: Path, payload: dict) -> None:
        _atomic_json_write(Path(path), payload)
