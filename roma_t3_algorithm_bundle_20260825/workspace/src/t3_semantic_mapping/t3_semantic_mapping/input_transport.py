"""Validated ROS 2 transport policy for the mapper's standard inputs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from message_filters import (
    ApproximateTimeSynchronizer,
    SimpleFilter,
    TimeSynchronizer,
)
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


_RELIABILITY_POLICIES = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}
_SYNC_MODES = {"exact", "approximate"}
_TIMESTAMP_SOURCES = {"sensor", "arrival"}


class TimestampSourceFilter(SimpleFilter):
    """Optionally replace image stamps before message-filter synchronisation.

    ``arrival`` is an explicit replay compatibility mode for recordings whose
    image ``Header.stamp`` fields are missing or corrupt.  Normal live runs use
    ``sensor`` and therefore preserve the acquisition timestamps unchanged.
    """

    def __init__(self, source_filter, *, timestamp_source, now_callback):
        super().__init__()
        self.timestamp_source = _normalised_choice(
            timestamp_source,
            name="timestamp_source",
            choices=_TIMESTAMP_SOURCES,
        )
        self._now_callback = now_callback
        self._source_filter = source_filter
        self._source_stamp_by_message_id = OrderedDict()
        self._source_stamp_capacity = 2048
        source_filter.registerCallback(self._on_message)

    @staticmethod
    def _stamp_ns(message) -> int:
        return (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _on_message(self, message) -> None:
        if self.timestamp_source == "arrival":
            message_id = id(message)
            self._source_stamp_by_message_id[message_id] = self._stamp_ns(
                message
            )
            self._source_stamp_by_message_id.move_to_end(message_id)
            while (
                len(self._source_stamp_by_message_id)
                > self._source_stamp_capacity
            ):
                self._source_stamp_by_message_id.popitem(last=False)
            message.header.stamp = self._now_callback()
        self.signalMessage(message)

    def pop_source_stamp_ns(self, message) -> int | None:
        """Return the pre-arrival-rewrite stamp without changing transport."""

        if self.timestamp_source == "sensor":
            return self._stamp_ns(message)
        return self._source_stamp_by_message_id.pop(id(message), None)


def _normalised_choice(value: str, *, name: str, choices: set[str]) -> str:
    normalised = str(value).strip().lower()
    if normalised not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")
    return normalised


@dataclass(frozen=True)
class InputTransportConfiguration:
    """Immutable input QoS and timestamp-synchronisation configuration."""

    stereo_reliability: str = "reliable"
    semantic_reliability: str = "reliable"
    imu_reliability: str = "reliable"
    image_qos_depth: int = 10
    imu_qos_depth: int = 10
    sync_mode: str = "exact"
    sync_queue_size: int = 50
    sync_slop_sec: float = 0.02
    stereo_max_skew_sec: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "stereo_reliability",
            "semantic_reliability",
            "imu_reliability",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalised_choice(
                    getattr(self, field_name),
                    name=field_name,
                    choices=set(_RELIABILITY_POLICIES),
                ),
            )
        object.__setattr__(
            self,
            "sync_mode",
            _normalised_choice(
                self.sync_mode,
                name="sync_mode",
                choices=_SYNC_MODES,
            ),
        )
        if self.image_qos_depth <= 0:
            raise ValueError("image_qos_depth must be positive")
        if self.imu_qos_depth <= 0:
            raise ValueError("imu_qos_depth must be positive")
        if self.sync_queue_size <= 0:
            raise ValueError("sync_queue_size must be positive")
        if self.sync_slop_sec < 0.0:
            raise ValueError("sync_slop_sec cannot be negative")
        if self.stereo_max_skew_sec < 0.0:
            raise ValueError("stereo_max_skew_sec cannot be negative")
        if self.sync_mode == "approximate" and self.sync_slop_sec == 0.0:
            raise ValueError(
                "sync_slop_sec must be positive in approximate mode"
            )
        if (
            self.sync_mode == "approximate"
            and self.stereo_max_skew_sec > self.sync_slop_sec
        ):
            raise ValueError(
                "stereo_max_skew_sec cannot exceed sync_slop_sec"
            )

    @staticmethod
    def qos_profile(reliability: str, depth: int) -> QoSProfile:
        """Return the volatile keep-last profile for one input stream."""

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=_RELIABILITY_POLICIES[reliability],
            durability=DurabilityPolicy.VOLATILE,
        )

    @property
    def stereo_qos(self) -> QoSProfile:
        return self.qos_profile(
            self.stereo_reliability, self.image_qos_depth
        )

    @property
    def semantic_qos(self) -> QoSProfile:
        return self.qos_profile(
            self.semantic_reliability, self.image_qos_depth
        )

    @property
    def imu_qos(self) -> QoSProfile:
        return self.qos_profile(self.imu_reliability, self.imu_qos_depth)

    def accepts_stereo_skew(self, left_stamp_ns: int, right_stamp_ns: int) -> bool:
        """Protect stereo geometry when semantic matching is approximate."""

        if self.sync_mode == "exact":
            return left_stamp_ns == right_stamp_ns
        maximum_ns = int(round(self.stereo_max_skew_sec * 1e9))
        return abs(left_stamp_ns - right_stamp_ns) <= maximum_ns

    def create_synchronizer(self, filters: Sequence[object]):
        """Build a header-stamp synchronizer; headerless input is forbidden."""

        if self.sync_mode == "exact":
            return TimeSynchronizer(filters, self.sync_queue_size)
        return ApproximateTimeSynchronizer(
            filters,
            self.sync_queue_size,
            self.sync_slop_sec,
            allow_headerless=False,
        )

    @property
    def sync_description(self) -> str:
        if self.sync_mode == "exact":
            return f"exact (queue={self.sync_queue_size})"
        return (
            f"approximate (queue={self.sync_queue_size}, "
            f"slop={self.sync_slop_sec:.6f}s, "
            f"stereo_max_skew={self.stereo_max_skew_sec:.6f}s, "
            "header required)"
        )
