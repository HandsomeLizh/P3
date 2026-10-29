"""Thread-safe timestamp pairing for one ordered LiDAR message stream."""

from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Deque, Optional, Tuple


class LidarMessageBuffer:
    """Keep recent LiDAR messages and select the true nearest timestamp.

    A valid predecessor alone is not enough to prove that it is nearest.  In
    the live path :meth:`nearest` waits until the ordered stream reaches the
    target, then compares predecessor and successor.  At stream end it may
    return an in-gate predecessor after the wall-clock timeout.

    Messages are retained by reference, matching the previous node-local
    deque behaviour.  Duplicate timestamps replace the previous message and
    capacity trimming always keeps the newest sensor timestamps.
    """

    def __init__(
        self,
        *,
        max_size: int = 20,
        max_time_offset_sec: float = 0.15,
        wait_timeout_sec: float = 1.0,
    ) -> None:
        size = int(max_size)
        gate = float(max_time_offset_sec)
        timeout = float(wait_timeout_sec)
        if size <= 0:
            raise ValueError("LiDAR message buffer max_size must be positive")
        if not math.isfinite(gate) or gate < 0.0:
            raise ValueError(
                "LiDAR max_time_offset_sec must be finite and non-negative"
            )
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError(
                "LiDAR wait_timeout_sec must be finite and non-negative"
            )

        self.max_size = size
        self.max_time_offset_sec = gate
        self.wait_timeout_sec = timeout
        self._condition = threading.Condition()
        self._entries: Deque[Tuple[int, object]] = deque(maxlen=size)

    def __len__(self) -> int:
        with self._condition:
            return len(self._entries)

    @property
    def time_range_ns(self) -> Optional[Tuple[int, int]]:
        with self._condition:
            if not self._entries:
                return None
            return int(self._entries[0][0]), int(self._entries[-1][0])

    @staticmethod
    def _positive_timestamp(value: int, *, name: str) -> int:
        try:
            timestamp = int(value)
        except (TypeError, ValueError, OverflowError) as exception:
            raise ValueError(
                f"{name} must be a positive integer"
            ) from exception
        if timestamp <= 0:
            raise ValueError(f"{name} must be positive")
        return timestamp

    def add(self, timestamp_ns: int, message: object) -> None:
        """Insert or replace one message and wake waiting consumers."""

        timestamp = self._positive_timestamp(
            timestamp_ns,
            name="LiDAR message timestamp_ns",
        )
        with self._condition:
            entries = [
                entry
                for entry in self._entries
                if int(entry[0]) != timestamp
            ]
            entries.append((timestamp, message))
            entries.sort(key=lambda entry: int(entry[0]))
            self._entries = deque(
                entries[-self.max_size :],
                maxlen=self.max_size,
            )
            self._condition.notify_all()

    def nearest(
        self,
        target_timestamp_ns: int,
    ) -> Tuple[Optional[object], float]:
        """Return ``(message, signed_offset_sec)`` within the configured gate."""

        target = self._positive_timestamp(
            target_timestamp_ns,
            name="LiDAR target timestamp_ns",
        )
        deadline = time.monotonic() + self.wait_timeout_sec
        nearest_message: Optional[object] = None
        nearest_offset_sec = float("inf")

        with self._condition:
            while True:
                entries = tuple(self._entries)
                if entries:
                    timestamp, nearest_message = min(
                        entries,
                        key=lambda entry: (
                            abs(int(entry[0]) - target),
                            int(entry[0]) > target,
                        ),
                    )
                    nearest_offset_sec = (int(timestamp) - target) * 1e-9

                    # Once the ordered stream has reached the target, later
                    # messages can only be farther into the future.
                    if int(entries[-1][0]) >= target:
                        if (
                            abs(nearest_offset_sec)
                            <= self.max_time_offset_sec
                        ):
                            return nearest_message, nearest_offset_sec
                        return None, nearest_offset_sec

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    if (
                        nearest_message is not None
                        and abs(nearest_offset_sec)
                        <= self.max_time_offset_sec
                    ):
                        return nearest_message, nearest_offset_sec
                    return None, nearest_offset_sec
                self._condition.wait(timeout=remaining)
