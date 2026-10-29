"""Shared ROS type, QoS and timestamp validation helpers."""

from __future__ import annotations

import re
from typing import Type

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosidl_runtime_py.utilities import get_message


_TYPE_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*/msg/[A-Za-z][A-Za-z0-9_]*$"
)
_RELIABILITY = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}


def load_message_type(type_name: str) -> Type[object]:
    """Load an installed ROS message class from ``package/msg/Type``."""

    normalised = str(type_name).strip()
    if _TYPE_PATTERN.fullmatch(normalised) is None:
        raise ValueError(
            "input_message_type must use package/msg/Type syntax; "
            f"got {normalised!r}"
        )
    try:
        return get_message(normalised)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exception:
        raise RuntimeError(
            f"Cannot load {normalised}. Build and source the official "
            "Task 1/Task 2 interface package first."
        ) from exception


def qos_profile(reliability: str, depth: int) -> QoSProfile:
    normalised = str(reliability).strip().lower()
    if normalised not in _RELIABILITY:
        raise ValueError("reliability must be reliable or best_effort")
    if int(depth) <= 0:
        raise ValueError("QoS depth must be positive")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=_RELIABILITY[normalised],
        durability=DurabilityPolicy.VOLATILE,
    )


def stamp_nanoseconds(message: object) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None or not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
        raise ValueError("message has no valid Header.stamp")
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if nanoseconds < 0 or nanoseconds >= 1_000_000_000:
        raise ValueError(
            f"Header.stamp.nanosec must be in [0, 1e9); got {nanoseconds}"
        )
    return seconds * 1_000_000_000 + nanoseconds
