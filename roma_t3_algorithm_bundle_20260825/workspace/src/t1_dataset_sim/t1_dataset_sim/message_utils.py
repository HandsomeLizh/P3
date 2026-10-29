from __future__ import annotations

import array as array_module
from typing import Any

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image, Imu


def uint8_array_from_buffer(values: Any) -> array_module.array:
    """Copy one contiguous byte buffer through ROS's typed-array fast path."""

    view = memoryview(values)
    if not view.c_contiguous:
        raise ValueError("uint8 ROS buffer must be C-contiguous")
    typed_data = array_module.array("B")
    typed_data.frombytes(view.cast("B"))
    return typed_data


def nanoseconds_to_time(timestamp_ns: int) -> Time:
    """Convert an integer nanosecond sensor timestamp without using float."""
    timestamp_ns = int(timestamp_ns)
    message = Time()
    message.sec = timestamp_ns // 1_000_000_000
    message.nanosec = timestamp_ns % 1_000_000_000
    return message


def seconds_to_nanoseconds(timestamp_seconds: Any) -> int:
    """Convert a numeric sensor time in seconds to integer nanoseconds."""
    # Decimal through string avoids most binary float artefacts from pandas.
    from decimal import Decimal

    value = Decimal(str(timestamp_seconds))
    return int((value * Decimal(1_000_000_000)).to_integral_value())


def image_to_message(
    image: np.ndarray,
    stamp: Time,
    frame_id: str,
    encoding: str,
) -> Image:
    """Convert a contiguous NumPy image to sensor_msgs/Image without cv_bridge."""
    array = np.ascontiguousarray(image)

    if encoding == "bgr8":
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"bgr8 expects uint8 HxWx3, got {array.dtype} {array.shape}")
        channels = 3
    elif encoding in {"mono8", "8UC1"}:
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError(f"mono8 expects uint8 HxW, got {array.dtype} {array.shape}")
        channels = 1
        encoding = "mono8"
    elif encoding == "32FC1":
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError(f"32FC1 expects float32 HxW, got {array.dtype} {array.shape}")
        channels = 1
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    height, width = array.shape[:2]

    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = int(height)
    message.width = int(width)
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(width * channels * array.dtype.itemsize)
    # Generated ROS sequence setters validate a generic ``bytes`` value one
    # Python element at a time.  ``array('B')`` is their documented typed
    # buffer path and preserves the identical byte sequence without that
    # multi-megabyte validation loop.
    message.data = uint8_array_from_buffer(array)
    return message


def _set_fixed_sequence(target, values) -> None:
    if len(target) != len(values):
        raise ValueError(f"Fixed sequence length {len(target)} != {len(values)}")
    for index, value in enumerate(values):
        target[index] = float(value)


def make_camera_info(
    *,
    stamp: Time,
    frame_id: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline_m: float,
    is_right: bool,
) -> CameraInfo:
    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.width = int(width)
    message.height = int(height)
    message.distortion_model = "plumb_bob"
    message.d = [0.0] * 5

    _set_fixed_sequence(
        message.k,
        [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    )
    _set_fixed_sequence(
        message.r,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )

    tx = -float(fx) * float(baseline_m) if is_right else 0.0
    _set_fixed_sequence(
        message.p,
        [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    return message


def make_imu_message(
    *,
    stamp: Time,
    frame_id: str,
    angular_velocity_rad_s: np.ndarray,
) -> Imu:
    """Create an IMU message containing only angular velocity."""
    gyro = np.asarray(angular_velocity_rad_s, dtype=np.float64).reshape(3)

    message = Imu()
    message.header.stamp = stamp
    message.header.frame_id = frame_id

    # Orientation and linear acceleration are deliberately unavailable.
    message.orientation_covariance[0] = -1.0
    message.linear_acceleration_covariance[0] = -1.0

    message.angular_velocity.x = float(gyro[0])
    message.angular_velocity.y = float(gyro[1])
    message.angular_velocity.z = float(gyro[2])
    return message
