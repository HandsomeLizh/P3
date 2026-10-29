"""Template adapter from one Task 1 custom fusion message to standard topics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2

from .field_path import optional_field, resolve_field
from .ros_helpers import load_message_type, qos_profile, stamp_nanoseconds


class T1FusionAdapterNode(Node):
    """Unpack nested standard messages without changing sensor data."""

    def __init__(self) -> None:
        super().__init__("t1_fusion_adapter")

        self.declare_parameter(
            "input_message_type", "CHANGE_ME_T1_INTERFACES/msg/FusionData"
        )
        self.declare_parameter("input_topic", "/Car/T1/fusion/multimodal")
        self.declare_parameter("left_image_field", "stereo.left_image")
        self.declare_parameter("right_image_field", "stereo.right_image")
        self.declare_parameter("imu_field", "imu_samples")
        self.declare_parameter("lidar_field", "dense_cloud")
        self.declare_parameter("odometry_field", "")
        self.declare_parameter(
            "left_output_topic", "/Car/T1/fusion/stereo/left/image_raw"
        )
        self.declare_parameter(
            "right_output_topic", "/Car/T1/fusion/stereo/right/image_raw"
        )
        self.declare_parameter("imu_output_topic", "/Car/T1/fusion/imu/data")
        self.declare_parameter(
            "lidar_output_topic", "/Car/T1/fusion/lidar/points"
        )
        self.declare_parameter("odometry_output_topic", "/Odometry")
        self.declare_parameter("input_reliability", "reliable")
        self.declare_parameter("input_qos_depth", 10)
        self.declare_parameter("image_output_reliability", "reliable")
        self.declare_parameter("imu_output_reliability", "reliable")
        self.declare_parameter("lidar_output_reliability", "best_effort")
        self.declare_parameter("odometry_output_reliability", "best_effort")
        self.declare_parameter("output_qos_depth", 10)
        self.declare_parameter("required_image_encoding", "bgr8")
        self.declare_parameter("maximum_stereo_skew_sec", 0.0)

        self.left_field = self._text_parameter("left_image_field", required=True)
        self.right_field = self._text_parameter("right_image_field", required=True)
        self.imu_field = self._text_parameter("imu_field")
        self.lidar_field = self._text_parameter("lidar_field")
        self.odometry_field = self._text_parameter("odometry_field")
        self.required_image_encoding = self._text_parameter(
            "required_image_encoding"
        ).lower()
        self.maximum_stereo_skew_ns = int(
            round(
                float(self.get_parameter("maximum_stereo_skew_sec").value)
                * 1_000_000_000
            )
        )
        if self.maximum_stereo_skew_ns < 0:
            raise ValueError("maximum_stereo_skew_sec cannot be negative")

        input_type_name = self._text_parameter(
            "input_message_type", required=True
        )
        input_type = load_message_type(input_type_name)
        input_qos = qos_profile(
            self._text_parameter("input_reliability", required=True),
            int(self.get_parameter("input_qos_depth").value),
        )
        output_depth = int(self.get_parameter("output_qos_depth").value)
        image_qos = qos_profile(
            self._text_parameter("image_output_reliability", required=True),
            output_depth,
        )
        imu_qos = qos_profile(
            self._text_parameter("imu_output_reliability", required=True),
            output_depth,
        )
        lidar_qos = qos_profile(
            self._text_parameter("lidar_output_reliability", required=True),
            output_depth,
        )
        odometry_qos = qos_profile(
            self._text_parameter(
                "odometry_output_reliability", required=True
            ),
            output_depth,
        )

        self.left_publisher = self.create_publisher(
            Image,
            self._text_parameter("left_output_topic", required=True),
            image_qos,
        )
        self.right_publisher = self.create_publisher(
            Image,
            self._text_parameter("right_output_topic", required=True),
            image_qos,
        )
        self.imu_publisher = (
            self.create_publisher(
                Imu,
                self._text_parameter("imu_output_topic", required=True),
                imu_qos,
            )
            if self.imu_field
            else None
        )
        self.lidar_publisher = (
            self.create_publisher(
                PointCloud2,
                self._text_parameter("lidar_output_topic", required=True),
                lidar_qos,
            )
            if self.lidar_field
            else None
        )
        self.odometry_publisher = (
            self.create_publisher(
                Odometry,
                self._text_parameter("odometry_output_topic", required=True),
                odometry_qos,
            )
            if self.odometry_field
            else None
        )
        self.subscription = self.create_subscription(
            input_type,
            self._text_parameter("input_topic", required=True),
            self._on_fusion_message,
            input_qos,
        )
        self.received_count = 0
        self.published_count = 0
        self.rejected_count = 0
        self.get_logger().info(
            f"Task 1 adapter ready: {input_type_name}; "
            f"left={self.left_field}, right={self.right_field}, "
            f"imu={self.imu_field or 'off'}, "
            f"lidar={self.lidar_field or 'off'}, "
            f"odometry={self.odometry_field or 'off'}"
        )

    def _text_parameter(self, name: str, *, required: bool = False) -> str:
        value = str(self.get_parameter(name).value).strip()
        if required and not value:
            raise ValueError(f"{name} cannot be empty")
        return value

    @staticmethod
    def _require_type(value: object, expected_type: type, field: str):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{field} must contain {expected_type.__name__}; "
                f"got {type(value).__name__}"
            )
        return value

    def _validate_stereo(self, left: Image, right: Image) -> None:
        if int(left.width) <= 0 or int(left.height) <= 0:
            raise ValueError("left image dimensions are invalid")
        if (int(left.width), int(left.height)) != (
            int(right.width),
            int(right.height),
        ):
            raise ValueError("left and right image dimensions differ")
        if self.required_image_encoding:
            for name, image in (("left", left), ("right", right)):
                if str(image.encoding).lower() != self.required_image_encoding:
                    raise ValueError(
                        f"{name} image encoding is {image.encoding!r}; "
                        f"expected {self.required_image_encoding!r}"
                    )
        for name, image in (("left", left), ("right", right)):
            step = int(image.step)
            if step <= 0:
                raise ValueError(f"{name} image step must be positive")
            required_payload = step * int(image.height)
            if len(image.data) < required_payload:
                raise ValueError(
                    f"{name} image payload has {len(image.data)} bytes; "
                    f"requires at least {required_payload}"
                )
        skew = abs(stamp_nanoseconds(left) - stamp_nanoseconds(right))
        if skew > self.maximum_stereo_skew_ns:
            raise ValueError(
                f"stereo acquisition skew {skew / 1e6:.3f} ms exceeds "
                f"{self.maximum_stereo_skew_ns / 1e6:.3f} ms"
            )
        if not str(left.header.frame_id).strip() or not str(
            right.header.frame_id
        ).strip():
            raise ValueError("stereo frame_id cannot be empty")

    def _imu_messages(self, value: Optional[object]) -> Tuple[Imu, ...]:
        if value is None:
            return ()
        if isinstance(value, Imu):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            messages = tuple(value)
            for message in messages:
                self._require_type(message, Imu, self.imu_field)
            return messages
        raise TypeError(
            f"{self.imu_field} must contain Imu or a sequence of Imu"
        )

    def _on_fusion_message(self, message: object) -> None:
        self.received_count += 1
        try:
            left = self._require_type(
                resolve_field(message, self.left_field),
                Image,
                self.left_field,
            )
            right = self._require_type(
                resolve_field(message, self.right_field),
                Image,
                self.right_field,
            )
            imu_messages = self._imu_messages(
                optional_field(message, self.imu_field)
            )
            lidar_value = optional_field(message, self.lidar_field)
            lidar = (
                self._require_type(
                    lidar_value, PointCloud2, self.lidar_field
                )
                if lidar_value is not None
                else None
            )
            odometry_value = optional_field(message, self.odometry_field)
            odometry = (
                self._require_type(
                    odometry_value, Odometry, self.odometry_field
                )
                if odometry_value is not None
                else None
            )
            self._validate_stereo(left, right)
            for imu_message in imu_messages:
                stamp_nanoseconds(imu_message)
            if lidar is not None:
                stamp_nanoseconds(lidar)
                if not str(lidar.header.frame_id).strip():
                    raise ValueError("LiDAR frame_id cannot be empty")
            if odometry is not None:
                stamp_nanoseconds(odometry)
                if not str(odometry.header.frame_id).strip():
                    raise ValueError("Odometry parent frame_id cannot be empty")
                if not str(odometry.child_frame_id).strip():
                    raise ValueError("Odometry child_frame_id cannot be empty")
        except (AttributeError, IndexError, TypeError, ValueError) as exception:
            self.rejected_count += 1
            self.get_logger().error(
                f"Rejected Task 1 fusion message #{self.received_count}: "
                f"{exception}"
            )
            return

        # Validate the entire container before publishing any part of it.
        if self.imu_publisher is not None:
            for imu_message in imu_messages:
                self.imu_publisher.publish(imu_message)
        if self.lidar_publisher is not None and lidar is not None:
            self.lidar_publisher.publish(lidar)
        if self.odometry_publisher is not None and odometry is not None:
            self.odometry_publisher.publish(odometry)
        self.left_publisher.publish(left)
        self.right_publisher.publish(right)
        self.published_count += 1
        if self.published_count == 1 or self.published_count % 100 == 0:
            self.get_logger().info(
                f"Task 1 adapter published {self.published_count} stereo "
                f"containers; rejected={self.rejected_count}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = T1FusionAdapterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
