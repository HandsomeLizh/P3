"""Template adapter from SceneSemanticModel to a mono8 semantic image."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .field_path import resolve_field
from .ros_helpers import load_message_type, qos_profile, stamp_nanoseconds


class T2SceneAdapterNode(Node):
    """Extract a dense label raster while preserving the source-frame stamp."""

    def __init__(self) -> None:
        super().__init__("t2_scene_adapter")

        self.declare_parameter(
            "input_message_type",
            "CHANGE_ME_T2_INTERFACES/msg/SceneSemanticModel",
        )
        self.declare_parameter("input_topic", "/T2/semantic/scene")
        self.declare_parameter("label_image_field", "semantic.label_image")
        self.declare_parameter("raw_label_data_field", "")
        self.declare_parameter("raw_label_height_field", "")
        self.declare_parameter("raw_label_width_field", "")
        self.declare_parameter("raw_label_header_field", "header")
        self.declare_parameter(
            "output_topic", "/Car/T2/semantic/label_image"
        )
        self.declare_parameter("input_reliability", "reliable")
        self.declare_parameter("output_reliability", "reliable")
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("expected_width", 0)
        self.declare_parameter("expected_height", 0)
        self.declare_parameter("class_lut", [0, 1, 2, 3, 4])

        self.label_image_field = self._text_parameter("label_image_field")
        self.raw_data_field = self._text_parameter("raw_label_data_field")
        self.raw_height_field = self._text_parameter(
            "raw_label_height_field"
        )
        self.raw_width_field = self._text_parameter("raw_label_width_field")
        self.raw_header_field = self._text_parameter(
            "raw_label_header_field"
        )
        if self.label_image_field:
            if self.raw_data_field or self.raw_height_field or self.raw_width_field:
                raise ValueError(
                    "Configure either label_image_field or raw_label_* fields, "
                    "not both"
                )
            self.input_mode = "nested_image"
        else:
            required_raw = (
                self.raw_data_field,
                self.raw_height_field,
                self.raw_width_field,
                self.raw_header_field,
            )
            if not all(required_raw):
                raise ValueError(
                    "Raw mode requires data, height, width and header fields"
                )
            self.input_mode = "raw_raster"

        self.expected_width = int(self.get_parameter("expected_width").value)
        self.expected_height = int(
            self.get_parameter("expected_height").value
        )
        if self.expected_width < 0 or self.expected_height < 0:
            raise ValueError("expected image dimensions cannot be negative")
        self.translation_table = self._translation_table(
            self.get_parameter("class_lut").value
        )

        input_type_name = self._text_parameter(
            "input_message_type", required=True
        )
        input_type = load_message_type(input_type_name)
        depth = int(self.get_parameter("qos_depth").value)
        input_qos = qos_profile(
            self._text_parameter("input_reliability", required=True), depth
        )
        output_qos = qos_profile(
            self._text_parameter("output_reliability", required=True), depth
        )
        self.publisher = self.create_publisher(
            Image,
            self._text_parameter("output_topic", required=True),
            output_qos,
        )
        self.subscription = self.create_subscription(
            input_type,
            self._text_parameter("input_topic", required=True),
            self._on_scene_message,
            input_qos,
        )
        self.received_count = 0
        self.published_count = 0
        self.rejected_count = 0
        self.get_logger().info(
            f"Task 2 adapter ready: {input_type_name}; "
            f"mode={self.input_mode}; "
            f"field={self.label_image_field or self.raw_data_field}"
        )

    def _text_parameter(self, name: str, *, required: bool = False) -> str:
        value = str(self.get_parameter(name).value).strip()
        if required and not value:
            raise ValueError(f"{name} cannot be empty")
        return value

    @staticmethod
    def _translation_table(values: Iterable[object]) -> bytes:
        configured = tuple(int(value) for value in values)
        if not configured or len(configured) > 256:
            raise ValueError("class_lut must contain between 1 and 256 IDs")
        if any(value < 0 or value > 255 for value in configured):
            raise ValueError("class_lut output IDs must be uint8")
        table = bytearray(256)
        for source_id, target_id in enumerate(configured):
            table[source_id] = target_id
        return bytes(table)

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("semantic image dimensions must be positive")
        if self.expected_width and width != self.expected_width:
            raise ValueError(
                f"semantic width {width} != configured {self.expected_width}"
            )
        if self.expected_height and height != self.expected_height:
            raise ValueError(
                f"semantic height {height} != configured {self.expected_height}"
            )

    def _translate_pixels(self, payload: bytes) -> bytes:
        return payload.translate(self.translation_table)

    def _from_nested_image(self, message: object) -> Image:
        source = resolve_field(message, self.label_image_field)
        if not isinstance(source, Image):
            raise TypeError(
                f"{self.label_image_field} must contain sensor_msgs/Image; "
                f"got {type(source).__name__}"
            )
        encoding = str(source.encoding).strip().lower()
        if encoding not in {"mono8", "8uc1"}:
            raise ValueError(
                f"semantic image encoding must be mono8/8UC1; got {source.encoding!r}"
            )
        width = int(source.width)
        height = int(source.height)
        step = int(source.step) or width
        self._validate_dimensions(width, height)
        if step < width:
            raise ValueError("semantic image step is smaller than width")
        payload = bytes(source.data)
        required = (height - 1) * step + width
        if len(payload) < required:
            raise ValueError(
                f"semantic image payload has {len(payload)} bytes; "
                f"requires at least {required}"
            )
        compact = bytearray(width * height)
        for row in range(height):
            source_start = row * step
            target_start = row * width
            compact[target_start : target_start + width] = payload[
                source_start : source_start + width
            ]
        output = Image()
        output.header = deepcopy(source.header)
        output.height = height
        output.width = width
        output.encoding = "mono8"
        output.is_bigendian = 0
        output.step = width
        output.data = self._translate_pixels(bytes(compact))
        return output

    def _from_raw_raster(self, message: object) -> Image:
        header = resolve_field(message, self.raw_header_field)
        if not isinstance(header, Header):
            raise TypeError(
                f"{self.raw_header_field} must contain std_msgs/Header"
            )
        width = int(resolve_field(message, self.raw_width_field))
        height = int(resolve_field(message, self.raw_height_field))
        self._validate_dimensions(width, height)
        try:
            payload = bytes(resolve_field(message, self.raw_data_field))
        except (TypeError, ValueError) as exception:
            raise TypeError(
                f"{self.raw_data_field} must be a uint8-compatible sequence"
            ) from exception
        required = width * height
        if len(payload) != required:
            raise ValueError(
                f"raw semantic payload has {len(payload)} bytes; "
                f"expected exactly {required}"
            )
        output = Image()
        output.header = deepcopy(header)
        output.height = height
        output.width = width
        output.encoding = "mono8"
        output.is_bigendian = 0
        output.step = width
        output.data = self._translate_pixels(payload)
        return output

    def _on_scene_message(self, message: object) -> None:
        self.received_count += 1
        try:
            if self.input_mode == "nested_image":
                output = self._from_nested_image(message)
            else:
                output = self._from_raw_raster(message)
            stamp_nanoseconds(output)
            if not str(output.header.frame_id).strip():
                raise ValueError("semantic image frame_id cannot be empty")
        except (AttributeError, IndexError, TypeError, ValueError) as exception:
            self.rejected_count += 1
            self.get_logger().error(
                f"Rejected Task 2 scene message #{self.received_count}: "
                f"{exception}"
            )
            return

        self.publisher.publish(output)
        self.published_count += 1
        if self.published_count == 1 or self.published_count % 100 == 0:
            self.get_logger().info(
                f"Task 2 adapter published {self.published_count} label "
                f"images; rejected={self.rejected_count}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = T2SceneAdapterNode()
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
