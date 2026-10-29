from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from .segmenter import BasicSemanticSegmenter, CLASS_NAMES


class BasicSemanticNode(Node):
    def __init__(self) -> None:
        super().__init__("basic_semantic_node")

        self.declare_parameter(
            "input_topic", "/Car/T1/fusion/stereo/left/image_raw"
        )
        self.declare_parameter(
            "label_topic", "/Car/T2/semantic/label_image"
        )
        self.declare_parameter(
            "color_topic", "/Car/T2/semantic/color_image"
        )
        self.declare_parameter("publish_color", False)
        self.declare_parameter("input_reliability", "reliable")
        self.declare_parameter("input_qos_depth", 10)

        input_reliability = str(
            self.get_parameter("input_reliability").value
        ).strip().lower()
        reliability_policies = {
            "reliable": ReliabilityPolicy.RELIABLE,
            "best_effort": ReliabilityPolicy.BEST_EFFORT,
        }
        if input_reliability not in reliability_policies:
            raise ValueError(
                "input_reliability must be 'reliable' or 'best_effort', "
                f"got {input_reliability!r}"
            )

        input_qos_depth = int(self.get_parameter("input_qos_depth").value)
        if input_qos_depth <= 0:
            raise ValueError("input_qos_depth must be greater than zero")

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=input_qos_depth,
            reliability=reliability_policies[input_reliability],
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()
        self.segmenter = BasicSemanticSegmenter()
        self.label_pub = self.create_publisher(
            Image, str(self.get_parameter("label_topic").value), output_qos
        )
        self.color_pub = self.create_publisher(
            Image, str(self.get_parameter("color_topic").value), output_qos
        )
        self.publish_color = bool(self.get_parameter("publish_color").value)

        self.subscription = self.create_subscription(
            Image,
            str(self.get_parameter("input_topic").value),
            self.on_image,
            input_qos,
        )

        self.get_logger().info(
            "Basic semantic segmentation ready. "
            f"Input QoS: {input_reliability}, depth={input_qos_depth}. "
            f"Classes: {CLASS_NAMES}"
        )

    def on_image(self, message: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        labels = self.segmenter.segment(bgr)

        label_message = self.bridge.cv2_to_imgmsg(labels, encoding="mono8")
        label_message.header = message.header
        self.label_pub.publish(label_message)

        if self.publish_color:
            color = self.segmenter.colorize(labels)
            color_message = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            color_message.header = message.header
            self.color_pub.publish(color_message)



def main(args=None) -> None:
    rclpy.init(args=args)
    node = BasicSemanticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
