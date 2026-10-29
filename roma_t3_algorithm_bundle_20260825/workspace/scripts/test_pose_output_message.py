#!/usr/bin/env python3
"""Wire contracts for the pure Odometry and TF message constructors."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE / "src" / "t3_semantic_mapping"
MAPPER_PATH = PACKAGE / (
    "t3_semantic_mapping/roma_semantic_mapper_node.py"
)
sys.path.insert(0, str(PACKAGE))

from t3_semantic_mapping.odometry_output import (  # noqa: E402
    OdometryOutputSample,
)
from t3_semantic_mapping.pose_output_message import (  # noqa: E402
    make_odometry_message,
    make_transform_stamped_message,
)


def _header() -> Header:
    header = Header()
    header.stamp.sec = 17
    header.stamp.nanosec = 23
    header.frame_id = "ignored_sensor_frame"
    return header


def _sample() -> OdometryOutputSample:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler(
        "xyz", [0.2, -0.3, 1.1]
    ).as_matrix()
    pose[:3, 3] = [1.25, -2.5, 3.75]
    return OdometryOutputSample(
        pose_map_child=pose,
        twist_angular_linear=np.array(
            [0.1, -0.2, 0.3, 1.0, -2.0, 3.0],
            dtype=np.float64,
        ),
        pose_covariance_ros=np.arange(36, dtype=np.float64).reshape(6, 6),
        twist_covariance_ros=(
            np.arange(36, dtype=np.float64).reshape(6, 6) + 100.0
        ),
        velocity_method="frozen_test",
    )


def _frozen_odometry_reference(
    *,
    header: Header,
    map_frame: str,
    child_frame: str,
    output: OdometryOutputSample,
) -> Odometry:
    pose_map_child = output.pose_map_child
    velocity_child = output.twist_angular_linear
    quaternion = Rotation.from_matrix(
        np.asarray(pose_map_child[:3, :3], dtype=np.float64).copy()
    ).as_quat()

    message = Odometry()
    message.header.stamp = header.stamp
    message.header.frame_id = map_frame
    message.child_frame_id = child_frame
    message.pose.pose.position.x = float(pose_map_child[0, 3])
    message.pose.pose.position.y = float(pose_map_child[1, 3])
    message.pose.pose.position.z = float(pose_map_child[2, 3])
    message.pose.pose.orientation.x = float(quaternion[0])
    message.pose.pose.orientation.y = float(quaternion[1])
    message.pose.pose.orientation.z = float(quaternion[2])
    message.pose.pose.orientation.w = float(quaternion[3])
    message.pose.covariance = output.pose_covariance_ros.reshape(-1).tolist()
    message.twist.twist.angular.x = float(velocity_child[0])
    message.twist.twist.angular.y = float(velocity_child[1])
    message.twist.twist.angular.z = float(velocity_child[2])
    message.twist.twist.linear.x = float(velocity_child[3])
    message.twist.twist.linear.y = float(velocity_child[4])
    message.twist.twist.linear.z = float(velocity_child[5])
    message.twist.covariance = output.twist_covariance_ros.reshape(-1).tolist()
    return message


def _frozen_transform_reference(odometry: Odometry) -> TransformStamped:
    message = TransformStamped()
    message.header = odometry.header
    message.child_frame_id = odometry.child_frame_id
    message.transform.translation.x = odometry.pose.pose.position.x
    message.transform.translation.y = odometry.pose.pose.position.y
    message.transform.translation.z = odometry.pose.pose.position.z
    message.transform.rotation = odometry.pose.pose.orientation
    return message


class PoseOutputMessageTest(unittest.TestCase):
    def test_odometry_is_byte_identical_to_frozen_reference(self):
        sample = _sample()
        actual = make_odometry_message(
            header=_header(),
            map_frame="map",
            child_frame="camera_optical",
            output=sample,
        )
        expected = _frozen_odometry_reference(
            header=_header(),
            map_frame="map",
            child_frame="camera_optical",
            output=sample,
        )
        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )

    def test_transform_is_byte_identical_to_frozen_reference(self):
        odometry = make_odometry_message(
            header=_header(),
            map_frame="map",
            child_frame="base_link",
            output=_sample(),
        )
        actual = make_transform_stamped_message(odometry)
        expected = _frozen_transform_reference(odometry)
        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )

    def test_odometry_owns_detached_numeric_payload(self):
        sample = _sample()
        message = make_odometry_message(
            header=_header(),
            map_frame="map",
            child_frame="camera_optical",
            output=sample,
        )
        before = bytes(serialize_message(message))
        sample.pose_map_child[:] = np.eye(4)
        sample.twist_angular_linear[:] = 99.0
        sample.pose_covariance_ros[:] = 88.0
        sample.twist_covariance_ros[:] = 77.0
        self.assertEqual(bytes(serialize_message(message)), before)

    def test_mapper_delegates_but_owns_publish_and_broadcast(self):
        tree = ast.parse(MAPPER_PATH.read_text(encoding="utf-8"))
        mapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RomaSemanticMapperNode"
        )
        method = next(
            node
            for node in mapper.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_publish_pose"
        )
        named_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ]
        names = {node.func.id for node in named_calls}
        self.assertIn("make_odometry_message", names)
        self.assertIn("make_transform_stamped_message", names)
        self.assertNotIn("Odometry", names)
        self.assertNotIn("TransformStamped", names)

        odometry_builder = next(
            node
            for node in named_calls
            if node.func.id == "make_odometry_message"
        )
        odometry_publish = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "publish"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "odometry_pub"
        )
        transform_builder = next(
            node
            for node in named_calls
            if node.func.id == "make_transform_stamped_message"
        )
        broadcast = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sendTransform"
        )
        self.assertLess(odometry_builder.lineno, odometry_publish.lineno)
        self.assertLess(transform_builder.lineno, broadcast.lineno)

        tf_guard = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "tf_broadcaster"
        )
        guarded_names = {
            node.func.id
            for node in ast.walk(tf_guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        guarded_attributes = {
            node.func.attr
            for node in ast.walk(tf_guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("make_transform_stamped_message", guarded_names)
        self.assertIn("sendTransform", guarded_attributes)

        node_owned_calls = {
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        }
        self.assertIn("_publish_path_if_due", node_owned_calls)
        self.assertIn("_write_tum", node_owned_calls)


if __name__ == "__main__":
    unittest.main()
