"""Pure constructors for the stable Odometry and TF wire messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from .coordinate_utils import matrix_to_quaternion_xyzw

if TYPE_CHECKING:
    from .odometry_output import OdometryOutputSample


def make_odometry_message(
    *,
    header: Header,
    map_frame: str,
    child_frame: str,
    output: "OdometryOutputSample",
) -> Odometry:
    """Build one complete Odometry message without publishing side effects."""

    pose_map_child = output.pose_map_child
    velocity_child = output.twist_angular_linear
    quaternion = matrix_to_quaternion_xyzw(pose_map_child[:3, :3])

    message = Odometry()
    message.header.stamp = header.stamp
    message.header.frame_id = str(map_frame)
    message.child_frame_id = str(child_frame)
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


def make_transform_stamped_message(odometry: Odometry) -> TransformStamped:
    """Build the matching parent-to-child TF from an Odometry message."""

    message = TransformStamped()
    message.header = odometry.header
    message.child_frame_id = odometry.child_frame_id
    message.transform.translation.x = odometry.pose.pose.position.x
    message.transform.translation.y = odometry.pose.pose.position.y
    message.transform.translation.z = odometry.pose.pose.position.z
    message.transform.rotation = odometry.pose.pose.orientation
    return message


def make_transform_stamped_from_matrix(
    *,
    header: Header,
    parent_frame: str,
    child_frame: str,
    transform_parent_child: np.ndarray,
) -> TransformStamped:
    """Build a stamped TF from a homogeneous parent-to-child transform."""

    transform = np.asarray(transform_parent_child, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("TF transform must be a finite 4x4 matrix")
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    message = TransformStamped()
    message.header.stamp = header.stamp
    message.header.frame_id = str(parent_frame)
    message.child_frame_id = str(child_frame)
    message.transform.translation.x = float(transform[0, 3])
    message.transform.translation.y = float(transform[1, 3])
    message.transform.translation.z = float(transform[2, 3])
    message.transform.rotation.x = float(quaternion[0])
    message.transform.rotation.y = float(quaternion[1])
    message.transform.rotation.z = float(quaternion[2])
    message.transform.rotation.w = float(quaternion[3])
    return message
