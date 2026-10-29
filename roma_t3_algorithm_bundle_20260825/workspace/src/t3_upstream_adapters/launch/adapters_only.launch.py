from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start only the optional Task 1/Task 2 boundary adapters."""

    adapter_share = Path(
        get_package_share_directory("t3_upstream_adapters")
    )
    arguments = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("enable_t1_adapter", default_value="true"),
        DeclareLaunchArgument("enable_t2_adapter", default_value="true"),
        DeclareLaunchArgument(
            "adapter_params_file",
            default_value=str(
                adapter_share / "config" / "interface_mapping.yaml"
            ),
        ),
    ]
    parameters = [
        LaunchConfiguration("adapter_params_file"),
        {"use_sim_time": LaunchConfiguration("use_sim_time")},
    ]
    t1_adapter = Node(
        package="t3_upstream_adapters",
        executable="t1_fusion_adapter",
        name="t1_fusion_adapter",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("enable_t1_adapter")),
        parameters=parameters,
    )
    t2_adapter = Node(
        package="t3_upstream_adapters",
        executable="t2_scene_adapter",
        name="t2_scene_adapter",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("enable_t2_adapter")),
        parameters=parameters,
    )
    return LaunchDescription(arguments + [t1_adapter, t2_adapter])
