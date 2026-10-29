from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from t3_semantic_mapping.launch_options import (
    input_launch_arguments,
    input_parameter_overrides,
    localization_launch_arguments,
    localization_parameter_overrides,
    mapping_launch_arguments,
    mapping_parameter_overrides,
)


def generate_launch_description():
    """Start both boundary adapters and the unchanged Task 3 mapper."""

    home = Path.home()
    adapter_share = Path(
        get_package_share_directory("t3_upstream_adapters")
    )
    mapper_share = Path(
        get_package_share_directory("t3_semantic_mapping")
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
        DeclareLaunchArgument(
            "data_dir",
            default_value=EnvironmentVariable(
                "T3_DATA_DIR",
                default_value=str(home / "t3_data"),
            ),
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value=EnvironmentVariable(
                "T3_OUTPUT_DIR",
                default_value=str(home / "P3" / "output"),
            ),
        ),
        DeclareLaunchArgument(
            "algorithm_root",
            default_value=EnvironmentVariable(
                "T3_ALGORITHM_ROOT",
                default_value=str(
                    home / "t3_gridmap_v2_ws" / "third_party" / "RoMa"
                ),
            ),
        ),
    ]
    # Reuse the production definitions rather than maintaining a partial,
    # drifting copy of VO/LIO/fusion/mapping controls in this optional layer.
    arguments.extend(input_launch_arguments())
    arguments.extend(localization_launch_arguments())
    arguments.extend(mapping_launch_arguments())

    adapter_parameters = [
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
        parameters=adapter_parameters,
    )
    t2_adapter = Node(
        package="t3_upstream_adapters",
        executable="t2_scene_adapter",
        name="t2_scene_adapter",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("enable_t2_adapter")),
        parameters=adapter_parameters,
    )
    mapper = Node(
        package="t3_semantic_mapping",
        executable="roma_semantic_mapper",
        name="roma_semantic_mapper_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            str(mapper_share / "config" / "semantic_map.yaml"),
            {
                **input_parameter_overrides(),
                **localization_parameter_overrides(),
                **mapping_parameter_overrides(),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "algorithm_root": LaunchConfiguration("algorithm_root"),
                "calib_path": LaunchConfiguration("data_dir"),
                "output_dir": LaunchConfiguration("output_dir"),
            },
        ],
    )

    return LaunchDescription(arguments + [t1_adapter, t2_adapter, mapper])
