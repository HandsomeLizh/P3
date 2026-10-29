from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
    home = Path.home()
    mapping_share = Path(
        get_package_share_directory("t3_semantic_mapping")
    )

    arguments = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "data_dir",
            default_value=EnvironmentVariable(
                "T3_DATA_DIR",
                default_value=str(home / "RoMa" / "data" / "data1223"),
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
                default_value=str(home / "RoMa"),
            ),
        ),
    ]
    arguments.extend(input_launch_arguments())
    arguments.extend(localization_launch_arguments())
    arguments.extend(mapping_launch_arguments())

    mapper = Node(
        package="t3_semantic_mapping",
        executable="roma_semantic_mapper",
        name="roma_semantic_mapper_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            str(mapping_share / "config" / "semantic_map.yaml"),
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

    return LaunchDescription(arguments + [mapper])
