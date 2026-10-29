from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from t3_semantic_mapping.launch_options import (
    dataset_playback_launch_arguments,
    dataset_playback_parameter_overrides,
    dataset_lio_parameter_overrides,
    input_launch_arguments,
    input_parameter_overrides,
    input_topic_parameter_overrides,
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
        DeclareLaunchArgument(
            "data_dir",
            default_value=EnvironmentVariable(
                "T3_DATA_DIR",
                default_value=str(home / "RoMa" / "data" / "data1223"),
            ),
        ),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("max_stereo_skew_ms", default_value="20.0"),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("start_delay_sec", default_value="8.0"),
        DeclareLaunchArgument(
            "algorithm_root",
            default_value=EnvironmentVariable(
                "T3_ALGORITHM_ROOT",
                default_value=str(home / "RoMa"),
            ),
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value=EnvironmentVariable(
                "T3_OUTPUT_DIR",
                default_value=str(home / "P3" / "output"),
            ),
        ),
        DeclareLaunchArgument("lidar_publish_stride", default_value="1"),
        DeclareLaunchArgument(
            "lidar_publish_max_points", default_value="5000"
        ),
    ]
    arguments.extend(input_launch_arguments())
    arguments.extend(dataset_playback_launch_arguments())
    arguments.extend(
        localization_launch_arguments(include_dataset_lio=True)
    )
    arguments.extend(mapping_launch_arguments())

    publisher = Node(
        package="t1_dataset_sim",
        executable="dataset_sensor_publisher",
        name="dataset_sensor_publisher",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                **dataset_playback_parameter_overrides(),
                **dataset_lio_parameter_overrides(),
                **input_topic_parameter_overrides(),
                "data_dir": LaunchConfiguration("data_dir"),
                "use_precomputed_semantics": False,
                "max_frames": LaunchConfiguration("max_frames"),
                "max_stereo_skew_ms": LaunchConfiguration(
                    "max_stereo_skew_ms"
                ),
                "playback_rate": LaunchConfiguration("playback_rate"),
                "start_delay_sec": LaunchConfiguration("start_delay_sec"),
                "playback_summary_path": PathJoinSubstitution([
                    LaunchConfiguration("output_dir"),
                    "dataset_playback_summary.json",
                ]),
                "publish_imu": LaunchConfiguration("enable_imu"),
                "publish_lidar": LaunchConfiguration(
                    "enable_lidar_mapping"
                ),
                "lidar_publish_stride": LaunchConfiguration(
                    "lidar_publish_stride"
                ),
                "lidar_publish_max_points": LaunchConfiguration(
                    "lidar_publish_max_points"
                ),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "lidar_frame_id": LaunchConfiguration(
                    "lidar_expected_frame_id"
                ),
            }
        ],
    )

    semantics = Node(
        package="t2_basic_semantics",
        executable="basic_semantic_node",
        name="basic_semantic_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "input_topic": LaunchConfiguration("left_topic"),
                "label_topic": LaunchConfiguration("semantic_topic"),
                "publish_color": False,
            }
        ],
    )

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
                "algorithm_root": LaunchConfiguration("algorithm_root"),
                "calib_path": LaunchConfiguration("data_dir"),
                "output_dir": LaunchConfiguration("output_dir"),
                "publish_input_processed": True,
            },
        ],
    )

    return LaunchDescription(arguments + [publisher, semantics, mapper])
