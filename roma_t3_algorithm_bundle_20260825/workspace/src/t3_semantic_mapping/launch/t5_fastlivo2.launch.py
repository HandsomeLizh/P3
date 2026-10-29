from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    home = Path.home()
    mapping_share = Path(get_package_share_directory("t3_semantic_mapping"))

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("with_gridmap", default_value="true"),
        DeclareLaunchArgument(
            "full_global_map",
            default_value="true",
            description=(
                "Publish the complete high-resolution global GridMap and "
                "save dense NPZ. Set false for large simulations while "
                "retaining local GridMap, global overview and SQLite."
            ),
        ),
        DeclareLaunchArgument(
            "fastlivo_params_file",
            default_value=str(mapping_share / "config" / "t5_fastlivo2.yaml"),
        ),
        DeclareLaunchArgument(
            "camera_params_file",
            default_value=str(
                mapping_share / "config" / "t5_fastlivo2_camera.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "calib_path",
            default_value=str(mapping_share / "config" / "t5_rgb10_rgb11.yaml"),
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value=EnvironmentVariable(
                "T3_OUTPUT_DIR",
                default_value=str(home / "P3" / "output"),
            ),
        ),
        DeclareLaunchArgument(
            "left_topic", default_value="/Car/T5/Cam_Left/image_raw/color"
        ),
        DeclareLaunchArgument(
            "right_topic", default_value="/Car/T5/Cam_Right/image_raw/color"
        ),
        DeclareLaunchArgument("semantic_topic", default_value="/task2/seg_mask"),
        DeclareLaunchArgument("lidar_topic", default_value="/Car/T5/OS1/points"),
        DeclareLaunchArgument("imu_topic", default_value="/Car/T5/OS1/imu"),
    ]

    camera_parameters = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="parameter_blackboard",
        parameters=[LaunchConfiguration("camera_params_file")],
        output="screen",
    )

    fastlivo = Node(
        package="fast_livo",
        executable="fastlivo_mapping",
        name="laserMapping",
        parameters=[
            LaunchConfiguration("fastlivo_params_file"),
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "common.img_topic": LaunchConfiguration("left_topic"),
                "common.lid_topic": LaunchConfiguration("lidar_topic"),
                "common.imu_topic": LaunchConfiguration("imu_topic"),
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    # The T3 mapper consumes FAST-LIVO2 odometry and raw Ouster geometry.  In
    # direct-LiDAR mode it bypasses RoMa, while left/right/mask synchronization
    # still determines when semantic observations enter the 2.5D grid.
    mapper = Node(
        condition=IfCondition(LaunchConfiguration("with_gridmap")),
        package="t3_semantic_mapping",
        executable="roma_semantic_mapper",
        name="roma_semantic_mapper_node",
        parameters=[
            str(mapping_share / "config" / "semantic_map.yaml"),
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "calib_path": LaunchConfiguration("calib_path"),
                "output_dir": LaunchConfiguration("output_dir"),
                "left_topic": LaunchConfiguration("left_topic"),
                "right_topic": LaunchConfiguration("right_topic"),
                "semantic_topic": LaunchConfiguration("semantic_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "enable_imu": False,
                "map_frame": "map",
                "odom_frame": "odom",
                "local_grid_map_frame": "odom",
                "child_frame": "base_link",
                "odometry_camera_frame": "car_stereo_left_optical_frame",
                "odometry_child_transform_confirmed": True,
                # Latest 2026-08-17 calibration. The measured vehicle frame
                # is FRD; this maps ROS FLU base_link into rectified RGB10.
                "odometry_child_to_camera_transform": [
                    0.007669960290140, -0.999891919173566,
                    0.012542793969095, 0.115249653386463,
                    -0.467167589008267, -0.014673162435303,
                    -0.884047024815055, 0.984673225690091,
                    0.884135518735308, 0.000921018756979,
                    -0.467229639722373, -0.079985681086572,
                    0.0, 0.0, 0.0, 1.0,
                ],
                "stereo_input_reliability": "best_effort",
                "semantic_input_reliability": "reliable",
                "image_input_qos_depth": 5,
                "image_sync_mode": "approximate",
                "image_sync_queue_size": 100,
                "image_sync_slop_sec": 0.05,
                "stereo_sync_max_skew_sec": 0.02,
                "localization_mode": "lio",
                "fusion_backend": "none",
                "enable_gp_prior": False,
                "lio_odometry_topic": "/aft_mapped_to_init",
                "lio_expected_parent_frame": "camera_init",
                "lio_expected_child_frame": "aft_mapped",
                "lio_extrinsics_path": LaunchConfiguration("calib_path"),
                "lio_child_to_lidar_confirmed": True,
                # p_sensor = T_sensor_imu * p_imu from Ouster metadata.
                "lio_child_to_lidar_transform": [
                    1.0, 0.0, 0.0, -0.002441,
                    0.0, 1.0, 0.0, -0.009725,
                    0.0, 0.0, 1.0, 0.007533,
                    0.0, 0.0, 0.0, 1.0,
                ],
                "lio_max_time_offset_sec": 0.20,
                "lio_wait_timeout_sec": 0.50,
                "enable_global_loop_closure": False,
                "enable_lidar_mapping": True,
                "lidar_mapping_mode": "direct",
                "lidar_align_map_to_sensor_frame": False,
                "lidar_extrinsics_path": LaunchConfiguration("calib_path"),
                "lidar_expected_frame_id": "sensor_frame",
                "lidar_max_time_offset_sec": 0.20,
                "lidar_wait_timeout_sec": 1.0,
                "lidar_temporal_mode": "none",
                "lidar_fallback_to_stereo": False,
                "lidar_voxel_size_m": 0.20,
                "dense_map_window_mode": "rolling",
                "map_resolution": 0.20,
                "map_length_x": 64.0,
                "map_length_y": 64.0,
                "publish_global_grid_map": ParameterValue(
                    LaunchConfiguration("full_global_map"),
                    value_type=bool,
                ),
                "save_dense_global_grid_map": ParameterValue(
                    LaunchConfiguration("full_global_map"),
                    value_type=bool,
                ),
                "publish_feature_matches": True,
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(args + [camera_parameters, fastlivo, mapper])
