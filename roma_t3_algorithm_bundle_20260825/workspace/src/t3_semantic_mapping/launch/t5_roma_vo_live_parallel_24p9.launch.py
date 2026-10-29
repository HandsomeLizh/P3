from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from t3_semantic_mapping.roma_runtime_defaults import launch_default


def generate_launch_description():
    home = Path.home()
    mapping_share = Path(
        get_package_share_directory("t3_semantic_mapping")
    )

    arguments = [
        DeclareLaunchArgument(
            "calib_path",
            default_value=str(
                mapping_share / "config" / "t5_rgb10_rgb11.yaml"
            ),
            description=(
                "Measured RGB10/RGB11 stereo calibration file or directory."
            ),
        ),
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
        DeclareLaunchArgument(
            "left_topic",
            default_value="/Car/T5/Cam_Left/image_raw/color",
        ),
        DeclareLaunchArgument(
            "right_topic",
            default_value="/Car/T5/Cam_Right/image_raw/color",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/Car/T5/wit/imu",
        ),
        DeclareLaunchArgument(
            "semantic_topic",
            default_value="/task2/seg_mask",
        ),
        DeclareLaunchArgument(
            "enable_semantics",
            default_value="false",
            description=(
                "Subscribe to the external semantic mask. When false, RoMa "
                "runs from stereo only and uses zero labels internally."
            ),
        ),
        DeclareLaunchArgument(
            "launch_basic_semantics",
            default_value="false",
            description=(
                "Launch the development-only basic semantic producer. Keep "
                "false when task 2 publishes /task2/seg_mask."
            ),
        ),
        DeclareLaunchArgument(
            "image_timestamp_source",
            default_value="sensor",
            description=(
                "sensor preserves acquisition Header.stamp; arrival is an "
                "explicit compatibility mode for publishers with bad stamps."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "Use the standard ROS /clock time source. Set true together "
                "with image_timestamp_source:=arrival when replaying a "
                "simulated publishers that provide /clock."
            ),
        ),
        DeclareLaunchArgument(
            "enable_imu",
            default_value="false",
            description=(
                "Use the measured IMU-to-camera rotation as a gyro-derived "
                "PnP initial guess. Disabled by default because this is not "
                "a full visual-inertial fusion backend."
            ),
        ),
        DeclareLaunchArgument(
            "enable_pointcloud_fusion",
            default_value="false",
            description=(
                "Fuse OS1 and left/right TOF PointCloud2 height evidence "
                "after each accepted RoMa map keyframe."
            ),
        ),
        DeclareLaunchArgument(
            "pointcloud_fusion_config",
            default_value=str(
                mapping_share / "config" / "pointcloud_extrinsic.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "image_sync_mode",
            default_value="approximate",
            description="Use exact or approximate three-image timestamp sync.",
        ),
        DeclareLaunchArgument(
            "image_sync_slop_sec", default_value="0.03"
        ),
        DeclareLaunchArgument(
            "stereo_sync_max_skew_sec", default_value="0.01"
        ),
        DeclareLaunchArgument(
            "coarse_res",
            default_value=launch_default("coarse_res"),
            description="RoMa coarse inference resolution (multiple of 14).",
        ),
        DeclareLaunchArgument(
            "upsample_res",
            default_value=launch_default("upsample_res"),
            description="RoMa refinement resolution (multiple of 14).",
        ),
        DeclareLaunchArgument(
            "stereo_upsample_refinement",
            default_value=launch_default("stereo_upsample_refinement"),
            description="Refine left/right stereo matches with a second pass.",
        ),
        DeclareLaunchArgument(
            "temporal_upsample_refinement",
            default_value=launch_default("temporal_upsample_refinement"),
            description="Refine keyframe/current temporal matches.",
        ),
        DeclareLaunchArgument(
            "symmetric_matching",
            default_value=launch_default("symmetric_matching"),
            description="Run bidirectional RoMa matching.",
        ),
        DeclareLaunchArgument(
            "use_custom_corr",
            default_value=launch_default("use_custom_corr"),
            description="Use the installed fused CUDA local correlation.",
        ),
        DeclareLaunchArgument(
            "enable_feature_reuse",
            default_value=launch_default("enable_feature_reuse"),
            description=(
                "Reuse deterministic RoMa image features across stereo, "
                "temporal, and equal-resolution refinement passes."
            ),
        ),
        DeclareLaunchArgument(
            "batch_coarse_pairs",
            default_value=launch_default("batch_coarse_pairs"),
            description=(
                "Batch stereo and temporal RoMa coarse pairs in one model "
                "call. Validated on the target Orin; disable for strict "
                "legacy comparison."
            ),
        ),
        DeclareLaunchArgument(
            "enable_klt_temporal_tracking",
            default_value=launch_default("enable_klt_temporal_tracking"),
            description=(
                "Use guarded forward/backward KLT for temporal tracking and "
                "fall back to cached-feature RoMa on failure. Disabled by "
                "default so temporal matching uses RoMa on every frame."
            ),
        ),
        DeclareLaunchArgument(
            "incremental_publish_mode",
            default_value="fixed",
            description=(
                "Legacy IncrementalSemanticMap mode; external publication is "
                "disabled by default in favour of local GridMap plus queries."
            ),
        ),
        DeclareLaunchArgument(
            "incremental_publish_period_sec",
            default_value="1.0",
            description=(
                "Minimum period between coalesced full-tile delta batches; "
                "map fusion itself still runs on every synchronized frame."
            ),
        ),
        DeclareLaunchArgument(
            "dense_map_window_mode",
            default_value="rolling",
            description=(
                "Use a rolling 64x64 m local GridMap centred near the robot."
            ),
        ),
        DeclareLaunchArgument(
            "map_length_x",
            default_value="64.0",
            description="Width of the compatibility GridMap window in metres.",
        ),
        DeclareLaunchArgument(
            "map_length_y",
            default_value="64.0",
            description="Height of the compatibility GridMap window in metres.",
        ),
        DeclareLaunchArgument(
            "full_global_map",
            default_value="true",
            description=(
                "Publish the complete high-resolution global GridMap and "
                "save its dense NPZ matrix. Set false for large simulations; "
                "the rolling local map, global overview and sparse SQLite "
                "map remain enabled."
            ),
        ),
        DeclareLaunchArgument(
            "publish_layer_images",
            default_value="false",
            description=(
                "Publish optional semantic/elevation visualization images."
            ),
        ),
        DeclareLaunchArgument(
            "publish_map_pointcloud",
            default_value="false",
            description="Publish the optional accumulated VO debug cloud.",
        ),
        DeclareLaunchArgument(
            "path_publish_stride",
            default_value="10",
            description=(
                "Publish the accumulated debug Path every N poses; Odometry "
                "and map fusion remain per-frame."
            ),
        ),
        DeclareLaunchArgument(
            "publish_feature_matches",
            default_value="true",
            description=(
                "Advertise the reused stereo/temporal match debug image. "
                "Rendering occurs only while a subscriber is connected."
            ),
        ),
        DeclareLaunchArgument(
            "feature_matches_topic",
            default_value="/Car/T2/feature_matching/visualization",
            description=(
                "Task 2 feature-match visualization image topic."
            ),
        ),
        DeclareLaunchArgument(
            "feature_visualization_crop_width",
            default_value="1280",
            description="P1 center-crop width before enhancement resize.",
        ),
        DeclareLaunchArgument(
            "feature_visualization_crop_height",
            default_value="720",
            description="P1 center-crop height before enhancement resize.",
        ),
        DeclareLaunchArgument(
            "feature_visualization_width",
            default_value="640",
            description="P1-compatible feature visualization panel width.",
        ),
        DeclareLaunchArgument(
            "feature_visualization_height",
            default_value="480",
            description="P1-compatible feature visualization panel height.",
        ),
        DeclareLaunchArgument(
            "publish_input_processed",
            default_value="false",
            description=(
                "Publish the per-callback completion receipt used only by "
                "completion-paced offline dataset tools."
            ),
        ),
        DeclareLaunchArgument(
            "publish_realtime_metrics",
            default_value="false",
            description="Publish exact per-frame timing telemetry as JSON.",
        ),
        DeclareLaunchArgument(
            "realtime_metrics_topic",
            default_value="/Car/T3/metrics/frame_timing",
        ),
        DeclareLaunchArgument(
            "enable_legacy_output_aliases",
            default_value="false",
            description=(
                "Also publish the historical duplicate odometry and local "
                "GridMap topic names."
            ),
        ),
        DeclareLaunchArgument(
            "feature_match_output_width",
            default_value="1280",
        ),
        DeclareLaunchArgument(
            "feature_match_max_lines",
            default_value="200",
        ),
    ]

    semantics = Node(
        package="t2_basic_semantics",
        executable="basic_semantic_node",
        name="basic_semantic_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("launch_basic_semantics")),
        parameters=[
            {
                "input_topic": LaunchConfiguration("left_topic"),
                "label_topic": LaunchConfiguration("semantic_topic"),
                "publish_color": False,
                "input_reliability": "best_effort",
                "input_qos_depth": 1,
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
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "algorithm_root": LaunchConfiguration("algorithm_root"),
                "calib_path": LaunchConfiguration("calib_path"),
                "output_dir": LaunchConfiguration("output_dir"),
                "left_topic": LaunchConfiguration("left_topic"),
                "right_topic": LaunchConfiguration("right_topic"),
                "semantic_topic": LaunchConfiguration("semantic_topic"),
                "enable_semantics": LaunchConfiguration("enable_semantics"),
                "image_timestamp_source": LaunchConfiguration(
                    "image_timestamp_source"
                ),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "enable_imu": LaunchConfiguration("enable_imu"),
                "enable_pointcloud_fusion": ParameterValue(
                    LaunchConfiguration("enable_pointcloud_fusion"),
                    value_type=bool,
                ),
                "pointcloud_fusion_config": LaunchConfiguration(
                    "pointcloud_fusion_config"
                ),
                "map_frame": "map",
                "odom_frame": "odom",
                "local_grid_map_frame": "odom",
                "child_frame": "base_link",
                "odometry_camera_frame": "car_stereo_left_optical_frame",
                "odometry_child_transform_confirmed": True,
                # Spreadsheet intrinsics with the confirmed simulation mounting:
                # parallel stereo cameras pitched down 24.9 degrees. This maps
                # ROS FLU base_link into rectified RGB10.
                "odometry_child_to_camera_transform": [
                    0.0, -1.0, 0.0, 0.129759135607494,
                    -0.421035813367491, 0.0,
                    -0.907044014291465, 0.977452854090656,
                    0.907044014291465, 0.0,
                    -0.421035813367491, -0.130422536135199,
                    0.0, 0.0, 0.0, 1.0,
                ],
                # Bag replay publishes reliable images.  A reliable queue
                # prevents either large stereo image from being discarded
                # when both arrive close together at accelerated rates.
                # Live sensor publishers commonly use SensorDataQoS. Keep only
                # the newest synchronized pair so inference cannot lag behind.
                "stereo_input_reliability": "best_effort",
                "semantic_input_reliability": "reliable",
                "imu_input_reliability": "reliable",
                "image_input_qos_depth": 1,
                "imu_input_qos_depth": 10,
                "image_sync_mode": LaunchConfiguration("image_sync_mode"),
                "image_sync_queue_size": 50,
                "image_sync_slop_sec": LaunchConfiguration(
                    "image_sync_slop_sec"
                ),
                "stereo_sync_max_skew_sec": LaunchConfiguration(
                    "stereo_sync_max_skew_sec"
                ),
                "coarse_res": ParameterValue(
                    LaunchConfiguration("coarse_res"), value_type=int
                ),
                "upsample_res": ParameterValue(
                    LaunchConfiguration("upsample_res"), value_type=int
                ),
                "stereo_upsample_refinement": ParameterValue(
                    LaunchConfiguration("stereo_upsample_refinement"),
                    value_type=bool,
                ),
                "temporal_upsample_refinement": ParameterValue(
                    LaunchConfiguration("temporal_upsample_refinement"),
                    value_type=bool,
                ),
                "symmetric_matching": ParameterValue(
                    LaunchConfiguration("symmetric_matching"),
                    value_type=bool,
                ),
                "use_custom_corr": ParameterValue(
                    LaunchConfiguration("use_custom_corr"), value_type=bool
                ),
                "enable_feature_reuse": ParameterValue(
                    LaunchConfiguration("enable_feature_reuse"),
                    value_type=bool,
                ),
                "batch_coarse_pairs": ParameterValue(
                    LaunchConfiguration("batch_coarse_pairs"),
                    value_type=bool,
                ),
                "enable_klt_temporal_tracking": ParameterValue(
                    LaunchConfiguration("enable_klt_temporal_tracking"),
                    value_type=bool,
                ),
                "incremental_publish_mode": LaunchConfiguration(
                    "incremental_publish_mode"
                ),
                "incremental_publish_period_sec": ParameterValue(
                    LaunchConfiguration("incremental_publish_period_sec"),
                    value_type=float,
                ),
                "dense_map_window_mode": LaunchConfiguration(
                    "dense_map_window_mode"
                ),
                "map_length_x": ParameterValue(
                    LaunchConfiguration("map_length_x"), value_type=float
                ),
                "map_length_y": ParameterValue(
                    LaunchConfiguration("map_length_y"), value_type=float
                ),
                "publish_global_grid_map": ParameterValue(
                    LaunchConfiguration("full_global_map"),
                    value_type=bool,
                ),
                "save_dense_global_grid_map": ParameterValue(
                    LaunchConfiguration("full_global_map"),
                    value_type=bool,
                ),
                "publish_layer_images": ParameterValue(
                    LaunchConfiguration("publish_layer_images"),
                    value_type=bool,
                ),
                "publish_map_pointcloud": ParameterValue(
                    LaunchConfiguration("publish_map_pointcloud"),
                    value_type=bool,
                ),
                "path_publish_stride": ParameterValue(
                    LaunchConfiguration("path_publish_stride"),
                    value_type=int,
                ),
                "publish_feature_matches": ParameterValue(
                    LaunchConfiguration("publish_feature_matches"),
                    value_type=bool,
                ),
                "feature_matches_topic": LaunchConfiguration(
                    "feature_matches_topic"
                ),
                "feature_visualization_crop_width": ParameterValue(
                    LaunchConfiguration("feature_visualization_crop_width"),
                    value_type=int,
                ),
                "feature_visualization_crop_height": ParameterValue(
                    LaunchConfiguration("feature_visualization_crop_height"),
                    value_type=int,
                ),
                "feature_visualization_width": ParameterValue(
                    LaunchConfiguration("feature_visualization_width"),
                    value_type=int,
                ),
                "feature_visualization_height": ParameterValue(
                    LaunchConfiguration("feature_visualization_height"),
                    value_type=int,
                ),
                "publish_input_processed": ParameterValue(
                    LaunchConfiguration("publish_input_processed"),
                    value_type=bool,
                ),
                "publish_realtime_metrics": ParameterValue(
                    LaunchConfiguration("publish_realtime_metrics"),
                    value_type=bool,
                ),
                "realtime_metrics_topic": LaunchConfiguration(
                    "realtime_metrics_topic"
                ),
                "publish_odometry_compat": ParameterValue(
                    LaunchConfiguration("enable_legacy_output_aliases"),
                    value_type=bool,
                ),
                "publish_grid_map_odom_alias": ParameterValue(
                    LaunchConfiguration("enable_legacy_output_aliases"),
                    value_type=bool,
                ),
                "feature_match_output_width": ParameterValue(
                    LaunchConfiguration("feature_match_output_width"),
                    value_type=int,
                ),
                "feature_match_max_lines": ParameterValue(
                    LaunchConfiguration("feature_match_max_lines"),
                    value_type=int,
                ),
                "localization_mode": "vo",
                "fusion_backend": "none",
                "enable_global_loop_closure": False,
                "enable_lidar_mapping": False,
                "lidar_fallback_to_stereo": False,
            },
        ],
    )

    return LaunchDescription(arguments + [semantics, mapper])
