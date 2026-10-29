"""Shared ROS 2 launch arguments for the task-three mapper."""

from typing import List

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def input_launch_arguments():
    """Stable standard-message topics plus configurable DDS transport."""

    return [
        DeclareLaunchArgument(
            "left_topic",
            default_value="/Car/T1/fusion/stereo/left/image_raw",
        ),
        DeclareLaunchArgument(
            "right_topic",
            default_value="/Car/T1/fusion/stereo/right/image_raw",
        ),
        DeclareLaunchArgument(
            "semantic_topic",
            default_value="/Car/T2/semantic/label_image",
        ),
        DeclareLaunchArgument(
            "imu_topic", default_value="/Car/T1/fusion/imu/data"
        ),
        DeclareLaunchArgument(
            "input_processed_topic",
            default_value="/Car/T3/debug/input_processed",
            description=(
                "Standard Header receipt emitted after T3 finishes each "
                "synchronized image callback"
            ),
        ),
        DeclareLaunchArgument(
            "enable_imu",
            default_value="true",
            description="Subscribe to and use IMU measurements in visual SLAM",
        ),
        DeclareLaunchArgument(
            "stereo_input_reliability",
            default_value="reliable",
            description="Stereo subscription reliability: reliable/best_effort",
        ),
        DeclareLaunchArgument(
            "semantic_input_reliability",
            default_value="reliable",
            description=(
                "Semantic-image subscription reliability: "
                "reliable/best_effort"
            ),
        ),
        DeclareLaunchArgument(
            "imu_input_reliability",
            default_value="reliable",
            description="IMU subscription reliability: reliable/best_effort",
        ),
        DeclareLaunchArgument("image_input_qos_depth", default_value="10"),
        DeclareLaunchArgument("imu_input_qos_depth", default_value="10"),
        DeclareLaunchArgument(
            "image_sync_mode",
            default_value="exact",
            description="Header.stamp matching: exact or approximate",
        ),
        DeclareLaunchArgument("image_sync_queue_size", default_value="50"),
        DeclareLaunchArgument(
            "image_sync_slop_sec",
            default_value="0.02",
            description="Maximum three-image stamp span in approximate mode",
        ),
        DeclareLaunchArgument(
            "stereo_sync_max_skew_sec",
            default_value="0.0",
            description=(
                "Maximum left/right acquisition skew in approximate mode; "
                "zero preserves exact stereo pairs"
            ),
        ),
    ]


def input_topic_parameter_overrides():
    return {
        "left_topic": LaunchConfiguration("left_topic"),
        "right_topic": LaunchConfiguration("right_topic"),
        "semantic_topic": LaunchConfiguration("semantic_topic"),
        "imu_topic": LaunchConfiguration("imu_topic"),
        "input_processed_topic": LaunchConfiguration(
            "input_processed_topic"
        ),
    }


def input_parameter_overrides():
    return {
        **input_topic_parameter_overrides(),
        "enable_imu": LaunchConfiguration("enable_imu"),
        "stereo_input_reliability": LaunchConfiguration(
            "stereo_input_reliability"
        ),
        "semantic_input_reliability": LaunchConfiguration(
            "semantic_input_reliability"
        ),
        "imu_input_reliability": LaunchConfiguration(
            "imu_input_reliability"
        ),
        "image_input_qos_depth": LaunchConfiguration(
            "image_input_qos_depth"
        ),
        "imu_input_qos_depth": LaunchConfiguration(
            "imu_input_qos_depth"
        ),
        "image_sync_mode": LaunchConfiguration("image_sync_mode"),
        "image_sync_queue_size": LaunchConfiguration(
            "image_sync_queue_size"
        ),
        "image_sync_slop_sec": LaunchConfiguration(
            "image_sync_slop_sec"
        ),
        "stereo_sync_max_skew_sec": LaunchConfiguration(
            "stereo_sync_max_skew_sec"
        ),
    }


def dataset_playback_launch_arguments():
    """Completion-aware controls used only by dataset practice launchers."""

    return [
        DeclareLaunchArgument(
            "lossless_playback",
            default_value="true",
            description=(
                "Wait for T3 input completion before publishing the next "
                "image frame; non-image sensor events keep their order"
            ),
        ),
        DeclareLaunchArgument(
            "processing_ack_timeout_sec",
            default_value="120.0",
            description=(
                "Fail-stop timeout for T3 readiness or one input completion "
                "receipt"
            ),
        ),
        DeclareLaunchArgument(
            "lossless_max_in_flight_frames",
            default_value="2",
            description=(
                "Bounded image pipeline depth; every frame still requires "
                "its own T3 completion receipt"
            ),
        ),
    ]


def dataset_playback_parameter_overrides():
    return {
        "lossless_playback": LaunchConfiguration("lossless_playback"),
        "processing_ack_timeout_sec": LaunchConfiguration(
            "processing_ack_timeout_sec"
        ),
        "lossless_max_in_flight_frames": LaunchConfiguration(
            "lossless_max_in_flight_frames"
        ),
    }


def localization_launch_arguments(*, include_dataset_lio: bool = False):
    arguments = [
        DeclareLaunchArgument(
            "localization_mode",
            default_value="vo",
            description="Localisation source: vo, lio, or vo_lio",
        ),
        DeclareLaunchArgument(
            "fusion_backend",
            default_value="none",
            description="Local fusion backend: none or gtsam_gp",
        ),
        DeclareLaunchArgument("enable_gp_prior", default_value="true"),
        DeclareLaunchArgument("fusion_allow_vo_only", default_value="true"),
        DeclareLaunchArgument(
            "lio_odometry_topic", default_value="/Odometry"
        ),
        DeclareLaunchArgument(
            "lio_expected_parent_frame", default_value="camera_init"
        ),
        DeclareLaunchArgument(
            "lio_expected_child_frame", default_value="body"
        ),
        DeclareLaunchArgument("lio_extrinsics_path", default_value=""),
        DeclareLaunchArgument(
            "lio_child_to_lidar_confirmed",
            default_value="false",
            description=(
                "Explicit acknowledgement that T_lidar_child matches the "
                "FAST-LIO2 body/IMU extrinsic"
            ),
        ),
        DeclareLaunchArgument(
            "lio_child_to_lidar_transform",
            default_value=(
                "[1.0, 0.0, 0.0, 0.0, "
                "0.0, 1.0, 0.0, 0.0, "
                "0.0, 0.0, 1.0, 0.0, "
                "0.0, 0.0, 0.0, 1.0]"
            ),
            description=(
                "Row-major T_lidar_child, p_lidar=T_lidar_child*p_child"
            ),
        ),
        DeclareLaunchArgument(
            "lio_max_time_offset_sec", default_value="0.15"
        ),
        DeclareLaunchArgument("lio_wait_timeout_sec", default_value="0.20"),
        DeclareLaunchArgument(
            "odometry_camera_frame",
            default_value="car_stereo_left_optical_frame",
            description=(
                "Physical camera frame estimated by the localisation state"
            ),
        ),
        DeclareLaunchArgument(
            "child_frame",
            default_value="car_stereo_left_optical_frame",
            description=(
                "Published Odometry/TF child; a different physical frame "
                "requires a confirmed T_camera_child"
            ),
        ),
        DeclareLaunchArgument(
            "odometry_child_transform_confirmed",
            default_value="false",
            description=(
                "Explicit acknowledgement that T_camera_child is calibrated"
            ),
        ),
        DeclareLaunchArgument(
            "odometry_child_to_camera_transform",
            default_value=(
                "[1.0, 0.0, 0.0, 0.0, "
                "0.0, 1.0, 0.0, 0.0, "
                "0.0, 0.0, 1.0, 0.0, "
                "0.0, 0.0, 0.0, 1.0]"
            ),
            description=(
                "Row-major T_camera_child, p_camera=T_camera_child*p_child"
            ),
        ),
        DeclareLaunchArgument(
            "odometry_velocity_max_interval_sec", default_value="1.0"
        ),
        DeclareLaunchArgument(
            "odometry_unknown_covariance_variance", default_value="1.0e6"
        ),
        DeclareLaunchArgument(
            "enable_global_loop_closure",
            default_value="false",
            description=(
                "Enable the independent verified global loop layer for "
                "lio/vo_lio; pure VO remains unchanged"
            ),
        ),
        DeclareLaunchArgument(
            "global_loop_candidate_source",
            default_value="mprf",
            description="Loop proposals: lidar, mprf, or hybrid",
        ),
        DeclareLaunchArgument(
            "global_loop_confirmation_count",
            default_value="2",
            description="Require one or two temporally consistent verified loops",
        ),
        DeclareLaunchArgument(
            "mprf_endpoint", default_value="http://127.0.0.1:8765"
        ),
        DeclareLaunchArgument(
            "mprf_request_timeout_sec", default_value="20.0"
        ),
        DeclareLaunchArgument("mprf_queue_size", default_value="8"),
        DeclareLaunchArgument("mprf_transient_retries", default_value="2"),
        DeclareLaunchArgument("mprf_poll_period_sec", default_value="0.10"),
        DeclareLaunchArgument(
            "mprf_auth_token_env", default_value="MPRF_SERVICE_TOKEN"
        ),
    ]
    if include_dataset_lio:
        arguments.extend(
            [
                DeclareLaunchArgument(
                    "publish_lio_odometry",
                    default_value="false",
                    description=(
                        "Replay saved lidar_fastlio/trajectory.npz only for "
                        "ROS interface and fusion integration tests"
                    ),
                ),
                DeclareLaunchArgument(
                    "lio_publish_stride", default_value="1"
                ),
            ]
        )
    return arguments


def localization_parameter_overrides():
    return {
        "localization_mode": LaunchConfiguration("localization_mode"),
        "fusion_backend": LaunchConfiguration("fusion_backend"),
        "enable_gp_prior": LaunchConfiguration("enable_gp_prior"),
        "fusion_allow_vo_only": LaunchConfiguration(
            "fusion_allow_vo_only"
        ),
        "lio_odometry_topic": LaunchConfiguration("lio_odometry_topic"),
        "lio_expected_parent_frame": LaunchConfiguration(
            "lio_expected_parent_frame"
        ),
        "lio_expected_child_frame": LaunchConfiguration(
            "lio_expected_child_frame"
        ),
        "lio_extrinsics_path": ParameterValue(
            LaunchConfiguration("lio_extrinsics_path"), value_type=str
        ),
        "lio_child_to_lidar_confirmed": LaunchConfiguration(
            "lio_child_to_lidar_confirmed"
        ),
        "lio_child_to_lidar_transform": ParameterValue(
            LaunchConfiguration("lio_child_to_lidar_transform"),
            value_type=List[float],
        ),
        "lio_max_time_offset_sec": LaunchConfiguration(
            "lio_max_time_offset_sec"
        ),
        "lio_wait_timeout_sec": LaunchConfiguration(
            "lio_wait_timeout_sec"
        ),
        "odometry_camera_frame": LaunchConfiguration(
            "odometry_camera_frame"
        ),
        "child_frame": LaunchConfiguration("child_frame"),
        "odometry_child_transform_confirmed": LaunchConfiguration(
            "odometry_child_transform_confirmed"
        ),
        "odometry_child_to_camera_transform": ParameterValue(
            LaunchConfiguration("odometry_child_to_camera_transform"),
            value_type=List[float],
        ),
        "odometry_velocity_max_interval_sec": LaunchConfiguration(
            "odometry_velocity_max_interval_sec"
        ),
        "odometry_unknown_covariance_variance": LaunchConfiguration(
            "odometry_unknown_covariance_variance"
        ),
        "enable_global_loop_closure": LaunchConfiguration(
            "enable_global_loop_closure"
        ),
        "global_loop_candidate_source": LaunchConfiguration(
            "global_loop_candidate_source"
        ),
        "global_loop_confirmation_count": LaunchConfiguration(
            "global_loop_confirmation_count"
        ),
        "mprf_endpoint": ParameterValue(
            LaunchConfiguration("mprf_endpoint"), value_type=str
        ),
        "mprf_request_timeout_sec": LaunchConfiguration(
            "mprf_request_timeout_sec"
        ),
        "mprf_queue_size": LaunchConfiguration("mprf_queue_size"),
        "mprf_transient_retries": LaunchConfiguration(
            "mprf_transient_retries"
        ),
        "mprf_poll_period_sec": LaunchConfiguration(
            "mprf_poll_period_sec"
        ),
        "mprf_auth_token_env": ParameterValue(
            LaunchConfiguration("mprf_auth_token_env"), value_type=str
        ),
    }


def dataset_lio_parameter_overrides():
    return {
        "publish_lio_odometry": LaunchConfiguration(
            "publish_lio_odometry"
        ),
        "lio_publish_stride": LaunchConfiguration("lio_publish_stride"),
        "lio_odometry_topic": LaunchConfiguration("lio_odometry_topic"),
        "lio_parent_frame_id": LaunchConfiguration(
            "lio_expected_parent_frame"
        ),
        "lio_child_frame_id": LaunchConfiguration(
            "lio_expected_child_frame"
        ),
    }


def mapping_launch_arguments():
    """Shared map-source, publication and provenance launch controls."""
    return [
        DeclareLaunchArgument(
            "incremental_publish_mode",
            default_value="fixed",
            description=(
                "IncrementalSemanticMap mode: fixed, rolling_window, "
                "or full_tiles"
            ),
        ),
        DeclareLaunchArgument(
            "incremental_publish_period_sec",
            default_value="1.0",
            description=(
                "Minimum period between coalesced full-tile delta batches; "
                "map fusion itself remains per-frame"
            ),
        ),
        DeclareLaunchArgument(
            "dense_map_window_mode",
            default_value="rolling",
            description=(
                "Dense GridMap window: follow_incremental, fixed, or rolling"
            ),
        ),
        DeclareLaunchArgument(
            "incremental_map_batch_topic",
            default_value="/Car/T3/semantic/map_batch",
            description=(
                "Companion BEGIN/COMPLETE metadata for full_tiles payloads"
            ),
        ),
        DeclareLaunchArgument(
            "incremental_map_replay_service",
            default_value="/Car/T3/mapping/replay_full_map",
            description="Trigger a frozen complete full_tiles replay",
        ),
        DeclareLaunchArgument(
            "publish_unchanged_map_outputs",
            default_value="false",
            description=(
                "Republish an unchanged map window on every valid pose; "
                "normally disabled because map outputs are transient-local"
            ),
        ),
        DeclareLaunchArgument(
            "publish_layer_images",
            default_value="false",
            description="Publish semantic/elevation visualization images",
        ),
        DeclareLaunchArgument(
            "publish_map_pointcloud",
            default_value="false",
            description="Publish the map debug PointCloud2",
        ),
        DeclareLaunchArgument(
            "path_publish_stride",
            default_value="1",
            description=(
                "Publish the complete debug Path every N valid poses; "
                "Odometry remains per-pose"
            ),
        ),
        DeclareLaunchArgument(
            "stereo_map_max_depth_m",
            default_value="5.0",
            description=(
                "Maximum pure-stereo optical depth fused into the map; "
                "zero disables the map-only gate"
            ),
        ),
        DeclareLaunchArgument(
            "keyframe_interval",
            default_value="5",
            description=(
                "Map/visual keyframe interval; direct LIO counts accepted "
                "LIO poses and always includes the first valid pose"
            ),
        ),
        DeclareLaunchArgument(
            "enable_lidar_mapping",
            default_value="false",
            description=(
                "Use PointCloud2 geometry for 2.5D mapping independently "
                "of the selected localisation source"
            ),
        ),
        DeclareLaunchArgument(
            "lidar_mapping_mode",
            default_value="direct",
            description="PointCloud2 map mode: direct or assist",
        ),
        DeclareLaunchArgument(
            "lidar_topic", default_value="/Car/T1/fusion/lidar/points"
        ),
        DeclareLaunchArgument(
            "lidar_extrinsics_path",
            default_value="",
            description=(
                "Optional extrinsics.json containing LiDAR_to_Cam_L.RT; "
                "empty discovers it from data_dir"
            ),
        ),
        DeclareLaunchArgument(
            "lidar_expected_frame_id", default_value="velodyne"
        ),
        DeclareLaunchArgument(
            "lidar_max_time_offset_sec", default_value="0.15"
        ),
        DeclareLaunchArgument(
            "lidar_wait_timeout_sec", default_value="1.0"
        ),
        DeclareLaunchArgument(
            "lidar_temporal_mode",
            default_value="none",
            description=(
                "LiDAR timing contract: none, upstream_deskewed, or "
                "per_point"
            ),
        ),
        DeclareLaunchArgument(
            "lidar_point_time_field",
            default_value="time",
            description=(
                "Scalar PointCloud2 field containing a time offset from "
                "header.stamp in per_point mode"
            ),
        ),
        DeclareLaunchArgument(
            "lidar_point_time_unit",
            default_value="sec",
            description="Unit of the header-relative point-time field",
        ),
        DeclareLaunchArgument(
            "lidar_point_time_max_abs_sec", default_value="0.20"
        ),
        DeclareLaunchArgument(
            "lidar_deskew_time_bin_sec",
            default_value="0.001",
            description=(
                "Pose-query bin for per-point compensation; zero uses every "
                "distinct point time"
            ),
        ),
        DeclareLaunchArgument("lidar_min_range_m", default_value="0.5"),
        DeclareLaunchArgument("lidar_max_range_m", default_value="30.0"),
        DeclareLaunchArgument("lidar_voxel_size_m", default_value="0.10"),
        DeclareLaunchArgument(
            "lidar_max_points_per_keyframe", default_value="5000"
        ),
        DeclareLaunchArgument(
            "lidar_min_points_per_keyframe", default_value="100"
        ),
        DeclareLaunchArgument("lidar_buffer_size", default_value="20"),
        DeclareLaunchArgument(
            "lidar_ground_max_slope_deg",
            default_value="35.0",
            description="Maximum slope of a direct-LiDAR ground surface",
        ),
        DeclareLaunchArgument(
            "lidar_ground_neighbor_count",
            default_value="12",
            description="Neighbour count for direct-LiDAR local normals",
        ),
        DeclareLaunchArgument(
            "lidar_ground_max_neighbor_distance_m",
            default_value="0.75",
            description=(
                "Maximum local support radius for direct-LiDAR endpoint "
                "classification"
            ),
        ),
        DeclareLaunchArgument(
            "lidar_fallback_to_stereo",
            default_value="false",
            description=(
                "Use bounded stereo geometry when a keyframe lacks a "
                "time-valid LiDAR cloud"
            ),
        ),
        DeclareLaunchArgument(
            "lidar_align_map_to_sensor_frame",
            default_value="true",
            description=(
                "Align map axes to the calibrated REP-103 LiDAR frame"
            ),
        ),
        DeclareLaunchArgument(
            "pointcloud_assist_pixel_radius", default_value="2.0"
        ),
        DeclareLaunchArgument(
            "pointcloud_assist_unmatched_stereo_max_depth_m",
            default_value="5.0",
        ),
        DeclareLaunchArgument(
            "pointcloud_assist_replace_stereo_min_depth_m",
            default_value="5.0",
        ),
        DeclareLaunchArgument(
            "pointcloud_assist_min_matches", default_value="20"
        ),
        DeclareLaunchArgument(
            "pointcloud_assist_max_points_per_keyframe",
            default_value="20000",
        ),
        DeclareLaunchArgument(
            "map_admission_enabled",
            default_value="true",
            description=(
                "Protect pure-VO map evidence after tracking loss; pose "
                "publication remains unchanged"
            ),
        ),
        DeclareLaunchArgument(
            "map_admission_min_recovery_inliers",
            default_value="25",
        ),
        DeclareLaunchArgument(
            "map_admission_min_recovery_inlier_ratio",
            default_value="0.25",
        ),
        DeclareLaunchArgument(
            "map_admission_stable_frames",
            default_value="3",
        ),
        DeclareLaunchArgument(
            "map_admission_max_translation_step_m",
            default_value="1.0",
        ),
        DeclareLaunchArgument(
            "map_admission_max_rotation_step_deg",
            default_value="20.0",
        ),
        DeclareLaunchArgument(
            "map_observation_archive_enabled",
            default_value="true",
            description=(
                "Losslessly page keyframe map observations to output_dir; "
                "disabling restores the old unbounded-RAM store"
            ),
        ),
        DeclareLaunchArgument(
            "map_observation_cache_size",
            default_value="64",
            description="Maximum full keyframe observations resident in RAM",
        ),
        DeclareLaunchArgument(
            "map_observation_compression_level",
            default_value="1",
            description="Lossless zlib level 0..9 for the SQLite archive",
        ),
    ]


def mapping_parameter_overrides():
    return {
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
        "incremental_map_batch_topic": LaunchConfiguration(
            "incremental_map_batch_topic"
        ),
        "incremental_map_replay_service": LaunchConfiguration(
            "incremental_map_replay_service"
        ),
        "publish_unchanged_map_outputs": LaunchConfiguration(
            "publish_unchanged_map_outputs"
        ),
        "publish_layer_images": LaunchConfiguration(
            "publish_layer_images"
        ),
        "publish_map_pointcloud": LaunchConfiguration(
            "publish_map_pointcloud"
        ),
        "path_publish_stride": LaunchConfiguration(
            "path_publish_stride"
        ),
        "stereo_map_max_depth_m": LaunchConfiguration(
            "stereo_map_max_depth_m"
        ),
        "keyframe_interval": LaunchConfiguration("keyframe_interval"),
        "enable_lidar_mapping": LaunchConfiguration(
            "enable_lidar_mapping"
        ),
        "lidar_mapping_mode": LaunchConfiguration("lidar_mapping_mode"),
        "lidar_topic": LaunchConfiguration("lidar_topic"),
        "lidar_extrinsics_path": ParameterValue(
            LaunchConfiguration("lidar_extrinsics_path"), value_type=str
        ),
        "lidar_expected_frame_id": LaunchConfiguration(
            "lidar_expected_frame_id"
        ),
        "lidar_max_time_offset_sec": LaunchConfiguration(
            "lidar_max_time_offset_sec"
        ),
        "lidar_wait_timeout_sec": LaunchConfiguration(
            "lidar_wait_timeout_sec"
        ),
        "lidar_temporal_mode": LaunchConfiguration(
            "lidar_temporal_mode"
        ),
        "lidar_point_time_field": LaunchConfiguration(
            "lidar_point_time_field"
        ),
        "lidar_point_time_unit": LaunchConfiguration(
            "lidar_point_time_unit"
        ),
        "lidar_point_time_max_abs_sec": LaunchConfiguration(
            "lidar_point_time_max_abs_sec"
        ),
        "lidar_deskew_time_bin_sec": LaunchConfiguration(
            "lidar_deskew_time_bin_sec"
        ),
        "lidar_min_range_m": LaunchConfiguration("lidar_min_range_m"),
        "lidar_max_range_m": LaunchConfiguration("lidar_max_range_m"),
        "lidar_voxel_size_m": LaunchConfiguration("lidar_voxel_size_m"),
        "lidar_max_points_per_keyframe": LaunchConfiguration(
            "lidar_max_points_per_keyframe"
        ),
        "lidar_min_points_per_keyframe": LaunchConfiguration(
            "lidar_min_points_per_keyframe"
        ),
        "lidar_buffer_size": LaunchConfiguration("lidar_buffer_size"),
        "lidar_ground_max_slope_deg": LaunchConfiguration(
            "lidar_ground_max_slope_deg"
        ),
        "lidar_ground_neighbor_count": LaunchConfiguration(
            "lidar_ground_neighbor_count"
        ),
        "lidar_ground_max_neighbor_distance_m": LaunchConfiguration(
            "lidar_ground_max_neighbor_distance_m"
        ),
        "lidar_fallback_to_stereo": LaunchConfiguration(
            "lidar_fallback_to_stereo"
        ),
        "lidar_align_map_to_sensor_frame": LaunchConfiguration(
            "lidar_align_map_to_sensor_frame"
        ),
        "pointcloud_assist_pixel_radius": LaunchConfiguration(
            "pointcloud_assist_pixel_radius"
        ),
        "pointcloud_assist_unmatched_stereo_max_depth_m": (
            LaunchConfiguration(
                "pointcloud_assist_unmatched_stereo_max_depth_m"
            )
        ),
        "pointcloud_assist_replace_stereo_min_depth_m": LaunchConfiguration(
            "pointcloud_assist_replace_stereo_min_depth_m"
        ),
        "pointcloud_assist_min_matches": LaunchConfiguration(
            "pointcloud_assist_min_matches"
        ),
        "pointcloud_assist_max_points_per_keyframe": LaunchConfiguration(
            "pointcloud_assist_max_points_per_keyframe"
        ),
        "map_admission_enabled": LaunchConfiguration(
            "map_admission_enabled"
        ),
        "map_admission_min_recovery_inliers": LaunchConfiguration(
            "map_admission_min_recovery_inliers"
        ),
        "map_admission_min_recovery_inlier_ratio": LaunchConfiguration(
            "map_admission_min_recovery_inlier_ratio"
        ),
        "map_admission_stable_frames": LaunchConfiguration(
            "map_admission_stable_frames"
        ),
        "map_admission_max_translation_step_m": LaunchConfiguration(
            "map_admission_max_translation_step_m"
        ),
        "map_admission_max_rotation_step_deg": LaunchConfiguration(
            "map_admission_max_rotation_step_deg"
        ),
        "map_observation_archive_enabled": LaunchConfiguration(
            "map_observation_archive_enabled"
        ),
        "map_observation_cache_size": LaunchConfiguration(
            "map_observation_cache_size"
        ),
        "map_observation_compression_level": LaunchConfiguration(
            "map_observation_compression_level"
        ),
    }
