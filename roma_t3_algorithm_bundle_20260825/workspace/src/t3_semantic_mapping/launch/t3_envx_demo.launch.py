"""Native Orin/Env_X demo: stereo VO, elevation and registered LiDAR XYZ."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
import numpy as np
import yaml

from t3_semantic_mapping.roma_runtime_defaults import ROMA_RUNTIME_DEFAULTS


def _nodes(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    def boolean(name):
        result = value(name).lower()
        if result not in {"true", "false"}:
            raise ValueError(f"{name} must be true or false")
        return result == "true"

    bundle = Path(value("bundle_root"))
    calibration_path = Path(value("calib_path") or str(
        bundle / "config/calibration_spreadsheets_parallel_24p9.yaml"))
    cloud_config = Path(value("pointcloud_config") or str(
        bundle / "config/pointcloud_extrinsic_os1_mapping_only.yaml"))
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    # The supplied profile has parallel optical axes; use its measured
    # mounting translation and rotation to publish the body pose in ROS FLU.
    right_to_left = np.asarray(calibration["raw_stereo"]["transform_right_to_left"])
    if not np.allclose(right_to_left[:3, :3], np.eye(3), atol=1e-7):
        raise ValueError("This demo profile requires parallel stereo cameras")
    body_from_camera = np.diag([1., -1., -1., 1.]) @ np.asarray(
        calibration["mounting"]["transform_vehicle_left"], dtype=float)
    camera_from_body = np.linalg.inv(body_from_camera)
    share = Path(get_package_share_directory("t3_semantic_mapping"))
    # Merge first, then pass ONE parameter dictionary. This avoids YAML vs
    # launch override precedence differences across installed ROS versions.
    params = yaml.safe_load((share / "config/semantic_map.yaml").read_text())[
        "roma_semantic_mapper_node"]["ros__parameters"]
    params.update(dict(ROMA_RUNTIME_DEFAULTS))
    params.update({
        "algorithm_root": str(bundle / "third_party/RoMa"),
        "calib_path": str(calibration_path),
        "output_dir": value("output_dir"),
        "left_topic": value("left_topic"),
        "right_topic": value("right_topic"),
        "semantic_topic": value("semantic_topic"),
        "enable_semantics": boolean("enable_semantics"),
        "enable_imu": False,
        "use_sim_time": boolean("use_sim_time"),
        "image_timestamp_source": "sensor",
        "image_sync_mode": "approximate",
        "stereo_input_reliability": value("image_reliability"),
        "semantic_input_reliability": "best_effort",
        "image_input_qos_depth": 2,
        "image_sync_queue_size": 10,
        "image_sync_slop_sec": 0.07,
        "stereo_sync_max_skew_sec": 0.07,
        "map_frame": "map",
        "odom_frame": "odom",
        "local_grid_map_frame": "odom",
        "child_frame": "base_link",
        "odometry_camera_frame": "car_stereo_left_optical_frame",
        "odometry_child_transform_confirmed": True,
        "odometry_child_to_camera_transform": camera_from_body.reshape(-1).tolist(),
        "camera_pitch_deg": float(calibration["mounting"]["downward_pitch_deg"]),
        "localization_mode": "vo",
        "fusion_backend": "none",
        "enable_backend": False,
        "enable_loop_closing": False,
        "enable_global_loop_closure": False,
        "enable_lidar_mapping": False,
        "enable_pointcloud_fusion": True,
        "elevation_fusion_mode": value("elevation_fusion_mode"),
        "pointcloud_fusion_config": str(cloud_config),
        "publish_registered_lidar": True,
        "registered_lidar_topic": "/T3/mapping/lidar_keyframe",
        "odometry_topic": "/T3/semantic/current_pose",
        "publish_odometry_compat": True,
        "odometry_compat_topic": "/Car/T3/localization/odometry",
        "trajectory_topic": "/T3/semantic/trajectory",
        "path_publish_stride": 1,
        "grid_map_topic": "/T3/mapping/grid_map",
        "global_grid_map_topic": "/T3/mapping/global_grid_map",
        "global_overview_topic": "/T3/mapping/global_overview",
        "global_map_revision_topic": "/T3/mapping/global_map_revision",
        "grid_map_query_service": "/T3/mapping/get_grid_map",
        "publish_incremental_map": True,
        "incremental_map_topic": "/T3/semantic/incremental_map",
        "incremental_publish_mode": "rolling_window",
        "dense_map_window_mode": "rolling",
        "map_length_x": 64.0,
        "map_length_y": 64.0,
        "publish_global_grid_map": True,
        "save_dense_global_grid_map": True,
        "publish_layer_images": True,
        "publish_map_pointcloud": False,
        "publish_feature_matches": False,
        "publish_realtime_metrics": True,
        "publish_input_processed": boolean("publish_input_processed"),
        "map_persistence_workers": 2,
        "global_dense_min_period_sec": 10.0,
        "map_observation_cache_size": 32,
        "visual_keyframe_image_cache": 8,
    })
    return [
        Node(package="t3_semantic_mapping", executable="visual_lidar_cloud_mapper",
             output="screen", parameters=[{
                 "output_dir": value("output_dir"),
                 "use_sim_time": boolean("use_sim_time"),
                 "publish_period": float(value("cloud_publish_period")),
                 "preview_max_points": 200000,
             }]),
        Node(package="t3_semantic_mapping", executable="roma_semantic_mapper",
             name="roma_semantic_mapper_node", output="screen", parameters=[params]),
    ]


def generate_launch_description():
    arguments = {
        "bundle_root": EnvironmentVariable(
            "T3_BUNDLE_ROOT", default_value=str(Path.home() / "P3/roma_t3_algorithm_bundle_20260825")),
        "output_dir": EnvironmentVariable("T3_OUTPUT_DIR", default_value=""),
        "calib_path": "",
        "pointcloud_config": "",
        "elevation_fusion_mode": EnvironmentVariable(
            "T3_ELEVATION_FUSION", default_value="variance"),
        "left_topic": "/Car/T5/Cam_Left/image_raw/color",
        "right_topic": "/Car/T5/Cam_Right/image_raw/color",
        "semantic_topic": "/task2/seg_mask",
        "enable_semantics": "false",
        "image_reliability": "reliable",
        "cloud_publish_period": "2.0",
        "publish_input_processed": "false",
        "use_sim_time": "false",
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=default) for name, default in arguments.items()),
        OpaqueFunction(function=_nodes),
    ])
