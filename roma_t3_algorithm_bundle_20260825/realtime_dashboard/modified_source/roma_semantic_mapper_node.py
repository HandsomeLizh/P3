from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.srv import GetGridMap
from message_filters import Subscriber
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import Header, String, UInt64
from std_srvs.srv import Trigger
from t1_dataset_sim.dataset_formats import StereoRectifier, load_calibration
from t3_interfaces.msg import IncrementalSemanticMap, SemanticMapBatch
from tf2_ros import TransformBroadcaster

from .coordinate_utils import (
    matrix_to_quaternion_xyzw,
    opencv_world_to_map_rotation,
    points_opencv_to_map,
    pose_opencv_to_map,
)
from .compact_grid_store import LiveCompactGlobalMapWriter
from .dense_grid_store import LiveDenseGlobalMapWriter
from .feature_match_debug import render_feature_match_debug
from .grid_map_message import make_grid_map_message
from .input_processing_monitor import (
    CallbackStageTimer,
    InputProcessingMonitor,
)
from .input_transport import InputTransportConfiguration, TimestampSourceFilter
from .lidar_message_buffer import LidarMessageBuffer
from .lidar_mapping import (
    LidarMappingProcessor,
    point_time_unit_scale,
    pointcloud2_xyz_array,
    pointcloud2_xyz_time_arrays,
    stamp_nanoseconds,
)
from .lidar_temporal_alignment import (
    LidarTemporalConfiguration,
    align_lidar_points_to_reference,
    lidar_required_pose_time_range,
)
from .localization import (
    ExternalOdometryAdapter,
    FrameMismatchError,
    LocalizationConfiguration,
    LocalizationCoordinator,
    LocalizationUnavailableError,
)
from .localization.calibration import (
    compose_lio_child_to_camera_transform,
    load_rectified_lidar_to_camera_transform,
    load_rectified_mounting_to_camera_transform,
    row_major_transform,
)
from .localization.global_loop_closure import GlobalLoopClosure
from .localization.lidar_loop_closure import LidarLoopConfiguration
from .localization.mprf_client import (
    MprfAsyncClient,
    MprfClientConfiguration,
)
from .localization.pose_math import inverse_pose, pose_change_norm
from .map_admission import MapAdmissionConfig, MapAdmissionGate
from .map_contribution_index import MapContributionIndex
from .map_correction_worker import (
    MapCorrectionCompletion,
    MapCorrectionWorker,
)
from .roma_runtime_defaults import ROMA_RUNTIME_DEFAULTS
from .map_output_message import (
    make_incremental_semantic_map_message,
    make_occupancy_grid_from_array,
    make_occupancy_grid_message,
)
from .map_transfer_message import (
    make_map_batch_begin,
    make_map_batch_complete,
)
from .map_transfer_protocol import (
    MapTransferBatchPlan,
    MapTransferSession,
    TileTransferPlan,
)
from .odometry_output import OdometryOutputModel
from .pointcloud_utils import (
    finite_point_sample_indices,
    make_semantic_point_cloud,
)
from .pose_output_message import (
    make_odometry_message,
    make_transform_stamped_from_matrix,
    make_transform_stamped_message,
)
from .ros_imu_buffer import RosImuBuffer
from .semantic_map_rebuilder import (
    KeyFrameObservationStore,
    SemanticKeyFrameObservation,
)
from .semantic_grid import (
    CLASS_NAMES,
)
from .stereo_observation_builder import (
    StereoObservationBuilder,
    StereoObservationConfiguration,
)
from .terrain_pointcloud_mapper import TerrainPointCloudMapper
from .tiled_semantic_map import TiledSemanticMapManager
from .visual_tracking_adapter import (
    process_visual_tracking,
    visual_frontend_required,
)


# ROS 2 parameters are serialized through rcl_interfaces/ParameterValue. Some
# supported distributions reject NaN during declaration, so use a finite
# out-of-band value for "centre the map".
_AUTO_MAP_ORIGIN = -1.0e300
_GRID_MAP_LAYER_NAMES = (
    "elevation",
    "elevation_variance",
    "occupancy",
    "semantic_id",
    "semantic_confidence",
    "observation_count",
    "height_range",
    "roughness",
    "color",
)


class LidarObservationBuildError(RuntimeError):
    """Retriable map-observation failure retaining its selected cloud.

    Point-wise deskew may need a pose slightly newer than the image callback.
    The selected PointCloud2 must survive that wait: selecting again on retry
    can silently switch geometry or fail forever after a high-rate DDS stream
    evicts the original cloud from its small live buffer.
    """

    def __init__(self, message: str, *, lidar_message: PointCloud2) -> None:
        super().__init__(str(message))
        self.lidar_message = lidar_message


@dataclass
class _TileBatchReplayState:
    """One immutable, streamed full/delta tile batch."""

    snapshot: TiledSemanticMapManager
    map_header: Header
    batch: MapTransferBatchPlan
    pending_keys: deque
    payload_update_ids: list[int]
    published_revisions: dict[tuple[int, int], int]


@dataclass(frozen=True)
class _MapCorrectionRequest:
    """Immutable inputs captured for one background map correction."""

    geometry_revision: int
    base_map_revision: int
    live_grid: TiledSemanticMapManager
    contribution_index: MapContributionIndex
    pending_observations: tuple[SemanticKeyFrameObservation, ...]
    keyframe_poses_cv: dict[int, np.ndarray]


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _stamp_text(stamp) -> str:
    return f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"


def _stamp_nanoseconds_value(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class RomaSemanticMapperNode(Node):
    """Task 3 localisation and formal 2.5D semantic GridMap node."""

    def __init__(self) -> None:
        super().__init__("roma_semantic_mapper_node")

        home = Path.home()
        default_data = home / "RoMa" / "data" / "data1223"
        default_output = home / "P3" / "output"

        # Algorithm and calibration.
        self.declare_parameter("algorithm_root", str(home / "RoMa"))
        self.declare_parameter("algorithm_module", "roma_vo")
        self.declare_parameter("algorithm_class", "StereoSLAMv2")
        self.declare_parameter("calib_path", str(default_data / "calib.yaml"))
        self.declare_parameter("output_dir", str(default_output))

        # Inputs.
        self.declare_parameter(
            "left_topic", "/Car/T1/fusion/stereo/left/image_raw"
        )
        self.declare_parameter(
            "right_topic", "/Car/T1/fusion/stereo/right/image_raw"
        )
        self.declare_parameter(
            "semantic_topic", "/Car/T2/semantic/label_image"
        )
        self.declare_parameter("enable_semantics", True)
        self.declare_parameter("image_timestamp_source", "sensor")
        self.declare_parameter("imu_topic", "/Car/T1/fusion/imu/data")
        self.declare_parameter(
            "input_processed_topic", "/Car/T3/debug/input_processed"
        )
        self.declare_parameter("publish_input_processed", False)
        self.declare_parameter(
            "realtime_metrics_topic", "/Car/T3/metrics/frame_timing"
        )
        self.declare_parameter("publish_realtime_metrics", False)
        # Keep the established standard-message boundary as the default, but
        # make DDS compatibility and timestamp matching explicit for live
        # integration.  Formal T1/T2 custom messages belong in thin adapters
        # once their real .msg definitions are supplied.
        self.declare_parameter("stereo_input_reliability", "reliable")
        self.declare_parameter("semantic_input_reliability", "reliable")
        self.declare_parameter("imu_input_reliability", "reliable")
        self.declare_parameter("image_input_qos_depth", 10)
        self.declare_parameter("imu_input_qos_depth", 10)
        self.declare_parameter("image_sync_mode", "exact")
        self.declare_parameter("image_sync_queue_size", 50)
        self.declare_parameter("image_sync_slop_sec", 0.02)
        self.declare_parameter("stereo_sync_max_skew_sec", 0.0)
        self.declare_parameter(
            "lidar_topic", "/Car/T1/fusion/lidar/points"
        )
        self.declare_parameter("lio_odometry_topic", "/Odometry")

        # Outputs.
        self.declare_parameter(
            "odometry_topic", "/Car/T3/localization/odometry"
        )
        self.declare_parameter(
            "odometry_compat_topic", "/Car/T3/semantic/current_pose"
        )
        self.declare_parameter("publish_odometry_compat", False)
        self.declare_parameter(
            "trajectory_topic", "/Car/T3/debug/trajectory"
        )
        self.declare_parameter(
            "feature_matches_topic",
            "/Car/T2/feature_matching/visualization",
        )
        self.declare_parameter("feature_visualization_crop_width", 1280)
        self.declare_parameter("feature_visualization_crop_height", 720)
        self.declare_parameter("feature_visualization_width", 640)
        self.declare_parameter("feature_visualization_height", 480)
        self.declare_parameter(
            "pointcloud_topic", "/Car/T3/mapping/map_points"
        )
        self.declare_parameter(
            "occupancy_topic", "/Car/T3/mapping/occupancy_grid"
        )
        self.declare_parameter(
            "semantic_grid_topic", "/Car/T3/mapping/semantic_grid"
        )
        self.declare_parameter(
            "semantic_color_topic",
            "/Car/T3/mapping/semantic_grid_color",
        )
        self.declare_parameter(
            "elevation_topic", "/Car/T3/mapping/elevation_grid"
        )
        self.declare_parameter(
            "elevation_variance_topic",
            "/Car/T3/mapping/elevation_variance_grid",
        )
        self.declare_parameter(
            "elevation_color_topic",
            "/Car/T3/mapping/elevation_grid_color",
        )
        self.declare_parameter(
            "grid_map_topic", "/Car/T3/mapping/grid_map"
        )
        self.declare_parameter(
            "grid_map_odom_topic", "/Car/T3/mapping/grid_map_odom"
        )
        self.declare_parameter("publish_grid_map_odom_alias", False)
        self.declare_parameter(
            "global_grid_map_topic", "/Car/T3/mapping/global_grid_map"
        )
        self.declare_parameter(
            "global_overview_topic", "/Car/T3/mapping/global_overview"
        )
        self.declare_parameter(
            "global_map_revision_topic",
            "/Car/T3/mapping/global_map_revision",
        )
        self.declare_parameter(
            "grid_map_query_service", "/Car/T3/mapping/get_grid_map"
        )
        self.declare_parameter(
            "incremental_map_topic",
            "/Car/T3/semantic/incremental_map",
        )
        self.declare_parameter(
            "incremental_map_batch_topic",
            "/Car/T3/semantic/map_batch",
        )
        self.declare_parameter(
            "incremental_map_replay_service",
            "/Car/T3/mapping/replay_full_map",
        )
        self.declare_parameter(
            "incremental_publish_mode",
            "fixed",
        )
        self.declare_parameter("incremental_publish_period_sec", 1.0)
        self.declare_parameter(
            "dense_map_window_mode", "rolling"
        )
        # Publication controls never change the formal topic schema.  The
        # default suppresses only equivalent map payloads (the ROS header may
        # still differ between frames);
        # optional debug products retain their established defaults.
        self.declare_parameter("publish_unchanged_map_outputs", False)
        self.declare_parameter("publish_layer_images", False)
        self.declare_parameter("publish_map_pointcloud", False)
        self.declare_parameter("publish_feature_matches", True)
        self.declare_parameter("feature_match_output_width", 1280)
        self.declare_parameter("feature_match_max_lines", 200)
        self.declare_parameter("publish_local_occupancy_grid", False)
        self.declare_parameter("publish_incremental_map", False)
        self.declare_parameter(
            "local_grid_map_layers",
            ["occupancy", "semantic_id", "elevation", "roughness"],
        )
        self.declare_parameter("publish_global_overview", True)
        self.declare_parameter("publish_global_grid_map", True)
        self.declare_parameter(
            "global_grid_map_layers",
            ["elevation"],
        )
        self.declare_parameter("global_overview_resolution", 1.0)
        self.declare_parameter("global_overview_period_sec", 20.0)
        self.declare_parameter("grid_map_query_max_cells", 1000000)
        self.declare_parameter(
            "compact_global_map_filename", "global_grid_map.sqlite3"
        )
        self.declare_parameter("compact_global_map_compression_level", 6)
        self.declare_parameter(
            "map_persistence_workers",
            min(4, max(1, os.cpu_count() or 1)),
        )
        self.declare_parameter("save_dense_global_grid_map", True)
        self.declare_parameter(
            "dense_global_grid_map_filename", "global_grid_map.npz"
        )
        self.declare_parameter("path_publish_stride", 1)

        # Coordinate frames and extrinsic offsets.
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("local_grid_map_frame", "odom")
        self.declare_parameter(
            "child_frame", "car_stereo_left_optical_frame"
        )
        # The estimator state is the left-camera optical centre.  Publishing
        # another physical child frame requires a calibrated T_camera_child;
        # changing only child_frame would silently mislabel the pose.
        self.declare_parameter(
            "odometry_camera_frame", "car_stereo_left_optical_frame"
        )
        self.declare_parameter(
            "odometry_child_transform_confirmed", False
        )
        self.declare_parameter(
            "odometry_child_to_camera_transform",
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        self.declare_parameter(
            "odometry_velocity_max_interval_sec", 1.0
        )
        self.declare_parameter(
            "odometry_unknown_covariance_variance", 1.0e6
        )
        self.declare_parameter("camera_pitch_deg", 30.0)
        self.declare_parameter("map_x_offset_m", 0.0)
        self.declare_parameter("map_y_offset_m", 0.0)
        self.declare_parameter("map_z_offset_m", 0.0)
        self.declare_parameter("publish_tf", True)

        # Grid geometry.
        self.declare_parameter("map_resolution", 0.20)
        self.declare_parameter("map_length_x", 64.0)
        self.declare_parameter("map_length_y", 64.0)
        self.declare_parameter("map_tile_cells", 256)
        self.declare_parameter("map_id", "t3_semantic_map")
        self.declare_parameter("free_class_ids", [1])
        self.declare_parameter("occupied_class_ids", [3, 4])
        self.declare_parameter("map_origin_x", _AUTO_MAP_ORIGIN)
        self.declare_parameter("map_origin_y", _AUTO_MAP_ORIGIN)
        self.declare_parameter("max_rays_per_update", 1200)
        # Complete map provenance is losslessly page-backed by default. Only
        # the LRU cache is bounded; every keyframe observation remains
        # available for BA/loop/factor-graph map reconstruction.
        self.declare_parameter("map_observation_archive_enabled", True)
        self.declare_parameter("map_observation_cache_size", 64)
        self.declare_parameter("map_observation_compression_level", 1)
        # Pure-VO map admission after tracking loss.  This does not suppress
        # Odometry and does not alter RoMa; it only protects persistent map
        # evidence from a weak or discontinuous recovery pose.
        self.declare_parameter("map_admission_enabled", True)
        self.declare_parameter("map_admission_min_recovery_inliers", 25)
        self.declare_parameter("map_admission_min_recovery_inlier_ratio", 0.25)
        self.declare_parameter("map_admission_stable_frames", 3)
        self.declare_parameter(
            "map_admission_max_translation_step_m", 1.0
        )
        self.declare_parameter(
            "map_admission_max_rotation_step_deg", 20.0
        )

        # RoMa and SLAM settings.
        self.declare_parameter("device", "cuda")
        self.declare_parameter(
            "coarse_res", ROMA_RUNTIME_DEFAULTS["coarse_res"]
        )
        self.declare_parameter(
            "upsample_res", ROMA_RUNTIME_DEFAULTS["upsample_res"]
        )
        self.declare_parameter(
            "use_custom_corr", ROMA_RUNTIME_DEFAULTS["use_custom_corr"]
        )
        self.declare_parameter(
            "stereo_upsample_refinement",
            ROMA_RUNTIME_DEFAULTS["stereo_upsample_refinement"],
        )
        self.declare_parameter(
            "temporal_upsample_refinement",
            ROMA_RUNTIME_DEFAULTS["temporal_upsample_refinement"],
        )
        self.declare_parameter(
            "symmetric_matching",
            ROMA_RUNTIME_DEFAULTS["symmetric_matching"],
        )
        self.declare_parameter(
            "enable_feature_reuse",
            ROMA_RUNTIME_DEFAULTS["enable_feature_reuse"],
        )
        self.declare_parameter(
            "batch_coarse_pairs",
            ROMA_RUNTIME_DEFAULTS["batch_coarse_pairs"],
        )
        self.declare_parameter(
            "enable_klt_temporal_tracking",
            ROMA_RUNTIME_DEFAULTS["enable_klt_temporal_tracking"],
        )
        self.declare_parameter("klt_max_points", 800)
        self.declare_parameter("klt_min_tracks", 60)
        self.declare_parameter("klt_min_track_ratio", 0.25)
        self.declare_parameter("klt_fb_max_error", 1.5)
        self.declare_parameter("klt_image_scale", 0.25)
        self.declare_parameter("klt_min_pnp_inlier_ratio", 0.40)
        self.declare_parameter(
            "stereo_matches", ROMA_RUNTIME_DEFAULTS["stereo_matches"]
        )
        self.declare_parameter(
            "temporal_matches", ROMA_RUNTIME_DEFAULTS["temporal_matches"]
        )
        self.declare_parameter("keyframe_interval", 5)
        self.declare_parameter("enable_backend", False)
        self.declare_parameter("enable_loop_closing", False)
        self.declare_parameter("enable_imu", True)
        self.declare_parameter("pointcloud_max_points", 50000)
        self.declare_parameter("visible_max_points", 30000)
        self.declare_parameter("stereo_map_max_depth_m", 5.0)
        self.declare_parameter("path_max_length", 5000)
        self.declare_parameter("save_on_shutdown", True)

        # Localisation source selection.  The established VO path remains the
        # default and does not import GTSAM or require an external process.
        self.declare_parameter("localization_mode", "vo")
        self.declare_parameter("fusion_backend", "none")
        self.declare_parameter("enable_gp_prior", True)
        self.declare_parameter("fusion_allow_vo_only", True)

        # External FAST-LIO2 Odometry adapter.  LiDAR-to-camera and the LIO
        # child-to-LiDAR transform are explicit because stock FAST-LIO2's
        # ``body`` child frame is the IMU frame, not necessarily the LiDAR.
        self.declare_parameter("lio_expected_parent_frame", "camera_init")
        self.declare_parameter("lio_expected_child_frame", "body")
        self.declare_parameter("lio_extrinsics_path", "")
        self.declare_parameter("lio_child_to_lidar_confirmed", False)
        self.declare_parameter(
            "lio_child_to_lidar_transform",
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        self.declare_parameter("lio_max_time_offset_sec", 0.15)
        self.declare_parameter("lio_wait_timeout_sec", 0.20)
        self.declare_parameter("lio_buffer_size", 2000)
        self.declare_parameter("lio_buffer_age_sec", 300.0)

        # Local fixed-lag GTSAM backend.  Rotation entries precede translation
        # entries, matching the Pose3 tangent convention.
        self.declare_parameter("fusion_lag_sec", 10.0)
        self.declare_parameter("fusion_solver", "incremental")
        self.declare_parameter(
            "fusion_gp_qc_rotation", [1.0, 1.0, 2.0]
        )
        self.declare_parameter(
            "fusion_gp_qc_translation", [2.0, 2.0, 2.0]
        )
        self.declare_parameter(
            "fusion_vo_rotation_sigma_deg", [1.5, 1.5, 2.5]
        )
        self.declare_parameter(
            "fusion_vo_translation_sigma_m", [0.08, 0.08, 0.12]
        )
        self.declare_parameter(
            "fusion_lio_rotation_sigma_deg", [0.5, 0.5, 0.8]
        )
        self.declare_parameter(
            "fusion_lio_translation_sigma_m", [0.03, 0.03, 0.05]
        )
        self.declare_parameter("fusion_robust_huber_k", 1.5)
        self.declare_parameter(
            "fusion_map_rebuild_translation_m", 0.02
        )
        self.declare_parameter(
            "fusion_map_rebuild_rotation_deg", 0.5
        )

        # Optional global loop layer.  It is deliberately independent from
        # the local fixed-lag estimator and off by default.  MPRF proposes a
        # coarse candidate; only the in-process LiDAR verifier can turn it
        # into a robust iSAM2 factor.
        self.declare_parameter("enable_global_loop_closure", False)
        self.declare_parameter("global_loop_candidate_source", "mprf")
        self.declare_parameter("global_loop_submap_keyframes", 3)
        self.declare_parameter("global_loop_voxel_size_m", 0.25)
        self.declare_parameter("global_loop_max_points", 5000)
        self.declare_parameter("global_loop_geometry_cache_records", 16)
        self.declare_parameter("global_loop_min_keyframe_separation", 20)
        self.declare_parameter("global_loop_min_time_separation_sec", 20.0)
        self.declare_parameter("global_loop_scan_context_min_similarity", 0.35)
        self.declare_parameter("global_loop_icp_max_correspondence_m", 1.0)
        self.declare_parameter("global_loop_icp_min_inliers", 120)
        self.declare_parameter("global_loop_icp_min_overlap", 0.25)
        self.declare_parameter(
            "global_loop_icp_min_unique_target_ratio", 0.50
        )
        self.declare_parameter("global_loop_icp_max_rmse_m", 0.30)
        self.declare_parameter("global_loop_confirmation_count", 2)
        self.declare_parameter("global_loop_max_correction_m", 4.0)
        self.declare_parameter("global_loop_max_correction_deg", 25.0)
        self.declare_parameter("global_graph_robust_huber_k", 1.5)
        self.declare_parameter(
            "global_graph_odometry_rotation_sigma_deg",
            [0.8, 0.8, 1.2],
        )
        self.declare_parameter(
            "global_graph_odometry_translation_sigma_m",
            [0.05, 0.05, 0.08],
        )
        self.declare_parameter(
            "mprf_endpoint", "http://127.0.0.1:8765"
        )
        self.declare_parameter("mprf_request_timeout_sec", 20.0)
        self.declare_parameter("mprf_queue_size", 8)
        self.declare_parameter("mprf_transient_retries", 2)
        self.declare_parameter("mprf_poll_period_sec", 0.10)
        self.declare_parameter(
            "mprf_auth_token_env", "MPRF_SERVICE_TOKEN"
        )

        # Optional LiDAR geometry for 2.5D mapping.  This is deliberately
        # independent of the localisation selector: VO can map LiDAR points,
        # and LIO can still map stereo points.  The option remains off by
        # default so the established ROS behaviour is unchanged.
        self.declare_parameter("enable_lidar_mapping", False)
        self.declare_parameter("lidar_mapping_mode", "direct")
        self.declare_parameter("lidar_extrinsics_path", "")
        self.declare_parameter("lidar_expected_frame_id", "velodyne")
        self.declare_parameter("lidar_max_time_offset_sec", 0.15)
        self.declare_parameter("lidar_wait_timeout_sec", 1.0)
        self.declare_parameter("lidar_temporal_mode", "none")
        self.declare_parameter("lidar_point_time_field", "time")
        self.declare_parameter("lidar_point_time_unit", "sec")
        self.declare_parameter("lidar_point_time_max_abs_sec", 0.20)
        self.declare_parameter("lidar_deskew_time_bin_sec", 0.001)
        self.declare_parameter("lidar_min_range_m", 0.5)
        self.declare_parameter("lidar_max_range_m", 30.0)
        self.declare_parameter("lidar_voxel_size_m", 0.10)
        self.declare_parameter("lidar_max_points_per_keyframe", 5000)
        self.declare_parameter("lidar_min_points_per_keyframe", 100)
        self.declare_parameter("lidar_buffer_size", 20)
        self.declare_parameter("lidar_ground_max_slope_deg", 35.0)
        self.declare_parameter("lidar_ground_neighbor_count", 12)
        self.declare_parameter(
            "lidar_ground_max_neighbor_distance_m", 0.75
        )
        self.declare_parameter("lidar_fallback_to_stereo", False)
        self.declare_parameter("lidar_align_map_to_sensor_frame", True)
        self.declare_parameter("pointcloud_assist_pixel_radius", 2.0)
        self.declare_parameter(
            "pointcloud_assist_unmatched_stereo_max_depth_m", 5.0
        )
        self.declare_parameter(
            "pointcloud_assist_replace_stereo_min_depth_m", 5.0
        )
        self.declare_parameter("pointcloud_assist_min_matches", 20)
        self.declare_parameter(
            "pointcloud_assist_max_points_per_keyframe", 20000
        )
        self.declare_parameter("enable_pointcloud_fusion", False)
        self.declare_parameter(
            "pointcloud_fusion_config", ""
        )

        self.input_transport = InputTransportConfiguration(
            stereo_reliability=str(
                self.get_parameter("stereo_input_reliability").value
            ),
            semantic_reliability=str(
                self.get_parameter("semantic_input_reliability").value
            ),
            imu_reliability=str(
                self.get_parameter("imu_input_reliability").value
            ),
            image_qos_depth=int(
                self.get_parameter("image_input_qos_depth").value
            ),
            imu_qos_depth=int(
                self.get_parameter("imu_input_qos_depth").value
            ),
            sync_mode=str(self.get_parameter("image_sync_mode").value),
            sync_queue_size=int(
                self.get_parameter("image_sync_queue_size").value
            ),
            sync_slop_sec=float(
                self.get_parameter("image_sync_slop_sec").value
            ),
            stereo_max_skew_sec=float(
                self.get_parameter("stereo_sync_max_skew_sec").value
            ),
        )

        self.map_frame = str(
            self.get_parameter("map_frame").value
        ).strip().lstrip("/")
        self.odom_frame = str(
            self.get_parameter("odom_frame").value
        ).strip().lstrip("/")
        self.local_grid_map_frame = str(
            self.get_parameter("local_grid_map_frame").value
        ).strip().lstrip("/")
        if not self.map_frame or not self.odom_frame:
            raise ValueError("map_frame and odom_frame cannot be empty")
        if self.map_frame == self.odom_frame:
            raise ValueError("map_frame and odom_frame must be different")
        if self.local_grid_map_frame not in {
            self.map_frame,
            self.odom_frame,
        }:
            raise ValueError(
                "local_grid_map_frame must equal map_frame or odom_frame"
            )
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.odometry_output = OdometryOutputModel(
            camera_frame=str(
                self.get_parameter("odometry_camera_frame").value
            ),
            child_frame=self.child_frame,
            transform_camera_child=row_major_transform(
                self.get_parameter(
                    "odometry_child_to_camera_transform"
                ).value,
                name="Odometry child-to-camera transform",
            ),
            child_transform_confirmed=bool(
                self.get_parameter(
                    "odometry_child_transform_confirmed"
                ).value
            ),
            velocity_max_interval_sec=float(
                self.get_parameter(
                    "odometry_velocity_max_interval_sec"
                ).value
            ),
            unknown_covariance_variance=float(
                self.get_parameter(
                    "odometry_unknown_covariance_variance"
                ).value
            ),
        )
        # Use the validated, ROS-normalised frame name everywhere the output
        # child is referenced (Odometry, TF and Path semantics).
        self.child_frame = self.odometry_output.child_frame
        self.last_transform_map_odom = np.eye(4, dtype=np.float64)
        self.publish_tf_enabled = bool(
            self.get_parameter("publish_tf").value
        )
        self.enable_imu = bool(self.get_parameter("enable_imu").value)
        self.enable_semantics = bool(
            self.get_parameter("enable_semantics").value
        )
        self.image_timestamp_source = str(
            self.get_parameter("image_timestamp_source").value
        ).strip().lower()
        if self.image_timestamp_source not in {"sensor", "arrival"}:
            raise ValueError(
                "image_timestamp_source must be 'sensor' or 'arrival'"
            )
        self.publish_unchanged_map_outputs = bool(
            self.get_parameter("publish_unchanged_map_outputs").value
        )
        self.publish_layer_images = bool(
            self.get_parameter("publish_layer_images").value
        )
        self.publish_map_pointcloud = bool(
            self.get_parameter("publish_map_pointcloud").value
        )
        self.publish_feature_matches = bool(
            self.get_parameter("publish_feature_matches").value
        )
        self.feature_match_output_width = int(
            self.get_parameter("feature_match_output_width").value
        )
        self.feature_match_max_lines = int(
            self.get_parameter("feature_match_max_lines").value
        )
        if self.feature_match_output_width < 320:
            raise ValueError("feature_match_output_width must be at least 320")
        if self.feature_match_max_lines <= 0:
            raise ValueError("feature_match_max_lines must be positive")
        self.publish_local_occupancy_grid = bool(
            self.get_parameter("publish_local_occupancy_grid").value
        )
        self.publish_incremental_map = bool(
            self.get_parameter("publish_incremental_map").value
        )
        self.local_grid_map_layers = tuple(
            str(value)
            for value in self.get_parameter("local_grid_map_layers").value
        )
        if not self.local_grid_map_layers:
            raise ValueError("local_grid_map_layers must not be empty")
        unknown_local_layers = sorted(
            set(self.local_grid_map_layers) - set(_GRID_MAP_LAYER_NAMES)
        )
        if unknown_local_layers:
            raise ValueError(
                "Unknown local_grid_map_layers: "
                + ", ".join(unknown_local_layers)
            )
        if len(set(self.local_grid_map_layers)) != len(
            self.local_grid_map_layers
        ):
            raise ValueError("local_grid_map_layers must not contain duplicates")
        self.publish_global_overview = bool(
            self.get_parameter("publish_global_overview").value
        )
        self.publish_global_grid_map = bool(
            self.get_parameter("publish_global_grid_map").value
        )
        self.global_grid_map_layers = tuple(
            str(value)
            for value in self.get_parameter("global_grid_map_layers").value
        )
        if not self.global_grid_map_layers:
            raise ValueError("global_grid_map_layers must not be empty")
        unknown_global_layers = sorted(
            set(self.global_grid_map_layers) - set(_GRID_MAP_LAYER_NAMES)
        )
        if unknown_global_layers:
            raise ValueError(
                "Unknown global_grid_map_layers: "
                + ", ".join(unknown_global_layers)
            )
        if len(set(self.global_grid_map_layers)) != len(
            self.global_grid_map_layers
        ):
            raise ValueError(
                "global_grid_map_layers must not contain duplicates"
            )
        self.global_overview_resolution = float(
            self.get_parameter("global_overview_resolution").value
        )
        self.global_overview_period_sec = float(
            self.get_parameter("global_overview_period_sec").value
        )
        if (
            not np.isfinite(self.global_overview_resolution)
            or self.global_overview_resolution <= 0.0
        ):
            raise ValueError("global_overview_resolution must be positive")
        if (
            not np.isfinite(self.global_overview_period_sec)
            or self.global_overview_period_sec <= 0.0
        ):
            raise ValueError("global_overview_period_sec must be positive")
        self.grid_map_query_max_cells = int(
            self.get_parameter("grid_map_query_max_cells").value
        )
        if self.grid_map_query_max_cells <= 0:
            raise ValueError("grid_map_query_max_cells must be positive")
        self.compact_global_map_filename = str(
            self.get_parameter("compact_global_map_filename").value
        ).strip()
        compact_name = Path(self.compact_global_map_filename)
        if (
            not self.compact_global_map_filename
            or compact_name.name != self.compact_global_map_filename
        ):
            raise ValueError(
                "compact_global_map_filename must be one plain file name"
            )
        self.compact_global_map_compression_level = int(
            self.get_parameter(
                "compact_global_map_compression_level"
            ).value
        )
        if not 0 <= self.compact_global_map_compression_level <= 9:
            raise ValueError(
                "compact_global_map_compression_level must be between 0 and 9"
            )
        requested_map_workers = int(
            self.get_parameter("map_persistence_workers").value
        )
        if requested_map_workers <= 0:
            raise ValueError("map_persistence_workers must be positive")
        self.map_persistence_workers = min(
            requested_map_workers,
            max(1, os.cpu_count() or 1),
        )
        self.save_dense_global_grid_map = bool(
            self.get_parameter("save_dense_global_grid_map").value
        )
        self.dense_global_grid_map_filename = str(
            self.get_parameter("dense_global_grid_map_filename").value
        ).strip()
        dense_name = Path(self.dense_global_grid_map_filename)
        if (
            not self.dense_global_grid_map_filename
            or dense_name.name != self.dense_global_grid_map_filename
            or dense_name.suffix.lower() != ".npz"
        ):
            raise ValueError(
                "dense_global_grid_map_filename must be one plain .npz file name"
            )
        self.path_publish_stride = int(
            self.get_parameter("path_publish_stride").value
        )
        if self.path_publish_stride <= 0:
            raise ValueError("path_publish_stride must be positive")
        self.path_max_length = int(
            self.get_parameter("path_max_length").value
        )
        self.pointcloud_max_points = int(
            self.get_parameter("pointcloud_max_points").value
        )
        visible_max_points = int(
            self.get_parameter("visible_max_points").value
        )
        stereo_map_max_depth_m = float(
            self.get_parameter("stereo_map_max_depth_m").value
        )
        self.save_on_shutdown = bool(
            self.get_parameter("save_on_shutdown").value
        )
        self.localization_configuration = LocalizationConfiguration(
            mode=str(self.get_parameter("localization_mode").value),
            fusion_backend=str(self.get_parameter("fusion_backend").value),
            enable_gp_prior=bool(
                self.get_parameter("enable_gp_prior").value
            ),
            allow_vo_only_fusion=bool(
                self.get_parameter("fusion_allow_vo_only").value
            ),
        )
        self.localization_mode = self.localization_configuration.mode
        self.map_admission_requested = bool(
            self.get_parameter("map_admission_enabled").value
        )
        self.map_admission_gate = MapAdmissionGate(
            MapAdmissionConfig(
                # LIO and fused modes map with an external/fused pose.  A
                # visual-pose recovery gate must not suppress those trusted
                # geometry sources merely because RoMa lost texture.
                enabled=(
                    self.map_admission_requested
                    and self.localization_mode == "vo"
                ),
                min_recovery_inliers=int(
                    self.get_parameter(
                        "map_admission_min_recovery_inliers"
                    ).value
                ),
                min_recovery_inlier_ratio=float(
                    self.get_parameter(
                        "map_admission_min_recovery_inlier_ratio"
                    ).value
                ),
                stable_frames_required=int(
                    self.get_parameter(
                        "map_admission_stable_frames"
                    ).value
                ),
                max_translation_step_m=float(
                    self.get_parameter(
                        "map_admission_max_translation_step_m"
                    ).value
                ),
                max_rotation_step_deg=float(
                    self.get_parameter(
                        "map_admission_max_rotation_step_deg"
                    ).value
                ),
            )
        )
        self.lio_max_time_offset_sec = float(
            self.get_parameter("lio_max_time_offset_sec").value
        )
        self.lio_wait_timeout_sec = float(
            self.get_parameter("lio_wait_timeout_sec").value
        )
        self.lio_buffer_size = int(
            self.get_parameter("lio_buffer_size").value
        )
        self.lio_buffer_age_sec = float(
            self.get_parameter("lio_buffer_age_sec").value
        )
        self.fusion_map_rebuild_translation_m = float(
            self.get_parameter("fusion_map_rebuild_translation_m").value
        )
        self.fusion_map_rebuild_rotation_rad = np.deg2rad(
            float(
                self.get_parameter(
                    "fusion_map_rebuild_rotation_deg"
                ).value
            )
        )
        if self.lio_max_time_offset_sec < 0.0:
            raise ValueError("lio_max_time_offset_sec cannot be negative")
        if self.lio_wait_timeout_sec < 0.0:
            raise ValueError("lio_wait_timeout_sec cannot be negative")
        if self.lio_buffer_size <= 0 or self.lio_buffer_age_sec <= 0.0:
            raise ValueError("LIO buffer size and age must be positive")
        if (
            self.fusion_map_rebuild_translation_m < 0.0
            or self.fusion_map_rebuild_rotation_rad < 0.0
        ):
            raise ValueError("fusion map-rebuild thresholds cannot be negative")
        if self.localization_mode != "vo" and (
            bool(self.get_parameter("enable_backend").value)
            or bool(self.get_parameter("enable_loop_closing").value)
        ):
            raise ValueError(
                "External localisation modes currently require the internal "
                "VO backend and loop closing to be disabled.  Local fusion "
                "must be validated before the visual loop graph is connected "
                "as the global map->odom correction layer."
            )
        self.enable_lidar_mapping = bool(
            self.get_parameter("enable_lidar_mapping").value
        )
        self.lidar_mapping_mode = str(
            self.get_parameter("lidar_mapping_mode").value
        ).strip().lower()
        if self.lidar_mapping_mode not in {"direct", "assist"}:
            raise ValueError(
                "lidar_mapping_mode must be 'direct' or 'assist', got "
                f"{self.lidar_mapping_mode!r}"
            )
        self.lidar_expected_frame_id = str(
            self.get_parameter("lidar_expected_frame_id").value
        ).strip()
        self.lidar_max_time_offset_sec = float(
            self.get_parameter("lidar_max_time_offset_sec").value
        )
        self.lidar_wait_timeout_sec = float(
            self.get_parameter("lidar_wait_timeout_sec").value
        )
        self.lidar_temporal_configuration = LidarTemporalConfiguration(
            mode=str(self.get_parameter("lidar_temporal_mode").value),
            max_point_time_offset_sec=float(
                self.get_parameter("lidar_point_time_max_abs_sec").value
            ),
            pose_time_bin_sec=float(
                self.get_parameter("lidar_deskew_time_bin_sec").value
            ),
        )
        self.lidar_point_time_field = str(
            self.get_parameter("lidar_point_time_field").value
        ).strip()
        self.lidar_point_time_unit = str(
            self.get_parameter("lidar_point_time_unit").value
        ).strip()
        point_time_unit_scale(self.lidar_point_time_unit)
        if (
            self.lidar_temporal_configuration.requires_point_times
            and not self.lidar_point_time_field
        ):
            raise ValueError(
                "lidar_point_time_field cannot be empty in per_point mode"
            )
        if (
            self.enable_lidar_mapping
            and self.lidar_temporal_configuration.requires_external_pose
            and not self.localization_configuration.requires_external_odometry
        ):
            raise ValueError(
                "Accurate LiDAR temporal modes require external LIO poses; "
                "select localization_mode='lio' or 'vo_lio', or use the "
                "backward-compatible lidar_temporal_mode='none'"
            )
        self.lidar_min_points_per_keyframe = int(
            self.get_parameter("lidar_min_points_per_keyframe").value
        )
        self.lidar_buffer_size = int(
            self.get_parameter("lidar_buffer_size").value
        )
        self.lidar_message_buffer = LidarMessageBuffer(
            max_size=self.lidar_buffer_size,
            max_time_offset_sec=self.lidar_max_time_offset_sec,
            wait_timeout_sec=self.lidar_wait_timeout_sec,
        )
        self.lidar_fallback_to_stereo = bool(
            self.get_parameter("lidar_fallback_to_stereo").value
        )
        self.visual_frontend_enabled = visual_frontend_required(
            localization_mode=self.localization_mode,
            enable_lidar_mapping=self.enable_lidar_mapping,
            lidar_mapping_mode=self.lidar_mapping_mode,
            lidar_fallback_to_stereo=self.lidar_fallback_to_stereo,
        )
        self.lidar_align_map_to_sensor_frame = bool(
            self.get_parameter("lidar_align_map_to_sensor_frame").value
        )
        self.stereo_observation_configuration = (
            StereoObservationConfiguration(
                visible_max_points=visible_max_points,
                stereo_map_max_depth_m=stereo_map_max_depth_m,
                assist_pixel_radius=float(
                    self.get_parameter(
                        "pointcloud_assist_pixel_radius"
                    ).value
                ),
                assist_unmatched_stereo_max_depth_m=float(
                    self.get_parameter(
                        "pointcloud_assist_unmatched_stereo_max_depth_m"
                    ).value
                ),
                assist_replace_stereo_min_depth_m=float(
                    self.get_parameter(
                        "pointcloud_assist_replace_stereo_min_depth_m"
                    ).value
                ),
                assist_min_matches=int(
                    self.get_parameter(
                        "pointcloud_assist_min_matches"
                    ).value
                ),
            )
        )
        self.pointcloud_assist_max_points_per_keyframe = int(
            self.get_parameter(
                "pointcloud_assist_max_points_per_keyframe"
            ).value
        )
        if self.lidar_min_points_per_keyframe <= 0:
            raise ValueError("lidar_min_points_per_keyframe must be positive")
        if self.pointcloud_assist_max_points_per_keyframe <= 0:
            raise ValueError(
                "pointcloud_assist_max_points_per_keyframe must be positive"
            )

        self.enable_pointcloud_fusion = bool(
            self.get_parameter("enable_pointcloud_fusion").value
        )
        self.pointcloud_fusion_config = str(
            self.get_parameter("pointcloud_fusion_config").value
        ).strip()
        if self.enable_pointcloud_fusion and self.enable_lidar_mapping:
            raise ValueError(
                "enable_pointcloud_fusion and enable_lidar_mapping cannot "
                "be enabled together; the former is elevation-only while "
                "the latter changes stereo map observations"
            )
        if self.enable_pointcloud_fusion and any(
            bool(self.get_parameter(name).value)
            for name in (
                "enable_backend",
                "enable_loop_closing",
                "enable_global_loop_closure",
            )
        ):
            raise ValueError(
                "Elevation-only point-cloud fusion currently requires the "
                "frozen RoMa VO geometry (backend and loop correction off)"
            )

        self.enable_global_loop_closure = bool(
            self.get_parameter("enable_global_loop_closure").value
        )
        self.global_loop_candidate_source = str(
            self.get_parameter("global_loop_candidate_source").value
        ).strip().lower()
        if self.global_loop_candidate_source not in {
            "lidar",
            "mprf",
            "hybrid",
        }:
            raise ValueError(
                "global_loop_candidate_source must be lidar, mprf or hybrid"
            )
        if self.enable_global_loop_closure and self.localization_mode == "vo":
            raise ValueError(
                "The independent global loop layer is currently scoped to "
                "localization_mode='lio' or 'vo_lio'. Pure VO remains the "
                "frozen RoMa baseline with its established internal switches."
            )
        if self.enable_global_loop_closure and not self.enable_lidar_mapping:
            raise ValueError(
                "Global loop closure requires enable_lidar_mapping=true "
                "because every candidate must pass metric LiDAR verification"
            )
        self.global_loop_detector_configuration = LidarLoopConfiguration(
            enabled=(
                self.enable_global_loop_closure
                and self.global_loop_candidate_source in {"lidar", "hybrid"}
            ),
            submap_keyframes=int(
                self.get_parameter("global_loop_submap_keyframes").value
            ),
            voxel_size_m=float(
                self.get_parameter("global_loop_voxel_size_m").value
            ),
            max_points=int(
                self.get_parameter("global_loop_max_points").value
            ),
            geometry_cache_records=int(
                self.get_parameter(
                    "global_loop_geometry_cache_records"
                ).value
            ),
            descriptor_min_similarity=float(
                self.get_parameter(
                    "global_loop_scan_context_min_similarity"
                ).value
            ),
            min_keyframe_separation=int(
                self.get_parameter(
                    "global_loop_min_keyframe_separation"
                ).value
            ),
            min_time_separation_sec=float(
                self.get_parameter(
                    "global_loop_min_time_separation_sec"
                ).value
            ),
            icp_max_correspondence_m=float(
                self.get_parameter(
                    "global_loop_icp_max_correspondence_m"
                ).value
            ),
            icp_min_inliers=int(
                self.get_parameter("global_loop_icp_min_inliers").value
            ),
            icp_min_overlap=float(
                self.get_parameter("global_loop_icp_min_overlap").value
            ),
            icp_min_unique_target_ratio=float(
                self.get_parameter(
                    "global_loop_icp_min_unique_target_ratio"
                ).value
            ),
            icp_max_rmse_m=float(
                self.get_parameter("global_loop_icp_max_rmse_m").value
            ),
            max_correction_translation_m=float(
                self.get_parameter("global_loop_max_correction_m").value
            ),
            max_correction_rotation_deg=float(
                self.get_parameter("global_loop_max_correction_deg").value
            ),
            confirmation_count=int(
                self.get_parameter("global_loop_confirmation_count").value
            ),
        )
        self.global_loop_graph_options = {
            "odometry_rotation_sigma_deg": tuple(
                float(value)
                for value in self.get_parameter(
                    "global_graph_odometry_rotation_sigma_deg"
                ).value
            ),
            "odometry_translation_sigma_m": tuple(
                float(value)
                for value in self.get_parameter(
                    "global_graph_odometry_translation_sigma_m"
                ).value
            ),
            "robust_huber_k": float(
                self.get_parameter("global_graph_robust_huber_k").value
            ),
        }
        mprf_selected = self.global_loop_candidate_source in {
            "mprf",
            "hybrid",
        }
        self.mprf_client_configuration = MprfClientConfiguration(
            enabled=(self.enable_global_loop_closure and mprf_selected),
            endpoint=str(self.get_parameter("mprf_endpoint").value),
            request_timeout_sec=float(
                self.get_parameter("mprf_request_timeout_sec").value
            ),
            queue_size=int(self.get_parameter("mprf_queue_size").value),
            transient_retries=int(
                self.get_parameter("mprf_transient_retries").value
            ),
            auth_token_env=str(
                self.get_parameter("mprf_auth_token_env").value
            ),
        )
        self.mprf_poll_period_sec = float(
            self.get_parameter("mprf_poll_period_sec").value
        )
        if (
            not np.isfinite(self.mprf_poll_period_sec)
            or self.mprf_poll_period_sec <= 0.0
        ):
            raise ValueError("mprf_poll_period_sec must be finite and positive")

        pitch_deg = float(self.get_parameter("camera_pitch_deg").value)
        self.rotation_map_world = opencv_world_to_map_rotation(pitch_deg)
        self.map_translation_offset = np.array(
            [
                float(self.get_parameter("map_x_offset_m").value),
                float(self.get_parameter("map_y_offset_m").value),
                float(self.get_parameter("map_z_offset_m").value),
            ],
            dtype=np.float64,
        )

        origin_x = float(self.get_parameter("map_origin_x").value)
        origin_y = float(self.get_parameter("map_origin_y").value)
        if origin_x == _AUTO_MAP_ORIGIN:
            origin_x = 0.0
        elif not np.isfinite(origin_x):
            raise ValueError("map_origin_x must be finite")
        if origin_y == _AUTO_MAP_ORIGIN:
            origin_y = 0.0
        elif not np.isfinite(origin_y):
            raise ValueError("map_origin_y must be finite")

        self.local_map_length_x = float(
            self.get_parameter("map_length_x").value
        )
        self.local_map_length_y = float(
            self.get_parameter("map_length_y").value
        )
        if self.local_map_length_x <= 0.0 or self.local_map_length_y <= 0.0:
            raise ValueError("map_length_x and map_length_y must be positive")

        self.incremental_publish_mode = str(
            self.get_parameter("incremental_publish_mode").value
        ).strip().lower()
        supported_incremental_modes = {
            "fixed",
            "rolling_window",
            "full_tiles",
        }
        if self.incremental_publish_mode not in supported_incremental_modes:
            raise ValueError(
                "incremental_publish_mode must be one of "
                f"{sorted(supported_incremental_modes)}, got "
                f"{self.incremental_publish_mode!r}"
            )
        self.incremental_publish_period_sec = float(
            self.get_parameter("incremental_publish_period_sec").value
        )
        if (
            not np.isfinite(self.incremental_publish_period_sec)
            or self.incremental_publish_period_sec <= 0.0
        ):
            raise ValueError(
                "incremental_publish_period_sec must be finite and positive"
            )
        requested_dense_window_mode = str(
            self.get_parameter("dense_map_window_mode").value
        ).strip().lower()
        if requested_dense_window_mode == "follow_incremental":
            self.dense_map_window_mode = (
                "fixed"
                if self.incremental_publish_mode == "fixed"
                else "rolling"
            )
        elif requested_dense_window_mode in {"fixed", "rolling"}:
            self.dense_map_window_mode = requested_dense_window_mode
        else:
            raise ValueError(
                "dense_map_window_mode must be follow_incremental, fixed, "
                f"or rolling, got {requested_dense_window_mode!r}"
            )
        # The original dense map defaulted to [-30, 30) m for a 60 m map.
        # Pinning the compatibility window to the tile-lattice origin exactly
        # reproduces that geometry with the default (0, 0) lattice origin.
        self.fixed_map_center_x = origin_x
        self.fixed_map_center_y = origin_y

        self._grid_kwargs = {
            "resolution": float(self.get_parameter("map_resolution").value),
            "tile_cells": int(self.get_parameter("map_tile_cells").value),
            "origin_x": origin_x,
            "origin_y": origin_y,
            "class_names": CLASS_NAMES,
            "map_id": str(self.get_parameter("map_id").value),
            "free_class_ids": [
                int(value)
                for value in self.get_parameter("free_class_ids").value
            ],
            "occupied_class_ids": [
                int(value)
                for value in self.get_parameter("occupied_class_ids").value
            ],
            "max_rays_per_update": int(
                self.get_parameter("max_rays_per_update").value
            ),
            "lidar_ground_max_slope_deg": float(
                self.get_parameter("lidar_ground_max_slope_deg").value
            ),
            "lidar_ground_neighbor_count": int(
                self.get_parameter("lidar_ground_neighbor_count").value
            ),
            "lidar_ground_max_neighbor_distance_m": float(
                self.get_parameter(
                    "lidar_ground_max_neighbor_distance_m"
                ).value
            ),
        }
        self.map_observation_archive_enabled = bool(
            self.get_parameter("map_observation_archive_enabled").value
        )
        self.map_observation_cache_size = int(
            self.get_parameter("map_observation_cache_size").value
        )
        self.map_observation_compression_level = int(
            self.get_parameter(
                "map_observation_compression_level"
            ).value
        )
        if self.map_observation_cache_size <= 0:
            raise ValueError("map_observation_cache_size must be positive")
        if not 0 <= self.map_observation_compression_level <= 9:
            raise ValueError(
                "map_observation_compression_level must be between 0 and 9"
            )

        self.output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Keep the run root limited to public deliverables. Reconstruction,
        # camera-frame and paging artifacts live under one clearly internal
        # directory and are not part of the external file interface.
        self.internal_output_dir = self.output_dir / "_internal"
        self.internal_output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.internal_output_dir / "grid_map_snapshot"
        self.compact_global_map_path = (
            self.output_dir / self.compact_global_map_filename
        )
        self.dense_global_grid_map_path = (
            self.output_dir / self.dense_global_grid_map_filename
        )
        observation_archive_path = None
        if self.map_observation_archive_enabled:
            archive_directory = (
                self.internal_output_dir / "map_observation_archive"
            )
            archive_directory.mkdir(parents=True, exist_ok=True)
            observation_archive_path = archive_directory / (
                f"observations_{time.time_ns()}.sqlite3"
            )

        self._map_lock = threading.RLock()
        self.grid = self._make_empty_grid()
        overview_factor = round(
            self.global_overview_resolution / self.grid.resolution
        )
        if overview_factor <= 0 or not np.isclose(
            self.global_overview_resolution,
            overview_factor * self.grid.resolution,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "global_overview_resolution must be an integer multiple of "
                "map_resolution"
            )
        self.map_transfer_session = MapTransferSession(
            map_id=self.grid.map_id
        )
        self.map_observations = KeyFrameObservationStore(
            observation_archive_path,
            memory_cache_size=(
                self.map_observation_cache_size
                if self.map_observation_archive_enabled
                else None
            ),
            compression_level=self.map_observation_compression_level,
        )
        self.map_contribution_index = MapContributionIndex()
        self.map_correction_worker = MapCorrectionWorker()
        self.pending_map_observations: dict[
            int, SemanticKeyFrameObservation
        ] = {}
        # keyframe_created 是一次性信号：观测构建（快照读取/语义采样）
        # 异常时缓存该关键帧的同帧语义图，在统一同步边界重试补建，
        # 否则该关键帧的地图证据永久丢失。事务契约只覆盖暂存之后的失败。
        self.missed_keyframe_observations: dict[int, dict] = {}
        self.missed_observation_cache_limit = 8
        self.missed_observation_warning_interval = 20
        self.map_revision = 0
        self.map_message_update_id = 0
        self._last_published_map_signature = None
        self._last_published_global_overview_revision = -1
        self._last_published_global_grid_map_revision = -1
        self._last_enqueued_dense_grid_map_revision = -1
        self._last_published_compact_map_revision = -1
        self._last_compact_map_error_text = None
        self.map_output_publish_count = 0
        self.map_output_skip_count = 0
        self.applied_geometry_revision = 0
        self.map_is_stale = False
        self.map_integrity_fault: Optional[str] = None
        self.compact_global_map_writer = LiveCompactGlobalMapWriter(
            self.compact_global_map_path,
            initial_grid=self.grid.clone_for_update(),
            frame_id=self.map_frame,
            compression_level=self.compact_global_map_compression_level,
            parallel_workers=self.map_persistence_workers,
        )
        self.dense_global_grid_map_writer = (
            LiveDenseGlobalMapWriter(
                self.dense_global_grid_map_path,
                frame_id=self.map_frame,
            )
            if self.save_dense_global_grid_map
            else None
        )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.received_frame_count = 0
        self.tracking_failure_count = 0
        self.processing = False
        self.input_monitor = InputProcessingMonitor()
        self.last_stamp = None
        self.last_header = None
        self.last_pose_map = np.eye(4, dtype=np.float64)
        self.last_path_publish_frame_count = 0
        self.pose_stamps: dict[int, tuple[int, int]] = {}
        self.trajectory_file_stamps: dict[int, tuple[int, int]] = {}
        self.lidar_skipped_count = 0
        self._lio_condition = threading.Condition()
        self.lio_rejected_count = 0
        self.last_lio_diagnostic = "not requested"
        # Raw local estimates are retained separately from globally corrected
        # output.  This is the core no-double-counting boundary.
        self.localization_odom_pose_history: dict[int, np.ndarray] = {}
        self.localization_pose_history: dict[int, np.ndarray] = {}
        self.localization_keyframe_odom_poses: dict[int, np.ndarray] = {}
        self.localization_keyframe_poses: dict[int, np.ndarray] = {}
        self.localization_keyframe_output_indices: dict[int, int] = {}
        self.localization_geometry_revision = 0
        self.global_loop_closure = None
        self.mprf_client = None
        self._global_loop_registered_nodes: set[int] = set()
        self._mprf_submitted_nodes: set[int] = set()
        self._last_global_lidar_timestamp = float("-inf")
        self._global_path_rebuild_pending = False
        self._global_loop_error_count = 0

        self.imu_buffer = RosImuBuffer(
            camera_pitch_deg=pitch_deg,
            max_age_sec=300.0,
            max_boundary_extrapolation_sec=1.0,
        )
        self.imu_alignment_source = "legacy camera_pitch_deg fallback"
        self.map_alignment_source = "camera_pitch_deg fallback"

        self.calibration, self.slam = self._create_slam()
        self.stereo_rectifier = StereoRectifier(self.calibration)
        calibrated_imu_transform = (
            load_rectified_mounting_to_camera_transform(
                self.calibration,
                sensor_key="transform_vehicle_imu",
            )
        )
        if calibrated_imu_transform is not None:
            self.imu_buffer.set_rotation_camera_imu(
                calibrated_imu_transform[:3, :3]
            )
            self.imu_alignment_source = (
                "calibrated vehicle/IMU mounting + left rectification"
            )
        calibrated_alignment = self.calibration.get("map_alignment", {}).get(
            "rotation_camera_to_map"
        )
        if calibrated_alignment is not None:
            calibrated_rotation = np.asarray(
                calibrated_alignment, dtype=np.float64
            ).reshape(3, 3)
            if not np.isfinite(calibrated_rotation).all() or not np.allclose(
                calibrated_rotation @ calibrated_rotation.T,
                np.eye(3),
                atol=1e-6,
            ):
                raise ValueError(
                    "calibration map_alignment.rotation_camera_to_map is invalid"
                )
            self.rotation_map_world = calibrated_rotation
            calibrated_translation = np.asarray(
                self.calibration.get("map_alignment", {}).get(
                    "translation_camera_to_map", [0.0, 0.0, 0.0]
                ),
                dtype=np.float64,
            ).reshape(3)
            if not np.isfinite(calibrated_translation).all():
                raise ValueError(
                    "calibration map_alignment.translation_camera_to_map "
                    "is invalid"
                )
            # Keep the explicit launch offsets as an optional global shift,
            # but place the unshifted map origin at the initial vehicle body
            # rather than at the left-camera optical centre.
            self.map_translation_offset = (
                calibrated_translation + self.map_translation_offset
            )
            self.map_alignment_source = (
                "calibrated initial base_link + left rectification"
            )
        self.stereo_observation_builder = (
            StereoObservationBuilder(
                self.slam,
                self.stereo_observation_configuration,
            )
            if self.slam is not None
            else None
        )
        self.applied_geometry_revision = (
            int(getattr(self.slam, "geometry_revision", 0))
            if self.localization_mode == "vo"
            else 0
        )
        self.external_odometry_adapter = None
        self.lio_transform_camera_lidar = None
        self.localization_coordinator = (
            self._create_localization_coordinator()
        )
        if (
            self.localization_configuration.requires_external_odometry
            and self.lidar_align_map_to_sensor_frame
        ):
            # The external adapter has already converted body/LiDAR odometry
            # into a first-camera-relative trajectory.  Align that first
            # camera world to the calibrated REP-103 LiDAR axes even when raw
            # LiDAR geometry is not selected for map evidence.
            self.rotation_map_world = (
                self.lio_transform_camera_lidar[:3, :3].T.copy()
            )
            self.map_alignment_source = "LiDAR sensor frame"
        self.lidar_processor = None
        if self.enable_lidar_mapping:
            configured_extrinsics = str(
                self.get_parameter("lidar_extrinsics_path").value
            ).strip()
            self.lidar_processor = LidarMappingProcessor.from_calibration(
                self.calibration,
                extrinsics_path=(
                    Path(configured_extrinsics)
                    if configured_extrinsics
                    else None
                ),
                class_count=len(CLASS_NAMES),
                min_range_m=float(
                    self.get_parameter("lidar_min_range_m").value
                ),
                max_range_m=float(
                    self.get_parameter("lidar_max_range_m").value
                ),
                voxel_size_m=float(
                    self.get_parameter("lidar_voxel_size_m").value
                ),
                max_points=int(
                    self.pointcloud_assist_max_points_per_keyframe
                    if self.lidar_mapping_mode == "assist"
                    else self.get_parameter(
                        "lidar_max_points_per_keyframe"
                    ).value
                ),
            )
            if self.lidar_temporal_configuration.requires_external_pose:
                if self.lio_transform_camera_lidar is None:
                    raise RuntimeError(
                        "LiDAR temporal alignment has no verified external "
                        "LIO calibration chain"
                    )
                if not np.allclose(
                    self.lio_transform_camera_lidar,
                    self.lidar_processor.transform_camera_lidar,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "LiDAR temporal alignment requires the mapping input "
                        "frame and external-LIO LiDAR frame to use the same "
                        "full T_camera_sensor calibration"
                    )
            if self.lidar_align_map_to_sensor_frame:
                # At the first frame the SLAM world is the rectified optical
                # camera frame.  The calibrated inverse rotation maps that
                # world into the LiDAR frame, whose REP-103 axes are the
                # desired map X-forward, Y-left, Z-up convention.  This also
                # removes the need to guess camera_pitch_deg in LiDAR mode.
                lidar_rotation = (
                    self.lidar_processor.rotation_camera_lidar.copy()
                )
                if (
                    self.lio_transform_camera_lidar is not None
                    and not np.allclose(
                        self.lio_transform_camera_lidar[:3, :3],
                        lidar_rotation,
                        atol=1e-6,
                    )
                ):
                    raise ValueError(
                        "LIO and raw-LiDAR mapping use inconsistent "
                        "LiDAR-to-camera rotations"
                    )
                self.rotation_map_world = lidar_rotation.T

        self.terrain_pointcloud_mapper = None
        if self.enable_pointcloud_fusion:
            if not self.pointcloud_fusion_config:
                raise ValueError(
                    "pointcloud_fusion_config is required when point-cloud "
                    "fusion is enabled"
                )
            mapper = TerrainPointCloudMapper(
                Path(self.pointcloud_fusion_config),
                logger=self.get_logger(),
            )
            if mapper.base_frame != self.child_frame:
                raise ValueError(
                    "Point-cloud base frame does not match odometry child: "
                    f"{mapper.base_frame!r} != {self.child_frame!r}"
                )
            if mapper.map_frame != self.map_frame:
                raise ValueError(
                    "Point-cloud map frame does not match mapper frame: "
                    f"{mapper.map_frame!r} != {self.map_frame!r}"
                )
            self.terrain_pointcloud_mapper = mapper

        if self.enable_global_loop_closure:
            self.global_loop_closure = GlobalLoopClosure(
                detector_configuration=(
                    self.global_loop_detector_configuration
                ),
                graph_options=self.global_loop_graph_options,
                detector_archive_directory=(
                    self.internal_output_dir
                    / "loop_closure_archive"
                    / f"run_{time.time_ns()}"
                ),
            )
            self.mprf_client = MprfAsyncClient(
                self.mprf_client_configuration
            )

        self._open_trajectory_files()

        self._create_ros_interfaces()
        self._log_startup_configuration()

    def _create_ros_interfaces(self) -> None:
        """Create ROS publishers, subscriptions, timers, and services."""

        stereo_input_qos = self.input_transport.stereo_qos
        semantic_input_qos = self.input_transport.semantic_qos
        imu_input_qos = self.input_transport.imu_qos
        live_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        debug_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        incremental_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=(
                10
                if self.incremental_publish_mode == "full_tiles"
                else 1
            ),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        map_batch_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.odometry_pub = self.create_publisher(
            Odometry,
            str(self.get_parameter("odometry_topic").value),
            live_output_qos,
        )
        odometry_topic = str(
            self.get_parameter("odometry_topic").value
        ).strip()
        odometry_compat_topic = str(
            self.get_parameter("odometry_compat_topic").value
        ).strip()
        self.odometry_compat_pub = (
            self.create_publisher(
                Odometry,
                odometry_compat_topic,
                live_output_qos,
            )
            if (
                bool(
                    self.get_parameter("publish_odometry_compat").value
                )
                and
                odometry_compat_topic
                and odometry_compat_topic != odometry_topic
            )
            else None
        )
        self.input_processed_pub = (
            self.create_publisher(
                Header,
                str(self.get_parameter("input_processed_topic").value),
                live_output_qos,
            )
            if bool(
                self.get_parameter("publish_input_processed").value
            )
            else None
        )
        self.realtime_metrics_pub = (
            self.create_publisher(
                String,
                str(self.get_parameter("realtime_metrics_topic").value),
                live_output_qos,
            )
            if bool(self.get_parameter("publish_realtime_metrics").value)
            else None
        )
        self.realtime_metrics_sequence = 0
        self.path_pub = self.create_publisher(
            NavPath,
            str(self.get_parameter("trajectory_topic").value),
            map_output_qos,
        )
        self.feature_matches_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("feature_matches_topic").value),
                debug_image_qos,
            )
            if self.publish_feature_matches and self.visual_frontend_enabled
            else None
        )
        self.feature_visualization_crop_width = int(
            self.get_parameter("feature_visualization_crop_width").value
        )
        self.feature_visualization_crop_height = int(
            self.get_parameter("feature_visualization_crop_height").value
        )
        self.feature_visualization_width = int(
            self.get_parameter("feature_visualization_width").value
        )
        self.feature_visualization_height = int(
            self.get_parameter("feature_visualization_height").value
        )
        if (
            self.feature_visualization_crop_width <= 0
            or self.feature_visualization_crop_height <= 0
            or self.feature_visualization_width <= 0
            or self.feature_visualization_height <= 0
        ):
            raise ValueError(
                "Feature visualization dimensions must be positive"
            )
        self.pointcloud_pub = (
            self.create_publisher(
                PointCloud2,
                str(self.get_parameter("pointcloud_topic").value),
                map_output_qos,
            )
            if self.publish_map_pointcloud
            else None
        )
        self.occupancy_pub = (
            self.create_publisher(
                OccupancyGrid,
                str(self.get_parameter("occupancy_topic").value),
                map_output_qos,
            )
            if self.publish_local_occupancy_grid
            else None
        )
        self.semantic_grid_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("semantic_grid_topic").value),
                map_output_qos,
            )
            if self.publish_layer_images
            else None
        )
        self.semantic_color_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("semantic_color_topic").value),
                map_output_qos,
            )
            if self.publish_layer_images
            else None
        )
        self.elevation_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("elevation_topic").value),
                map_output_qos,
            )
            if self.publish_layer_images
            else None
        )
        self.elevation_variance_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("elevation_variance_topic").value),
                map_output_qos,
            )
            if self.publish_layer_images
            else None
        )
        self.elevation_color_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("elevation_color_topic").value),
                map_output_qos,
            )
            if self.publish_layer_images
            else None
        )

        from grid_map_msgs.msg import GridMap

        self.grid_map_pub = self.create_publisher(
            GridMap,
            str(self.get_parameter("grid_map_topic").value),
            map_output_qos,
        )
        grid_map_topic = str(
            self.get_parameter("grid_map_topic").value
        ).strip()
        grid_map_odom_topic = str(
            self.get_parameter("grid_map_odom_topic").value
        ).strip()
        self.grid_map_odom_pub = (
            self.create_publisher(
                GridMap,
                grid_map_odom_topic,
                map_output_qos,
            )
            if (
                bool(
                    self.get_parameter(
                        "publish_grid_map_odom_alias"
                    ).value
                )
                and grid_map_odom_topic
                and grid_map_odom_topic != grid_map_topic
            )
            else None
        )
        self.global_grid_map_pub = (
            self.create_publisher(
                GridMap,
                str(self.get_parameter("global_grid_map_topic").value),
                map_output_qos,
            )
            if self.publish_global_grid_map
            else None
        )
        self.global_overview_pub = (
            self.create_publisher(
                OccupancyGrid,
                str(self.get_parameter("global_overview_topic").value),
                map_output_qos,
            )
            if self.publish_global_overview
            else None
        )
        self.global_map_revision_pub = self.create_publisher(
            UInt64,
            str(self.get_parameter("global_map_revision_topic").value),
            map_output_qos,
        )
        self.incremental_map_pub = (
            self.create_publisher(
                IncrementalSemanticMap,
                str(self.get_parameter("incremental_map_topic").value),
                incremental_output_qos,
            )
            if self.publish_incremental_map
            else None
        )
        self.incremental_map_batch_pub = (
            self.create_publisher(
                SemanticMapBatch,
                str(
                    self.get_parameter(
                        "incremental_map_batch_topic"
                    ).value
                ),
                map_batch_output_qos,
            )
            if (
                self.publish_incremental_map
                and self.incremental_publish_mode == "full_tiles"
            )
            else None
        )
        self._incremental_subscription_count = 0
        self._incremental_subscription_gids = frozenset()
        self._incremental_replay_state = None
        self._incremental_replay_requested = False
        self._incremental_delta_requested = False
        self._incremental_next_delta_monotonic = 0.0
        self.incremental_replay_timer = (
            self.create_timer(
                0.10,
                self._on_incremental_replay_timer,
            )
            if (
                self.publish_incremental_map
                and self.incremental_publish_mode == "full_tiles"
            )
            else None
        )
        self.global_overview_timer = (
            self.create_timer(
                self.global_overview_period_sec,
                self._publish_global_overview_if_changed,
            )
            if self.publish_global_overview
            else None
        )
        self.compact_map_status_timer = self.create_timer(
            0.5,
            self._publish_compact_map_status,
        )
        # A correction that finishes after the robot stops producing images
        # must still become visible. The timer is idle while a job is running
        # and only enters the commit path once its immutable result is ready.
        self.map_correction_timer = self.create_timer(
            0.05,
            self._on_map_correction_timer,
        )
        # The robot may stop immediately after submitting its last keyframe.
        # Polling only from the image callback would strand a completed MPRF
        # result until another image or process shutdown.  Keep this timer in
        # the default mutually-exclusive group so graph/map mutations remain
        # serial with the main mapping transaction.
        self.mprf_result_timer = (
            self.create_timer(
                self.mprf_poll_period_sec,
                self._on_mprf_result_timer,
            )
            if self.mprf_client is not None
            else None
        )

        self.path_message = NavPath()
        self.path_message.header.frame_id = self.map_frame

        self.tf_broadcaster = (
            TransformBroadcaster(self)
            if self.publish_tf_enabled
            else None
        )

        # IMU buffering is lock-protected and does not touch RoMa or map
        # state.  A separate mutually-exclusive group lets it keep receiving
        # samples while the default group performs long image inference.
        self.imu_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if self.enable_imu
            else None
        )
        self.imu_subscription = (
            self.create_subscription(
                Imu,
                str(self.get_parameter("imu_topic").value),
                self._on_imu,
                imu_input_qos,
                callback_group=self.imu_callback_group,
            )
            if self.enable_imu
            else None
        )
        lidar_input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=min(self.lidar_buffer_size, 10),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # RoMa inference can occupy the default callback group for more than
        # one LiDAR period. Give each lightweight sensor buffer its own
        # mutually-exclusive group; main() supplies one worker per active
        # group while all image/SLAM/map work remains serialised.
        self.lidar_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if self.enable_lidar_mapping
            else None
        )
        self.lidar_subscription = (
            self.create_subscription(
                PointCloud2,
                str(self.get_parameter("lidar_topic").value),
                self._on_lidar,
                lidar_input_qos,
                callback_group=self.lidar_callback_group,
            )
            if self.enable_lidar_mapping
            else None
        )
        pointcloud_fusion_input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pointcloud_fusion_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if self.enable_pointcloud_fusion
            else None
        )
        self.pointcloud_fusion_subscriptions = []
        if self.terrain_pointcloud_mapper is not None:
            for sensor_name, sensor in (
                self.terrain_pointcloud_mapper.sensors.items()
            ):
                subscription = self.create_subscription(
                    PointCloud2,
                    sensor.topic,
                    lambda message, name=sensor_name: (
                        self._on_terrain_pointcloud(name, message)
                    ),
                    pointcloud_fusion_input_qos,
                    callback_group=self.pointcloud_fusion_callback_group,
                )
                self.pointcloud_fusion_subscriptions.append(subscription)

        lio_input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=min(self.lio_buffer_size, 100),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.lio_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if self.localization_configuration.requires_external_odometry
            else None
        )
        self.lio_subscription = (
            self.create_subscription(
                Odometry,
                str(self.get_parameter("lio_odometry_topic").value),
                self._on_lio_odometry,
                lio_input_qos,
                callback_group=self.lio_callback_group,
            )
            if self.localization_configuration.requires_external_odometry
            else None
        )

        self.left_subscriber = Subscriber(
            self,
            Image,
            str(self.get_parameter("left_topic").value),
            qos_profile=stereo_input_qos,
        )
        self.right_subscriber = Subscriber(
            self,
            Image,
            str(self.get_parameter("right_topic").value),
            qos_profile=stereo_input_qos,
        )
        self.semantic_subscriber = (
            Subscriber(
                self,
                Image,
                str(self.get_parameter("semantic_topic").value),
                qos_profile=semantic_input_qos,
            )
            if self.enable_semantics
            else None
        )
        now_callback = lambda: self.get_clock().now().to_msg()
        self.left_input_filter = TimestampSourceFilter(
            self.left_subscriber,
            timestamp_source=self.image_timestamp_source,
            now_callback=now_callback,
        )
        self.right_input_filter = TimestampSourceFilter(
            self.right_subscriber,
            timestamp_source=self.image_timestamp_source,
            now_callback=now_callback,
        )
        self.semantic_input_filter = (
            TimestampSourceFilter(
                self.semantic_subscriber,
                timestamp_source=self.image_timestamp_source,
                now_callback=now_callback,
            )
            if self.semantic_subscriber is not None
            else None
        )
        self._last_input_diagnostic_counts = (0, 0, 0)
        self._last_input_diagnostic_sync_count = 0
        self._empty_input_diagnostic_intervals = 0
        self._empty_input_warning_reported = False
        synchronized_filters = [
            self.left_input_filter,
            self.right_input_filter,
        ]
        if self.semantic_input_filter is not None:
            synchronized_filters.append(self.semantic_input_filter)
        self.synchronizer = self.input_transport.create_synchronizer(
            synchronized_filters
        )
        if self.enable_semantics:
            self.synchronizer.registerCallback(self._on_synchronized_input)
        else:
            self.synchronizer.registerCallback(self._on_synchronized_stereo)
        # Register passive diagnostics after the primary synchronizer.  Even
        # if diagnostic metadata is malformed, it must never block mapping.
        diagnostic_subscribers = [
            ("left", self.left_subscriber),
            ("right", self.right_subscriber),
        ]
        if self.semantic_subscriber is not None:
            diagnostic_subscribers.append(
                ("semantic", self.semantic_subscriber)
            )
        for name, subscriber in diagnostic_subscribers:
            subscriber.registerCallback(self._record_image_input, name)
        self.input_diagnostic_timer = self.create_timer(
            5.0, self._report_unsynchronized_input
        )

        self.save_service = self.create_service(
            Trigger,
            "/Car/T3/mapping/save_grid_map",
            self._on_save_service,
        )
        self.grid_map_query_server = self.create_service(
            GetGridMap,
            str(self.get_parameter("grid_map_query_service").value),
            self._on_get_grid_map,
        )
        self.incremental_replay_service = (
            self.create_service(
                Trigger,
                str(
                    self.get_parameter(
                        "incremental_map_replay_service"
                    ).value
                ),
                self._on_incremental_replay_service,
            )
            if (
                self.publish_incremental_map
                and self.incremental_publish_mode == "full_tiles"
            )
            else None
        )

    def _record_image_input(self, message: Image, source: str) -> None:
        """Record only transport metadata; image payloads are never copied."""

        stamp_ns = None
        try:
            stamp_ns = _stamp_nanoseconds_value(
                message.header.stamp
            )
        except (AttributeError, TypeError, ValueError):
            # Diagnostics are deliberately non-authoritative. The formal
            # synchronizer remains responsible for enforcing Header.stamp.
            pass
        self.input_monitor.record_delivered(source, stamp_ns)

    def _report_unsynchronized_input(self) -> None:
        """Explain a live input stall without producing periodic log spam."""

        names = (
            ("left", "right", "semantic")
            if self.enable_semantics
            else ("left", "right")
        )
        counts = tuple(
            self.input_monitor.delivered_counts[name] for name in names
        )
        if not any(counts):
            self._empty_input_diagnostic_intervals += 1
            if (
                self._empty_input_diagnostic_intervals >= 2
                and not self._empty_input_warning_reported
            ):
                topic_parameters = ["left_topic", "right_topic"]
                if self.enable_semantics:
                    topic_parameters.append("semantic_topic")
                topics = tuple(
                    str(self.get_parameter(parameter).value)
                    for parameter in topic_parameters
                )
                publishers = tuple(
                    self.count_publishers(topic) for topic in topics
                )
                self.get_logger().warning(
                    "No image input received during the first 10 seconds "
                    "after mapper readiness. "
                    f"discovered_publishers({'/'.join(names)})={publishers}. "
                    "Check topic names and publisher/subscriber QoS."
                )
                self._empty_input_warning_reported = True
            return
        if counts == self._last_input_diagnostic_counts:
            return

        previous_sync_count = self._last_input_diagnostic_sync_count
        self._last_input_diagnostic_counts = counts
        self._last_input_diagnostic_sync_count = (
            self.input_monitor.synchronized_count
        )
        if self.input_monitor.synchronized_count != previous_sync_count:
            return

        latest = ", ".join(
            f"{name}={self.input_monitor.latest_stamp_ns[name]}"
            for name in names
        )
        self.get_logger().warning(
            "Image inputs arrived but formed no new synchronized group in "
            "the last diagnostic interval. "
            f"counts(left/right/semantic)={counts}, latest_stamp_ns({latest}), "
            f"policy={self.input_transport.sync_description}. Check missing "
            "streams, publisher/subscriber QoS, and acquisition Header.stamp."
        )

    def _on_synchronized_stereo(
        self,
        left_message: Image,
        right_message: Image,
    ) -> None:
        """Run visual-only tests without requiring a semantic producer."""

        self._on_synchronized_input(left_message, right_message, None)

    def _log_startup_configuration(self) -> None:
        """Log the effective map, localisation, and sensor configuration."""
        dense_window_description = (
            "fixed at the original map centre"
            if self.dense_map_window_mode == "fixed"
            else "centred near the robot"
        )
        if not self.enable_lidar_mapping:
            depth_limit = (
                self.stereo_observation_configuration.stereo_map_max_depth_m
            )
            map_geometry_description = (
                f"stereo (map-only optical depth Z<={depth_limit:.3f}m)"
                if depth_limit > 0.0
                else "stereo (map depth gate disabled)"
            )
        elif self.lidar_mapping_mode == "assist":
            map_geometry_description = (
                "PointCloud2-assisted stereo depth at RoMa keyframes"
            )
        else:
            map_geometry_description = (
                "direct PointCloud2 geometry at RoMa keyframes"
                if self.visual_frontend_enabled
                else "direct PointCloud2 geometry at periodic LIO keyframes"
            )
        map_alignment_description = self.map_alignment_source
        temporal_mode = self.lidar_temporal_configuration.mode
        temporal_description = {
            "none": "compatibility/no motion correction",
            "upstream_deskewed": (
                "upstream point deskew + LIO cloud-to-image alignment"
            ),
            "per_point": (
                "header-relative point time + LIO point/cloud alignment"
            ),
        }[temporal_mode]
        admission_config = self.map_admission_gate.config
        if admission_config.enabled:
            map_admission_description = (
                "active for pure VO "
                f"(inliers>={admission_config.min_recovery_inliers}, "
                f"ratio>={admission_config.min_recovery_inlier_ratio:.2f}, "
                f"stable_frames={admission_config.stable_frames_required}, "
                f"step<={admission_config.max_translation_step_m:.2f}m/"
                f"{admission_config.max_rotation_step_deg:.1f}deg)"
            )
        elif self.map_admission_requested:
            map_admission_description = (
                "bypassed because localisation uses an external/fused pose"
            )
        else:
            map_admission_description = "disabled by parameter"
        map_publication_description = (
            "every valid pose"
            if self.publish_unchanged_map_outputs
            else "map revision/window-cell changes"
        )
        imu_input_description = (
            f"{self.get_parameter('imu_topic').value} "
            f"(sensor_msgs/Imu, "
            f"{self.input_transport.imu_reliability}, "
            f"depth={self.input_transport.imu_qos_depth}, "
            f"alignment={self.imu_alignment_source})"
            if self.enable_imu
            else "disabled"
        )
        semantic_input_description = (
            f"{self.get_parameter('semantic_topic').value} "
            f"(sensor_msgs/Image, "
            f"{self.input_transport.semantic_reliability}, "
            f"depth={self.input_transport.image_qos_depth})"
            if self.enable_semantics
            else "disabled (zero labels used internally)"
        )
        pointcloud_fusion_description = (
            self.terrain_pointcloud_mapper.description
            if self.terrain_pointcloud_mapper is not None
            else "disabled"
        )
        self.get_logger().info(
            "Mapper ready. Waiting for synchronized image input.\n"
            f"  stereo input: {self.get_parameter('left_topic').value}, "
            f"{self.get_parameter('right_topic').value} "
            f"(sensor_msgs/Image, "
            f"{self.input_transport.stereo_reliability}, "
            f"depth={self.input_transport.image_qos_depth})\n"
            f"  semantic input: {semantic_input_description}\n"
            f"  IMU input: {imu_input_description}\n"
            f"  point-cloud elevation fusion: "
            f"{pointcloud_fusion_description}\n"
            f"  image timestamp sync: "
            f"{self.input_transport.sync_description}, "
            f"source={self.image_timestamp_source}\n"
            f"  input completion receipt: "
            f"{self.get_parameter('input_processed_topic').value if self.input_processed_pub is not None else 'disabled'}\n"
            f"  localisation: {self.localization_mode} "
            f"(backend={self.localization_configuration.fusion_backend})\n"
            f"  visual frontend: "
            f"{'enabled' if self.visual_frontend_enabled else 'bypassed'}\n"
            f"  feature-match debug: "
            f"{self.get_parameter('feature_matches_topic').value} "
            f"({'subscriber-gated' if self.feature_matches_pub is not None else 'disabled'}, "
            f"width={self.feature_match_output_width}, "
            f"lines<={self.feature_match_max_lines})\n"
            f"  RoMa inference: coarse="
            f"{self.get_parameter('coarse_res').value}, upsample="
            f"{self.get_parameter('upsample_res').value}, "
            f"stereo_refinement="
            f"{self.get_parameter('stereo_upsample_refinement').value}, "
            f"temporal_refinement="
            f"{self.get_parameter('temporal_upsample_refinement').value}, "
            f"symmetric="
            f"{self.get_parameter('symmetric_matching').value}, "
            f"custom_corr={self.get_parameter('use_custom_corr').value}, "
            f"feature_reuse="
            f"{bool(getattr(self.slam, 'feature_reuse_active', False))}, "
            f"batch_coarse="
            f"{self.get_parameter('batch_coarse_pairs').value}, "
            f"temporal_klt="
            f"{self.get_parameter('enable_klt_temporal_tracking').value}\n"
            f"  TF: {self.map_frame}->{self.odom_frame}->"
            f"{self.child_frame}\n"
            f"  Odometry: {self.get_parameter('odometry_topic').value} "
            f"({self.odom_frame}->{self.child_frame}, "
            f"estimated frame={self.odometry_output.camera_frame}, "
            "velocity=source-or-backward-difference, "
            f"unknown variance="
            f"{self.odometry_output.unknown_covariance_variance:.3g})\n"
            f"  GridMap: {self.get_parameter('grid_map_topic').value} "
            f"(frame={self.local_grid_map_frame})\n"
            f"  global map: sparse {self.grid.tile_cells}x"
            f"{self.grid.tile_cells}-cell tiles @ "
            f"{self.grid.resolution:.3f} m/cell\n"
            f"  dense publish window: {self.local_map_length_x:.1f}x"
            f"{self.local_map_length_y:.1f} m "
            f"({dense_window_description})\n"
            f"  GridMap layers: {', '.join(self.local_grid_map_layers)}\n"
            f"  complete global GridMap: "
            f"{self.get_parameter('global_grid_map_topic').value} "
            f"({'revision-triggered' if self.global_grid_map_pub is not None else 'disabled'}, "
            f"layers={','.join(self.global_grid_map_layers)})\n"
            f"  IncrementalSemanticMap: "
            f"{'enabled (' + self.incremental_publish_mode + ')' if self.publish_incremental_map else 'disabled'}\n"
            f"  global overview: "
            f"{self.get_parameter('global_overview_topic').value} @ "
            f"{self.global_overview_resolution:.3f} m/cell every "
            f"{self.global_overview_period_sec:.1f}s\n"
            f"  regional query: "
            f"{self.get_parameter('grid_map_query_service').value} "
            f"(max_cells={self.grid_map_query_max_cells})\n"
            f"  live compact global map: {self.compact_global_map_path}\n"
            f"  map persistence workers: {self.map_persistence_workers}\n"
            f"  live dense complete map: "
            f"{self.dense_global_grid_map_path if self.save_dense_global_grid_map else 'disabled'}\n"
            f"  persisted global-map revision: "
            f"{self.get_parameter('global_map_revision_topic').value}\n"
            f"  map publication: "
            f"{map_publication_description}\n"
            f"  optional outputs: layer_images="
            f"{self.publish_layer_images}, pointcloud="
            f"{self.publish_map_pointcloud}, path_stride="
            f"{self.path_publish_stride}\n"
            f"  map offset: {self.map_translation_offset.tolist()}\n"
            f"  map geometry source: "
            f"{map_geometry_description}"
            f"\n  visual map admission: {map_admission_description}"
            + (
                f"\n  full-tile batch protocol: "
                f"{self.get_parameter('incremental_map_batch_topic').value}, "
                f"session={self.map_transfer_session.session_id}, "
                f"replay={self.get_parameter('incremental_map_replay_service').value}, "
                f"delta_period={self.incremental_publish_period_sec:.3f}s"
                if (
                    self.publish_incremental_map
                    and self.incremental_publish_mode == "full_tiles"
                )
                else ""
            )
            + (
                f"\n  lossless observation archive: "
                f"{self.map_observations.archive_path} "
                f"(RAM cache={self.map_observation_cache_size})"
                if self.map_observation_archive_enabled
                else "\n  observation archive: disabled (unbounded RAM)"
            )
            + (
                f"\n  LiDAR topic: "
                f"{self.get_parameter('lidar_topic').value}, "
                f"max time offset="
                f"{self.lidar_max_time_offset_sec:.3f}s"
                f", mode={self.lidar_mapping_mode}"
                f", temporal={temporal_mode} ({temporal_description})"
                f", map alignment="
                f"{map_alignment_description}"
                if self.enable_lidar_mapping
                else ""
            )
            + (
                f"\n  LIO Odometry: "
                f"{self.get_parameter('lio_odometry_topic').value}, "
                f"frames={self.get_parameter('lio_expected_parent_frame').value}"
                f"->{self.get_parameter('lio_expected_child_frame').value}, "
                f"max time offset={self.lio_max_time_offset_sec:.3f}s"
                if self.localization_configuration.requires_external_odometry
                else ""
            )
            + (
                f"\n  global loop layer: "
                f"provider={self.global_loop_candidate_source}, "
                "candidate->bidirectional LiDAR ICP->robust iSAM2, "
                f"MPRF={self.mprf_client_configuration.endpoint}"
                if self.enable_global_loop_closure
                else "\n  global loop layer: disabled"
            )
        )
        if self.enable_lidar_mapping and temporal_mode == "none":
            self.get_logger().warning(
                "LiDAR mapping uses lidar_temporal_mode='none': this keeps "
                "the original behaviour but does not correct scan motion or "
                "cloud/image time offset. Accuracy acceptance should use "
                "'upstream_deskewed' or 'per_point' with external LIO poses."
            )

    def _make_empty_grid(self) -> TiledSemanticMapManager:
        """Create an empty, unbounded tiled map with fixed layer semantics."""
        return TiledSemanticMapManager(**self._grid_kwargs)

    def _create_slam(self):
        calibration_path = Path(
            str(self.get_parameter("calib_path").value)
        ).expanduser()
        if not calibration_path.exists():
            raise FileNotFoundError(
                f"Calibration file/directory missing: {calibration_path}"
            )

        calibration = load_calibration(calibration_path)
        if not self.visual_frontend_enabled:
            self.get_logger().info(
                "RoMa visual frontend is not loaded: direct LIO provides "
                "the pose, direct PointCloud2 provides map geometry, and "
                "stereo fallback is disabled."
            )
            return calibration, None

        algorithm_root = Path(
            str(self.get_parameter("algorithm_root").value)
        ).expanduser()
        module_name = str(self.get_parameter("algorithm_module").value)
        class_name = str(self.get_parameter("algorithm_class").value)

        if not algorithm_root.is_dir():
            raise FileNotFoundError(f"Algorithm root missing: {algorithm_root}")

        if str(algorithm_root) not in sys.path:
            sys.path.insert(0, str(algorithm_root))

        module = importlib.import_module(module_name)
        slam_class = getattr(module, class_name)

        self.get_logger().info(
            f"Loaded algorithm: {module.__file__}::{class_name}"
        )

        requested_device = str(self.get_parameter("device").value)
        torch = importlib.import_module("torch")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warning(
                "CUDA requested but unavailable; falling back to CPU."
            )
            device = "cpu"
        else:
            device = requested_device

        self.get_logger().info(
            f"Initialising RoMa SLAM on {device}; this can take a while."
        )
        start = time.perf_counter()

        slam = slam_class(
            calib=calibration,
            device=device,
            coarse_res=int(self.get_parameter("coarse_res").value),
            upsample_res=int(self.get_parameter("upsample_res").value),
            use_custom_corr=bool(
                self.get_parameter("use_custom_corr").value
            ),
            stereo_upsample_refinement=bool(
                self.get_parameter("stereo_upsample_refinement").value
            ),
            temporal_upsample_refinement=bool(
                self.get_parameter("temporal_upsample_refinement").value
            ),
            symmetric_matching=bool(
                self.get_parameter("symmetric_matching").value
            ),
            enable_feature_reuse=bool(
                self.get_parameter("enable_feature_reuse").value
            ),
            batch_coarse_pairs=bool(
                self.get_parameter("batch_coarse_pairs").value
            ),
            enable_klt_temporal_tracking=bool(
                self.get_parameter(
                    "enable_klt_temporal_tracking"
                ).value
            ),
            klt_max_points=int(
                self.get_parameter("klt_max_points").value
            ),
            klt_min_tracks=int(
                self.get_parameter("klt_min_tracks").value
            ),
            klt_min_track_ratio=float(
                self.get_parameter("klt_min_track_ratio").value
            ),
            klt_fb_max_error=float(
                self.get_parameter("klt_fb_max_error").value
            ),
            klt_image_scale=float(
                self.get_parameter("klt_image_scale").value
            ),
            klt_min_pnp_inlier_ratio=float(
                self.get_parameter("klt_min_pnp_inlier_ratio").value
            ),
            stereo_matches=int(
                self.get_parameter("stereo_matches").value
            ),
            temporal_matches=int(
                self.get_parameter("temporal_matches").value
            ),
            match_resize=None,
            min_depth=float(
                calibration.get("depth", {}).get("min", 0.1)
            ),
            max_depth=float(
                calibration.get("depth", {}).get("max", 100.0)
            ),
            epipolar_threshold=float(
                calibration.get("stereo", {}).get(
                    "epipolar_threshold_px", 2.0
                )
            ),
            keyframe_interval=int(
                self.get_parameter("keyframe_interval").value
            ),
            keyframe_min_inliers=int(
                calibration.get("tracking", {}).get(
                    "keyframe_min_inliers", 50
                )
            ),
            keyframe_min_gap=int(
                calibration.get("tracking", {}).get(
                    "keyframe_min_gap", 1
                )
            ),
            keyframe_interval_from_last=bool(
                calibration.get("tracking", {}).get(
                    "keyframe_interval_from_last", False
                )
            ),
            enable_backend=bool(
                self.get_parameter("enable_backend").value
            ),
            enable_loop_closing=bool(
                self.get_parameter("enable_loop_closing").value
            ),
            imu_helper=self.imu_buffer if self.enable_imu else None,
        )

        self.get_logger().info(
            f"RoMa SLAM initialised in {time.perf_counter() - start:.2f}s."
        )
        return calibration, slam

    def _create_localization_coordinator(self) -> LocalizationCoordinator:
        adapter = None
        if self.localization_configuration.requires_external_odometry:
            if not bool(
                self.get_parameter(
                    "lio_child_to_lidar_confirmed"
                ).value
            ):
                raise ValueError(
                    "External LIO mode requires an explicitly verified "
                    "lio_child_to_lidar_transform. Set "
                    "lio_child_to_lidar_confirmed:=true only after checking "
                    "the FAST-LIO2 IMU/body-to-LiDAR extrinsic."
                )
            configured_extrinsics = str(
                self.get_parameter("lio_extrinsics_path").value
            ).strip()
            if not configured_extrinsics:
                configured_extrinsics = str(
                    self.get_parameter("lidar_extrinsics_path").value
                ).strip()
            transform_camera_lidar = load_rectified_lidar_to_camera_transform(
                self.calibration,
                extrinsics_path=(
                    Path(configured_extrinsics).expanduser()
                    if configured_extrinsics
                    else None
                ),
            )
            self.lio_transform_camera_lidar = transform_camera_lidar.copy()
            transform_lidar_child = row_major_transform(
                self.get_parameter(
                    "lio_child_to_lidar_transform"
                ).value,
                name="lio_child_to_lidar_transform",
            )
            transform_camera_child = (
                compose_lio_child_to_camera_transform(
                    transform_camera_lidar,
                    transform_lidar_child,
                )
            )
            adapter = ExternalOdometryAdapter(
                transform_camera_child=transform_camera_child,
                expected_parent_frame=str(
                    self.get_parameter("lio_expected_parent_frame").value
                ),
                expected_child_frame=str(
                    self.get_parameter("lio_expected_child_frame").value
                ),
                max_buffer_size=self.lio_buffer_size,
                max_buffer_age_sec=self.lio_buffer_age_sec,
            )
            self.external_odometry_adapter = adapter
            identity_child_transform = np.allclose(
                transform_lidar_child, np.eye(4), atol=1e-9
            )
            self.get_logger().info(
                "Verified LIO extrinsic chain loaded: "
                "T_camera_child=T_camera_lidar*T_lidar_child; "
                f"T_lidar_child_identity={identity_child_transform}."
            )

        fusion_options = {
            "lag_sec": float(
                self.get_parameter("fusion_lag_sec").value
            ),
            "solver": str(
                self.get_parameter("fusion_solver").value
            ),
            "qc_rotation": self.get_parameter(
                "fusion_gp_qc_rotation"
            ).value,
            "qc_translation": self.get_parameter(
                "fusion_gp_qc_translation"
            ).value,
            "vo_rotation_sigma_deg": self.get_parameter(
                "fusion_vo_rotation_sigma_deg"
            ).value,
            "vo_translation_sigma_m": self.get_parameter(
                "fusion_vo_translation_sigma_m"
            ).value,
            "lio_rotation_sigma_deg": self.get_parameter(
                "fusion_lio_rotation_sigma_deg"
            ).value,
            "lio_translation_sigma_m": self.get_parameter(
                "fusion_lio_translation_sigma_m"
            ).value,
            "robust_huber_k": float(
                self.get_parameter("fusion_robust_huber_k").value
            ),
        }
        coordinator = LocalizationCoordinator(
            self.localization_configuration,
            external_adapter=adapter,
            external_max_time_offset_sec=self.lio_max_time_offset_sec,
            fusion_options=fusion_options,
        )
        gp_prior_description = (
            self.localization_configuration.enable_gp_prior
            if coordinator.uses_factor_graph
            else "inactive"
        )
        self.get_logger().info(
            "Localisation configured: "
            f"mode={self.localization_mode}, "
            f"fusion_backend="
            f"{self.localization_configuration.fusion_backend}, "
            f"gp_prior="
            f"{gp_prior_description}"
        )
        return coordinator

    def _open_trajectory_files(self) -> None:
        self.tum_cv_path = (
            self.internal_output_dir / "trajectory_opencv.tum"
        )
        self.tum_map_path = self.output_dir / "trajectory_map.tum"
        self.tum_cv_file = self.tum_cv_path.open(
            "w", encoding="utf-8", buffering=1
        )
        self.tum_map_file = self.tum_map_path.open(
            "w", encoding="utf-8", buffering=1
        )
        self._trajectory_header = "# timestamp tx ty tz qx qy qz qw\n"
        self.tum_cv_file.write(self._trajectory_header)
        self.tum_map_file.write(self._trajectory_header)

    def _on_imu(self, message: Imu) -> None:
        if self.enable_imu:
            self.imu_buffer.add_message(message)

    def _on_lio_odometry(self, message: Odometry) -> None:
        if self.external_odometry_adapter is None:
            return
        timestamp = _stamp_seconds(message.header.stamp)
        if timestamp <= 0.0 or not np.isfinite(timestamp):
            self.lio_rejected_count += 1
            return

        pose = message.pose.pose
        try:
            self.external_odometry_adapter.add_components(
                timestamp=timestamp,
                parent_frame=message.header.frame_id,
                child_frame=message.child_frame_id,
                position_xyz=[
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ],
                quaternion_xyzw=[
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                # FAST-LIO covariance is expressed for its own child pose.
                # Do not republish it as camera/map covariance until the full
                # SE(3) adjoint (including the lever arm) is applied.
                covariance=None,
            )
        except (FrameMismatchError, ValueError) as exception:
            self.lio_rejected_count += 1
            if (
                self.lio_rejected_count == 1
                or self.lio_rejected_count % 20 == 0
            ):
                self.get_logger().warning(
                    f"Rejected external LIO Odometry: {exception}; "
                    f"total_rejected={self.lio_rejected_count}"
                )
            return

        with self._lio_condition:
            self._lio_condition.notify_all()

    def _wait_for_external_buffer_until(self, timestamp: float) -> None:
        if self.external_odometry_adapter is None:
            return
        target = float(timestamp)
        deadline = time.monotonic() + self.lio_wait_timeout_sec
        with self._lio_condition:
            while True:
                time_range = self.external_odometry_adapter.buffer.time_range
                if time_range is not None and time_range[1] >= target:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._lio_condition.wait(timeout=remaining)

    def _wait_for_external_pose(self, timestamp: float):
        if self.external_odometry_adapter is None:
            return None
        target = float(timestamp)
        self._wait_for_external_buffer_until(target)

        query = self.localization_coordinator.query_external(target)
        if query is None:
            time_range = self.external_odometry_adapter.buffer.time_range
            self.last_lio_diagnostic = (
                "no time-valid pose; buffer="
                f"{time_range}, target={target:.9f}"
            )
            return None
        self.last_lio_diagnostic = (
            f"{query.method}, offset={query.nearest_offset_sec:+.3f}s, "
            f"span={query.bracket_span_sec:.3f}s"
        )
        return query

    def _query_bracketed_external_camera_pose(
        self, timestamp: float
    ) -> Optional[np.ndarray]:
        """Query an unaligned, interpolated/exact LIO pose for deskew.

        Nearest-sample fallback is deliberately disabled: accepting one here
        would turn a missing motion interval into a plausible-looking but
        geometrically uncorrected point cloud.
        """

        if self.external_odometry_adapter is None:
            return None
        query = self.external_odometry_adapter.query(
            float(timestamp),
            max_time_offset_sec=self.lio_max_time_offset_sec,
            align_to_first_query=False,
            allow_nearest=False,
        )
        return None if query is None else query.pose

    def _on_terrain_pointcloud(
        self, sensor_name: str, message: PointCloud2
    ) -> None:
        mapper = self.terrain_pointcloud_mapper
        if mapper is not None:
            mapper.record_message(sensor_name, message)

    def _on_lidar(self, message: PointCloud2) -> None:
        if not self.enable_lidar_mapping:
            return
        frame_id = str(message.header.frame_id).strip().lstrip("/")
        expected = self.lidar_expected_frame_id.lstrip("/")
        if expected and frame_id != expected:
            self.lidar_skipped_count += 1
            if self.lidar_skipped_count == 1 or self.lidar_skipped_count % 20 == 0:
                self.get_logger().warning(
                    "Skipping LiDAR cloud with unexpected frame_id: "
                    f"expected={self.lidar_expected_frame_id!r}, "
                    f"received={message.header.frame_id!r}. Set "
                    "lidar_expected_frame_id to the calibrated sensor frame."
                )
            return

        timestamp_ns = stamp_nanoseconds(message)
        if timestamp_ns <= 0:
            self.lidar_skipped_count += 1
            return
        self.lidar_message_buffer.add(timestamp_ns, message)

    def _prepare_lidar_keyframe_data(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        header: Header,
        lidar_message: Optional[PointCloud2] = None,
    ):
        image_timestamp_ns = (
            int(header.stamp.sec) * 1_000_000_000
            + int(header.stamp.nanosec)
        )
        if lidar_message is None:
            message, offset_sec = self.lidar_message_buffer.nearest(
                image_timestamp_ns
            )
            if message is None:
                raise RuntimeError(
                    "no PointCloud2 message within "
                    f"{self.lidar_max_time_offset_sec:.3f}s of keyframe "
                    f"{keyframe_id}; nearest offset={offset_sec:+.3f}s, "
                    f"buffer_ns={self.lidar_message_buffer.time_range_ns}"
                )
        else:
            message = lidar_message
            offset_sec = (
                stamp_nanoseconds(message) - image_timestamp_ns
            ) * 1e-9
            if abs(offset_sec) > self.lidar_max_time_offset_sec:
                raise LidarObservationBuildError(
                    "retained PointCloud2 moved outside the configured "
                    f"time gate for keyframe {keyframe_id}: "
                    f"offset={offset_sec:+.3f}s, "
                    f"gate={self.lidar_max_time_offset_sec:.3f}s",
                    lidar_message=message,
                )

        try:
            cloud_timestamp = stamp_nanoseconds(message) * 1e-9
            reference_timestamp = image_timestamp_ns * 1e-9
            point_time_offsets = None
            if self.lidar_temporal_configuration.requires_point_times:
                points_lidar, point_time_offsets = (
                    pointcloud2_xyz_time_arrays(
                        message,
                        time_field=self.lidar_point_time_field,
                        time_unit=self.lidar_point_time_unit,
                    )
                )
            else:
                # In upstream_deskewed mode, point-time-like fields are
                # intentionally ignored so an already corrected FAST-LIO2
                # cloud can never be deskewed a second time.
                points_lidar = pointcloud2_xyz_array(message)

            required_time_range = lidar_required_pose_time_range(
                cloud_timestamp=cloud_timestamp,
                reference_timestamp=reference_timestamp,
                point_count=len(points_lidar),
                configuration=self.lidar_temporal_configuration,
                point_time_offsets_sec=point_time_offsets,
            )
            if required_time_range is not None:
                self._wait_for_external_buffer_until(required_time_range[1])
            alignment = align_lidar_points_to_reference(
                points_lidar,
                cloud_timestamp=cloud_timestamp,
                reference_timestamp=reference_timestamp,
                transform_camera_sensor=(
                    self.lidar_processor.transform_camera_lidar
                ),
                configuration=self.lidar_temporal_configuration,
                pose_query=self._query_bracketed_external_camera_pose,
                point_time_offsets_sec=point_time_offsets,
            )
            observation = self.lidar_processor.make_observation(
                alignment.points_sensor,
                semantic_image,
                sensor_origin_lidar=alignment.sensor_origin_sensor,
            )
            selected = int(observation.statistics["selected_points"])
            if selected < self.lidar_min_points_per_keyframe:
                raise RuntimeError(
                    f"PointCloud2 keyframe {keyframe_id} retained only "
                    f"{selected} points; minimum="
                    f"{self.lidar_min_points_per_keyframe}"
                )
            statistics = dict(observation.statistics)
            statistics.update(alignment.statistics)
            statistics["time_offset_sec"] = float(offset_sec)
            statistics["keyframe_id"] = int(keyframe_id)
            statistics["cloud_timestamp_sec"] = float(cloud_timestamp)
            statistics["reference_timestamp_sec"] = float(
                reference_timestamp
            )
        except LidarObservationBuildError:
            raise
        except Exception as exception:
            raise LidarObservationBuildError(
                str(exception), lidar_message=message
            ) from exception
        return observation, statistics, message, alignment.points_sensor

    def _make_lidar_keyframe_observation(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        header: Header,
        visual_image_bgr: Optional[np.ndarray] = None,
        lidar_message: Optional[PointCloud2] = None,
    ) -> SemanticKeyFrameObservation:
        observation, statistics, message, points_sensor = (
            self._prepare_lidar_keyframe_data(
                keyframe_id=keyframe_id,
                semantic_image=semantic_image,
                header=header,
                lidar_message=lidar_message,
            )
        )
        self._register_global_lidar_keyframe(
            keyframe_id=keyframe_id,
            header=header,
            points_sensor=points_sensor,
            visual_image_bgr=visual_image_bgr,
        )
        selected = int(observation.statistics["selected_points"])
        self.get_logger().info(
            f"Direct PointCloud2 map observation for keyframe {keyframe_id}: "
            f"selected={selected}, "
            f"semantic_projected="
            f"{statistics['semantic_projected_points']}, "
            f"time_offset={statistics['time_offset_sec']:+.3f}s, "
            f"temporal_mode={self.lidar_temporal_configuration.mode}, "
            f"correction_p95="
            f"{statistics['correction_p95_m']:.3f}m."
        )
        return SemanticKeyFrameObservation(
            keyframe_id=int(keyframe_id),
            sensor_timestamp=(
                statistics["reference_timestamp_sec"]
                if self.lidar_temporal_configuration.requires_external_pose
                else stamp_nanoseconds(message) * 1e-9
            ),
            camera_points=observation.camera_points,
            semantic_labels=observation.semantic_labels,
            sensor_origin_camera=observation.sensor_origin_camera,
            require_positive_depth=False,
            source="lidar",
        )

    def _discovered_incremental_subscription_gids(self) -> frozenset:
        """Return stable DDS identities for discovered payload consumers.

        Subscription count alone cannot detect a consumer replacement when
        one endpoint disconnects in the same polling interval that another
        endpoint connects.  Endpoint GIDs close that gap.  The count remains
        the compatibility fallback for older rclpy implementations.
        """
        graph_query = getattr(
            self,
            "get_subscriptions_info_by_topic",
            None,
        )
        topic_name = getattr(self.incremental_map_pub, "topic_name", None)
        if not callable(graph_query) or not topic_name:
            return frozenset()
        try:
            endpoints = graph_query(topic_name)
        except (RuntimeError, TypeError, ValueError):
            return frozenset()
        return frozenset(
            tuple(int(component) for component in endpoint.endpoint_gid)
            for endpoint in endpoints
            if endpoint.endpoint_gid
        )

    def _on_incremental_replay_timer(self) -> None:
        """Stream one frozen authoritative tile per timer callback."""
        if (
            not self.publish_incremental_map
            or self.incremental_publish_mode != "full_tiles"
            or self.incremental_map_pub is None
        ):
            return
        subscription_count = int(
            self.incremental_map_pub.get_subscription_count()
        )
        subscription_gids = self._discovered_incremental_subscription_gids()
        endpoint_replaced = bool(
            self._incremental_subscription_gids
            and subscription_gids - self._incremental_subscription_gids
        )
        if (
            subscription_count > self._incremental_subscription_count
            or endpoint_replaced
        ):
            self._incremental_replay_requested = True
        if subscription_count == 0:
            self._incremental_replay_state = None
            self._incremental_replay_requested = False
            self._incremental_delta_requested = False
            self._incremental_next_delta_monotonic = 0.0
        self._incremental_subscription_count = subscription_count
        self._incremental_subscription_gids = (
            subscription_gids if subscription_count > 0 else frozenset()
        )

        if subscription_count == 0 or self.last_header is None:
            return

        if (
            self._incremental_replay_state is None
            and self._incremental_replay_requested
        ):
            try:
                self._start_incremental_replay()
                self._incremental_replay_requested = False
            except Exception as exception:
                self.get_logger().error(
                    f"Could not start complete-map replay: {exception}"
                )
                return

        if (
            self._incremental_replay_state is None
            and self._incremental_delta_requested
            and time.monotonic() >= self._incremental_next_delta_monotonic
        ):
            try:
                self._start_incremental_delta(self.last_header)
                self._incremental_delta_requested = False
                self._incremental_next_delta_monotonic = (
                    time.monotonic() + self.incremental_publish_period_sec
                )
            except Exception as exception:
                self.get_logger().error(
                    f"Could not start incremental-map delta: {exception}"
                )
                return

        state = self._incremental_replay_state
        if state is None:
            return
        if not state.pending_keys:
            self._finish_incremental_replay()
            return

        key = state.pending_keys.popleft()
        try:
            update = state.snapshot.tile_update(key, full_tile=True)
            update_id = self._publish_incremental_window_snapshot(
                map_header=state.map_header,
                window=update.window,
            )
            state.payload_update_ids.append(update_id)
            if not state.pending_keys:
                self._finish_incremental_replay()
        except Exception as exception:
            # No COMPLETE descriptor is emitted for a partial replay. A
            # receiver can discard those payloads, and the next timer retries
            # from a new frozen snapshot/batch ID.
            self.get_logger().error(
                "Incremental-map tile batch aborted before commit: "
                f"{exception}"
            )
            self._incremental_replay_state = None
            if state.batch.full_snapshot:
                self._incremental_replay_requested = True
            else:
                self._incremental_delta_requested = True

    def _start_incremental_replay(self) -> None:
        if self.last_header is None:
            raise RuntimeError("cannot replay before the first map header")
        with self._map_lock:
            snapshot = self.grid.clone_for_update()
            map_revision = int(self.map_revision)
            keys = snapshot.replay_tile_keys()
            tiles = tuple(
                self._tile_transfer_plan(
                    snapshot.tile_update_metadata(key)
                )
                for key in keys
            )
        map_header = Header()
        map_header.stamp = self.last_header.stamp
        map_header.frame_id = self.map_frame
        batch = self.map_transfer_session.begin_batch(
            map_revision=map_revision,
            full_snapshot=True,
            tiles=tiles,
        )
        self._publish_map_batch_begin(map_header, batch)
        self._incremental_replay_state = _TileBatchReplayState(
            snapshot=snapshot,
            map_header=map_header,
            batch=batch,
            pending_keys=deque(keys),
            payload_update_ids=[],
            published_revisions={
                key: int(snapshot.tile_revisions.get(key, 0))
                for key in keys
            },
        )
        self.get_logger().info(
            "Incremental-map consumer requested a frozen full replay: "
            f"session={batch.session_id}, batch={batch.batch_id}, "
            f"map_revision={batch.map_revision}, tiles={len(keys)}."
        )
        if not keys:
            self._finish_incremental_replay()

    def _request_incremental_delta(self, map_header: Header) -> None:
        """Coalesce dirty full tiles for the bounded-rate transfer timer.

        Map fusion remains per-frame.  The transfer channel publishes at most
        once per configured period, so fast revisions of the same tile merge
        naturally into one latest-revision full-tile batch.
        """
        del map_header
        if self.incremental_map_pub is None:
            return
        if self.incremental_map_pub.get_subscription_count() == 0:
            return
        self._incremental_delta_requested = True

    def _start_incremental_delta(self, map_header: Header) -> bool:
        """Freeze one dirty-tile batch; the timer streams its payloads."""
        with self._map_lock:
            snapshot = self.grid.clone_for_update()
            map_revision = int(self.map_revision)
            keys = snapshot.pending_dirty_tile_keys()
            tiles = tuple(
                self._tile_transfer_plan(
                    snapshot.tile_update_metadata(key)
                )
                for key in keys
            )
            revisions = {
                key: int(snapshot.tile_revisions.get(key, 0))
                for key in keys
            }
        if not keys:
            return False

        batch_header = Header()
        batch_header.stamp = map_header.stamp
        batch_header.frame_id = self.map_frame
        batch = self.map_transfer_session.begin_batch(
            map_revision=map_revision,
            full_snapshot=False,
            tiles=tiles,
        )
        self._publish_map_batch_begin(batch_header, batch)
        self._incremental_replay_state = _TileBatchReplayState(
            snapshot=snapshot,
            map_header=batch_header,
            batch=batch,
            pending_keys=deque(keys),
            payload_update_ids=[],
            published_revisions=revisions,
        )
        return True

    def _finish_incremental_replay(self) -> None:
        state = self._incremental_replay_state
        if state is None:
            return
        completed = self.map_transfer_session.complete_batch(
            state.batch,
            state.payload_update_ids,
        )
        self._publish_map_batch_complete(state.map_header, completed)
        self.map_transfer_session.mark_complete(completed)
        with self._map_lock:
            self.grid.mark_tiles_published_at_revisions(
                state.published_revisions
            )
        self._incremental_replay_state = None
        batch_kind = (
            "full-map replay"
            if completed.plan.full_snapshot
            else "incremental delta"
        )
        self.get_logger().info(
            f"Completed frozen {batch_kind}: "
            f"session={completed.plan.session_id}, "
            f"batch={completed.plan.batch_id}, "
            f"tiles={len(completed.plan.tiles)}."
        )

    def _on_incremental_replay_service(self, request, response):
        del request
        if self.last_header is None:
            response.success = False
            response.message = "No map header is available yet"
            return response
        if (
            self.incremental_map_pub is None
            or self.incremental_map_pub.get_subscription_count() == 0
        ):
            response.success = False
            response.message = (
                "No IncrementalSemanticMap subscriber is connected"
            )
            return response
        self._incremental_replay_requested = True
        response.success = True
        response.message = (
            "Complete full-tile replay queued"
            if self._incremental_replay_state is None
            else "Replay already active; one fresh replay queued after it"
        )
        return response

    def _global_pose_for_output_index(
        self, output_index: int, local_pose: np.ndarray
    ) -> np.ndarray:
        if self.global_loop_closure is None:
            return np.asarray(local_pose, dtype=np.float64).copy()
        return self.global_loop_closure.correct_pose(
            node_id=int(output_index),
            local_pose=local_pose,
        )

    def _pose_change_requires_map_rebuild(
        self, previous: np.ndarray, current: np.ndarray
    ) -> bool:
        translation_m, rotation_rad = pose_change_norm(previous, current)
        return bool(
            translation_m >= self.fusion_map_rebuild_translation_m
            or rotation_rad >= self.fusion_map_rebuild_rotation_rad
        )

    def _refresh_global_corrected_state(self) -> bool:
        """Recompute every derived output from immutable local estimates."""

        if self.global_loop_closure is None:
            return False
        corrected_history = self.global_loop_closure.correct_poses(
            self.localization_odom_pose_history
        )
        path_changed = False
        for output_index, corrected in corrected_history.items():
            previous = self.localization_pose_history.get(output_index)
            if previous is not None and self._pose_change_requires_map_rebuild(
                previous, corrected
            ):
                path_changed = True
            self.localization_pose_history[output_index] = np.asarray(
                corrected, dtype=np.float64
            ).copy()

        map_changed = False
        for keyframe_id, local_pose in tuple(
            self.localization_keyframe_odom_poses.items()
        ):
            output_index = self.localization_keyframe_output_indices.get(
                keyframe_id
            )
            if output_index is None:
                continue
            corrected = self._global_pose_for_output_index(
                output_index, local_pose
            )
            previous = self.localization_keyframe_poses.get(keyframe_id)
            if previous is not None and self._pose_change_requires_map_rebuild(
                previous, corrected
            ):
                map_changed = True
            self.localization_keyframe_poses[keyframe_id] = corrected

        if map_changed:
            self.localization_geometry_revision += 1
        if path_changed or map_changed:
            self._global_path_rebuild_pending = True
        return map_changed

    def _apply_global_loop_update(self, update) -> bool:
        if not bool(getattr(update, "geometry_corrected", False)):
            return False
        return self._refresh_global_corrected_state()

    def _register_global_odometry_node(
        self,
        *,
        node_id: int,
        timestamp: float,
        local_pose: np.ndarray,
    ) -> bool:
        if self.global_loop_closure is None:
            return False
        if self.global_loop_closure.graph.has_node(node_id):
            return False
        try:
            self.global_loop_closure.register_odometry_keyframe(
                node_id=int(node_id),
                timestamp=float(timestamp),
                local_camera_pose=local_pose,
            )
            return True
        except Exception as exception:
            self._global_loop_error_count += 1
            self.get_logger().error(
                "Global graph keyframe registration failed without stopping "
                f"local mapping: node={int(node_id)}, error={exception}"
            )
            return False

    def _poll_mprf_loop_results(self) -> bool:
        if self.mprf_client is None or self.global_loop_closure is None:
            return False
        geometry_corrected = False
        for result in self.mprf_client.poll():
            if result.error is not None:
                self._global_loop_error_count += 1
                if (
                    self._global_loop_error_count == 1
                    or self._global_loop_error_count % 20 == 0
                ):
                    self.get_logger().warning(
                        "MPRF service result was ignored; local odometry and "
                        f"mapping continue unchanged: {result.error}"
                    )
                continue
            if not result.candidates:
                continue
            try:
                update = self.global_loop_closure.submit_candidates(
                    result.candidates
                )
                geometry_corrected = (
                    self._apply_global_loop_update(update)
                    or geometry_corrected
                )
                self.get_logger().info(
                    "MPRF shortlist LiDAR verification: "
                    "references="
                    f"{[value.reference_node for value in result.candidates]}, "
                    f"query={result.query_node}, "
                    f"status={update.detection.status}, "
                    f"geometrically_verified="
                    f"{update.detection.geometrically_verified}, "
                    f"accepted_factors={update.accepted_loop_count}."
                )
            except Exception as exception:
                self._global_loop_error_count += 1
                self.get_logger().warning(
                    "MPRF candidate could not enter the safe LiDAR gate; "
                    f"local output is unchanged: {exception}"
                )
        return geometry_corrected

    def _on_mprf_result_timer(self) -> None:
        """Apply completed MPRF work even when no new sensor frame arrives."""

        try:
            map_geometry_changed = self._poll_mprf_loop_results()
            header = self.last_header or Header()
            if map_geometry_changed:
                self._synchronize_map_state(
                    header=header,
                    publish=rclpy.ok(context=self.context),
                )
            if self._global_path_rebuild_pending and self.last_header is not None:
                self._rebuild_path_and_trajectory(
                    self.last_header,
                    publish=rclpy.ok(context=self.context),
                )
                self._global_path_rebuild_pending = False
        except Exception as exception:
            self._global_loop_error_count += 1
            if (
                self._global_loop_error_count == 1
                or self._global_loop_error_count % 20 == 0
            ):
                self.get_logger().warning(
                    "Deferred MPRF result application failed; local output "
                    f"remains active and the timer will retry: {exception}"
                )

    def _register_global_lidar_keyframe(
        self,
        *,
        keyframe_id: int,
        header: Header,
        points_sensor: np.ndarray,
        visual_image_bgr: Optional[np.ndarray],
    ) -> None:
        if self.global_loop_closure is None or self.lidar_processor is None:
            return
        output_index = self.localization_keyframe_output_indices.get(
            int(keyframe_id)
        )
        local_pose = self.localization_keyframe_odom_poses.get(
            int(keyframe_id)
        )
        if output_index is None or local_pose is None:
            self.get_logger().warning(
                "Global loop data skipped because its ordered local graph "
                f"node is unavailable: keyframe={int(keyframe_id)}"
            )
            return
        node_id = int(output_index)
        timestamp = _stamp_seconds(header.stamp)
        if node_id not in self._global_loop_registered_nodes:
            if timestamp <= self._last_global_lidar_timestamp:
                # A delayed map-observation retry must never be inserted into
                # the chronological submap history after newer scans.
                self.get_logger().warning(
                    "Delayed LiDAR loop keyframe skipped to preserve ordered "
                    f"submaps: node={node_id}, time={timestamp:.9f}."
                )
                self._global_loop_registered_nodes.add(node_id)
                return
            try:
                update = self.global_loop_closure.register_lidar_keyframe(
                    node_id=node_id,
                    keyframe_id=int(keyframe_id),
                    timestamp=timestamp,
                    local_camera_pose=local_pose,
                    points_sensor=points_sensor,
                    transform_camera_sensor=(
                        self.lidar_processor.transform_camera_lidar
                    ),
                )
                self._global_loop_registered_nodes.add(node_id)
                self._last_global_lidar_timestamp = timestamp
                self._apply_global_loop_update(update)
                if update.detection.candidates_considered:
                    self.get_logger().info(
                        "LiDAR loop search: "
                        f"node={node_id}, status={update.detection.status}, "
                        f"candidates={update.detection.candidates_considered}, "
                        f"verified={update.detection.geometrically_verified}, "
                        f"accepted_factors={update.accepted_loop_count}."
                    )
            except Exception as exception:
                self._global_loop_error_count += 1
                self.get_logger().warning(
                    "LiDAR loop keyframe was ignored without affecting the "
                    f"map transaction: node={node_id}, error={exception}"
                )

        if (
            self.mprf_client is not None
            and self.mprf_client.configuration.enabled
            and visual_image_bgr is not None
            and node_id not in self._mprf_submitted_nodes
            and node_id in self._global_loop_registered_nodes
        ):
            try:
                submitted = self.mprf_client.submit(
                    node_id=node_id,
                    timestamp_ns=_stamp_nanoseconds_value(header.stamp),
                    image_bgr=visual_image_bgr,
                    points_sensor=points_sensor,
                    camera_matrix=self.lidar_processor.camera_matrix,
                    transform_camera_sensor=(
                        self.lidar_processor.transform_camera_lidar
                    ),
                    calibration_width=(
                        self.lidar_processor.calibration_width
                    ),
                    calibration_height=(
                        self.lidar_processor.calibration_height
                    ),
                )
                if submitted:
                    self._mprf_submitted_nodes.add(node_id)
            except Exception as exception:
                self._global_loop_error_count += 1
                self.get_logger().warning(
                    "MPRF keyframe submission failed; local mapping remains "
                    f"active: node={node_id}, error={exception}"
                )

    def _record_external_localization_update(
        self,
        update,
        tracking_status,
    ) -> bool:
        """Record fused/direct-LIO poses and detect map-relevant correction."""
        geometry_corrected = False
        for output_index, pose in update.active_poses.items():
            output_index = int(output_index)
            local_value = np.asarray(pose, dtype=np.float64).copy()
            self.localization_odom_pose_history[output_index] = local_value
            value = self._global_pose_for_output_index(
                output_index, local_value
            )
            self.localization_pose_history[output_index] = value
            for keyframe_id, registered_index in tuple(
                self.localization_keyframe_output_indices.items()
            ):
                if int(registered_index) != output_index:
                    continue
                self.localization_keyframe_odom_poses[keyframe_id] = (
                    local_value.copy()
                )
                previous = self.localization_keyframe_poses.get(keyframe_id)
                if previous is None:
                    self.localization_keyframe_poses[keyframe_id] = value.copy()
                    continue
                if self._pose_change_requires_map_rebuild(previous, value):
                    geometry_corrected = True
                self.localization_keyframe_poses[keyframe_id] = value.copy()

        if bool(getattr(tracking_status, "success", False)) and bool(
            getattr(tracking_status, "keyframe_created", False)
        ):
            keyframe_id = getattr(tracking_status, "keyframe_id", None)
            if keyframe_id is None:
                raise RuntimeError(
                    "SLAM reported keyframe_created without keyframe_id"
                )
            keyframe_id = int(keyframe_id)
            output_index = int(update.measurement_index)
            self.localization_keyframe_output_indices[keyframe_id] = (
                output_index
            )
            local_value = np.asarray(
                update.estimate.pose, dtype=np.float64
            ).copy()
            self.localization_keyframe_odom_poses[keyframe_id] = local_value
            self.localization_keyframe_poses[keyframe_id] = (
                self._global_pose_for_output_index(output_index, local_value)
            )
            self._register_global_odometry_node(
                node_id=output_index,
                timestamp=float(update.estimate.timestamp),
                local_pose=local_value,
            )

        if geometry_corrected:
            self.localization_geometry_revision += 1
        return geometry_corrected

    def _update_map_without_vo_tracking(
        self,
        *,
        localization_update,
        semantic_image: np.ndarray,
        header: Header,
        visual_image_bgr: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Add periodic direct-cloud evidence without a visual keyframe.

        Assisted stereo needs a valid visual keyframe and therefore cannot
        manufacture map evidence during visual tracking loss. Direct cloud
        mode uses the accepted LIO-pose count, so the first valid pose always
        seeds the map and rejected external poses do not consume its cadence.
        """
        keyframe_interval = max(
            1, int(self.get_parameter("keyframe_interval").value)
        )
        create_mapping_keyframe = (
            self.enable_lidar_mapping
            and self.lidar_mapping_mode == "direct"
            and (
                self.frame_count == 1
                or self.frame_count % keyframe_interval == 0
            )
        )
        if create_mapping_keyframe:
            # Negative IDs cannot collide with RoMa's non-negative keyframes.
            keyframe_id = -int(self.received_frame_count)
            output_index = int(localization_update.measurement_index)
            self.localization_keyframe_output_indices[keyframe_id] = (
                output_index
            )
            local_pose = np.asarray(
                localization_update.estimate.pose, dtype=np.float64
            ).copy()
            self.localization_keyframe_odom_poses[keyframe_id] = local_pose
            self.localization_keyframe_poses[keyframe_id] = (
                self._global_pose_for_output_index(output_index, local_pose)
            )
            self._register_global_odometry_node(
                node_id=output_index,
                timestamp=float(localization_update.estimate.timestamp),
                local_pose=local_pose,
            )
            try:
                observation = self._make_lidar_keyframe_observation(
                    keyframe_id=keyframe_id,
                    semantic_image=semantic_image,
                    header=header,
                    visual_image_bgr=visual_image_bgr,
                )
                self._stage_keyframe_observation(observation)
            except Exception as exception:
                self.lidar_skipped_count += 1
                self._record_missed_keyframe_observation(
                    keyframe_id=keyframe_id,
                    semantic_image=semantic_image,
                    header=header,
                    exception=exception,
                    visual_image_bgr=visual_image_bgr,
                )
        return self._synchronize_map_state(header=header, publish=True)

    def _on_synchronized_input(
        self,
        left_message: Image,
        right_message: Image,
        semantic_message: Optional[Image],
    ) -> None:
        callback_start = time.perf_counter()
        left_stamp_ns = _stamp_nanoseconds_value(left_message.header.stamp)
        right_stamp_ns = _stamp_nanoseconds_value(right_message.header.stamp)
        source_stamp_ns = self.left_input_filter.pop_source_stamp_ns(
            left_message
        )
        metric_stamp_ns = (
            left_stamp_ns if source_stamp_ns is None else source_stamp_ns
        )
        self.input_monitor.record_synchronized(
            metric_stamp_ns, callback_start
        )
        if not self.input_transport.accepts_stereo_skew(
            left_stamp_ns, right_stamp_ns
        ):
            self._finish_synchronized_input(
                left_message.header,
                outcome="stereo_skew_rejected",
                wall_time=time.perf_counter(),
            )
            rejected_count = (
                self.input_monitor.rejected_stereo_skew_count
            )
            if (
                rejected_count == 1
                or rejected_count % 50 == 0
            ):
                self.get_logger().warning(
                    "Rejected approximately synchronized input because "
                    "left/right acquisition skew would corrupt stereo "
                    f"geometry: skew="
                    f"{abs(left_stamp_ns - right_stamp_ns) * 1e-9:.6f}s, "
                    f"limit={self.input_transport.stereo_max_skew_sec:.6f}s, "
                    f"rejected={rejected_count}."
                )
            return
        if self.processing:
            self._finish_synchronized_input(
                left_message.header,
                outcome="reentrant_rejected",
                wall_time=time.perf_counter(),
            )
            self.get_logger().warning(
                "Previous frame is still processing; dropping synchronized input."
            )
            return

        self.processing = True
        self.input_monitor.record_processing_started()
        self.received_frame_count += 1
        start = callback_start
        visual_admission_observed = False
        processing_outcome = "completed"
        stage_timer = CallbackStageTimer(callback_start)
        pose_publish_latency_sec = None
        map_update_latency_sec = None
        localization_solution_start_monotonic_ns = int(
            round(callback_start * 1_000_000_000)
        )
        pose_publish_completion_monotonic_ns = None
        map_update_start_monotonic_ns = None
        map_publish_completion_monotonic_ns = None

        try:
            left = self.bridge.imgmsg_to_cv2(
                left_message, desired_encoding="bgr8"
            )
            right = self.bridge.imgmsg_to_cv2(
                right_message, desired_encoding="bgr8"
            )
            if semantic_message is None:
                semantic = np.zeros(left.shape[:2], dtype=np.uint8)
            else:
                semantic = self.bridge.imgmsg_to_cv2(
                    semantic_message, desired_encoding="mono8"
                )
                semantic = np.asarray(semantic, dtype=np.uint8)
            if semantic.shape[:2] != left.shape[:2]:
                raise ValueError(
                    "Semantic and left-image sizes differ: "
                    f"{semantic.shape[:2]} != {left.shape[:2]}"
                )

            left, right = self.stereo_rectifier.rectify(left, right)
            semantic = self.stereo_rectifier.rectify_left(
                semantic, interpolation=0
            )

            timestamp = _stamp_seconds(left_message.header.stamp)
            if self.enable_imu:
                self.imu_buffer.register_image_timestamp(timestamp)

            stage_timer.switch(
                "visual_tracking", time.perf_counter()
            )
            raw_vo_pose, tracking_status = process_visual_tracking(
                self.slam,
                left,
                right,
                visual_required=self.localization_mode == "vo",
                frame_index=self.received_frame_count - 1,
                geometry_revision=self.applied_geometry_revision,
                enabled=self.visual_frontend_enabled,
            )
            self._publish_feature_match_debug(
                left=left,
                right=right,
                header=left_message.header,
            )
            stage_timer.switch(
                "localization_and_pose", time.perf_counter()
            )
            vo_pose_is_valid = (
                raw_vo_pose.shape == (4, 4)
                and np.isfinite(raw_vo_pose).all()
            )
            vo_tracking_success = bool(
                tracking_status.success
            ) and vo_pose_is_valid
            if not vo_tracking_success:
                visual_tracking_attempted = bool(
                    getattr(tracking_status, "attempted", True)
                )
                if visual_tracking_attempted:
                    self.tracking_failure_count += 1
                if self.localization_mode == "vo":
                    self.map_admission_gate.record_tracking_failure()
                    visual_admission_observed = True
                    status_revision = int(
                        getattr(
                            tracking_status,
                            "geometry_revision",
                            self.applied_geometry_revision,
                        )
                    )
                    geometry_changed = bool(
                        getattr(
                            tracking_status,
                            "geometry_corrected",
                            False,
                        )
                    ) or status_revision != self.applied_geometry_revision
                    with self._map_lock:
                        map_repair_required = bool(
                            self.pending_map_observations
                        ) or bool(self.map_is_stale) or geometry_changed
                    if map_repair_required:
                        try:
                            self._synchronize_map_state(
                                header=left_message.header,
                                publish=True,
                            )
                        except Exception as exception:
                            self.get_logger().error(
                                "Pending GridMap transaction retry failed "
                                "after a tracking failure: "
                                f"{exception}"
                            )
                    if geometry_changed:
                        try:
                            self._rebuild_path_and_trajectory(
                                left_message.header
                            )
                        except Exception as exception:
                            self.get_logger().error(
                                "Corrected Path publication failed after a "
                                f"tracking failure: {exception}"
                            )
                if visual_tracking_attempted:
                    self.get_logger().warning(
                        (
                            "Tracking failed; failed-frame fusion skipped: "
                            if self.localization_mode == "vo"
                            else "VO tracking failed; continuing with the "
                            "external localisation path: "
                        )
                        + f"input_frame={self.received_frame_count}, "
                        f"algorithm_frame={tracking_status.frame_index}, "
                        f"reason={tracking_status.reason}, "
                        f"inliers={tracking_status.inlier_count}, "
                        f"correspondences="
                        f"{tracking_status.correspondence_count}, "
                        f"total_failures={self.tracking_failure_count}"
                    )
                if self.localization_mode == "vo":
                    processing_outcome = "tracking_failure"
                    return

            external_query = (
                self._wait_for_external_pose(timestamp)
                if self.localization_configuration.requires_external_odometry
                else None
            )
            try:
                localization_update = self.localization_coordinator.estimate(
                    timestamp=timestamp,
                    vo_pose=(
                        raw_vo_pose if vo_tracking_success else None
                    ),
                    external_query=external_query,
                )
            except LocalizationUnavailableError as exception:
                if vo_tracking_success:
                    self.tracking_failure_count += 1
                processing_outcome = "localization_unavailable"
                self.get_logger().warning(
                    "Localisation output skipped: "
                    f"{exception}; LIO={self.last_lio_diagnostic}"
                )
                return

            global_geometry_corrected = self._poll_mprf_loop_results()

            if self.localization_mode == "vo":
                pose_index = int(
                    getattr(tracking_status, "pose_index", -1)
                )
                if pose_index < 0:
                    raise RuntimeError(
                        "Successful tracking did not provide a valid "
                        "pose_index"
                    )
                geometry_corrected = bool(
                    getattr(
                        tracking_status, "geometry_corrected", False
                    )
                )
                pose_cv = np.asarray(
                    localization_update.estimate.pose, dtype=np.float64
                )
            else:
                pose_index = int(localization_update.measurement_index)
                geometry_corrected = (
                    self._record_external_localization_update(
                        localization_update,
                        tracking_status,
                    )
                    or global_geometry_corrected
                )
                pose_cv = np.asarray(
                    self.localization_pose_history[pose_index],
                    dtype=np.float64,
                )
            pose_map = pose_opencv_to_map(
                pose_cv,
                self.rotation_map_world,
                self.map_translation_offset,
            )
            pose_odom_cv = (
                np.asarray(
                    self.localization_odom_pose_history.get(
                        pose_index, pose_cv
                    ),
                    dtype=np.float64,
                )
                if self.localization_mode != "vo"
                else pose_cv
            )
            pose_odom = pose_opencv_to_map(
                pose_odom_cv,
                self.rotation_map_world,
                self.map_translation_offset,
            )

            self.frame_count += 1
            self.last_stamp = copy.deepcopy(left_message.header.stamp)
            self.last_header = copy.deepcopy(left_message.header)
            # Rolling-map geometry follows the calibrated vehicle origin,
            # not the left-camera optical centre.
            self.last_pose_map = self.odometry_output.transform_pose(pose_map)
            map_admission_decision = None
            if vo_tracking_success and self.localization_mode == "vo":
                map_admission_decision = (
                    self.map_admission_gate.observe_tracking_success(
                        raw_vo_pose,
                        reason=str(tracking_status.reason),
                        inlier_count=int(tracking_status.inlier_count),
                        correspondence_count=int(
                            tracking_status.correspondence_count
                        ),
                    )
                )
                visual_admission_observed = True
                if map_admission_decision.reason == "recovery_confirmed":
                    self.get_logger().info(
                        "Visual map admission reopened after "
                        f"{map_admission_decision.stable_frames} stable "
                        "tracking frames."
                    )
            self._publish_pose(
                pose_cv,
                pose_map,
                pose_odom,
                left_message.header,
                pose_index=pose_index,
                geometry_corrected=geometry_corrected,
                trajectory_stamp_ns=metric_stamp_ns,
                pose_covariance=localization_update.estimate.covariance,
                velocity_body=localization_update.estimate.velocity_body,
            )
            pose_published_wall = time.perf_counter()
            pose_publish_completion_monotonic_ns = int(
                round(pose_published_wall * 1_000_000_000)
            )
            pose_publish_latency_sec = (
                pose_published_wall - callback_start
            )
            stage_timer.switch(
                "map_update_and_publish", pose_published_wall
            )
            map_update_started_wall = pose_published_wall
            map_update_start_monotonic_ns = pose_publish_completion_monotonic_ns
            if vo_tracking_success:
                admit_new_observation = (
                    True
                    if map_admission_decision is None
                    else map_admission_decision.allow_new_observation
                )
                admission_motion = ""
                if (
                    map_admission_decision is not None
                    and map_admission_decision.translation_step_m is not None
                    and map_admission_decision.rotation_step_deg is not None
                ):
                    admission_motion = (
                        f", step="
                        f"{map_admission_decision.translation_step_m:.3f}m/"
                        f"{map_admission_decision.rotation_step_deg:.2f}deg"
                    )
                admission_reason = (
                    "not_applicable"
                    if map_admission_decision is None
                    else (
                        f"{map_admission_decision.reason}, "
                        f"state={map_admission_decision.state.value}, "
                        f"stable={map_admission_decision.stable_frames}/"
                        f"{map_admission_decision.stable_frames_required}, "
                        f"inliers={map_admission_decision.inlier_count}/"
                        f"{map_admission_decision.correspondence_count} "
                        f"({map_admission_decision.inlier_ratio:.3f})"
                        f"{admission_motion}"
                    )
                )
                map_statistics = self._update_map(
                    tracking_status=tracking_status,
                    semantic_image=semantic,
                    header=left_message.header,
                    admit_new_observation=admit_new_observation,
                    admission_reason=admission_reason,
                    visual_image_bgr=left,
                    pointcloud_target_ns=metric_stamp_ns,
                )
            else:
                map_statistics = self._update_map_without_vo_tracking(
                    localization_update=localization_update,
                    semantic_image=semantic,
                    header=left_message.header,
                    visual_image_bgr=left,
                )
            map_publish_completed_wall = time.perf_counter()
            map_publish_completion_monotonic_ns = int(
                round(map_publish_completed_wall * 1_000_000_000)
            )
            map_update_latency_sec = (
                map_publish_completed_wall - map_update_started_wall
            )

            if self._global_path_rebuild_pending:
                try:
                    self._rebuild_path_and_trajectory(left_message.header)
                    self._global_path_rebuild_pending = False
                except Exception as exception:
                    self.get_logger().warning(
                        "Globally corrected Path rewrite will be retried: "
                        f"{exception}"
                    )

            stage_timer.switch(
                "callback_bookkeeping", time.perf_counter()
            )
            elapsed = time.perf_counter() - start
            diagnostic = self.imu_buffer.last_diagnostic.message
            position = pose_map[:3, 3]

            self.get_logger().info(
                f"Frame {self.frame_count}: "
                f"sensor_time={_stamp_text(left_message.header.stamp)}, "
                f"elapsed={elapsed:.3f}s, "
                f"map_position=({position[0]:.3f}, "
                f"{position[1]:.3f}, {position[2]:.3f}), "
                f"source={localization_update.estimate.source}, "
                f"{diagnostic}"
                + (
                    f", LIO={self.last_lio_diagnostic}"
                    if self.localization_configuration.requires_external_odometry
                    else ""
                )
            )

            if map_statistics is not None:
                self.get_logger().info(
                    f"2.5D map revision {self.map_revision}: "
                    f"fused_points={map_statistics.get('accepted_points', 0)}, "
                    f"known_cells={map_statistics.get('known_cells', 0)}, "
                    f"known_area="
                    f"{map_statistics.get('known_area_m2', 0.0):.2f}m², "
                    f"elevation=["
                    f"{map_statistics.get('elevation_min', float('nan')):.3f}, "
                    f"{map_statistics.get('elevation_max', float('nan')):.3f}]m"
                )

        except Exception as exception:
            processing_outcome = "processing_exception"
            if (
                self.localization_mode == "vo"
                and not visual_admission_observed
            ):
                self.map_admission_gate.record_tracking_failure()
            self.tracking_failure_count += 1
            self.get_logger().error(
                f"Synchronized frame processing failed: {exception}\n"
                f"{traceback.format_exc()}\n"
                f"Total tracking/processing failures: "
                f"{self.tracking_failure_count}"
            )
        finally:
            self.processing = False
            terminal_wall = time.perf_counter()
            try:
                stage_durations = stage_timer.finish(terminal_wall)
            except Exception as exception:
                # Profiling is passive diagnostics. Never let it suppress the
                # authoritative terminal accounting or completion receipt.
                stage_durations = None
                self.get_logger().error(
                    "Callback stage timing was discarded: "
                    f"{type(exception).__name__}: {exception}"
                )
            self._finish_synchronized_input(
                left_message.header,
                outcome=processing_outcome,
                wall_time=terminal_wall,
                duration_sec=terminal_wall - callback_start,
                stage_durations_sec=stage_durations,
                sensor_stamp_ns=metric_stamp_ns,
                pose_publish_latency_sec=pose_publish_latency_sec,
                map_update_latency_sec=map_update_latency_sec,
                localization_solution_start_monotonic_ns=localization_solution_start_monotonic_ns,
                pose_publish_completion_monotonic_ns=pose_publish_completion_monotonic_ns,
                map_update_start_monotonic_ns=map_update_start_monotonic_ns,
                map_publish_completion_monotonic_ns=map_publish_completion_monotonic_ns,
            )

    def _publish_feature_match_debug(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        header: Header,
    ) -> bool:
        """Publish reused stereo/temporal matches only while subscribed."""
        publisher = self.feature_matches_pub
        if publisher is None or publisher.get_subscription_count() <= 0:
            return False
        try:
            getter = getattr(self.slam, "get_feature_match_snapshot", None)
            if not callable(getter):
                raise RuntimeError(
                    "Configured SLAM class has no get_feature_match_snapshot()"
                )
            snapshot = getter()
            if snapshot is None:
                return False
            visualization_left = self._feature_visualization_image(left)
            visualization_right = self._feature_visualization_image(right)
            visualization_snapshot = self._feature_visualization_snapshot(
                snapshot,
                source_shape=left.shape[:2],
            )
            debug_image = render_feature_match_debug(
                visualization_snapshot,
                current_left=visualization_left,
                current_right=visualization_right,
                output_width=self.feature_match_output_width,
                maximum_lines=self.feature_match_max_lines,
            )
            message = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            message.header = copy.deepcopy(header)
            publisher.publish(message)
            return True
        except Exception as exception:
            failures = int(
                getattr(self, "_feature_match_debug_failures", 0)
            ) + 1
            self._feature_match_debug_failures = failures
            if failures == 1 or failures % 20 == 0:
                self.get_logger().warning(
                    "Feature-match debug image was skipped without affecting "
                    f"tracking: {type(exception).__name__}: {exception}"
                )
            return False

    def _feature_visualization_snapshot(
        self,
        snapshot: dict,
        *,
        source_shape: tuple[int, int],
    ) -> dict:
        transformed = dict(snapshot)
        transformed["stereo_left_points"] = self._feature_visualization_points(
            snapshot["stereo_left_points"], source_shape
        )
        transformed["stereo_right_points"] = self._feature_visualization_points(
            snapshot["stereo_right_points"], source_shape
        )
        transformed["temporal_reference_points"] = (
            self._feature_visualization_points(
                snapshot["temporal_reference_points"], source_shape
            )
        )
        transformed["temporal_current_points"] = (
            self._feature_visualization_points(
                snapshot["temporal_current_points"], source_shape
            )
        )
        reference_image = snapshot.get("temporal_reference_image")
        if reference_image is not None:
            transformed["temporal_reference_image"] = (
                self._feature_visualization_image(reference_image)
            )
        return transformed

    def _feature_visualization_image(
        self, image: np.ndarray
    ) -> np.ndarray:
        value = np.asarray(image)
        source_height, source_width = value.shape[:2]
        crop_width = min(self.feature_visualization_crop_width, source_width)
        crop_height = min(
            self.feature_visualization_crop_height, source_height
        )
        crop_x = max(0, (source_width - crop_width) // 2)
        crop_y = max(0, (source_height - crop_height) // 2)
        cropped = value[
            crop_y:crop_y + crop_height,
            crop_x:crop_x + crop_width,
        ]
        return cv2.resize(
            cropped,
            (
                self.feature_visualization_width,
                self.feature_visualization_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    def _feature_visualization_points(
        self,
        points: np.ndarray,
        source_shape: tuple[int, int],
    ) -> np.ndarray:
        source_height, source_width = source_shape
        crop_width = min(self.feature_visualization_crop_width, source_width)
        crop_height = min(
            self.feature_visualization_crop_height, source_height
        )
        crop_x = max(0, (source_width - crop_width) // 2)
        crop_y = max(0, (source_height - crop_height) // 2)
        result = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        result[:, 0] = (
            (result[:, 0] - crop_x)
            * self.feature_visualization_width
            / float(crop_width)
        )
        result[:, 1] = (
            (result[:, 1] - crop_y)
            * self.feature_visualization_height
            / float(crop_height)
        )
        valid = (
            np.isfinite(result).all(axis=1)
            & (result[:, 0] >= 0.0)
            & (result[:, 0] < self.feature_visualization_width)
            & (result[:, 1] >= 0.0)
            & (result[:, 1] < self.feature_visualization_height)
        )
        result[~valid] = np.nan
        return result

    def _finish_synchronized_input(
        self,
        header: Header,
        *,
        outcome: str,
        wall_time: float,
        duration_sec: Optional[float] = None,
        stage_durations_sec: Optional[dict] = None,
        sensor_stamp_ns: Optional[int] = None,
        pose_publish_latency_sec: Optional[float] = None,
        map_update_latency_sec: Optional[float] = None,
        localization_solution_start_monotonic_ns: Optional[int] = None,
        pose_publish_completion_monotonic_ns: Optional[int] = None,
        map_update_start_monotonic_ns: Optional[int] = None,
        map_publish_completion_monotonic_ns: Optional[int] = None,
    ) -> None:
        """Close one callback and publish its acquisition-stamp receipt."""

        try:
            self.input_monitor.record_terminal(
                outcome,
                wall_time=wall_time,
                duration_sec=duration_sec,
                stage_durations_sec=stage_durations_sec,
                sensor_stamp_ns=sensor_stamp_ns,
                pose_publish_latency_sec=pose_publish_latency_sec,
                map_update_latency_sec=map_update_latency_sec,
                localization_solution_start_monotonic_ns=localization_solution_start_monotonic_ns,
                pose_publish_completion_monotonic_ns=pose_publish_completion_monotonic_ns,
                map_update_start_monotonic_ns=map_update_start_monotonic_ns,
                map_publish_completion_monotonic_ns=map_publish_completion_monotonic_ns,
            )
        except ValueError as exception:
            if stage_durations_sec is None:
                raise
            self.get_logger().error(
                "Callback stage accounting was discarded: "
                f"{exception}"
            )
            self.input_monitor.record_terminal(
                outcome,
                wall_time=wall_time,
                duration_sec=duration_sec,
                sensor_stamp_ns=sensor_stamp_ns,
                pose_publish_latency_sec=pose_publish_latency_sec,
                map_update_latency_sec=map_update_latency_sec,
                localization_solution_start_monotonic_ns=localization_solution_start_monotonic_ns,
                pose_publish_completion_monotonic_ns=pose_publish_completion_monotonic_ns,
                map_update_start_monotonic_ns=map_update_start_monotonic_ns,
                map_publish_completion_monotonic_ns=map_publish_completion_monotonic_ns,
            )
        if self.realtime_metrics_pub is not None:
            self.realtime_metrics_sequence += 1
            effective_stamp_ns = (
                int(sensor_stamp_ns)
                if sensor_stamp_ns is not None
                else int(header.stamp.sec) * 1_000_000_000
                + int(header.stamp.nanosec)
            )
            telemetry = String()
            telemetry.data = json.dumps(
                {
                    "schema_version": 1,
                    "sequence": self.realtime_metrics_sequence,
                    "sensor_stamp_ns": effective_stamp_ns,
                    "sensor_stamp_sec": effective_stamp_ns * 1e-9,
                    "frame_id": str(header.frame_id),
                    "outcome": str(outcome),
                    "pose_processing_sec": pose_publish_latency_sec,
                    "map_update_sec": map_update_latency_sec,
                    "end_to_end_sec": duration_sec,
                    "localization_solution_start_monotonic_ns": (
                        localization_solution_start_monotonic_ns
                    ),
                    "pose_publish_completion_monotonic_ns": (
                        pose_publish_completion_monotonic_ns
                    ),
                    "map_update_start_monotonic_ns": map_update_start_monotonic_ns,
                    "map_publish_completion_monotonic_ns": (
                        map_publish_completion_monotonic_ns
                    ),
                },
                separators=(",", ":"),
                allow_nan=False,
            )
            try:
                self.realtime_metrics_pub.publish(telemetry)
            except Exception as exception:
                self.get_logger().error(
                    "Could not publish realtime frame telemetry "
                    f"for stamp={_stamp_text(header.stamp)}: {exception}"
                )
        if self.input_processed_pub is not None:
            completion = Header()
            completion.stamp = copy.deepcopy(header.stamp)
            completion.frame_id = str(header.frame_id)
            try:
                self.input_processed_pub.publish(completion)
                self.input_monitor.record_completion_notification()
            except Exception as exception:
                self.get_logger().error(
                    "Could not publish mapper input completion notification "
                    f"for stamp={_stamp_text(header.stamp)}: {exception}"
                )

    def _get_map_points_cv(self) -> np.ndarray:
        if self.localization_mode != "vo":
            with self._map_lock:
                poses = {
                    keyframe_id: np.asarray(pose, dtype=np.float64).copy()
                    for keyframe_id, pose
                    in self.localization_keyframe_poses.items()
                }
                return self.map_observations.transformed_point_sample(
                    keyframe_poses_cv=poses,
                    max_points=self.pointcloud_max_points,
                )
        try:
            points, _ = self.slam.get_map_points(
                include_bad=False,
                min_obs=1,
            )
        except TypeError:
            points, _ = self.slam.get_map_points()
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def _make_pointcloud_assist_fallback_observation(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        header: Header,
    ) -> SemanticKeyFrameObservation:
        """Keep only the configured reliable stereo range without a cloud.

        Assisted mode must not silently re-introduce the unsupported far
        stereo estimates it was selected to reject.  This fallback is scoped
        to ``lidar_mapping_mode=assist``. Stereo-only mapping is unaffected;
        an unavailable selected geometry source follows the common retry and
        fail-stop completeness policy.
        """

        observation, fallback_statistics = (
            self.stereo_observation_builder.make_fallback(
                keyframe_id=keyframe_id,
                semantic_image=semantic_image,
                sensor_timestamp=_stamp_seconds(header.stamp),
            )
        )
        maximum_depth = float(fallback_statistics["maximum_depth_m"])
        output_points = int(fallback_statistics["output_points"])
        if maximum_depth > 0.0:
            diagnostic = (
                f"PointCloud2-assisted fallback for keyframe {keyframe_id}: "
                f"retained {output_points} stereo point(s) with "
                f"Z<={maximum_depth:.3f}m."
            )
        else:
            diagnostic = (
                f"PointCloud2-assisted fallback for keyframe {keyframe_id}: "
                f"depth gate disabled, retained {output_points} "
                "stereo point(s)."
            )
        self.get_logger().info(diagnostic)
        return observation

    def _make_pointcloud_assisted_stereo_observation(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        header: Header,
        visual_image_bgr: Optional[np.ndarray] = None,
        lidar_message: Optional[PointCloud2] = None,
    ) -> SemanticKeyFrameObservation:
        cloud_observation, cloud_statistics, _, points_sensor = (
            self._prepare_lidar_keyframe_data(
                keyframe_id=keyframe_id,
                semantic_image=semantic_image,
                header=header,
                lidar_message=lidar_message,
            )
        )
        self._register_global_lidar_keyframe(
            keyframe_id=keyframe_id,
            header=header,
            points_sensor=points_sensor,
            visual_image_bgr=visual_image_bgr,
        )
        observation, assist_statistics = (
            self.stereo_observation_builder.make_assisted(
                keyframe_id=keyframe_id,
                semantic_image=semantic_image,
                sensor_timestamp=_stamp_seconds(header.stamp),
                pointcloud_camera_points=(
                    cloud_observation.camera_points
                ),
                camera_matrix=self.lidar_processor.camera_matrix,
                calibration_width=(
                    self.lidar_processor.calibration_width
                ),
                calibration_height=(
                    self.lidar_processor.calibration_height
                ),
            )
        )
        stereo_count = int(
            assist_statistics.pop("input_stereo_points")
        )
        supported = int(
            assist_statistics["pointcloud_supported_points"]
        )
        statistics = dict(cloud_statistics)
        statistics.update(
            {
                f"assist_{name}": value
                for name, value in assist_statistics.items()
            }
        )
        self.get_logger().info(
            f"PointCloud2-assisted stereo observation for keyframe "
            f"{keyframe_id}: stereo={stereo_count}, "
            f"supported={supported}, output="
            f"{assist_statistics['output_points']}, dropped_far="
            f"{assist_statistics['unmatched_stereo_dropped']}, "
            f"time_offset={cloud_statistics['time_offset_sec']:+.3f}s."
        )
        return observation

    def _make_keyframe_observation(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        header: Header,
        visual_image_bgr: Optional[np.ndarray] = None,
        lidar_message: Optional[PointCloud2] = None,
    ) -> Optional[SemanticKeyFrameObservation]:
        if self.enable_lidar_mapping:
            try:
                if self.lidar_mapping_mode == "assist":
                    return self._make_pointcloud_assisted_stereo_observation(
                        keyframe_id=keyframe_id,
                        semantic_image=semantic_image,
                        header=header,
                        visual_image_bgr=visual_image_bgr,
                        lidar_message=lidar_message,
                    )
                return self._make_lidar_keyframe_observation(
                    keyframe_id=keyframe_id,
                    semantic_image=semantic_image,
                    header=header,
                    visual_image_bgr=visual_image_bgr,
                    lidar_message=lidar_message,
                )
            except Exception as exception:
                self.lidar_skipped_count += 1
                if (
                    self.lidar_skipped_count == 1
                    or self.lidar_skipped_count % 20 == 0
                ):
                    action = (
                        "falling back to stereo geometry"
                        if self.lidar_fallback_to_stereo
                        else "blocking map publication until a retry succeeds"
                    )
                    self.get_logger().warning(
                        f"PointCloud2 mapping unavailable for keyframe "
                        f"{keyframe_id}: {exception}; {action}."
                    )
                if not self.lidar_fallback_to_stereo:
                    raise
                if self.lidar_mapping_mode == "assist":
                    return self._make_pointcloud_assist_fallback_observation(
                        keyframe_id=keyframe_id,
                        semantic_image=semantic_image,
                        header=header,
                    )
        if self.stereo_observation_builder is None:
            raise RuntimeError(
                "Stereo map geometry was requested while the visual "
                "frontend is intentionally disabled"
            )
        return self.stereo_observation_builder.make_stereo(
            keyframe_id=keyframe_id,
            semantic_image=semantic_image,
            sensor_timestamp=_stamp_seconds(header.stamp),
        )

    def _get_keyframe_pose_state(self) -> tuple[int, dict[int, np.ndarray]]:
        if self.localization_mode != "vo":
            return self.localization_geometry_revision, {
                int(keyframe_id): np.asarray(
                    pose, dtype=np.float64
                ).copy()
                for keyframe_id, pose in (
                    self.localization_keyframe_poses.items()
                )
            }
        getter = getattr(self.slam, "get_keyframe_pose_snapshot", None)
        if not callable(getter):
            raise RuntimeError(
                "The configured SLAM class does not expose "
                "get_keyframe_pose_snapshot()"
            )
        revision, poses = getter(include_bad=False)
        return int(revision), {
            int(keyframe_id): np.asarray(pose, dtype=np.float64).copy()
            for keyframe_id, pose in poses.items()
        }

    def _stage_keyframe_observation(
        self,
        observation: SemanticKeyFrameObservation,
    ) -> bool:
        """Stage an observation without claiming that map fusion succeeded."""
        keyframe_id = int(observation.keyframe_id)
        with self._map_lock:
            committed = self.map_observations.get(keyframe_id)
            if committed is not None:
                if not self.map_observations.observations_equal(
                    committed, observation
                ):
                    raise ValueError(
                        f"Committed keyframe {keyframe_id} conflicts with "
                        "a new observation"
                    )
                return False

            pending = self.pending_map_observations.get(keyframe_id)
            if pending is not None:
                if not self.map_observations.observations_equal(
                    pending, observation
                ):
                    raise ValueError(
                        f"Pending keyframe {keyframe_id} conflicts with "
                        "a new observation"
                    )
                return False

            self.pending_map_observations[keyframe_id] = observation
            return True

    def _record_missed_keyframe_observation(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        header: Header,
        exception: Exception,
        visual_image_bgr: Optional[np.ndarray] = None,
    ) -> None:
        """Cache a failed observation build for retry at sync boundaries.

        观测必须用关键帧自己那一帧的语义图采样，所以这里缓存的是图像
        本身而不仅是关键帧号。缓存有上限；如果上限耗尽，则进入显式
        fail-stop 状态并冻结地图发布，绝不丢掉旧证据后继续发布一张看似
        正常但实际不完整的地图。
        """
        keyframe_id = int(keyframe_id)
        capacity_fault = None
        with self._map_lock:
            if keyframe_id not in self.missed_keyframe_observations:
                limit = max(
                    1,
                    int(getattr(self, "missed_observation_cache_limit", 8)),
                )
                if len(self.missed_keyframe_observations) >= limit:
                    capacity_fault = (
                        "Missed-observation cache is full; cannot retain "
                        f"keyframe {keyframe_id}. Map publication is frozen "
                        "to prevent incomplete output."
                    )
                    self.map_integrity_fault = capacity_fault
                else:
                    self.missed_keyframe_observations[keyframe_id] = {
                        "semantic_image": np.array(semantic_image, copy=True),
                        "header": copy.deepcopy(header),
                        "attempts": 0,
                    }
                    if visual_image_bgr is not None:
                        self.missed_keyframe_observations[keyframe_id][
                            "visual_image_bgr"
                        ] = np.array(visual_image_bgr, copy=True)
                    lidar_message = getattr(exception, "lidar_message", None)
                    if lidar_message is not None:
                        # DDS callback messages are immutable after delivery.
                        # Keep the exact selected cloud by reference; the
                        # surrounding cache is already strictly bounded.
                        self.missed_keyframe_observations[keyframe_id][
                            "lidar_message"
                        ] = lidar_message
        if capacity_fault is not None:
            self.get_logger().fatal(capacity_fault)
            raise RuntimeError(capacity_fault)
        self.get_logger().error(
            f"Keyframe {keyframe_id} observation build failed; cached its "
            f"frame for retry: {exception}"
        )

    def _retry_missed_keyframe_observations(self) -> None:
        """Rebuild cached failed observations before the next map commit."""
        with self._map_lock:
            entries = list(self.missed_keyframe_observations.items())
        if not entries:
            return
        warning_interval = max(
            1,
            int(getattr(self, "missed_observation_warning_interval", 20)),
        )
        for keyframe_id, entry in entries:
            try:
                observation_arguments = {
                    "keyframe_id": keyframe_id,
                    "semantic_image": entry["semantic_image"],
                    "header": entry["header"],
                }
                if entry.get("visual_image_bgr") is not None:
                    observation_arguments["visual_image_bgr"] = entry[
                        "visual_image_bgr"
                    ]
                if entry.get("lidar_message") is not None:
                    observation_arguments["lidar_message"] = entry[
                        "lidar_message"
                    ]
                observation = self._make_keyframe_observation(
                    **observation_arguments
                )
                if observation is not None:
                    self._stage_keyframe_observation(observation)
            except Exception as exception:
                attempts = 0
                with self._map_lock:
                    current = self.missed_keyframe_observations.get(
                        keyframe_id
                    )
                    if current is not None:
                        current["attempts"] = int(current["attempts"]) + 1
                        attempts = int(current["attempts"])
                if attempts > 0 and attempts % warning_interval == 0:
                    self.get_logger().error(
                        f"Keyframe {keyframe_id} observation retry failed "
                        f"{attempts} times; evidence remains retained and "
                        f"map publication remains blocked: {exception}"
                    )
                else:
                    self.get_logger().warning(
                        f"Keyframe {keyframe_id} observation retry failed; "
                        f"will retry again: {exception}"
                    )
                continue
            with self._map_lock:
                self.missed_keyframe_observations.pop(keyframe_id, None)
            self.get_logger().info(
                f"Keyframe {keyframe_id} map observation resolved after "
                "retry."
            )

    def _pending_observation_snapshot(
        self,
    ) -> tuple[SemanticKeyFrameObservation, ...]:
        with self._map_lock:
            return tuple(
                self.pending_map_observations[keyframe_id]
                for keyframe_id in sorted(self.pending_map_observations)
            )

    def _commit_map_candidate(
        self,
        *,
        candidate: TiledSemanticMapManager,
        pending_observations: tuple[SemanticKeyFrameObservation, ...],
        geometry_revision: int,
        contribution_index: Optional[MapContributionIndex] = None,
        base_map_revision: Optional[int] = None,
    ) -> None:
        """Commit provenance and its already-built grid in one lock scope."""
        with self._map_lock:
            if (
                base_map_revision is not None
                and int(self.map_revision) != int(base_map_revision)
            ):
                raise RuntimeError(
                    "Live map changed while corrected candidate was built"
                )
            for observation in pending_observations:
                keyframe_id = int(observation.keyframe_id)
                current = self.pending_map_observations.get(keyframe_id)
                if current is None or not self.map_observations.observations_equal(
                    current, observation
                ):
                    raise RuntimeError(
                        f"Pending keyframe {keyframe_id} changed during commit"
                    )
                if self.map_observations.get(keyframe_id) is not None:
                    raise RuntimeError(
                        f"Pending keyframe {keyframe_id} is already committed"
                    )

            if contribution_index is not None:
                contribution_index.assert_consistent()
                expected_keyframes = set(self.map_observations.keyframe_ids)
                expected_keyframes.update(
                    int(observation.keyframe_id)
                    for observation in pending_observations
                )
                if not contribution_index.is_complete_for(
                    expected_keyframes
                ):
                    raise RuntimeError(
                        "Corrected-map contribution index is incomplete"
                    )

            # add_many() performs all conflict checks before mutating the
            # provenance store. The assignments and pop() calls below cannot
            # expose an intermediate state while this lock is held.
            self.map_observations.add_many(pending_observations)
            self.grid = candidate
            if contribution_index is not None:
                self.map_contribution_index = contribution_index
            for observation in pending_observations:
                self.pending_map_observations.pop(
                    int(observation.keyframe_id),
                    None,
                )
            self.applied_geometry_revision = int(geometry_revision)
            self.map_revision += 1
            self.map_is_stale = False
            compact_writer = getattr(
                self, "compact_global_map_writer", None
            )
            if compact_writer is not None:
                last_header = getattr(self, "last_header", None)
                timestamp_text = (
                    _stamp_text(last_header.stamp)
                    if last_header is not None
                    else f"wall_{time.time_ns()}"
                )
                compact_writer.enqueue(
                    self.grid.clone_for_update(),
                    map_revision=self.map_revision,
                    timestamp_text=timestamp_text,
                )

    def _rebuild_historical_map(
        self,
        *,
        header: Header,
        publish: bool,
    ) -> dict:
        geometry_revision, keyframe_poses = self._get_keyframe_pose_state()
        pending_observations = self._pending_observation_snapshot()
        with self._map_lock:
            live_grid = self.grid
            contribution_index = getattr(
                self, "map_contribution_index", None
            )
            base_map_revision = int(self.map_revision)
        start = time.perf_counter()
        try:
            candidate, statistics, candidate_index = (
                self.map_observations.build_corrected_candidate(
                    live_grid=live_grid,
                    grid_factory=self._make_empty_grid,
                    contribution_index=contribution_index,
                    pending_observations=pending_observations,
                    keyframe_poses_cv=keyframe_poses,
                    rotation_map_world=self.rotation_map_world,
                    map_translation_offset=self.map_translation_offset,
                )
            )
            self._commit_map_candidate(
                candidate=candidate,
                pending_observations=pending_observations,
                geometry_revision=geometry_revision,
                contribution_index=candidate_index,
                base_map_revision=base_map_revision,
            )
        except Exception:
            with self._map_lock:
                self.map_is_stale = True
            raise

        statistics["rebuild_seconds"] = time.perf_counter() - start
        self.get_logger().info(
            f"Atomically rebuilt GridMap revision {self.map_revision} via "
            f"{statistics.get('rebuild_strategy', 'full')} replay "
            f"({statistics.get('replayed_keyframes', 0)} keyframes, "
            f"{statistics.get('affected_tiles', 0)} tiles) in "
            f"{statistics['rebuild_seconds']:.3f}s "
            f"(SLAM geometry revision {geometry_revision})."
        )
        if publish:
            self._publish_map_if_changed(header)
        return statistics

    def _start_background_map_correction(
        self,
        *,
        geometry_revision: int,
        keyframe_poses: dict[int, np.ndarray],
    ) -> bool:
        """Capture immutable inputs and start one candidate build."""
        pending_observations = self._pending_observation_snapshot()
        with self._map_lock:
            request = _MapCorrectionRequest(
                geometry_revision=int(geometry_revision),
                base_map_revision=int(self.map_revision),
                live_grid=self.grid,
                contribution_index=self.map_contribution_index,
                pending_observations=pending_observations,
                keyframe_poses_cv={
                    int(keyframe_id): np.asarray(
                        pose, dtype=np.float64
                    ).copy()
                    for keyframe_id, pose in keyframe_poses.items()
                },
            )

        def build_candidate():
            return self.map_observations.build_corrected_candidate(
                live_grid=request.live_grid,
                grid_factory=self._make_empty_grid,
                contribution_index=request.contribution_index,
                pending_observations=request.pending_observations,
                keyframe_poses_cv=request.keyframe_poses_cv,
                rotation_map_world=self.rotation_map_world,
                map_translation_offset=self.map_translation_offset,
            )

        submitted = self.map_correction_worker.submit(
            metadata=request,
            build=build_candidate,
        )
        if submitted:
            with self._map_lock:
                self.map_is_stale = True
            self.get_logger().info(
                "Started background GridMap correction for geometry "
                f"revision {geometry_revision} from map revision "
                f"{request.base_map_revision}."
            )
        return submitted

    def _commit_background_map_correction(
        self,
        *,
        completion: MapCorrectionCompletion,
        current_geometry_revision: int,
        current_keyframe_poses: dict[int, np.ndarray],
        header: Header,
        publish: bool,
    ) -> Optional[dict]:
        """Validate and atomically switch one completed candidate.

        ``None`` means the result became obsolete and should be rebuilt from
        a newer pose/map revision.  Build failures remain retryable but are
        raised so callers and save services never report false success.
        """
        request = completion.metadata
        if not isinstance(request, _MapCorrectionRequest):
            raise RuntimeError("Unknown background map-correction metadata")
        if completion.exception is not None:
            with self._map_lock:
                self.map_is_stale = True
            raise completion.exception

        candidate, statistics, candidate_index = completion.value
        if candidate_index is None:
            raise RuntimeError(
                "Background correction did not produce a contribution index"
            )

        with self._map_lock:
            base_is_current = (
                int(self.map_revision) == request.base_map_revision
            )
        geometry_is_current = (
            int(current_geometry_revision) == request.geometry_revision
        )
        poses_are_current = geometry_is_current
        if poses_are_current:
            for keyframe_id in candidate_index.keyframe_ids:
                requested_pose = request.keyframe_poses_cv.get(keyframe_id)
                current_pose = current_keyframe_poses.get(keyframe_id)
                if (
                    requested_pose is None
                    or current_pose is None
                    or not np.array_equal(requested_pose, current_pose)
                ):
                    poses_are_current = False
                    break

        if not base_is_current or not poses_are_current:
            reason = (
                "live map revision changed"
                if not base_is_current
                else "keyframe geometry advanced"
            )
            self.get_logger().info(
                "Discarded obsolete background GridMap candidate: "
                f"{reason}."
            )
            with self._map_lock:
                self.map_is_stale = True
            return None

        # Frames may continue arriving while the worker builds. Fuse every
        # newly staged observation into the private candidate before the one
        # atomic switch, so no published revision knowingly omits evidence
        # already accepted at this synchronization boundary.
        pending_now = self._pending_observation_snapshot()
        captured_ids = {
            int(observation.keyframe_id)
            for observation in request.pending_observations
        }
        additional_pending = tuple(
            observation
            for observation in pending_now
            if int(observation.keyframe_id) not in captured_ids
        )
        if additional_pending:
            preceding_keys = [
                self.map_observations.observation_order_key(observation)
                for observation in request.pending_observations
            ]
            archive_key = (
                self.map_observations.latest_archive_order_key
            )
            if archive_key is not None:
                preceding_keys.append(archive_key)
            latest_preceding = (
                max(preceding_keys) if preceding_keys else None
            )
            earliest_additional = min(
                self.map_observations.observation_order_key(observation)
                for observation in additional_pending
            )
            if (
                latest_preceding is not None
                and earliest_additional <= latest_preceding
            ):
                self.get_logger().info(
                    "Restarting background GridMap correction to insert "
                    "a recovered historical observation in sensor-time "
                    "order."
                )
                with self._map_lock:
                    self.map_is_stale = True
                return None
        extension_start = time.perf_counter()
        if additional_pending:
            tile_keys_by_keyframe: dict[int, tuple] = {}
            self.map_observations.fuse_observations(
                grid=candidate,
                observations=additional_pending,
                keyframe_poses_cv=current_keyframe_poses,
                rotation_map_world=self.rotation_map_world,
                map_translation_offset=self.map_translation_offset,
                tile_keys_by_keyframe=tile_keys_by_keyframe,
            )
            candidate_index.record_many(
                poses_world_camera={
                    int(observation.keyframe_id): current_keyframe_poses[
                        int(observation.keyframe_id)
                    ]
                    for observation in additional_pending
                },
                tile_keys_by_keyframe=tile_keys_by_keyframe,
            )
        candidate_index.assert_consistent()

        self._commit_map_candidate(
            candidate=candidate,
            pending_observations=pending_now,
            geometry_revision=current_geometry_revision,
            contribution_index=candidate_index,
            base_map_revision=request.base_map_revision,
        )
        extension_seconds = time.perf_counter() - extension_start
        added_count = len(additional_pending)
        statistics["replayed_keyframes"] = int(
            statistics.get("replayed_keyframes", 0)
        ) + added_count
        statistics["fusion_operations"] = int(
            statistics.get("fusion_operations", 0)
        ) + added_count
        statistics["total_keyframes"] = len(candidate_index)
        statistics["accepted_points"] = int(
            candidate.total_accepted_points
        )
        statistics["background_seconds"] = float(
            completion.elapsed_seconds
        )
        statistics["rebuild_seconds"] = float(
            completion.elapsed_seconds + extension_seconds
        )
        self.get_logger().info(
            f"Atomically switched GridMap revision {self.map_revision} via "
            f"background {statistics.get('rebuild_strategy', 'full')} "
            f"replay ({statistics.get('replayed_keyframes', 0)} keyframes, "
            f"{statistics.get('affected_tiles', 0)} tiles) in "
            f"{statistics['rebuild_seconds']:.3f}s."
        )
        if publish:
            self._publish_map_if_changed(header)
        return statistics

    def _synchronize_background_map_correction(
        self,
        *,
        header: Header,
        publish: bool,
        wait: bool,
    ) -> Optional[dict]:
        """Poll/start correction; block only for explicit save/shutdown."""
        while True:
            completion = self.map_correction_worker.take_completed()
            if completion is None and self.map_correction_worker.active:
                if not wait:
                    if publish:
                        self._publish_map_if_changed(header)
                    return {
                        "rebuild_strategy": "background_pending",
                        "background_state": "building",
                    }
                completion = self.map_correction_worker.take_completed(
                    wait=True
                )

            geometry_revision, keyframe_poses = (
                self._get_keyframe_pose_state()
            )
            if completion is not None:
                committed = self._commit_background_map_correction(
                    completion=completion,
                    current_geometry_revision=geometry_revision,
                    current_keyframe_poses=keyframe_poses,
                    header=header,
                    publish=publish,
                )
                if committed is not None:
                    return committed
                continue

            with self._map_lock:
                rebuild_required = bool(self.map_is_stale) or (
                    geometry_revision != self.applied_geometry_revision
                )
            if not rebuild_required:
                if publish:
                    self._publish_map_if_changed(header)
                return None

            started = self._start_background_map_correction(
                geometry_revision=geometry_revision,
                keyframe_poses=keyframe_poses,
            )
            if not started:
                continue
            if not wait:
                if publish:
                    self._publish_map_if_changed(header)
                return {
                    "rebuild_strategy": "background_pending",
                    "background_state": "started",
                }

    def _on_map_correction_timer(self) -> None:
        """Commit a ready result even when no later image frame arrives."""
        worker = getattr(self, "map_correction_worker", None)
        if worker is None or not worker.ready or self.last_header is None:
            return
        try:
            self._synchronize_background_map_correction(
                header=self.last_header,
                publish=True,
                wait=False,
            )
        except Exception as exception:
            # The completion was consumed and map_is_stale stays set. A new
            # image, manual save, or shutdown boundary will retry; the timer
            # does not enter a 20 Hz failure loop.
            self.get_logger().error(
                "Background GridMap correction commit failed; retained the "
                f"last complete map for retry: {exception}"
            )

    def _apply_pointcloud_elevation_to_candidate(
        self,
        candidate: TiledSemanticMapManager,
        target_stamp_ns: Optional[int],
    ):
        mapper = self.terrain_pointcloud_mapper
        if mapper is None or target_stamp_ns is None:
            return None, None
        update = mapper.prepare_update(
            target_stamp_ns=int(target_stamp_ns),
            transform_map_base=self.last_pose_map,
            map_resolution=candidate.resolution,
        )
        if len(update.points_map) == 0:
            return update, None
        statistics = candidate.update_elevation_only(
            points_map=update.points_map
        )
        return update, statistics

    def _commit_pending_map_observations(
        self,
        *,
        header: Header,
        publish: bool,
        geometry_revision: int,
        keyframe_poses: dict[int, np.ndarray],
        pointcloud_target_ns: Optional[int] = None,
    ) -> Optional[dict]:
        pending_observations = self._pending_observation_snapshot()
        if not pending_observations:
            if publish:
                self._publish_map_if_changed(header)
            return None

        start = time.perf_counter()
        # The live grid remains untouched even if deepcopy, pose lookup or a
        # later fusion operation raises. Pending observations are retained.
        with self._map_lock:
            contribution_index = getattr(
                self, "map_contribution_index", None
            )
            tile_keys_by_keyframe = (
                {} if contribution_index is not None else None
            )
            candidate, statistics = (
                self.map_observations.build_incremental_candidate(
                    live_grid=self.grid,
                    observations=pending_observations,
                    keyframe_poses_cv=keyframe_poses,
                    rotation_map_world=self.rotation_map_world,
                    map_translation_offset=self.map_translation_offset,
                    tile_keys_by_keyframe=tile_keys_by_keyframe,
                )
            )
            candidate_index = None
            if contribution_index is not None:
                candidate_index = contribution_index.clone()
                candidate_index.record_many(
                    poses_world_camera={
                        int(observation.keyframe_id): keyframe_poses[
                            int(observation.keyframe_id)
                        ]
                        for observation in pending_observations
                    },
                    tile_keys_by_keyframe=tile_keys_by_keyframe,
                )
                candidate_index.assert_consistent()
            pointcloud_update = None
            pointcloud_statistics = None
            try:
                pointcloud_update, pointcloud_statistics = (
                    self._apply_pointcloud_elevation_to_candidate(
                        candidate, pointcloud_target_ns
                    )
                )
            except Exception as exception:
                self.get_logger().warning(
                    "Point-cloud elevation evidence was skipped without "
                    "affecting the stereo map transaction: "
                    f"{type(exception).__name__}: {exception}"
                )
            if pointcloud_update is not None:
                statistics.update(
                    {
                        f"pointcloud_{key}": value
                        for key, value in (
                            pointcloud_update.statistics.items()
                        )
                    }
                )
            if pointcloud_statistics is not None:
                statistics["accepted_points"] = int(
                    statistics.get("accepted_points", 0)
                ) + int(pointcloud_statistics["accepted_points"])
                for key in (
                    "known_cells",
                    "known_area_m2",
                    "elevation_min",
                    "elevation_max",
                    "surface_cells",
                    "tile_count",
                ):
                    if key in pointcloud_statistics:
                        statistics[key] = pointcloud_statistics[key]

            self._commit_map_candidate(
                candidate=candidate,
                pending_observations=pending_observations,
                geometry_revision=geometry_revision,
                contribution_index=candidate_index,
                base_map_revision=self.map_revision,
            )
            if pointcloud_statistics is not None:
                self.terrain_pointcloud_mapper.mark_committed(
                    pointcloud_update.selected_stamps_ns
                )
                self.get_logger().info(
                    "Point-cloud elevation committed: sensors="
                    f"{pointcloud_update.statistics['selected_sensors']}, "
                    "cell_observations="
                    f"{pointcloud_update.statistics['cell_observations']}, "
                    "accepted="
                    f"{pointcloud_statistics['accepted_points']}."
                )

        statistics["commit_seconds"] = time.perf_counter() - start
        self.get_logger().info(
            f"Atomically committed {len(pending_observations)} keyframe "
            f"observation(s) to GridMap revision {self.map_revision} in "
            f"{statistics['commit_seconds']:.3f}s."
        )
        if publish:
            self._publish_map_if_changed(header)
        return statistics

    def _synchronize_map_state(
        self,
        *,
        header: Header,
        publish: bool,
        wait_for_correction: bool = False,
        pointcloud_target_ns: Optional[int] = None,
    ) -> Optional[dict]:
        """Commit pending evidence or repair stale corrected geometry."""
        if getattr(self, "missed_keyframe_observations", None):
            self._retry_missed_keyframe_observations()
        with self._map_lock:
            integrity_fault = getattr(self, "map_integrity_fault", None)
            unresolved_keyframes = tuple(
                sorted(
                    getattr(self, "missed_keyframe_observations", {})
                )
            )
        if integrity_fault is not None:
            raise RuntimeError(integrity_fault)
        if unresolved_keyframes:
            raise RuntimeError(
                "Refusing to publish an incomplete map; observation build "
                f"is still unresolved for keyframe(s) {unresolved_keyframes}."
            )
        geometry_revision, keyframe_poses = self._get_keyframe_pose_state()
        with self._map_lock:
            pending_snapshot = tuple(
                self.pending_map_observations[keyframe_id]
                for keyframe_id in sorted(self.pending_map_observations)
            )
            pending_exists = bool(pending_snapshot)
            historical_pending = (
                pending_exists
                and not self.map_observations.observations_follow_archive(
                    pending_snapshot
                )
            )
            full_rebuild_required = bool(self.map_is_stale) or (
                geometry_revision != self.applied_geometry_revision
            ) or historical_pending

        if full_rebuild_required:
            if getattr(self, "map_correction_worker", None) is not None:
                return self._synchronize_background_map_correction(
                    header=header,
                    publish=publish,
                    wait=wait_for_correction,
                )
            return self._rebuild_historical_map(
                header=header,
                publish=publish,
            )
        if pending_exists:
            return self._commit_pending_map_observations(
                header=header,
                publish=publish,
                geometry_revision=geometry_revision,
                keyframe_poses=keyframe_poses,
                pointcloud_target_ns=pointcloud_target_ns,
            )
        if publish:
            self._publish_map_if_changed(header)
        return None

    def _update_map(
        self,
        *,
        tracking_status,
        semantic_image: np.ndarray,
        header: Header,
        admit_new_observation: bool = True,
        admission_reason: str = "not_gated",
        visual_image_bgr: Optional[np.ndarray] = None,
        pointcloud_target_ns: Optional[int] = None,
    ) -> Optional[dict]:
        if bool(getattr(tracking_status, "keyframe_created", False)):
            keyframe_id = getattr(tracking_status, "keyframe_id", None)
            if keyframe_id is None:
                raise RuntimeError(
                    "SLAM reported keyframe_created without keyframe_id"
                )
            if not admit_new_observation:
                # This is an intentional accuracy decision, not an observation
                # construction failure.  Never put it into the retry/fail-stop
                # path, because doing so would later reintroduce rejected
                # evidence into the complete map.
                self.get_logger().warning(
                    "Visual keyframe map observation intentionally rejected: "
                    f"keyframe_id={int(keyframe_id)}, {admission_reason}. "
                    "Odometry publication is unchanged."
                )
            else:
                try:
                    observation_arguments = {
                        "keyframe_id": int(keyframe_id),
                        "semantic_image": semantic_image,
                        "header": header,
                    }
                    if visual_image_bgr is not None:
                        observation_arguments["visual_image_bgr"] = (
                            visual_image_bgr
                        )
                    new_observation = self._make_keyframe_observation(
                        **observation_arguments
                    )
                    if new_observation is None:
                        self.get_logger().warning(
                            "Visual keyframe contains no reliable stereo "
                            "map evidence after configured selection; "
                            f"keyframe_id={int(keyframe_id)} is skipped for "
                            "mapping while odometry remains unchanged."
                        )
                    else:
                        self._stage_keyframe_observation(new_observation)
                except Exception as exception:
                    self._record_missed_keyframe_observation(
                        keyframe_id=int(keyframe_id),
                        semantic_image=semantic_image,
                        header=header,
                        exception=exception,
                        visual_image_bgr=visual_image_bgr,
                    )

        # Every call remains a transaction retry boundary. Publication is
        # revision/window-aware, so an ordinary frame does not rebuild an
        # identical transient-local map payload.
        return self._synchronize_map_state(
            header=header,
            publish=True,
            pointcloud_target_ns=pointcloud_target_ns,
        )

    def _map_output_signature(self) -> tuple:
        """Identify the complete map payload without materialising arrays."""
        with self._map_lock:
            if self.dense_map_window_mode == "fixed":
                center_x = self.fixed_map_center_x
                center_y = self.fixed_map_center_y
            else:
                center_x = float(self.last_pose_map[0, 3])
                center_y = float(self.last_pose_map[1, 3])
            geometry = self.grid.window_geometry(
                center_x=center_x,
                center_y=center_y,
                length_x=self.local_map_length_x,
                length_y=self.local_map_length_y,
            )
            return (
                int(self.map_revision),
                int(geometry.width),
                int(geometry.height),
                float(geometry.origin_x),
                float(geometry.origin_y),
            )

    def _publish_map_if_changed(self, header: Header) -> bool:
        """Publish once per map revision or cell-aligned window position.

        The signature is recorded only after every map publisher succeeds.
        A partial DDS/conversion failure is therefore retried at the next
        synchronization boundary instead of being mistaken for success.
        """
        signature = self._map_output_signature()
        if (
            not self.publish_unchanged_map_outputs
            and signature == self._last_published_map_signature
        ):
            self.map_output_skip_count += 1
            return False

        self._publish_current_map_outputs(header)
        self._last_published_map_signature = signature
        self.map_output_publish_count += 1
        return True

    def _publish_current_map_outputs(self, header: Header) -> None:
        all_points_map = None
        point_semantics = None
        global_window = None
        global_revision = None
        if self.publish_map_pointcloud:
            all_points_cv = self._get_map_points_cv()
            # The VO map may contain millions of accumulated points.  The
            # debug cloud publishes at most ``pointcloud_max_points``, so keep
            # the established finite/global-linspace sample but take it in
            # the source frame before the O(N) coordinate transform.  A
            # finite rigid transform preserves both validity and ordering,
            # making the published sample equivalent while bounding work.
            sample_indices = finite_point_sample_indices(
                all_points_cv, self.pointcloud_max_points
            )
            all_points_cv = all_points_cv[sample_indices]
            all_points_map = points_opencv_to_map(
                all_points_cv,
                self.rotation_map_world,
                self.map_translation_offset,
            )
        with self._map_lock:
            if all_points_map is not None:
                point_semantics = self.grid.semantic_ids_at(all_points_map)
            if self.dense_map_window_mode == "fixed":
                window_center_x = self.fixed_map_center_x
                window_center_y = self.fixed_map_center_y
            else:
                window_center_x = float(self.last_pose_map[0, 3])
                window_center_y = float(self.last_pose_map[1, 3])
            local_window = self.grid.extract_window(
                center_x=window_center_x,
                center_y=window_center_y,
                length_x=self.local_map_length_x,
                length_y=self.local_map_length_y,
            )
            current_revision = int(self.map_revision)
            need_global_grid_map = (
                self.global_grid_map_pub is not None
                and current_revision
                != self._last_published_global_grid_map_revision
            )
            need_dense_grid_save = (
                self.dense_global_grid_map_writer is not None
                and current_revision
                != self._last_enqueued_dense_grid_map_revision
            )
            if need_global_grid_map or need_dense_grid_save:
                global_window = self.grid.extract_complete_window()
                global_revision = current_revision
            if (
                self.publish_incremental_map
                and self.incremental_publish_mode == "full_tiles"
            ):
                # Payload windows are materialised one at a time by the
                # transfer timer from a frozen copy-on-write snapshot.
                dirty_keys = ()
            else:
                dirty_keys = self.grid.pending_dirty_tile_keys()

        self._publish_map_outputs(
            header=header,
            all_points_map=all_points_map,
            point_semantics=point_semantics,
            local_window=local_window,
            global_window=global_window,
            global_revision=global_revision,
        )
        if dirty_keys:
            with self._map_lock:
                self.grid.mark_tiles_published(dirty_keys)

    def _publish_pose(
        self,
        pose_cv: np.ndarray,
        pose_map: np.ndarray,
        pose_odom: np.ndarray,
        header: Header,
        *,
        pose_index: int,
        geometry_corrected: bool,
        trajectory_stamp_ns: Optional[int] = None,
        pose_covariance: Optional[np.ndarray] = None,
        velocity_body: Optional[np.ndarray] = None,
    ) -> None:
        output = self.odometry_output.update(
            timestamp=_stamp_seconds(header.stamp),
            pose_map_camera=pose_odom,
            velocity_camera=velocity_body,
            pose_covariance_camera=pose_covariance,
            reset_velocity_history=geometry_corrected,
        )
        pose_odom_child = output.pose_map_child
        pose_map_child = self.odometry_output.transform_pose(pose_map)
        transform_map_odom = pose_map_child @ inverse_pose(pose_odom_child)
        self.last_transform_map_odom = transform_map_odom.copy()
        odometry = make_odometry_message(
            header=header,
            map_frame=self.odom_frame,
            child_frame=self.child_frame,
            output=output,
        )
        self.odometry_pub.publish(odometry)
        if self.odometry_compat_pub is not None:
            self.odometry_compat_pub.publish(odometry)

        self.pose_stamps[int(pose_index)] = (
            int(header.stamp.sec),
            int(header.stamp.nanosec),
        )
        file_stamp_ns = (
            _stamp_nanoseconds_value(header.stamp)
            if trajectory_stamp_ns is None
            else int(trajectory_stamp_ns)
        )
        self.trajectory_file_stamps[int(pose_index)] = (
            int(file_stamp_ns // 1_000_000_000),
            int(file_stamp_ns % 1_000_000_000),
        )
        if geometry_corrected:
            try:
                self._rebuild_path_and_trajectory(header)
            except Exception as exception:
                self.get_logger().error(
                    f"Corrected Path publication failed: {exception}"
                )
        else:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = header.stamp
            pose_stamped.header.frame_id = self.map_frame
            pose_stamped.pose.position.x = float(pose_map_child[0, 3])
            pose_stamped.pose.position.y = float(pose_map_child[1, 3])
            pose_stamped.pose.position.z = float(pose_map_child[2, 3])
            path_quaternion = matrix_to_quaternion_xyzw(
                pose_map_child[:3, :3]
            )
            pose_stamped.pose.orientation.x = float(path_quaternion[0])
            pose_stamped.pose.orientation.y = float(path_quaternion[1])
            pose_stamped.pose.orientation.z = float(path_quaternion[2])
            pose_stamped.pose.orientation.w = float(path_quaternion[3])
            self.path_message.header.stamp = header.stamp
            self.path_message.header.frame_id = self.map_frame
            self.path_message.poses.append(pose_stamped)
            if (
                self.path_max_length > 0
                and len(self.path_message.poses) > self.path_max_length
            ):
                self.path_message.poses = self.path_message.poses[
                    -self.path_max_length:
                ]
            try:
                self._publish_path_if_due()
            except Exception as exception:
                # Path is a derived debug product. Its transport must not
                # suppress this frame's formal map transaction; the complete
                # accumulated path will be retried on a later stride/finalize.
                self.get_logger().error(
                    f"Debug Path publication failed: {exception}"
                )

        if self.tf_broadcaster is not None:
            transform_map_odom_message = make_transform_stamped_from_matrix(
                header=header,
                parent_frame=self.map_frame,
                child_frame=self.odom_frame,
                transform_parent_child=transform_map_odom,
            )
            transform_odom_child = make_transform_stamped_message(odometry)
            self.tf_broadcaster.sendTransform(
                [transform_map_odom_message, transform_odom_child]
            )

        if not geometry_corrected:
            stamp_sec, stamp_nanosec = self.trajectory_file_stamps[
                int(pose_index)
            ]
            file_stamp = (stamp_sec, stamp_nanosec)
            self._write_tum(self.tum_cv_file, pose_cv, file_stamp)
            self._write_tum(
                self.tum_map_file, pose_map_child, file_stamp
            )

    def _publish_path_if_due(self, *, force: bool = False) -> bool:
        """Publish the complete debug path without affecting Odometry."""
        if not self.path_message.poses:
            return False
        if not force:
            stride_boundary = (
                self.frame_count == 1
                or self.frame_count % self.path_publish_stride == 0
            )
            if (
                not stride_boundary
                or self.last_path_publish_frame_count == self.frame_count
            ):
                return False
        self.path_pub.publish(self.path_message)
        self.last_path_publish_frame_count = self.frame_count
        return True

    def _rebuild_path_and_trajectory(
        self,
        header: Header,
        *,
        publish: bool = True,
    ) -> None:
        if self.localization_mode == "vo":
            getter = getattr(self.slam, "get_pose_history_snapshot", None)
            if not callable(getter):
                raise RuntimeError(
                    "The configured SLAM class does not expose "
                    "get_pose_history_snapshot()"
                )
            _, pose_sequence = getter()
            poses_cv = {
                int(index): np.asarray(pose, dtype=np.float64)
                for index, pose in enumerate(pose_sequence)
            }
        else:
            poses_cv = {
                int(index): np.asarray(pose, dtype=np.float64).copy()
                for index, pose in self.localization_pose_history.items()
            }

        corrected_path = NavPath()
        corrected_path.header.stamp = header.stamp
        corrected_path.header.frame_id = self.map_frame
        for pose_index in sorted(poses_cv):
            pose_cv = poses_cv[pose_index]
            stamp = self.pose_stamps.get(pose_index)
            if stamp is None:
                continue
            pose_map_camera = pose_opencv_to_map(
                pose_cv,
                self.rotation_map_world,
                self.map_translation_offset,
            )
            pose_map = self.odometry_output.transform_pose(
                pose_map_camera
            )
            quaternion = matrix_to_quaternion_xyzw(pose_map[:3, :3])
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.map_frame
            pose_stamped.header.stamp.sec = int(stamp[0])
            pose_stamped.header.stamp.nanosec = int(stamp[1])
            pose_stamped.pose.position.x = float(pose_map[0, 3])
            pose_stamped.pose.position.y = float(pose_map[1, 3])
            pose_stamped.pose.position.z = float(pose_map[2, 3])
            pose_stamped.pose.orientation.x = float(quaternion[0])
            pose_stamped.pose.orientation.y = float(quaternion[1])
            pose_stamped.pose.orientation.z = float(quaternion[2])
            pose_stamped.pose.orientation.w = float(quaternion[3])
            corrected_path.poses.append(pose_stamped)

        if (
            self.path_max_length > 0
            and len(corrected_path.poses) > self.path_max_length
        ):
            corrected_path.poses = corrected_path.poses[-self.path_max_length:]
        self.path_message = corrected_path
        if publish:
            self._publish_path_if_due(force=True)
        self._rewrite_trajectory_files(poses_cv)

    def _rewrite_trajectory_files(self, poses_cv) -> None:
        for file_object in (self.tum_cv_file, self.tum_map_file):
            file_object.seek(0)
            file_object.truncate(0)
            file_object.write(self._trajectory_header)

        for pose_index in sorted(self.pose_stamps):
            if isinstance(poses_cv, dict):
                pose_value = poses_cv.get(pose_index)
            else:
                pose_value = (
                    poses_cv[pose_index]
                    if 0 <= pose_index < len(poses_cv)
                    else None
                )
            if pose_value is None:
                continue
            stamp_sec, stamp_nanosec = self.trajectory_file_stamps.get(
                pose_index, self.pose_stamps[pose_index]
            )
            pose_cv = np.asarray(pose_value, dtype=np.float64)
            pose_map_camera = pose_opencv_to_map(
                pose_cv,
                self.rotation_map_world,
                self.map_translation_offset,
            )
            pose_map = self.odometry_output.transform_pose(
                pose_map_camera
            )
            self._write_tum_values(
                self.tum_cv_file,
                pose_cv,
                stamp_sec,
                stamp_nanosec,
                flush=False,
            )
            self._write_tum_values(
                self.tum_map_file,
                pose_map,
                stamp_sec,
                stamp_nanosec,
                flush=False,
            )
        self.tum_cv_file.flush()
        self.tum_map_file.flush()

    @staticmethod
    def _write_tum(file_object, pose: np.ndarray, stamp) -> None:
        if isinstance(stamp, tuple):
            stamp_sec, stamp_nanosec = stamp
        else:
            stamp_sec = int(stamp.sec)
            stamp_nanosec = int(stamp.nanosec)
        RomaSemanticMapperNode._write_tum_values(
            file_object,
            pose,
            int(stamp_sec),
            int(stamp_nanosec),
            flush=True,
        )

    @staticmethod
    def _write_tum_values(
        file_object,
        pose: np.ndarray,
        stamp_sec: int,
        stamp_nanosec: int,
        *,
        flush: bool,
    ) -> None:
        # SciPy 1.5 (the version shipped on the target Ubuntu 20.04 host)
        # requests a writable buffer here.  Localization estimates are
        # intentionally immutable, so pass an owned copy of the rotation.
        rotation = np.asarray(pose[:3, :3], dtype=np.float64).copy()
        quaternion = Rotation.from_matrix(rotation).as_quat()
        translation = pose[:3, 3]
        file_object.write(
            f"{int(stamp_sec)}.{int(stamp_nanosec):09d} "
            f"{translation[0]:.9f} {translation[1]:.9f} "
            f"{translation[2]:.9f} "
            f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
            f"{quaternion[2]:.9f} {quaternion[3]:.9f}\n"
        )
        if flush:
            file_object.flush()

    def _publish_map_outputs(
        self,
        *,
        header: Header,
        all_points_map: Optional[np.ndarray],
        point_semantics: Optional[np.ndarray],
        local_window,
        global_window=None,
        global_revision: Optional[int] = None,
    ) -> None:
        geometry = local_window.geometry

        map_header = Header()
        map_header.stamp = header.stamp
        map_header.frame_id = self.local_grid_map_frame
        transform_local_map = (
            inverse_pose(self.last_transform_map_odom)
            if self.local_grid_map_frame == self.odom_frame
            else None
        )
        global_header = Header()
        global_header.stamp = header.stamp
        global_header.frame_id = self.map_frame

        if getattr(self, "publish_local_occupancy_grid", True):
            if self.occupancy_pub is None:
                raise RuntimeError("local OccupancyGrid publisher is unavailable")
            occupancy_message = make_occupancy_grid_message(
                header=map_header,
                window=local_window,
            )
            self.occupancy_pub.publish(occupancy_message)

        if self.publish_layer_images:
            semantic = local_window.semantic_id_layer()
            elevation = local_window.elevation_layer()
            elevation_variance = local_window.elevation_variance_layer()
            # Human-viewing images are vertically flipped so map +Y appears
            # upward. These are optional views; GridMap and incremental map
            # below always retain every numerical layer.
            semantic_message = self.bridge.cv2_to_imgmsg(
                np.flipud(semantic),
                encoding="mono8",
            )
            semantic_message.header = map_header
            self.semantic_grid_pub.publish(semantic_message)

            semantic_color_message = self.bridge.cv2_to_imgmsg(
                np.flipud(local_window.semantic_color_bgr()),
                encoding="bgr8",
            )
            semantic_color_message.header = map_header
            self.semantic_color_pub.publish(semantic_color_message)

            elevation_message = self.bridge.cv2_to_imgmsg(
                np.flipud(elevation.astype(np.float32)),
                encoding="32FC1",
            )
            elevation_message.header = map_header
            self.elevation_pub.publish(elevation_message)

            elevation_variance_message = self.bridge.cv2_to_imgmsg(
                np.flipud(elevation_variance.astype(np.float32)),
                encoding="32FC1",
            )
            elevation_variance_message.header = map_header
            self.elevation_variance_pub.publish(
                elevation_variance_message
            )

            elevation_color_message = self.bridge.cv2_to_imgmsg(
                np.flipud(local_window.elevation_color_bgr()),
                encoding="bgr8",
            )
            elevation_color_message.header = map_header
            self.elevation_color_pub.publish(elevation_color_message)

        configured_layers = getattr(self, "local_grid_map_layers", None)
        layer_map = local_window.layers(configured_layers)
        basic_layers = (
            ["elevation"]
            if "elevation" in layer_map
            else [next(iter(layer_map))]
        )
        grid_map_message = make_grid_map_message(
            header=map_header,
            geometry=geometry,
            layers=layer_map,
            basic_layers=basic_layers,
            transform_output_map=transform_local_map,
        )
        self.grid_map_pub.publish(grid_map_message)
        if self.grid_map_odom_pub is not None:
            self.grid_map_odom_pub.publish(grid_map_message)

        if global_window is not None:
            if global_revision is None:
                raise RuntimeError(
                    "complete global GridMap publication state is invalid"
                )
            if (
                self.global_grid_map_pub is not None
                and global_revision
                != self._last_published_global_grid_map_revision
            ):
                global_layers = global_window.layers(
                    self.global_grid_map_layers
                )
                global_basic_layers = (
                    ["elevation"]
                    if "elevation" in global_layers
                    else [next(iter(global_layers))]
                )
                global_message = make_grid_map_message(
                    header=global_header,
                    geometry=global_window.geometry,
                    layers=global_layers,
                    basic_layers=global_basic_layers,
                )
                self.global_grid_map_pub.publish(global_message)
                self._last_published_global_grid_map_revision = int(
                    global_revision
                )
            dense_writer = self.dense_global_grid_map_writer
            if (
                dense_writer is not None
                and global_revision
                != self._last_enqueued_dense_grid_map_revision
            ):
                dense_writer.enqueue(
                    global_window,
                    map_revision=global_revision,
                    timestamp_text=_stamp_text(map_header.stamp),
                )
                self._last_enqueued_dense_grid_map_revision = int(
                    global_revision
                )

        if getattr(self, "publish_incremental_map", True):
            if self.incremental_publish_mode == "full_tiles":
                self._request_incremental_delta(map_header)
            else:
                # Fixed and rolling-window modes both preserve the original
                # complete-rectangle wire meaning. Only the centre differs.
                self._publish_incremental_window_snapshot(
                    map_header=map_header,
                    window=local_window,
                )

        if self.publish_map_pointcloud:
            if (
                self.pointcloud_pub is None
                or all_points_map is None
                or point_semantics is None
            ):
                raise RuntimeError("Map point-cloud inputs were not prepared")
            cloud = make_semantic_point_cloud(
                header=map_header,
                points_xyz=all_points_map,
                semantic_ids=point_semantics,
                max_points=self.pointcloud_max_points,
            )
            self.pointcloud_pub.publish(cloud)

    @staticmethod
    def _tile_transfer_plan(tile) -> TileTransferPlan:
        return TileTransferPlan(
            tile_x=int(tile.tile_x),
            tile_y=int(tile.tile_y),
            tile_revision=int(tile.tile_revision),
            cleared=bool(tile.cleared),
        )

    def _publish_map_batch_begin(
        self,
        map_header: Header,
        batch: MapTransferBatchPlan,
    ) -> None:
        if self.incremental_map_batch_pub is None:
            raise RuntimeError("full-tile batch publisher is unavailable")
        self.incremental_map_batch_pub.publish(
            make_map_batch_begin(header=map_header, plan=batch)
        )

    def _publish_map_batch_complete(self, map_header, batch) -> None:
        if self.incremental_map_batch_pub is None:
            raise RuntimeError("full-tile batch publisher is unavailable")
        self.incremental_map_batch_pub.publish(
            make_map_batch_complete(header=map_header, batch=batch)
        )

    def _publish_incremental_window_snapshot(
        self,
        *,
        map_header: Header,
        window,
    ) -> int:
        """Publish one complete rectangular map payload using the old schema."""
        if self.incremental_map_pub is None:
            raise RuntimeError("incremental map publisher is disabled")
        self.map_message_update_id += 1
        incremental = make_incremental_semantic_map_message(
            header=map_header,
            update_id=self.map_message_update_id,
            window=window,
        )
        self.incremental_map_pub.publish(incremental)
        return int(incremental.update_id)

    def _publish_global_overview_if_changed(self, *, force: bool = False) -> bool:
        """Publish one low-rate global occupancy overview when it changed."""

        if self.global_overview_pub is None or self.last_header is None:
            return False
        with self._map_lock:
            revision = int(self.map_revision)
            if (
                not force
                and revision == self._last_published_global_overview_revision
            ):
                return False
            geometry, occupancy = self.grid.extract_occupancy_overview(
                resolution=self.global_overview_resolution
            )
        header = Header()
        header.stamp = self.last_header.stamp
        header.frame_id = self.map_frame
        message = make_occupancy_grid_from_array(
            header=header,
            geometry=geometry,
            occupancy=occupancy,
        )
        self.global_overview_pub.publish(message)
        self._last_published_global_overview_revision = revision
        return True

    def _publish_compact_map_status(self) -> None:
        """Publish the newest complete revision committed to the live DB."""

        writer = self.compact_global_map_writer
        error = writer.last_error
        error_text = None if error is None else str(error)
        if error_text != self._last_compact_map_error_text:
            if error_text is not None:
                self.get_logger().error(
                    f"Live global-map database update failed: {error_text}"
                )
            elif self._last_compact_map_error_text is not None:
                self.get_logger().info(
                    "Live global-map database writer recovered"
                )
            self._last_compact_map_error_text = error_text
        revision = writer.persisted_revision
        if revision == self._last_published_compact_map_revision:
            return
        message = UInt64()
        message.data = int(revision)
        self.global_map_revision_pub.publish(message)
        self._last_published_compact_map_revision = revision

    def _on_get_grid_map(self, request, response):
        """Return one bounded high-resolution ROI from the live global map."""

        requested_frame = str(request.frame_id).strip()
        if requested_frame and requested_frame != self.map_frame:
            raise ValueError(
                f"GetGridMap frame must be {self.map_frame!r}, got "
                f"{requested_frame!r}"
            )
        length_x = float(request.length_x)
        length_y = float(request.length_y)
        if (
            not np.isfinite([length_x, length_y]).all()
            or length_x <= 0.0
            or length_y <= 0.0
        ):
            raise ValueError("GetGridMap lengths must be finite and positive")
        width = max(1, int(np.ceil(length_x / self.grid.resolution)))
        height = max(1, int(np.ceil(length_y / self.grid.resolution)))
        if width * height > self.grid_map_query_max_cells:
            raise ValueError(
                "GetGridMap request exceeds grid_map_query_max_cells: "
                f"{width * height} > {self.grid_map_query_max_cells}"
            )
        requested_layers = tuple(str(name) for name in request.layers)
        with self._map_lock:
            window = self.grid.extract_window(
                center_x=float(request.position_x),
                center_y=float(request.position_y),
                length_x=length_x,
                length_y=length_y,
            )
            layer_map = window.layers(
                requested_layers if requested_layers else None
            )
        header = Header()
        if self.last_header is not None:
            header.stamp = self.last_header.stamp
        else:
            header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame
        basic_layers = (
            ["elevation"]
            if "elevation" in layer_map
            else [next(iter(layer_map))]
        )
        response.map = make_grid_map_message(
            header=header,
            geometry=window.geometry,
            layers=layer_map,
            basic_layers=basic_layers,
        )
        return response

    def _on_save_service(self, request, response):
        del request
        try:
            paths = self.save_snapshot()
            response.success = True
            response.message = ", ".join(
                f"{name}={path}" for name, path in paths.items()
            )
        except Exception as exception:
            response.success = False
            response.message = str(exception)
        return response

    def save_snapshot(self):
        # A manual save is also a retry boundary: never label an older grid as
        # the latest snapshot while an immutable keyframe transaction is still
        # pending or corrected geometry is stale.
        self._synchronize_map_state(
            header=self.last_header or Header(),
            publish=rclpy.ok(context=self.context),
            wait_for_correction=True,
        )
        if self.last_stamp is None:
            timestamp_text = "no_frames"
        else:
            timestamp_text = _stamp_text(self.last_stamp)
        with self._map_lock:
            paths = self.grid.save_snapshot(
                self.snapshot_dir,
                timestamp_text=timestamp_text,
                frame_id=self.map_frame,
                parallel_workers=self.map_persistence_workers,
            )
            compact_revision = int(self.map_revision)
            dense_window = (
                self.grid.extract_complete_window()
                if self.dense_global_grid_map_writer is not None
                else None
            )
            self.compact_global_map_writer.enqueue(
                self.grid.clone_for_update(),
                map_revision=compact_revision,
                timestamp_text=timestamp_text,
            )
        self.compact_global_map_writer.flush(compact_revision)
        paths["compact_global_map"] = self.compact_global_map_path
        dense_writer = self.dense_global_grid_map_writer
        if dense_writer is not None and dense_window is not None:
            dense_writer.enqueue(
                dense_window,
                map_revision=compact_revision,
                timestamp_text=timestamp_text,
            )
            self._last_enqueued_dense_grid_map_revision = max(
                self._last_enqueued_dense_grid_map_revision,
                compact_revision,
            )
            dense_writer.flush(compact_revision)
            paths["dense_global_grid_map"] = self.dense_global_grid_map_path
        if rclpy.ok(context=self.context):
            self.get_logger().info(
                "Saved GridMap snapshot: "
                + ", ".join(str(path) for path in paths.values())
            )
        return paths

    def close_resources(self) -> None:
        can_publish = rclpy.ok(context=self.context)
        if self.mprf_client is not None:
            try:
                self.mprf_client.close(
                    join_timeout_sec=min(
                        self.mprf_client.configuration.request_timeout_sec
                        + 1.0,
                        30.0,
                    )
                )
                self._poll_mprf_loop_results()
            except Exception as exception:
                self.get_logger().warning(
                    f"MPRF shutdown result could not be applied: {exception}"
                )
        try:
            shutdown = getattr(self.slam, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as exception:
            self.get_logger().warning(f"SLAM shutdown warning: {exception}")

        # shutdown() may commit a final queued loop result. Rebuild before the
        # snapshot so disk output never predates the final SLAM geometry or a
        # staged keyframe observation whose first transaction attempt failed.
        try:
            geometry_revision, _ = self._get_keyframe_pose_state()
            geometry_changed = (
                geometry_revision != self.applied_geometry_revision
            )
            with self._map_lock:
                map_work_pending = (
                    bool(self.pending_map_observations)
                    or bool(self.map_is_stale)
                    or geometry_changed
                )
            publication_pending = (
                self.map_revision > 0
                and self._map_output_signature()
                != self._last_published_map_signature
            )
            if map_work_pending or (publication_pending and can_publish):
                rebuild_header = self.last_header or Header()
                self._synchronize_map_state(
                    header=rebuild_header,
                    publish=can_publish,
                    wait_for_correction=True,
                )
            if geometry_changed and self.last_header is not None:
                self._rebuild_path_and_trajectory(
                    self.last_header,
                    publish=can_publish,
                )
        except Exception as exception:
            self.get_logger().warning(
                f"Could not apply final GridMap correction: {exception}"
            )

        if (
            can_publish
            and self.path_message.poses
            and self.last_path_publish_frame_count != self.frame_count
        ):
            try:
                self._publish_path_if_due(force=True)
            except Exception as exception:
                self.get_logger().warning(
                    f"Could not publish final complete Path: {exception}"
                )

        if self.save_on_shutdown and self.map_revision > 0:
            try:
                self.save_snapshot()
            except Exception as exception:
                self.get_logger().warning(
                    f"Could not save final GridMap snapshot: {exception}"
                )

        if rclpy.ok(context=self.context):
            self.get_logger().info(
                "Map output publication summary: "
                f"map_revisions={self.map_revision}, "
                f"complete_publications={self.map_output_publish_count}, "
                f"unchanged_skips={self.map_output_skip_count}."
            )

        input_summary_path = self.output_dir / (
            "input_processing_summary.json"
        )
        try:
            input_summary = self.input_monitor.summary(
                processing_in_progress=self.processing,
                valid_pose_count=self.frame_count,
                tracking_or_processing_failure_count=(
                    self.tracking_failure_count
                ),
                map_revision=self.map_revision,
            )
            input_summary["input_configuration"] = {
                "left_topic": str(
                    self.get_parameter("left_topic").value
                ),
                "right_topic": str(
                    self.get_parameter("right_topic").value
                ),
                "semantic_topic": str(
                    self.get_parameter("semantic_topic").value
                ),
                "input_processed_topic": str(
                    self.get_parameter("input_processed_topic").value
                ),
                "input_processed_enabled": (
                    self.input_processed_pub is not None
                ),
                "synchronization": (
                    self.input_transport.sync_description
                ),
                "stereo_reliability": (
                    self.input_transport.stereo_reliability
                ),
                "semantic_reliability": (
                    self.input_transport.semantic_reliability
                ),
                "image_qos_depth": (
                    self.input_transport.image_qos_depth
                ),
                "imu_enabled": self.enable_imu,
                "image_timestamp_source": self.image_timestamp_source,
                "ros_time_is_active": bool(
                    getattr(self.get_clock(), "ros_time_is_active", False)
                ),
                "trajectory_file_timestamp_source": (
                    "input Header.stamp before optional arrival rewrite"
                ),
            }
            input_summary["roma_inference"] = {
                "coarse_res": int(
                    self.get_parameter("coarse_res").value
                ),
                "upsample_res": int(
                    self.get_parameter("upsample_res").value
                ),
                "stereo_upsample_refinement": bool(
                    self.get_parameter(
                        "stereo_upsample_refinement"
                    ).value
                ),
                "temporal_upsample_refinement": bool(
                    self.get_parameter(
                        "temporal_upsample_refinement"
                    ).value
                ),
                "symmetric_matching": bool(
                    self.get_parameter("symmetric_matching").value
                ),
                "use_custom_corr": bool(
                    self.get_parameter("use_custom_corr").value
                ),
                "enable_feature_reuse": bool(
                    self.get_parameter("enable_feature_reuse").value
                ),
                "batch_coarse_pairs": bool(
                    self.get_parameter("batch_coarse_pairs").value
                ),
                "feature_reuse_active": bool(
                    getattr(self.slam, "feature_reuse_active", False)
                ),
                "enable_klt_temporal_tracking": bool(
                    self.get_parameter(
                        "enable_klt_temporal_tracking"
                    ).value
                ),
                "temporal_tracking": {
                    "last_source": str(getattr(
                        self.slam,
                        "last_temporal_tracking_source",
                        "none",
                    )),
                    "klt_attempts": int(getattr(
                        self.slam, "temporal_klt_attempts", 0
                    )),
                    "klt_candidates": int(getattr(
                        self.slam, "temporal_klt_candidates", 0
                    )),
                    "klt_successes": int(getattr(
                        self.slam, "temporal_klt_successes", 0
                    )),
                    "roma_fallbacks": int(getattr(
                        self.slam, "temporal_roma_fallbacks", 0
                    )),
                },
            }
            input_summary["global_loop"] = {
                "enabled": bool(self.enable_global_loop_closure),
                "candidate_source": self.global_loop_candidate_source,
                "error_count": int(self._global_loop_error_count),
                "graph_node_count": (
                    int(self.global_loop_closure.graph.node_count)
                    if self.global_loop_closure is not None
                    else 0
                ),
                "accepted_loop_factor_count": (
                    int(self.global_loop_closure.graph.loop_count)
                    if self.global_loop_closure is not None
                    else 0
                ),
                "mprf": (
                    {
                        "enabled": bool(
                            self.mprf_client.configuration.enabled
                        ),
                        "submitted": int(self.mprf_client.submitted_count),
                        "dropped": int(self.mprf_client.dropped_count),
                        "completed": int(self.mprf_client.completed_count),
                        "failed": int(self.mprf_client.failed_count),
                    }
                    if self.mprf_client is not None
                    else None
                ),
            }
            self.input_monitor.write_summary(
                input_summary_path, input_summary
            )
            if rclpy.ok(context=self.context):
                self.get_logger().info(
                    "Input processing summary: "
                    f"synchronized="
                    f"{input_summary['synchronized_callbacks']}, "
                    f"terminated={input_summary['terminal_callbacks']}, "
                    f"completion_notifications="
                    f"{input_summary['completion_notifications_published']}, "
                    f"path={input_summary_path}."
                )
        except Exception as exception:
            self.get_logger().warning(
                "Could not write input processing summary "
                f"{input_summary_path}: {exception}"
            )

        terrain_mapper = getattr(self, "terrain_pointcloud_mapper", None)
        if terrain_mapper is not None:
            try:
                terrain_mapper.close()
            except Exception as exception:
                self.get_logger().warning(
                    "Could not stop point-cloud preprocessing worker "
                    f"cleanly: {exception}"
                )

        for file_object in [
            getattr(self, "tum_cv_file", None),
            getattr(self, "tum_map_file", None),
        ]:
            if file_object is not None and not file_object.closed:
                file_object.close()

        compact_writer = getattr(self, "compact_global_map_writer", None)
        if compact_writer is not None:
            try:
                compact_writer.close(flush_revision=self.map_revision)
            except Exception as exception:
                self.get_logger().warning(
                    "Could not stop live global-map writer cleanly: "
                    f"{exception}"
                )

        dense_writer = getattr(self, "dense_global_grid_map_writer", None)
        if dense_writer is not None:
            try:
                dense_writer.close(flush_revision=self.map_revision)
            except Exception as exception:
                self.get_logger().warning(
                    "Could not stop live dense GridMap writer cleanly: "
                    f"{exception}"
                )

        correction_worker = getattr(self, "map_correction_worker", None)
        if correction_worker is not None:
            try:
                correction_worker.shutdown()
            except Exception as exception:
                self.get_logger().warning(
                    "Could not stop map-correction worker cleanly: "
                    f"{exception}"
                )

        observation_store = getattr(self, "map_observations", None)
        if observation_store is not None:
            try:
                archive_path = observation_store.archive_path
                observation_count = len(observation_store)
                observation_store.close()
                if (
                    archive_path is not None
                    and rclpy.ok(context=self.context)
                ):
                    self.get_logger().info(
                        "Closed lossless observation archive: "
                        f"{archive_path} ({observation_count} keyframes, "
                        f"{observation_store.archive_size_bytes / (1024 ** 2):.1f} MiB)"
                    )
            except Exception as exception:
                self.get_logger().warning(
                    f"Could not close observation archive cleanly: {exception}"
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RomaSemanticMapperNode] = None
    executor = None
    try:
        node = RomaSemanticMapperNode()
        requires_external_callbacks = (
            node.localization_configuration.requires_external_odometry
        )
        executor_threads = 1 + sum(
            (
                int(node.enable_imu),
                int(node.enable_lidar_mapping),
                int(node.enable_pointcloud_fusion),
                int(requires_external_callbacks),
            )
        )
        if executor_threads > 1:
            executor = MultiThreadedExecutor(
                num_threads=executor_threads
            )
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.close_resources()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
