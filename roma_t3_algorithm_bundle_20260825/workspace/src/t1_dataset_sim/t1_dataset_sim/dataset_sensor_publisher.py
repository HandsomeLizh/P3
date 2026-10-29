from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import Header

from .dataset_formats import (
    ORGANIZED_FORMAT,
    StereoPairingStats,
    StereoRectifier,
    default_dataset_paths,
    detect_dataset_format,
    load_calibration,
    load_organized_imu_records,
    load_organized_stereo_records,
)
from .message_utils import (
    image_to_message,
    make_camera_info,
    make_imu_message,
    nanoseconds_to_time,
    seconds_to_nanoseconds,
)
from .lidar_io import (
    OrganizedLidarRecord,
    load_organized_lidar_records,
    read_binary_pcd_message,
)
from .fastlio_io import (
    OrganizedFastLioOdometryRecord,
    load_organized_fastlio_odometry,
)
from .playback_monitor import PlaybackMonitor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FrameRecord:
    timestamp_ns: int
    right_timestamp_ns: int
    left_path: Path
    right_path: Path
    semantic_path: Optional[Path]


@dataclass(frozen=True)
class ImuRecord:
    timestamp_ns: int
    gyro_rad_s: np.ndarray


@dataclass(frozen=True)
class Event:
    timestamp_ns: int
    priority: int
    kind: str
    index: int


def _natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _timestamp_from_image_name(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exception:
        raise ValueError(
            f"Image filename must be an integer nanosecond timestamp: {path.name}"
        ) from exception


def _discover_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=_natural_key)


def _find_column(columns, required_substrings):
    lowered = {str(column).strip().lower(): column for column in columns}
    for key, original in lowered.items():
        if all(part.lower() in key for part in required_substrings):
            return original
    raise KeyError(
        f"Could not find a column containing {required_substrings}; columns={list(columns)}"
    )


class DatasetSensorPublisher(Node):
    """Publish dataset images and IMU as independent ROS 2 topics.

    Both the original left/right/Measure.xlsx layout and Luna's
    batch_organized layout are supported.  Playback rate only changes
    wall-clock delivery speed; it never rescales sensor Header.stamp values.
    """

    def __init__(self) -> None:
        super().__init__("dataset_sensor_publisher")

        home = Path.home()
        default_data = home / "RoMa" / "data" / "data1223"

        self.declare_parameter("data_dir", str(default_data))
        self.declare_parameter("dataset_format", "auto")
        self.declare_parameter("imu_path", "")
        self.declare_parameter("calib_path", "")
        self.declare_parameter("semantic_dir", "")
        self.declare_parameter("use_precomputed_semantics", False)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("max_stereo_skew_ms", 20.0)
        self.declare_parameter("rectify_stereo", True)
        self.declare_parameter("playback_rate", 1.0)
        self.declare_parameter("start_delay_sec", 8.0)
        self.declare_parameter("stereo_publish_gap_sec", 0.05)
        self.declare_parameter("loop", False)
        self.declare_parameter("lossless_playback", False)
        self.declare_parameter("lossless_max_in_flight_frames", 2)
        self.declare_parameter("processing_ack_timeout_sec", 120.0)
        self.declare_parameter("playback_summary_path", "")
        self.declare_parameter("publish_imu", True)
        self.declare_parameter("publish_lidar", False)
        self.declare_parameter("lidar_publish_stride", 1)
        self.declare_parameter("lidar_publish_max_points", 0)
        self.declare_parameter("publish_lio_odometry", False)
        self.declare_parameter("lio_publish_stride", 1)
        self.declare_parameter("lio_trajectory_path", "")

        self.declare_parameter(
            "left_topic", "/Car/T1/fusion/stereo/left/image_raw"
        )
        self.declare_parameter(
            "right_topic", "/Car/T1/fusion/stereo/right/image_raw"
        )
        self.declare_parameter(
            "left_info_topic", "/Car/T1/fusion/stereo/left/camera_info"
        )
        self.declare_parameter(
            "right_info_topic", "/Car/T1/fusion/stereo/right/camera_info"
        )
        self.declare_parameter("imu_topic", "/Car/T1/fusion/imu/data")
        self.declare_parameter(
            "lidar_topic", "/Car/T1/fusion/lidar/points"
        )
        self.declare_parameter(
            "semantic_topic", "/Car/T2/semantic/label_image"
        )
        self.declare_parameter(
            "input_processed_topic", "/Car/T3/debug/input_processed"
        )
        self.declare_parameter("lio_odometry_topic", "/Odometry")

        self.declare_parameter(
            "left_frame_id", "car_stereo_left_optical_frame"
        )
        self.declare_parameter(
            "right_frame_id", "car_stereo_right_optical_frame"
        )
        self.declare_parameter("imu_frame_id", "car_imu_body_frame")
        self.declare_parameter("lidar_frame_id", "velodyne")
        self.declare_parameter("lio_parent_frame_id", "camera_init")
        self.declare_parameter("lio_child_frame_id", "body")

        self.data_dir = Path(
            str(self.get_parameter("data_dir").value)
        ).expanduser()
        self.dataset_format = detect_dataset_format(
            self.data_dir,
            str(self.get_parameter("dataset_format").value),
        )
        default_paths = default_dataset_paths(
            self.data_dir, self.dataset_format
        )

        def resolved_path(parameter_name: str, default_key: str) -> Path:
            configured = str(self.get_parameter(parameter_name).value).strip()
            if configured:
                return Path(configured).expanduser()
            return default_paths[default_key]

        self.imu_path = resolved_path("imu_path", "imu")
        self.calib_path = resolved_path("calib_path", "calibration")
        self.semantic_dir = resolved_path("semantic_dir", "semantics")

        self.use_precomputed = bool(
            self.get_parameter("use_precomputed_semantics").value
        )
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.max_stereo_skew_ns = int(round(
            float(self.get_parameter("max_stereo_skew_ms").value) * 1e6
        ))
        self.rectify_stereo = bool(
            self.get_parameter("rectify_stereo").value
        )
        self.playback_rate = float(self.get_parameter("playback_rate").value)
        self.start_delay_sec = float(
            self.get_parameter("start_delay_sec").value
        )
        self.stereo_gap = max(
            0.0, float(self.get_parameter("stereo_publish_gap_sec").value)
        )
        self.loop_enabled = bool(self.get_parameter("loop").value)
        self.lossless_playback = bool(
            self.get_parameter("lossless_playback").value
        )
        self.lossless_max_in_flight_frames = int(
            self.get_parameter("lossless_max_in_flight_frames").value
        )
        self.processing_ack_timeout_sec = float(
            self.get_parameter("processing_ack_timeout_sec").value
        )
        summary_path = str(
            self.get_parameter("playback_summary_path").value
        ).strip()
        self.playback_summary_path = (
            Path(summary_path).expanduser() if summary_path else None
        )
        self.publish_imu = bool(
            self.get_parameter("publish_imu").value
        )
        self.publish_lidar = bool(
            self.get_parameter("publish_lidar").value
        )
        self.lidar_publish_stride = int(
            self.get_parameter("lidar_publish_stride").value
        )
        self.lidar_publish_max_points = int(
            self.get_parameter("lidar_publish_max_points").value
        )
        self.publish_lio_odometry = bool(
            self.get_parameter("publish_lio_odometry").value
        )
        self.lio_publish_stride = int(
            self.get_parameter("lio_publish_stride").value
        )

        if self.playback_rate <= 0.0:
            raise ValueError("playback_rate must be positive")
        if self.max_frames < 0:
            raise ValueError("max_frames cannot be negative")
        if self.processing_ack_timeout_sec <= 0.0:
            raise ValueError("processing_ack_timeout_sec must be positive")
        if self.lossless_max_in_flight_frames <= 0:
            raise ValueError(
                "lossless_max_in_flight_frames must be positive"
            )
        if self.lossless_playback and self.loop_enabled:
            raise ValueError(
                "lossless_playback cannot be combined with loop because "
                "repeated acquisition timestamps make completion ambiguous"
            )
        if self.lidar_publish_stride <= 0:
            raise ValueError("lidar_publish_stride must be positive")
        if self.lidar_publish_max_points < 0:
            raise ValueError("lidar_publish_max_points cannot be negative")
        if self.lio_publish_stride <= 0:
            raise ValueError("lio_publish_stride must be positive")

        self.left_frame_id = str(self.get_parameter("left_frame_id").value)
        self.right_frame_id = str(self.get_parameter("right_frame_id").value)
        self.imu_frame_id = str(self.get_parameter("imu_frame_id").value)
        self.lidar_frame_id = str(
            self.get_parameter("lidar_frame_id").value
        )
        self.lio_parent_frame_id = str(
            self.get_parameter("lio_parent_frame_id").value
        )
        self.lio_child_frame_id = str(
            self.get_parameter("lio_child_frame_id").value
        )

        self.frames, self.input_selection_summary = self._load_frames()
        self.imu_records = self._load_imu() if self.publish_imu else []
        self.lidar_records = self._load_lidar()
        self.lio_odometry_records = self._load_lio_odometry()
        self.calibration = load_calibration(self.calib_path)
        self.rectifier = StereoRectifier(
            self.calibration, enabled=self.rectify_stereo
        )
        self.events = self._build_events()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.mapper_input_topics = (
            str(self.get_parameter("left_topic").value),
            str(self.get_parameter("right_topic").value),
            str(self.get_parameter("semantic_topic").value),
        )
        self.input_processed_topic = str(
            self.get_parameter("input_processed_topic").value
        )

        self.left_pub = self.create_publisher(
            Image, self.mapper_input_topics[0], qos
        )
        self.right_pub = self.create_publisher(
            Image, self.mapper_input_topics[1], qos
        )
        self.left_info_pub = self.create_publisher(
            CameraInfo, str(self.get_parameter("left_info_topic").value), qos
        )
        self.right_info_pub = self.create_publisher(
            CameraInfo, str(self.get_parameter("right_info_topic").value), qos
        )
        self.imu_pub = (
            self.create_publisher(
                Imu, str(self.get_parameter("imu_topic").value), qos
            )
            if self.publish_imu
            else None
        )
        lidar_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.lidar_pub = (
            self.create_publisher(
                PointCloud2,
                str(self.get_parameter("lidar_topic").value),
                lidar_qos,
            )
            if self.publish_lidar
            else None
        )
        self.lio_odometry_pub = (
            self.create_publisher(
                Odometry,
                str(self.get_parameter("lio_odometry_topic").value),
                lidar_qos,
            )
            if self.publish_lio_odometry
            else None
        )
        self.semantic_pub = self.create_publisher(
            Image, self.mapper_input_topics[2], qos
        )
        self.input_processed_subscription = self.create_subscription(
            Header,
            self.input_processed_topic,
            self._on_input_processed,
            qos,
        )

        self.event_index = 0
        self.play_started = False
        self.finished = False
        self.pending_frame_acks = deque()
        self.node_start_wall = time.monotonic()
        self.mapper_readiness_wait_wall = None
        self.mapper_readiness_wait_logged = False
        self.mapper_ready_confirmed = False
        self.mapper_readiness_wait_sec = None
        self.play_start_wall = 0.0
        self.sensor_start_ns = self.events[0].timestamp_ns
        self.playback_monitor = PlaybackMonitor(
            expected_frame_count=len(self.frames),
            expected_first_stamp_ns=self.frames[0].timestamp_ns,
            expected_last_stamp_ns=self.frames[-1].timestamp_ns,
            requested_playback_rate=self.playback_rate,
            lossless_playback=self.lossless_playback,
        )

        self.timer = self.create_timer(0.002, self._on_timer)

        self.get_logger().info(
            "Dataset live publisher is ready.\n"
            f"  dataset format: {self.dataset_format}\n"
            f"  selected stereo frames: {len(self.frames)}\n"
            f"  stereo-skew eligible/shared pairs: "
            f"{self.input_selection_summary['stereo_skew_eligible_pair_count']}"
            "/"
            f"{self.input_selection_summary['shared_candidate_pair_count']}\n"
            f"  publish raw IMU: {self.publish_imu}\n"
            f"  IMU samples: {len(self.imu_records)}\n"
            f"  raw LiDAR frames: {len(self.lidar_records)}\n"
            f"  saved FAST-LIO odometry: "
            f"{len(self.lio_odometry_records)}\n"
            f"  sensor time: {self.events[0].timestamp_ns / 1e9:.9f} -> "
            f"{self.events[-1].timestamp_ns / 1e9:.9f}\n"
            f"  playback rate: {self.playback_rate}x\n"
            f"  lossless playback: {self.lossless_playback}\n"
            f"  lossless max in-flight image frames: "
            f"{self.lossless_max_in_flight_frames}\n"
            f"  processing acknowledgement timeout: "
            f"{self.processing_ack_timeout_sec}s\n"
            f"  start delay: {self.start_delay_sec}s\n"
            f"  stereo rectification: {self.rectifier.enabled}\n"
            f"  precomputed semantics: {self.use_precomputed}\n"
            f"  publish raw LiDAR: {self.publish_lidar}\n"
            f"  publish saved FAST-LIO odometry: "
            f"{self.publish_lio_odometry} (interface test only)\n"
            f"  LiDAR playback point cap: "
            f"{self.lidar_publish_max_points or 'complete cloud'}"
        )

    def _load_frames(self) -> Tuple[List[FrameRecord], dict]:
        if self.dataset_format == ORGANIZED_FORMAT:
            return self._load_organized_frames()

        left_paths = _discover_images(self.data_dir / "left")
        right_paths = _discover_images(self.data_dir / "right")

        right_by_timestamp = {
            _timestamp_from_image_name(path): path for path in right_paths
        }

        max_frames = self.max_frames
        left_timestamps = {
            _timestamp_from_image_name(path) for path in left_paths
        }
        right_timestamps = set(right_by_timestamp)
        shared_timestamps = left_timestamps.intersection(right_timestamps)
        records: List[FrameRecord] = []

        for left_path in left_paths:
            timestamp_ns = _timestamp_from_image_name(left_path)
            right_path = right_by_timestamp.get(timestamp_ns)
            if right_path is None:
                self.get_logger().warning(
                    f"No right image for left timestamp {timestamp_ns}"
                )
                continue

            semantic_path = None
            if self.use_precomputed:
                candidate = self.semantic_dir / left_path.name
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"Precomputed semantic label is missing: {candidate}"
                    )
                semantic_path = candidate

            records.append(
                FrameRecord(
                    timestamp_ns=timestamp_ns,
                    right_timestamp_ns=timestamp_ns,
                    left_path=left_path,
                    right_path=right_path,
                    semantic_path=semantic_path,
                )
            )
            if max_frames > 0 and len(records) >= max_frames:
                break

        if not records:
            raise RuntimeError("No valid stereo frame pairs were found")
        stats = StereoPairingStats(
            left_rows=len(left_paths),
            right_rows=len(right_paths),
            shared_rows=len(shared_timestamps),
            eligible_rows=len(shared_timestamps),
            paired_rows=len(records),
            skipped_for_skew=0,
            skipped_for_max_frames=(
                len(shared_timestamps) - len(records)
            ),
            missing_left=len(right_timestamps.difference(left_timestamps)),
            missing_right=len(left_timestamps.difference(right_timestamps)),
            max_abs_skew_ns=0,
        )
        return records, stats.selection_summary(
            pairing_method="exact_filename_timestamp",
            configured_max_abs_skew_ns=self.max_stereo_skew_ns,
            max_frames=max_frames,
        )

    def _load_organized_frames(self) -> Tuple[List[FrameRecord], dict]:
        max_frames = self.max_frames
        sources, stats = load_organized_stereo_records(
            self.data_dir,
            max_abs_skew_ns=self.max_stereo_skew_ns,
            max_frames=max_frames,
        )
        if stats.skipped_for_skew:
            self.get_logger().warning(
                "Skipped "
                f"{stats.skipped_for_skew} stereo pairs exceeding "
                f"{self.max_stereo_skew_ns / 1e6:.3f} ms; observed max "
                f"skew={stats.max_abs_skew_ns / 1e6:.3f} ms."
            )
        if stats.missing_left or stats.missing_right:
            self.get_logger().warning(
                "Unpaired timestamp rows: "
                f"missing_left={stats.missing_left}, "
                f"missing_right={stats.missing_right}."
            )

        records: List[FrameRecord] = []
        for source in sources:
            semantic_path = None
            if self.use_precomputed:
                candidate = self.semantic_dir / source.left_path.name
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"Precomputed semantic label is missing: {candidate}"
                    )
                semantic_path = candidate
            records.append(
                FrameRecord(
                    timestamp_ns=source.left_timestamp_ns,
                    right_timestamp_ns=source.right_timestamp_ns,
                    left_path=source.left_path,
                    right_path=source.right_path,
                    semantic_path=semantic_path,
                )
            )

        if not records:
            raise RuntimeError(
                "No organized stereo pairs passed the timestamp-skew limit. "
                "Choose a synchronized sequence or explicitly increase "
                "max_stereo_skew_ms."
            )
        return records, stats.selection_summary(
            pairing_method="frame_index",
            configured_max_abs_skew_ns=self.max_stereo_skew_ns,
            max_frames=max_frames,
        )

    def _load_imu(self) -> List[ImuRecord]:
        if self.dataset_format == ORGANIZED_FORMAT:
            all_records = self._load_organized_imu()
        else:
            all_records = self._load_legacy_imu()

        all_records.sort(key=lambda record: record.timestamp_ns)
        if not all_records:
            raise RuntimeError("No valid IMU rows were loaded")

        first_image_ns = self.frames[0].timestamp_ns
        last_image_ns = self.frames[-1].timestamp_ns
        before = [
            record for record in all_records
            if record.timestamp_ns < first_image_ns
        ]
        inside = [
            record
            for record in all_records
            if first_image_ns <= record.timestamp_ns <= last_image_ns
        ]
        after = [
            record for record in all_records
            if record.timestamp_ns > last_image_ns
        ]

        selected: List[ImuRecord] = []
        if before:
            selected.append(before[-1])
        selected.extend(inside)
        if after:
            selected.append(after[0])
        return selected or all_records

    def _load_lidar(self) -> List[OrganizedLidarRecord]:
        if not self.publish_lidar:
            return []
        if self.dataset_format != ORGANIZED_FORMAT:
            raise RuntimeError(
                "Dataset LiDAR publication currently requires "
                "batch_organized/lidar_raw"
            )
        # Keep one cloud just outside each image boundary so nearest-time
        # association remains possible at the beginning and end.
        margin_ns = 200_000_000
        return load_organized_lidar_records(
            self.data_dir,
            first_timestamp_ns=self.frames[0].timestamp_ns - margin_ns,
            last_timestamp_ns=self.frames[-1].timestamp_ns + margin_ns,
            stride=self.lidar_publish_stride,
        )

    def _load_lio_odometry(self) -> List[OrganizedFastLioOdometryRecord]:
        if not self.publish_lio_odometry:
            return []
        if self.dataset_format != ORGANIZED_FORMAT:
            raise RuntimeError(
                "Saved FAST-LIO odometry publication requires "
                "batch_organized/lidar_fastlio/trajectory.npz"
            )
        configured = str(
            self.get_parameter("lio_trajectory_path").value
        ).strip()
        margin_ns = 200_000_000
        return load_organized_fastlio_odometry(
            self.data_dir,
            trajectory_path=(
                Path(configured) if configured else None
            ),
            first_timestamp_ns=(
                self.frames[0].timestamp_ns - margin_ns
            ),
            last_timestamp_ns=(
                self.frames[-1].timestamp_ns + margin_ns
            ),
            stride=self.lio_publish_stride,
        )

    def _load_organized_imu(self) -> List[ImuRecord]:
        return [
            ImuRecord(
                timestamp_ns=record.timestamp_ns,
                gyro_rad_s=record.gyro_rad_s,
            )
            for record in load_organized_imu_records(self.imu_path)
        ]

    def _load_legacy_imu(self) -> List[ImuRecord]:
        if not self.imu_path.is_file():
            raise FileNotFoundError(f"IMU Excel file does not exist: {self.imu_path}")

        import pandas as pd

        dataframe = pd.read_excel(self.imu_path)
        time_column = _find_column(dataframe.columns, ["星时"])
        wx_column = _find_column(dataframe.columns, ["wbix"])
        wy_column = _find_column(dataframe.columns, ["wbiy"])
        wz_column = _find_column(dataframe.columns, ["wbiz"])

        all_records: List[ImuRecord] = []
        for row in dataframe.itertuples(index=False, name=None):
            mapping = dict(zip(dataframe.columns, row))
            timestamp_ns = seconds_to_nanoseconds(mapping[time_column])
            gyro_deg_s = np.array(
                [mapping[wx_column], mapping[wy_column], mapping[wz_column]],
                dtype=np.float64,
            )
            if not np.isfinite(gyro_deg_s).all():
                continue
            all_records.append(
                ImuRecord(
                    timestamp_ns=timestamp_ns,
                    gyro_rad_s=np.deg2rad(gyro_deg_s),
                )
            )

        return all_records

    def _build_events(self) -> List[Event]:
        events = [
            Event(record.timestamp_ns, 0, "imu", index)
            for index, record in enumerate(self.imu_records)
        ]
        events.extend(
            Event(record.timestamp_ns, 2, "frame", index)
            for index, record in enumerate(self.frames)
        )
        events.extend(
            Event(record.timestamp_ns, 1, "lidar", index)
            for index, record in enumerate(self.lidar_records)
        )
        events.extend(
            Event(record.timestamp_ns, 1, "lio", index)
            for index, record in enumerate(self.lio_odometry_records)
        )
        events.sort(key=lambda event: (event.timestamp_ns, event.priority))
        return events

    def _on_timer(self) -> None:
        if self.finished:
            return

        now = time.monotonic()
        if not self.play_started:
            if now - self.node_start_wall < self.start_delay_sec:
                return
            if self.lossless_playback:
                if self.mapper_readiness_wait_wall is None:
                    self.mapper_readiness_wait_wall = now
                ready, readiness_description = self._mapper_is_ready()
                if not ready:
                    if not self.mapper_readiness_wait_logged:
                        self.get_logger().info(
                            "Lossless playback is waiting for mapper ROS "
                            "interfaces: " + readiness_description
                        )
                        self.mapper_readiness_wait_logged = True
                    if (
                        now - self.mapper_readiness_wait_wall
                        >= self.processing_ack_timeout_sec
                    ):
                        self._fail_playback(
                            "Timed out waiting for mapper ROS interfaces: "
                            + readiness_description
                        )
                    return
                self.mapper_ready_confirmed = True
                self.mapper_readiness_wait_sec = max(
                    0.0, now - self.mapper_readiness_wait_wall
                )
                self.get_logger().info(
                    "Mapper ROS interfaces are ready for lossless playback "
                    f"after {self.mapper_readiness_wait_sec:.3f}s."
                )
            self.play_started = True
            self.play_start_wall = now
            self.get_logger().info("Dataset playback started.")

        if (
            self.lossless_playback
            and self.pending_frame_acks
            and now - self.pending_frame_acks[0][1]
            >= self.processing_ack_timeout_sec
        ):
            oldest_stamp_ns, _ = self.pending_frame_acks[0]
            self._fail_playback(
                "Timed out waiting for mapper completion notification for "
                f"stamp_ns={oldest_stamp_ns} after "
                f"{self.processing_ack_timeout_sec:.3f}s"
            )
            return

        while self.event_index < len(self.events):
            event = self.events[self.event_index]
            due_wall = self.play_start_wall + (
                (event.timestamp_ns - self.sensor_start_ns)
                / 1_000_000_000.0
                / self.playback_rate
            )
            if time.monotonic() < due_wall:
                break
            if (
                event.kind == "frame"
                and self.lossless_playback
                and len(self.pending_frame_acks)
                >= self.lossless_max_in_flight_frames
            ):
                break

            if event.kind == "imu":
                self._publish_imu(self.imu_records[event.index])
            elif event.kind == "lidar":
                self._publish_lidar(self.lidar_records[event.index])
            elif event.kind == "lio":
                self._publish_lio_odometry(
                    self.lio_odometry_records[event.index]
                )
            else:
                frame_wall = time.monotonic()
                published = self._publish_frame(
                    event.index, self.frames[event.index]
                )
                self.playback_monitor.record_frame_attempt(
                    event.timestamp_ns,
                    published=published,
                    wall_time=frame_wall,
                    schedule_lateness_sec=frame_wall - due_wall,
                )
                if not published and self.lossless_playback:
                    self.event_index += 1
                    self._fail_playback(
                        "Failed to publish a complete stereo/semantic frame "
                        f"for stamp_ns={event.timestamp_ns}"
                    )
                    return
                if published and self.lossless_playback:
                    self.pending_frame_acks.append(
                        (int(event.timestamp_ns), time.monotonic())
                    )

            self.event_index += 1

        if self.event_index >= len(self.events):
            if self.loop_enabled:
                self.event_index = 0
                self.play_start_wall = time.monotonic()
                self.get_logger().info("Dataset playback loop restarted.")
            else:
                if not self.playback_monitor.publication_finished:
                    self.playback_monitor.mark_publication_finished()
                    self.get_logger().info(
                        "Dataset sensor publication finished: "
                        f"published={self.playback_monitor.published_frame_count}/"
                        f"{len(self.frames)}, mapper_completed="
                        f"{self.playback_monitor.acknowledged_frame_count}/"
                        f"{self.playback_monitor.published_frame_count}."
                    )
                    self._write_playback_summary()
                if (
                    self.lossless_playback
                    and self.pending_frame_acks
                ):
                    return
                self.finished = True
                self._report_completion_if_ready()

    def _mapper_is_ready(self):
        completion_publishers = self.count_publishers(
            self.input_processed_topic
        )
        input_subscribers = tuple(
            self.count_subscribers(topic)
            for topic in self.mapper_input_topics
        )
        ready = completion_publishers > 0 and all(
            count > 0 for count in input_subscribers
        )
        description = (
            f"completion_publishers={completion_publishers}, "
            "input_subscribers(left/right/semantic)="
            f"{input_subscribers}"
        )
        return ready, description

    def _on_input_processed(self, message: Header) -> None:
        stamp_ns = (
            int(message.stamp.sec) * 1_000_000_000
            + int(message.stamp.nanosec)
        )
        accepted = self.playback_monitor.acknowledge(
            stamp_ns, time.monotonic()
        )
        if not accepted:
            count = self.playback_monitor.unexpected_notification_count
            if count == 1 or count % 50 == 0:
                self.get_logger().warning(
                    "Ignored an unmatched or duplicate mapper completion "
                    f"notification: stamp_ns={stamp_ns}, count={count}."
                )
            return

        pending_index = next(
            (
                index
                for index, (pending_stamp_ns, _) in enumerate(
                    self.pending_frame_acks
                )
                if pending_stamp_ns == stamp_ns
            ),
            None,
        )
        if pending_index is not None:
            del self.pending_frame_acks[pending_index]
        elif self.lossless_playback:
            self._fail_playback(
                "Mapper completion did not match an in-flight lossless "
                f"frame: in_flight={list(self.pending_frame_acks)}, "
                f"received_stamp_ns={stamp_ns}"
            )
            return

        if self.playback_monitor.publication_finished:
            self._report_completion_if_ready()
            self._write_playback_summary()

    def _report_completion_if_ready(self) -> None:
        if not self.playback_monitor.is_complete:
            return
        self.finished = True
        summary = self.playback_monitor.summary()
        effective_rate = summary["effective_playback_rate_x"]
        effective_text = (
            "n/a" if effective_rate is None else f"{effective_rate:.3f}x"
        )
        self.get_logger().info(
            "Dataset playback and mapper input consumption are complete: "
            f"published={summary['published_frame_count']}, "
            f"completed={summary['acknowledged_frame_count']}, "
            f"requested_rate={self.playback_rate:.3f}x, "
            f"effective_rate={effective_text}. The node remains alive; "
            "press Ctrl+C to exit."
        )

    def _fail_playback(self, reason: str) -> None:
        self.playback_monitor.mark_failed(reason)
        self.finished = True
        self.get_logger().error(
            "Dataset playback stopped to preserve input completeness: "
            f"{reason}. No additional image frame will be published."
        )
        self._write_playback_summary()

    def _write_playback_summary(self) -> None:
        if self.playback_summary_path is None:
            return
        try:
            summary = self.playback_monitor.summary()
            summary["complete_input_consumption_scope"] = (
                "selected_frame_count_after_pairing_and_configured_filters"
            )
            summary["input_selection"] = self.input_selection_summary
            summary["playback_configuration"] = {
                "data_dir": str(self.data_dir),
                "dataset_format": self.dataset_format,
                "max_frames": self.max_frames,
                "max_stereo_skew_ms": (
                    self.max_stereo_skew_ns / 1_000_000.0
                ),
                "start_delay_sec": self.start_delay_sec,
                "processing_ack_timeout_sec": (
                    self.processing_ack_timeout_sec
                ),
                "lossless_max_in_flight_frames": (
                    self.lossless_max_in_flight_frames
                ),
                "input_processed_topic": self.input_processed_topic,
            }
            summary["mapper_readiness"] = {
                "required": self.lossless_playback,
                "confirmed_before_playback": self.mapper_ready_confirmed,
                "wait_sec_after_start_delay": (
                    self.mapper_readiness_wait_sec
                ),
            }
            self.playback_monitor.write_summary(
                self.playback_summary_path,
                summary,
            )
        except Exception as exception:
            self.get_logger().error(
                "Could not write dataset playback summary "
                f"{self.playback_summary_path}: {exception}"
            )

    def close_resources(self) -> None:
        self._write_playback_summary()

    def _publish_imu(self, record: ImuRecord) -> None:
        if self.imu_pub is None:
            raise RuntimeError("Cannot publish IMU while publish_imu is false")
        message = make_imu_message(
            stamp=nanoseconds_to_time(record.timestamp_ns),
            frame_id=self.imu_frame_id,
            angular_velocity_rad_s=record.gyro_rad_s,
        )
        self.imu_pub.publish(message)

    def _publish_lidar(self, record: OrganizedLidarRecord) -> None:
        message = read_binary_pcd_message(
            record.path,
            stamp=nanoseconds_to_time(record.timestamp_ns),
            frame_id=self.lidar_frame_id or record.frame_id,
            max_points=self.lidar_publish_max_points,
        )
        self.lidar_pub.publish(message)

    def _publish_lio_odometry(
        self, record: OrganizedFastLioOdometryRecord
    ) -> None:
        message = Odometry()
        message.header.stamp = nanoseconds_to_time(record.timestamp_ns)
        message.header.frame_id = self.lio_parent_frame_id
        message.child_frame_id = self.lio_child_frame_id
        message.pose.pose.position.x = float(record.position[0])
        message.pose.pose.position.y = float(record.position[1])
        message.pose.pose.position.z = float(record.position[2])
        message.pose.pose.orientation.x = float(record.orientation_xyzw[0])
        message.pose.pose.orientation.y = float(record.orientation_xyzw[1])
        message.pose.pose.orientation.z = float(record.orientation_xyzw[2])
        message.pose.pose.orientation.w = float(record.orientation_xyzw[3])
        message.pose.covariance = record.pose_covariance.reshape(-1).tolist()
        message.twist.twist.linear.x = float(record.linear_velocity[0])
        message.twist.twist.linear.y = float(record.linear_velocity[1])
        message.twist.twist.linear.z = float(record.linear_velocity[2])
        message.twist.twist.angular.x = float(record.angular_velocity[0])
        message.twist.twist.angular.y = float(record.angular_velocity[1])
        message.twist.twist.angular.z = float(record.angular_velocity[2])
        message.twist.covariance = record.twist_covariance.reshape(-1).tolist()
        self.lio_odometry_pub.publish(message)

    def _publish_frame(self, frame_index: int, record: FrameRecord) -> bool:
        left = cv2.imread(str(record.left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(record.right_path), cv2.IMREAD_COLOR)

        if left is None or right is None:
            self.get_logger().error(
                f"Failed to read stereo frame: {record.left_path}, {record.right_path}"
            )
            return False
        if left.shape != right.shape:
            self.get_logger().error(
                f"Stereo image shapes differ: left={left.shape}, right={right.shape}"
            )
            return False

        try:
            left, right = self.rectifier.rectify(left, right)
        except (ValueError, cv2.error) as exception:
            self.get_logger().error(
                f"Stereo rectification failed for frame {frame_index}: {exception}"
            )
            return False

        stamp = nanoseconds_to_time(record.timestamp_ns)
        height, width = left.shape[:2]
        camera = self.rectifier.camera_parameters(width, height)
        baseline = float(self.calibration["baseline"])

        right_message = image_to_message(
            right, stamp, self.right_frame_id, "bgr8"
        )
        left_message = image_to_message(
            left, stamp, self.left_frame_id, "bgr8"
        )
        left_info = make_camera_info(
            stamp=stamp,
            frame_id=self.left_frame_id,
            width=width,
            height=height,
            fx=float(camera["fx"]),
            fy=float(camera["fy"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
            baseline_m=baseline,
            is_right=False,
        )
        right_info = make_camera_info(
            stamp=stamp,
            frame_id=self.right_frame_id,
            width=width,
            height=height,
            fx=float(camera["fx"]),
            fy=float(camera["fy"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
            baseline_m=baseline,
            is_right=True,
        )

        # Space the two large image payloads to avoid losing the second UDP sample
        # under WSL/Cyclone DDS. Header.stamp remains exactly identical.
        self.right_info_pub.publish(right_info)
        self.right_pub.publish(right_message)
        if self.stereo_gap > 0.0:
            time.sleep(self.stereo_gap)
        self.left_info_pub.publish(left_info)
        self.left_pub.publish(left_message)

        if self.use_precomputed and record.semantic_path is not None:
            semantic = cv2.imread(str(record.semantic_path), cv2.IMREAD_UNCHANGED)
            if semantic is None:
                self.get_logger().error(
                    f"Failed to read semantic label: {record.semantic_path}"
                )
                return False
            else:
                if semantic.ndim == 3:
                    semantic = semantic[:, :, 0]
                semantic = semantic.astype(np.uint8, copy=False)
                try:
                    semantic = self.rectifier.rectify_left(
                        semantic,
                        interpolation=cv2.INTER_NEAREST,
                    )
                except (ValueError, cv2.error) as exception:
                    self.get_logger().error(
                        "Semantic rectification failed for frame "
                        f"{frame_index}: {exception}"
                    )
                    return False
                semantic_message = image_to_message(
                    semantic, stamp, self.left_frame_id, "mono8"
                )
                self.semantic_pub.publish(semantic_message)

        self.get_logger().info(
            f"Published stereo frame {frame_index + 1}/{len(self.frames)} "
            f"at sensor time {record.timestamp_ns // 1_000_000_000}."
            f"{record.timestamp_ns % 1_000_000_000:09d}"
        )
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DatasetSensorPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close_resources()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
