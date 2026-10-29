"""Low-intrusion PointCloud2 evidence for the elevation layer only.

The helper owns no ROS publishers and never touches localisation. Sensor
callbacks only retain bounded messages. A ROMA map-keyframe transaction asks
for the nearest samples, transforms them with the already-determined
T_map_base and commits the selected stamps only after the map candidate is
successfully committed.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Dict, Mapping, Optional

import numpy as np
import yaml
from sensor_msgs.msg import PointCloud2

from .lidar_mapping import pointcloud2_xyz_array


@dataclass(frozen=True)
class TerrainPointCloudSensor:
    name: str
    topic: str
    expected_frame_id: str
    max_time_offset_sec: float
    queue_size: int
    min_range_m: float
    max_range_m: float
    voxel_size_m: float
    max_points: int
    transform_base_sensor: np.ndarray


@dataclass(frozen=True)
class TerrainPointCloudUpdate:
    points_map: np.ndarray
    selected_stamps_ns: Mapping[str, int]
    statistics: Mapping[str, float]
    registered_points_map: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _PreparedTerrainPointCloud:
    points_base: np.ndarray
    statistics: Mapping[str, int]


@dataclass
class _BufferedTerrainPointCloud:
    timestamp_ns: int
    future: Future
    users: int = 0


def _normalise_frame(value: str) -> str:
    return str(value).strip().lstrip("/")


def _stamp_ns(message: PointCloud2) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def _validated_transform(value, *, sensor_name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(
            f"{sensor_name}: transform_base_sensor must be a finite 4x4 matrix"
        )
    if not np.allclose(
        transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-9
    ):
        raise ValueError(
            f"{sensor_name}: transform_base_sensor has an invalid last row"
        )
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6):
        raise ValueError(
            f"{sensor_name}: transform_base_sensor rotation is not orthonormal"
        )
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(
            f"{sensor_name}: transform_base_sensor rotation determinant is not +1"
        )
    result = transform.copy()
    result.setflags(write=False)
    return result


def _sensor_from_mapping(name: str, value: Mapping) -> TerrainPointCloudSensor:
    topic = str(value.get("topic", "")).strip()
    frame = _normalise_frame(value.get("expected_frame_id", ""))
    if not topic or not frame:
        raise ValueError(f"{name}: topic and expected_frame_id are required")

    max_time_offset_sec = float(value.get("max_time_offset_sec", 0.0))
    queue_size = int(value.get("queue_size", 0))
    min_range_m = float(value.get("min_range_m", 0.0))
    max_range_m = float(value.get("max_range_m", 0.0))
    voxel_size_m = float(value.get("voxel_size_m", 0.0))
    max_points = int(value.get("max_points", 0))
    if max_time_offset_sec <= 0.0:
        raise ValueError(f"{name}: max_time_offset_sec must be positive")
    if queue_size <= 0 or max_points <= 0:
        raise ValueError(f"{name}: queue_size and max_points must be positive")
    if min_range_m < 0.0 or max_range_m <= min_range_m:
        raise ValueError(f"{name}: invalid range limits")
    if voxel_size_m < 0.0:
        raise ValueError(f"{name}: voxel_size_m cannot be negative")

    return TerrainPointCloudSensor(
        name=str(name),
        topic=topic,
        expected_frame_id=frame,
        max_time_offset_sec=max_time_offset_sec,
        queue_size=queue_size,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        voxel_size_m=voxel_size_m,
        max_points=max_points,
        transform_base_sensor=_validated_transform(
            value.get("transform_base_sensor"), sensor_name=str(name)
        ),
    )


class TerrainPointCloudMapper:
    """Buffer and prepare independently calibrated elevation sources."""

    def __init__(self, config_path: Path, *, logger,
                 retain_registered_points: bool = False) -> None:
        self.retain_registered_points = bool(retain_registered_points)
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Point-cloud fusion configuration does not exist: {path}"
            )
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError("Point-cloud fusion configuration must be a mapping")
        if int(document.get("version", 0)) != 1:
            raise ValueError("Unsupported point-cloud fusion configuration version")

        self.config_path = path
        self.base_frame = _normalise_frame(document.get("base_frame", ""))
        self.map_frame = _normalise_frame(document.get("map_frame", ""))
        if not self.base_frame or not self.map_frame:
            raise ValueError("base_frame and map_frame are required")

        sensor_values = document.get("sensors")
        if not isinstance(sensor_values, Mapping) or not sensor_values:
            raise ValueError("Point-cloud fusion requires at least one sensor")
        self.sensors: Dict[str, TerrainPointCloudSensor] = {
            str(name): _sensor_from_mapping(str(name), value)
            for name, value in sensor_values.items()
        }
        topics = [sensor.topic for sensor in self.sensors.values()]
        if len(set(topics)) != len(topics):
            raise ValueError("Point-cloud fusion topics must be unique")

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="terrain_pointcloud"
        )
        self._closed = False
        self._logger = logger
        self._lock = threading.Lock()
        self._buffers = {
            name: deque(maxlen=sensor.queue_size)
            for name, sensor in self.sensors.items()
        }
        self._last_committed_stamp_ns = {
            name: -1 for name in self.sensors
        }
        self._warning_counts: Dict[str, int] = {}

    def _warn(self, key: str, message: str) -> None:
        count = self._warning_counts.get(key, 0) + 1
        self._warning_counts[key] = count
        if count == 1 or count % 50 == 0:
            self._logger.warning(f"{message} (count={count})")

    def record_message(self, sensor_name: str, message: PointCloud2) -> None:
        sensor = self.sensors.get(sensor_name)
        if sensor is None:
            self._warn(
                "unknown_sensor",
                f"Ignoring unconfigured point-cloud sensor {sensor_name!r}",
            )
            return
        frame = _normalise_frame(message.header.frame_id)
        if frame != sensor.expected_frame_id:
            self._warn(
                f"{sensor_name}:frame",
                f"Skipping {sensor_name} PointCloud2: expected frame_id "
                f"{sensor.expected_frame_id!r}, received "
                f"{message.header.frame_id!r}",
            )
            return
        timestamp_ns = _stamp_ns(message)
        if timestamp_ns <= 0:
            self._warn(
                f"{sensor_name}:stamp",
                f"Skipping {sensor_name} PointCloud2 with a zero timestamp",
            )
            return
        with self._lock:
            if self._closed:
                return
            future = self._executor.submit(
                self._prepare_sensor_base_points, sensor, message
            )
            buffer = self._buffers[sensor_name]
            if len(buffer) == buffer.maxlen:
                dropped = buffer.popleft()
                if dropped.users == 0:
                    dropped.future.cancel()
            buffer.append(
                _BufferedTerrainPointCloud(timestamp_ns, future)
            )

    def _nearest(
        self, sensor_name: str, target_stamp_ns: int
    ) -> Optional[tuple[_BufferedTerrainPointCloud, float]]:
        sensor = self.sensors[sensor_name]
        with self._lock:
            last_used = self._last_committed_stamp_ns[sensor_name]
            candidates = tuple(
                item
                for item in self._buffers[sensor_name]
                if item.timestamp_ns > last_used
            )
            if not candidates:
                return None
            item = min(
                candidates,
                key=lambda value: abs(value.timestamp_ns - target_stamp_ns),
            )
            offset_sec = (item.timestamp_ns - target_stamp_ns) * 1.0e-9
            if abs(offset_sec) > sensor.max_time_offset_sec:
                return None
            item.users += 1
            return item, offset_sec

    def _release(self, item: _BufferedTerrainPointCloud) -> None:
        with self._lock:
            item.users = max(0, item.users - 1)

    @staticmethod
    def _voxel_sample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
        if voxel_size_m <= 0.0 or len(points) == 0:
            return points
        keys = np.floor(points / voxel_size_m).astype(np.int64)
        _, first_indices = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(first_indices)]

    @staticmethod
    def _cell_medians(points_map: np.ndarray, resolution: float) -> np.ndarray:
        if len(points_map) == 0:
            return np.empty((0, 3), dtype=np.float64)
        columns = np.floor(points_map[:, 0] / resolution).astype(np.int64)
        rows = np.floor(points_map[:, 1] / resolution).astype(np.int64)
        order = np.lexsort((points_map[:, 2], columns, rows))
        rows_sorted = rows[order]
        columns_sorted = columns[order]
        z_sorted = points_map[order, 2]
        boundaries = np.flatnonzero(
            (rows_sorted[1:] != rows_sorted[:-1])
            | (columns_sorted[1:] != columns_sorted[:-1])
        ) + 1
        starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
        ends = np.concatenate(
            (boundaries, np.array([len(order)], dtype=np.int64))
        )
        counts = ends - starts
        lower = starts + (counts - 1) // 2
        upper = starts + counts // 2
        median_z = 0.5 * (z_sorted[lower] + z_sorted[upper])
        representative = np.column_stack(
            (
                (columns_sorted[starts] + 0.5) * resolution,
                (rows_sorted[starts] + 0.5) * resolution,
                median_z,
            )
        )
        return np.ascontiguousarray(representative, dtype=np.float64)

    def _prepare_sensor_base_points(
        self,
        sensor: TerrainPointCloudSensor,
        message: PointCloud2,
    ) -> _PreparedTerrainPointCloud:
        points = pointcloud2_xyz_array(message).reshape(-1, 3)
        input_points = len(points)
        finite = np.isfinite(points).all(axis=1)
        with np.errstate(invalid="ignore", over="ignore"):
            ranges = np.linalg.norm(points, axis=1)
        valid = (
            finite
            & (ranges >= sensor.min_range_m)
            & (ranges <= sensor.max_range_m)
        )
        points = points[valid]
        points = self._voxel_sample(points, sensor.voxel_size_m)
        if len(points) > sensor.max_points:
            sample_indices = np.linspace(
                0, len(points) - 1, sensor.max_points, dtype=np.int64
            )
            points = points[sample_indices]

        transform = sensor.transform_base_sensor
        points_base = points @ transform[:3, :3].T
        points_base += transform[:3, 3]
        return _PreparedTerrainPointCloud(
            points_base=points_base,
            statistics={
                "input_points": int(input_points),
                "finite_range_points": int(len(points)),
            },
        )

    def _prepare_sensor_points(
        self,
        prepared: _PreparedTerrainPointCloud,
        transform_map_base: np.ndarray,
        resolution: float,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        points_map = prepared.points_base @ transform_map_base[:3, :3].T
        points_map += transform_map_base[:3, 3]
        representatives = self._cell_medians(points_map, resolution)
        statistics = dict(prepared.statistics)
        statistics["cell_observations"] = int(len(representatives))
        return representatives, points_map, statistics

    def prepare_update(
        self,
        *,
        target_stamp_ns: int,
        transform_map_base: np.ndarray,
        map_resolution: float,
    ) -> TerrainPointCloudUpdate:
        if target_stamp_ns <= 0:
            return TerrainPointCloudUpdate(
                points_map=np.empty((0, 3), dtype=np.float64),
                selected_stamps_ns={},
                statistics={"selected_sensors": 0, "cell_observations": 0},
            )
        transform = np.asarray(transform_map_base, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_map_base must be a finite 4x4 matrix")
        if not np.isfinite(map_resolution) or map_resolution <= 0.0:
            raise ValueError("map_resolution must be positive")

        prepared = []
        registered_clouds = []
        selected_stamps: Dict[str, int] = {}
        statistics: Dict[str, float] = {}
        for name, sensor in self.sensors.items():
            selected = self._nearest(name, target_stamp_ns)
            if selected is None:
                continue
            item, offset_sec = selected
            try:
                sensor_cloud = item.future.result()
                points, registered_points, sensor_statistics = self._prepare_sensor_points(
                    sensor_cloud, transform, map_resolution
                )
            except Exception as exception:
                self._warn(
                    f"{name}:decode",
                    f"Skipping {name} PointCloud2 elevation evidence: "
                    f"{type(exception).__name__}: {exception}",
                )
                continue
            finally:
                self._release(item)
            if len(points) == 0:
                self._warn(
                    f"{name}:empty",
                    f"Skipping {name} PointCloud2 with no valid terrain points",
                )
                continue
            prepared.append(points)
            if self.retain_registered_points:
                registered_clouds.append(registered_points)
            selected_stamps[name] = item.timestamp_ns
            statistics[f"{name}_time_offset_sec"] = float(offset_sec)
            for key, value in sensor_statistics.items():
                statistics[f"{name}_{key}"] = value

        points_map = (
            np.ascontiguousarray(np.vstack(prepared), dtype=np.float64)
            if prepared
            else np.empty((0, 3), dtype=np.float64)
        )
        statistics["selected_sensors"] = int(len(selected_stamps))
        statistics["cell_observations"] = int(len(points_map))
        return TerrainPointCloudUpdate(
            points_map=points_map,
            selected_stamps_ns=selected_stamps,
            statistics=statistics,
            registered_points_map=(
                np.ascontiguousarray(np.vstack(registered_clouds), dtype=np.float32)
                if registered_clouds else None
            ),
        )

    def mark_committed(self, selected_stamps_ns: Mapping[str, int]) -> None:
        with self._lock:
            for name, timestamp_ns in selected_stamps_ns.items():
                if name not in self._buffers:
                    continue
                timestamp_ns = int(timestamp_ns)
                self._last_committed_stamp_ns[name] = max(
                    self._last_committed_stamp_ns[name], timestamp_ns
                )
                buffer = self._buffers[name]
                while buffer and buffer[0].timestamp_ns <= timestamp_ns:
                    removed = buffer.popleft()
                    if removed.users == 0:
                        removed.future.cancel()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for buffer in self._buffers.values():
                for item in buffer:
                    if item.users == 0:
                        item.future.cancel()
                buffer.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)

    @property
    def description(self) -> str:
        values = ", ".join(
            f"{sensor.name}={sensor.topic}[{sensor.expected_frame_id}]"
            for sensor in self.sensors.values()
        )
        return f"{values}; config={self.config_path}"
