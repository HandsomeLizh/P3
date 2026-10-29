from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .localization.pose_math import inverse_pose, validated_pose_matrix


LIDAR_TEMPORAL_MODES = frozenset(
    {"none", "upstream_deskewed", "per_point"}
)


@dataclass(frozen=True)
class LidarTemporalConfiguration:
    """Explicit temporal contract for one PointCloud2 input stream.

    ``none`` preserves the original uncorrected-cloud geometry contract.  It
    is a compatibility mode and performs no motion correction; the caller may
    still use the safer predecessor/successor nearest-cloud pairing policy.

    ``upstream_deskewed`` asserts that every point has already been deskewed
    into the sensor frame at ``PointCloud2.header.stamp``.  This component
    only moves that rigid cloud from the cloud time to the image time.

    ``per_point`` treats the configured PointCloud2 scalar field as a
    relative offset from ``header.stamp`` and compensates every point before
    moving the cloud to the image time.  Pose-time binning changes no point
    count; it only bounds the number of interpolation queries.
    """

    mode: str = "none"
    max_point_time_offset_sec: float = 0.20
    pose_time_bin_sec: float = 0.001

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        maximum = float(self.max_point_time_offset_sec)
        bin_size = float(self.pose_time_bin_sec)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "max_point_time_offset_sec", maximum)
        object.__setattr__(self, "pose_time_bin_sec", bin_size)
        if mode not in LIDAR_TEMPORAL_MODES:
            raise ValueError(
                "lidar_temporal_mode must be one of "
                f"{sorted(LIDAR_TEMPORAL_MODES)}, got {mode!r}"
            )
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(
                "lidar_point_time_max_abs_sec must be finite and positive"
            )
        if not np.isfinite(bin_size) or bin_size < 0.0:
            raise ValueError(
                "lidar_deskew_time_bin_sec must be finite and non-negative"
            )
        if bin_size > maximum:
            raise ValueError(
                "lidar_deskew_time_bin_sec cannot exceed "
                "lidar_point_time_max_abs_sec"
            )

    @property
    def requires_external_pose(self) -> bool:
        return self.mode != "none"

    @property
    def requires_point_times(self) -> bool:
        return self.mode == "per_point"


@dataclass(frozen=True)
class LidarTemporalAlignment:
    """A cloud expressed in the sensor axes at the image reference time."""

    points_sensor: np.ndarray
    sensor_origin_sensor: np.ndarray
    statistics: dict[str, float]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_sensor, dtype=np.float64).reshape(-1, 3)
        origin = np.asarray(
            self.sensor_origin_sensor, dtype=np.float64
        ).reshape(3)
        if not np.isfinite(origin).all():
            raise ValueError("Aligned LiDAR sensor origin must be finite")
        points = points.copy()
        origin = origin.copy()
        points.setflags(write=False)
        origin.setflags(write=False)
        object.__setattr__(self, "points_sensor", points)
        object.__setattr__(self, "sensor_origin_sensor", origin)
        object.__setattr__(self, "statistics", dict(self.statistics))


PoseQuery = Callable[[float], Optional[np.ndarray]]


def _validated_point_offsets(
    point_time_offsets_sec: Optional[np.ndarray],
    *,
    point_count: int,
    configuration: LidarTemporalConfiguration,
) -> Optional[np.ndarray]:
    if not configuration.requires_point_times:
        return None
    if point_time_offsets_sec is None:
        raise ValueError(
            "lidar_temporal_mode='per_point' requires a point-time field"
        )
    offsets = np.asarray(
        point_time_offsets_sec, dtype=np.float64
    ).reshape(-1)
    if len(offsets) != int(point_count):
        raise ValueError(
            "PointCloud2 XYZ and point-time counts differ: "
            f"{point_count} != {len(offsets)}"
        )
    if not np.isfinite(offsets).all():
        raise ValueError("PointCloud2 point-time field contains non-finite values")
    maximum = float(np.max(np.abs(offsets), initial=0.0))
    if maximum > configuration.max_point_time_offset_sec:
        raise ValueError(
            "PointCloud2 point-time offset exceeds the configured safety "
            f"limit: {maximum:.6f}s > "
            f"{configuration.max_point_time_offset_sec:.6f}s; check the "
            "field unit and header-relative-time contract"
        )
    return offsets.copy()


def lidar_required_pose_time_range(
    *,
    cloud_timestamp: float,
    reference_timestamp: float,
    point_count: int,
    configuration: LidarTemporalConfiguration,
    point_time_offsets_sec: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float]]:
    """Return the exact pose-buffer interval needed by one alignment."""

    cloud_time = float(cloud_timestamp)
    reference_time = float(reference_timestamp)
    if not np.isfinite(cloud_time) or not np.isfinite(reference_time):
        raise ValueError("LiDAR cloud/reference timestamps must be finite")
    if not configuration.requires_external_pose:
        return None
    offsets = _validated_point_offsets(
        point_time_offsets_sec,
        point_count=point_count,
        configuration=configuration,
    )
    times = [cloud_time, reference_time]
    if offsets is not None and len(offsets):
        bin_size = configuration.pose_time_bin_sec
        motion_offsets = (
            np.rint(offsets / bin_size) * bin_size
            if bin_size > 0.0
            else offsets
        )
        times.extend(
            [
                cloud_time + float(np.min(motion_offsets)),
                cloud_time + float(np.max(motion_offsets)),
            ]
        )
    return float(min(times)), float(max(times))


def align_lidar_points_to_reference(
    points_sensor: np.ndarray,
    *,
    cloud_timestamp: float,
    reference_timestamp: float,
    transform_camera_sensor: np.ndarray,
    configuration: LidarTemporalConfiguration,
    pose_query: Optional[PoseQuery] = None,
    point_time_offsets_sec: Optional[np.ndarray] = None,
) -> LidarTemporalAlignment:
    """Align a cloud to an image time without changing or dropping points.

    The pose callback supplies ``T_world_camera(t)``.  The returned points
    use the original sensor axes, but their coordinate frame is the rigid
    sensor frame at ``reference_timestamp``.  Consequently the established
    LiDAR-to-camera projection and keyframe-pose fusion path can remain
    unchanged.
    """

    points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    cloud_time = float(cloud_timestamp)
    reference_time = float(reference_timestamp)
    if not np.isfinite(cloud_time) or not np.isfinite(reference_time):
        raise ValueError("LiDAR cloud/reference timestamps must be finite")
    offsets = _validated_point_offsets(
        point_time_offsets_sec,
        point_count=len(points),
        configuration=configuration,
    )

    if not configuration.requires_external_pose:
        return LidarTemporalAlignment(
            points_sensor=points,
            sensor_origin_sensor=np.zeros(3, dtype=np.float64),
            statistics={
                "temporal_compensation_applied": 0.0,
                "point_deskew_applied": 0.0,
                "cloud_to_reference_sec": reference_time - cloud_time,
                "point_time_min_sec": 0.0,
                "point_time_max_sec": 0.0,
                "point_time_span_sec": 0.0,
                "pose_time_bin_max_error_sec": 0.0,
                "pose_query_count": 0.0,
                "correction_mean_m": 0.0,
                "correction_p95_m": 0.0,
                "correction_max_m": 0.0,
            },
        )
    if pose_query is None:
        raise ValueError(
            f"lidar_temporal_mode={configuration.mode!r} requires a pose query"
        )

    camera_sensor = validated_pose_matrix(
        transform_camera_sensor,
        name="LiDAR temporal camera-to-sensor calibration",
    )
    sensor_camera = inverse_pose(camera_sensor)
    pose_cache: dict[float, np.ndarray] = {}

    def require_pose(timestamp: float) -> np.ndarray:
        key = float(timestamp)
        cached = pose_cache.get(key)
        if cached is not None:
            return cached
        value = pose_query(key)
        if value is None:
            raise RuntimeError(
                "External LIO pose buffer does not bracket LiDAR temporal "
                f"query t={key:.9f}s"
            )
        pose = validated_pose_matrix(
            value,
            name=f"external camera pose at {key:.9f}s",
        )
        pose_cache[key] = pose
        return pose

    world_camera_reference = require_pose(reference_time)
    sensor_reference_world = (
        sensor_camera @ inverse_pose(world_camera_reference)
    )

    def sensor_motion(timestamp: float) -> np.ndarray:
        return (
            sensor_reference_world
            @ require_pose(timestamp)
            @ camera_sensor
        )

    header_motion = sensor_motion(cloud_time)
    aligned = np.empty_like(points)
    if offsets is None:
        aligned[:] = (
            points @ header_motion[:3, :3].T
            + header_motion[:3, 3]
        )
        motion_offsets = np.zeros(len(points), dtype=np.float64)
        bin_error = 0.0
    else:
        bin_size = configuration.pose_time_bin_sec
        motion_offsets = (
            np.rint(offsets / bin_size) * bin_size
            if bin_size > 0.0
            else offsets.copy()
        )
        bin_error = float(
            np.max(np.abs(motion_offsets - offsets), initial=0.0)
        )
        unique_offsets, inverse = np.unique(
            motion_offsets, return_inverse=True
        )
        for index, offset in enumerate(unique_offsets):
            selected = inverse == index
            motion = sensor_motion(cloud_time + float(offset))
            aligned[selected] = (
                points[selected] @ motion[:3, :3].T
                + motion[:3, 3]
            )

    correction = np.linalg.norm(aligned - points, axis=1)
    finite_correction = correction[np.isfinite(correction)]
    if len(finite_correction):
        mean_correction = float(np.mean(finite_correction))
        p95_correction = float(np.percentile(finite_correction, 95.0))
        maximum_correction = float(np.max(finite_correction))
    else:
        mean_correction = p95_correction = maximum_correction = 0.0
    if offsets is None or not len(offsets):
        time_min = time_max = time_span = 0.0
    else:
        time_min = float(np.min(offsets))
        time_max = float(np.max(offsets))
        time_span = time_max - time_min

    return LidarTemporalAlignment(
        points_sensor=aligned,
        sensor_origin_sensor=header_motion[:3, 3],
        statistics={
            "temporal_compensation_applied": 1.0,
            "point_deskew_applied": float(offsets is not None),
            "cloud_to_reference_sec": reference_time - cloud_time,
            "point_time_min_sec": time_min,
            "point_time_max_sec": time_max,
            "point_time_span_sec": time_span,
            "pose_time_bin_max_error_sec": bin_error,
            "pose_query_count": float(len(pose_cache)),
            "correction_mean_m": mean_correction,
            "correction_p95_m": p95_correction,
            "correction_max_m": maximum_correction,
        },
    )
