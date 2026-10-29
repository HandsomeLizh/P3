from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from .localization.calibration import (
    load_rectified_lidar_to_camera_transform,
)


_POINT_FIELD_DTYPES = {
    PointField.INT8: "i1",
    PointField.UINT8: "u1",
    PointField.INT16: "i2",
    PointField.UINT16: "u2",
    PointField.INT32: "i4",
    PointField.UINT32: "u4",
    PointField.FLOAT32: "f4",
    PointField.FLOAT64: "f8",
}

_POINT_TIME_UNIT_SCALES = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1.0e-3,
    "msec": 1.0e-3,
    "millisecond": 1.0e-3,
    "milliseconds": 1.0e-3,
    "us": 1.0e-6,
    "usec": 1.0e-6,
    "microsecond": 1.0e-6,
    "microseconds": 1.0e-6,
    "ns": 1.0e-9,
    "nsec": 1.0e-9,
    "nanosecond": 1.0e-9,
    "nanoseconds": 1.0e-9,
}


def stamp_nanoseconds(message: PointCloud2) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def pointcloud2_xyz_array(message: PointCloud2) -> np.ndarray:
    """Decode x/y/z without Python-per-point unpacking.

    The structured view honours field offsets, point padding, row padding,
    and byte order.  A detached float64 Nx3 array is returned so callers may
    safely retain it after the ROS callback finishes.
    """
    fields = {field.name: field for field in message.fields}
    missing = {name for name in ("x", "y", "z") if name not in fields}
    if missing:
        raise ValueError(
            f"PointCloud2 is missing required fields: {sorted(missing)}"
        )
    if message.point_step <= 0 or message.row_step <= 0:
        raise ValueError(
            "PointCloud2 point_step and row_step must both be positive"
        )

    names = []
    formats = []
    offsets = []
    endian = ">" if bool(message.is_bigendian) else "<"
    for name in ("x", "y", "z"):
        field = fields[name]
        if int(field.count) != 1:
            raise ValueError(f"PointCloud2 field {name!r} must have count=1")
        scalar = _POINT_FIELD_DTYPES.get(int(field.datatype))
        if scalar not in {"f4", "f8"}:
            raise ValueError(
                f"PointCloud2 field {name!r} must be FLOAT32/FLOAT64; "
                f"got datatype={field.datatype}"
            )
        names.append(name)
        formats.append(np.dtype(endian + scalar))
        offsets.append(int(field.offset))

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )
    required_bytes = int(message.row_step) * int(message.height)
    if len(message.data) < required_bytes:
        raise ValueError(
            f"PointCloud2 data is truncated: {len(message.data)} < "
            f"{required_bytes} bytes"
        )
    structured = np.ndarray(
        shape=(int(message.height), int(message.width)),
        dtype=dtype,
        buffer=message.data,
        strides=(int(message.row_step), int(message.point_step)),
    )
    points = np.empty((structured.size, 3), dtype=np.float64)
    for column, name in enumerate(("x", "y", "z")):
        points[:, column] = structured[name].reshape(-1)
    return points


def point_time_unit_scale(unit: str) -> float:
    """Return the explicit PointCloud2 relative-time unit in seconds."""

    normalized = str(unit).strip().lower().replace("µ", "u")
    scale = _POINT_TIME_UNIT_SCALES.get(normalized)
    if scale is None:
        raise ValueError(
            "lidar_point_time_unit must describe seconds, milliseconds, "
            f"microseconds, or nanoseconds; got {unit!r}"
        )
    return float(scale)


def pointcloud2_xyz_time_arrays(
    message: PointCloud2,
    *,
    time_field: str = "time",
    time_unit: str = "sec",
) -> tuple[np.ndarray, np.ndarray]:
    """Decode XYZ and one scalar header-relative point-time field.

    The field is intentionally not guessed.  Its configured unit is converted
    to seconds, while its sign and zero remain unchanged: both negative
    scan-end-relative offsets and positive scan-start-relative offsets are
    valid when ``header.stamp + offset`` is the acquisition time.
    """

    points = pointcloud2_xyz_array(message)
    name = str(time_field).strip()
    if not name:
        raise ValueError("lidar_point_time_field cannot be empty")
    fields = {field.name: field for field in message.fields}
    field = fields.get(name)
    if field is None:
        raise ValueError(
            f"PointCloud2 is missing configured point-time field {name!r}"
        )
    if int(field.count) != 1:
        raise ValueError(
            f"PointCloud2 point-time field {name!r} must have count=1"
        )
    scalar = _POINT_FIELD_DTYPES.get(int(field.datatype))
    if scalar is None:
        raise ValueError(
            f"PointCloud2 point-time field {name!r} has unsupported "
            f"datatype={field.datatype}"
        )
    endian = ">" if bool(message.is_bigendian) else "<"
    scalar_dtype = np.dtype(endian + scalar)
    offset = int(field.offset)
    if offset < 0 or offset + scalar_dtype.itemsize > int(message.point_step):
        raise ValueError(
            f"PointCloud2 point-time field {name!r} exceeds point_step"
        )
    dtype = np.dtype(
        {
            "names": [name],
            "formats": [scalar_dtype],
            "offsets": [offset],
            "itemsize": int(message.point_step),
        }
    )
    structured = np.ndarray(
        shape=(int(message.height), int(message.width)),
        dtype=dtype,
        buffer=message.data,
        strides=(int(message.row_step), int(message.point_step)),
    )
    offsets_sec = (
        structured[name].reshape(-1).astype(np.float64, copy=True)
        * point_time_unit_scale(time_unit)
    )
    if len(points) != len(offsets_sec):
        raise RuntimeError("Decoded PointCloud2 XYZ/time sizes differ")
    return points, offsets_sec


@dataclass(frozen=True)
class LidarObservation:
    camera_points: np.ndarray
    semantic_labels: np.ndarray
    sensor_origin_camera: np.ndarray
    statistics: Dict[str, float]


@dataclass(frozen=True)
class PointCloudAssistedStereoObservation:
    """Stereo rays whose metric depth is corrected by projected point cloud.

    ``source_indices`` maps every returned point back to the caller's stereo
    keypoint/label arrays.  Point-cloud measurements only supply optical depth;
    the resulting point remains on the stereo pixel ray, so the observation
    origin is still the left camera rather than the displaced LiDAR origin.
    """

    camera_points: np.ndarray
    source_indices: np.ndarray
    pointcloud_supported: np.ndarray
    statistics: Dict[str, float]


def assist_stereo_depth_from_pointcloud(
    *,
    keypoints: np.ndarray,
    stereo_camera_points: np.ndarray,
    pointcloud_camera_points: np.ndarray,
    camera_matrix: np.ndarray,
    image_width: int,
    image_height: int,
    pixel_radius: float = 2.0,
    unmatched_stereo_max_depth_m: float = 5.0,
    replace_stereo_min_depth_m: float = 5.0,
) -> PointCloudAssistedStereoObservation:
    """Correct sparse stereo depth with a time-aligned ``PointCloud2`` cloud.

    Point-cloud points must already be expressed in the rectified left-camera
    frame.  A nearest-depth z-buffer prevents a farther surface from winning
    when several returns project to the same pixel.  Matching is bounded in
    image space and deterministic; no learned interpolation is introduced.

    Stereo estimates no farther than ``replace_stereo_min_depth_m`` are kept
    unchanged by default, even when a cloud return projects nearby.  Farther
    stereo estimates are replaced only where cloud support exists.  Points
    without such support are retained only up to
    ``unmatched_stereo_max_depth_m``.  A non-positive limit disables the
    corresponding threshold.  These rules apply only to assisted mode and do
    not alter the established stereo-only or direct-cloud paths.
    """

    keypoints = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    stereo_points = np.asarray(
        stereo_camera_points, dtype=np.float64
    ).reshape(-1, 3)
    cloud_points = np.asarray(
        pointcloud_camera_points, dtype=np.float64
    ).reshape(-1, 3)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    width = int(image_width)
    height = int(image_height)
    radius = float(pixel_radius)
    unmatched_limit = float(unmatched_stereo_max_depth_m)
    replacement_minimum = float(replace_stereo_min_depth_m)

    if len(keypoints) != len(stereo_points):
        raise ValueError(
            "keypoints and stereo_camera_points differ: "
            f"{len(keypoints)} != {len(stereo_points)}"
        )
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be finite")
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("pixel_radius must be finite and non-negative")
    if not np.isfinite(unmatched_limit):
        raise ValueError(
            "unmatched_stereo_max_depth_m must be finite"
        )
    if not np.isfinite(replacement_minimum):
        raise ValueError("replace_stereo_min_depth_m must be finite")

    stereo_valid = (
        np.isfinite(keypoints).all(axis=1)
        & np.isfinite(stereo_points).all(axis=1)
        & (stereo_points[:, 2] > 0.1)
        & (keypoints[:, 0] >= 0.0)
        & (keypoints[:, 0] < width)
        & (keypoints[:, 1] >= 0.0)
        & (keypoints[:, 1] < height)
    )
    source_indices = np.flatnonzero(stereo_valid)
    valid_keypoints = keypoints[source_indices]
    output_points = stereo_points[source_indices].copy()

    cloud_valid = (
        np.isfinite(cloud_points).all(axis=1)
        & (cloud_points[:, 2] > 0.1)
    )
    cloud_indices = np.flatnonzero(cloud_valid)
    projected_u = np.empty(0, dtype=np.float64)
    projected_v = np.empty(0, dtype=np.float64)
    projected_points = np.empty((0, 3), dtype=np.float64)
    unique_projected_count = 0

    if len(cloud_indices):
        candidate_points = cloud_points[cloud_indices]
        candidate_depth = candidate_points[:, 2]
        candidate_u = fx * candidate_points[:, 0] / candidate_depth + cx
        candidate_v = fy * candidate_points[:, 1] / candidate_depth + cy
        rounded_u = np.rint(candidate_u).astype(np.int64)
        rounded_v = np.rint(candidate_v).astype(np.int64)
        inside = (
            np.isfinite(candidate_u)
            & np.isfinite(candidate_v)
            & (rounded_u >= 0)
            & (rounded_u < width)
            & (rounded_v >= 0)
            & (rounded_v < height)
        )
        candidate_points = candidate_points[inside]
        candidate_u = candidate_u[inside]
        candidate_v = candidate_v[inside]
        rounded_u = rounded_u[inside]
        rounded_v = rounded_v[inside]

        # Sort near-to-far before choosing the first entry per rounded pixel.
        # A stable sort keeps the result deterministic for equal-depth points.
        if len(candidate_points):
            depth_order = np.argsort(candidate_points[:, 2], kind="stable")
            ordered_pixels = (
                rounded_v[depth_order] * width + rounded_u[depth_order]
            )
            _, first_positions = np.unique(
                ordered_pixels, return_index=True
            )
            selected = depth_order[first_positions]
            projected_points = candidate_points[selected]
            projected_u = candidate_u[selected]
            projected_v = candidate_v[selected]
            projected_pixel_u = rounded_u[selected]
            projected_pixel_v = rounded_v[selected]
            unique_projected_count = len(selected)
        else:
            projected_pixel_u = np.empty(0, dtype=np.int64)
            projected_pixel_v = np.empty(0, dtype=np.int64)
    else:
        projected_pixel_u = np.empty(0, dtype=np.int64)
        projected_pixel_v = np.empty(0, dtype=np.int64)

    best_cloud_index = np.full(len(valid_keypoints), -1, dtype=np.int64)
    best_distance_squared = np.full(
        len(valid_keypoints), np.inf, dtype=np.float64
    )
    if len(projected_points) and len(valid_keypoints):
        z_buffer = np.full((height, width), -1, dtype=np.int64)
        z_buffer[projected_pixel_v, projected_pixel_u] = np.arange(
            len(projected_points), dtype=np.int64
        )
        centre_u = np.rint(valid_keypoints[:, 0]).astype(np.int64)
        centre_v = np.rint(valid_keypoints[:, 1]).astype(np.int64)
        integer_radius = int(np.ceil(radius))
        radius_squared = radius * radius + 1e-12
        for offset_v in range(-integer_radius, integer_radius + 1):
            query_v = centre_v + offset_v
            valid_v = (query_v >= 0) & (query_v < height)
            if not np.any(valid_v):
                continue
            for offset_u in range(-integer_radius, integer_radius + 1):
                query_u = centre_u + offset_u
                valid_query = valid_v & (query_u >= 0) & (query_u < width)
                query_indices = np.flatnonzero(valid_query)
                if not len(query_indices):
                    continue
                cloud_at_query = z_buffer[
                    query_v[query_indices], query_u[query_indices]
                ]
                populated = cloud_at_query >= 0
                query_indices = query_indices[populated]
                cloud_at_query = cloud_at_query[populated]
                if not len(query_indices):
                    continue
                distance_squared = (
                    projected_u[cloud_at_query]
                    - valid_keypoints[query_indices, 0]
                ) ** 2 + (
                    projected_v[cloud_at_query]
                    - valid_keypoints[query_indices, 1]
                ) ** 2
                better = (
                    (distance_squared <= radius_squared)
                    & (
                        distance_squared
                        < best_distance_squared[query_indices]
                    )
                )
                accepted_queries = query_indices[better]
                accepted_cloud = cloud_at_query[better]
                best_distance_squared[accepted_queries] = (
                    distance_squared[better]
                )
                best_cloud_index[accepted_queries] = accepted_cloud

    matched = best_cloud_index >= 0
    replace_with_pointcloud = matched & (
        (replacement_minimum <= 0.0)
        | (output_points[:, 2] > replacement_minimum)
    )
    if np.any(replace_with_pointcloud):
        supported_depth = projected_points[
            best_cloud_index[replace_with_pointcloud], 2
        ]
        output_points[replace_with_pointcloud, 0] = (
            (valid_keypoints[replace_with_pointcloud, 0] - cx)
            / fx
            * supported_depth
        )
        output_points[replace_with_pointcloud, 1] = (
            (valid_keypoints[replace_with_pointcloud, 1] - cy)
            / fy
            * supported_depth
        )
        output_points[replace_with_pointcloud, 2] = supported_depth

    keep = replace_with_pointcloud.copy()
    if unmatched_limit <= 0.0:
        keep |= ~replace_with_pointcloud
    else:
        keep |= (~replace_with_pointcloud) & (
            output_points[:, 2] <= unmatched_limit
        )

    kept_points = output_points[keep]
    kept_indices = source_indices[keep]
    kept_supported = replace_with_pointcloud[keep]
    stereo_retained_count = int(
        np.count_nonzero(keep & ~replace_with_pointcloud)
    )
    stereo_dropped_count = int(
        np.count_nonzero(~keep & ~replace_with_pointcloud)
    )
    statistics = {
        "input_stereo_points": int(len(keypoints)),
        "valid_stereo_points": int(len(source_indices)),
        "input_pointcloud_points": int(len(cloud_points)),
        "forward_pointcloud_points": int(np.count_nonzero(cloud_valid)),
        "projected_pointcloud_pixels": int(unique_projected_count),
        "pointcloud_matched_points": int(np.count_nonzero(matched)),
        "pointcloud_supported_points": int(
            np.count_nonzero(replace_with_pointcloud)
        ),
        # The established names are retained for report compatibility.  A
        # geometrically matched near stereo point is intentionally not
        # replaced, so the clearer aliases below distinguish policy from
        # literal pixel matching.
        "unmatched_stereo_points": int(
            stereo_retained_count + stereo_dropped_count
        ),
        "unmatched_stereo_kept": stereo_retained_count,
        "unmatched_stereo_dropped": stereo_dropped_count,
        "stereo_depth_retained_points": stereo_retained_count,
        "stereo_depth_dropped_points": stereo_dropped_count,
        "output_points": int(len(kept_points)),
    }
    return PointCloudAssistedStereoObservation(
        camera_points=kept_points,
        source_indices=kept_indices,
        pointcloud_supported=kept_supported,
        statistics=statistics,
    )


class LidarMappingProcessor:
    """Convert raw LiDAR returns into the rectified left-camera frame."""

    def __init__(
        self,
        *,
        transform_camera_lidar: np.ndarray,
        rectification_rotation: np.ndarray,
        camera_matrix: np.ndarray,
        calibration_width: int,
        calibration_height: int,
        class_count: int,
        min_range_m: float = 0.5,
        max_range_m: float = 30.0,
        voxel_size_m: float = 0.10,
        max_points: int = 5000,
    ) -> None:
        transform = np.asarray(
            transform_camera_lidar, dtype=np.float64
        ).reshape(4, 4)
        rectification = np.asarray(
            rectification_rotation, dtype=np.float64
        ).reshape(3, 3)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(transform).all():
            raise ValueError("LiDAR-to-camera transform must be finite")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("LiDAR-to-camera transform has an invalid last row")
        for name, rotation in (
            ("LiDAR-to-camera", transform[:3, :3]),
            ("camera rectification", rectification),
        ):
            error = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
            if error > 1e-3 or np.linalg.det(rotation) <= 0.0:
                raise ValueError(f"{name} rotation is invalid (error={error})")
        if int(calibration_width) <= 0 or int(calibration_height) <= 0:
            raise ValueError("Calibration image dimensions must be positive")
        if int(class_count) <= 0:
            raise ValueError("class_count must be positive")
        if float(min_range_m) < 0.0 or float(max_range_m) <= float(min_range_m):
            raise ValueError("LiDAR range limits are invalid")
        if float(voxel_size_m) < 0.0:
            raise ValueError("voxel_size_m cannot be negative")
        if int(max_points) < 0:
            raise ValueError("max_points cannot be negative")

        # Raw calibration maps LiDAR -> original Cam_L.  The mapper consumes
        # rectified stereo, so compose Cam_L -> Cam_L_Rect once here.
        self.rotation_camera_lidar = (
            rectification @ transform[:3, :3]
        )
        self.translation_camera_lidar = (
            rectification @ transform[:3, 3]
        )
        self.sensor_origin_camera = self.translation_camera_lidar.copy()
        self.transform_camera_lidar = np.eye(4, dtype=np.float64)
        self.transform_camera_lidar[:3, :3] = self.rotation_camera_lidar
        self.transform_camera_lidar[:3, 3] = self.translation_camera_lidar
        self.camera_matrix = camera_matrix.copy()
        self.calibration_width = int(calibration_width)
        self.calibration_height = int(calibration_height)
        self.class_count = int(class_count)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.voxel_size_m = float(voxel_size_m)
        self.max_points = int(max_points)

    @classmethod
    def from_calibration(
        cls,
        calibration: Dict,
        *,
        extrinsics_path: Optional[Path] = None,
        class_count: int,
        min_range_m: float,
        max_range_m: float,
        voxel_size_m: float,
        max_points: int,
    ) -> "LidarMappingProcessor":
        transform_camera_lidar = (
            load_rectified_lidar_to_camera_transform(
                calibration,
                extrinsics_path=extrinsics_path,
            )
        )
        camera = calibration["camera"]
        camera_matrix = np.array(
            [
                [camera["fx"], 0.0, camera["cx"]],
                [0.0, camera["fy"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return cls(
            # The shared loader already composes left-camera rectification.
            transform_camera_lidar=transform_camera_lidar,
            rectification_rotation=np.eye(3, dtype=np.float64),
            camera_matrix=camera_matrix,
            calibration_width=int(camera["width"]),
            calibration_height=int(camera["height"]),
            class_count=class_count,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            voxel_size_m=voxel_size_m,
            max_points=max_points,
        )

    def _filter_and_sample(
        self,
        points_lidar: np.ndarray,
        *,
        sensor_origin_lidar: Optional[np.ndarray] = None,
    ) -> tuple:
        points = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
        origin = (
            np.zeros(3, dtype=np.float64)
            if sensor_origin_lidar is None
            else np.asarray(sensor_origin_lidar, dtype=np.float64).reshape(3)
        )
        if not np.isfinite(origin).all():
            raise ValueError("LiDAR sensor origin must be finite")
        input_count = len(points)
        finite = np.isfinite(points).all(axis=1)
        ranges = np.linalg.norm(points - origin, axis=1)
        valid = (
            finite
            & (ranges >= self.min_range_m)
            & (ranges <= self.max_range_m)
        )
        points = points[valid]
        range_filtered_count = len(points)

        if self.voxel_size_m > 0.0 and len(points):
            voxel = np.floor(points / self.voxel_size_m).astype(np.int64)
            _, unique_indices = np.unique(voxel, axis=0, return_index=True)
            points = points[np.sort(unique_indices)]
        voxel_filtered_count = len(points)

        if self.max_points > 0 and len(points) > self.max_points:
            indices = np.linspace(
                0, len(points) - 1, self.max_points, dtype=np.int64
            )
            points = points[indices]
        return points, {
            "input_points": int(input_count),
            "range_filtered_points": int(range_filtered_count),
            "voxel_filtered_points": int(voxel_filtered_count),
            "selected_points": int(len(points)),
        }

    def make_observation(
        self,
        points_lidar: np.ndarray,
        semantic_image: np.ndarray,
        *,
        sensor_origin_lidar: Optional[np.ndarray] = None,
    ) -> LidarObservation:
        origin_lidar = (
            np.zeros(3, dtype=np.float64)
            if sensor_origin_lidar is None
            else np.asarray(sensor_origin_lidar, dtype=np.float64).reshape(3)
        )
        points_lidar, statistics = self._filter_and_sample(
            points_lidar,
            sensor_origin_lidar=origin_lidar,
        )
        points_camera = (
            points_lidar @ self.rotation_camera_lidar.T
            + self.translation_camera_lidar
        )
        sensor_origin_camera = (
            origin_lidar @ self.rotation_camera_lidar.T
            + self.translation_camera_lidar
        )
        semantic = np.asarray(semantic_image, dtype=np.uint8)
        if semantic.ndim != 2:
            raise ValueError(
                f"LiDAR semantic projection expects mono8 HxW, got {semantic.shape}"
            )
        height, width = semantic.shape
        scale_x = width / float(self.calibration_width)
        scale_y = height / float(self.calibration_height)
        fx = float(self.camera_matrix[0, 0]) * scale_x
        fy = float(self.camera_matrix[1, 1]) * scale_y
        cx = float(self.camera_matrix[0, 2]) * scale_x
        cy = float(self.camera_matrix[1, 2]) * scale_y

        labels = np.zeros(len(points_camera), dtype=np.uint8)
        forward = points_camera[:, 2] > 0.1
        indices = np.flatnonzero(forward)
        if len(indices):
            z = points_camera[indices, 2]
            u = np.rint(fx * points_camera[indices, 0] / z + cx).astype(np.int64)
            v = np.rint(fy * points_camera[indices, 1] / z + cy).astype(np.int64)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            projected_indices = indices[inside]
            projected_labels = semantic[v[inside], u[inside]]
            valid_labels = projected_labels < self.class_count
            labels[projected_indices[valid_labels]] = projected_labels[valid_labels]
            projected_count = int(np.count_nonzero(valid_labels))
        else:
            projected_count = 0

        statistics = dict(statistics)
        statistics.update(
            {
                "semantic_projected_points": projected_count,
                "unknown_semantic_points": int(np.count_nonzero(labels == 0)),
            }
        )
        return LidarObservation(
            camera_points=points_camera,
            semantic_labels=labels,
            sensor_origin_camera=sensor_origin_camera,
            statistics=statistics,
        )
