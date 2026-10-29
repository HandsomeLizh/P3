#!/usr/bin/env python3
"""Evaluate raw or temporally aligned LiDAR against organized ``depth_gt``.

This is an offline diagnostic for the optional mapper input.  It uses the
same nearest-timestamp rule, LiDAR-to-left-camera extrinsic, stereo
rectification, range gate, and camera projection as the ROS 2 mapper.  The
default remains the historical uncompensated baseline. ``per_point`` uses
the saved FAST-LIO trajectory only as the motion source for the same temporal
adapter used online; it never treats that estimator output as ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from builtin_interfaces.msg import Time


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t1_dataset_sim"))
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stereo_depth_accuracy import (
    _depth_band_metrics,
    _error_metrics,
    _reference_depth_in_rectified_camera,
)
from t1_dataset_sim.dataset_formats import (
    StereoRectifier,
    load_calibration,
    load_organized_stereo_records,
)
from t1_dataset_sim.lidar_io import (
    load_organized_lidar_records,
    read_binary_pcd_message,
)
from t3_semantic_mapping.lidar_mapping import (
    LidarMappingProcessor,
    pointcloud2_xyz_array,
    pointcloud2_xyz_time_arrays,
)
from t3_semantic_mapping.lidar_temporal_alignment import (
    LidarTemporalConfiguration,
    align_lidar_points_to_reference,
)
from t3_semantic_mapping.localization import ExternalOdometryAdapter
from t3_semantic_mapping.localization.calibration import (
    compose_lio_child_to_camera_transform,
    row_major_transform,
)
from t1_dataset_sim.fastlio_io import load_organized_fastlio_odometry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-evaluated-frames", type=int, default=0)
    parser.add_argument("--first-frame-index", type=int, default=0)
    parser.add_argument("--max-time-offset-sec", type=float, default=0.15)
    parser.add_argument("--max-stereo-skew-ms", type=float, default=20.0)
    parser.add_argument(
        "--temporal-mode", choices=("none", "per_point"), default="none"
    )
    parser.add_argument("--point-time-field", default="time")
    parser.add_argument("--point-time-unit", default="sec")
    parser.add_argument("--point-time-max-abs-sec", type=float, default=0.20)
    parser.add_argument("--deskew-time-bin-sec", type=float, default=0.001)
    parser.add_argument(
        "--lio-child-to-lidar-transform",
        type=float,
        nargs=16,
        metavar=(
            "R00", "R01", "R02", "TX",
            "R10", "R11", "R12", "TY",
            "R20", "R21", "R22", "TZ",
            "Z0", "Z1", "Z2", "ONE",
        ),
        help=(
            "Required in per_point mode: row-major T_lidar_child matching "
            "the saved FAST-LIO child/body pose"
        ),
    )
    parser.add_argument("--min-range-m", type=float, default=0.5)
    parser.add_argument("--max-range-m", type=float, default=30.0)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_time_offset_sec < 0.0:
        raise ValueError("--max-time-offset-sec cannot be negative")
    if args.max_stereo_skew_ms < 0.0:
        raise ValueError("--max-stereo-skew-ms cannot be negative")
    sequence_dir = args.sequence_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    manifest = json.loads(
        (sequence_dir / "manifest.json").read_text(encoding="utf-8")
    )
    depth_description = manifest.get("depth_gt", {})
    if not depth_description.get("available", False):
        raise RuntimeError(f"depth_gt is unavailable in {sequence_dir}")
    depth_scale = float(depth_description["scale_to_m"])

    calibration = load_calibration(sequence_dir)
    rectifier = StereoRectifier(calibration, enabled=True)
    processor = LidarMappingProcessor.from_calibration(
        calibration,
        class_count=5,
        min_range_m=float(args.min_range_m),
        max_range_m=float(args.max_range_m),
        voxel_size_m=0.0,
        max_points=0,
    )
    temporal_configuration = LidarTemporalConfiguration(
        mode=str(args.temporal_mode),
        max_point_time_offset_sec=float(args.point_time_max_abs_sec),
        pose_time_bin_sec=float(args.deskew_time_bin_sec),
    )
    external_adapter = None
    if temporal_configuration.requires_external_pose:
        if args.lio_child_to_lidar_transform is None:
            raise ValueError(
                "--lio-child-to-lidar-transform is required in per_point mode"
            )
        transform_lidar_child = row_major_transform(
            args.lio_child_to_lidar_transform,
            name="evaluation LIO child-to-LiDAR transform",
        )
        transform_camera_child = compose_lio_child_to_camera_transform(
            processor.transform_camera_lidar,
            transform_lidar_child,
        )
        external_adapter = ExternalOdometryAdapter(
            transform_camera_child=transform_camera_child,
        )
        for record in load_organized_fastlio_odometry(sequence_dir):
            external_adapter.add_components(
                timestamp=record.timestamp_ns * 1e-9,
                parent_frame="camera_init",
                child_frame="body",
                position_xyz=record.position,
                quaternion_xyzw=record.orientation_xyzw,
                covariance=None,
            )
    camera = calibration["camera"]
    width = int(camera["width"])
    height = int(camera["height"])
    rectified_camera = rectifier.camera_parameters(width, height)
    fx = float(rectified_camera["fx"])
    fy = float(rectified_camera["fy"])
    cx = float(rectified_camera["cx"])
    cy = float(rectified_camera["cy"])

    stereo_records, _ = load_organized_stereo_records(
        sequence_dir,
        max_abs_skew_ns=int(round(args.max_stereo_skew_ms * 1_000_000.0)),
    )
    eligible = [
        record
        for record in stereo_records
        if int(record.frame_index) >= int(args.first_frame_index)
    ]
    selected = eligible[:: int(args.frame_stride)]
    if args.max_evaluated_frames > 0:
        selected = selected[: int(args.max_evaluated_frames)]
    lidar_records = load_organized_lidar_records(sequence_dir)
    lidar_timestamps = np.asarray(
        [record.timestamp_ns for record in lidar_records], dtype=np.int64
    )

    estimated_parts = []
    reference_parts = []
    offsets = []
    temporal_correction_p95 = []
    temporal_correction_max = []
    per_frame = []
    blank_semantic = np.zeros((height, width), dtype=np.uint8)
    started = time.perf_counter()

    for ordinal, stereo in enumerate(selected, start=1):
        frame_started = time.perf_counter()
        target_ns = int(stereo.left_timestamp_ns)
        position = int(np.searchsorted(lidar_timestamps, target_ns))
        candidates = [
            index
            for index in (position - 1, position)
            if 0 <= index < len(lidar_records)
        ]
        lidar_index = min(
            candidates,
            key=lambda index: abs(int(lidar_timestamps[index]) - target_ns),
        )
        lidar_record = lidar_records[lidar_index]
        offset_sec = (lidar_record.timestamp_ns - target_ns) * 1e-9
        if abs(offset_sec) > float(args.max_time_offset_sec):
            raise RuntimeError(
                f"No LiDAR within the time gate for frame "
                f"{stereo.frame_index}: offset={offset_sec:+.6f}s"
            )

        message = read_binary_pcd_message(
            lidar_record.path,
            stamp=Time(
                sec=int(lidar_record.timestamp_ns // 1_000_000_000),
                nanosec=int(lidar_record.timestamp_ns % 1_000_000_000),
            ),
            frame_id=lidar_record.frame_id,
        )
        point_offsets = None
        if temporal_configuration.requires_point_times:
            cloud, point_offsets = pointcloud2_xyz_time_arrays(
                message,
                time_field=args.point_time_field,
                time_unit=args.point_time_unit,
            )
        else:
            cloud = pointcloud2_xyz_array(message)
        alignment = align_lidar_points_to_reference(
            cloud,
            cloud_timestamp=lidar_record.timestamp_ns * 1e-9,
            reference_timestamp=target_ns * 1e-9,
            transform_camera_sensor=processor.transform_camera_lidar,
            configuration=temporal_configuration,
            pose_query=(
                None
                if external_adapter is None
                else lambda timestamp: (
                    None
                    if (
                        query := external_adapter.query(
                            timestamp,
                            max_time_offset_sec=float(args.max_time_offset_sec),
                            align_to_first_query=False,
                            allow_nearest=False,
                        )
                    )
                    is None
                    else query.pose
                )
            ),
            point_time_offsets_sec=point_offsets,
        )
        observation = processor.make_observation(
            alignment.points_sensor,
            blank_semantic,
            sensor_origin_lidar=alignment.sensor_origin_sensor,
        )
        points = observation.camera_points
        forward_indices = np.flatnonzero(points[:, 2] > 0.1)
        forward = points[forward_indices]
        u = fx * forward[:, 0] / forward[:, 2] + cx
        v = fy * forward[:, 1] / forward[:, 2] + cy
        inside = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        forward_indices = forward_indices[inside]
        u = u[inside]
        v = v[inside]

        # One nearest return per rectified pixel prevents occluded surfaces
        # behind a foreground return from being counted as visible geometry.
        integer_u = np.rint(u).astype(np.int64)
        integer_v = np.rint(v).astype(np.int64)
        inside_rounded = (
            (integer_u >= 0)
            & (integer_u < width)
            & (integer_v >= 0)
            & (integer_v < height)
        )
        forward_indices = forward_indices[inside_rounded]
        u = u[inside_rounded]
        v = v[inside_rounded]
        integer_u = integer_u[inside_rounded]
        integer_v = integer_v[inside_rounded]
        linear = integer_v * width + integer_u
        order = np.argsort(points[forward_indices, 2])
        _, first = np.unique(linear[order], return_index=True)
        selected_indices = forward_indices[order[first]]
        selected_u = u[order[first]]
        selected_v = v[order[first]]

        depth_path = sequence_dir / "depth_gt" / f"{stereo.frame_index}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Could not read {depth_path}")
        keypoints = np.column_stack((selected_u, selected_v))
        reference = _reference_depth_in_rectified_camera(
            depth,
            keypoints,
            depth_scale_to_m=depth_scale,
            rectification=calibration["rectification"],
            left_maps=rectifier._left_maps,
        )
        estimate = points[selected_indices, 2]
        valid = (
            np.isfinite(reference)
            & np.isfinite(estimate)
            & (reference > 0.1)
            & (estimate > 0.1)
        )
        estimate = estimate[valid]
        reference = reference[valid]
        metrics = _error_metrics(estimate, reference)
        estimated_parts.append(estimate)
        reference_parts.append(reference)
        offsets.append(offset_sec)
        temporal_correction_p95.append(
            alignment.statistics["correction_p95_m"]
        )
        temporal_correction_max.append(
            alignment.statistics["correction_max_m"]
        )
        per_frame.append(
            {
                "frame_index": stereo.frame_index,
                "image_timestamp_ns": target_ns,
                "lidar_timestamp_ns": int(lidar_record.timestamp_ns),
                "time_offset_sec": float(offset_sec),
                "raw_points": int(message.width * message.height),
                "temporal_correction_p95_m": float(
                    alignment.statistics["correction_p95_m"]
                ),
                "temporal_correction_max_m": float(
                    alignment.statistics["correction_max_m"]
                ),
                "projected_unique_pixels": int(len(selected_indices)),
                "evaluated_points": int(len(estimate)),
                "absolute_error_m": metrics.get("absolute_error_m", {}),
                "elapsed_s": float(time.perf_counter() - frame_started),
            }
        )
        print(
            f"[{ordinal:03d}/{len(selected):03d}] "
            f"frame={stereo.frame_index} offset={offset_sec:+.3f}s "
            f"points={len(estimate)} "
            f"RMSE={metrics.get('absolute_error_m', {}).get('rmse', float('nan')):.3f}m",
            flush=True,
        )

    estimate = np.concatenate(estimated_parts) if estimated_parts else np.zeros(0)
    reference = (
        np.concatenate(reference_parts) if reference_parts else np.zeros(0)
    )
    report = {
        "status": "ok" if len(estimate) else "no_valid_points",
        "sequence_dir": str(sequence_dir),
        "geometry_source": (
            "raw lidar_raw PCD; no temporal compensation"
            if args.temporal_mode == "none"
            else (
                "raw lidar_raw PCD; point-time compensation and image-time "
                "alignment from saved FastLIO poses (poses are not ground truth)"
            )
        ),
        "temporal_mode": str(args.temporal_mode),
        "frame_stride": int(args.frame_stride),
        "first_frame_index": int(args.first_frame_index),
        "max_stereo_skew_ms": float(args.max_stereo_skew_ms),
        "evaluated_frames": int(len(per_frame)),
        "time_offset_sec": {
            "min": float(np.min(offsets)),
            "mean_abs": float(np.mean(np.abs(offsets))),
            "max": float(np.max(offsets)),
        },
        "temporal_correction_m": {
            "mean_frame_p95": float(np.mean(temporal_correction_p95)),
            "maximum": float(np.max(temporal_correction_max)),
        },
        "all": _error_metrics(estimate, reference),
        "by_reference_depth": _depth_band_metrics(
            estimate,
            reference,
            (
                (0.1, 5.0, "0.1_to_5_m"),
                (5.0, 10.0, "5_to_10_m"),
                (10.0, 20.0, "10_to_20_m"),
            ),
        ),
        "per_frame": per_frame,
        "elapsed_s": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
