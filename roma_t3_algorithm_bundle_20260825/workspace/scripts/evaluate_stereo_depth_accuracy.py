#!/usr/bin/env python3
"""Evaluate the sparse RoMa stereo depths against organized ``depth_gt``.

The evaluator deliberately exercises the same stereo rectification, RoMa
matcher settings, confidence threshold, epipolar gate, and triangulation used
by the ROS 2 mapper.  LiDAR-derived ``depth_gt`` is used only as an offline
reference; it is never fed to localisation or map construction.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch


def _percentile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, quantile))


def _error_metrics(estimate: np.ndarray, reference: np.ndarray) -> Dict:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(estimate)
        & np.isfinite(reference)
        & (estimate > 0.0)
        & (reference > 0.0)
    )
    estimate = estimate[valid]
    reference = reference[valid]
    if estimate.size == 0:
        return {"count": 0}

    signed = estimate - reference
    absolute = np.abs(signed)
    relative = absolute / reference
    ratio = np.maximum(estimate / reference, reference / estimate)
    squared = signed * signed
    log_error = np.log(estimate) - np.log(reference)
    result = {
        "count": int(estimate.size),
        "reference_depth_m": {
            "min": float(reference.min()),
            "median": float(np.median(reference)),
            "p95": _percentile(reference, 0.95),
            "max": float(reference.max()),
        },
        "estimated_depth_m": {
            "min": float(estimate.min()),
            "median": float(np.median(estimate)),
            "p95": _percentile(estimate, 0.95),
            "max": float(estimate.max()),
        },
        "signed_error_m": {
            "mean_bias": float(signed.mean()),
            "median_bias": float(np.median(signed)),
        },
        "absolute_error_m": {
            "mae": float(absolute.mean()),
            "median": float(np.median(absolute)),
            "p95": _percentile(absolute, 0.95),
            "max": float(absolute.max()),
            "rmse": float(np.sqrt(squared.mean())),
        },
        "standard_depth_metrics": {
            "abs_rel": float(relative.mean()),
            "median_abs_rel": float(np.median(relative)),
            "p95_abs_rel": _percentile(relative, 0.95),
            "sq_rel_m": float((squared / reference).mean()),
            "rmse_log": float(np.sqrt(np.mean(log_error * log_error))),
            "delta_lt_1_25": float(np.mean(ratio < 1.25)),
            "delta_lt_1_25_squared": float(np.mean(ratio < 1.25 ** 2)),
            "delta_lt_1_25_cubed": float(np.mean(ratio < 1.25 ** 3)),
        },
        "within_absolute_error": {
            "0.10_m": float(np.mean(absolute <= 0.10)),
            "0.20_m": float(np.mean(absolute <= 0.20)),
            "0.50_m": float(np.mean(absolute <= 0.50)),
            "1.00_m": float(np.mean(absolute <= 1.00)),
        },
    }
    return result


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    map_x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    map_y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        np.asarray(image, dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return sampled.reshape(-1).astype(np.float64)


def _reference_depth_in_rectified_camera(
    depth_raw: np.ndarray,
    rectified_keypoints: np.ndarray,
    *,
    depth_scale_to_m: float,
    rectification: Dict,
    left_maps,
) -> np.ndarray:
    """Sample original-camera depth and rotate it into the rectified camera."""
    keypoints = np.asarray(rectified_keypoints, dtype=np.float64).reshape(-1, 2)
    map_x, map_y = left_maps
    source_x = _bilinear_sample(map_x, keypoints[:, 0], keypoints[:, 1])
    source_y = _bilinear_sample(map_y, keypoints[:, 0], keypoints[:, 1])
    source_depth = _bilinear_sample(
        np.asarray(depth_raw, dtype=np.float32) * float(depth_scale_to_m),
        source_x,
        source_y,
    )

    source_k = np.asarray(rectification["left_k"], dtype=np.float64)
    rotation = np.asarray(rectification["left_r"], dtype=np.float64)
    source_points = np.column_stack(
        (
            (source_x - source_k[0, 2]) * source_depth / source_k[0, 0],
            (source_y - source_k[1, 2]) * source_depth / source_k[1, 1],
            source_depth,
        )
    )
    rectified_points = source_points @ rotation.T
    return rectified_points[:, 2]


def _depth_band_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    bands: Iterable,
) -> Dict:
    result = {}
    for lower, upper, label in bands:
        mask = reference >= lower
        if upper is not None:
            mask &= reference < upper
        result[label] = _error_metrics(estimate[mask], reference[mask])
    return result


def _threshold_subset_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    values: np.ndarray,
    thresholds: Iterable,
    *,
    keep_below: bool,
) -> Dict:
    result = {}
    for threshold in thresholds:
        mask = values <= threshold if keep_below else values >= threshold
        result[str(threshold)] = _error_metrics(estimate[mask], reference[mask])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--algorithm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coarse-res", type=int, default=560)
    parser.add_argument("--upsample-res", type=int, default=560)
    parser.add_argument("--stereo-matches", type=int, default=5000)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Evaluate every Nth organized stereo frame (default: 5).",
    )
    parser.add_argument(
        "--max-evaluated-frames",
        type=int,
        default=0,
        help="Optional cap after applying frame stride; zero means no cap.",
    )
    parser.add_argument(
        "--evaluate-pointcloud-assist",
        action="store_true",
        help=(
            "Also evaluate the mapper's time-aligned PointCloud2-assisted "
            "stereo-depth policy using lidar_raw."
        ),
    )
    parser.add_argument("--assist-pixel-radius", type=float, default=2.0)
    parser.add_argument(
        "--assist-unmatched-stereo-max-depth-m", type=float, default=5.0
    )
    parser.add_argument(
        "--assist-replace-stereo-min-depth-m", type=float, default=5.0
    )
    parser.add_argument(
        "--assist-lidar-max-time-offset-sec", type=float, default=0.15
    )
    parser.add_argument(
        "--assist-lidar-min-range-m", type=float, default=0.5
    )
    parser.add_argument(
        "--assist-lidar-max-range-m", type=float, default=30.0
    )
    parser.add_argument(
        "--assist-lidar-voxel-size-m", type=float, default=0.10
    )
    parser.add_argument(
        "--assist-lidar-max-points", type=int, default=20000
    )
    args = parser.parse_args()

    sequence_dir = args.sequence_dir.expanduser().resolve()
    algorithm_root = args.algorithm_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    if not algorithm_root.is_dir():
        raise FileNotFoundError(algorithm_root)

    sys.path.insert(0, str(algorithm_root))
    from roma_vo import RoMaMatcher, StereoTriangulator
    from t1_dataset_sim.dataset_formats import (
        StereoRectifier,
        load_calibration,
        load_organized_stereo_records,
    )
    if args.evaluate_pointcloud_assist:
        from builtin_interfaces.msg import Time
        from t1_dataset_sim.lidar_io import (
            load_organized_lidar_records,
            read_binary_pcd_message,
        )
        from t3_semantic_mapping.lidar_mapping import (
            LidarMappingProcessor,
            assist_stereo_depth_from_pointcloud,
            pointcloud2_xyz_array,
        )

    manifest = json.loads(
        (sequence_dir / "manifest.json").read_text(encoding="utf-8")
    )
    depth_description = manifest.get("depth_gt", {})
    if not depth_description.get("available", False):
        raise RuntimeError(f"depth_gt is unavailable in {sequence_dir}")
    depth_scale = float(depth_description["scale_to_m"])

    calibration = load_calibration(sequence_dir)
    rectifier = StereoRectifier(calibration, enabled=True)
    camera = calibration["camera"]
    intrinsic = np.array(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangulator = StereoTriangulator(
        intrinsic,
        float(calibration["baseline"]),
        min_depth=float(calibration.get("depth", {}).get("min", 0.1)),
        max_depth=float(calibration.get("depth", {}).get("max", 100.0)),
        epipolar_threshold=float(
            calibration.get("stereo", {}).get("epipolar_threshold_px", 2.0)
        ),
    )

    records, pairing = load_organized_stereo_records(
        sequence_dir, max_abs_skew_ns=20_000_000
    )
    selected = records[:: args.frame_stride]
    if args.max_evaluated_frames > 0:
        selected = selected[: args.max_evaluated_frames]

    lidar_records = []
    lidar_processor = None
    if args.evaluate_pointcloud_assist:
        lidar_records = load_organized_lidar_records(sequence_dir)
        lidar_processor = LidarMappingProcessor.from_calibration(
            calibration,
            class_count=5,
            min_range_m=args.assist_lidar_min_range_m,
            max_range_m=args.assist_lidar_max_range_m,
            voxel_size_m=args.assist_lidar_voxel_size_m,
            max_points=args.assist_lidar_max_points,
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    matcher = RoMaMatcher(
        device=args.device,
        coarse_res=args.coarse_res,
        upsample_res=args.upsample_res,
    )

    estimated_parts: List[np.ndarray] = []
    reference_parts: List[np.ndarray] = []
    confidence_parts: List[np.ndarray] = []
    epipolar_residual_parts: List[np.ndarray] = []
    assisted_estimated_parts: List[np.ndarray] = []
    assisted_reference_parts: List[np.ndarray] = []
    assisted_support_parts: List[np.ndarray] = []
    supported_stereo_before_parts: List[np.ndarray] = []
    supported_reference_parts: List[np.ndarray] = []
    assisted_statistics = []
    per_frame = []
    start = time.perf_counter()
    for ordinal, record in enumerate(selected, start=1):
        frame_start = time.perf_counter()
        left = cv2.imread(str(record.left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(record.right_path), cv2.IMREAD_COLOR)
        depth_path = sequence_dir / "depth_gt" / f"{record.frame_index}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if left is None or right is None or depth is None:
            raise RuntimeError(
                f"Missing image/depth for organized frame {record.frame_index}"
            )
        left_rectified, right_rectified = rectifier.rectify(left, right)
        keypoints_left, keypoints_right, confidence = matcher.match(
            left_rectified,
            right_rectified,
            num_matches=args.stereo_matches,
            resize=None,
        )
        confident = confidence > float(args.confidence_threshold)
        keypoints_left = keypoints_left[confident]
        keypoints_right = keypoints_right[confident]
        points_camera, triangulated_indices = triangulator.triangulate(
            keypoints_left, keypoints_right
        )
        valid_left = keypoints_left[triangulated_indices]
        valid_right = keypoints_right[triangulated_indices]
        triangulated_confidence = confidence[confident][triangulated_indices]
        epipolar_residual = np.abs(valid_left[:, 1] - valid_right[:, 1])
        reference_depth = _reference_depth_in_rectified_camera(
            depth,
            valid_left,
            depth_scale_to_m=depth_scale,
            rectification=calibration["rectification"],
            left_maps=rectifier._left_maps,
        )
        frame_assisted_metrics = None
        frame_assisted_statistics = None
        if args.evaluate_pointcloud_assist:
            lidar_record = min(
                lidar_records,
                key=lambda item: abs(
                    int(item.timestamp_ns) - int(record.left_timestamp_ns)
                ),
            )
            offset_sec = (
                int(lidar_record.timestamp_ns)
                - int(record.left_timestamp_ns)
            ) * 1e-9
            if abs(offset_sec) <= args.assist_lidar_max_time_offset_sec:
                stamp = Time(
                    sec=int(lidar_record.timestamp_ns // 1_000_000_000),
                    nanosec=int(lidar_record.timestamp_ns % 1_000_000_000),
                )
                cloud_message = read_binary_pcd_message(
                    lidar_record.path,
                    stamp=stamp,
                    frame_id=lidar_record.frame_id,
                )
                cloud = pointcloud2_xyz_array(cloud_message)
                cloud_observation = lidar_processor.make_observation(
                    cloud,
                    np.zeros(left_rectified.shape[:2], dtype=np.uint8),
                )
                assisted = assist_stereo_depth_from_pointcloud(
                    keypoints=valid_left,
                    stereo_camera_points=points_camera,
                    pointcloud_camera_points=cloud_observation.camera_points,
                    camera_matrix=intrinsic,
                    image_width=left_rectified.shape[1],
                    image_height=left_rectified.shape[0],
                    pixel_radius=args.assist_pixel_radius,
                    unmatched_stereo_max_depth_m=(
                        args.assist_unmatched_stereo_max_depth_m
                    ),
                    replace_stereo_min_depth_m=(
                        args.assist_replace_stereo_min_depth_m
                    ),
                )
                assisted_reference = reference_depth[
                    assisted.source_indices
                ]
                assisted_estimate = assisted.camera_points[:, 2]
                stereo_before_assistance = points_camera[
                    assisted.source_indices, 2
                ]
                assisted_valid = (
                    np.isfinite(assisted_reference)
                    & np.isfinite(assisted_estimate)
                    & (assisted_reference > 0.1)
                    & (assisted_estimate > 0.1)
                )
                assisted_estimate = assisted_estimate[assisted_valid]
                assisted_reference = assisted_reference[assisted_valid]
                assisted_support = assisted.pointcloud_supported[
                    assisted_valid
                ]
                stereo_before_assistance = stereo_before_assistance[
                    assisted_valid
                ]
                assisted_estimated_parts.append(assisted_estimate)
                assisted_reference_parts.append(assisted_reference)
                assisted_support_parts.append(assisted_support)
                supported_stereo_before_parts.append(
                    stereo_before_assistance[assisted_support]
                )
                supported_reference_parts.append(
                    assisted_reference[assisted_support]
                )
                frame_assisted_metrics = _error_metrics(
                    assisted_estimate, assisted_reference
                )
                frame_assisted_statistics = dict(assisted.statistics)
                frame_assisted_statistics.update(
                    {
                        "lidar_time_offset_sec": float(offset_sec),
                        "lidar_input_points": int(len(cloud)),
                        "lidar_selected_points": int(
                            cloud_observation.statistics["selected_points"]
                        ),
                    }
                )
                assisted_statistics.append(frame_assisted_statistics)
        estimated_depth = points_camera[:, 2]
        valid = (
            np.isfinite(reference_depth)
            & np.isfinite(estimated_depth)
            & (reference_depth > 0.1)
            & (estimated_depth > 0.1)
        )
        estimated_depth = estimated_depth[valid]
        reference_depth = reference_depth[valid]
        estimated_parts.append(estimated_depth)
        reference_parts.append(reference_depth)
        confidence_parts.append(triangulated_confidence[valid])
        epipolar_residual_parts.append(epipolar_residual[valid])
        frame_metrics = _error_metrics(estimated_depth, reference_depth)
        per_frame.append(
            {
                "frame_index": record.frame_index,
                "left_timestamp_ns": int(record.left_timestamp_ns),
                "stereo_skew_ms": float(record.skew_ns) / 1.0e6,
                "sampled_matches": int(len(confidence)),
                "confidence_filtered_matches": int(confident.sum()),
                "triangulated_points": int(len(points_camera)),
                "evaluated_points": int(valid.sum()),
                "absolute_error_m": frame_metrics.get("absolute_error_m", {}),
                "standard_depth_metrics": frame_metrics.get(
                    "standard_depth_metrics", {}
                ),
                "elapsed_s": float(time.perf_counter() - frame_start),
                "pointcloud_assisted": (
                    {
                        "statistics": frame_assisted_statistics,
                        "absolute_error_m": (
                            frame_assisted_metrics or {}
                        ).get("absolute_error_m", {}),
                    }
                    if frame_assisted_metrics is not None
                    else None
                ),
            }
        )
        print(
            f"[{ordinal:03d}/{len(selected):03d}] frame={record.frame_index} "
            f"points={int(valid.sum())} "
            f"MAE={frame_metrics.get('absolute_error_m', {}).get('mae', float('nan')):.3f}m "
            f"RMSE={frame_metrics.get('absolute_error_m', {}).get('rmse', float('nan')):.3f}m",
            flush=True,
        )

    estimate = np.concatenate(estimated_parts) if estimated_parts else np.zeros(0)
    reference = np.concatenate(reference_parts) if reference_parts else np.zeros(0)
    evaluated_confidence = (
        np.concatenate(confidence_parts) if confidence_parts else np.zeros(0)
    )
    evaluated_epipolar_residual = (
        np.concatenate(epipolar_residual_parts)
        if epipolar_residual_parts
        else np.zeros(0)
    )
    assisted_estimate = (
        np.concatenate(assisted_estimated_parts)
        if assisted_estimated_parts
        else np.zeros(0)
    )
    assisted_reference = (
        np.concatenate(assisted_reference_parts)
        if assisted_reference_parts
        else np.zeros(0)
    )
    assisted_support = (
        np.concatenate(assisted_support_parts)
        if assisted_support_parts
        else np.zeros(0, dtype=bool)
    )
    supported_stereo_before = (
        np.concatenate(supported_stereo_before_parts)
        if supported_stereo_before_parts
        else np.zeros(0)
    )
    supported_reference = (
        np.concatenate(supported_reference_parts)
        if supported_reference_parts
        else np.zeros(0)
    )
    report = {
        "status": "ok" if estimate.size else "no_valid_points",
        "evaluation_scope": (
            "Sparse RoMa stereo depth after the same rectification and gates "
            "used by the ROS 2 mapper"
        ),
        "reference_note": (
            "depth_gt is a LiDAR-derived enhanced offline proxy, not certified "
            "ground truth, and was not used as an algorithm input"
        ),
        "sequence_dir": str(sequence_dir),
        "algorithm_root": str(algorithm_root),
        "parameters": {
            "device": args.device,
            "coarse_res": args.coarse_res,
            "upsample_res": args.upsample_res,
            "stereo_matches": args.stereo_matches,
            "confidence_threshold": args.confidence_threshold,
            "epipolar_threshold_px": triangulator.epipolar_threshold,
            "min_depth_m": triangulator.min_depth,
            "max_depth_m": triangulator.max_depth,
            "frame_stride": args.frame_stride,
            "depth_gt_scale_to_m": depth_scale,
            "evaluate_pointcloud_assist": bool(
                args.evaluate_pointcloud_assist
            ),
            "assist_pixel_radius": args.assist_pixel_radius,
            "assist_unmatched_stereo_max_depth_m": (
                args.assist_unmatched_stereo_max_depth_m
            ),
            "assist_replace_stereo_min_depth_m": (
                args.assist_replace_stereo_min_depth_m
            ),
            "assist_lidar_max_time_offset_sec": (
                args.assist_lidar_max_time_offset_sec
            ),
            "assist_lidar_min_range_m": args.assist_lidar_min_range_m,
            "assist_lidar_max_range_m": args.assist_lidar_max_range_m,
            "assist_lidar_voxel_size_m": args.assist_lidar_voxel_size_m,
            "assist_lidar_max_points": args.assist_lidar_max_points,
        },
        "calibration": {
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "baseline_m": float(calibration["baseline"]),
        },
        "counts": {
            "dataset_paired_stereo_frames": int(len(records)),
            "evaluated_frames": int(len(selected)),
            "evaluated_depth_points": int(estimate.size),
            "pairing_skipped_for_skew": int(pairing.skipped_for_skew),
            "sampled_matches": int(sum(v["sampled_matches"] for v in per_frame)),
            "confidence_filtered_matches": int(
                sum(v["confidence_filtered_matches"] for v in per_frame)
            ),
            "triangulated_points": int(
                sum(v["triangulated_points"] for v in per_frame)
            ),
        },
        "all_points": _error_metrics(estimate, reference),
        "pointcloud_assisted": (
            {
                "reference_note": (
                    "Raw LiDAR and depth_gt are correlated products; this "
                    "comparison is a diagnostic, not independent truth."
                ),
                "evaluated_frames": int(len(assisted_statistics)),
                "all_output_points": _error_metrics(
                    assisted_estimate, assisted_reference
                ),
                "pointcloud_supported_points": _error_metrics(
                    assisted_estimate[assisted_support],
                    assisted_reference[assisted_support],
                ),
                "stereo_before_correction_on_supported_points": (
                    _error_metrics(
                        supported_stereo_before, supported_reference
                    )
                ),
                "retained_unmatched_stereo_points": _error_metrics(
                    assisted_estimate[~assisted_support],
                    assisted_reference[~assisted_support],
                ),
                "by_reference_depth_band": _depth_band_metrics(
                    assisted_estimate,
                    assisted_reference,
                    (
                        (0.0, 5.0, "0_to_5_m"),
                        (5.0, 10.0, "5_to_10_m"),
                        (10.0, 20.0, "10_to_20_m"),
                        (20.0, None, "20_m_and_farther"),
                    ),
                ),
                "per_frame_statistics": assisted_statistics,
            }
            if args.evaluate_pointcloud_assist
            else None
        ),
        "by_reference_depth_band": _depth_band_metrics(
            estimate,
            reference,
            (
                (0.0, 5.0, "0_to_5_m"),
                (5.0, 10.0, "5_to_10_m"),
                (10.0, 20.0, "10_to_20_m"),
                (20.0, None, "20_m_and_farther"),
            ),
        ),
        "diagnostic_filter_subsets": {
            "note": (
                "These are post-hoc subsets of the same matches, intended to "
                "select safer mapping gates; they are not additional runs."
            ),
            "by_minimum_match_confidence": _threshold_subset_metrics(
                estimate,
                reference,
                evaluated_confidence,
                (0.5, 0.7, 0.8, 0.9, 0.95),
                keep_below=False,
            ),
            "by_maximum_epipolar_residual_px": _threshold_subset_metrics(
                estimate,
                reference,
                evaluated_epipolar_residual,
                (0.5, 1.0, 2.0, 3.0, 5.0),
                keep_below=True,
            ),
            "by_maximum_estimated_depth_m": _threshold_subset_metrics(
                estimate,
                reference,
                estimate,
                (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 100.0),
                keep_below=True,
            ),
        },
        "per_frame": per_frame,
        "runtime": {
            "total_s": float(time.perf_counter() - start),
            "mean_per_evaluated_frame_s": float(
                (time.perf_counter() - start) / max(len(selected), 1)
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return 0 if estimate.size else 2


if __name__ == "__main__":
    raise SystemExit(main())
