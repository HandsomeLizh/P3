#!/usr/bin/env python3
"""ROS 2 realtime telemetry client with a dependency-free web dashboard."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


HISTORY_LIMIT = 600


def stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_arrays(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    q = pose.orientation
    return position, quaternion_matrix(q.x, q.y, q.z, q.w)


def rotation_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class DashboardState:
    def __init__(self, cache_path: Path | None = None) -> None:
        self.lock = threading.RLock()
        self.cache_path = cache_path
        self.data: dict[str, Any] = {
            "dashboard_schema_version": 2,
            "started_unix_sec": time.time(),
            "topics": {},
            "pose": {},
            "timing": {},
            "map": {
                "resolution_m": None,
                "known_cells": 0,
                "observed_area_m2": 0.0,
                "observed_area_km2": 0.0,
                "extent_area_m2": 0.0,
                "revision_messages": 0,
            },
            "storage": {
                "all_map_files_bytes": 0,
                "published_map_files_bytes": 0,
                "gb_per_km2": None,
            },
            "stability_period_sec": 500.0,
            "stability_window": {
                "elapsed_wall_sec": 0.0,
                "required_wall_sec": 500.0,
            },
            "stability_current": None,
            "stability_results": [],
            "history": {
                "position_error": [],
                "pose_processing": [],
                "map_update": [],
            },
            "counters": {
                "estimate_messages": 0,
                "reference_messages": 0,
                "matched_poses": 0,
                "timing_messages": 0,
                "map_messages": 0,
            },
        }
        if cache_path is not None and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("dashboard_schema_version") == 2:
                    self.data = cached
            except (OSError, TypeError, ValueError):
                pass

    def snapshot(self) -> bytes:
        with self.lock:
            return json.dumps(
                self.data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")

    def append_history(self, key: str, row: list[Any]) -> None:
        history = self.data["history"][key]
        history.append(row)
        if len(history) > HISTORY_LIMIT:
            del history[: len(history) - HISTORY_LIMIT]

    def save(self) -> None:
        if self.cache_path is None:
            return
        with self.lock:
            payload = json.dumps(
                self.data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self.cache_path)
        except OSError:
            return


class RunResultsWriter:
    """Append complete realtime samples and periodically write requirement reports."""

    POSE_FIELDS = [
        "sensor_stamp_sec", "reference_stamp_sec",
        "reference_x", "reference_y", "reference_z",
        "estimate_x", "estimate_y", "estimate_z", "position_error_m",
        "reference_roll_deg", "reference_pitch_deg", "reference_yaw_deg",
        "estimate_roll_deg", "estimate_pitch_deg", "estimate_yaw_deg",
        "alignment",
    ]
    TIMING_FIELDS = [
        "sequence", "sensor_stamp_sec", "frame_id", "outcome",
        "processing_time_sec", "start_monotonic_ns", "completion_monotonic_ns",
    ]
    MAP_FIELDS = [
        "sensor_stamp_sec", "resolution_m", "known_cells", "observed_area_m2",
        "observed_area_km2", "extent_area_m2", "length_x_m", "length_y_m",
        "revision_messages",
    ]
    STABILITY_FIELDS = [
        "period_index", "start_stamp_sec", "end_stamp_sec", "elapsed_wall_sec",
        "elapsed_sensor_sec", "normalization_sec", "roll_difference_deg",
        "pitch_difference_deg", "yaw_difference_deg", "stability_deg_per_sec",
        "threshold_deg_per_sec", "status",
    ]

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pose_errors: list[float] = []
        self.pose_times: list[float] = []
        self.map_times: list[float] = []
        self.stability_values: list[float] = []
        self.position_alignment_modes: set[str] = set()
        self.reference_distance_m = 0.0
        self.last_reference_position: np.ndarray | None = None
        self.last_keys: dict[str, float] = {}
        self._ensure_artifact_headers()
        self._load_existing()

    def _path(self, name: str) -> Path:
        return self.output_dir / name

    def _ensure_artifact_headers(self) -> None:
        definitions = {
            "realtime_pose_samples.csv": self.POSE_FIELDS,
            "pose_processing_time.csv": self.TIMING_FIELDS,
            "map_update_time.csv": self.TIMING_FIELDS,
            "map_stats_history.csv": self.MAP_FIELDS,
            "orientation_stability_500s.csv": self.STABILITY_FIELDS,
        }
        for name, fieldnames in definitions.items():
            path = self._path(name)
            if path.exists() and path.stat().st_size > 0:
                continue
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    @staticmethod
    def _number(row: dict[str, str], key: str) -> float | None:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _load_rows(self, name: str) -> list[dict[str, str]]:
        path = self._path(name)
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        except OSError:
            return []

    def _load_existing(self) -> None:
        pose_rows = self._load_rows("realtime_pose_samples.csv")
        for row in pose_rows:
            error = self._number(row, "position_error_m")
            position_values = [self._number(row, key) for key in (
                "reference_x", "reference_y", "reference_z"
            )]
            if error is not None:
                self.pose_errors.append(error)
            if row.get("alignment"):
                self.position_alignment_modes.add(row["alignment"])
            if all(value is not None for value in position_values):
                position = np.asarray(position_values, dtype=np.float64)
                if self.last_reference_position is not None:
                    self.reference_distance_m += float(
                        np.linalg.norm(position - self.last_reference_position)
                    )
                self.last_reference_position = position
        if pose_rows:
            self.last_keys["realtime_pose_samples.csv"] = float(
                pose_rows[-1]["sensor_stamp_sec"]
            )
        for name, target in (
            ("pose_processing_time.csv", self.pose_times),
            ("map_update_time.csv", self.map_times),
        ):
            rows = self._load_rows(name)
            target.extend(
                value for row in rows
                if (value := self._number(row, "processing_time_sec")) is not None
            )
            if rows:
                self.last_keys[name] = float(rows[-1]["sensor_stamp_sec"])
        for name, key in (
            ("map_stats_history.csv", "sensor_stamp_sec"),
            ("orientation_stability_500s.csv", "period_index"),
        ):
            rows = self._load_rows(name)
            if name == "orientation_stability_500s.csv":
                self.stability_values.extend(
                    value for row in rows
                    if (
                        value := self._number(row, "stability_deg_per_sec")
                    ) is not None
                )
            if rows:
                self.last_keys[name] = float(rows[-1][key])

    def _append(
        self,
        name: str,
        fieldnames: list[str],
        row: dict[str, Any],
        unique_key: str,
    ) -> bool:
        key = float(row[unique_key])
        if key <= self.last_keys.get(name, -math.inf):
            return False
        path = self._path(name)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow({field: row.get(field) for field in fieldnames})
            stream.flush()
        self.last_keys[name] = key
        return True

    def append_pose(self, live: dict[str, Any]) -> None:
        reference = np.asarray(live["reference_position_m"], dtype=np.float64)
        estimate = live["estimate_position_m"]
        reference_rpy = live["reference_rpy_deg"]
        estimate_rpy = live["estimate_rpy_deg"]
        row = {
            "sensor_stamp_sec": live["sensor_stamp_sec"],
            "reference_stamp_sec": live["reference_stamp_sec"],
            "reference_x": reference[0], "reference_y": reference[1],
            "reference_z": reference[2], "estimate_x": estimate[0],
            "estimate_y": estimate[1], "estimate_z": estimate[2],
            "position_error_m": live["position_error_m"],
            "reference_roll_deg": reference_rpy[0],
            "reference_pitch_deg": reference_rpy[1],
            "reference_yaw_deg": reference_rpy[2],
            "estimate_roll_deg": estimate_rpy[0],
            "estimate_pitch_deg": estimate_rpy[1],
            "estimate_yaw_deg": estimate_rpy[2],
            "alignment": live["alignment"],
        }
        if not self._append(
            "realtime_pose_samples.csv", self.POSE_FIELDS, row, "sensor_stamp_sec"
        ):
            return
        self.pose_errors.append(float(live["position_error_m"]))
        self.position_alignment_modes.add(str(live["alignment"]))
        if self.last_reference_position is not None:
            self.reference_distance_m += float(
                np.linalg.norm(reference - self.last_reference_position)
            )
        self.last_reference_position = reference

    def append_timing(self, timing: dict[str, Any]) -> None:
        common = {
            "sequence": timing.get("sequence"),
            "sensor_stamp_sec": timing.get("sensor_stamp_sec"),
            "frame_id": timing.get("frame_id"),
            "outcome": timing.get("outcome"),
        }
        for name, duration_key, start_key, end_key, target in (
            (
                "pose_processing_time.csv", "pose_processing_sec",
                "localization_solution_start_monotonic_ns",
                "pose_publish_completion_monotonic_ns", self.pose_times,
            ),
            (
                "map_update_time.csv", "map_update_sec",
                "map_update_start_monotonic_ns",
                "map_publish_completion_monotonic_ns", self.map_times,
            ),
        ):
            duration = finite_float(timing.get(duration_key))
            stamp = finite_float(timing.get("sensor_stamp_sec"))
            if duration is None or stamp is None:
                continue
            row = dict(common)
            row.update({
                "processing_time_sec": duration,
                "start_monotonic_ns": timing.get(start_key),
                "completion_monotonic_ns": timing.get(end_key),
            })
            if self._append(name, self.TIMING_FIELDS, row, "sensor_stamp_sec"):
                target.append(duration)

    def append_map(self, map_state: dict[str, Any]) -> None:
        self._append(
            "map_stats_history.csv", self.MAP_FIELDS, map_state, "sensor_stamp_sec"
        )

    def append_stability(self, result: dict[str, Any]) -> None:
        difference = result["difference_rpy_deg"]
        row = dict(result)
        row.update({
            "roll_difference_deg": difference[0],
            "pitch_difference_deg": difference[1],
            "yaw_difference_deg": difference[2],
        })
        if self._append(
            "orientation_stability_500s.csv",
            self.STABILITY_FIELDS,
            row,
            "period_index",
        ):
            self.stability_values.append(float(result["stability_deg_per_sec"]))

    def write_partial_stability(self, result: dict[str, Any] | None) -> None:
        """Atomically preserve the latest not-yet-500-second stability window."""
        path = self._path("orientation_stability_partial.csv")
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.STABILITY_FIELDS)
            writer.writeheader()
            if result is not None:
                difference = result["difference_rpy_deg"]
                row = dict(result)
                row.update({
                    "roll_difference_deg": difference[0],
                    "pitch_difference_deg": difference[1],
                    "yaw_difference_deg": difference[2],
                })
                writer.writerow({field: row.get(field) for field in self.STABILITY_FIELDS})
        temporary.replace(path)

    @staticmethod
    def _statistics(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "p95": None, "maximum": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "p95": float(np.percentile(array, 95.0)),
            "maximum": float(np.max(array)),
        }

    @staticmethod
    def _status(value: float | None, threshold: float, lower_is_better: bool) -> str:
        if value is None:
            return "INCOMPLETE"
        passed = value < threshold if lower_is_better else value >= threshold
        return "PASS" if passed else "FAIL"

    def export(self, state: DashboardState) -> None:
        snapshot = json.loads(state.snapshot())
        map_state = snapshot.get("map", {})
        storage = snapshot.get("storage", {})
        position_rmse = (
            math.sqrt(sum(value * value for value in self.pose_errors) / len(self.pose_errors))
            if self.pose_errors else None
        )
        position_percent = (
            position_rmse / self.reference_distance_m * 100.0
            if position_rmse is not None and self.reference_distance_m > 0.0 else None
        )
        stability_current = snapshot.get("stability_current")
        if self.stability_values:
            stability_value = max(self.stability_values)
            stability_basis = "complete_500s_periods"
            stability_normalization_sec = 500.0
        elif stability_current:
            stability_value = finite_float(
                stability_current.get("stability_deg_per_sec")
            )
            stability_basis = "current_actual_runtime"
            stability_normalization_sec = finite_float(
                stability_current.get("normalization_sec")
            )
        else:
            stability_value = None
            stability_basis = "no_valid_period"
            stability_normalization_sec = None
        pose_stats = self._statistics(self.pose_times)
        map_time_stats = self._statistics(self.map_times)
        resolution = finite_float(map_state.get("resolution_m"))
        map_area = finite_float(map_state.get("observed_area_km2"))
        storage_rate = finite_float(storage.get("gb_per_km2"))
        requirements = {
            "orientation_stability": {
                "requirement": "每500 s发布姿态稳定度，阈值<0.005 deg/s",
                "value_deg_per_sec": stability_value,
                "complete_period_count": len(self.stability_values),
                "evaluation_basis": stability_basis,
                "normalization_sec": stability_normalization_sec,
                "partial_period_available": bool(stability_current),
                "current_window_wall_sec": snapshot.get("stability_window", {}).get(
                    "elapsed_wall_sec", 0.0
                ),
                "threshold_deg_per_sec": 0.005,
                "status": self._status(stability_value, 0.005, True),
            },
            "position_accuracy": {
                "requirement": "实时真值/估计值及误差，RMSE/D<1%",
                "rmse_m": position_rmse,
                "reference_trajectory_length_m": self.reference_distance_m,
                "rmse_over_distance_percent": position_percent,
                "threshold_percent": 1.0,
                "alignment": sorted(self.position_alignment_modes),
                "status": self._status(position_percent, 1.0, True),
            },
            "pose_processing_time": {
                "requirement": "发布处理时间戳和位姿处理时间，阈值<2 s",
                **pose_stats,
                "threshold_sec": 2.0,
                "status": self._status(pose_stats["maximum"], 2.0, True),
            },
            "map_construction_range": {
                "requirement": "点云建图，分辨率0.2 m，范围>=0.5 km2",
                "resolution_m": resolution,
                "known_cells": map_state.get("known_cells", 0),
                "observed_area_km2": map_area,
                "threshold_km2": 0.5,
                "resolution_status": (
                    "PASS" if resolution is not None and abs(resolution - 0.2) < 1e-9
                    else "FAIL" if resolution is not None else "INCOMPLETE"
                ),
                "status": self._status(map_area, 0.5, False),
            },
            "map_update_time": {
                "requirement": "发布地图更新时间戳和处理时间，阈值<2 s",
                **map_time_stats,
                "threshold_sec": 2.0,
                "status": self._status(map_time_stats["maximum"], 2.0, True),
            },
            "lightweight_map_storage": {
                "requirement": "显示地图文件大小及单位面积存储量，阈值<3 GB/km2",
                "all_map_files_bytes": storage.get("all_map_files_bytes", 0),
                "published_map_files_bytes": storage.get(
                    "published_map_files_bytes", 0
                ),
                "gb_per_km2": storage_rate,
                "threshold_gb_per_km2": 3.0,
                "status": self._status(storage_rate, 3.0, True),
            },
            "global_map_publication": {
                "requirement": "发布全局地图供RViz显示",
                "topic": snapshot.get("topics", {}).get("global_map"),
                "message_count": snapshot.get("counters", {}).get("map_messages", 0),
                "status": (
                    "PASS" if snapshot.get("counters", {}).get("map_messages", 0) > 0
                    else "INCOMPLETE"
                ),
            },
        }
        report = {
            "schema_version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "output_directory": str(self.output_dir),
            "topics": snapshot.get("topics", {}),
            "counters": snapshot.get("counters", {}),
            "requirements": requirements,
            "artifacts": {
                "pose_samples": "realtime_pose_samples.csv",
                "pose_processing": "pose_processing_time.csv",
                "map_update": "map_update_time.csv",
                "map_history": "map_stats_history.csv",
                "orientation_stability": "orientation_stability_500s.csv",
                "orientation_stability_partial": "orientation_stability_partial.csv",
                "global_map_npz": "global_grid_map.npz",
                "global_map_sqlite3": "global_grid_map.sqlite3",
            },
        }
        json_path = self._path("requirements_test_results.json")
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(json_path)
        self._write_markdown(report)

    def _write_markdown(self, report: dict[str, Any]) -> None:
        requirements = report["requirements"]
        def value(name: str) -> str:
            item = requirements[name]
            if name == "orientation_stability":
                result = item["value_deg_per_sec"]
                if result is None:
                    return "无有效运行周期"
                period = item.get("normalization_sec")
                suffix = (
                    f"（实际运行 {period:.3f} s）"
                    if item.get("evaluation_basis") == "current_actual_runtime"
                    and period is not None else "（完整500 s周期）"
                )
                return f"{result:.6f} deg/s {suffix}"
            if name == "position_accuracy":
                result = item["rmse_over_distance_percent"]
                return "无有效真值" if result is None else f"{result:.6f}%"
            if name in ("pose_processing_time", "map_update_time"):
                result = item["maximum"]
                return "无数据" if result is None else f"max={result:.6f} s"
            if name == "map_construction_range":
                result = item["observed_area_km2"]
                return "无地图" if result is None else f"{result:.6f} km2"
            if name == "lightweight_map_storage":
                result = item["gb_per_km2"]
                return "无地图" if result is None else f"{result:.6f} GB/km2"
            return f"{item['message_count']} messages"
        rows = [
            ("定姿稳定度", "orientation_stability", "orientation_stability_500s.csv"),
            ("定位精度", "position_accuracy", "realtime_pose_samples.csv"),
            ("位姿处理时间", "pose_processing_time", "pose_processing_time.csv"),
            ("地图构建范围", "map_construction_range", "map_stats_history.csv"),
            ("地图更新时间", "map_update_time", "map_update_time.csv"),
            ("轻量级地图存储量", "lightweight_map_storage", "global_grid_map.*"),
            ("全局地图发布", "global_map_publication", "/Car/T3/mapping/global_grid_map"),
        ]
        lines = [
            "# 实时运行需求指标测试结果", "",
            f"- 生成时间：`{report['generated_at']}`",
            f"- 输出目录：`{report['output_directory']}`", "",
            "| 需求指标 | 测试结果 | 状态 | 证据文件/话题 |", "|---|---:|---|---|",
        ]
        for label, name, artifact in rows:
            lines.append(
                f"| {label} | {value(name)} | **{requirements[name]['status']}** | `{artifact}` |"
            )
        lines.extend([
            "", "## 判定说明", "",
            "- `PASS`：本次运行有完整数据且满足冻结阈值。",
            "- `FAIL`：本次运行有完整数据但不满足冻结阈值。",
            "- 定姿稳定度不足500 s时，以本次实际运行时长为分母计算并参与判定；达到500 s后使用完整周期结果。",
            "- `INCOMPLETE`：除上述定姿短周期规则外，其他指标数据不足时不作通过判定。",
            "- 网页曲线为实时显示；CSV 文件保存全部实时样本，不受网页显示点数限制。",
            "- 定位误差对齐方式：`"
            + "; ".join(requirements["position_accuracy"]["alignment"])
            + "`。",
        ])
        path = self._path("需求指标测试结果.md")
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)


class T3RealtimeDashboard(Node):
    def __init__(self, args: argparse.Namespace, state: DashboardState) -> None:
        super().__init__("t3_realtime_dashboard")
        self.args = args
        self.state = state
        self.results = RunResultsWriter(args.output_dir)
        self.references: deque[tuple[int, np.ndarray, np.ndarray]] = deque(maxlen=10000)
        self.reference_stamps: deque[int] = deque(maxlen=10000)
        self.matched_positions: deque[
            tuple[int, np.ndarray, np.ndarray]
        ] = deque(maxlen=10000)
        self.align_rotation: np.ndarray | None = None
        self.align_translation: np.ndarray | None = None
        self.attitude_rpy_offset: np.ndarray | None = None
        self.stability_start: tuple[
            int, int, np.ndarray, np.ndarray, np.ndarray
        ] | None = None
        self.stability_index = 0

        live_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        reference_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Odometry, args.estimate_topic, self.on_estimate, live_qos)
        self.create_subscription(PoseStamped, args.reference_topic, self.on_reference, reference_qos)
        self.create_subscription(String, args.timing_topic, self.on_timing, live_qos)
        self.create_subscription(GridMap, args.global_map_topic, self.on_map, map_qos)
        self.live_pub = self.create_publisher(
            String, "/Car/T3/metrics/live_pose_error", live_qos
        )
        self.stability_pub = self.create_publisher(
            String, "/Car/T3/metrics/orientation_stability", live_qos
        )
        self.map_stats_pub = self.create_publisher(
            String, "/Car/T3/metrics/map_stats", live_qos
        )
        self.create_timer(args.storage_period_sec, self.update_storage)
        with self.state.lock:
            self.state.data["stability_period_sec"] = args.stability_period_sec
            self.state.data["stability_window"]["required_wall_sec"] = (
                args.stability_period_sec
            )
            self.state.data["topics"] = {
                "estimate": args.estimate_topic,
                "reference": args.reference_topic,
                "timing": args.timing_topic,
                "global_map": args.global_map_topic,
                "live_pose_error": "/Car/T3/metrics/live_pose_error",
                "orientation_stability": "/Car/T3/metrics/orientation_stability",
                "map_stats": "/Car/T3/metrics/map_stats",
            }
        self.get_logger().info(
            f"Dashboard telemetry ready; HTTP http://{args.host}:{args.port}"
        )

    def on_reference(self, message: PoseStamped) -> None:
        timestamp = stamp_ns(message)
        position, rotation = pose_arrays(message.pose)
        self.references.append((timestamp, position, rotation))
        self.reference_stamps.append(timestamp)
        with self.state.lock:
            self.state.data["counters"]["reference_messages"] += 1

    def nearest_reference(
        self, timestamp: int
    ) -> tuple[int, np.ndarray, np.ndarray] | None:
        if not self.references:
            return None
        stamps = list(self.reference_stamps)
        index = bisect.bisect_left(stamps, timestamp)
        candidates = []
        if index < len(stamps):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        if not candidates:
            return None
        best = min(candidates, key=lambda item: abs(stamps[item] - timestamp))
        reference = list(self.references)[best]
        if abs(reference[0] - timestamp) > int(
            self.args.reference_max_offset_sec * 1e9
        ):
            return None
        return reference

    def on_estimate(self, message: Odometry) -> None:
        timestamp = stamp_ns(message)
        estimate_position, estimate_rotation = pose_arrays(message.pose.pose)
        with self.state.lock:
            self.state.data["counters"]["estimate_messages"] += 1
        reference = self.nearest_reference(timestamp)
        if reference is None:
            return
        reference_timestamp, reference_position, reference_rotation = reference
        self.matched_positions.append(
            (timestamp, estimate_position.copy(), reference_position.copy())
        )
        estimate_points = np.asarray(
            [row[1] for row in self.matched_positions], dtype=np.float64
        )
        reference_points = np.asarray(
            [row[2] for row in self.matched_positions], dtype=np.float64
        )
        self.align_rotation, self.align_translation = self.fit_rigid_alignment(
            estimate_points, reference_points
        )
        assert self.align_translation is not None
        aligned_position = self.align_rotation @ estimate_position + self.align_translation
        position_error = float(np.linalg.norm(aligned_position - reference_position))
        reference_rpy = rotation_rpy_deg(reference_rotation)
        estimate_raw_rpy = rotation_rpy_deg(estimate_rotation)
        if self.attitude_rpy_offset is None:
            self.attitude_rpy_offset = wrap_degrees(reference_rpy - estimate_raw_rpy)
        estimate_rpy = wrap_degrees(estimate_raw_rpy + self.attitude_rpy_offset)
        row_time = timestamp * 1e-9
        live = {
            "sensor_stamp_sec": row_time,
            "reference_stamp_sec": reference_timestamp * 1e-9,
            "reference_position_m": reference_position.tolist(),
            "estimate_position_m": aligned_position.tolist(),
            "position_error_m": position_error,
            "reference_rpy_deg": reference_rpy.tolist(),
            "estimate_rpy_deg": estimate_rpy.tolist(),
            "alignment": (
                "position: online cumulative SE3 (no scale); "
                "attitude: initial RPY offset"
            ),
        }
        with self.state.lock:
            self.state.data["pose"] = live
            self.state.data["counters"]["matched_poses"] += 1
            aligned_points = (
                estimate_points @ self.align_rotation.T + self.align_translation
            )
            errors = np.linalg.norm(aligned_points - reference_points, axis=1)
            self.state.data["history"]["position_error"] = [
                [row[0] * 1e-9, float(error)]
                for row, error in zip(self.matched_positions, errors)
            ][-HISTORY_LIMIT:]
        output = String()
        output.data = json.dumps(live, separators=(",", ":"), allow_nan=False)
        self.live_pub.publish(output)
        self.results.append_pose(live)
        self.update_stability(timestamp, reference_rpy, estimate_raw_rpy)

    @staticmethod
    def fit_rigid_alignment(
        estimate_points: np.ndarray, reference_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if estimate_points.shape != reference_points.shape:
            raise ValueError("estimate/reference point arrays must have equal shapes")
        if estimate_points.ndim != 2 or estimate_points.shape[1] != 3:
            raise ValueError("alignment points must have shape (N, 3)")
        if estimate_points.shape[0] == 0:
            raise ValueError("at least one alignment point is required")
        if estimate_points.shape[0] == 1:
            rotation = np.eye(3, dtype=np.float64)
        else:
            estimate_centered = estimate_points - estimate_points.mean(axis=0)
            reference_centered = reference_points - reference_points.mean(axis=0)
            u_matrix, _, vt_matrix = np.linalg.svd(
                estimate_centered.T @ reference_centered
            )
            rotation = vt_matrix.T @ u_matrix.T
            if np.linalg.det(rotation) < 0.0:
                vt_matrix[-1, :] *= -1.0
                rotation = vt_matrix.T @ u_matrix.T
        translation = (
            reference_points.mean(axis=0)
            - rotation @ estimate_points.mean(axis=0)
        )
        return rotation, translation

    def update_stability(
        self,
        timestamp: int,
        reference_rpy: np.ndarray,
        estimate_rpy: np.ndarray,
    ) -> None:
        wall_now_ns = time.monotonic_ns()
        orientation_error = wrap_degrees(estimate_rpy - reference_rpy)
        if self.stability_start is None:
            self.stability_start = (
                wall_now_ns,
                timestamp,
                reference_rpy.copy(),
                estimate_rpy.copy(),
                orientation_error.copy(),
            )
            return
        (
            start_wall_ns,
            start_stamp,
            start_reference,
            start_estimate,
            start_error,
        ) = self.stability_start
        elapsed_wall = (wall_now_ns - start_wall_ns) * 1e-9
        elapsed_sensor = (timestamp - start_stamp) * 1e-9
        if elapsed_wall <= 0.0:
            return
        reference_change = wrap_degrees(reference_rpy - start_reference)
        estimate_change = wrap_degrees(estimate_rpy - start_estimate)
        difference = wrap_degrees(orientation_error - start_error)
        is_partial = elapsed_wall < self.args.stability_period_sec
        normalization_sec = (
            elapsed_wall if is_partial else self.args.stability_period_sec
        )
        stability = float(np.max(np.abs(difference)) / normalization_sec)
        result = {
            "period_index": self.stability_index + 1,
            "period_type": "partial" if is_partial else "complete",
            "is_partial_period": is_partial,
            "start_stamp_sec": start_stamp * 1e-9,
            "end_stamp_sec": timestamp * 1e-9,
            "elapsed_wall_sec": elapsed_wall,
            "elapsed_sensor_sec": elapsed_sensor,
            "normalization_sec": normalization_sec,
            "reference_change_rpy_deg": reference_change.tolist(),
            "estimate_change_rpy_deg": estimate_change.tolist(),
            "difference_rpy_deg": difference.tolist(),
            "stability_deg_per_sec": stability,
            "threshold_deg_per_sec": 0.005,
            "status": "pass" if stability < 0.005 else "fail",
        }
        with self.state.lock:
            self.state.data["stability_window"] = {
                "elapsed_wall_sec": elapsed_wall,
                "required_wall_sec": self.args.stability_period_sec,
            }
            self.state.data["stability_current"] = result
        output = String()
        output.data = json.dumps(result, separators=(",", ":"), allow_nan=False)
        self.stability_pub.publish(output)
        if is_partial:
            self.results.write_partial_stability(result)
            return
        self.stability_index += 1
        result["period_index"] = self.stability_index
        with self.state.lock:
            self.state.data["stability_results"].append(result)
            self.state.data["stability_results"] = self.state.data[
                "stability_results"
            ][-20:]
            self.state.data["stability_current"] = None
        self.results.append_stability(result)
        self.results.write_partial_stability(None)
        self.stability_start = (
            wall_now_ns,
            timestamp,
            reference_rpy.copy(),
            estimate_rpy.copy(),
            orientation_error.copy(),
        )

    def on_timing(self, message: String) -> None:
        try:
            timing = json.loads(message.data)
        except (TypeError, ValueError) as exception:
            self.get_logger().warning(f"Invalid timing JSON: {exception}")
            return
        pose_sec = finite_float(timing.get("pose_processing_sec"))
        map_sec = finite_float(timing.get("map_update_sec"))
        sensor_sec = finite_float(timing.get("sensor_stamp_sec"))
        with self.state.lock:
            self.state.data["timing"] = timing
            self.state.data["counters"]["timing_messages"] += 1
            if sensor_sec is not None and pose_sec is not None:
                self.state.append_history("pose_processing", [sensor_sec, pose_sec])
            if sensor_sec is not None and map_sec is not None:
                self.state.append_history("map_update", [sensor_sec, map_sec])
        self.results.append_timing(timing)

    def on_map(self, message: GridMap) -> None:
        resolution = float(message.info.resolution)
        known_cells = 0
        if "elevation" in message.layers:
            index = list(message.layers).index("elevation")
            values = np.asarray(message.data[index].data, dtype=np.float32)
            known_cells = int(np.isfinite(values).sum())
        area_m2 = known_cells * resolution * resolution
        extent_m2 = float(message.info.length_x) * float(message.info.length_y)
        map_stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        map_state = {
            "sensor_stamp_sec": map_stamp_ns * 1e-9,
            "resolution_m": resolution,
            "known_cells": known_cells,
            "observed_area_m2": area_m2,
            "observed_area_km2": area_m2 / 1e6,
            "extent_area_m2": extent_m2,
            "length_x_m": float(message.info.length_x),
            "length_y_m": float(message.info.length_y),
        }
        with self.state.lock:
            map_state["revision_messages"] = (
                self.state.data["map"].get("revision_messages", 0) + 1
            )
            self.state.data["map"] = map_state
            self.state.data["counters"]["map_messages"] += 1
        self.publish_map_stats()
        self.results.append_map(map_state)

    def map_storage_files(self) -> tuple[int, int]:
        output_dir = Path(self.args.output_dir).expanduser()
        if not output_dir.is_dir():
            return 0, 0
        published = 0
        all_map = 0
        for path in output_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(output_dir).as_posix()
            size = path.stat().st_size
            if relative in ("global_grid_map.npz", "global_grid_map.sqlite3"):
                published += size
                all_map += size
            elif relative.startswith("_internal/grid_map_snapshot/"):
                all_map += size
            elif relative.startswith("_internal/map_observation_archive/"):
                all_map += size
        return all_map, published

    def update_storage(self) -> None:
        try:
            all_map, published = self.map_storage_files()
        except OSError as exception:
            self.get_logger().warning(f"Could not scan map storage: {exception}")
            return
        with self.state.lock:
            area_km2 = float(self.state.data["map"].get("observed_area_km2", 0.0))
            self.state.data["storage"] = {
                "all_map_files_bytes": all_map,
                "published_map_files_bytes": published,
                "all_map_files_mb": all_map / 1e6,
                "published_map_files_mb": published / 1e6,
                "gb_per_km2": (
                    all_map / 1e9 / area_km2 if area_km2 > 0.0 else None
                ),
                "output_dir": str(Path(self.args.output_dir).expanduser()),
            }
        self.publish_map_stats()
        self.state.save()
        self.results.export(self.state)

    def publish_map_stats(self) -> None:
        with self.state.lock:
            payload = {
                "map": self.state.data["map"],
                "storage": self.state.data["storage"],
            }
        output = String()
        output.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.map_stats_pub.publish(output)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T3 实时定位与建图监控</title>
<style>
:root{--bg:#f5f7f8;--panel:#fff;--line:#d8dee2;--ink:#172126;--muted:#66747c;--teal:#087e8b;--red:#d64545;--green:#2e8b57;--amber:#b7791f;--blue:#3567a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"Noto Sans SC",sans-serif;font-size:14px;letter-spacing:0}
header{height:56px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 20px;position:sticky;top:0;z-index:5}
h1{font-size:18px;margin:0;font-weight:700}h2{font-size:14px;margin:0 0 12px;font-weight:700}main{max-width:1600px;margin:0 auto;padding:16px 20px 28px}
.status{display:flex;gap:12px;align-items:center;color:var(--muted);font-size:12px}.dot{width:9px;height:9px;border-radius:50%;background:var(--amber)}.dot.ok{background:var(--green)}
.metrics{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));border:1px solid var(--line);background:#fff;margin-bottom:14px}.metric{padding:12px 14px;border-right:1px solid var(--line);min-width:0}.metric:last-child{border-right:0}.metric label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}.metric b{font-size:20px;font-variant-numeric:tabular-nums;white-space:nowrap}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:14px;min-width:0}.wide{grid-column:1/-1}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.values{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}.values th,.values td{padding:7px 8px;border-bottom:1px solid #edf0f2;text-align:right}.values th:first-child,.values td:first-child{text-align:left;color:var(--muted)}
canvas{width:100%;height:250px;display:block;background:#fff}.chart-title{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:6px}.topic{font-family:Consolas,monospace;color:var(--blue);overflow-wrap:anywhere}
.stability{width:100%;border-collapse:collapse}.stability th,.stability td{padding:7px;border-bottom:1px solid #edf0f2;text-align:right;font-variant-numeric:tabular-nums}.stability th:first-child,.stability td:first-child{text-align:left}.pass{color:var(--green);font-weight:700}.fail{color:var(--red);font-weight:700}.empty{color:var(--muted);padding:20px 0;text-align:center}
@media(max-width:1000px){.metrics{grid-template-columns:repeat(3,1fr)}.metric:nth-child(3){border-right:0}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.pair{grid-template-columns:1fr}}
@media(max-width:620px){header{padding:0 12px}.status span:last-child{display:none}main{padding:10px}.metrics{grid-template-columns:1fr 1fr}.metric:nth-child(3){border-right:1px solid var(--line)}.metric:nth-child(even){border-right:0}.pair{grid-template-columns:1fr}.metric b{font-size:17px}}
</style>
</head>
<body>
<header><h1>T3 实时定位与建图监控</h1><div class="status"><i id="dot" class="dot"></i><span id="connection">等待 ROS 数据</span><span id="updated">--</span></div></header>
<main>
  <section class="metrics">
    <div class="metric"><label>当前定位误差</label><b id="positionError">-- m</b></div>
    <div class="metric"><label>最新稳定度</label><b id="stabilityValue">-- °/s</b></div>
    <div class="metric"><label>位姿处理时间</label><b id="poseTime">-- s</b></div>
    <div class="metric"><label>地图更新时间</label><b id="mapTime">-- s</b></div>
    <div class="metric"><label>有效地图范围</label><b id="mapArea">-- m²</b></div>
    <div class="metric"><label>单位面积存储</label><b id="storageRate">-- GB/km²</b></div>
  </section>
  <div class="grid">
    <section class="panel"><h2>位置真值与估计值</h2><table class="values"><thead><tr><th>坐标</th><th>真值 (m)</th><th>估计 (m)</th></tr></thead><tbody id="positionRows"></tbody></table><p>对齐方式：<span id="alignment">--</span></p></section>
    <section class="panel"><h2>姿态真值与估计值</h2><table class="values"><thead><tr><th>姿态</th><th>真值 (°)</th><th>估计 (°)</th></tr></thead><tbody id="attitudeRows"></tbody></table><p><span class="stabilityPeriod">500</span> s 稳定度话题：<span class="topic">/Car/T3/metrics/orientation_stability</span></p></section>
    <section class="panel wide"><div class="chart-title"><h2>实时位置误差曲线</h2><span>m</span></div><canvas id="errorChart"></canvas></section>
    <section class="panel"><div class="chart-title"><h2>位姿处理时间（增量柱状图）</h2><span id="poseStamp">时间戳 --</span></div><canvas id="poseChart"></canvas></section>
    <section class="panel"><div class="chart-title"><h2>地图更新时间（增量柱状图）</h2><span id="mapStamp">时间戳 --</span></div><canvas id="mapChart"></canvas></section>
    <section class="panel"><h2>全局地图</h2><table class="values"><tbody><tr><td>分辨率</td><td id="resolution">-- m</td></tr><tr><td>有效栅格数</td><td id="knownCells">--</td></tr><tr><td>有效观测面积</td><td id="observedArea">-- m²</td></tr><tr><td>矩阵外包范围</td><td id="extentArea">-- m²</td></tr><tr><td>全局地图话题</td><td class="topic">/Car/T3/mapping/global_grid_map</td></tr></tbody></table></section>
    <section class="panel"><h2>轻量级地图存储</h2><table class="values"><tbody><tr><td>全部地图数据文件</td><td id="allStorage">-- MB</td></tr><tr><td>发布地图文件</td><td id="publishedStorage">-- MB</td></tr><tr><td>单位面积存储</td><td id="storageDetail">-- GB/km²</td></tr><tr><td>输出目录</td><td id="outputDir" class="topic">--</td></tr></tbody></table></section>
    <section class="panel wide"><h2>每 <span class="stabilityPeriod">500</span> s 姿态稳定度</h2><div id="stabilityEmpty" class="empty">等待第一个完整周期</div><table id="stabilityTable" class="stability" hidden><thead><tr><th>周期</th><th>结束时间戳</th><th>Roll 差</th><th>Pitch 差</th><th>Yaw 差</th><th>稳定度</th><th>状态</th></tr></thead><tbody id="stabilityRows"></tbody></table></section>
  </div>
</main>
<script>
const $=id=>document.getElementById(id), fmt=(v,n=3)=>v==null?'--':Number(v).toFixed(n), vec=(a,i)=>Array.isArray(a)?fmt(a[i]):'--';
function setupCanvas(c){const d=window.devicePixelRatio||1,r=c.getBoundingClientRect();c.width=Math.max(300,Math.floor(r.width*d));c.height=Math.floor(250*d);const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,250]}
function lineChart(id,rows,color){const c=$(id),[x,w,h]=setupCanvas(c);x.clearRect(0,0,w,h);x.strokeStyle='#d8dee2';x.fillStyle='#66747c';x.font='11px Arial';const p={l:46,r:12,t:12,b:28};if(!rows.length){x.fillText('等待数据',w/2-22,h/2);return}const vals=rows.map(r=>Number(r[1])),minX=Number(rows[0][0]),maxX=Number(rows.at(-1)[0]),maxY=Math.max(...vals,0.001)*1.1;x.beginPath();for(let i=0;i<5;i++){let yy=p.t+(h-p.t-p.b)*i/4;x.moveTo(p.l,yy);x.lineTo(w-p.r,yy);x.fillText(fmt(maxY*(1-i/4),2),4,yy+4)}x.stroke();x.strokeStyle=color;x.lineWidth=1.8;x.beginPath();rows.forEach((r,i)=>{const xx=p.l+(w-p.l-p.r)*(Number(r[0])-minX)/Math.max(maxX-minX,1e-6),yy=h-p.b-(h-p.t-p.b)*Number(r[1])/maxY;i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke();x.fillStyle='#66747c';x.fillText('最近 '+rows.length+' 点',p.l,h-8)}
function barChart(id,rows,color){const c=$(id),[x,w,h]=setupCanvas(c);x.clearRect(0,0,w,h);const p={l:44,r:10,t:12,b:28},view=rows.slice(-80);if(!view.length){x.fillStyle='#66747c';x.fillText('等待数据',w/2-22,h/2);return}const maxY=Math.max(...view.map(r=>Number(r[1])),.001)*1.15,bw=(w-p.l-p.r)/view.length;x.strokeStyle='#d8dee2';x.fillStyle='#66747c';x.font='11px Arial';x.beginPath();for(let i=0;i<5;i++){let yy=p.t+(h-p.t-p.b)*i/4;x.moveTo(p.l,yy);x.lineTo(w-p.r,yy);x.fillText(fmt(maxY*(1-i/4),2),3,yy+4)}x.stroke();x.fillStyle=color;view.forEach((r,i)=>{const bh=(h-p.t-p.b)*Number(r[1])/maxY;x.fillRect(p.l+i*bw,h-p.b-bh,Math.max(1,bw-1),bh)});x.fillStyle='#66747c';x.fillText('最近 '+view.length+' 帧',p.l,h-8)}
function rows3(target,labels,a,b){$(target).innerHTML=labels.map((v,i)=>`<tr><td>${v}</td><td>${vec(a,i)}</td><td>${vec(b,i)}</td></tr>`).join('')}
async function refresh(){try{const d=await (await fetch('/api/state',{cache:'no-store'})).json(),p=d.pose||{},t=d.timing||{},m=d.map||{},s=d.storage||{},st=d.stability_results||[],cur=d.stability_current||null,all=cur?[...st,cur]:st,latest=cur||(st.length?st.at(-1):null);$('dot').className='dot ok';$('connection').textContent='ROS 客户端运行中';$('updated').textContent=new Date().toLocaleTimeString();$('positionError').textContent=fmt(p.position_error_m,3)+' m';$('poseTime').textContent=fmt(t.pose_processing_sec,3)+' s';$('mapTime').textContent=fmt(t.map_update_sec,3)+' s';$('mapArea').textContent=fmt(m.observed_area_m2,2)+' m²';$('storageRate').textContent=fmt(s.gb_per_km2,3)+' GB/km²';$('stabilityValue').textContent=latest?fmt(latest.stability_deg_per_sec,6)+' °/s':'-- °/s';rows3('positionRows',['X','Y','Z'],p.reference_position_m,p.estimate_position_m);rows3('attitudeRows',['Roll','Pitch','Yaw'],p.reference_rpy_deg,p.estimate_rpy_deg);$('alignment').textContent=p.alignment||'--';$('poseStamp').textContent='时间戳 '+fmt(t.sensor_stamp_sec,6);$('mapStamp').textContent='时间戳 '+fmt(t.sensor_stamp_sec,6);$('resolution').textContent=fmt(m.resolution_m,3)+' m';$('knownCells').textContent=(m.known_cells||0).toLocaleString();$('observedArea').textContent=fmt(m.observed_area_m2,2)+' m²';$('extentArea').textContent=fmt(m.extent_area_m2,2)+' m²';$('allStorage').textContent=fmt(s.all_map_files_mb,2)+' MB';$('publishedStorage').textContent=fmt(s.published_map_files_mb,2)+' MB';$('storageDetail').textContent=fmt(s.gb_per_km2,3)+' GB/km²';$('outputDir').textContent=s.output_dir||'--';lineChart('errorChart',d.history.position_error,'#d64545');barChart('poseChart',d.history.pose_processing,'#087e8b');barChart('mapChart',d.history.map_update,'#b7791f');$('stabilityEmpty').hidden=!!all.length;$('stabilityTable').hidden=!all.length;$('stabilityRows').innerHTML=all.slice().reverse().map(v=>`<tr><td>${v.is_partial_period?'当前实际周期':v.period_index}</td><td>${fmt(v.end_stamp_sec,3)}</td><td>${fmt(v.difference_rpy_deg[0],4)}°</td><td>${fmt(v.difference_rpy_deg[1],4)}°</td><td>${fmt(v.difference_rpy_deg[2],4)}°</td><td>${fmt(v.stability_deg_per_sec,6)}°/s<br><small>${fmt(v.normalization_sec,1)} s</small></td><td class="${v.status}">${v.status}</td></tr>`).join('')}catch(e){$('dot').className='dot';$('connection').textContent='客户端数据连接失败'}}
async function refreshPeriod(){try{const d=await (await fetch('/api/state',{cache:'no-store'})).json(),w=d.stability_window||{};document.querySelectorAll('.stabilityPeriod').forEach(e=>e.textContent=fmt(d.stability_period_sec,0));if(!(d.stability_results||[]).length&&!d.stability_current)$('stabilityEmpty').textContent=`等待有效周期（${fmt(w.elapsed_wall_sec,1)} / ${fmt(w.required_wall_sec,0)} s 实际运行时间）`}catch(e){}}
setInterval(refresh,500);setInterval(refreshPeriod,500);addEventListener('resize',refresh);refresh();refreshPeriod();
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/index.html"):
            payload = DASHBOARD_HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path.startswith("/api/state"):
            payload = self.state.snapshot()
            content_type = "application/json; charset=utf-8"
        elif self.path == "/healthz":
            payload = b'{"status":"ok"}'
            content_type = "application/json"
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--estimate-topic", default="/Car/T3/localization/odometry")
    parser.add_argument("--reference-topic", default="/car/pose")
    parser.add_argument("--timing-topic", default="/Car/T3/metrics/frame_timing")
    parser.add_argument("--global-map-topic", default="/Car/T3/mapping/global_grid_map")
    parser.add_argument("--stability-period-sec", type=float, default=500.0)
    parser.add_argument("--reference-max-offset-sec", type=float, default=0.2)
    parser.add_argument("--storage-period-sec", type=float, default=2.0)
    args, ros_args = parser.parse_known_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.stability_period_sec <= 0.0:
        parser.error("--stability-period-sec must be positive")
    return args, ros_args


def main() -> int:
    args, ros_args = parse_args()
    state = DashboardState(
        Path(args.output_dir).expanduser() / "dashboard_state_cache.json"
    )
    DashboardHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    rclpy.init(args=ros_args)
    node = T3RealtimeDashboard(args, state)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        state.save()
        node.results.export(state)
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
