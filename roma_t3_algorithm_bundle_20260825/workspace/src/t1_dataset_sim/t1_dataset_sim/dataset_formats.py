from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml


ORGANIZED_FORMAT = "batch_organized"
LEGACY_FORMAT = "legacy"


@dataclass(frozen=True)
class OrganizedStereoRecord:
    frame_index: str
    left_timestamp_ns: int
    right_timestamp_ns: int
    left_path: Path
    right_path: Path

    @property
    def timestamp_ns(self) -> int:
        """Canonical pose time: the left-camera exposure time."""
        return self.left_timestamp_ns

    @property
    def skew_ns(self) -> int:
        return self.right_timestamp_ns - self.left_timestamp_ns


@dataclass(frozen=True)
class StereoPairingStats:
    left_rows: int
    right_rows: int
    shared_rows: int
    eligible_rows: int
    paired_rows: int
    skipped_for_skew: int
    skipped_for_max_frames: int
    missing_left: int
    missing_right: int
    max_abs_skew_ns: int

    def selection_summary(
        self,
        *,
        pairing_method: str,
        configured_max_abs_skew_ns: int,
        max_frames: int,
    ) -> Dict[str, Any]:
        """Describe raw candidates, eligibility filters, and final selection."""
        configured_skew_ms = (
            None
            if configured_max_abs_skew_ns < 0
            else configured_max_abs_skew_ns / 1_000_000.0
        )
        return {
            "schema_version": 1,
            "pairing_method": str(pairing_method),
            "eligibility_definition": (
                "shared pair whose absolute left/right timestamp skew does "
                "not exceed configured_max_stereo_skew_ms"
                if configured_max_abs_skew_ns >= 0
                else "shared pair; stereo timestamp-skew rejection disabled"
            ),
            "raw_left_row_count": int(self.left_rows),
            "raw_right_row_count": int(self.right_rows),
            "shared_candidate_pair_count": int(self.shared_rows),
            "stereo_skew_eligible_pair_count": int(self.eligible_rows),
            "selected_frame_count": int(self.paired_rows),
            "excluded_by_stereo_skew_count": int(self.skipped_for_skew),
            "excluded_by_max_frames_count": int(
                self.skipped_for_max_frames
            ),
            "missing_left_count": int(self.missing_left),
            "missing_right_count": int(self.missing_right),
            "configured_max_stereo_skew_ms": configured_skew_ms,
            "observed_max_abs_stereo_skew_ms": (
                self.max_abs_skew_ns / 1_000_000.0
            ),
            "configured_max_frames": int(max_frames),
            "all_timestamp_rows_have_partner": (
                self.missing_left == 0 and self.missing_right == 0
            ),
            "all_shared_candidates_pass_stereo_skew": (
                self.eligible_rows == self.shared_rows
            ),
            "all_stereo_skew_eligible_pairs_selected": (
                self.skipped_for_max_frames == 0
            ),
        }


@dataclass(frozen=True)
class OrganizedImuRecord:
    timestamp_ns: int
    gyro_rad_s: np.ndarray


def detect_dataset_format(data_dir: Path, requested: str = "auto") -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", ORGANIZED_FORMAT, LEGACY_FORMAT}:
        raise ValueError(
            "dataset_format must be auto, batch_organized, or legacy; "
            f"got {requested!r}"
        )
    if requested != "auto":
        return requested

    data_dir = Path(data_dir)
    if (
        (data_dir / "images" / "timestamps.tsv").is_file()
        and (data_dir / "images" / "left").is_dir()
        and (data_dir / "images" / "right").is_dir()
    ):
        return ORGANIZED_FORMAT
    return LEGACY_FORMAT


def default_dataset_paths(data_dir: Path, dataset_format: str) -> Dict[str, Path]:
    data_dir = Path(data_dir)
    if dataset_format == ORGANIZED_FORMAT:
        return {
            "imu": data_dir / "imu" / "imu.csv",
            "calibration": data_dir,
            "semantics": data_dir / "semantic_labels",
        }
    return {
        "imu": data_dir / "Measure.xlsx",
        "calibration": data_dir / "calib.yaml",
        "semantics": data_dir / "semantic_labels",
    }


def _read_timestamp_rows(path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Stereo timestamp table does not exist: {path}")

    by_side: Dict[str, Dict[str, Dict[str, str]]] = {
        "left": {},
        "right": {},
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"side", "frame_index", "timestamp_ns", "filename"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Invalid stereo timestamp columns in {path}: {reader.fieldnames}"
            )
        for row in reader:
            side = str(row["side"]).strip().lower()
            if side not in by_side:
                continue
            frame_index = str(row["frame_index"]).strip()
            if frame_index in by_side[side]:
                raise ValueError(
                    f"Duplicate {side} frame_index {frame_index!r} in {path}"
                )
            by_side[side][frame_index] = row
    return by_side


def load_organized_stereo_records(
    data_dir: Path,
    *,
    max_abs_skew_ns: int = 20_000_000,
    max_frames: int = 0,
) -> Tuple[List[OrganizedStereoRecord], StereoPairingStats]:
    """Load frame-indexed stereo pairs without inventing image timestamps.

    Both images are later published at the left exposure time because the
    downstream SLAM estimates the left-camera pose.  The original right time
    is retained here so pairs outside the configured tolerance can be rejected.
    A negative max_abs_skew_ns disables skew rejection.
    """

    data_dir = Path(data_dir)
    rows = _read_timestamp_rows(data_dir / "images" / "timestamps.tsv")
    left_rows = rows["left"]
    right_rows = rows["right"]
    shared = sorted(
        set(left_rows).intersection(right_rows),
        key=lambda value: (
            (0, int(value)) if value.isdigit() else (1, value)
        ),
    )

    records: List[OrganizedStereoRecord] = []
    eligible_rows = 0
    skipped_for_skew = 0
    observed_max_skew = 0
    for frame_index in shared:
        left = left_rows[frame_index]
        right = right_rows[frame_index]
        left_timestamp_ns = int(left["timestamp_ns"])
        right_timestamp_ns = int(right["timestamp_ns"])
        abs_skew = abs(right_timestamp_ns - left_timestamp_ns)
        observed_max_skew = max(observed_max_skew, abs_skew)
        if max_abs_skew_ns >= 0 and abs_skew > max_abs_skew_ns:
            skipped_for_skew += 1
            continue

        eligible_rows += 1
        if max_frames > 0 and len(records) >= max_frames:
            continue

        left_path = data_dir / str(left["filename"])
        right_path = data_dir / str(right["filename"])
        if not left_path.is_file():
            raise FileNotFoundError(
                f"Left image listed in TSV is missing: {left_path}"
            )
        if not right_path.is_file():
            raise FileNotFoundError(
                f"Right image listed in TSV is missing: {right_path}"
            )
        records.append(
            OrganizedStereoRecord(
                frame_index=frame_index,
                left_timestamp_ns=left_timestamp_ns,
                right_timestamp_ns=right_timestamp_ns,
                left_path=left_path,
                right_path=right_path,
            )
        )

    stats = StereoPairingStats(
        left_rows=len(left_rows),
        right_rows=len(right_rows),
        shared_rows=len(shared),
        eligible_rows=eligible_rows,
        paired_rows=len(records),
        skipped_for_skew=skipped_for_skew,
        skipped_for_max_frames=eligible_rows - len(records),
        missing_left=len(set(right_rows).difference(left_rows)),
        missing_right=len(set(left_rows).difference(right_rows)),
        max_abs_skew_ns=observed_max_skew,
    )
    return records, stats


def load_organized_imu_records(path: Path) -> List[OrganizedImuRecord]:
    """Read the organized ROS IMU export; angular velocity is already rad/s."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IMU CSV file does not exist: {path}")

    records: List[OrganizedImuRecord] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid organized IMU columns: {reader.fieldnames}")
        time_column = (
            "header_time_ns"
            if "header_time_ns" in reader.fieldnames
            else "record_time_ns"
        )
        if time_column not in reader.fieldnames:
            raise ValueError(
                "Organized IMU CSV requires header_time_ns or record_time_ns"
            )
        for row in reader:
            gyro_rad_s = np.array(
                [
                    row["angular_velocity_x"],
                    row["angular_velocity_y"],
                    row["angular_velocity_z"],
                ],
                dtype=np.float64,
            )
            if not np.isfinite(gyro_rad_s).all():
                continue
            records.append(
                OrganizedImuRecord(
                    timestamp_ns=int(row[time_column]),
                    gyro_rad_s=gyro_rad_s,
                )
            )
    return records


def _resolution_to_size(value: Any) -> Tuple[int, int]:
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size != 2 or np.any(values <= 0):
        raise ValueError(f"Invalid camera resolution: {value!r}")
    # The organized calibration stores [height, width].
    return int(values[1]), int(values[0])


def _translation_metres(value: Any) -> np.ndarray:
    translation = np.asarray(value, dtype=np.float64).reshape(3)
    # This dataset stores camera translations in millimetres.  Retain support
    # for future files that already use metres.
    if np.linalg.norm(translation) > 10.0:
        translation = translation / 1000.0
    return translation


def _load_organized_calibration(sequence_dir: Path) -> Dict[str, Any]:
    calibration_dir = sequence_dir / "calibration"
    if sequence_dir.name == "calibration":
        calibration_dir = sequence_dir

    intrinsics_path = calibration_dir / "intrinsics.json"
    extrinsics_path = calibration_dir / "extrinsics.json"
    if not intrinsics_path.is_file() or not extrinsics_path.is_file():
        raise FileNotFoundError(
            "Organized calibration requires intrinsics.json and extrinsics.json "
            f"under {calibration_dir}"
        )

    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    extrinsics = json.loads(extrinsics_path.read_text(encoding="utf-8"))

    left = intrinsics["Cam_L"]
    right = intrinsics["Cam_R"]
    width, height = _resolution_to_size(left["resolution"])
    right_size = _resolution_to_size(right["resolution"])
    if right_size != (width, height):
        raise ValueError(
            f"Stereo calibration resolutions differ: {(width, height)} vs {right_size}"
        )

    left_k = np.asarray(left["K"], dtype=np.float64).reshape(3, 3)
    right_k = np.asarray(right["K"], dtype=np.float64).reshape(3, 3)

    # The organized images are already undistorted.  Only the stereo rotation
    # must be applied.  Cam_R_to_Cam_L maps right coordinates into left, while
    # OpenCV stereoRectify expects the left-to-right transform.
    right_to_left = extrinsics["Cam_R_to_Cam_L"]
    rotation_right_to_left = np.asarray(
        right_to_left["R"], dtype=np.float64
    ).reshape(3, 3)
    translation_right_to_left = _translation_metres(right_to_left["T"])
    rotation_left_to_right = rotation_right_to_left.T
    translation_left_to_right = (
        -rotation_right_to_left.T @ translation_right_to_left
    )
    baseline_m = float(np.linalg.norm(translation_left_to_right))

    zeros = np.zeros(5, dtype=np.float64)
    left_r, right_r, left_p, _right_p, _q, _roi_l, _roi_r = cv2.stereoRectify(
        left_k,
        zeros,
        right_k,
        zeros,
        (width, height),
        rotation_left_to_right,
        translation_left_to_right,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    rectified_k = np.asarray(left_p[:, :3], dtype=np.float64)

    return {
        "camera": {
            "fx": float(rectified_k[0, 0]),
            "fy": float(rectified_k[1, 1]),
            "cx": float(rectified_k[0, 2]),
            "cy": float(rectified_k[1, 2]),
            "width": width,
            "height": height,
        },
        "baseline": baseline_m,
        "depth": {"min": 0.1, "max": 100.0},
        "stereo": {
            # Residual calibration error in this capture is around 2--4 px
            # after geometric rectification; 5 px keeps the constraint useful.
            "epipolar_threshold_px": 5.0,
        },
        "tracking": {
            # RoMa/PnP on these high-resolution lunar frames commonly yields
            # 30--50 verified inliers.  Treating every result below 50 as a
            # keyframe otherwise grows the map by thousands of points/frame.
            "keyframe_min_inliers": 25,
            "keyframe_min_gap": 2,
            "keyframe_interval_from_last": True,
        },
        "rectification": {
            "enabled": True,
            "source_undistorted": True,
            "width": width,
            "height": height,
            "left_k": left_k.tolist(),
            "right_k": right_k.tolist(),
            "left_d": zeros.tolist(),
            "right_d": zeros.tolist(),
            "left_r": left_r.tolist(),
            "right_r": right_r.tolist(),
            "rectified_k": rectified_k.tolist(),
        },
        "source": {
            "format": ORGANIZED_FORMAT,
            "intrinsics_path": str(intrinsics_path),
            "extrinsics_path": str(extrinsics_path),
        },
    }


def _complete_raw_stereo_calibration(calibration: Dict[str, Any]) -> Dict[str, Any]:
    """Build the runtime rectifier from a measured raw-stereo description."""

    raw = calibration["raw_stereo"]
    width = int(raw["width"])
    height = int(raw["height"])
    if width <= 0 or height <= 0:
        raise ValueError("raw_stereo width/height must be positive")

    left_k = np.asarray(raw["left"]["k"], dtype=np.float64).reshape(3, 3)
    right_k = np.asarray(raw["right"]["k"], dtype=np.float64).reshape(3, 3)
    left_d = np.asarray(raw["left"]["d"], dtype=np.float64).reshape(-1)
    right_d = np.asarray(raw["right"]["d"], dtype=np.float64).reshape(-1)
    right_to_left = np.asarray(
        raw["transform_right_to_left"], dtype=np.float64
    ).reshape(4, 4)
    if not np.isfinite(right_to_left).all() or not np.allclose(
        right_to_left[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9
    ):
        raise ValueError("raw_stereo transform_right_to_left is not homogeneous")

    rotation_right_to_left = right_to_left[:3, :3]
    translation_right_to_left = right_to_left[:3, 3]
    rotation_left_to_right = rotation_right_to_left.T
    translation_left_to_right = (
        -rotation_left_to_right @ translation_right_to_left
    )
    baseline_m = float(np.linalg.norm(translation_left_to_right))
    if not np.isfinite(baseline_m) or baseline_m <= 0.0:
        raise ValueError("raw_stereo baseline must be finite and positive")

    left_r, right_r, left_p, _right_p, _q, _roi_l, _roi_r = cv2.stereoRectify(
        left_k,
        left_d,
        right_k,
        right_d,
        (width, height),
        rotation_left_to_right,
        translation_left_to_right,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    rectified_k = np.asarray(left_p[:, :3], dtype=np.float64)

    calibration["camera"] = {
        "fx": float(rectified_k[0, 0]),
        "fy": float(rectified_k[1, 1]),
        "cx": float(rectified_k[0, 2]),
        "cy": float(rectified_k[1, 2]),
        "width": width,
        "height": height,
    }
    calibration["baseline"] = baseline_m
    calibration["rectification"] = {
        "enabled": True,
        "source_undistorted": False,
        "width": width,
        "height": height,
        "left_k": left_k.tolist(),
        "right_k": right_k.tolist(),
        "left_d": left_d.tolist(),
        "right_d": right_d.tolist(),
        "left_r": left_r.tolist(),
        "right_r": right_r.tolist(),
        "rectified_k": rectified_k.tolist(),
    }

    mounting = calibration.get("mounting", {})
    if "transform_vehicle_left" in mounting:
        vehicle_to_left = np.asarray(
            mounting["transform_vehicle_left"], dtype=np.float64
        ).reshape(4, 4)
        # Calibration vehicle axes are X forward, Y right, Z down. ROS map
        # axes are X forward, Y left, Z up. The SLAM camera is rectified, so
        # undo the left rectification rotation before applying the mounting.
        vehicle_frd_to_map_flu = np.diag([1.0, -1.0, -1.0])
        rotation_camera_to_map = (
            vehicle_frd_to_map_flu
            @ vehicle_to_left[:3, :3]
            @ left_r.T
        )
        translation_camera_to_map = (
            vehicle_frd_to_map_flu @ vehicle_to_left[:3, 3]
        )
        calibration["map_alignment"] = {
            "rotation_camera_to_map": rotation_camera_to_map.tolist(),
            "translation_camera_to_map": (
                translation_camera_to_map.tolist()
            ),
            "origin": "initial_vehicle_body_origin",
            "axes": "x_forward_y_left_z_up",
        }

    return calibration


def load_calibration(source: Path) -> Dict[str, Any]:
    """Load either legacy calib.yaml or an organized sequence/calibration dir."""

    source = Path(source).expanduser()
    if source.is_dir():
        organized_intrinsics = source / "calibration" / "intrinsics.json"
        if source.name == "calibration":
            organized_intrinsics = source / "intrinsics.json"
        if organized_intrinsics.is_file():
            return _load_organized_calibration(source)
        legacy_path = source / "calib.yaml"
        if legacy_path.is_file():
            source = legacy_path
        else:
            raise FileNotFoundError(
                "Calibration directory contains neither organized JSON nor "
                f"calib.yaml: {source}"
            )
    if not source.is_file():
        raise FileNotFoundError(f"Calibration source does not exist: {source}")
    if source.suffix.lower() == ".json":
        return _load_organized_calibration(source.parent)
    with source.open("r", encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    if not isinstance(calibration, dict):
        raise ValueError(f"Calibration YAML must contain a mapping: {source}")
    if "raw_stereo" in calibration:
        calibration = _complete_raw_stereo_calibration(calibration)
    if "camera" not in calibration or "baseline" not in calibration:
        raise ValueError(
            f"Calibration YAML is missing camera/baseline entries: {source}"
        )
    return calibration


def _scale_camera_matrix(
    matrix: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    scaled = np.asarray(matrix, dtype=np.float64).copy()
    scaled[0, :] *= float(scale_x)
    scaled[1, :] *= float(scale_y)
    scaled[2, :] = [0.0, 0.0, 1.0]
    return scaled


class StereoRectifier:
    """Cache OpenCV maps for already-undistorted organized stereo images."""

    def __init__(self, calibration: Dict[str, Any], enabled: bool = True) -> None:
        self.calibration = calibration
        self.enabled = bool(enabled and calibration.get("rectification", {}).get(
            "enabled", False
        ))
        self._size: Tuple[int, int] | None = None
        self._left_maps = None
        self._right_maps = None
        self._output_k = None

    def _prepare(self, width: int, height: int) -> None:
        size = (int(width), int(height))
        if self._size == size:
            return

        rectification = self.calibration.get("rectification", {})
        source_width = int(rectification.get("width", width))
        source_height = int(rectification.get("height", height))
        scale_x = width / float(source_width)
        scale_y = height / float(source_height)
        target_k = _scale_camera_matrix(
            np.asarray(rectification["rectified_k"], dtype=np.float64),
            scale_x,
            scale_y,
        )
        left_k = _scale_camera_matrix(
            np.asarray(rectification["left_k"], dtype=np.float64),
            scale_x,
            scale_y,
        )
        right_k = _scale_camera_matrix(
            np.asarray(rectification["right_k"], dtype=np.float64),
            scale_x,
            scale_y,
        )
        self._left_maps = cv2.initUndistortRectifyMap(
            left_k,
            np.asarray(rectification["left_d"], dtype=np.float64),
            np.asarray(rectification["left_r"], dtype=np.float64),
            target_k,
            size,
            cv2.CV_32FC1,
        )
        self._right_maps = cv2.initUndistortRectifyMap(
            right_k,
            np.asarray(rectification["right_d"], dtype=np.float64),
            np.asarray(rectification["right_r"], dtype=np.float64),
            target_k,
            size,
            cv2.CV_32FC1,
        )
        self._output_k = target_k
        self._size = size

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return left, right
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(
                f"Stereo image shapes differ: {left.shape} vs {right.shape}"
            )
        height, width = left.shape[:2]
        self._prepare(width, height)
        return (
            self.rectify_left(left),
            cv2.remap(right, *self._right_maps, interpolation=cv2.INTER_LINEAR),
        )

    def rectify_left(
        self,
        image: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        if not self.enabled:
            return image
        height, width = image.shape[:2]
        self._prepare(width, height)
        return cv2.remap(
            image,
            *self._left_maps,
            interpolation=interpolation,
        )

    def camera_parameters(self, width: int, height: int) -> Dict[str, float]:
        if self.enabled:
            self._prepare(width, height)
            matrix = self._output_k
            return {
                "fx": float(matrix[0, 0]),
                "fy": float(matrix[1, 1]),
                "cx": float(matrix[0, 2]),
                "cy": float(matrix[1, 2]),
            }

        camera = self.calibration["camera"]
        source_width = float(camera.get("width", width))
        source_height = float(camera.get("height", height))
        return {
            "fx": float(camera["fx"]) * width / source_width,
            "fy": float(camera["fy"]) * height / source_height,
            "cx": float(camera["cx"]) * width / source_width,
            "cy": float(camera["cy"]) * height / source_height,
        }
