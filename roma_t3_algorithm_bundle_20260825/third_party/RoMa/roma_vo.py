#!/usr/bin/env python3
"""
RoMa-based Stereo Visual SLAM — 增强版 v2
==========================================
在原始 RoMa SLAM 基础上集成：
  1. 滑动窗口 Bundle Adjustment（scipy，多轮 outlier 剔除）
  2. ORB 外观候选 + 双向 PnP 几何验证 + scipy SE(3) 位姿图
  3. IMU 陀螺仪预积分辅助 PnP（降低旋转估计误差）
  4. 改进双目三角化（极线约束 + 视差范围检查）
  5. 地图点关联验证（重投影误差二次过滤）
  6. 全局地图点管理（分离优化剔除与点云保留）
  7. 最终全局 BA 接口

依赖：
    pip install romatch kornia scipy tqdm pyyaml matplotlib

使用：
    python roma_slam_v2.py \
        --data_dir data/data1223 \
        --calib data/data1223/calib.yaml \
        --enable_backend --enable_loop_closing --final_ba \
        --imu_data data/Measure.xlsx --imu_timestamps data/timestamps.txt
"""

import os
import time
import queue
import threading
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2
import yaml
import torch
from PIL import Image
from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot

# RoMa
from romatch import roma_outdoor

torch.set_float32_matmul_precision("highest")


# ──────────────────────────────────────────────────────────────
# g2o 可选导入
# ──────────────────────────────────────────────────────────────
g2o = None
G2O_POSE_GRAPH_OK = False
try:
    import g2o as _g2o
    _opt = _g2o.SparseOptimizer()
    _g2o.VertexSE3Expmap(); _g2o.EdgeSE3Expmap()
    g2o = _g2o
    G2O_POSE_GRAPH_OK = True
    print("[Backend] 检测到 g2o；回环仍使用已验证的 scipy SE(3) 后端")
except Exception:
    print("[Backend] g2o 未安装；回环使用 scipy SE(3) 后端")


def is_g2o_available() -> bool:
    return g2o is not None


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class MapPoint:
    """全局地图点"""
    id: int
    position: np.ndarray
    color: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))
    observations: Dict[int, np.ndarray] = field(default_factory=dict)
    descriptor: Optional[np.ndarray] = None
    n_visible: int = 0
    n_found: int = 0
    is_bad: bool = False
    ba_outlier_count: int = 0

    def add_observation(self, kf_id: int, keypoint: np.ndarray):
        self.observations[kf_id] = keypoint
        self.n_visible += 1
        self.n_found += 1

    def remove_observation(self, kf_id: int):
        self.observations.pop(kf_id, None)


@dataclass
class KeyFrame:
    """关键帧"""
    id: int
    timestamp: float
    pose: np.ndarray
    image: Optional[np.ndarray] = None
    keypoints: List[np.ndarray] = field(default_factory=list)
    # 与 keypoints/mappoints 按索引对齐。矫正双目下 BA 使用
    # (u_left, v_left, u_right) 观测持续约束公制尺度。
    right_keypoints: List[Optional[np.ndarray]] = field(default_factory=list)
    # 关键帧创建时由双目直接测得的左相机坐标，与 keypoints 严格对齐。
    # 回环几何验证使用它，避免依赖可能已漂移的全局地图坐标。
    camera_points: List[Optional[np.ndarray]] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    mappoints: List[Optional[MapPoint]] = field(default_factory=list)
    covisible_kfs: Dict[int, int] = field(default_factory=dict)
    loop_edges: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    K: Optional[np.ndarray] = None
    is_bad: bool = False
    image_size_hint: Optional[Tuple[int, int]] = None

    def add_mappoint(self, idx: int, mp: "MapPoint"):
        while len(self.mappoints) <= idx:
            self.mappoints.append(None)
        self.mappoints[idx] = mp


@dataclass(frozen=True)
class TrackingStatus:
    """Outcome of the latest ``StereoSLAMv2.process_frame`` call.

    ``process_frame`` keeps returning a 4x4 pose for backwards compatibility.
    On failure that pose is only the last known pose, so callers must inspect
    this status before using the frame as a new mapping observation.
    """

    success: bool = False
    frame_index: int = 0
    reason: str = "not_processed"
    pose_updated: bool = False
    initialized: bool = False
    inlier_count: int = 0
    correspondence_count: int = 0
    pose_index: int = -1
    keyframe_created: bool = False
    keyframe_id: Optional[int] = None
    geometry_revision: int = 0
    geometry_corrected: bool = False


@dataclass(frozen=True)
class MappingKeyFrameSnapshot:
    """Immutable keyframe data exposed to external mapping consumers.

    ``camera_points`` stay in the keyframe's original left-camera frame.  They
    are therefore stable raw observations that can be replayed with a later
    BA/pose-graph-corrected keyframe pose without retaining live SLAM objects.
    """

    id: int
    timestamp: float
    pose: np.ndarray
    keypoints: np.ndarray
    camera_points: np.ndarray
    is_bad: bool = False


@dataclass(frozen=True)
class LoopKeyFrameSnapshot:
    """Immutable data owned by the loop-closing worker.

    The worker never receives a live ``KeyFrame`` or the RoMa model. Camera
    points are already expressed in this keyframe's left-camera coordinates,
    so global pose-graph corrections do not invalidate geometric verification.
    """

    id: int
    pose: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    camera_points: np.ndarray
    image_size: Tuple[int, int]
    previous_id: Optional[int] = None
    T_previous_current: Optional[np.ndarray] = None
    # Indices in the source KeyFrame arrays before spatial subsampling.  Loop
    # closing only needs geometry, while relocalization uses these indices to
    # reconnect verified inliers to existing MapPoints. Kept after the
    # pre-existing optional fields to preserve positional construction.
    source_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32))


@dataclass(frozen=True)
class LoopConstraint:
    """Independently measured relative transform ``T_loop_current``."""

    current_id: int
    loop_id: int
    T_loop_current: np.ndarray
    descriptor_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_rmse: float
    appearance_score: float
    # Verified feature indices in the immutable loop/current snapshots.
    loop_feature_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32))
    current_feature_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32))


@dataclass(frozen=True)
class PoseGraphEdge:
    from_id: int
    to_id: int
    measurement: np.ndarray
    weight: float = 1.0
    kind: str = "odometry"


@dataclass(frozen=True)
class LoopClosureResult:
    """Worker-to-main-thread message containing incremental world corrections."""

    constraint: LoopConstraint
    corrections: Dict[int, np.ndarray]
    optimized_poses: Dict[int, np.ndarray]


def make_keyframe_geometry_snapshot(
        kf: KeyFrame,
        K: np.ndarray,
        previous_id: Optional[int] = None,
        T_previous_current: Optional[np.ndarray] = None
        ) -> LoopKeyFrameSnapshot:
    """Build a detached, spatially bounded keyframe geometry snapshot.

    This builder is shared by loop closing and front-end relocalization.  It
    never retains a live ``KeyFrame`` or image and never accesses the RoMa GPU
    model, so relocalization remains available when the optional backend is
    disabled.
    """
    intrinsic = np.asarray(K, dtype=np.float64)
    keypoint_count = len(kf.keypoints)
    keypoints = np.full((keypoint_count, 2), np.nan, dtype=np.float64)
    for index, point in enumerate(kf.keypoints):
        value = np.asarray(point, dtype=np.float64).reshape(-1)
        if value.size >= 2:
            keypoints[index] = value[:2]

    descriptors = np.zeros((keypoint_count, 32), dtype=np.uint8)
    if kf.descriptors is not None:
        source = np.asarray(kf.descriptors, dtype=np.uint8)
        if source.ndim == 2 and source.shape[1] == 32:
            count = min(keypoint_count, len(source))
            descriptors[:count] = source[:count]

    camera_points = np.full((keypoint_count, 3), np.nan, dtype=np.float64)
    pose = np.asarray(kf.pose, dtype=np.float64)
    R_cw = pose[:3, :3].T
    t_wc = pose[:3, 3]
    for index in range(keypoint_count):
        if index < len(kf.camera_points):
            raw_point = kf.camera_points[index]
            if raw_point is not None:
                value = np.asarray(raw_point, dtype=np.float64).reshape(-1)
                if (value.size >= 3 and np.all(np.isfinite(value[:3]))
                        and value[2] > 0.1):
                    camera_points[index] = value[:3]
                    continue
        if index >= len(kf.mappoints):
            continue
        mp = kf.mappoints[index]
        if mp is None or mp.is_bad:
            continue
        point_camera = R_cw @ (mp.position - t_wc)
        if np.all(np.isfinite(point_camera)) and point_camera[2] > 0.1:
            camera_points[index] = point_camera

    if kf.image is not None and np.asarray(kf.image).ndim >= 2:
        image_size = tuple(int(value) for value in kf.image.shape[:2])
    elif getattr(kf, "image_size_hint", None) is not None:
        image_size = tuple(kf.image_size_hint)
    else:
        image_size = (
            max(1, int(round(2.0 * float(intrinsic[1, 2])))),
            max(1, int(round(2.0 * float(intrinsic[0, 2])))),
        )

    # Dense RoMa keyframes may contain ~5000 points. Brute-force ORB place
    # retrieval on all of them is quadratic, so keep the established spatially
    # balanced 1200-feature subset. This is the same selection previously used
    # only by the loop-closing backend.
    valid = np.flatnonzero(
        np.all(np.isfinite(keypoints), axis=1)
        & np.any(descriptors != 0, axis=1)
        & np.all(np.isfinite(camera_points), axis=1)
        & (camera_points[:, 2] > 0.1))
    max_geometry_features = 1200
    if len(valid) > max_geometry_features:
        height, width = image_size
        grid_rows, grid_cols = 6, 8
        per_cell = max(
            1, max_geometry_features // (grid_rows * grid_cols))
        selected = []
        selected_set = set()
        x_cells = np.clip(
            (keypoints[valid, 0] * grid_cols / max(width, 1)).astype(int),
            0, grid_cols - 1)
        y_cells = np.clip(
            (keypoints[valid, 1] * grid_rows / max(height, 1)).astype(int),
            0, grid_rows - 1)
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell = valid[(x_cells == col) & (y_cells == row)]
                if len(cell) == 0:
                    continue
                if len(cell) > per_cell:
                    positions = np.linspace(
                        0, len(cell) - 1, per_cell, dtype=np.int32)
                    cell = cell[positions]
                for index in cell:
                    selected.append(int(index))
                    selected_set.add(int(index))
        if len(selected) < max_geometry_features:
            for index in valid:
                if int(index) not in selected_set:
                    selected.append(int(index))
                    if len(selected) >= max_geometry_features:
                        break
        valid = np.asarray(
            selected[:max_geometry_features], dtype=np.int32)
    else:
        valid = valid.astype(np.int32, copy=False)

    keypoints = keypoints[valid].copy()
    descriptors = descriptors[valid].copy()
    camera_points = camera_points[valid].copy()
    source_indices = valid.copy()

    frozen_pose = pose.copy()
    frozen_odometry = None
    if T_previous_current is not None:
        frozen_odometry = np.asarray(
            T_previous_current, dtype=np.float64).copy()
    for array in (
            frozen_pose, keypoints, descriptors, camera_points,
            source_indices, frozen_odometry):
        if array is None:
            continue
        array.setflags(write=False)
    return LoopKeyFrameSnapshot(
        id=int(kf.id),
        pose=frozen_pose,
        keypoints=keypoints,
        descriptors=descriptors,
        camera_points=camera_points,
        image_size=image_size,
        source_indices=source_indices,
        previous_id=previous_id,
        T_previous_current=frozen_odometry)


# ══════════════════════════════════════════════════════════════
#  IMU 预积分辅助
# ══════════════════════════════════════════════════════════════

class IMUHelper:
    """
    IMU 陀螺仪预积分 — 为帧间 PnP 提供旋转初始值

    坐标系：
        IMU/车体:  X前, Y右, Z上
        相机:      X右, Y下, Z前 (OpenCV)
        安装俯角:  camera_pitch_deg 度（正 = 向下俯拍）
    """

    def __init__(self, imu_data_path: str, timestamp_path: str,
                 camera_pitch_deg: float = 30.0,
                 gravity: float = 1.626):
        import pandas as pd
        df = pd.read_excel(imu_data_path)
        self.imu_times = df['%星时(s)'].values
        self.gyro = np.column_stack([
            df['wbix(deg/s)'].values,
            df['wbiy(deg/s)'].values,
            df['wbiz(deg/s)'].values
        ])
        self.imu_dt = 0.2  # 5 Hz

        df_ts = pd.read_csv(timestamp_path, sep='\t', encoding='utf-8')
        self.image_times = df_ts.iloc[:, 0].values
        self.R_cam_imu = self._build_R_cam_imu(camera_pitch_deg)
        print(f"[IMU] {len(self.imu_times)} 条数据, 相机俯角={camera_pitch_deg}°")

    def _build_R_cam_imu(self, pitch_deg: float) -> np.ndarray:
        R_align = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
        th = np.deg2rad(pitch_deg)
        R_pitch = np.array([[1, 0, 0],
                             [0, np.cos(th), -np.sin(th)],
                             [0, np.sin(th),  np.cos(th)]], dtype=np.float64)
        return R_pitch @ R_align

    @staticmethod
    def _exp_so3(omega: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(omega)
        if theta < 1e-10:
            return np.eye(3)
        ax = omega / theta
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    def get_timestamp(self, frame_idx: int) -> float:
        if 1 <= frame_idx <= len(self.image_times):
            return float(self.image_times[frame_idx - 1])
        return -1.0

    def get_delta_R_cam(self, frame_from: int, frame_to: int) -> Optional[np.ndarray]:
        t0 = self.get_timestamp(frame_from)
        t1 = self.get_timestamp(frame_to)
        if t0 < 0 or t1 < 0 or t1 <= t0:
            return None
        mask = (self.imu_times > t0) & (self.imu_times <= t1)
        if not np.any(mask):
            return None
        dR_body = np.eye(3)
        for gyro in self.gyro[mask]:
            dR_body = dR_body @ self._exp_so3(np.deg2rad(gyro) * self.imu_dt)
        return self.R_cam_imu @ dR_body @ self.R_cam_imu.T


# ══════════════════════════════════════════════════════════════
#  RoMa 匹配器封装（支持 numpy array 输入）
# ══════════════════════════════════════════════════════════════

class RoMaMatcher:
    """
    对 RoMa 模型的轻量封装
    - 支持文件路径 或 numpy BGR 图像输入
    - 返回 (kpts1, kpts2, confidence) numpy arrays
    """

    def __init__(self, device: str = "cuda",
                 coarse_res: int = 560,
                 upsample_res: int = 560,
                 use_custom_corr: bool = False,
                 enable_upsample_refinement: bool = True,
                 symmetric_matching: bool = True):
        self.device = device
        print("[RoMa] 加载模型...")
        self.model = roma_outdoor(
            device=device,
            use_custom_corr=use_custom_corr,
            coarse_res=coarse_res,
            upsample_res=upsample_res
        )
        # RoMa defaults to an additional high-resolution refinement pass and
        # bidirectional matching.  Both remain enabled by default for backward
        # compatibility, but may be disabled for real-time VO.
        self.model.upsample_preds = bool(enable_upsample_refinement)
        self.model.symmetric = bool(symmetric_matching)
        print("[RoMa] 模型加载完成")

    @staticmethod
    def _to_pil(src) -> Tuple[Image.Image, int, int]:
        """将路径或 numpy BGR 转为 PIL RGB 并返回 (pil, W, H)"""
        if isinstance(src, (str, Path)):
            img = Image.open(src).convert("RGB")
        else:  # numpy BGR
            img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        W, H = img.size
        return img, W, H

    def match(self, src1, src2,
              num_matches: int = 2000,
              resize: Optional[Tuple[int, int]] = None,
              upsample_refinement: Optional[bool] = None,
              symmetric_matching: Optional[bool] = None,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        匹配两张图像。

        Args:
            src1, src2: 文件路径 或 numpy BGR 图像
            num_matches: 采样点数
            resize: (W, H) 若提供则先缩放（用于加速）

        Returns:
            kpts1, kpts2: (N,2) float32 像素坐标
            confidence:   (N,)  float32
        """
        pil1, W1, H1 = self._to_pil(src1)
        pil2, W2, H2 = self._to_pil(src2)

        if resize is not None:
            pil1 = pil1.resize(resize, Image.BILINEAR)
            pil2 = pil2.resize(resize, Image.BILINEAR)
            rW, rH = resize
            sx1, sy1 = W1 / rW, H1 / rH
            sx2, sy2 = W2 / rW, H2 / rH
        else:
            rW, rH = W1, H1
            sx1 = sy1 = sx2 = sy2 = 1.0

        warp, certainty = self.model.match(
            pil1,
            pil2,
            device=self.device,
            upsample_preds=upsample_refinement,
            symmetric=symmetric_matching,
        )
        matches, cert = self.model.sample(warp, certainty, num=num_matches)
        kp1, kp2 = self.model.to_pixel_coordinates(matches, rH, rW, rH, rW)

        kp1 = kp1.cpu().numpy().astype(np.float32)
        kp2 = kp2.cpu().numpy().astype(np.float32)
        cert = cert.cpu().numpy().astype(np.float32)

        # 还原到原始分辨率
        kp1[:, 0] *= sx1;  kp1[:, 1] *= sy1
        kp2[:, 0] *= sx2;  kp2[:, 1] *= sy2

        return kp1, kp2, cert

    def supports_feature_reuse(
            self,
            stereo_refinement: bool,
            temporal_refinement: bool,
            symmetric_matching: bool) -> bool:
        if symmetric_matching or temporal_refinement:
            return False
        if not stereo_refinement:
            return True
        return tuple(self.model.upsample_res) == (
            self.model.h_resized,
            self.model.w_resized,
        )

    def _prepare_source(
            self,
            src,
            resize: Optional[Tuple[int, int]] = None) -> dict:
        image, width, height = self._to_pil(src)
        if resize is not None:
            image = image.resize(resize, Image.BILINEAR)
            match_width, match_height = resize
        else:
            match_width, match_height = width, height
        return {
            "prepared": self.model.prepare_image(
                image, device=self.device
            ),
            "features": None,
            "source_width": int(width),
            "source_height": int(height),
            "match_width": int(match_width),
            "match_height": int(match_height),
            "scale_x": float(width / match_width),
            "scale_y": float(height / match_height),
            "resize": None if resize is None else tuple(resize),
        }

    def _cache_is_compatible(
            self,
            cache: Optional[dict],
            src,
            resize: Optional[Tuple[int, int]]) -> bool:
        if cache is None or cache.get("features") is None:
            return False
        if isinstance(src, (str, Path)):
            return False
        height, width = np.asarray(src).shape[:2]
        return (
            cache.get("source_width") == int(width)
            and cache.get("source_height") == int(height)
            and cache.get("resize") == (
                None if resize is None else tuple(resize)
            )
            and tuple(cache["prepared"].shape[-2:]) == (
                self.model.h_resized,
                self.model.w_resized,
            )
        )

    def freeze_feature_cache(self, cache: Optional[dict]) -> Optional[dict]:
        if cache is None:
            return None
        frozen = {
            key: value
            for key, value in cache.items()
            if key not in {"prepared", "features"}
        }
        frozen["prepared"] = cache["prepared"].detach().clone()
        frozen["features"] = self.model.clone_feature_pyramid(
            cache["features"]
        )
        return frozen

    def _sample_prepared_match(
            self,
            output,
            source_a: dict,
            source_b: dict,
            num_matches: int
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        warp, certainty = output
        matches, cert = self.model.sample(
            warp, certainty, num=num_matches
        )
        kp_a, kp_b = self.model.to_pixel_coordinates(
            matches,
            source_a["match_height"],
            source_a["match_width"],
            source_b["match_height"],
            source_b["match_width"],
        )
        kp_a = kp_a.cpu().numpy().astype(np.float32)
        kp_b = kp_b.cpu().numpy().astype(np.float32)
        cert = cert.cpu().numpy().astype(np.float32)
        kp_a[:, 0] *= source_a["scale_x"]
        kp_a[:, 1] *= source_a["scale_y"]
        kp_b[:, 0] *= source_b["scale_x"]
        kp_b[:, 1] *= source_b["scale_y"]
        return kp_a, kp_b, cert

    def match_frame_pairs(
            self,
            left,
            right,
            keyframe=None,
            keyframe_cache: Optional[dict] = None,
            *,
            stereo_matches: int = 5000,
            temporal_matches: int = 2000,
            resize: Optional[Tuple[int, int]] = None,
            stereo_refinement: bool = True,
            temporal_refinement: bool = False,
            symmetric_matching: bool = False,
            batch_coarse_pairs: bool = False):
        if not self.supports_feature_reuse(
                stereo_refinement,
                temporal_refinement,
                symmetric_matching):
            raise ValueError(
                "Current RoMa settings do not support feature reuse"
            )

        left_source = self._prepare_source(left, resize)
        right_source = self._prepare_source(right, resize)
        sources_to_encode = [left_source, right_source]

        keyframe_source = None
        if keyframe is not None:
            if self._cache_is_compatible(keyframe_cache, keyframe, resize):
                keyframe_source = keyframe_cache
            else:
                keyframe_source = self._prepare_source(keyframe, resize)
                sources_to_encode.append(keyframe_source)

        encoded = self.model.encode_prepared_images(
            [source["prepared"] for source in sources_to_encode]
        )
        for source, features in zip(sources_to_encode, encoded):
            source["features"] = features

        all_sources = [left_source, right_source]
        pairs = [(0, 1)]
        if keyframe_source is not None:
            all_sources.append(keyframe_source)
            pairs.append((2, 0))
        outputs = self.model.match_prepared_pairs(
            [source["prepared"] for source in all_sources],
            [source["features"] for source in all_sources],
            pairs,
            refine_pair_indices=(0,) if stereo_refinement else (),
            batch_pairs=batch_coarse_pairs,
        )

        stereo_result = self._sample_prepared_match(
            outputs[0], left_source, right_source, stereo_matches
        )
        temporal_result = None
        if keyframe_source is not None:
            temporal_result = self._sample_prepared_match(
                outputs[1],
                keyframe_source,
                left_source,
                temporal_matches,
            )
        return stereo_result, temporal_result, left_source

    def match_cached_pair(
            self,
            source_a,
            source_b,
            cache_a: Optional[dict] = None,
            cache_b: Optional[dict] = None,
            *,
            num_matches: int = 2000,
            resize: Optional[Tuple[int, int]] = None,
            upsample_refinement: bool = False
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match one pair while reusing any compatible prepared features.

        This is used by the KLT fallback path.  The current left image has
        already been encoded for stereo, so a RoMa temporal fallback only
        needs the decoder unless a cache is unavailable or incompatible.
        """
        if not self.supports_feature_reuse(
                upsample_refinement, False, False):
            raise ValueError(
                "Current RoMa settings do not support cached pair matching"
            )

        prepared_sources = []
        sources_to_encode = []
        for source, cache in (
                (source_a, cache_a), (source_b, cache_b)):
            if self._cache_is_compatible(cache, source, resize):
                prepared = cache
            else:
                prepared = self._prepare_source(source, resize)
                sources_to_encode.append(prepared)
            prepared_sources.append(prepared)

        if sources_to_encode:
            encoded = self.model.encode_prepared_images(
                [source["prepared"] for source in sources_to_encode]
            )
            for source, features in zip(sources_to_encode, encoded):
                source["features"] = features

        outputs = self.model.match_prepared_pairs(
            [source["prepared"] for source in prepared_sources],
            [source["features"] for source in prepared_sources],
            [(0, 1)],
            refine_pair_indices=(0,) if upsample_refinement else (),
        )
        return self._sample_prepared_match(
            outputs[0], prepared_sources[0], prepared_sources[1], num_matches
        )


# ══════════════════════════════════════════════════════════════
#  ORB 描述子提取（用于地图点关联二次验证）
# ══════════════════════════════════════════════════════════════

class ORBDescriptorExtractor:
    def __init__(self, n_features: int = 2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract_at_points(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        n = len(points)
        descs = np.zeros((n, 32), dtype=np.uint8)
        kps, valid_idx = [], []
        for i, pt in enumerate(points):
            x, y = int(pt[0]), int(pt[1])
            if 16 <= x < w - 16 and 16 <= y < h - 16:
                kps.append(cv2.KeyPoint(float(pt[0]), float(pt[1]), 31))
                valid_idx.append(i)
        if not kps:
            return descs
        _, des = self.orb.compute(gray, kps)
        if des is not None:
            for j, idx in enumerate(valid_idx):
                if j < len(des):
                    descs[idx] = des[j]
        return descs


# ══════════════════════════════════════════════════════════════
#  改进双目三角化
# ══════════════════════════════════════════════════════════════

class StereoTriangulator:
    """
    改进点：
      · 极线约束：左右 y 坐标差 < epipolar_threshold
      · 视差范围：由 min/max_depth 推导
    """

    def __init__(self, K: np.ndarray, baseline: float,
                 min_depth: float = 0.5, max_depth: float = 200.0,
                 epipolar_threshold: float = 2.0):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.epipolar_threshold = epipolar_threshold

    def triangulate(self, pts_l: np.ndarray,
                    pts_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            pts_3d:      (M, 3) 有效 3D 点（相机坐标系）
            valid_idx:   (M,)   在原始数组中的索引
        """
        y_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
        disparity = pts_l[:, 0] - pts_r[:, 0]
        min_d = self.fx * self.baseline / self.max_depth
        max_d = self.fx * self.baseline / self.min_depth
        valid = (y_diff < self.epipolar_threshold) & (disparity > min_d) & (disparity < max_d)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return np.zeros((0, 3)), idx
        d = disparity[idx]
        Z = self.fx * self.baseline / d
        X = (pts_l[idx, 0] - self.cx) * Z / self.fx
        Y = (pts_l[idx, 1] - self.cy) * Z / self.fy
        return np.stack([X, Y, Z], axis=1), idx


# ══════════════════════════════════════════════════════════════
#  地图点关联器（重投影验证）
# ══════════════════════════════════════════════════════════════

class MapPointAssociator:
    def __init__(self, K: np.ndarray,
                 pixel_threshold: float = 5.0,
                 reproj_threshold: float = 3.0):
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.pixel_threshold = pixel_threshold
        self.reproj_threshold = reproj_threshold

    def _reproj_error(self, pt3d: np.ndarray, pt2d: np.ndarray,
                      pose: np.ndarray) -> float:
        R, t = pose[:3, :3], pose[:3, 3]
        pc = R.T @ (pt3d - t)
        if pc[2] <= 0.1:
            return float('inf')
        u = self.fx * pc[0] / pc[2] + self.cx
        v = self.fy * pc[1] / pc[2] + self.cy
        return float(np.sqrt((u - pt2d[0])**2 + (v - pt2d[1])**2))

    def associate(self,
                  curr_kpts: np.ndarray,
                  curr_pose: np.ndarray,
                  matched_kf_kpts: np.ndarray,
                  matched_kf_indices: np.ndarray,
                  kf_mp_ids: List[int],
                  global_mappoints: Dict[int, MapPoint]
                  ) -> Dict[int, int]:
        """Returns {curr_kpt_idx -> mp_id}"""
        if len(curr_kpts) == 0 or len(matched_kf_kpts) == 0:
            return {}
        tree = cKDTree(curr_kpts)
        assoc = {}
        used_mp_ids = set()
        for mkpt, kf_idx in zip(matched_kf_kpts, matched_kf_indices):
            if kf_idx >= len(kf_mp_ids):
                continue
            mp_id = kf_mp_ids[kf_idx]
            if (mp_id < 0 or mp_id not in global_mappoints
                    or mp_id in used_mp_ids):
                continue
            mp = global_mappoints[mp_id]
            if mp.is_bad:
                continue
            dist, ci = tree.query(mkpt, k=1)
            if dist > self.pixel_threshold:
                continue
            if self._reproj_error(mp.position, curr_kpts[ci], curr_pose) > self.reproj_threshold:
                continue
            if ci not in assoc:
                assoc[ci] = mp_id
                used_mp_ids.add(mp_id)
        return assoc


# ══════════════════════════════════════════════════════════════
#  滑动窗口 Bundle Adjustment
# ══════════════════════════════════════════════════════════════

class SlidingWindowBA:
    """带公制双目约束的局部滑窗 BA。

    约定：``KeyFrame.pose`` 是左相机的 ``T_wc``，双目已经矫正，右相机
    相对左相机沿 +X 平移 ``baseline``。活动窗口的第一帧是固定边界帧，
    不进入优化变量；退出活动窗口只影响局部计算集合，不破坏全局关键帧、
    地图点观测或回环历史。
    """

    def __init__(self, window_size: int = 10, K: np.ndarray = None,
                 baseline: float = 0.0):
        self.window_size = max(2, int(window_size))
        self.K = K
        self.baseline = float(baseline)
        self.keyframes: List[KeyFrame] = []
        self.mappoints: Dict[int, MapPoint] = {}

        # 像素噪声按 sigma=1 px 处理。单目为 2 自由度，双目为 3 自由度。
        self.chi2_init_mono = 13.816   # chi2(2, 0.999)
        self.chi2_final_mono = 5.991   # chi2(2, 0.95)
        self.chi2_init_stereo = 16.266 # chi2(3, 0.999)
        self.chi2_final_stereo = 7.815 # chi2(3, 0.95)
        self.outlier_count_threshold = 5

        # 调用方可据此判断是否真的提交了优化结果。
        self.last_accepted = False
        self.last_mode = "none"
        self.last_failure_reason = "not_run"

    @staticmethod
    def _right_keypoint(kf: KeyFrame, index: int) -> Optional[np.ndarray]:
        if index >= len(kf.right_keypoints):
            return None
        value = kf.right_keypoints[index]
        if value is None:
            return None
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            return None
        return point[:2]

    def _rebuild_active_mappoints(self):
        active = {}
        for kf in self.keyframes:
            if kf.is_bad:
                continue
            for mp in kf.mappoints:
                if mp is not None and not mp.is_bad:
                    active[mp.id] = mp
        self.mappoints = active

    def add_keyframe(self, kf: KeyFrame):
        self.keyframes.append(kf)
        if len(self.keyframes) > self.window_size:
            self._marginalize()
        else:
            self._rebuild_active_mappoints()

    def _marginalize(self):
        """移出局部活动窗口，但保留完整 SLAM 历史。

        这里采用固定边界帧的局部 BA，而不是带 Schur 先验的固定滞后
        平滑器，所以准确说是窗口淘汰。绝不能因此把关键帧标为 bad 或删除
        地图点的全局观测，否则回环会失去历史数据。
        """
        if self.keyframes:
            self.keyframes.pop(0)
        self._rebuild_active_mappoints()

    @staticmethod
    def _snapshot_singletons(kfs: List[KeyFrame],
                             optimized_mp_ids) -> Dict[int, Tuple[MapPoint, int, np.ndarray]]:
        """保存未参加 BA 的单帧点在所属相机中的坐标。

        当关键帧位姿被修正时，这些点应随关键帧刚体移动，否则会留在旧位姿
        生成的位置。多帧点不能这样移动，它们必须由共同观测约束。
        """
        snapshots = {}
        optimized_mp_ids = set(optimized_mp_ids)
        for pi, kf in enumerate(kfs):
            R_old = kf.pose[:3, :3]
            t_old = kf.pose[:3, 3]
            for mp in kf.mappoints:
                if (mp is None or mp.is_bad or mp.id in optimized_mp_ids
                        or mp.id in snapshots or len(mp.observations) != 1):
                    continue
                pc = R_old.T @ (mp.position - t_old)
                if np.all(np.isfinite(pc)):
                    snapshots[mp.id] = (mp, pi, pc)
        return snapshots

    @staticmethod
    def _restore_singletons(kfs: List[KeyFrame], snapshots):
        for mp, pi, pc in snapshots.values():
            pose = kfs[pi].pose
            mp.position = pose[:3, :3] @ pc + pose[:3, 3]

    def _update_outlier_history(self, mp_list: List[MapPoint],
                                obs_mi: np.ndarray,
                                inlier: np.ndarray):
        """每次 BA 对一个地图点最多累计一次失败，避免按轮次重复处罚。"""
        for mi, mp in enumerate(mp_list):
            mask = obs_mi == mi
            total = int(mask.sum())
            rejected = int((~inlier[mask]).sum())
            majority_limit = max(2, (total + 1) // 2)
            if rejected >= majority_limit:
                mp.ba_outlier_count += 1
            else:
                mp.ba_outlier_count = max(0, mp.ba_outlier_count - 1)
            if mp.ba_outlier_count >= self.outlier_count_threshold:
                mp.is_bad = True
                mp.ba_outlier_count = 0

    def optimize(self, iterations: int = 10) -> Tuple[float, int]:
        """优化局部位姿和地图点；只有通过完整校验后才原子式写回。"""
        from scipy.sparse import lil_matrix

        self.last_accepted = False
        self.last_mode = "stereo_joint"
        self.last_failure_reason = "not_enough_data"

        kfs = [kf for kf in self.keyframes if not kf.is_bad]
        n_poses = len(kfs)
        if n_poses < 2 or self.K is None:
            return 0.0, 0

        # 收集活动窗口观测。只有多帧共视点进入优化：窗口内单观测点
        # 对位姿是零净约束（双目 3 残差恰好对 3 个点未知数），全局单帧点
        # 由 _snapshot_singletons 随所属关键帧刚体搬移，效果等同且不占
        # 变量；窗口外仍有历史观测的点保持不动，避免被单观测重新锚定。
        mp_observations: Dict[int, List] = {}
        for pi, kf in enumerate(kfs):
            seen_mp_ids = set()
            for ki, mp in enumerate(kf.mappoints):
                if (mp is None or mp.is_bad or mp.id in seen_mp_ids
                        or ki >= len(kf.keypoints)):
                    continue
                left = np.asarray(kf.keypoints[ki], dtype=np.float64).reshape(-1)
                if left.size < 2 or not np.all(np.isfinite(left[:2])):
                    continue
                right = self._right_keypoint(kf, ki)
                mp_observations.setdefault(mp.id, []).append(
                    (pi, ki, mp, left[:2], right))
                seen_mp_ids.add(mp.id)

        shared = [(mid, obs) for mid, obs in mp_observations.items()
                  if len(obs) >= 2]
        if len(shared) < 10 or self.baseline <= 0.0:
            return self._pose_only(kfs, iterations)

        candidates = sorted(
            shared, key=lambda item: len(item[1]), reverse=True)[:2500]

        mp_list = [obs[0][2] for _, obs in candidates]
        mp_id2i = {mp.id: i for i, mp in enumerate(mp_list)}
        n_pts = len(mp_list)

        obs_pi, obs_mi, obs_left, obs_right_u, obs_stereo = [], [], [], [], []
        for mid, observations in candidates:
            mi = mp_id2i[mid]
            for pi, _ki, _mp, left, right in observations:
                obs_pi.append(pi)
                obs_mi.append(mi)
                obs_left.append(left)
                is_stereo = right is not None
                obs_stereo.append(is_stereo)
                obs_right_u.append(float(right[0]) if is_stereo else np.nan)

        obs_pi = np.asarray(obs_pi, dtype=np.int32)
        obs_mi = np.asarray(obs_mi, dtype=np.int32)
        obs_left = np.asarray(obs_left, dtype=np.float64)
        obs_right_u = np.asarray(obs_right_u, dtype=np.float64)
        obs_stereo = np.asarray(obs_stereo, dtype=bool)
        n_obs = len(obs_pi)
        if n_obs < 20 or int(obs_stereo.sum()) < 20:
            return self._pose_only(kfs, iterations)

        # 第一帧不进入变量，明确消除全局 SE(3) gauge。
        n_free_poses = n_poses - 1
        pose_dim = 6 * n_free_poses
        x0 = np.zeros(pose_dim + 3 * n_pts, dtype=np.float64)
        for free_i, kf in enumerate(kfs[1:]):
            rv, _ = cv2.Rodrigues(kf.pose[:3, :3])
            x0[6*free_i:6*free_i+3] = rv.reshape(3)
            x0[6*free_i+3:6*free_i+6] = kf.pose[:3, 3]
        for mi, mp in enumerate(mp_list):
            x0[pose_dim+3*mi:pose_dim+3*mi+3] = mp.position

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        anchor_R = kfs[0].pose[:3, :3].copy()
        anchor_t = kfs[0].pose[:3, 3].copy()

        def decode(x):
            Rs = np.empty((n_poses, 3, 3), dtype=np.float64)
            ts = np.empty((n_poses, 3), dtype=np.float64)
            Rs[0], ts[0] = anchor_R, anchor_t
            for free_i in range(n_free_poses):
                base = 6 * free_i
                Rs[free_i + 1] = cv2.Rodrigues(x[base:base+3])[0]
                ts[free_i + 1] = x[base+3:base+6]
            pts = x[pose_dim:].reshape(n_pts, 3)
            return Rs, ts, pts

        def residual_matrix(x, indices):
            indices = np.asarray(indices, dtype=np.int32)
            Rs, ts, pts = decode(x)
            pi = obs_pi[indices]
            mi = obs_mi[indices]
            delta = pts[mi] - ts[pi]
            pc = np.einsum(
                'nij,nj->ni', np.transpose(Rs[pi], (0, 2, 1)), delta)
            z = pc[:, 2]
            valid_depth = np.isfinite(z) & (z > 0.1)
            safe_z = np.where(valid_depth, z, 1.0)
            errors = np.zeros((len(indices), 3), dtype=np.float64)
            errors[:, 0] = fx * pc[:, 0] / safe_z + cx - obs_left[indices, 0]
            errors[:, 1] = fy * pc[:, 1] / safe_z + cy - obs_left[indices, 1]
            stereo = obs_stereo[indices]
            errors[stereo, 2] = (
                fx * (pc[stereo, 0] - self.baseline) / safe_z[stereo]
                + cx - obs_right_u[indices[stereo]])
            errors[~valid_depth, 0:2] = 50.0
            errors[(~valid_depth) & stereo, 2] = 50.0
            errors[~np.isfinite(errors)] = 50.0
            return errors, valid_depth

        def observation_errors(x):
            errors, valid_depth = residual_matrix(x, np.arange(n_obs))
            chi2 = np.sum(errors * errors, axis=1)
            return chi2, valid_depth

        def build_sparsity(indices):
            indices = np.asarray(indices, dtype=np.int32)
            matrix = lil_matrix(
                (3 * len(indices), pose_dim + 3 * n_pts), dtype=np.int8)
            for row_i, oi in enumerate(indices):
                rows = slice(3 * row_i, 3 * row_i + (3 if obs_stereo[oi] else 2))
                pi = int(obs_pi[oi])
                if pi > 0:
                    col = 6 * (pi - 1)
                    matrix[rows, col:col+6] = 1
                point_col = pose_dim + 3 * int(obs_mi[oi])
                matrix[rows, point_col:point_col+3] = 1
            return matrix.tocsr()

        def solve(start, active, huber_chi2):
            indices = np.flatnonzero(active)
            return least_squares(
                lambda x: residual_matrix(x, indices)[0].reshape(-1),
                start,
                method='trf',
                jac_sparsity=build_sparsity(indices),
                loss='huber',
                f_scale=max(1.0, float(np.sqrt(huber_chi2))),
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
                max_nfev=max(50, int(iterations) * 20),
                verbose=0,
                x_scale='jac')

        active = np.ones(n_obs, dtype=bool)
        xc = x0.copy()
        rounds = [
            (self.chi2_init_mono * 2.0, self.chi2_init_stereo * 2.0),
            (self.chi2_init_mono, self.chi2_init_stereo),
            (self.chi2_final_mono, self.chi2_final_stereo),
        ]
        try:
            for round_i, (mono_limit, stereo_limit) in enumerate(rounds, start=1):
                result = solve(xc, active, max(mono_limit, stereo_limit))
                if not np.all(np.isfinite(result.x)):
                    raise ValueError("optimizer returned non-finite variables")
                xc = result.x
                chi2, valid_depth = observation_errors(xc)
                limits = np.where(obs_stereo, stereo_limit, mono_limit)
                next_active = active & valid_depth & np.isfinite(chi2) & (chi2 <= limits)
                newly_rejected = int((active & ~next_active).sum())
                active = next_active
                print(f"[BA R{round_i}] outlier+{newly_rejected} inlier={active.sum()}")
                if int(active.sum()) < 20:
                    self.last_failure_reason = "too_few_inliers"
                    return self._pose_only(kfs, iterations)

            # 用最终内点再求解一次，避免最后一轮剔除后直接写回。
            result = solve(xc, active, self.chi2_final_stereo)
            if not np.all(np.isfinite(result.x)):
                raise ValueError("final optimizer returned non-finite variables")
            xc = result.x
        except Exception as exc:
            self.last_failure_reason = f"optimizer_failed: {exc}"
            print(f"[BA] 放弃写回: {exc}")
            return 0.0, 0

        initial_chi2, _ = observation_errors(x0)
        final_chi2, final_depth = observation_errors(xc)
        final_limits = np.where(
            obs_stereo, self.chi2_final_stereo, self.chi2_final_mono)
        active &= final_depth & np.isfinite(final_chi2) & (final_chi2 <= final_limits)
        n_inliers = int(active.sum())
        if n_inliers < 20:
            self.last_failure_reason = "too_few_final_inliers"
            return 0.0, 0

        initial_error = float(np.mean(initial_chi2[active]))
        final_error = float(np.mean(final_chi2[active]))
        if (not np.isfinite(initial_error) or not np.isfinite(final_error)
                or final_error > initial_error * 1.01 + 1e-9):
            self.last_failure_reason = "cost_not_improved"
            print(f"[BA] 放弃写回: cost {initial_error:.6f} -> {final_error:.6f}")
            return 0.0, 0

        # 至此才提交：优化失败或校验失败不会污染任何外部状态。
        old_anchor = kfs[0].pose.copy()
        singleton_snapshots = self._snapshot_singletons(
            kfs, optimized_mp_ids=mp_id2i.keys())
        Rs, ts, points = decode(xc)
        for pi in range(1, n_poses):
            kfs[pi].pose[:3, :3] = Rs[pi]
            kfs[pi].pose[:3, 3] = ts[pi]
        for mi, mp in enumerate(mp_list):
            mp.position = points[mi].copy()
        self._restore_singletons(kfs, singleton_snapshots)
        self._update_outlier_history(mp_list, obs_mi, active)

        # 固定帧必须按位完全不变；这是运行期防回归断言。
        if not np.array_equal(kfs[0].pose, old_anchor):
            raise RuntimeError("BA anchor pose was modified")

        self.last_accepted = True
        self.last_failure_reason = ""
        return final_error, n_inliers

    def _pose_only(self, kfs: List[KeyFrame],
                   iterations: int) -> Tuple[float, int]:
        """地图点固定的安全降级路径；同样不优化窗口首帧。"""
        from scipy.sparse import lil_matrix

        self.last_accepted = False
        self.last_mode = "pose_only"
        self.last_failure_reason = "not_enough_shared_observations"
        n_poses = len(kfs)
        if n_poses < 2 or self.K is None:
            return 0.0, 0

        # 只用活动窗口内被至少两帧看到的点。单帧新点由当前位姿创建，若拿
        # 它反过来约束同一位姿，会形成虚假的自约束。
        counts = defaultdict(int)
        for kf in kfs:
            seen = set()
            for ki, mp in enumerate(kf.mappoints):
                if (mp is not None and not mp.is_bad and ki < len(kf.keypoints)
                        and mp.id not in seen):
                    counts[mp.id] += 1
                    seen.add(mp.id)

        obs_pi, obs_points, obs_left, obs_right_u, obs_stereo, obs_mps = [], [], [], [], [], []
        for pi, kf in enumerate(kfs[1:], start=1):
            seen_mp_ids = set()
            for ki, mp in enumerate(kf.mappoints):
                if (mp is None or mp.is_bad or counts[mp.id] < 2
                        or mp.id in seen_mp_ids or ki >= len(kf.keypoints)):
                    continue
                left = np.asarray(kf.keypoints[ki], dtype=np.float64).reshape(-1)
                if left.size < 2 or not np.all(np.isfinite(left[:2])):
                    continue
                right = self._right_keypoint(kf, ki)
                is_stereo = right is not None and self.baseline > 0.0
                obs_pi.append(pi)
                obs_points.append(mp.position.copy())
                obs_left.append(left[:2])
                obs_right_u.append(float(right[0]) if is_stereo else np.nan)
                obs_stereo.append(is_stereo)
                obs_mps.append(mp)
                seen_mp_ids.add(mp.id)
                if len(obs_pi) >= 3000:
                    break

        n_obs = len(obs_pi)
        if n_obs < 12:
            return 0.0, 0

        obs_pi = np.asarray(obs_pi, dtype=np.int32)
        obs_points = np.asarray(obs_points, dtype=np.float64)
        obs_left = np.asarray(obs_left, dtype=np.float64)
        obs_right_u = np.asarray(obs_right_u, dtype=np.float64)
        obs_stereo = np.asarray(obs_stereo, dtype=bool)
        unique_mps = {mp.id: mp for mp in obs_mps}
        mp_list = list(unique_mps.values())
        mp_id2i = {mp.id: i for i, mp in enumerate(mp_list)}
        obs_mi = np.asarray([mp_id2i[mp.id] for mp in obs_mps], dtype=np.int32)

        n_free_poses = n_poses - 1
        x0 = np.zeros(6 * n_free_poses, dtype=np.float64)
        for free_i, kf in enumerate(kfs[1:]):
            rv, _ = cv2.Rodrigues(kf.pose[:3, :3])
            x0[6*free_i:6*free_i+3] = rv.reshape(3)
            x0[6*free_i+3:6*free_i+6] = kf.pose[:3, 3]

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        def decode_poses(x):
            Rs = [kfs[0].pose[:3, :3]]
            ts = [kfs[0].pose[:3, 3]]
            for free_i in range(n_free_poses):
                base = 6 * free_i
                Rs.append(cv2.Rodrigues(x[base:base+3])[0])
                ts.append(x[base+3:base+6])
            return np.asarray(Rs), np.asarray(ts)

        def residual_matrix(x, indices):
            indices = np.asarray(indices, dtype=np.int32)
            Rs, ts = decode_poses(x)
            pi = obs_pi[indices]
            delta = obs_points[indices] - ts[pi]
            pc = np.einsum(
                'nij,nj->ni', np.transpose(Rs[pi], (0, 2, 1)), delta)
            z = pc[:, 2]
            valid_depth = np.isfinite(z) & (z > 0.1)
            safe_z = np.where(valid_depth, z, 1.0)
            errors = np.zeros((len(indices), 3), dtype=np.float64)
            errors[:, 0] = fx * pc[:, 0] / safe_z + cx - obs_left[indices, 0]
            errors[:, 1] = fy * pc[:, 1] / safe_z + cy - obs_left[indices, 1]
            stereo = obs_stereo[indices]
            errors[stereo, 2] = (
                fx * (pc[stereo, 0] - self.baseline) / safe_z[stereo]
                + cx - obs_right_u[indices[stereo]])
            errors[~valid_depth, 0:2] = 50.0
            errors[(~valid_depth) & stereo, 2] = 50.0
            errors[~np.isfinite(errors)] = 50.0
            return errors, valid_depth

        def observation_errors(x):
            errors, valid_depth = residual_matrix(x, np.arange(n_obs))
            return np.sum(errors * errors, axis=1), valid_depth

        def build_sparsity(indices):
            indices = np.asarray(indices, dtype=np.int32)
            matrix = lil_matrix((3 * len(indices), len(x0)), dtype=np.int8)
            for row_i, oi in enumerate(indices):
                rows = slice(3 * row_i, 3 * row_i + (3 if obs_stereo[oi] else 2))
                col = 6 * (int(obs_pi[oi]) - 1)
                matrix[rows, col:col+6] = 1
            return matrix.tocsr()

        def solve(start, active, huber_chi2):
            indices = np.flatnonzero(active)
            return least_squares(
                lambda x: residual_matrix(x, indices)[0].reshape(-1),
                start,
                method='trf',
                jac_sparsity=build_sparsity(indices),
                loss='huber',
                f_scale=max(1.0, float(np.sqrt(huber_chi2))),
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
                max_nfev=max(50, int(iterations) * 20),
                verbose=0,
                x_scale='jac')

        active = np.ones(n_obs, dtype=bool)
        xc = x0.copy()
        try:
            for mono_limit, stereo_limit in [
                    (self.chi2_init_mono, self.chi2_init_stereo),
                    (self.chi2_final_mono, self.chi2_final_stereo)]:
                result = solve(xc, active, max(mono_limit, stereo_limit))
                if not np.all(np.isfinite(result.x)):
                    raise ValueError("optimizer returned non-finite variables")
                xc = result.x
                chi2, valid_depth = observation_errors(xc)
                limits = np.where(obs_stereo, stereo_limit, mono_limit)
                active &= valid_depth & np.isfinite(chi2) & (chi2 <= limits)
                if int(active.sum()) < 12:
                    self.last_failure_reason = "too_few_pose_inliers"
                    return 0.0, 0
            result = solve(xc, active, self.chi2_final_stereo)
            if not np.all(np.isfinite(result.x)):
                raise ValueError("final optimizer returned non-finite variables")
            xc = result.x
        except Exception as exc:
            self.last_failure_reason = f"optimizer_failed: {exc}"
            print(f"[BA pose-only] 放弃写回: {exc}")
            return 0.0, 0

        initial_chi2, _ = observation_errors(x0)
        final_chi2, final_depth = observation_errors(xc)
        limits = np.where(obs_stereo, self.chi2_final_stereo, self.chi2_final_mono)
        active &= final_depth & np.isfinite(final_chi2) & (final_chi2 <= limits)
        n_inliers = int(active.sum())
        if n_inliers < 12:
            self.last_failure_reason = "too_few_final_pose_inliers"
            return 0.0, 0
        initial_error = float(np.mean(initial_chi2[active]))
        final_error = float(np.mean(final_chi2[active]))
        if (not np.isfinite(initial_error) or not np.isfinite(final_error)
                or final_error > initial_error * 1.01 + 1e-9):
            self.last_failure_reason = "pose_cost_not_improved"
            return 0.0, 0

        old_anchor = kfs[0].pose.copy()
        singleton_snapshots = self._snapshot_singletons(kfs, optimized_mp_ids=())
        Rs, ts = decode_poses(xc)
        for pi in range(1, n_poses):
            kfs[pi].pose[:3, :3] = Rs[pi]
            kfs[pi].pose[:3, 3] = ts[pi]
        self._restore_singletons(kfs, singleton_snapshots)
        self._update_outlier_history(mp_list, obs_mi, active)
        if not np.array_equal(kfs[0].pose, old_anchor):
            raise RuntimeError("pose-only BA anchor pose was modified")

        self.last_accepted = True
        self.last_failure_reason = ""
        return final_error, n_inliers


# ══════════════════════════════════════════════════════════════
#  回环检测与位姿图优化
# ══════════════════════════════════════════════════════════════

class LoopClosing:
    def __init__(self, task_queue: queue.Queue, K: np.ndarray,
                 matcher: "RoMaMatcher" = None,
                 min_dist_thresh: float = 3.0,
                 min_interval: int = 15,
                 max_candidates: int = 5,
                 ratio_test: float = 0.75,
                 min_descriptor_matches: int = 30,
                 min_pnp_inliers: int = 20,
                 min_inlier_ratio: float = 0.35,
                 pnp_reproj_threshold: float = 3.0,
                 max_reprojection_rmse: float = 2.5,
                 min_image_coverage: float = 0.005,
                 max_rotation_consistency_deg: float = 5.0,
                 max_translation_consistency: float = 0.30):
        self.q = task_queue
        self.K = np.asarray(K, dtype=np.float64).copy()
        # ``matcher`` is intentionally ignored. Sharing the RoMa GPU model with
        # the tracking thread caused data races and unpredictable CUDA failures.
        self.max_loop_translation = max(0.1, float(min_dist_thresh))
        self.min_interval = max(1, int(min_interval))
        self.max_candidates = max(1, int(max_candidates))
        self.ratio_test = float(ratio_test)
        self.min_descriptor_matches = max(8, int(min_descriptor_matches))
        self.min_pnp_inliers = max(6, int(min_pnp_inliers))
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.pnp_reproj_threshold = float(pnp_reproj_threshold)
        self.max_reprojection_rmse = float(max_reprojection_rmse)
        self.min_image_coverage = float(min_image_coverage)
        self.max_rotation_consistency = np.deg2rad(
            float(max_rotation_consistency_deg))
        self.max_translation_consistency = float(max_translation_consistency)

        self.kfs: Dict[int, LoopKeyFrameSnapshot] = {}
        self.pose_estimates: Dict[int, np.ndarray] = {}
        self.odometry_edges: List[PoseGraphEdge] = []
        self.loop_edges: List[PoseGraphEdge] = []
        self.running = False
        self.thread = None
        self.on_loop_result = None
        self.last_error = ""

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._run, name="roma-loop-closing", daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        # FIFO sentinel: all snapshots already queued are processed first.
        self.q.put(None)
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                print("[LoopClosing] 警告: 后台线程未在超时内退出")
        self.running = False

    def _run(self):
        while True:
            try:
                snapshot = self.q.get()
                if snapshot is None:
                    break
                self._process(snapshot)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[LoopClosing] 后台异常，已跳过当前关键帧: {self.last_error}")
        self.running = False

    @staticmethod
    def _valid_feature_indices(kf: LoopKeyFrameSnapshot) -> np.ndarray:
        n = min(len(kf.keypoints), len(kf.descriptors), len(kf.camera_points))
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        keypoints = kf.keypoints[:n]
        descriptors = kf.descriptors[:n]
        camera_points = kf.camera_points[:n]
        valid = (
            np.all(np.isfinite(keypoints), axis=1)
            & np.any(descriptors != 0, axis=1)
            & np.all(np.isfinite(camera_points), axis=1)
            & (camera_points[:, 2] > 0.1)
        )
        return np.flatnonzero(valid).astype(np.int32)

    def _mutual_ratio_matches(
            self,
            first: LoopKeyFrameSnapshot,
            second: LoopKeyFrameSnapshot) -> List[Tuple[int, int, float]]:
        first_indices = self._valid_feature_indices(first)
        second_indices = self._valid_feature_indices(second)
        if len(first_indices) < 2 or len(second_indices) < 2:
            return []

        first_desc = np.ascontiguousarray(
            first.descriptors[first_indices], dtype=np.uint8)
        second_desc = np.ascontiguousarray(
            second.descriptors[second_indices], dtype=np.uint8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        def ratio_matches(query, train):
            accepted = []
            for pair in matcher.knnMatch(query, train, k=2):
                if len(pair) != 2:
                    continue
                best, runner_up = pair
                if (runner_up.distance > 0.0
                        and best.distance < self.ratio_test * runner_up.distance):
                    accepted.append(best)
            return accepted

        forward = ratio_matches(first_desc, second_desc)
        reverse = ratio_matches(second_desc, first_desc)
        reverse_pairs = {
            (int(first_indices[match.trainIdx]),
             int(second_indices[match.queryIdx]))
            for match in reverse
        }

        mutual = []
        for match in forward:
            first_index = int(first_indices[match.queryIdx])
            second_index = int(second_indices[match.trainIdx])
            if (first_index, second_index) in reverse_pairs:
                mutual.append((first_index, second_index, float(match.distance)))
        mutual.sort(key=lambda item: item[2])
        return mutual

    def _process(self, kf: LoopKeyFrameSnapshot):
        if kf.id in self.kfs:
            return
        initial_pose = kf.pose.copy()
        if (kf.previous_id is not None
                and kf.previous_id in self.kfs
                and kf.T_previous_current is not None):
            self.odometry_edges.append(PoseGraphEdge(
                from_id=kf.previous_id,
                to_id=kf.id,
                measurement=kf.T_previous_current.copy(),
                weight=1.0,
                kind="odometry"))
            if kf.previous_id in self.pose_estimates:
                initial_pose = (
                    self.pose_estimates[kf.previous_id]
                    @ kf.T_previous_current)

        self.kfs[kf.id] = kf
        self.pose_estimates[kf.id] = initial_pose
        constraint = self._detect(kf)
        if constraint is not None:
            self._close(constraint)

    def _detect(self, kf: LoopKeyFrameSnapshot) -> Optional[LoopConstraint]:
        candidates = []
        current_valid = len(self._valid_feature_indices(kf))
        if current_valid < self.min_descriptor_matches:
            return None
        for candidate_id, candidate in self.kfs.items():
            if candidate_id == kf.id or abs(candidate_id - kf.id) < self.min_interval:
                continue
            matches = self._mutual_ratio_matches(candidate, kf)
            if len(matches) < self.min_descriptor_matches:
                continue
            candidate_valid = len(self._valid_feature_indices(candidate))
            score = len(matches) / max(1, min(current_valid, candidate_valid))
            candidates.append((len(matches), score, candidate_id, matches))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        verified = []
        for _count, score, candidate_id, matches in candidates[:self.max_candidates]:
            constraint = self._verify(
                self.kfs[candidate_id], kf, matches, appearance_score=score)
            if constraint is not None:
                verified.append(constraint)
        if not verified:
            return None
        verified.sort(
            key=lambda item: (item.inliers, item.inlier_ratio,
                              -item.reprojection_rmse),
            reverse=True)
        return verified[0]

    def _estimate_pnp(self, object_points: np.ndarray,
                      image_points: np.ndarray,
                      image_size: Tuple[int, int]):
        valid = (
            np.all(np.isfinite(object_points), axis=1)
            & (object_points[:, 2] > 0.1)
            & np.all(np.isfinite(image_points), axis=1)
        )
        valid_indices = np.flatnonzero(valid)
        object_points = np.asarray(object_points[valid], dtype=np.float64)
        image_points = np.asarray(image_points[valid], dtype=np.float64)
        if len(object_points) < self.min_pnp_inliers:
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.K,
            None,
            iterationsCount=500,
            reprojectionError=self.pnp_reproj_threshold,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None:
            return None
        inliers = inliers.reshape(-1)
        inlier_ratio = len(inliers) / max(1, len(object_points))
        if (len(inliers) < self.min_pnp_inliers
                or inlier_ratio < self.min_inlier_ratio):
            return None

        try:
            if hasattr(cv2, "solvePnPRefineLM") and len(inliers) >= 6:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[inliers], image_points[inliers],
                    self.K, None, rvec, tvec)
        except cv2.error:
            pass

        projected, _ = cv2.projectPoints(
            object_points[inliers], rvec, tvec, self.K, None)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points[inliers], axis=1)
        rmse = float(np.sqrt(np.mean(errors * errors)))
        if not np.isfinite(rmse) or rmse > self.max_reprojection_rmse:
            return None

        height, width = image_size
        if len(inliers) >= 3 and height > 0 and width > 0:
            hull = cv2.convexHull(
                image_points[inliers].astype(np.float32).reshape(-1, 1, 2))
            coverage = float(cv2.contourArea(hull)) / float(height * width)
            if coverage < self.min_image_coverage:
                return None

        R, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R
        transform[:3, 3] = np.asarray(tvec).reshape(3)
        if not np.all(np.isfinite(transform)):
            return None
        return transform, valid_indices[inliers], inlier_ratio, rmse

    def _verify(self,
                loop: LoopKeyFrameSnapshot,
                current: LoopKeyFrameSnapshot,
                matches: Optional[List[Tuple[int, int, float]]] = None,
                appearance_score: float = 0.0) -> Optional[LoopConstraint]:
        if matches is None:
            matches = self._mutual_ratio_matches(loop, current)
        if len(matches) < self.min_descriptor_matches:
            return None

        loop_indices = np.asarray([item[0] for item in matches], dtype=np.int32)
        current_indices = np.asarray([item[1] for item in matches], dtype=np.int32)

        # Forward PnP measures T_current_loop from loop-frame metric 3D points
        # and current-frame image observations. It does not use either global pose.
        forward = self._estimate_pnp(
            loop.camera_points[loop_indices],
            current.keypoints[current_indices],
            current.image_size)
        # Reciprocal PnP independently measures T_loop_current.
        reverse = self._estimate_pnp(
            current.camera_points[current_indices],
            loop.keypoints[loop_indices],
            loop.image_size)
        if forward is None or reverse is None:
            return None

        T_current_loop, inliers_f, _ratio_f, rmse_f = forward
        T_loop_current_reverse, inliers_r, _ratio_r, rmse_r = reverse
        T_loop_current = np.linalg.inv(T_current_loop)

        consistency = np.linalg.inv(T_loop_current_reverse) @ T_loop_current
        rotation_cos = np.clip(
            (np.trace(consistency[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        rotation_error = float(np.arccos(rotation_cos))
        translation_error = float(np.linalg.norm(consistency[:3, 3]))
        if (rotation_error > self.max_rotation_consistency
                or translation_error > self.max_translation_consistency):
            return None
        if np.linalg.norm(T_loop_current[:3, 3]) > self.max_loop_translation:
            return None

        shared_inliers = np.intersect1d(
            inliers_f, inliers_r, assume_unique=False)
        inliers = len(shared_inliers)
        inlier_ratio = inliers / max(1, len(matches))
        if (inliers < self.min_pnp_inliers
                or inlier_ratio < self.min_inlier_ratio):
            return None
        rmse = max(rmse_f, rmse_r)
        loop_feature_indices = loop_indices[shared_inliers].astype(
            np.int32, copy=True)
        current_feature_indices = current_indices[shared_inliers].astype(
            np.int32, copy=True)
        loop_feature_indices.setflags(write=False)
        current_feature_indices.setflags(write=False)
        return LoopConstraint(
            current_id=current.id,
            loop_id=loop.id,
            T_loop_current=T_loop_current.copy(),
            descriptor_matches=len(matches),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reprojection_rmse=rmse,
            appearance_score=float(appearance_score),
            loop_feature_indices=loop_feature_indices,
            current_feature_indices=current_feature_indices)

    def _close(self, constraint: LoopConstraint):
        if any(edge.from_id == constraint.loop_id
               and edge.to_id == constraint.current_id
               for edge in self.loop_edges):
            return
        loop_weight = min(
            5.0, max(1.0, np.sqrt(
                constraint.inliers / max(1, self.min_pnp_inliers))))
        edge = PoseGraphEdge(
            from_id=constraint.loop_id,
            to_id=constraint.current_id,
            measurement=constraint.T_loop_current.copy(),
            weight=float(loop_weight),
            kind="loop")
        self.loop_edges.append(edge)
        optimized = self._pgo_scipy()
        if optimized is None:
            self.loop_edges.pop()
            return
        corrections, optimized_poses = optimized
        result = LoopClosureResult(
            constraint=constraint,
            corrections=corrections,
            optimized_poses=optimized_poses)
        print(
            f"[LoopClosing] KF{constraint.current_id} → KF{constraint.loop_id}: "
            f"matches={constraint.descriptor_matches} "
            f"inliers={constraint.inliers} "
            f"rmse={constraint.reprojection_rmse:.2f}px")
        if self.on_loop_result is not None:
            self.on_loop_result(result)

    def _pgo_scipy(self):
        from scipy.sparse import lil_matrix

        ids = sorted(self.pose_estimates)
        if len(ids) < 2:
            return None
        edges = [
            edge for edge in self.odometry_edges + self.loop_edges
            if edge.from_id in self.pose_estimates
            and edge.to_id in self.pose_estimates
        ]
        if not edges:
            return None

        old_poses = {
            kid: self.pose_estimates[kid].copy() for kid in ids
        }
        anchor_id = ids[0]
        anchor_pose = old_poses[anchor_id]
        free_ids = ids[1:]
        id_to_free = {kid: index for index, kid in enumerate(free_ids)}
        x0 = np.zeros(6 * len(free_ids), dtype=np.float64)
        for index, kid in enumerate(free_ids):
            pose = old_poses[kid]
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            x0[6*index:6*index+3] = rvec.reshape(3)
            x0[6*index+3:6*index+6] = pose[:3, 3]

        def decode(x):
            poses = {anchor_id: anchor_pose}
            for index, kid in enumerate(free_ids):
                base = 6 * index
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = cv2.Rodrigues(x[base:base+3])[0]
                pose[:3, 3] = x[base+3:base+6]
                poses[kid] = pose
            return poses

        def residuals(x):
            poses = decode(x)
            values = []
            for edge in edges:
                estimate = (
                    np.linalg.inv(poses[edge.from_id]) @ poses[edge.to_id])
                error = np.linalg.inv(edge.measurement) @ estimate
                rotation_error, _ = cv2.Rodrigues(error[:3, :3])
                values.extend(rotation_error.reshape(3) * edge.weight)
                values.extend(error[:3, 3] * edge.weight)
            return np.asarray(values, dtype=np.float64)

        sparsity = lil_matrix((6 * len(edges), len(x0)), dtype=np.int8)
        for edge_index, edge in enumerate(edges):
            rows = slice(6 * edge_index, 6 * edge_index + 6)
            for kid in (edge.from_id, edge.to_id):
                if kid in id_to_free:
                    col = 6 * id_to_free[kid]
                    sparsity[rows, col:col+6] = 1

        initial_residuals = residuals(x0)
        initial_cost = float(np.dot(initial_residuals, initial_residuals))
        try:
            result = least_squares(
                residuals,
                x0,
                method="trf",
                jac_sparsity=sparsity.tocsr(),
                loss="huber",
                f_scale=1.0,
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-5,
                max_nfev=500,
                x_scale="jac",
                verbose=0)
        except Exception as exc:
            self.last_error = f"pose_graph_failed: {exc}"
            print(f"[LoopClosing] 位姿图未提交: {exc}")
            return None
        if not np.all(np.isfinite(result.x)):
            self.last_error = "pose_graph_non_finite"
            return None

        final_residuals = residuals(result.x)
        final_cost = float(np.dot(final_residuals, final_residuals))
        if (not np.isfinite(final_cost)
                or final_cost > initial_cost * 1.01 + 1e-12):
            self.last_error = (
                f"pose_graph_cost_not_improved: {initial_cost} -> {final_cost}")
            return None

        optimized_poses = {
            kid: pose.copy() for kid, pose in decode(result.x).items()
        }
        # The fixed anchor is guaranteed bit-identical.
        optimized_poses[anchor_id] = old_poses[anchor_id].copy()
        corrections = {
            kid: optimized_poses[kid] @ np.linalg.inv(old_poses[kid])
            for kid in ids
        }
        self.pose_estimates = {
            kid: pose.copy() for kid, pose in optimized_poses.items()
        }
        self.last_error = ""
        return corrections, optimized_poses


class Relocalizer:
    """Synchronous, backend-independent keyframe relocalization database.

    Keyframes are stored as detached CPU snapshots.  A query first retrieves
    candidates with mutual ORB matches and then reuses the loop closer's
    reciprocal-PnP, reprojection, coverage and transform-consistency checks.
    No global-pose prior is used, which is essential once tracking is lost.
    """

    def __init__(self, K: np.ndarray, max_relative_translation: float = 3.0):
        self.K = np.asarray(K, dtype=np.float64).copy()
        # This LoopClosing instance is only a geometry verifier.  It is never
        # started and owns neither a worker thread nor the RoMa matcher/GPU.
        self.geometry = LoopClosing(
            queue.Queue(),
            self.K,
            matcher=None,
            min_dist_thresh=max_relative_translation)
        self.kfs: Dict[int, LoopKeyFrameSnapshot] = {}
        self.last_error = ""

    def add_snapshot(self, snapshot: LoopKeyFrameSnapshot) -> None:
        """Register one already-detached snapshot by keyframe id."""
        if not isinstance(snapshot, LoopKeyFrameSnapshot):
            raise TypeError("relocalization snapshot has an invalid type")
        self.kfs[int(snapshot.id)] = snapshot

    def add_keyframe(self, kf: KeyFrame) -> LoopKeyFrameSnapshot:
        snapshot = make_keyframe_geometry_snapshot(kf, self.K)
        self.add_snapshot(snapshot)
        return snapshot

    def get_snapshot(self, keyframe_id: int) -> Optional[LoopKeyFrameSnapshot]:
        return self.kfs.get(int(keyframe_id))

    def remove(self, keyframe_id: int) -> None:
        self.kfs.pop(int(keyframe_id), None)

    def query(
            self,
            current: LoopKeyFrameSnapshot,
            allowed_ids: Optional[set] = None) -> Optional[LoopConstraint]:
        """Return the strongest fully verified historical match, if any."""
        try:
            allowed = None if allowed_ids is None else {
                int(keyframe_id) for keyframe_id in allowed_ids
            }
            current_valid = len(
                self.geometry._valid_feature_indices(current))
            if current_valid < self.geometry.min_descriptor_matches:
                self.last_error = "insufficient_current_features"
                return None

            candidates = []
            for candidate_id, candidate in self.kfs.items():
                if candidate_id == current.id:
                    continue
                if allowed is not None and candidate_id not in allowed:
                    continue
                matches = self.geometry._mutual_ratio_matches(
                    candidate, current)
                if len(matches) < self.geometry.min_descriptor_matches:
                    continue
                candidate_valid = len(
                    self.geometry._valid_feature_indices(candidate))
                score = len(matches) / max(
                    1, min(current_valid, candidate_valid))
                candidates.append(
                    (len(matches), score, candidate_id, matches))

            candidates.sort(
                key=lambda item: (item[0], item[1]), reverse=True)
            verified = []
            for _count, score, candidate_id, matches in candidates[
                    :self.geometry.max_candidates]:
                constraint = self.geometry._verify(
                    self.kfs[candidate_id],
                    current,
                    matches,
                    appearance_score=score)
                if constraint is not None:
                    verified.append(constraint)
            if not verified:
                self.last_error = "no_geometrically_verified_candidate"
                return None

            verified.sort(
                key=lambda item: (
                    item.inliers,
                    item.inlier_ratio,
                    -item.reprojection_rmse),
                reverse=True)
            self.last_error = ""
            return verified[0]
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


class BackendOptimizer:
    def __init__(self, K: np.ndarray, enable_loop_closing: bool = True,
                 matcher: "RoMaMatcher" = None,
                 loop_dist_thresh: float = 3.0):
        self.K = K
        self.all_kfs: Dict[int, KeyFrame] = {}
        self.lc_queue: Optional[queue.Queue] = None
        self.loop_closing: Optional[LoopClosing] = None
        self.loop_results: queue.Queue = queue.Queue()
        self._result_lock = threading.Lock()
        self._latest_optimized_poses: Dict[int, np.ndarray] = {}
        if enable_loop_closing:
            self.lc_queue = queue.Queue()
            self.loop_closing = LoopClosing(
                self.lc_queue, K, matcher=matcher,
                min_dist_thresh=loop_dist_thresh)
            self.loop_closing.on_loop_result = self._on_loop_result

    def _on_loop_result(self, result: LoopClosureResult):
        """Worker callback: enqueue immutable copies; never touch live SLAM state."""
        copied = LoopClosureResult(
            constraint=result.constraint,
            corrections={
                kid: correction.copy()
                for kid, correction in result.corrections.items()
            },
            optimized_poses={
                kid: pose.copy()
                for kid, pose in result.optimized_poses.items()
            })
        with self._result_lock:
            self._latest_optimized_poses = {
                kid: pose.copy()
                for kid, pose in copied.optimized_poses.items()
            }
        self.loop_results.put(copied)

    def start(self):
        if self.loop_closing:
            self.loop_closing.start()
            print("[Backend] 回环检测线程已启动 (ORB + reciprocal PnP + scipy SE3)")

    def stop(self):
        if self.loop_closing:
            self.loop_closing.stop()

    def add_keyframe(self, kf: KeyFrame,
                     base_snapshot: Optional[LoopKeyFrameSnapshot] = None):
        previous_id = max(self.all_kfs) if self.all_kfs else None
        T_previous_current = None
        if previous_id is not None:
            previous_pose = self.all_kfs[previous_id].pose.copy()
            T_previous_current = np.linalg.inv(previous_pose) @ kf.pose
        self.all_kfs[kf.id] = kf
        if self.lc_queue:
            self.lc_queue.put(self._make_loop_snapshot(
                kf,
                previous_id=previous_id,
                T_previous_current=T_previous_current,
                base_snapshot=base_snapshot))

    def _make_loop_snapshot(
            self,
            kf: KeyFrame,
            previous_id: Optional[int] = None,
            T_previous_current: Optional[np.ndarray] = None,
            base_snapshot: Optional[LoopKeyFrameSnapshot] = None
            ) -> LoopKeyFrameSnapshot:
        if base_snapshot is not None and int(base_snapshot.id) == int(kf.id):
            # 复用前端为重定位库构建的几何快照（关键点/描述子/相机坐标
            # 与位姿无关），省掉每关键帧重复一次的空间均衡特征采样；
            # 只补冻结的里程计边信息，并刷新可能已被 BA 修正的位姿。
            from dataclasses import replace as dataclass_replace
            frozen_odometry = None
            if T_previous_current is not None:
                frozen_odometry = np.asarray(
                    T_previous_current, dtype=np.float64).copy()
                frozen_odometry.setflags(write=False)
            frozen_pose = np.asarray(kf.pose, dtype=np.float64).copy()
            frozen_pose.setflags(write=False)
            return dataclass_replace(
                base_snapshot,
                pose=frozen_pose,
                previous_id=previous_id,
                T_previous_current=frozen_odometry)
        return make_keyframe_geometry_snapshot(
            kf,
            self.K,
            previous_id=previous_id,
            T_previous_current=T_previous_current)

    def consume_loop_results(self) -> List[LoopClosureResult]:
        results = []
        while True:
            try:
                results.append(self.loop_results.get_nowait())
            except queue.Empty:
                break
        return results

    def get_optimized_poses(self) -> Dict[int, np.ndarray]:
        with self._result_lock:
            return {
                kid: pose.copy()
                for kid, pose in self._latest_optimized_poses.items()
            }


# ══════════════════════════════════════════════════════════════
#  主 SLAM 系统
# ══════════════════════════════════════════════════════════════

class StereoSLAMv2:
    """
    RoMa 双目 SLAM — 增强版

    流程：
        1. 双目 RoMa 匹配 → 改进三角化 → 当前帧 3D 点
        2. 帧间 RoMa 匹配（关键帧 ↔ 当前帧）
        3. （可选）IMU 旋转预积分提供 PnP 初值
        4. PnP RANSAC → 当前帧位姿
        5. 关键帧判断 → 地图点关联（重投影验证）
        6. 滑动窗口 BA → 关键帧位姿 & 地图点精化
        7. 回环检测 → 位姿图优化
        8. 最终全局 BA（可选）
    """

    def __init__(self,
                 calib: dict,
                 device: str = "cuda",
                 # RoMa 参数
                 coarse_res: int = 560,
                 upsample_res: int = 560,
                 stereo_matches: int = 5000,
                 temporal_matches: int = 2000,
                 match_resize: Optional[Tuple[int, int]] = None,
                 use_custom_corr: bool = False,
                 enable_upsample_refinement: bool = True,
                 stereo_upsample_refinement: Optional[bool] = None,
                 temporal_upsample_refinement: Optional[bool] = None,
                 symmetric_matching: bool = True,
                 enable_feature_reuse: bool = True,
                 batch_coarse_pairs: bool = False,
                 enable_klt_temporal_tracking: bool = False,
                 klt_max_points: int = 800,
                 klt_min_tracks: int = 60,
                 klt_min_track_ratio: float = 0.25,
                 klt_fb_max_error: float = 1.5,
                 klt_image_scale: float = 0.25,
                 klt_min_pnp_inlier_ratio: float = 0.40,
                 # 深度范围
                 min_depth: float = 0.5,
                 max_depth: float = 200.0,
                 epipolar_threshold: float = 2.0,
                 # 关键帧策略
                 keyframe_interval: int = 5,
                 keyframe_min_inliers: int = 50,
                 keyframe_min_gap: int = 1,
                 keyframe_interval_from_last: bool = False,
                 # 重定位：失败帧两次全库检索之间的最小帧间隔（1=逐帧）
                 relocalization_min_interval: int = 1,
                 # 后端
                 enable_backend: bool = True,
                 enable_loop_closing: bool = True,
                 backend_window_size: int = 10,
                 ba_frequency: int = 3,
                 loop_dist_thresh: float = 3.0,
                 # IMU
                 imu_helper: Optional[IMUHelper] = None,
                 # 关联参数
                 assoc_pixel_thresh: float = 5.0,
                 assoc_reproj_thresh: float = 3.0):

        # ── 相机参数 ──
        cam = calib['camera']
        self.fx, self.fy = cam['fx'], cam['fy']
        self.cx, self.cy = cam['cx'], cam['cy']
        self.baseline = calib['baseline']
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float64)
        print(f"[Calib] fx={self.fx} fy={self.fy} cx={self.cx} cy={self.cy} B={self.baseline}")

        # ── 匹配器 ──
        self.matcher = RoMaMatcher(device=device, coarse_res=coarse_res,
                                   upsample_res=upsample_res,
                                   use_custom_corr=use_custom_corr,
                                   enable_upsample_refinement=(
                                       enable_upsample_refinement),
                                   symmetric_matching=symmetric_matching)
        self.stereo_matches  = stereo_matches
        self.temporal_matches = temporal_matches
        self.match_resize = match_resize
        self.stereo_upsample_refinement = bool(
            enable_upsample_refinement
            if stereo_upsample_refinement is None
            else stereo_upsample_refinement
        )
        self.temporal_upsample_refinement = bool(
            enable_upsample_refinement
            if temporal_upsample_refinement is None
            else temporal_upsample_refinement
        )
        self.symmetric_matching = bool(symmetric_matching)
        self.feature_reuse_requested = bool(enable_feature_reuse)
        self.batch_coarse_pairs = bool(batch_coarse_pairs)
        self.enable_klt_temporal_tracking = bool(
            enable_klt_temporal_tracking
        )
        self.klt_max_points = max(80, int(klt_max_points))
        self.klt_min_tracks = max(6, int(klt_min_tracks))
        self.klt_min_track_ratio = float(np.clip(
            klt_min_track_ratio, 0.0, 1.0
        ))
        self.klt_fb_max_error = max(0.1, float(klt_fb_max_error))
        self.klt_image_scale = float(np.clip(
            klt_image_scale, 0.1, 1.0
        ))
        self.klt_min_pnp_inlier_ratio = float(np.clip(
            klt_min_pnp_inlier_ratio, 0.0, 1.0
        ))
        self.feature_reuse_active = (
            self.feature_reuse_requested
            and self.matcher.supports_feature_reuse(
                self.stereo_upsample_refinement,
                self.temporal_upsample_refinement,
                self.symmetric_matching,
            )
        )
        if self.feature_reuse_requested and not self.feature_reuse_active:
            print(
                "[RoMa] 特征复用已自动关闭：当前精修/对称配置不受支持"
            )

        # ── 三角化 ──
        self.triangulator = StereoTriangulator(
            self.K,
            self.baseline,
            min_depth,
            max_depth,
            epipolar_threshold=epipolar_threshold,
        )

        # ── 描述子 ──
        self.desc_extractor = ORBDescriptorExtractor()

        # ── 地图点关联 ──
        self.associator = MapPointAssociator(
            self.K, assoc_pixel_thresh, assoc_reproj_thresh)

        # 重定位库始终可用，不依赖可选 BA/回环后端。它只保存
        # 脱离活状态的 CPU 几何快照，也不共享 RoMa/CUDA 模型。
        self.relocalizer = Relocalizer(
            self.K, max_relative_translation=loop_dist_thresh)
        self.relocalization_min_interval = max(
            1, int(relocalization_min_interval))
        self._last_relocalization_attempt_frame: Optional[int] = None

        # ── 状态 ──
        self.poses: List[np.ndarray] = [np.eye(4)]
        self.keyframes: List[KeyFrame] = []
        self.global_mps: Dict[int, MapPoint] = {}
        self.next_mp_id = 0
        self.next_kf_id = 0
        self.frame_count = 0
        self.kf_count = 0
        self.loop_count = 0
        self.relocalization_count = 0
        self.tracking_state = "NOT_INITIALIZED"
        # Monotonic revision of already-published geometry.  New keyframes do
        # not increment it: external mappers can append those observations.
        # Accepted BA/loop corrections do increment it because historical
        # observations must be replayed at corrected poses.
        self.geometry_revision = 0
        self._geometry_corrected_in_process = False
        self.last_tracking_status = TrackingStatus()
        self.keyframe_min_inliers = max(0, int(keyframe_min_inliers))
        self.keyframe_min_gap = max(1, int(keyframe_min_gap))
        self.keyframe_interval_from_last = bool(
            keyframe_interval_from_last
        )

        # 关键帧缓存
        self._kf_img: Optional[np.ndarray] = None
        self._kf_roma_cache: Optional[dict] = None
        self._kf_gray: Optional[np.ndarray] = None
        self._kf_klt_indices: Optional[np.ndarray] = None
        self._kf_tree: Optional[cKDTree] = None
        self._kf_kpts: Optional[np.ndarray] = None
        self._kf_pts3d: Optional[np.ndarray] = None   # 相机坐标系
        self._kf_mp_ids: Optional[List[int]] = None
        self._last_kf: Optional[KeyFrame] = None
        self._last_kf_frame_idx: int = 0
        self.temporal_klt_attempts = 0
        self.temporal_klt_candidates = 0
        self.temporal_klt_successes = 0
        self.temporal_roma_fallbacks = 0
        self.last_temporal_tracking_source = "none"
        # Detached numeric match data for optional ROS/debug visualisation.
        # Images and points are reused from the already-computed front-end;
        # this never triggers another RoMa or KLT pass.
        self._last_feature_match_data = None
        self._kf_id_to_pose_idx: Dict[int, int] = {}
        # 与 poses 等长：每个普通帧保存相对其参考关键帧的刚体变换，
        # 这样 BA/回环修正关键帧后可一致重建整条轨迹，而非只改关键帧点。
        self._pose_reference_kf_ids: List[Optional[int]] = [None]
        self._pose_relative_to_kf: List[np.ndarray] = [np.eye(4)]

        # ── IMU ──
        self.imu = imu_helper
        self.keyframe_interval = keyframe_interval
        self.ba_frequency = ba_frequency

        # ── 后端 ──
        self.enable_backend = enable_backend
        if enable_backend:
            self.sw_ba = SlidingWindowBA(
                backend_window_size, self.K, self.baseline)
            self.backend = BackendOptimizer(
                self.K, enable_loop_closing=enable_loop_closing,
                matcher=self.matcher, loop_dist_thresh=loop_dist_thresh)
            self.backend.start()
        else:
            self.sw_ba = None
            self.backend = None

    # ──────────────────────────────────────────────────────────
    #  核心：处理一帧
    # ──────────────────────────────────────────────────────────
    def _set_tracking_status(
            self,
            *,
            success: bool,
            reason: str,
            pose_updated: bool = False,
            initialized: bool = False,
            inlier_count: int = 0,
            correspondence_count: int = 0,
            pose_index: Optional[int] = None,
            keyframe_created: bool = False,
            keyframe_id: Optional[int] = None) -> None:
        if pose_index is None:
            pose_index = len(self.poses) - 1 if success else -1
        self.last_tracking_status = TrackingStatus(
            success=bool(success),
            frame_index=int(self.frame_count),
            reason=str(reason),
            pose_updated=bool(pose_updated),
            initialized=bool(initialized),
            inlier_count=int(inlier_count),
            correspondence_count=int(correspondence_count),
            pose_index=int(pose_index),
            keyframe_created=bool(keyframe_created),
            keyframe_id=(None if keyframe_id is None else int(keyframe_id)),
            geometry_revision=int(getattr(self, "geometry_revision", 0)),
            geometry_corrected=bool(
                getattr(self, "_geometry_corrected_in_process", False)),
        )

    def _mark_geometry_corrected(self) -> None:
        """Publish a monotonic revision after committed historical changes."""
        self.geometry_revision = int(getattr(self, "geometry_revision", 0)) + 1
        self._geometry_corrected_in_process = True

    def get_tracking_status(self) -> TrackingStatus:
        """Return the immutable status of the latest processed frame."""
        return self.last_tracking_status

    def _record_temporal_feature_matches(
            self,
            *,
            reference_image: Optional[np.ndarray],
            reference_points: np.ndarray,
            current_points: np.ndarray,
            inlier_indices: Optional[np.ndarray],
            source: str,
            pose_success: bool) -> None:
        """Attach already-computed temporal correspondences to debug state."""
        if self._last_feature_match_data is None:
            return
        reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
        current = np.asarray(current_points, dtype=np.float32).reshape(-1, 2)
        count = min(len(reference), len(current))
        reference = reference[:count]
        current = current[:count]
        inlier_mask = np.zeros(count, dtype=bool)
        if inlier_indices is not None and count:
            indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
            indices = indices[(indices >= 0) & (indices < count)]
            inlier_mask[indices] = True
        self._last_feature_match_data.update(
            {
                "temporal_reference_image": reference_image,
                "temporal_reference_points": reference,
                "temporal_current_points": current,
                "temporal_inlier_mask": inlier_mask,
                "temporal_source": str(source),
                "temporal_pose_success": bool(pose_success),
            }
        )

    def _mark_feature_match_tracking_success(self, success: bool) -> None:
        if self._last_feature_match_data is not None:
            self._last_feature_match_data["tracking_success"] = bool(success)

    def get_feature_match_snapshot(self) -> Optional[dict]:
        """Return the latest match evidence without running feature inference.

        Numeric arrays are detached and read-only.  The reference image is a
        previous keyframe array retained by this snapshot and must be treated
        as read-only by the immediate consumer.
        """
        data = self._last_feature_match_data
        if data is None:
            return None
        return {
            "frame_index": int(data["frame_index"]),
            "stereo_left_points": self._readonly_copy(
                data["stereo_left_points"], dtype=np.float32
            ),
            "stereo_right_points": self._readonly_copy(
                data["stereo_right_points"], dtype=np.float32
            ),
            "temporal_reference_image": data["temporal_reference_image"],
            "temporal_reference_points": self._readonly_copy(
                data["temporal_reference_points"], dtype=np.float32
            ),
            "temporal_current_points": self._readonly_copy(
                data["temporal_current_points"], dtype=np.float32
            ),
            "temporal_inlier_mask": self._readonly_copy(
                data["temporal_inlier_mask"], dtype=bool
            ),
            "temporal_source": str(data["temporal_source"]),
            "temporal_pose_success": bool(data["temporal_pose_success"]),
            "tracking_success": bool(data["tracking_success"]),
        }

    @staticmethod
    def _snapshot_source_index(
            snapshot: LoopKeyFrameSnapshot,
            feature_index: int) -> Optional[int]:
        """Map a bounded snapshot feature back to its keyframe array index."""
        feature_index = int(feature_index)
        if feature_index < 0 or feature_index >= len(snapshot.keypoints):
            return None
        source_indices = np.asarray(snapshot.source_indices).reshape(-1)
        if len(source_indices) == len(snapshot.keypoints):
            return int(source_indices[feature_index])
        # Compatibility for snapshots created before source indices existed.
        return feature_index

    def _make_relocalization_query(
            self,
            left: np.ndarray,
            pts3d_cam: np.ndarray,
            kp_l_valid: np.ndarray,
            kp_r_valid: np.ndarray
            ) -> Tuple[LoopKeyFrameSnapshot, np.ndarray]:
        """Build a detached current-frame query without touching SLAM state."""
        descriptors = self.desc_extractor.extract_at_points(
            left, kp_l_valid)
        query_keyframe = KeyFrame(
            id=self.next_kf_id,
            timestamp=time.time(),
            pose=np.eye(4, dtype=np.float64),
            image=left.copy(),
            keypoints=[point.tolist() for point in kp_l_valid],
            right_keypoints=[point.tolist() for point in kp_r_valid],
            camera_points=[point.tolist() for point in pts3d_cam],
            descriptors=descriptors,
            K=self.K)
        return (
            make_keyframe_geometry_snapshot(query_keyframe, self.K),
            descriptors)

    def _relocalization_associations(
            self,
            constraint: LoopConstraint,
            current_snapshot: LoopKeyFrameSnapshot
            ) -> Dict[int, int]:
        """Reconnect verified current features to live historical MapPoints."""
        relocalizer = getattr(self, "relocalizer", None)
        if relocalizer is None:
            return {}
        loop_snapshot = relocalizer.get_snapshot(constraint.loop_id)
        loop_keyframe = next(
            (item for item in self.keyframes
             if item.id == constraint.loop_id and not item.is_bad),
            None)
        if loop_snapshot is None or loop_keyframe is None:
            return {}

        associations: Dict[int, int] = {}
        used_map_points = set()
        pairs = zip(
            np.asarray(constraint.loop_feature_indices).reshape(-1),
            np.asarray(constraint.current_feature_indices).reshape(-1))
        for loop_feature_index, current_feature_index in pairs:
            loop_source_index = self._snapshot_source_index(
                loop_snapshot, int(loop_feature_index))
            current_source_index = self._snapshot_source_index(
                current_snapshot, int(current_feature_index))
            if loop_source_index is None or current_source_index is None:
                continue
            if (loop_source_index >= len(loop_keyframe.mappoints)
                    or current_source_index in associations):
                continue
            mp = loop_keyframe.mappoints[loop_source_index]
            if (mp is None or mp.is_bad or mp.id not in self.global_mps
                    or mp.id in used_map_points):
                continue
            associations[current_source_index] = mp.id
            used_map_points.add(mp.id)
        return associations

    def _commit_relocalized_frame(
            self,
            left: np.ndarray,
            pts3d_cam: np.ndarray,
            kp_l_valid: np.ndarray,
            kp_r_valid: np.ndarray,
            descriptors: np.ndarray,
            current_snapshot: LoopKeyFrameSnapshot,
            constraint: LoopConstraint,
            T_wc: np.ndarray) -> np.ndarray:
        """Atomically establish a new tracking anchor after verification."""
        T_wc = np.asarray(T_wc, dtype=np.float64).copy()
        if T_wc.shape != (4, 4) or not np.all(np.isfinite(T_wc)):
            raise ValueError("relocalized pose is not a finite 4x4 transform")

        # Prepare every array that is needed after the keyframe commit first.
        cached_image = left.copy()
        cached_points = pts3d_cam.copy()
        cached_keypoints = kp_l_valid.copy()
        pts3d_world = (
            T_wc[:3, :3] @ pts3d_cam.T + T_wc[:3, 3:4]).T
        associations = self._relocalization_associations(
            constraint, current_snapshot)

        pose_index = len(self.poses)
        keyframe_id = self.next_kf_id
        self.poses.append(T_wc.copy())
        self._pose_reference_kf_ids.append(keyframe_id)
        self._pose_relative_to_kf.append(np.eye(4, dtype=np.float64))
        try:
            keyframe = self._create_keyframe(
                left,
                pts3d_cam,
                pts3d_world,
                kp_l_valid,
                kp_r_valid,
                T_wc,
                associations,
                descriptors=descriptors)
        except Exception:
            # _create_keyframe performs all fallible preparation before its
            # live-map commit, so these are the only staged writes to undo.
            self.poses.pop()
            self._pose_reference_kf_ids.pop()
            self._pose_relative_to_kf.pop()
            raise

        self._kf_id_to_pose_idx[keyframe.id] = pose_index
        self._update_kf_cache(
            cached_image,
            cached_points,
            cached_keypoints,
            keyframe,
            roma_cache=None,
        )
        T_wc = self._submit_keyframe_to_backend(keyframe)

        self.relocalization_count += 1
        self.tracking_state = "TRACKING"
        self._set_tracking_status(
            success=True,
            reason="relocalized",
            pose_updated=True,
            inlier_count=constraint.inliers,
            correspondence_count=constraint.descriptor_matches,
            pose_index=pose_index,
            keyframe_created=True,
            keyframe_id=keyframe.id)
        print(
            f"[Relocalization] #{self.relocalization_count}: "
            f"KF{keyframe.id} → KF{constraint.loop_id}, "
            f"inliers={constraint.inliers}, "
            f"rmse={constraint.reprojection_rmse:.2f}px")
        return T_wc

    def _try_relocalize(
            self,
            left: np.ndarray,
            pts3d_cam: np.ndarray,
            kp_l_valid: np.ndarray,
            kp_r_valid: np.ndarray) -> Optional[np.ndarray]:
        """Verify a recovery without mutating state, then commit it once."""
        relocalizer = getattr(self, "relocalizer", None)
        if relocalizer is None or not relocalizer.kfs:
            return None
        # 失败帧的全库 ORB 检索跑在跟踪线程上，代价随历史关键帧数线性
        # 增长。距上次尝试不足 relocalization_min_interval 帧时跳过本帧，
        # 保护连续丢失期间的帧率。默认 1 保持逐帧尝试的原行为。
        min_interval = max(
            1, int(getattr(self, "relocalization_min_interval", 1)))
        last_attempt = getattr(
            self, "_last_relocalization_attempt_frame", None)
        if (min_interval > 1 and last_attempt is not None
                and self.frame_count - last_attempt < min_interval):
            return None
        self._last_relocalization_attempt_frame = self.frame_count
        try:
            current_snapshot, descriptors = self._make_relocalization_query(
                left, pts3d_cam, kp_l_valid, kp_r_valid)
            live_keyframes = {
                keyframe.id for keyframe in self.keyframes
                if not keyframe.is_bad
            }
            constraint = relocalizer.query(
                current_snapshot, allowed_ids=live_keyframes)
            if constraint is None:
                return None
            reference = next(
                (keyframe for keyframe in self.keyframes
                 if keyframe.id == constraint.loop_id
                 and not keyframe.is_bad),
                None)
            if reference is None:
                return None
            # T_loop_current maps current-camera coordinates into the matched
            # historical camera.  Compose it with the live (possibly BA/loop
            # corrected) historical pose to recover the current world pose.
            T_wc = reference.pose.copy() @ constraint.T_loop_current
        except Exception as exc:
            print(
                "[Relocalization] 候选验证异常，未修改 SLAM 状态: "
                f"{type(exc).__name__}: {exc}")
            return None

        # All acceptance checks completed above.  Any programming error during
        # commit is deliberately surfaced instead of being mislabeled as an
        # ordinary tracking failure after state has changed.
        return self._commit_relocalized_frame(
            left,
            pts3d_cam,
            kp_l_valid,
            kp_r_valid,
            descriptors,
            current_snapshot,
            constraint,
            T_wc)

    def _handle_tracking_failure(
            self,
            *,
            reason: str,
            left: np.ndarray,
            pts3d_cam: np.ndarray,
            kp_l_valid: np.ndarray,
            kp_r_valid: np.ndarray,
            inlier_count: int = 0,
            correspondence_count: int = 0,
            allow_relocalization: bool = True) -> np.ndarray:
        """Attempt recovery first; otherwise preserve the last known pose."""
        if allow_relocalization:
            recovered_pose = self._try_relocalize(
                left, pts3d_cam, kp_l_valid, kp_r_valid)
            if recovered_pose is not None:
                return recovered_pose
        self.tracking_state = (
            "LOST" if getattr(self, "keyframes", []) else "NOT_INITIALIZED")
        self._set_tracking_status(
            success=False,
            reason=reason,
            inlier_count=inlier_count,
            correspondence_count=correspondence_count)
        return self.poses[-1]

    def _build_klt_indices(self, points: np.ndarray) -> np.ndarray:
        """Choose a deterministic, spatially distributed KLT seed subset."""
        count = len(points)
        if count <= self.klt_max_points:
            return np.arange(count, dtype=np.int32)
        if self._kf_img is None:
            return np.arange(self.klt_max_points, dtype=np.int32)

        height, width = self._kf_img.shape[:2]
        columns, rows = 8, 6
        cell_x = np.clip(
            (points[:, 0] * columns / max(width, 1)).astype(np.int32),
            0,
            columns - 1,
        )
        cell_y = np.clip(
            (points[:, 1] * rows / max(height, 1)).astype(np.int32),
            0,
            rows - 1,
        )
        cell_ids = cell_y * columns + cell_x
        per_cell = int(np.ceil(self.klt_max_points / (columns * rows)))
        selected = []
        selected_mask = np.zeros(count, dtype=bool)
        for cell_id in range(columns * rows):
            indices = np.flatnonzero(cell_ids == cell_id)[:per_cell]
            selected.extend(indices.tolist())
            selected_mask[indices] = True
        if len(selected) < self.klt_max_points:
            remaining = np.flatnonzero(~selected_mask)
            selected.extend(
                remaining[:self.klt_max_points - len(selected)].tolist()
            )
        return np.asarray(selected[:self.klt_max_points], dtype=np.int32)

    def _track_keyframe_with_klt(
            self, current_left: np.ndarray
            ) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
        """Track cached keyframe 3-D observations with guarded KLT flow."""
        if (not self.enable_klt_temporal_tracking
                or self._kf_gray is None
                or self._kf_kpts is None
                or self._kf_klt_indices is None
                or len(self._kf_klt_indices) < 6):
            return None

        self.temporal_klt_attempts += 1
        original_keyframe_points = self._kf_kpts[
            self._kf_klt_indices
        ].astype(
            np.float32, copy=False
        ).reshape(-1, 1, 2)
        keyframe_points = (
            original_keyframe_points * self.klt_image_scale
        ).astype(np.float32, copy=False)
        current_gray_full = (
            cv2.cvtColor(current_left, cv2.COLOR_BGR2GRAY)
            if current_left.ndim == 3
            else np.asarray(current_left, dtype=np.uint8)
        )
        current_gray = cv2.resize(
            current_gray_full,
            None,
            fx=self.klt_image_scale,
            fy=self.klt_image_scale,
            interpolation=cv2.INTER_AREA,
        )

        initial_points = None
        flow_flags = 0
        if self.imu is not None and self._last_kf_frame_idx > 0:
            try:
                delta_rotation = self.imu.get_delta_R_cam(
                    self._last_kf_frame_idx, self.frame_count
                )
                if delta_rotation is not None:
                    flat = original_keyframe_points[:, 0, :].astype(
                        np.float64
                    )
                    rays = np.column_stack(
                        (flat, np.ones(len(flat), dtype=np.float64))
                    )
                    rotated = (
                        self.K @ np.asarray(delta_rotation, dtype=np.float64)
                        @ np.linalg.inv(self.K) @ rays.T
                    ).T
                    valid_depth = np.abs(rotated[:, 2]) > 1e-8
                    predicted = flat.copy()
                    predicted[valid_depth] = (
                        rotated[valid_depth, :2]
                        / rotated[valid_depth, 2:3]
                    )
                    if np.all(np.isfinite(predicted)):
                        initial_points = (
                            predicted * self.klt_image_scale
                        ).astype(np.float32).reshape(-1, 1, 2)
                        flow_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
            except Exception:
                initial_points = None
                flow_flags = 0

        lk_parameters = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
            "minEigThreshold": 1e-4,
        }
        tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self._kf_gray,
            current_gray,
            keyframe_points,
            initial_points,
            flags=flow_flags,
            **lk_parameters,
        )
        if tracked is None or forward_status is None:
            return None
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._kf_gray,
            tracked,
            None,
            **lk_parameters,
        )
        if backward is None or backward_status is None:
            return None

        seeds = keyframe_points[:, 0, :]
        tracked_flat = tracked[:, 0, :]
        backward_flat = backward[:, 0, :]
        fb_error = (
            np.linalg.norm(backward_flat - seeds, axis=1)
            / self.klt_image_scale
        )
        height, width = current_gray.shape[:2]
        valid = (
            (forward_status.reshape(-1) > 0)
            & (backward_status.reshape(-1) > 0)
            & np.all(np.isfinite(tracked_flat), axis=1)
            & (fb_error <= self.klt_fb_max_error)
            & (tracked_flat[:, 0] >= 2.0)
            & (tracked_flat[:, 0] < width - 2.0)
            & (tracked_flat[:, 1] >= 2.0)
            & (tracked_flat[:, 1] < height - 2.0)
        )
        valid_count = int(np.count_nonzero(valid))
        track_ratio = valid_count / max(len(seeds), 1)
        if (valid_count < self.klt_min_tracks
                or track_ratio < self.klt_min_track_ratio):
            return None

        valid_points = tracked_flat[valid] / self.klt_image_scale
        full_height, full_width = current_gray_full.shape[:2]
        coverage_x = np.clip(
            (valid_points[:, 0] * 6 / max(full_width, 1)).astype(np.int32),
            0,
            5,
        )
        coverage_y = np.clip(
            (valid_points[:, 1] * 4 / max(full_height, 1)).astype(np.int32),
            0,
            3,
        )
        occupied_cells = int(len(np.unique(coverage_y * 6 + coverage_x)))
        if occupied_cells < 6:
            return None

        self.temporal_klt_candidates += 1
        metrics = {
            "seed_count": int(len(seeds)),
            "track_count": valid_count,
            "track_ratio": float(track_ratio),
            "median_fb_error": float(np.median(fb_error[valid])),
            "occupied_cells": occupied_cells,
        }
        return (
            original_keyframe_points[:, 0, :][valid].astype(
                np.float32, copy=False
            ),
            valid_points.astype(np.float32, copy=False),
            metrics,
        )

    def _estimate_temporal_pose(
            self, keyframe_points: np.ndarray, current_points: np.ndarray
            ) -> Tuple[Optional[dict], dict]:
        """Associate temporal matches and solve the unchanged PnP problem."""
        if len(keyframe_points) < 10:
            return None, {
                "reason": "insufficient_temporal_matches",
                "inlier_count": 0,
                "correspondence_count": len(keyframe_points),
            }

        tree = self._kf_tree
        if tree is None:
            tree = cKDTree(self._kf_kpts)
        distances, indices = tree.query(keyframe_points, k=1)
        good = distances < 3.0
        points_3d = self._kf_pts3d[indices[good]]
        points_2d = current_points[good].astype(np.float32)
        keyframe_indices = indices[good]
        if len(points_3d) < 6:
            return None, {
                "reason": "insufficient_3d_2d_correspondences",
                "inlier_count": 0,
                "correspondence_count": len(points_3d),
            }

        rotation_guess = translation_guess = None
        use_guess = False
        if self.imu is not None and self._last_kf_frame_idx > 0:
            delta_rotation = self.imu.get_delta_R_cam(
                self._last_kf_frame_idx, self.frame_count
            )
            if delta_rotation is not None:
                rotation_guess, _ = cv2.Rodrigues(
                    delta_rotation.astype(np.float64)
                )
                translation_guess = np.zeros((3, 1), dtype=np.float64)
                use_guess = True

        success, rotation_vector, translation_vector, inliers = (
            cv2.solvePnPRansac(
                points_3d.astype(np.float32),
                points_2d,
                self.K.astype(np.float32),
                None,
                rvec=rotation_guess,
                tvec=translation_guess,
                useExtrinsicGuess=use_guess,
                iterationsCount=500,
                reprojectionError=2.0,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )
        if not success or inliers is None or len(inliers) < 6:
            return None, {
                "reason": "pnp_failed",
                "inlier_count": 0 if inliers is None else len(inliers),
                "correspondence_count": len(points_3d),
            }
        return {
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector,
            "inliers": inliers,
            "points_3d": points_3d,
            "points_2d": points_2d,
            "keyframe_indices": keyframe_indices,
            "good": good,
        }, {
            "reason": "tracking_success",
            "inlier_count": len(inliers),
            "correspondence_count": len(points_3d),
        }

    def _match_temporal_roma(
            self,
            current_left: np.ndarray,
            current_cache: Optional[dict]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.feature_reuse_active and current_cache is not None:
            return self.matcher.match_cached_pair(
                self._kf_img,
                current_left,
                self._kf_roma_cache,
                current_cache,
                num_matches=self.temporal_matches,
                resize=self.match_resize,
                upsample_refinement=self.temporal_upsample_refinement,
            )
        return self.matcher.match(
            self._kf_img,
            current_left,
            num_matches=self.temporal_matches,
            resize=self.match_resize,
            upsample_refinement=self.temporal_upsample_refinement,
            symmetric_matching=self.symmetric_matching,
        )

    def process_frame(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # 后台线程只产生不可变结果；所有 SLAM 状态变更统一在前端线程提交。
        self._geometry_corrected_in_process = False
        self._apply_pending_loop_results()
        self.frame_count += 1
        self._last_feature_match_data = None
        self._set_tracking_status(
            success=False,
            reason="processing",
        )

        # ── Step 1: 双目匹配 + 三角化 ──
        temporal_result = None
        current_roma_cache = None
        klt_result = self._track_keyframe_with_klt(left)
        roma_temporal_required = klt_result is None
        if self.feature_reuse_active:
            stereo_result, temporal_result, current_roma_cache = (
                self.matcher.match_frame_pairs(
                    left,
                    right,
                    keyframe=(
                        self._kf_img if roma_temporal_required else None
                    ),
                    keyframe_cache=self._kf_roma_cache,
                    stereo_matches=self.stereo_matches,
                    temporal_matches=self.temporal_matches,
                    resize=self.match_resize,
                    stereo_refinement=self.stereo_upsample_refinement,
                    temporal_refinement=self.temporal_upsample_refinement,
                    symmetric_matching=self.symmetric_matching,
                    batch_coarse_pairs=self.batch_coarse_pairs,
                )
            )
            kp_l, kp_r, conf_stereo = stereo_result
        else:
            kp_l, kp_r, conf_stereo = self.matcher.match(
                left, right,
                num_matches=self.stereo_matches,
                resize=self.match_resize,
                upsample_refinement=self.stereo_upsample_refinement,
                symmetric_matching=self.symmetric_matching)

        # 置信度过滤
        valid_conf = conf_stereo > 0.5
        kp_l, kp_r = kp_l[valid_conf], kp_r[valid_conf]

        pts3d_cam, tri_idx = self.triangulator.triangulate(kp_l, kp_r)
        kp_l_valid = kp_l[tri_idx]
        kp_r_valid = kp_r[tri_idx]
        self._last_feature_match_data = {
            "frame_index": int(self.frame_count),
            "stereo_left_points": kp_l_valid,
            "stereo_right_points": kp_r_valid,
            "temporal_reference_image": None,
            "temporal_reference_points": np.empty((0, 2), dtype=np.float32),
            "temporal_current_points": np.empty((0, 2), dtype=np.float32),
            "temporal_inlier_mask": np.empty(0, dtype=bool),
            "temporal_source": "none",
            "temporal_pose_success": False,
            "tracking_success": False,
        }

        if len(pts3d_cam) < 10:
            print(f"[W] 帧{self.frame_count}: 有效3D点不足({len(pts3d_cam)})")
            return self._handle_tracking_failure(
                reason="insufficient_stereo_points",
                left=left,
                pts3d_cam=pts3d_cam,
                kp_l_valid=kp_l_valid,
                kp_r_valid=kp_r_valid,
                correspondence_count=len(pts3d_cam),
                allow_relocalization=False,
            )

        # ── Step 2: 第一帧初始化 ──
        if self._kf_img is None:
            self._init_first_kf(
                left,
                pts3d_cam,
                kp_l_valid,
                kp_r_valid,
                roma_cache=current_roma_cache,
            )
            first_keyframe = getattr(self, "_last_kf", None)
            first_keyframe_id = (
                first_keyframe.id if first_keyframe is not None else None)
            self.tracking_state = "TRACKING"
            self._set_tracking_status(
                success=True,
                reason="initialized",
                pose_updated=True,
                initialized=True,
                correspondence_count=len(pts3d_cam),
                keyframe_created=True,
                keyframe_id=first_keyframe_id,
            )
            self._mark_feature_match_tracking_success(True)
            return self.poses[-1]

        # ── Step 3: 帧间匹配（关键帧 ↔ 当前帧）──
        if klt_result is not None:
            mkp_kf, mkp_curr, _ = klt_result
            temporal_source = "klt"
        elif self.feature_reuse_active:
            if temporal_result is None:
                raise RuntimeError(
                    "RoMa feature reuse did not return temporal matches"
                )
            mkp_kf, mkp_curr, _ = temporal_result
            temporal_source = "roma"
        else:
            mkp_kf, mkp_curr, _ = self.matcher.match(
                self._kf_img, left,
                num_matches=self.temporal_matches,
                resize=self.match_resize,
                upsample_refinement=self.temporal_upsample_refinement,
                symmetric_matching=self.symmetric_matching)
            temporal_source = "roma"

        debug_reference_image = self._kf_img
        pose_solution, pose_failure = self._estimate_temporal_pose(
            mkp_kf, mkp_curr
        )
        if pose_solution is not None and temporal_source == "klt":
            klt_correspondences = len(pose_solution["points_3d"])
            klt_inliers = len(pose_solution["inliers"])
            klt_inlier_ratio = klt_inliers / max(klt_correspondences, 1)
            if (klt_inliers < self.klt_min_tracks
                    or klt_inlier_ratio < self.klt_min_pnp_inlier_ratio):
                pose_solution = None
                pose_failure = {
                    "reason": "klt_pnp_quality_rejected",
                    "inlier_count": klt_inliers,
                    "correspondence_count": klt_correspondences,
                }
        if pose_solution is None and temporal_source == "klt":
            self.temporal_roma_fallbacks += 1
            mkp_kf, mkp_curr, _ = self._match_temporal_roma(
                left, current_roma_cache
            )
            temporal_source = "roma_fallback"
            pose_solution, pose_failure = self._estimate_temporal_pose(
                mkp_kf, mkp_curr
            )

        if pose_solution is None:
            self._record_temporal_feature_matches(
                reference_image=debug_reference_image,
                reference_points=mkp_kf,
                current_points=mkp_curr,
                inlier_indices=None,
                source=temporal_source,
                pose_success=False,
            )
            reason = str(pose_failure["reason"])
            if reason == "insufficient_temporal_matches":
                print(
                    f"[W] 帧{self.frame_count}: "
                    f"帧间匹配不足({pose_failure['correspondence_count']})"
                )
            elif reason == "insufficient_3d_2d_correspondences":
                print(
                    f"[W] 帧{self.frame_count}: 3D-2D 关联不足"
                    f"({pose_failure['correspondence_count']})"
                )
            else:
                print(f"[W] 帧{self.frame_count}: PnP 失败")
            return self._handle_tracking_failure(
                reason=reason,
                left=left,
                pts3d_cam=pts3d_cam,
                kp_l_valid=kp_l_valid,
                kp_r_valid=kp_r_valid,
                inlier_count=int(pose_failure["inlier_count"]),
                correspondence_count=int(
                    pose_failure["correspondence_count"]
                ),
            )

        if temporal_source == "klt":
            self.temporal_klt_successes += 1
        self.last_temporal_tracking_source = temporal_source
        rvec = pose_solution["rotation_vector"]
        tvec = pose_solution["translation_vector"]
        inliers = pose_solution["inliers"]
        pts3d_match = pose_solution["points_3d"]
        kf_idx_match = pose_solution["keyframe_indices"]
        good = pose_solution["good"]
        self._record_temporal_feature_matches(
            reference_image=debug_reference_image,
            reference_points=self._kf_kpts[kf_idx_match],
            current_points=pose_solution["points_2d"],
            inlier_indices=inliers,
            source=temporal_source,
            pose_success=True,
        )

        R, _ = cv2.Rodrigues(rvec)
        T_ck = np.eye(4); T_ck[:3,:3]=R; T_ck[:3,3]=tvec.flatten()
        T_wk = self._last_kf.pose.copy() if self._last_kf else self.poses[-1]
        T_wc = T_wk @ np.linalg.inv(T_ck)

        # 运动合理性检查（防漂移）
        dt = np.linalg.norm(T_ck[:3,3])
        if dt > 50.0:
            print(f"[W] 帧{self.frame_count}: 位移过大({dt:.2f}m)，跳过")
            return self._handle_tracking_failure(
                reason="implausible_translation",
                left=left,
                pts3d_cam=pts3d_cam,
                kp_l_valid=kp_l_valid,
                kp_r_valid=kp_r_valid,
                inlier_count=len(inliers),
                correspondence_count=len(pts3d_match),
            )

        self.poses.append(T_wc)
        reference_id = self._last_kf.id if self._last_kf is not None else None
        self._pose_reference_kf_ids.append(reference_id)
        self._pose_relative_to_kf.append(np.linalg.inv(T_ck))
        print(
            f"  Frame {self.frame_count}: inliers={len(inliers)} "
            f"t={dt:.3f}m temporal={temporal_source}"
        )

        # ── Step 6: 关键帧决策 ──
        created_keyframe = None
        if self._need_keyframe(T_ck, len(inliers)):
            R_wc, t_wc = T_wc[:3,:3], T_wc[:3,3]
            pts3d_world = (R_wc @ pts3d_cam.T).T + t_wc
            assoc = self.associator.associate(
                curr_kpts=kp_l_valid,
                curr_pose=T_wc,
                matched_kf_kpts=mkp_curr[good],
                matched_kf_indices=kf_idx_match,
                kf_mp_ids=self._kf_mp_ids or [],
                global_mappoints=self.global_mps)
            kf = self._create_keyframe(left, pts3d_cam, pts3d_world,
                                       kp_l_valid, kp_r_valid, T_wc, assoc)
            created_keyframe = kf
            current_pose_index = len(self.poses) - 1
            self._kf_id_to_pose_idx[kf.id] = current_pose_index
            self._pose_reference_kf_ids[current_pose_index] = kf.id
            self._pose_relative_to_kf[current_pose_index] = np.eye(4)
            self._update_kf_cache(
                left,
                pts3d_cam,
                kp_l_valid,
                kf,
                roma_cache=current_roma_cache,
            )
            T_wc = self._submit_keyframe_to_backend(kf)

        self.tracking_state = "TRACKING"
        self._set_tracking_status(
            success=True,
            reason="tracking_success",
            pose_updated=True,
            inlier_count=len(inliers),
            correspondence_count=len(pts3d_match),
            keyframe_created=created_keyframe is not None,
            keyframe_id=(
                created_keyframe.id if created_keyframe is not None else None),
        )
        self._mark_feature_match_tracking_success(True)
        return T_wc

    # ──────────────────────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────────────────────
    def _need_keyframe(self, T_ck: np.ndarray, n_inliers: int) -> bool:
        frames_since_keyframe = self.frame_count - self._last_kf_frame_idx
        if frames_since_keyframe < self.keyframe_min_gap:
            return False
        if self.keyframe_interval_from_last:
            interval_reached = frames_since_keyframe >= self.keyframe_interval
        else:
            interval_reached = self.frame_count % self.keyframe_interval == 0
        if interval_reached:
            return True
        t = np.linalg.norm(T_ck[:3,3])
        ang = np.arccos(np.clip((np.trace(T_ck[:3,:3])-1)/2, -1, 1))
        return (
            t > 0.1
            or ang > 0.1
            or n_inliers < self.keyframe_min_inliers
        )

    def _create_keyframe(self, img: np.ndarray,
                         pts3d_cam: np.ndarray, pts3d_world: np.ndarray,
                         kpts: np.ndarray, right_kpts: np.ndarray,
                         pose: np.ndarray,
                         assoc: Dict[int, int],
                         descriptors: Optional[np.ndarray] = None) -> KeyFrame:
        """Stage a complete keyframe, then commit it to all live stores."""
        count = len(kpts)
        if (len(pts3d_cam) != count
                or len(pts3d_world) != count
                or len(right_kpts) != count):
            raise ValueError("keyframe stereo arrays are not index-aligned")
        if descriptors is None:
            descs = self.desc_extractor.extract_at_points(img, kpts)
        else:
            descs = np.asarray(descriptors, dtype=np.uint8).copy()
        if descs.shape != (count, 32):
            raise ValueError("keyframe ORB descriptors must have shape (N, 32)")

        kf = KeyFrame(
            id=self.next_kf_id, timestamp=time.time(), pose=pose.copy(),
            image=img.copy(),
            keypoints=[k.tolist() for k in kpts],
            right_keypoints=[k.tolist() for k in right_kpts],
            camera_points=[p.tolist() for p in pts3d_cam],
            descriptors=descs, K=self.K)

        # Prepare all point associations without mutating the live map.  This
        # is important for relocalization: a rejected recovery must be a true
        # zero-write operation.
        mp_ids, n_new, n_reuse = [], 0, 0
        staged_new_mps: Dict[int, MapPoint] = {}
        staged_reused_observations = []
        next_mp_id = self.next_mp_id
        reused_mp_ids = set()
        for i, (p3w, p2d) in enumerate(zip(pts3d_world, kpts)):
            associated_id = assoc.get(i)
            can_reuse = (
                associated_id is not None
                and associated_id in self.global_mps
                and associated_id not in reused_mp_ids
                and not self.global_mps[associated_id].is_bad)
            if can_reuse:
                mp = self.global_mps[associated_id]
                kf.add_mappoint(i, mp)
                mp_ids.append(mp.id)
                staged_reused_observations.append(
                    (mp, np.asarray(p2d).copy()))
                reused_mp_ids.add(mp.id)
                n_reuse += 1
            else:
                mp = MapPoint(id=next_mp_id, position=p3w.copy())
                mp.descriptor = descs[i].copy()
                x, y = int(p2d[0]), int(p2d[1])
                if 0<=x<img.shape[1] and 0<=y<img.shape[0]:
                    mp.color = img[y,x,::-1].astype(np.uint8)
                mp.add_observation(kf.id, np.array(p2d))
                kf.add_mappoint(i, mp)
                staged_new_mps[next_mp_id] = mp
                mp_ids.append(next_mp_id)
                next_mp_id += 1
                n_new += 1

        relocalizer = getattr(self, "relocalizer", None)
        relocalization_snapshot = None
        if relocalizer is not None:
            relocalization_snapshot = make_keyframe_geometry_snapshot(
                kf, self.K)

        # Commit only after descriptors, map points and the detached geometry
        # snapshot have all been built successfully.
        for mp, observation in staged_reused_observations:
            mp.add_observation(kf.id, observation)
        self.global_mps.update(staged_new_mps)
        self.keyframes.append(kf)
        self.next_mp_id = next_mp_id
        self.kf_count += 1
        self.next_kf_id += 1
        self._last_kf = kf
        self._kf_mp_ids = mp_ids
        self._last_kf_frame_idx = self.frame_count
        if relocalization_snapshot is not None:
            relocalizer.add_snapshot(relocalization_snapshot)
        self.trim_keyframe_images(int(getattr(self, "keyframe_image_cache_limit", 0)))
        if n_reuse: print(f"  [KF{kf.id}] 复用{n_reuse} 新建{n_new}")
        return kf

    def trim_keyframe_images(self, keep_last: int) -> int:
        """Drop only redundant historical pixels, retaining all geometry/features.

        Tracking uses the separate current-keyframe caches. Relocalization
        stores detached descriptors/geometry, not images. Default 0 preserves
        historical behavior for callers that explicitly need every RGB frame.
        """
        if keep_last <= 0:
            return 0
        released = 0
        for keyframe in self.keyframes[:-keep_last]:
            if keyframe.image is not None:
                keyframe.image_size_hint = tuple(int(x) for x in keyframe.image.shape[:2])
                keyframe.image = None
                released += 1
        return released

    def _init_first_kf(self, img: np.ndarray, pts3d: np.ndarray,
                       kpts: np.ndarray, right_kpts: np.ndarray,
                       roma_cache: Optional[dict] = None):
        pose = self.poses[-1]
        pts3d_world = (pose[:3,:3] @ pts3d.T).T + pose[:3,3]
        kf = self._create_keyframe(
            img, pts3d, pts3d_world, kpts, right_kpts, pose, {})
        first_pose_index = len(self.poses) - 1
        self._kf_id_to_pose_idx[kf.id] = first_pose_index
        self._pose_reference_kf_ids[first_pose_index] = kf.id
        self._pose_relative_to_kf[first_pose_index] = np.eye(4)
        self._update_kf_cache(
            img, pts3d, kpts, kf, roma_cache=roma_cache
        )
        self._submit_keyframe_to_backend(kf)

    def _update_kf_cache(self, img: np.ndarray, pts3d: np.ndarray,
                         kpts: np.ndarray, kf: KeyFrame,
                         roma_cache: Optional[dict] = None):
        self._kf_img   = img.copy()
        self._kf_pts3d = pts3d.copy()
        self._kf_kpts  = kpts.copy()
        keyframe_gray = (
            cv2.cvtColor(self._kf_img, cv2.COLOR_BGR2GRAY)
            if self._kf_img.ndim == 3
            else np.asarray(self._kf_img, dtype=np.uint8).copy()
        )
        self._kf_gray = cv2.resize(
            keyframe_gray,
            None,
            fx=self.klt_image_scale,
            fy=self.klt_image_scale,
            interpolation=cv2.INTER_AREA,
        )
        self._kf_klt_indices = self._build_klt_indices(self._kf_kpts)
        self._kf_tree = cKDTree(self._kf_kpts)
        self._kf_roma_cache = self.matcher.freeze_feature_cache(
            roma_cache
        )

    def _submit_keyframe_to_backend(self, kf: KeyFrame) -> np.ndarray:
        """Run the unchanged optional backend path for a committed keyframe."""
        if self.enable_backend and self.sw_ba is not None:
            try:
                self.sw_ba.add_keyframe(kf)
                n_kfs = len(self.sw_ba.keyframes)
                if self.kf_count % self.ba_frequency == 0 and n_kfs >= 3:
                    print(f"[Backend] SW-BA ({n_kfs} KFs)...")
                    err, n_in = self.sw_ba.optimize(iterations=15)
                    print(f"[Backend] BA err={err:.4f} inliers={n_in}")
                    if self.sw_ba.last_accepted:
                        self._sync_ba_poses()
                        self._mark_geometry_corrected()
                    else:
                        print(
                            "[Backend] BA 未提交: "
                            f"{self.sw_ba.last_failure_reason}")
            except Exception as exc:
                print(f"[Backend] BA 异常: {exc}")
            if self.backend:
                try:
                    relocalizer = getattr(self, "relocalizer", None)
                    self.backend.add_keyframe(
                        kf,
                        base_snapshot=(
                            relocalizer.get_snapshot(kf.id)
                            if relocalizer is not None else None))
                except Exception as exc:
                    print(f"[Backend] 关键帧入队异常: {exc}")
        return kf.pose.copy()

    def _sync_pose_history_from_keyframes(self):
        """Rebuild every tracked-frame pose from its reference keyframe."""
        if (not hasattr(self, "_pose_reference_kf_ids")
                or not hasattr(self, "_pose_relative_to_kf")):
            return
        keyframe_poses = {
            kf.id: kf.pose for kf in self.keyframes if not kf.is_bad
        }
        count = min(
            len(self.poses),
            len(self._pose_reference_kf_ids),
            len(self._pose_relative_to_kf))
        for index in range(count):
            reference_id = self._pose_reference_kf_ids[index]
            if reference_id in keyframe_poses:
                self.poses[index] = (
                    keyframe_poses[reference_id]
                    @ self._pose_relative_to_kf[index])

    def _apply_pending_loop_results(self) -> bool:
        """Commit worker results on the SLAM/front-end thread only."""
        backend = getattr(self, "backend", None)
        if backend is None:
            return False
        results = backend.consume_loop_results()
        if not results:
            return False

        committed_any = False
        for result in results:
            raw_corrections = {
                int(kid): np.asarray(correction, dtype=np.float64).copy()
                for kid, correction in result.corrections.items()
                if np.asarray(correction).shape == (4, 4)
                and np.all(np.isfinite(correction))
            }
            if not raw_corrections:
                continue

            # If the front end has advanced while the worker was optimizing,
            # propagate the newest available correction rigidly to that tail.
            correction_ids = sorted(raw_corrections)
            effective_corrections = {}
            for kf in sorted(self.keyframes, key=lambda item: item.id):
                eligible = [kid for kid in correction_ids if kid <= kf.id]
                if eligible:
                    effective_corrections[kf.id] = raw_corrections[eligible[-1]]
            if not effective_corrections:
                continue

            # Sparse map points follow their first still-known observing
            # keyframe. Each point is transformed exactly once per result.
            # 按参考关键帧分组后批量变换：地图点可达数十万个，逐点做
            # 3x3 矩阵乘会在跟踪线程上停顿数秒。
            correction_groups: Dict[int, List[MapPoint]] = {}
            for mp in self.global_mps.values():
                reference_id = next(
                    (kid for kid in mp.observations
                     if kid in effective_corrections),
                    None)
                if reference_id is not None:
                    correction_groups.setdefault(reference_id, []).append(mp)
            for reference_id, points in correction_groups.items():
                correction = effective_corrections[reference_id]
                positions = np.asarray(
                    [mp.position for mp in points], dtype=np.float64)
                corrected = (
                    positions @ correction[:3, :3].T + correction[:3, 3])
                for mp, position in zip(points, corrected):
                    mp.position = position.copy()

            for kf in self.keyframes:
                correction = effective_corrections.get(kf.id)
                if correction is None:
                    continue
                corrected_pose = correction @ kf.pose
                corrected_pose[3] = np.array([0.0, 0.0, 0.0, 1.0])
                kf.pose = corrected_pose

            constraint = result.constraint
            current = next(
                (kf for kf in self.keyframes
                 if kf.id == constraint.current_id),
                None)
            if current is not None:
                current.loop_edges.append((
                    constraint.loop_id,
                    constraint.T_loop_current.copy()))
            self.loop_count += 1
            committed_any = True
            print(
                f"[SLAM] 回环 #{self.loop_count}: "
                f"KF{constraint.current_id} → KF{constraint.loop_id}, "
                f"inliers={constraint.inliers}, "
                f"rmse={constraint.reprojection_rmse:.2f}px")

        if committed_any:
            self._sync_pose_history_from_keyframes()
            self._refresh_last_kf_cache_from_map()
            self._mark_geometry_corrected()
        return committed_any

    def _sync_ba_poses(self):
        if not self.sw_ba: return
        self._sync_pose_history_from_keyframes()
        self._refresh_last_kf_cache_from_map()

    def _refresh_last_kf_cache_from_map(self):
        """让下一帧 PnP 使用与 BA 后位姿/地图一致的关键帧 3D 点。"""
        if self._last_kf is None or self._kf_pts3d is None:
            return
        pose = self._last_kf.pose
        R_cw = pose[:3, :3].T
        t_wc = pose[:3, 3]
        limit = min(len(self._kf_pts3d), len(self._last_kf.mappoints))
        for index in range(limit):
            mp = self._last_kf.mappoints[index]
            if mp is None or mp.is_bad:
                continue
            point_camera = R_cw @ (mp.position - t_wc)
            if np.all(np.isfinite(point_camera)) and point_camera[2] > 0.1:
                self._kf_pts3d[index] = point_camera

    # ──────────────────────────────────────────────────────────
    #  最终全局 BA
    # ──────────────────────────────────────────────────────────
    def final_optimization(self, iterations: int = 30):
        if not self.enable_backend or len(self.keyframes) < 3:
            return
        # Final BA must not race with a still-running pose-graph worker.
        if self.backend:
            self.backend.stop()
            self._apply_pending_loop_results()
        print(f"\n[Backend] 全局 BA ({len(self.keyframes)} KFs)...")
        gba = SlidingWindowBA(
            window_size=len(self.keyframes)+5,
            K=self.K,
            baseline=self.baseline)
        for kf in self.keyframes:
            gba.keyframes.append(kf)
            for mp in kf.mappoints:
                if mp and not mp.is_bad and mp.id not in gba.mappoints:
                    gba.mappoints[mp.id] = mp
        try:
            err, n_in = gba.optimize(iterations=iterations)
            print(f"[Backend] 全局 BA: err={err:.4f} inliers={n_in}")
            if not gba.last_accepted:
                print(f"[Backend] 全局 BA 未提交: {gba.last_failure_reason}")
                return
            self._sync_pose_history_from_keyframes()
            self._refresh_last_kf_cache_from_map()
            self._mark_geometry_corrected()
        except Exception as e:
            print(f"[Backend] 全局 BA 异常: {e}")

    def shutdown(self):
        if self.backend:
            self.backend.stop()
            self._apply_pending_loop_results()

    # ──────────────────────────────────────────────────────────
    #  结果访问
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _readonly_copy(array: np.ndarray, *, dtype=None) -> np.ndarray:
        result = np.asarray(array, dtype=dtype).copy()
        result.setflags(write=False)
        return result

    def get_mapping_keyframe_snapshot(
            self, keyframe_id: int) -> Optional[MappingKeyFrameSnapshot]:
        """Return copied raw observations for one keyframe.

        The arrays do not alias live front-end state and remain unchanged when
        BA or loop closing later updates the keyframe pose.
        """
        keyframe = next(
            (item for item in self.keyframes if item.id == int(keyframe_id)),
            None)
        if keyframe is None:
            return None

        count = min(len(keyframe.keypoints), len(keyframe.camera_points))
        keypoints = np.full((count, 2), np.nan, dtype=np.float64)
        camera_points = np.full((count, 3), np.nan, dtype=np.float64)
        for index in range(count):
            try:
                keypoint = np.asarray(
                    keyframe.keypoints[index], dtype=np.float64).reshape(-1)
                camera_point = np.asarray(
                    keyframe.camera_points[index], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if keypoint.size >= 2:
                keypoints[index] = keypoint[:2]
            if camera_point.size >= 3:
                camera_points[index] = camera_point[:3]

        valid = (
            np.all(np.isfinite(keypoints), axis=1)
            & np.all(np.isfinite(camera_points), axis=1)
            & (camera_points[:, 2] > 0.1)
        )
        return MappingKeyFrameSnapshot(
            id=int(keyframe.id),
            timestamp=float(keyframe.timestamp),
            pose=self._readonly_copy(keyframe.pose, dtype=np.float64),
            keypoints=self._readonly_copy(keypoints[valid], dtype=np.float64),
            camera_points=self._readonly_copy(
                camera_points[valid], dtype=np.float64),
            is_bad=bool(keyframe.is_bad),
        )

    def get_keyframe_pose_snapshot(
            self, include_bad: bool = False
            ) -> Tuple[int, Dict[int, np.ndarray]]:
        """Return geometry revision and detached current keyframe poses."""
        poses: Dict[int, np.ndarray] = {}
        for keyframe in self.keyframes:
            if keyframe.is_bad and not include_bad:
                continue
            poses[int(keyframe.id)] = self._readonly_copy(
                keyframe.pose, dtype=np.float64)
        return int(getattr(self, "geometry_revision", 0)), poses

    def get_pose_history_snapshot(self) -> Tuple[int, Tuple[np.ndarray, ...]]:
        """Return detached corrected poses indexed like ``self.poses``."""
        poses = tuple(
            self._readonly_copy(pose, dtype=np.float64)
            for pose in self.poses)
        return int(getattr(self, "geometry_revision", 0)), poses

    def get_map_points(self, include_bad: bool = False,
                       min_obs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        pts, cols = [], []
        for mp in self.global_mps.values():
            if not include_bad and mp.is_bad: continue
            if mp.n_found < min_obs: continue
            pts.append(mp.position)
            cols.append(mp.color if mp.color is not None else np.zeros(3))
        if not pts:
            return np.zeros((0,3)), np.zeros((0,3))
        return np.array(pts), np.array(cols)

    def get_dense_map_points(self, max_reproj: float = 10.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
        pts, cols = [], []
        for mp in self.global_mps.values():
            if mp.position is None: continue
            if not mp.observations:
                pts.append(mp.position)
                cols.append(mp.color if mp.color is not None else np.zeros(3))
                continue
            errs, cnt = 0., 0
            for kid, p2d in mp.observations.items():
                kf = next((k for k in self.keyframes if k.id==kid), None)
                if kf is None: continue
                R,t = kf.pose[:3,:3], kf.pose[:3,3]
                pc = R.T@(mp.position-t)
                if pc[2]<=0.1: continue
                iz=1./pc[2]
                u=self.fx*pc[0]*iz+self.cx; v=self.fy*pc[1]*iz+self.cy
                errs += np.sqrt((u-p2d[0])**2+(v-p2d[1])**2); cnt+=1
            if cnt==0 or errs/cnt<=max_reproj:
                pts.append(mp.position)
                cols.append(mp.color if mp.color is not None else np.zeros(3))
        if not pts:
            return np.zeros((0,3)), np.zeros((0,3))
        return np.array(pts), np.array(cols)


# ══════════════════════════════════════════════════════════════
#  I/O 辅助
# ══════════════════════════════════════════════════════════════

def save_trajectory_tum(poses: List[np.ndarray], path: str,
                        timestamps: Optional[List[float]] = None):
    with open(path, 'w') as f:
        for i, T in enumerate(poses):
            ts = timestamps[i] if timestamps else i*0.1
            t = T[:3,3]
            q = SciRot.from_matrix(T[:3,:3]).as_quat()
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def save_trajectory_kitti(poses: List[np.ndarray], path: str):
    with open(path, 'w') as f:
        for T in poses:
            f.write(' '.join(f"{x:.6e}" for x in T[:3,:4].flatten())+'\n')


def save_ply(points: np.ndarray, colors: np.ndarray, path: str):
    n = len(points)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for p, c in zip(points, colors):
            r,g,b = np.clip(c,0,255).astype(int)
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")


def plot_trajectory(poses: List[np.ndarray], path: str, loop_count: int = 0):
    import matplotlib.pyplot as plt
    pos = np.array([T[:3,3] for T in poses])
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    planes = [('X','Y',0,1),('X','Z',0,2),('Y','Z',1,2)]
    for ax, (xl, yl, xi, yi) in zip(axes, planes):
        ax.plot(pos[:,xi], pos[:,yi], 'b-', lw=1)
        ax.scatter(pos[0,xi], pos[0,yi], c='g', s=100, zorder=5, label='Start')
        ax.scatter(pos[-1,xi], pos[-1,yi], c='r', s=100, marker='x', zorder=5, label='End')
        ax.set_xlabel(f'{xl} (m)'); ax.set_ylabel(f'{yl} (m)')
        ax.set_title(f'{xl}{yl} 平面' + (f' (回环:{loop_count})' if loop_count else ''))
        ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[Save] 轨迹图: {path}")


# ══════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="RoMa Stereo SLAM v2")
    ap.add_argument("--data_dir",  default="data/data1223")
    ap.add_argument("--calib",     default=None)
    ap.add_argument("--output_dir",default="output_v2")
    ap.add_argument("--max_frames",type=int, default=None)
    ap.add_argument("--device",    default="cuda")
    # RoMa 参数
    ap.add_argument("--coarse_res",  type=int, default=672)
    ap.add_argument("--upsample_res",type=int, default=1024)
    ap.add_argument("--stereo_matches",  type=int, default=12000)
    ap.add_argument("--temporal_matches",type=int, default=6000)
    ap.add_argument("--match_resize", type=int, nargs=2, default=None,
                    help="RoMa 内部缩放 (W H)，加速但略降精度")
    # 深度
    ap.add_argument("--min_depth", type=float, default=0.5)
    ap.add_argument("--max_depth", type=float, default=200.0)
    ap.add_argument("--epipolar_threshold", type=float, default=2.0)
    # 关键帧
    ap.add_argument("--kf_interval", type=int, default=5)
    ap.add_argument("--keyframe_min_inliers", type=int, default=50)
    ap.add_argument("--keyframe_min_gap", type=int, default=1)
    ap.add_argument("--keyframe_interval_from_last", action="store_true")
    ap.add_argument(
        "--relocalization_min_interval", type=int, default=1,
        help="失败帧两次重定位全库检索的最小帧间隔；1 表示逐帧尝试")
    # 后端
    ap.add_argument("--enable_backend",      action="store_true")
    ap.add_argument("--enable_loop_closing", action="store_true")
    ap.add_argument("--window_size",  type=int,   default=10)
    ap.add_argument("--ba_frequency", type=int,   default=3)
    ap.add_argument(
        "--loop_dist", type=float, default=3.0,
        help="几何验证后允许的最大回环相对平移(m)，不用于候选检索")
    ap.add_argument("--final_ba",     action="store_true")
    # IMU
    ap.add_argument("--imu_data",       default=None)
    ap.add_argument("--imu_timestamps", default=None)
    ap.add_argument("--camera_pitch",   type=float, default=30.0)
    ap.add_argument("--lunar_gravity",  type=float, default=1.626)
    # 关联
    ap.add_argument("--assoc_pixel_thresh", type=float, default=5.0)
    ap.add_argument("--assoc_reproj_thresh",type=float, default=3.0)
    # 点云
    ap.add_argument("--save_all_points",   action="store_true")
    ap.add_argument("--save_dense_points", action="store_true")
    ap.add_argument("--dense_reproj_thresh",type=float,default=10.0)
    args = ap.parse_args()

    print(f"[Info] g2o 可用: {is_g2o_available()}")
    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = Path(args.data_dir)
    left_dir  = data_dir / 'left'
    right_dir = data_dir / 'right'
    calib_path = Path(args.calib) if args.calib else data_dir / 'calib.yaml'

    # 加载图像列表
    exts = ('*.png','*.jpg','*.PNG','*.JPG')
    left_imgs  = sorted(p for e in exts for p in left_dir.glob(e))
    right_imgs = sorted(p for e in exts for p in right_dir.glob(e))
    assert len(left_imgs)==len(right_imgs), "左右图像数量不匹配"
    if args.max_frames:
        left_imgs = left_imgs[:args.max_frames]
        right_imgs = right_imgs[:args.max_frames]
    print(f"[Data] {len(left_imgs)} 帧")

    # 加载标定
    with open(calib_path) as f:
        calib = yaml.safe_load(f)

    # IMU
    imu_helper = None
    if args.imu_data and args.imu_timestamps:
        try:
            imu_helper = IMUHelper(args.imu_data, args.imu_timestamps,
                                   args.camera_pitch, args.lunar_gravity)
        except Exception as e:
            print(f"[W] IMU 初始化失败: {e}")

    match_resize = tuple(args.match_resize) if args.match_resize else None

    slam = StereoSLAMv2(
        calib=calib,
        device=args.device if torch.cuda.is_available() else "cpu",
        coarse_res=args.coarse_res,
        upsample_res=args.upsample_res,
        stereo_matches=args.stereo_matches,
        temporal_matches=args.temporal_matches,
        match_resize=match_resize,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        epipolar_threshold=args.epipolar_threshold,
        keyframe_interval=args.kf_interval,
        keyframe_min_inliers=args.keyframe_min_inliers,
        keyframe_min_gap=args.keyframe_min_gap,
        keyframe_interval_from_last=args.keyframe_interval_from_last,
        relocalization_min_interval=args.relocalization_min_interval,
        enable_backend=args.enable_backend,
        enable_loop_closing=args.enable_loop_closing,
        backend_window_size=args.window_size,
        ba_frequency=args.ba_frequency,
        loop_dist_thresh=args.loop_dist,
        imu_helper=imu_helper,
        assoc_pixel_thresh=args.assoc_pixel_thresh,
        assoc_reproj_thresh=args.assoc_reproj_thresh,
    )

    times = []
    try:
        for lp, rp in tqdm(zip(left_imgs, right_imgs), total=len(left_imgs)):
            left  = cv2.imread(str(lp))
            right = cv2.imread(str(rp))
            if left is None or right is None:
                print(f"[W] 无法读取 {lp.name}")
                continue
            t0 = time.time()
            slam.process_frame(left, right)
            times.append(time.time()-t0)
    finally:
        if args.enable_backend and args.final_ba:
            slam.final_optimization(iterations=30)
        slam.shutdown()

    # ── 统计 ──
    print(f"\n[Stats] 帧数={len(slam.poses)} KFs={slam.kf_count} "
          f"MPs={len(slam.global_mps)} 回环={slam.loop_count}")
    n_bad = sum(1 for m in slam.global_mps.values() if m.is_bad)
    print(f"[Stats] bad MPs={n_bad}  avg={np.mean(times)*1e3:.1f}ms/帧")

    out = Path(args.output_dir)
    save_trajectory_tum(slam.poses,    out/"trajectory.txt")
    save_trajectory_kitti(slam.poses,  out/"trajectory_kitti.txt")

    pts, cols = slam.get_map_points()
    if len(pts): save_ply(pts, cols, out/"map_points.ply")
    print(f"[Save] 优化点云: {len(pts)} pts → {out/'map_points.ply'}")

    if args.save_all_points:
        pts_a, cols_a = slam.get_map_points(include_bad=True)
        if len(pts_a): save_ply(pts_a, cols_a, out/"map_points_all.ply")
        print(f"[Save] 完整点云: {len(pts_a)} pts")

    if args.save_dense_points:
        pts_d, cols_d = slam.get_dense_map_points(args.dense_reproj_thresh)
        if len(pts_d): save_ply(pts_d, cols_d, out/"map_points_dense.ply")
        print(f"[Save] 稠密点云: {len(pts_d)} pts")

    plot_trajectory(slam.poses, out/"trajectory.png", slam.loop_count)
    print("\n[Done]")


if __name__ == "__main__":
    main()
