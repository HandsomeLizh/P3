"""Low-rate LiDAR loop candidates and conservative metric verification.

The implementation is deliberately dependency-light (NumPy/SciPy only) so
the safe baseline works in the main ROS environment and on embedded targets.
Scan Context is used only to shortlist old places.  A candidate must then pass
trimmed bidirectional ICP, overlap, uniqueness, correction-size and temporal
confirmation gates before it can become a factor.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

from .loop_constraints import LoopCandidate, LoopConstraint
from .pose_math import (
    inverse_pose,
    pose_change_norm,
    relative_pose,
    validated_pose_matrix,
)


@dataclass(frozen=True)
class LidarLoopConfiguration:
    enabled: bool = False
    submap_keyframes: int = 3
    voxel_size_m: float = 0.25
    max_points: int = 5000
    descriptor_rings: int = 20
    descriptor_sectors: int = 60
    descriptor_max_radius_m: float = 35.0
    descriptor_shortlist: int = 12
    descriptor_min_similarity: float = 0.35
    min_keyframe_separation: int = 20
    min_time_separation_sec: float = 20.0
    max_candidates: int = 3
    icp_max_correspondence_m: float = 1.0
    icp_max_iterations: int = 30
    icp_trim_fraction: float = 0.70
    icp_min_inliers: int = 120
    icp_min_overlap: float = 0.25
    icp_min_unique_target_ratio: float = 0.50
    icp_max_rmse_m: float = 0.30
    icp_max_cycle_translation_m: float = 0.20
    icp_max_cycle_rotation_deg: float = 3.0
    max_correction_translation_m: float = 4.0
    max_correction_rotation_deg: float = 25.0
    confirmation_count: int = 2
    confirmation_max_query_gap: int = 2
    confirmation_max_reference_gap: int = 3
    confirmation_max_translation_m: float = 1.0
    confirmation_max_rotation_deg: float = 10.0
    archive_directory: Optional[Path] = None
    geometry_cache_records: int = 16

    def __post_init__(self) -> None:
        integer_positive = (
            "submap_keyframes",
            "max_points",
            "descriptor_rings",
            "descriptor_sectors",
            "descriptor_shortlist",
            "min_keyframe_separation",
            "max_candidates",
            "icp_max_iterations",
            "icp_min_inliers",
            "confirmation_count",
            "confirmation_max_query_gap",
            "confirmation_max_reference_gap",
            "geometry_cache_records",
        )
        for name in integer_positive:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        finite_nonnegative = (
            "voxel_size_m",
            "min_time_separation_sec",
            "icp_max_cycle_translation_m",
            "icp_max_cycle_rotation_deg",
            "max_correction_translation_m",
            "max_correction_rotation_deg",
            "confirmation_max_translation_m",
            "confirmation_max_rotation_deg",
        )
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        finite_positive = (
            "descriptor_max_radius_m",
            "icp_max_correspondence_m",
            "icp_max_rmse_m",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        unit_interval = (
            "descriptor_min_similarity",
            "icp_trim_fraction",
            "icp_min_overlap",
            "icp_min_unique_target_ratio",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        if int(self.confirmation_count) not in {1, 2}:
            raise ValueError("confirmation_count currently supports 1 or 2")
        archive = self.archive_directory
        if archive is not None:
            archive = Path(archive).resolve()
        object.__setattr__(self, "archive_directory", archive)
        object.__setattr__(self, "enabled", bool(self.enabled))


@dataclass(frozen=True)
class IcpVerificationResult:
    transform_target_source: np.ndarray
    converged: bool
    inlier_count: int
    overlap: float
    unique_target_ratio: float
    rmse_m: float
    iterations: int

    def __post_init__(self) -> None:
        transform = validated_pose_matrix(
            self.transform_target_source, name="ICP transform"
        )
        transform.setflags(write=False)
        object.__setattr__(self, "transform_target_source", transform)


@dataclass(frozen=True)
class LoopDetectionUpdate:
    node_id: int
    candidates_considered: int
    geometrically_verified: int
    accepted_constraints: tuple[LoopConstraint, ...]
    status: str


@dataclass
class _KeyframeRecord:
    node_id: int
    keyframe_id: int
    timestamp: float
    order: int
    local_camera_pose: np.ndarray
    points_sensor: Optional[np.ndarray]
    submap_sensor: Optional[np.ndarray]
    descriptor: np.ndarray
    ring_key: np.ndarray


@dataclass
class _PendingVerification:
    constraint: LoopConstraint
    query_order: int
    reference_order: int
    correction: np.ndarray


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_sample(
    points: np.ndarray, *, voxel_size_m: float, max_points: int
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    value = value[np.isfinite(value).all(axis=1)]
    if float(voxel_size_m) > 0.0 and len(value):
        voxel = np.floor(value / float(voxel_size_m)).astype(np.int64)
        _, indices = np.unique(voxel, axis=0, return_index=True)
        value = value[np.sort(indices)]
    limit = int(max_points)
    if limit > 0 and len(value) > limit:
        indices = np.linspace(0, len(value) - 1, limit, dtype=np.int64)
        value = value[indices]
    return value.copy()


def scan_context_descriptor(
    points_sensor: np.ndarray,
    *,
    rings: int,
    sectors: int,
    max_radius_m: float,
) -> np.ndarray:
    """Build a REP-103 LiDAR-frame polar height/occupancy descriptor."""

    points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    descriptor = np.zeros((int(rings), int(sectors)), dtype=np.float64)
    if not len(points):
        return descriptor
    radius = np.linalg.norm(points[:, :2], axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & (radius > 1.0e-6)
        & (radius <= float(max_radius_m))
    )
    points = points[valid]
    radius = radius[valid]
    if not len(points):
        return descriptor
    angle = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    ring = np.minimum(
        (radius / float(max_radius_m) * int(rings)).astype(np.int64),
        int(rings) - 1,
    )
    sector = np.minimum(
        (angle / (2.0 * np.pi) * int(sectors)).astype(np.int64),
        int(sectors) - 1,
    )
    # Median-centred height keeps the descriptor insensitive to a constant
    # sensor-height offset; +1 retains occupancy information on flat ground.
    height = np.maximum(points[:, 2] - np.median(points[:, 2]) + 1.0, 0.01)
    np.maximum.at(descriptor.reshape(-1), ring * int(sectors) + sector, height)
    return descriptor


def scan_context_similarity(
    reference: np.ndarray, query: np.ndarray
) -> tuple[float, int]:
    """Return maximum sector-wise cosine similarity and cyclic shift."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(query, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Scan Context descriptor shapes differ")
    best_score = 0.0
    best_shift = 0
    first_norm = np.linalg.norm(first, axis=0)
    for shift in range(first.shape[1]):
        shifted = np.roll(second, shift, axis=1)
        second_norm = np.linalg.norm(shifted, axis=0)
        populated = (first_norm > 1.0e-12) & (second_norm > 1.0e-12)
        if not np.any(populated):
            continue
        cosine = np.sum(first[:, populated] * shifted[:, populated], axis=0)
        cosine /= first_norm[populated] * second_norm[populated]
        score = float(np.clip(np.mean(cosine), 0.0, 1.0))
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_score, best_shift


def _rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("rigid alignment needs at least three paired points")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def trimmed_point_to_point_icp(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_transform_target_source: np.ndarray,
    max_correspondence_m: float,
    max_iterations: int,
    trim_fraction: float,
) -> IcpVerificationResult:
    """Deterministic robust ICP; source is transformed into target."""

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(
        initial_transform_target_source, dtype=np.float64
    ).reshape(4, 4).copy()
    if len(source) < 3 or len(target) < 3:
        return IcpVerificationResult(
            transform, False, 0, 0.0, 0.0, float("inf"), 0
        )
    tree = cKDTree(target)
    converged = False
    completed_iterations = 0
    for iteration in range(int(max_iterations)):
        completed_iterations = iteration + 1
        transformed = _transform_points(transform, source)
        distances, target_indices = tree.query(transformed, k=1, workers=1)
        valid_indices = np.flatnonzero(distances <= float(max_correspondence_m))
        if len(valid_indices) < 3:
            break
        keep_count = max(3, int(np.ceil(len(valid_indices) * trim_fraction)))
        order = np.argsort(distances[valid_indices], kind="stable")[:keep_count]
        selected = valid_indices[order]
        delta = _rigid_alignment(
            transformed[selected], target[target_indices[selected]]
        )
        transform = delta @ transform
        delta_translation, delta_rotation = pose_change_norm(np.eye(4), delta)
        if delta_translation < 1.0e-4 and delta_rotation < 1.0e-4:
            converged = True
            break

    transformed = _transform_points(transform, source)
    distances, target_indices = tree.query(transformed, k=1, workers=1)
    valid_indices = np.flatnonzero(distances <= float(max_correspondence_m))
    if len(valid_indices):
        keep_count = max(1, int(np.ceil(len(valid_indices) * trim_fraction)))
        order = np.argsort(distances[valid_indices], kind="stable")[:keep_count]
        selected = valid_indices[order]
        inlier_distances = distances[selected]
        unique_targets = len(np.unique(target_indices[selected]))
        rmse = float(np.sqrt(np.mean(inlier_distances ** 2)))
        overlap = min(
            len(selected) / float(len(source)),
            unique_targets / float(len(target)),
        )
        unique_ratio = unique_targets / float(len(selected))
    else:
        selected = np.empty(0, dtype=np.int64)
        rmse = float("inf")
        overlap = 0.0
        unique_ratio = 0.0
    return IcpVerificationResult(
        transform_target_source=transform,
        converged=converged,
        inlier_count=int(len(selected)),
        overlap=float(overlap),
        unique_target_ratio=float(unique_ratio),
        rmse_m=rmse,
        iterations=completed_iterations,
    )


class LidarLoopClosureDetector:
    """Store bounded keyframe submaps and emit only confirmed constraints."""

    def __init__(self, configuration: LidarLoopConfiguration) -> None:
        self.configuration = configuration
        self._records: list[_KeyframeRecord] = []
        self._by_node: dict[int, _KeyframeRecord] = {}
        self._pending: list[_PendingVerification] = []
        self._transform_camera_sensor: Optional[np.ndarray] = None
        self._geometry_cache: OrderedDict[
            int, tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._archive_directory = configuration.archive_directory
        if self._archive_directory is not None:
            self._archive_directory.mkdir(parents=True, exist_ok=True)

    def _archive_path(self, node_id: int) -> Path:
        if self._archive_directory is None:
            raise RuntimeError("LiDAR loop archive is disabled")
        return self._archive_directory / f"loop_{int(node_id):020d}.npz"

    def _cache_geometry(
        self,
        node_id: int,
        points_sensor: np.ndarray,
        submap_sensor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        value = (
            np.ascontiguousarray(points_sensor, dtype=np.float32),
            np.ascontiguousarray(submap_sensor, dtype=np.float32),
        )
        self._geometry_cache[int(node_id)] = value
        self._geometry_cache.move_to_end(int(node_id))
        while (
            len(self._geometry_cache)
            > self.configuration.geometry_cache_records
        ):
            self._geometry_cache.popitem(last=False)
        return value

    def _geometry(
        self, record: _KeyframeRecord
    ) -> tuple[np.ndarray, np.ndarray]:
        if record.points_sensor is not None and record.submap_sensor is not None:
            return record.points_sensor, record.submap_sensor
        cached = self._geometry_cache.get(record.node_id)
        if cached is not None:
            self._geometry_cache.move_to_end(record.node_id)
            return cached
        path = self._archive_path(record.node_id)
        with np.load(path, allow_pickle=False) as payload:
            node_id = int(np.asarray(payload["node_id"]).reshape(()))
            if node_id != record.node_id:
                raise RuntimeError(
                    f"LiDAR loop archive identity mismatch: {node_id}"
                )
            points = np.asarray(payload["points_sensor"], dtype=np.float32)
            submap = np.asarray(payload["submap_sensor"], dtype=np.float32)
        return self._cache_geometry(record.node_id, points, submap)

    def _archive_geometry(self, record: _KeyframeRecord) -> None:
        if self._archive_directory is None:
            return
        if record.points_sensor is None or record.submap_sensor is None:
            return
        destination = self._archive_path(record.node_id)
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=self._archive_directory,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        try:
            with temporary_file:
                np.savez_compressed(
                    temporary_file,
                    node_id=np.asarray(record.node_id, dtype=np.int64),
                    points_sensor=np.asarray(
                        record.points_sensor, dtype=np.float32
                    ),
                    submap_sensor=np.asarray(
                        record.submap_sensor, dtype=np.float32
                    ),
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
        self._cache_geometry(
            record.node_id, record.points_sensor, record.submap_sensor
        )
        record.points_sensor = None
        record.submap_sensor = None

    def _build_submap(
        self,
        *,
        current_local_camera_pose: np.ndarray,
        current_points_sensor: np.ndarray,
        transform_camera_sensor: np.ndarray,
    ) -> np.ndarray:
        current_sensor_pose = (
            current_local_camera_pose @ transform_camera_sensor
        )
        clouds = [current_points_sensor]
        history_count = max(0, self.configuration.submap_keyframes - 1)
        history = self._records[-history_count:] if history_count else []
        for record in history:
            previous_points, _ = self._geometry(record)
            previous_sensor_pose = (
                record.local_camera_pose @ transform_camera_sensor
            )
            transform_current_previous = (
                inverse_pose(current_sensor_pose) @ previous_sensor_pose
            )
            clouds.append(
                _transform_points(
                    transform_current_previous, previous_points
                )
            )
        return voxel_sample(
            np.vstack(clouds),
            voxel_size_m=self.configuration.voxel_size_m,
            max_points=self.configuration.max_points,
        )

    def _candidate_shortlist(
        self, query: _KeyframeRecord
    ) -> tuple[LoopCandidate, ...]:
        eligible = [
            record
            for record in self._records[:-1]
            if (
                query.order - record.order
                >= self.configuration.min_keyframe_separation
                and query.timestamp - record.timestamp
                >= self.configuration.min_time_separation_sec
            )
        ]
        if not eligible:
            return ()
        keys = np.stack([record.ring_key for record in eligible], axis=0)
        distances = np.linalg.norm(keys - query.ring_key[None, :], axis=1)
        shortlist_count = min(
            len(eligible), self.configuration.descriptor_shortlist
        )
        shortlist = np.argsort(distances, kind="stable")[:shortlist_count]
        scored = []
        for index in shortlist:
            reference = eligible[int(index)]
            similarity, sector_shift = scan_context_similarity(
                reference.descriptor, query.descriptor
            )
            if similarity < self.configuration.descriptor_min_similarity:
                continue
            scored.append(
                LoopCandidate(
                    reference_node=reference.node_id,
                    query_node=query.node_id,
                    source="lidar_scan_context",
                    score=similarity,
                    diagnostics={
                        "sector_shift": int(sector_shift),
                        "ring_key_distance": float(distances[int(index)]),
                    },
                )
            )
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return tuple(scored[: self.configuration.max_candidates])

    def _verify(self, candidate: LoopCandidate) -> Optional[LoopConstraint]:
        reference = self._by_node.get(candidate.reference_node)
        query = self._by_node.get(candidate.query_node)
        if reference is None or query is None:
            return None
        _, reference_submap = self._geometry(reference)
        _, query_submap = self._geometry(query)
        camera_sensor = self._transform_camera_sensor
        if camera_sensor is None:
            raise RuntimeError("LiDAR loop detector has no sensor extrinsic")
        sensor_camera = inverse_pose(camera_sensor)
        if candidate.coarse_relative_pose is None:
            reference_sensor_pose = (
                reference.local_camera_pose @ camera_sensor
            )
            query_sensor_pose = query.local_camera_pose @ camera_sensor
            initial_sensor = relative_pose(
                reference_sensor_pose, query_sensor_pose
            )
        else:
            initial_sensor = (
                sensor_camera
                @ candidate.coarse_relative_pose
                @ camera_sensor
            )

        forward = trimmed_point_to_point_icp(
            source_points=query_submap,
            target_points=reference_submap,
            initial_transform_target_source=initial_sensor,
            max_correspondence_m=(
                self.configuration.icp_max_correspondence_m
            ),
            max_iterations=self.configuration.icp_max_iterations,
            trim_fraction=self.configuration.icp_trim_fraction,
        )
        reverse = trimmed_point_to_point_icp(
            source_points=reference_submap,
            target_points=query_submap,
            initial_transform_target_source=inverse_pose(
                forward.transform_target_source
            ),
            max_correspondence_m=(
                self.configuration.icp_max_correspondence_m
            ),
            max_iterations=self.configuration.icp_max_iterations,
            trim_fraction=self.configuration.icp_trim_fraction,
        )
        correction_translation, correction_rotation = pose_change_norm(
            initial_sensor, forward.transform_target_source
        )
        cycle_translation, cycle_rotation = pose_change_norm(
            forward.transform_target_source,
            inverse_pose(reverse.transform_target_source),
        )
        accepted = (
            forward.converged
            and reverse.converged
            and forward.inlier_count >= self.configuration.icp_min_inliers
            and forward.overlap >= self.configuration.icp_min_overlap
            and forward.unique_target_ratio
            >= self.configuration.icp_min_unique_target_ratio
            and forward.rmse_m <= self.configuration.icp_max_rmse_m
            and cycle_translation
            <= self.configuration.icp_max_cycle_translation_m
            and cycle_rotation
            <= np.deg2rad(self.configuration.icp_max_cycle_rotation_deg)
            and correction_translation
            <= self.configuration.max_correction_translation_m
            and correction_rotation
            <= np.deg2rad(self.configuration.max_correction_rotation_deg)
        )
        if not accepted:
            return None

        measured_camera = (
            camera_sensor
            @ forward.transform_target_source
            @ sensor_camera
        )
        characteristic_radius = max(
            1.0,
            float(
                np.median(
                    np.linalg.norm(reference_submap[:, :2], axis=1)
                )
            ),
        )
        rotation_sigma = np.clip(
            forward.rmse_m / characteristic_radius,
            np.deg2rad(0.2),
            np.deg2rad(3.0),
        )
        translation_sigma = np.clip(
            forward.rmse_m / np.sqrt(max(forward.overlap, 0.05)),
            0.02,
            0.50,
        )
        rmse_quality = np.exp(
            -forward.rmse_m / self.configuration.icp_max_rmse_m
        )
        confidence = float(
            np.clip(
                candidate.score * forward.overlap * rmse_quality,
                0.0,
                1.0,
            )
        )
        return LoopConstraint(
            reference_node=reference.node_id,
            query_node=query.node_id,
            relative_pose_reference_query=measured_camera,
            sigmas=np.r_[
                np.full(3, rotation_sigma),
                np.full(3, translation_sigma),
            ],
            source=f"{candidate.source}+lidar_icp",
            confidence=confidence,
            diagnostics={
                "candidate_score": candidate.score,
                "icp_inliers": forward.inlier_count,
                "icp_overlap": forward.overlap,
                "icp_unique_target_ratio": forward.unique_target_ratio,
                "icp_rmse_m": forward.rmse_m,
                "icp_iterations": forward.iterations,
                "cycle_translation_m": cycle_translation,
                "cycle_rotation_deg": np.rad2deg(cycle_rotation),
                "correction_translation_m": correction_translation,
                "correction_rotation_deg": np.rad2deg(correction_rotation),
            },
        )

    def _confirm(
        self, constraint: LoopConstraint
    ) -> tuple[LoopConstraint, ...]:
        if self.configuration.confirmation_count == 1:
            return (constraint,)
        query = self._by_node[constraint.query_node]
        reference = self._by_node[constraint.reference_node]
        correction = (
            reference.local_camera_pose
            @ constraint.relative_pose_reference_query
            @ inverse_pose(query.local_camera_pose)
        )
        self._pending = [
            pending
            for pending in self._pending
            if query.order - pending.query_order
            <= self.configuration.confirmation_max_query_gap
        ]
        for pending_index, pending in enumerate(self._pending):
            if (
                query.order <= pending.query_order
                or query.order - pending.query_order
                > self.configuration.confirmation_max_query_gap
                or abs(reference.order - pending.reference_order)
                > self.configuration.confirmation_max_reference_gap
            ):
                continue
            translation_m, rotation_rad = pose_change_norm(
                pending.correction, correction
            )
            if (
                translation_m
                <= self.configuration.confirmation_max_translation_m
                and rotation_rad
                <= np.deg2rad(
                    self.configuration.confirmation_max_rotation_deg
                )
            ):
                self._pending.pop(pending_index)
                return pending.constraint, constraint
        self._pending.append(
            _PendingVerification(
                constraint=constraint,
                query_order=query.order,
                reference_order=reference.order,
                correction=correction,
            )
        )
        return ()

    def add_keyframe(
        self,
        *,
        node_id: int,
        keyframe_id: int,
        timestamp: float,
        local_camera_pose: np.ndarray,
        points_sensor: np.ndarray,
        transform_camera_sensor: np.ndarray,
    ) -> LoopDetectionUpdate:
        node = int(node_id)
        if node in self._by_node:
            return LoopDetectionUpdate(node, 0, 0, (), "duplicate_keyframe")
        camera_pose = validated_pose_matrix(
            local_camera_pose, name="LiDAR loop camera pose"
        )
        camera_sensor = validated_pose_matrix(
            transform_camera_sensor,
            name="LiDAR loop camera-sensor transform",
        )
        if self._transform_camera_sensor is None:
            self._transform_camera_sensor = camera_sensor.copy()
        elif not np.allclose(
            self._transform_camera_sensor, camera_sensor, atol=1.0e-8
        ):
            raise ValueError("LiDAR loop keyframes use inconsistent extrinsics")
        sampled = voxel_sample(
            points_sensor,
            voxel_size_m=self.configuration.voxel_size_m,
            max_points=self.configuration.max_points,
        )
        if len(sampled) < 3:
            return LoopDetectionUpdate(node, 0, 0, (), "insufficient_points")
        submap = self._build_submap(
            current_local_camera_pose=camera_pose,
            current_points_sensor=sampled,
            transform_camera_sensor=camera_sensor,
        )
        if self.configuration.enabled:
            descriptor = scan_context_descriptor(
                submap,
                rings=self.configuration.descriptor_rings,
                sectors=self.configuration.descriptor_sectors,
                max_radius_m=self.configuration.descriptor_max_radius_m,
            ).astype(np.float32)
            ring_key = np.mean(descriptor, axis=1)
        else:
            descriptor = np.empty((0, 0), dtype=np.float32)
            ring_key = np.empty(0, dtype=np.float32)
        record = _KeyframeRecord(
            node_id=node,
            keyframe_id=int(keyframe_id),
            timestamp=float(timestamp),
            order=len(self._records),
            local_camera_pose=camera_pose.copy(),
            points_sensor=sampled,
            submap_sensor=submap,
            descriptor=descriptor,
            ring_key=ring_key,
        )
        self._records.append(record)
        self._by_node[node] = record
        self._archive_geometry(record)
        if not self.configuration.enabled:
            return LoopDetectionUpdate(node, 0, 0, (), "disabled")

        candidates = self._candidate_shortlist(record)
        verified = [
            constraint
            for constraint in (self._verify(candidate) for candidate in candidates)
            if constraint is not None
        ]
        if not verified:
            return LoopDetectionUpdate(
                node, len(candidates), 0, (),
                "no_geometrically_verified_candidate",
            )
        best = max(verified, key=lambda constraint: constraint.confidence)
        accepted = self._confirm(best)
        status = "accepted" if accepted else "awaiting_temporal_confirmation"
        return LoopDetectionUpdate(
            node,
            len(candidates),
            len(verified),
            accepted,
            status,
        )

    def submit_candidates(
        self, candidates: Iterable[LoopCandidate]
    ) -> LoopDetectionUpdate:
        """Verify a provider shortlist and confirm only its best geometry.

        Treating one retrieval response as a batch prevents several
        appearance alternatives for the same query from polluting temporal
        confirmation state.  Every alternative is still independently
        checked by bidirectional ICP before confidence is compared.
        """

        values = tuple(candidates)
        if not values:
            raise ValueError("external loop candidate shortlist is empty")
        query_node = values[0].query_node
        if any(value.query_node != query_node for value in values):
            raise ValueError("external loop candidates use different queries")
        if query_node not in self._by_node:
            return LoopDetectionUpdate(
                query_node, len(values), 0, (), "unknown_query_node"
            )
        verified = tuple(
            constraint
            for constraint in (self._verify(value) for value in values)
            if constraint is not None
        )
        if not verified:
            return LoopDetectionUpdate(
                query_node, len(values), 0, (), "geometry_rejected"
            )
        best = max(verified, key=lambda constraint: constraint.confidence)
        accepted = self._confirm(best)
        return LoopDetectionUpdate(
            query_node,
            len(values),
            len(verified),
            accepted,
            "accepted" if accepted else "awaiting_temporal_confirmation",
        )

    def submit_candidate(self, candidate: LoopCandidate) -> LoopDetectionUpdate:
        """Verify one external candidate through the common batch gate."""

        return self.submit_candidates((candidate,))
