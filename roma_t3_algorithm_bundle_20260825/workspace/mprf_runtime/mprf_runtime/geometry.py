"""Dependency-light matching and robust metric coarse-pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def normalized_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("descriptors must be a matrix")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1.0e-12)


def normalized_vector(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("descriptor norm is zero or non-finite")
    return result / norm


@dataclass(frozen=True)
class FeatureMatches:
    reference_indices: np.ndarray
    query_indices: np.ndarray
    similarities: np.ndarray

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_indices, dtype=np.int64).reshape(-1)
        query = np.asarray(self.query_indices, dtype=np.int64).reshape(-1)
        score = np.asarray(self.similarities, dtype=np.float64).reshape(-1)
        if len(reference) != len(query) or len(query) != len(score):
            raise ValueError("feature match arrays differ in length")
        if not np.isfinite(score).all():
            raise ValueError("feature match scores contain non-finite values")
        object.__setattr__(self, "reference_indices", reference)
        object.__setattr__(self, "query_indices", query)
        object.__setattr__(self, "similarities", score)


def match_local_features(
    *,
    reference_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    mode: str,
    minimum_similarity: float,
    minimum_margin: float,
    maximum_matches: int,
) -> FeatureMatches:
    reference = normalized_rows(reference_descriptors)
    query = normalized_rows(query_descriptors)
    if not len(reference) or not len(query):
        return FeatureMatches([], [], [])
    similarity = query @ reference.T
    method = str(mode).strip().lower()
    if method == "hungarian":
        query_indices, reference_indices = linear_sum_assignment(-similarity)
        scores = similarity[query_indices, reference_indices]
        keep = scores >= float(minimum_similarity)
    elif method == "mutual":
        query_to_reference = np.argmax(similarity, axis=1)
        reference_to_query = np.argmax(similarity, axis=0)
        query_indices = np.arange(len(query), dtype=np.int64)
        reference_indices = query_to_reference
        mutual = reference_to_query[reference_indices] == query_indices
        best = similarity[query_indices, reference_indices]
        if similarity.shape[1] > 1:
            second = np.partition(similarity, -2, axis=1)[:, -2]
        else:
            second = np.full(len(query), -1.0, dtype=np.float32)
        keep = (
            mutual
            & (best >= float(minimum_similarity))
            & ((best - second) >= float(minimum_margin))
        )
        scores = best
    else:
        raise ValueError("feature match mode must be mutual or hungarian")
    query_indices = np.asarray(query_indices, dtype=np.int64)[keep]
    reference_indices = np.asarray(reference_indices, dtype=np.int64)[keep]
    scores = np.asarray(scores, dtype=np.float64)[keep]
    if len(scores) > int(maximum_matches):
        order = np.argsort(-scores, kind="stable")[: int(maximum_matches)]
        query_indices = query_indices[order]
        reference_indices = reference_indices[order]
        scores = scores[order]
    return FeatureMatches(reference_indices, query_indices, scores)


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("rigid alignment requires at least three point pairs")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    pose = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return value @ pose[:3, :3].T + pose[:3, 3]


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _spatial_spread(points: np.ndarray) -> float:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(value) < 3:
        return 0.0
    singular = np.linalg.svd(value - np.mean(value, axis=0), compute_uv=False)
    return float(singular[1] / np.sqrt(len(value)))


@dataclass(frozen=True)
class RansacPoseResult:
    transform_reference_query: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_ratio: float
    rmse_m: float
    median_error_m: float
    reference_spread_m: float
    query_spread_m: float
    iterations: int


def estimate_pose_ransac(
    *,
    reference_points: np.ndarray,
    query_points: np.ndarray,
    matches: FeatureMatches,
    maximum_iterations: int,
    residual_threshold_m: float,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
    maximum_rmse_m: float,
    minimum_spread_m: float,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
    random_seed: int,
) -> RansacPoseResult | None:
    reference = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query_points, dtype=np.float64).reshape(-1, 3)
    paired_reference = reference[matches.reference_indices]
    paired_query = query[matches.query_indices]
    pair_count = len(paired_query)
    if pair_count < max(3, int(minimum_inliers)):
        return None
    generator = np.random.default_rng(int(random_seed) & 0xFFFFFFFF)
    weights = np.clip(matches.similarities + 1.0e-6, 0.0, None)
    probability = weights / np.sum(weights) if np.sum(weights) > 0.0 else None
    best_mask = None
    best_count = 0
    best_median = float("inf")
    completed = 0
    target_iterations = int(maximum_iterations)
    for iteration in range(int(maximum_iterations)):
        completed = iteration + 1
        sample = generator.choice(
            pair_count, size=3, replace=False, p=probability
        )
        if (
            _spatial_spread(paired_query[sample]) < 1.0e-4
            or _spatial_spread(paired_reference[sample]) < 1.0e-4
        ):
            continue
        hypothesis = rigid_alignment(
            paired_query[sample], paired_reference[sample]
        )
        errors = np.linalg.norm(
            transform_points(hypothesis, paired_query) - paired_reference,
            axis=1,
        )
        mask = errors <= float(residual_threshold_m)
        count = int(np.count_nonzero(mask))
        median = float(np.median(errors[mask])) if count else float("inf")
        if count > best_count or (count == best_count and median < best_median):
            best_mask = mask
            best_count = count
            best_median = median
            ratio = count / float(pair_count)
            if ratio > 0.0:
                success = float(
                    np.clip(ratio ** 3, 1.0e-12, 1.0 - 1.0e-12)
                )
                adaptive = np.log(1.0 - 0.999) / np.log(1.0 - success)
                if np.isfinite(adaptive):
                    target_iterations = min(
                        target_iterations, max(iteration + 1, int(np.ceil(adaptive)))
                    )
        if completed >= target_iterations:
            break
    if best_mask is None:
        return None
    ratio = best_count / float(pair_count)
    if best_count < int(minimum_inliers) or ratio < float(minimum_inlier_ratio):
        return None
    transform = rigid_alignment(
        paired_query[best_mask], paired_reference[best_mask]
    )
    errors = np.linalg.norm(
        transform_points(transform, paired_query) - paired_reference, axis=1
    )
    inlier_mask = errors <= float(residual_threshold_m)
    if np.count_nonzero(inlier_mask) >= 3:
        transform = rigid_alignment(
            paired_query[inlier_mask], paired_reference[inlier_mask]
        )
        errors = np.linalg.norm(
            transform_points(transform, paired_query) - paired_reference,
            axis=1,
        )
        inlier_mask = errors <= float(residual_threshold_m)
    inlier_errors = errors[inlier_mask]
    count = int(len(inlier_errors))
    ratio = count / float(pair_count)
    if count < int(minimum_inliers) or ratio < float(minimum_inlier_ratio):
        return None
    rmse = float(np.sqrt(np.mean(inlier_errors ** 2)))
    median = float(np.median(inlier_errors))
    reference_spread = _spatial_spread(paired_reference[inlier_mask])
    query_spread = _spatial_spread(paired_query[inlier_mask])
    translation = float(np.linalg.norm(transform[:3, 3]))
    rotation_deg = float(np.rad2deg(_rotation_angle(transform[:3, :3])))
    if (
        rmse > float(maximum_rmse_m)
        or reference_spread < float(minimum_spread_m)
        or query_spread < float(minimum_spread_m)
        or translation > float(maximum_translation_m)
        or rotation_deg > float(maximum_rotation_deg)
    ):
        return None
    result_pose = transform.copy()
    result_pose.setflags(write=False)
    mask_copy = np.asarray(inlier_mask, dtype=bool).copy()
    mask_copy.setflags(write=False)
    return RansacPoseResult(
        transform_reference_query=result_pose,
        inlier_mask=mask_copy,
        inlier_count=count,
        inlier_ratio=ratio,
        rmse_m=rmse,
        median_error_m=median,
        reference_spread_m=reference_spread,
        query_spread_m=query_spread,
        iterations=completed,
    )
