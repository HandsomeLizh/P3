"""Online two-stage MPRF retrieval and multimodal coarse-pose engine."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from .config import (
    PROTOCOL_VERSION,
    SOURCE_BUNDLE_LABEL,
    MprfRuntimeConfiguration,
)
from .contracts import KeyframePacket, PoseFeatureSet
from .geometry import (
    estimate_pose_ransac,
    match_local_features,
    normalized_vector,
)
from .models import TorchMprfFeatureBackend
from .storage import SessionKeyframeStore, StoredKeyframe


class MprfOnlineEngine:
    """Serial GPU inference with persistent keyframes and bounded feature LRU."""

    def __init__(
        self,
        configuration: MprfRuntimeConfiguration,
        *,
        feature_backend=None,
    ) -> None:
        self.configuration = configuration
        self.feature_backend = (
            feature_backend
            if feature_backend is not None
            else TorchMprfFeatureBackend(configuration)
        )
        self._store: SessionKeyframeStore | None = None
        self._pose_cache: OrderedDict[tuple[str, int], PoseFeatureSet] = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self._request_count = 0
        self._candidate_count = 0
        self._last_status = "ready"
        self._started_at = time.time()

    def _select_store(self, session_id: str) -> SessionKeyframeStore:
        if self._store is None or self._store.session_id != session_id:
            self._store = SessionKeyframeStore(
                self.configuration.state_root, session_id
            )
            self._pose_cache.clear()
        return self._store

    def _cache_pose_features(
        self, packet: KeyframePacket, features: PoseFeatureSet
    ) -> None:
        key = (packet.session_id, packet.node_id)
        self._pose_cache[key] = features
        self._pose_cache.move_to_end(key)
        while len(self._pose_cache) > self.configuration.max_pose_cache_records:
            self._pose_cache.popitem(last=False)

    def _pose_features(
        self,
        packet: KeyframePacket,
    ) -> PoseFeatureSet:
        key = (packet.session_id, packet.node_id)
        cached = self._pose_cache.get(key)
        if cached is not None:
            self._pose_cache.move_to_end(key)
            return cached
        features = self.feature_backend.extract_pose_features(packet)
        self._cache_pose_features(packet, features)
        return features

    def _eligible_records(
        self, store: SessionKeyframeStore, packet: KeyframePacket
    ) -> tuple[StoredKeyframe, ...]:
        result = []
        for record in store.records():
            if (
                packet.node_id - record.node_id
                < self.configuration.min_keyframe_separation
            ):
                continue
            separation = (packet.timestamp_ns - record.timestamp_ns) * 1.0e-9
            if separation < self.configuration.min_time_separation_sec:
                continue
            result.append(record)
        return tuple(result)

    @staticmethod
    def _cosines(query: np.ndarray, descriptors) -> np.ndarray:
        query_value = normalized_vector(query)
        database = np.stack(
            [normalized_vector(value) for value in descriptors], axis=0
        )
        return np.clip(
            np.asarray(database @ query_value, dtype=np.float64),
            -1.0,
            1.0,
        )

    @staticmethod
    def _rank_records(
        query: np.ndarray,
        records: tuple[StoredKeyframe, ...],
        *,
        descriptor_name: str,
        maximum_results: int,
        minimum_score: float,
        chunk_size: int = 256,
    ) -> tuple[tuple[StoredKeyframe, float], ...]:
        """Exact cosine top-k with bounded temporary memory.

        This deliberately avoids truncating the historical database.  It is
        slower than an approximate index in the extreme long-run case, but
        deterministic, information-preserving and predictable on embedded
        memory.  The expensive 8448-D matrix is never materialised at once.
        """

        query_value = normalized_vector(query)
        scored: list[tuple[StoredKeyframe, float]] = []
        for start in range(0, len(records), int(chunk_size)):
            block_records = records[start : start + int(chunk_size)]
            block = np.stack(
                [
                    np.asarray(
                        getattr(record, descriptor_name), dtype=np.float32
                    )
                    for record in block_records
                ],
                axis=0,
            )
            scores = np.clip(block @ query_value, -1.0, 1.0)
            scored.extend(
                (record, float(score))
                for record, score in zip(block_records, scores)
                if float(score) >= float(minimum_score)
            )
        scored.sort(key=lambda item: (-item[1], item[0].node_id))
        return tuple(scored[: int(maximum_results)])

    def process(self, packet: KeyframePacket) -> dict[str, object]:
        with self._lock:
            request_start = time.perf_counter()
            self._request_count += 1
            store = self._select_store(packet.session_id)
            duplicate = store.get(packet.node_id)
            if duplicate is not None:
                if duplicate.timestamp_ns != packet.timestamp_ns:
                    raise ValueError("duplicate MPRF node has conflicting timestamp")
                previous_response = store.load_response(duplicate)
                if previous_response is not None:
                    self._last_status = "idempotent_replay"
                    return previous_response
                self._last_status = "duplicate"
                return self._base_response(packet, status="duplicate")
            if len(packet.points_sensor) < 32:
                self._last_status = "insufficient_points"
                response = self._base_response(
                    packet, status="insufficient_points"
                )
                response["point_count"] = int(len(packet.points_sensor))
                return response

            retrieval_start = time.perf_counter()
            salad, dino = self.feature_backend.extract_retrieval(
                packet.image_bgr
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
            eligible = self._eligible_records(store, packet)
            candidate_responses: list[dict[str, object]] = []
            pose_ms = 0.0
            pose_solved_count = 0
            shortlisted_count = 0
            refined_count = 0
            if eligible:
                ranked = self._rank_records(
                    salad,
                    eligible,
                    descriptor_name="salad_descriptor",
                    maximum_results=self.configuration.retrieval_shortlist,
                    minimum_score=self.configuration.min_salad_similarity,
                )
                shortlisted_count = len(ranked)
                if ranked:
                    shortlist = [item[0] for item in ranked]
                    salad_short = np.asarray(
                        [item[1] for item in ranked], dtype=np.float64
                    )
                    dino_scores = self._cosines(
                        dino,
                        [record.dino_descriptor for record in shortlist],
                    )
                    weight_sum = (
                        self.configuration.salad_score_weight
                        + self.configuration.dino_score_weight
                    )
                    combined = (
                        self.configuration.salad_score_weight * salad_short
                        + self.configuration.dino_score_weight * dino_scores
                    ) / weight_sum
                    refined_order = np.argsort(-combined, kind="stable")
                    refined_order = [
                        int(index)
                        for index in refined_order
                        if dino_scores[int(index)]
                        >= self.configuration.min_dino_similarity
                    ][: self.configuration.refined_shortlist]
                    refined_count = len(refined_order)
                    if refined_order:
                        pose_start = time.perf_counter()
                        query_features = self._pose_features(packet)
                        for refined_index in refined_order:
                            reference_record = shortlist[refined_index]
                            reference_packet = store.load_packet(reference_record)
                            retrieval_score = float(combined[refined_index])
                            # Place retrieval is already useful even when the
                            # cross-modal local matcher cannot recover a
                            # trustworthy coarse pose.  Such a proposal is
                            # explicitly marked retrieval-only; the ROS side
                            # will initialise ICP from local odometry and must
                            # still pass the same bidirectional LiDAR gate.
                            candidate_response: dict[str, object] = {
                                "reference_node": reference_record.node_id,
                                "reference_timestamp_ns": (
                                    reference_record.timestamp_ns
                                ),
                                "query_node": packet.node_id,
                                "query_timestamp_ns": packet.timestamp_ns,
                                "score": retrieval_score,
                                "salad_similarity": float(
                                    salad_short[refined_index]
                                ),
                                "dino_similarity": float(
                                    dino_scores[refined_index]
                                ),
                                "retrieval_score": retrieval_score,
                                "candidate_mode": "retrieval_only",
                                "matches": 0,
                            }
                            reference_features = self._pose_features(
                                reference_packet
                            )
                            if (
                                len(query_features.points_sensor) < 3
                                or len(reference_features.points_sensor) < 3
                                or query_features.descriptors.shape[1]
                                != reference_features.descriptors.shape[1]
                            ):
                                candidate_response["pose_rejection"] = (
                                    "insufficient_projected_features"
                                )
                                candidate_responses.append(candidate_response)
                                continue
                            matches = match_local_features(
                                reference_descriptors=(
                                    reference_features.descriptors
                                ),
                                query_descriptors=query_features.descriptors,
                                mode=self.configuration.match_mode,
                                minimum_similarity=(
                                    self.configuration.match_min_similarity
                                ),
                                minimum_margin=(
                                    self.configuration.match_min_margin
                                ),
                                maximum_matches=(
                                    self.configuration.match_max_count
                                ),
                            )
                            candidate_response["matches"] = int(
                                len(matches.similarities)
                            )
                            result = estimate_pose_ransac(
                                reference_points=(
                                    reference_features.points_sensor
                                ),
                                query_points=query_features.points_sensor,
                                matches=matches,
                                maximum_iterations=(
                                    self.configuration.ransac_iterations
                                ),
                                residual_threshold_m=(
                                    self.configuration.ransac_residual_m
                                ),
                                minimum_inliers=(
                                    self.configuration.ransac_min_inliers
                                ),
                                minimum_inlier_ratio=(
                                    self.configuration.ransac_min_inlier_ratio
                                ),
                                maximum_rmse_m=(
                                    self.configuration.ransac_max_rmse_m
                                ),
                                minimum_spread_m=(
                                    self.configuration.ransac_min_spread_m
                                ),
                                maximum_translation_m=(
                                    self.configuration.coarse_max_translation_m
                                ),
                                maximum_rotation_deg=(
                                    self.configuration.coarse_max_rotation_deg
                                ),
                                random_seed=(
                                    packet.timestamp_ns
                                    ^ reference_record.timestamp_ns
                                ),
                            )
                            if result is None:
                                candidate_response["pose_rejection"] = (
                                    "ransac_or_quality_gate"
                                )
                                candidate_responses.append(candidate_response)
                                continue
                            sensor_reference_query = (
                                result.transform_reference_query
                            )
                            camera_reference_query = (
                                reference_packet.transform_camera_sensor
                                @ sensor_reference_query
                                @ np.linalg.inv(
                                    packet.transform_camera_sensor
                                )
                            )
                            geometric_quality = (
                                result.inlier_ratio
                                * np.exp(
                                    -result.rmse_m
                                    / max(
                                        self.configuration.ransac_max_rmse_m,
                                        1.0e-6,
                                    )
                                )
                            )
                            score = float(
                                np.clip(
                                    retrieval_score * geometric_quality,
                                    0.0,
                                    1.0,
                                )
                            )
                            candidate_response.update({
                                "score": score,
                                "candidate_mode": "multimodal_pose",
                                "relative_pose_reference_query": (
                                    camera_reference_query.reshape(-1).tolist()
                                ),
                                "ransac_inliers": result.inlier_count,
                                "ransac_inlier_ratio": result.inlier_ratio,
                                "ransac_rmse_m": result.rmse_m,
                                "ransac_median_error_m": (
                                    result.median_error_m
                                ),
                                "reference_spread_m": (
                                    result.reference_spread_m
                                ),
                                "query_spread_m": result.query_spread_m,
                                "ransac_iterations": result.iterations,
                            })
                            pose_solved_count += 1
                            candidate_responses.append(candidate_response)
                        pose_ms = (time.perf_counter() - pose_start) * 1000.0

            total_ms = (time.perf_counter() - request_start) * 1000.0
            status = "candidate" if candidate_responses else "stored"
            self._last_status = status
            if candidate_responses:
                self._candidate_count += 1
            response = self._base_response(packet, status=status)
            response.update(
                {
                    "database_records": len(store) + 1,
                    "eligible_records": len(eligible),
                    "shortlisted_records": shortlisted_count,
                    "refined_records": refined_count,
                    "pose_solved_records": pose_solved_count,
                    "timings_ms": {
                        "retrieval": retrieval_ms,
                        "pose": pose_ms,
                        "total": total_ms,
                    },
                }
            )
            if candidate_responses:
                # Keep the singular field for protocol-v1 compatibility and
                # add the complete shortlist for safe best-of-N LiDAR
                # verification.  No item is a graph factor at this boundary.
                response["candidate"] = candidate_responses[0]
                response["candidates"] = candidate_responses
            store.add(
                packet,
                salad_descriptor=salad,
                dino_descriptor=dino,
                response_payload=response,
            )
            return response

    def _base_response(
        self, packet: KeyframePacket, *, status: str
    ) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "status": str(status),
            "session_id": packet.session_id,
            "query_node": packet.node_id,
            "query_timestamp_ns": packet.timestamp_ns,
        }

    def reset(self, session_id: str) -> dict[str, object]:
        with self._lock:
            self._store = SessionKeyframeStore(
                self.configuration.state_root, session_id
            )
            self._pose_cache.clear()
            self._last_status = "reset"
            return {
                "protocol_version": PROTOCOL_VERSION,
                "status": "ready",
                "session_id": session_id,
                "database_records": len(self._store),
            }

    def health(self) -> dict[str, object]:
        with self._lock:
            store = self._store
            metadata = self.feature_backend.metadata()
            return {
                "protocol_version": PROTOCOL_VERSION,
                "status": "ready",
                "last_status": self._last_status,
                "session_id": None if store is None else store.session_id,
                "database_records": 0 if store is None else len(store),
                "pose_cache_records": len(self._pose_cache),
                "requests": self._request_count,
                "candidates": self._candidate_count,
                "uptime_sec": max(0.0, time.time() - self._started_at),
                "source_bundle": SOURCE_BUNDLE_LABEL,
                "models": metadata,
            }
