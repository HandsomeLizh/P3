"""Independent global keyframe graph for loop-closure corrections.

This graph intentionally sits *after* the local fixed-lag VO/LIO estimator.
It consumes frozen local keyframe increments and verified loop constraints;
its result is a global correction layer and is never fed back as another local
measurement.  That separation prevents double counting and keeps real-time
odometry available even when global loop closure is disabled or delayed.
"""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .loop_constraints import LoopConstraint
from .pose_math import (
    interpolate_pose,
    inverse_pose,
    pose_change_norm,
    relative_pose,
    validated_pose_matrix,
)


class GlobalPoseGraphDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class GlobalPoseGraphUpdate:
    node_poses: Mapping[int, np.ndarray]
    changed_nodes: tuple[int, ...]
    loop_constraints_added: int

    def __post_init__(self) -> None:
        poses = {}
        for node_id, pose in self.node_poses.items():
            value = validated_pose_matrix(
                pose, name=f"global graph pose {int(node_id)}"
            )
            value.setflags(write=False)
            poses[int(node_id)] = value
        object.__setattr__(self, "node_poses", MappingProxyType(poses))
        object.__setattr__(
            self,
            "changed_nodes",
            tuple(int(node_id) for node_id in self.changed_nodes),
        )
        object.__setattr__(
            self,
            "loop_constraints_added",
            int(self.loop_constraints_added),
        )


def _positive_sigmas(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(6)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain six positive finite values")
    return result.copy()


class GtsamGlobalPoseGraph:
    """Unbounded iSAM2 Pose3 graph with a global-to-local correction model."""

    def __init__(
        self,
        *,
        prior_sigmas=(
            np.deg2rad(0.05),
            np.deg2rad(0.05),
            np.deg2rad(0.05),
            1.0e-3,
            1.0e-3,
            1.0e-3,
        ),
        odometry_rotation_sigma_deg=(0.8, 0.8, 1.2),
        odometry_translation_sigma_m=(0.05, 0.05, 0.08),
        robust_huber_k: float = 1.5,
        relinearize_threshold: float = 0.01,
    ) -> None:
        try:
            import gtsam
        except ImportError as exception:
            raise GlobalPoseGraphDependencyError(
                "global loop closure requires the Python gtsam module"
            ) from exception

        self.gtsam = gtsam
        self.prior_sigmas = _positive_sigmas(
            prior_sigmas, name="global prior sigmas"
        )
        rotation = np.asarray(
            odometry_rotation_sigma_deg, dtype=np.float64
        ).reshape(3)
        translation = np.asarray(
            odometry_translation_sigma_m, dtype=np.float64
        ).reshape(3)
        self.odometry_sigmas = _positive_sigmas(
            np.r_[np.deg2rad(rotation), translation],
            name="global odometry sigmas",
        )
        self.robust_huber_k = float(robust_huber_k)
        if not np.isfinite(self.robust_huber_k):
            raise ValueError("global robust_huber_k must be finite")
        threshold = float(relinearize_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("global relinearize_threshold must be positive")

        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(threshold)
        params.relinearizeSkip = 1
        self._isam = gtsam.ISAM2(params)
        self._local_poses: dict[int, np.ndarray] = {}
        self._global_poses: dict[int, np.ndarray] = {}
        self._timestamps: dict[int, float] = {}
        self._node_order: list[int] = []
        self._loop_pairs: set[tuple[int, int]] = set()
        self._lock = threading.RLock()

    def _x(self, node_id: int) -> int:
        return self.gtsam.symbol("x", int(node_id))

    def _pose3(self, pose: np.ndarray):
        return self.gtsam.Pose3(validated_pose_matrix(pose))

    def _diagonal_noise(self, sigmas: np.ndarray):
        return self.gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    def _loop_noise(self, sigmas: np.ndarray):
        base = self._diagonal_noise(sigmas)
        if self.robust_huber_k <= 0.0:
            return base
        return self.gtsam.noiseModel.Robust.Create(
            self.gtsam.noiseModel.mEstimator.Huber.Create(
                self.robust_huber_k
            ),
            base,
        )

    def _refresh_estimate(
        self,
        previous: Mapping[int, np.ndarray],
        *,
        loops_added: int,
    ) -> GlobalPoseGraphUpdate:
        estimate = self._isam.calculateEstimate()
        poses = {}
        changed = []
        for node_id in self._node_order:
            key = self._x(node_id)
            if not estimate.exists(key):
                raise RuntimeError(
                    f"global graph estimate omitted node {node_id}"
                )
            pose = np.asarray(
                estimate.atPose3(key).matrix(), dtype=np.float64
            )
            poses[node_id] = pose
            old = previous.get(node_id)
            if old is None:
                continue
            translation_m, rotation_rad = pose_change_norm(old, pose)
            if translation_m > 1.0e-9 or rotation_rad > 1.0e-9:
                changed.append(node_id)
        self._global_poses = poses
        return GlobalPoseGraphUpdate(
            node_poses=poses,
            changed_nodes=tuple(changed),
            loop_constraints_added=loops_added,
        )

    def add_keyframe(
        self,
        *,
        node_id: int,
        timestamp: float,
        local_pose: np.ndarray,
    ) -> GlobalPoseGraphUpdate:
        """Append one immutable local-odometry support pose.

        A retry with the same node is idempotent.  The original local snapshot
        remains the graph measurement even if the fixed-lag estimator later
        refines its live history; exact graph-node corrections remain
        authoritative for the global map.
        """

        node = int(node_id)
        sensor_time = float(timestamp)
        if node < 0:
            raise ValueError("global graph node IDs must be non-negative")
        if not np.isfinite(sensor_time):
            raise ValueError("global graph timestamp must be finite")
        local = validated_pose_matrix(local_pose, name="local keyframe pose")
        with self._lock:
            if node in self._local_poses:
                return GlobalPoseGraphUpdate(
                    node_poses=self._global_poses,
                    changed_nodes=(),
                    loop_constraints_added=0,
                )
            if self._node_order:
                previous_node = self._node_order[-1]
                if node <= previous_node:
                    raise ValueError(
                        "global graph nodes must be appended in increasing "
                        f"order: {node} <= {previous_node}"
                    )
                if sensor_time <= self._timestamps[previous_node]:
                    raise ValueError(
                        "global graph timestamps must be strictly increasing"
                    )

            graph = self.gtsam.NonlinearFactorGraph()
            values = self.gtsam.Values()
            if not self._node_order:
                initial = self._pose3(local)
                values.insert(self._x(node), initial)
                graph.add(
                    self.gtsam.PriorFactorPose3(
                        self._x(node),
                        initial,
                        self._diagonal_noise(self.prior_sigmas),
                    )
                )
            else:
                previous_node = self._node_order[-1]
                increment = relative_pose(
                    self._local_poses[previous_node], local
                )
                graph.add(
                    self.gtsam.BetweenFactorPose3(
                        self._x(previous_node),
                        self._x(node),
                        self._pose3(increment),
                        self._diagonal_noise(self.odometry_sigmas),
                    )
                )
                initial_matrix = (
                    self._global_poses[previous_node] @ increment
                )
                values.insert(self._x(node), self._pose3(initial_matrix))

            previous = dict(self._global_poses)
            self._local_poses[node] = local
            self._timestamps[node] = sensor_time
            self._node_order.append(node)
            self._isam.update(graph, values)
            return self._refresh_estimate(previous, loops_added=0)

    def add_loop_constraints(
        self, constraints: Iterable[LoopConstraint]
    ) -> GlobalPoseGraphUpdate:
        """Atomically add new, already verified loop factors."""

        requested = tuple(constraints)
        with self._lock:
            graph = self.gtsam.NonlinearFactorGraph()
            new_pairs = []
            for constraint in requested:
                reference = int(constraint.reference_node)
                query = int(constraint.query_node)
                if (
                    reference not in self._local_poses
                    or query not in self._local_poses
                ):
                    raise KeyError(
                        "loop constraint references an unknown graph node: "
                        f"{reference}->{query}"
                    )
                pair = tuple(sorted((reference, query)))
                if pair in self._loop_pairs or pair in new_pairs:
                    continue
                graph.add(
                    self.gtsam.BetweenFactorPose3(
                        self._x(reference),
                        self._x(query),
                        self._pose3(
                            constraint.relative_pose_reference_query
                        ),
                        self._loop_noise(constraint.sigmas),
                    )
                )
                new_pairs.append(pair)

            if not new_pairs:
                return GlobalPoseGraphUpdate(
                    node_poses=self._global_poses,
                    changed_nodes=(),
                    loop_constraints_added=0,
                )
            previous = dict(self._global_poses)
            self._isam.update(graph, self.gtsam.Values())
            # Force one additional wildfire/relinearisation pass.  Loop
            # closure is low-rate, so this small cost is preferable to
            # exposing a half-propagated correction to map reconstruction.
            self._isam.update()
            self._loop_pairs.update(new_pairs)
            return self._refresh_estimate(
                previous, loops_added=len(new_pairs)
            )

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._node_order)

    @property
    def loop_count(self) -> int:
        with self._lock:
            return len(self._loop_pairs)

    def has_node(self, node_id: int) -> bool:
        with self._lock:
            return int(node_id) in self._local_poses

    def node_poses(self) -> dict[int, np.ndarray]:
        with self._lock:
            return {
                node_id: pose.copy()
                for node_id, pose in self._global_poses.items()
            }

    def correction_for_index(self, node_id: int) -> np.ndarray:
        """Return a smooth global<-local correction at an output index."""

        query = int(node_id)
        with self._lock:
            if not self._node_order:
                return np.eye(4, dtype=np.float64)
            position = bisect.bisect_left(self._node_order, query)
            if position < len(self._node_order) and (
                self._node_order[position] == query
            ):
                anchor = self._node_order[position]
                return (
                    self._global_poses[anchor]
                    @ inverse_pose(self._local_poses[anchor])
                )
            if position == 0:
                anchor = self._node_order[0]
                return (
                    self._global_poses[anchor]
                    @ inverse_pose(self._local_poses[anchor])
                )
            if position == len(self._node_order):
                anchor = self._node_order[-1]
                return (
                    self._global_poses[anchor]
                    @ inverse_pose(self._local_poses[anchor])
                )

            before = self._node_order[position - 1]
            after = self._node_order[position]
            before_correction = (
                self._global_poses[before]
                @ inverse_pose(self._local_poses[before])
            )
            after_correction = (
                self._global_poses[after]
                @ inverse_pose(self._local_poses[after])
            )
            ratio = (query - before) / float(after - before)
            return interpolate_pose(
                before_correction, after_correction, ratio
            )

    def correct_pose(self, *, node_id: int, local_pose: np.ndarray) -> np.ndarray:
        """Correct a local pose; exact graph nodes use their optimized pose."""

        node = int(node_id)
        local = validated_pose_matrix(local_pose, name="local pose to correct")
        with self._lock:
            exact = self._global_poses.get(node)
            if exact is not None:
                return exact.copy()
        return self.correction_for_index(node) @ local

    def correct_poses(
        self, local_poses: Mapping[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        return {
            int(node_id): self.correct_pose(
                node_id=int(node_id), local_pose=pose
            )
            for node_id, pose in local_poses.items()
        }
