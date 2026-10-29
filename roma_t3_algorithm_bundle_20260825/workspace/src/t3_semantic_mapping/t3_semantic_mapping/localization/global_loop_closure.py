"""Coordinator for the lightweight global graph and LiDAR safety gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from .global_pose_graph import GlobalPoseGraphUpdate, GtsamGlobalPoseGraph
from .lidar_loop_closure import (
    LidarLoopClosureDetector,
    LidarLoopConfiguration,
    LoopDetectionUpdate,
)
from .loop_constraints import LoopCandidate


@dataclass(frozen=True)
class GlobalLoopClosureUpdate:
    graph: GlobalPoseGraphUpdate
    detection: LoopDetectionUpdate

    @property
    def accepted_loop_count(self) -> int:
        return int(self.graph.loop_constraints_added)

    @property
    def geometry_corrected(self) -> bool:
        return bool(
            self.graph.loop_constraints_added and self.graph.changed_nodes
        )


class GlobalLoopClosure:
    """Own one global graph while leaving the local estimator untouched."""

    def __init__(
        self,
        *,
        detector_configuration: LidarLoopConfiguration,
        graph_options=None,
        detector_archive_directory: Optional[Path] = None,
    ) -> None:
        if detector_archive_directory is not None:
            detector_configuration = replace(
                detector_configuration,
                archive_directory=Path(detector_archive_directory),
            )
        self.detector = LidarLoopClosureDetector(detector_configuration)
        self.graph = GtsamGlobalPoseGraph(**dict(graph_options or {}))

    def register_odometry_keyframe(
        self,
        *,
        node_id: int,
        timestamp: float,
        local_camera_pose: np.ndarray,
    ) -> GlobalPoseGraphUpdate:
        """Append the ordered local support pose before LiDAR may arrive."""

        return self.graph.add_keyframe(
            node_id=node_id,
            timestamp=timestamp,
            local_pose=local_camera_pose,
        )

    def register_lidar_keyframe(
        self,
        *,
        node_id: int,
        keyframe_id: int,
        timestamp: float,
        local_camera_pose: np.ndarray,
        points_sensor: np.ndarray,
        transform_camera_sensor: np.ndarray,
    ) -> GlobalLoopClosureUpdate:
        """Attach metric data to an already ordered graph node."""

        if not self.graph.has_node(node_id):
            raise KeyError(
                f"global graph node {int(node_id)} must be registered before LiDAR"
            )
        detection = self.detector.add_keyframe(
            node_id=node_id,
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            local_camera_pose=local_camera_pose,
            points_sensor=points_sensor,
            transform_camera_sensor=transform_camera_sensor,
        )
        if detection.accepted_constraints:
            graph_update = self.graph.add_loop_constraints(
                detection.accepted_constraints
            )
        else:
            graph_update = GlobalPoseGraphUpdate(
                node_poses={},
                changed_nodes=(),
                loop_constraints_added=0,
            )
        return GlobalLoopClosureUpdate(graph_update, detection)

    def register_keyframe(
        self,
        *,
        node_id: int,
        keyframe_id: int,
        timestamp: float,
        local_camera_pose: np.ndarray,
        points_sensor: np.ndarray,
        transform_camera_sensor: np.ndarray,
    ) -> GlobalLoopClosureUpdate:
        graph_update = self.register_odometry_keyframe(
            node_id=node_id,
            timestamp=timestamp,
            local_camera_pose=local_camera_pose,
        )
        lidar_update = self.register_lidar_keyframe(
            node_id=node_id,
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            local_camera_pose=local_camera_pose,
            points_sensor=points_sensor,
            transform_camera_sensor=transform_camera_sensor,
        )
        if lidar_update.graph.loop_constraints_added:
            graph_update = lidar_update.graph
        return GlobalLoopClosureUpdate(graph_update, lidar_update.detection)

    def submit_candidates(
        self, candidates: Iterable[LoopCandidate]
    ) -> GlobalLoopClosureUpdate:
        detection = self.detector.submit_candidates(candidates)
        if detection.accepted_constraints:
            graph_update = self.graph.add_loop_constraints(
                detection.accepted_constraints
            )
        else:
            graph_update = GlobalPoseGraphUpdate(
                node_poses={},
                changed_nodes=(),
                loop_constraints_added=0,
            )
        return GlobalLoopClosureUpdate(graph_update, detection)

    def submit_candidate(
        self, candidate: LoopCandidate
    ) -> GlobalLoopClosureUpdate:
        return self.submit_candidates((candidate,))

    def correct_pose(self, *, node_id: int, local_pose: np.ndarray) -> np.ndarray:
        return self.graph.correct_pose(node_id=node_id, local_pose=local_pose)

    def correct_poses(
        self, local_poses: Mapping[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        return self.graph.correct_poses(local_poses)
