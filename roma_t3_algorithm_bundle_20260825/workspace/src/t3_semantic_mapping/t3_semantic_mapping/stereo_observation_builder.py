"""Build immutable stereo map observations from read-only keyframe state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .lidar_mapping import assist_stereo_depth_from_pointcloud
from .semantic_map_rebuilder import SemanticKeyFrameObservation


@dataclass(frozen=True)
class StereoObservationConfiguration:
    """Point selection and optional PointCloud2-assist policy."""

    visible_max_points: int = 30000
    stereo_map_max_depth_m: float = 5.0
    assist_pixel_radius: float = 2.0
    assist_unmatched_stereo_max_depth_m: float = 5.0
    assist_replace_stereo_min_depth_m: float = 5.0
    assist_min_matches: int = 20

    def __post_init__(self) -> None:
        visible_maximum = int(self.visible_max_points)
        stereo_map_depth = float(self.stereo_map_max_depth_m)
        pixel_radius = float(self.assist_pixel_radius)
        unmatched_depth = float(
            self.assist_unmatched_stereo_max_depth_m
        )
        replacement_depth = float(
            self.assist_replace_stereo_min_depth_m
        )
        minimum_matches = int(self.assist_min_matches)

        object.__setattr__(self, "visible_max_points", visible_maximum)
        object.__setattr__(
            self, "stereo_map_max_depth_m", stereo_map_depth
        )
        object.__setattr__(self, "assist_pixel_radius", pixel_radius)
        object.__setattr__(
            self,
            "assist_unmatched_stereo_max_depth_m",
            unmatched_depth,
        )
        object.__setattr__(
            self,
            "assist_replace_stereo_min_depth_m",
            replacement_depth,
        )
        object.__setattr__(self, "assist_min_matches", minimum_matches)

        if not np.isfinite(stereo_map_depth) or stereo_map_depth < 0.0:
            raise ValueError(
                "stereo_map_max_depth_m must be finite and non-negative"
            )
        if not np.isfinite(pixel_radius) or pixel_radius < 0.0:
            raise ValueError(
                "pointcloud_assist_pixel_radius must be finite and "
                "non-negative"
            )
        if not np.isfinite(unmatched_depth):
            raise ValueError(
                "pointcloud_assist_unmatched_stereo_max_depth_m must be "
                "finite"
            )
        if not np.isfinite(replacement_depth):
            raise ValueError(
                "pointcloud_assist_replace_stereo_min_depth_m must be finite"
            )
        if minimum_matches <= 0:
            raise ValueError(
                "pointcloud_assist_min_matches must be positive"
            )


class StereoObservationBuilder:
    """Convert a RoMa keyframe snapshot into replayable map evidence.

    The source is read only and is expected to expose
    ``get_mapping_keyframe_snapshot(keyframe_id)``.  This component owns no
    ROS publishers, retry state, map transaction, or mutable SLAM state.
    """

    def __init__(
        self,
        snapshot_source: object,
        configuration: StereoObservationConfiguration = (
            StereoObservationConfiguration()
        ),
    ) -> None:
        if not isinstance(configuration, StereoObservationConfiguration):
            raise TypeError(
                "configuration must be StereoObservationConfiguration"
            )
        self.snapshot_source = snapshot_source
        self.configuration = configuration

    def components(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        maximum_depth_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned keypoints, camera points, and semantic labels."""

        snapshot_getter = getattr(
            self.snapshot_source,
            "get_mapping_keyframe_snapshot",
            None,
        )
        if not callable(snapshot_getter):
            raise RuntimeError(
                "The configured SLAM class does not expose "
                "get_mapping_keyframe_snapshot()"
            )
        snapshot = snapshot_getter(int(keyframe_id))
        if snapshot is None:
            raise RuntimeError(f"Missing keyframe snapshot: {keyframe_id}")
        if bool(snapshot.is_bad):
            raise RuntimeError(
                f"Newly created keyframe {keyframe_id} is already bad"
            )

        keypoints = np.asarray(
            snapshot.keypoints,
            dtype=np.float64,
        ).reshape(-1, 2)
        camera_points = np.asarray(
            snapshot.camera_points,
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(keypoints) != len(camera_points):
            raise ValueError(
                f"Keyframe {keyframe_id} snapshot is not aligned: "
                f"{len(keypoints)} != {len(camera_points)}"
            )

        semantic = np.asarray(semantic_image)
        height, width = semantic.shape[:2]
        u_integer = np.rint(keypoints[:, 0]).astype(np.int64)
        v_integer = np.rint(keypoints[:, 1]).astype(np.int64)
        valid = (
            np.isfinite(keypoints).all(axis=1)
            & np.isfinite(camera_points).all(axis=1)
            & (camera_points[:, 2] > 0.1)
            & (u_integer >= 0)
            & (u_integer < width)
            & (v_integer >= 0)
            & (v_integer < height)
        )
        if maximum_depth_m is not None and float(maximum_depth_m) > 0.0:
            valid &= camera_points[:, 2] <= float(maximum_depth_m)
        candidate_indices = np.where(valid)[0]

        # Preserve the established nearest-depth-per-pixel ordering exactly.
        if len(candidate_indices):
            pixel_indices = (
                v_integer[candidate_indices] * width
                + u_integer[candidate_indices]
            )
            order = np.argsort(camera_points[candidate_indices, 2])
            ordered_pixels = pixel_indices[order]
            _, first_positions = np.unique(
                ordered_pixels,
                return_index=True,
            )
            selected = candidate_indices[order[first_positions]]
        else:
            selected = np.zeros(0, dtype=np.int64)

        maximum = self.configuration.visible_max_points
        if maximum > 0 and len(selected) > maximum:
            selected = selected[
                np.linspace(
                    0,
                    len(selected) - 1,
                    maximum,
                    dtype=np.int64,
                )
            ]

        labels = semantic[
            v_integer[selected],
            u_integer[selected],
        ].astype(np.uint8)
        return keypoints[selected], camera_points[selected], labels

    def make_stereo(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        sensor_timestamp: float,
    ) -> Optional[SemanticKeyFrameObservation]:
        """Build pure-stereo map evidence, or ``None`` when none is safe.

        This map-only depth gate never mutates the keyframe or removes far
        features used by visual localisation. A zero limit explicitly keeps
        the previous unbounded map-observation behaviour.
        """

        _, camera_points, labels = self.components(
            keyframe_id=keyframe_id,
            semantic_image=semantic_image,
            maximum_depth_m=self.configuration.stereo_map_max_depth_m,
        )
        if not len(camera_points):
            return None
        return SemanticKeyFrameObservation(
            keyframe_id=int(keyframe_id),
            sensor_timestamp=float(sensor_timestamp),
            camera_points=camera_points,
            semantic_labels=labels,
        )

    def make_fallback(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        sensor_timestamp: float,
    ) -> Tuple[SemanticKeyFrameObservation, Dict[str, float]]:
        """Build the explicit near-stereo fallback for assist mode."""

        _, camera_points, labels = self.components(
            keyframe_id=keyframe_id,
            semantic_image=semantic_image,
        )
        input_count = len(camera_points)
        maximum_depth = (
            self.configuration.assist_unmatched_stereo_max_depth_m
        )
        if maximum_depth > 0.0:
            keep = camera_points[:, 2] <= maximum_depth
            camera_points = camera_points[keep]
            labels = labels[keep]
        if not len(camera_points):
            raise RuntimeError(
                "PointCloud2-assisted fallback has no stereo points inside "
                f"the reliable depth range for keyframe {keyframe_id}"
            )

        observation = SemanticKeyFrameObservation(
            keyframe_id=int(keyframe_id),
            sensor_timestamp=float(sensor_timestamp),
            camera_points=camera_points,
            semantic_labels=labels,
            source="pointcloud_assisted_stereo_fallback",
        )
        return observation, {
            "input_stereo_points": int(input_count),
            "output_points": int(len(camera_points)),
            "maximum_depth_m": float(maximum_depth),
        }

    def make_assisted(
        self,
        *,
        keyframe_id: int,
        semantic_image: np.ndarray,
        sensor_timestamp: float,
        pointcloud_camera_points: np.ndarray,
        camera_matrix: np.ndarray,
        calibration_width: int,
        calibration_height: int,
    ) -> Tuple[SemanticKeyFrameObservation, Dict[str, float]]:
        """Build a stereo observation corrected by time-aligned cloud depth."""

        keypoints, stereo_points, labels = self.components(
            keyframe_id=keyframe_id,
            semantic_image=semantic_image,
        )
        height, width = semantic_image.shape[:2]
        matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
        scale_x = width / float(calibration_width)
        scale_y = height / float(calibration_height)
        matrix[0, 0] *= scale_x
        matrix[0, 2] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[1, 2] *= scale_y

        assisted = assist_stereo_depth_from_pointcloud(
            keypoints=keypoints,
            stereo_camera_points=stereo_points,
            pointcloud_camera_points=pointcloud_camera_points,
            camera_matrix=matrix,
            image_width=width,
            image_height=height,
            pixel_radius=self.configuration.assist_pixel_radius,
            unmatched_stereo_max_depth_m=(
                self.configuration.assist_unmatched_stereo_max_depth_m
            ),
            replace_stereo_min_depth_m=(
                self.configuration.assist_replace_stereo_min_depth_m
            ),
        )
        supported = int(
            assisted.statistics["pointcloud_supported_points"]
        )
        if supported < self.configuration.assist_min_matches:
            raise RuntimeError(
                f"PointCloud2-assisted keyframe {keyframe_id} matched only "
                f"{supported} stereo points; minimum="
                f"{self.configuration.assist_min_matches}"
            )

        observation = SemanticKeyFrameObservation(
            keyframe_id=int(keyframe_id),
            sensor_timestamp=float(sensor_timestamp),
            camera_points=assisted.camera_points,
            semantic_labels=labels[assisted.source_indices],
            source="pointcloud_assisted_stereo",
        )
        statistics = dict(assisted.statistics)
        statistics["input_stereo_points"] = int(len(stereo_points))
        return observation, statistics
