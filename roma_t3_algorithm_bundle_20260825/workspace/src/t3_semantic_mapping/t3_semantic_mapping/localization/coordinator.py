from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from .config import LocalizationConfiguration
from .external_odometry import ExternalOdometryAdapter, ExternalPoseQuery
from .pose_math import validated_pose_matrix
from .types import FusionUpdate, PoseEstimate


class LocalizationUnavailableError(RuntimeError):
    pass


class LocalizationCoordinator:
    """Select direct VO/LIO poses or invoke the optional fusion backend.

    The class is ROS-independent.  A ROS node only feeds the external adapter
    from ``nav_msgs/Odometry`` and calls :meth:`estimate` at image timestamps.
    """

    def __init__(
        self,
        configuration: LocalizationConfiguration,
        *,
        external_adapter: Optional[ExternalOdometryAdapter] = None,
        external_max_time_offset_sec: float = 0.15,
        fusion_options: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.configuration = configuration
        self.external_adapter = external_adapter
        self.external_max_time_offset_sec = float(
            external_max_time_offset_sec
        )
        if self.external_max_time_offset_sec < 0.0:
            raise ValueError(
                "external_max_time_offset_sec cannot be negative"
            )
        if (
            self.configuration.requires_external_odometry
            and self.external_adapter is None
        ):
            raise ValueError(
                f"localization_mode={self.configuration.mode!r} requires an "
                "ExternalOdometryAdapter"
            )

        self._fusion = None
        if self.configuration.requires_fusion_backend:
            from .gtsam_gp import GtsamGpWnoaFusion

            options = dict(fusion_options or {})
            options["enable_gp_prior"] = self.configuration.enable_gp_prior
            self._fusion = GtsamGpWnoaFusion(**options)
        self._direct_index = 0

    @property
    def uses_factor_graph(self) -> bool:
        return self._fusion is not None

    def query_external(self, timestamp: float) -> Optional[ExternalPoseQuery]:
        if self.external_adapter is None:
            return None
        return self.external_adapter.query(
            timestamp,
            max_time_offset_sec=self.external_max_time_offset_sec,
            align_to_first_query=True,
        )

    def estimate(
        self,
        *,
        timestamp: float,
        vo_pose: Optional[np.ndarray],
        external_query: Optional[ExternalPoseQuery] = None,
    ) -> FusionUpdate:
        sensor_time = float(timestamp)
        mode = self.configuration.mode
        vo = (
            validated_pose_matrix(vo_pose, name="VO pose")
            if vo_pose is not None
            else None
        )

        if mode == "vo":
            if vo is None:
                raise LocalizationUnavailableError(
                    "VO localisation is unavailable for this frame"
                )
            index = self._direct_index
            self._direct_index += 1
            return FusionUpdate(
                estimate=PoseEstimate(
                    timestamp=sensor_time,
                    pose=vo,
                    source="vo",
                ),
                measurement_index=index,
                active_poses={index: vo},
            )

        external = external_query or self.query_external(sensor_time)
        if mode == "lio":
            if external is None:
                raise LocalizationUnavailableError(
                    "No external LIO pose passed the configured time gate"
                )
            index = self._direct_index
            self._direct_index += 1
            return FusionUpdate(
                estimate=PoseEstimate(
                    timestamp=sensor_time,
                    pose=external.pose,
                    covariance=external.covariance,
                    source="lio",
                    diagnostics={
                        "external_query_method": external.method,
                        "external_time_offset_sec": (
                            external.nearest_offset_sec
                        ),
                        "external_bracket_span_sec": (
                            external.bracket_span_sec
                        ),
                    },
                ),
                measurement_index=index,
                active_poses={index: external.pose},
            )

        if external is None and not (
            self.configuration.allow_vo_only_fusion and vo is not None
        ):
            raise LocalizationUnavailableError(
                "VO-LIO fusion has no time-valid LIO pose"
            )
        if external is None and vo is None:
            raise LocalizationUnavailableError(
                "VO-LIO fusion has no valid pose input"
            )
        update = self._fusion.add_measurement(
            timestamp=sensor_time,
            vo_pose=vo,
            lio_pose=None if external is None else external.pose,
        )
        if external is None:
            diagnostics = dict(update.estimate.diagnostics)
            diagnostics["lio_fallback"] = "vo_only"
            return FusionUpdate(
                estimate=PoseEstimate(
                    timestamp=update.estimate.timestamp,
                    pose=update.estimate.pose,
                    source=update.estimate.source,
                    velocity_body=update.estimate.velocity_body,
                    covariance=update.estimate.covariance,
                    diagnostics=diagnostics,
                ),
                measurement_index=update.measurement_index,
                active_poses=update.active_poses,
            )
        return update
