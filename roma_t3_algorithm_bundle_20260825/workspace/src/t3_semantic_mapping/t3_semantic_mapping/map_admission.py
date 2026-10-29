"""Confidence gate for admitting visual keyframes into the 2.5D map.

The gate is deliberately independent of ROS and RoMa internals.  It observes
the public tracking result and the reported camera pose, but it never changes
the localisation output.  Its only decision is whether the current successful
frame may contribute *new* map evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .localization.pose_math import pose_change_norm, validated_pose_matrix


class MapAdmissionState(str, Enum):
    """Visual-map admission state."""

    STARTING = "starting"
    TRACKING = "tracking"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class MapAdmissionConfig:
    """Thresholds used only after visual tracking has failed."""

    enabled: bool = True
    min_recovery_inliers: int = 25
    min_recovery_inlier_ratio: float = 0.25
    stable_frames_required: int = 3
    max_translation_step_m: float = 1.0
    max_rotation_step_deg: float = 20.0

    def __post_init__(self) -> None:
        if self.min_recovery_inliers <= 0:
            raise ValueError("min_recovery_inliers must be positive")
        ratio = float(self.min_recovery_inlier_ratio)
        if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "min_recovery_inlier_ratio must be finite and in [0, 1]"
            )
        if self.stable_frames_required <= 0:
            raise ValueError("stable_frames_required must be positive")
        for name, value in (
            ("max_translation_step_m", self.max_translation_step_m),
            ("max_rotation_step_deg", self.max_rotation_step_deg),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MapAdmissionDecision:
    """Immutable explanation of one successful-frame admission decision."""

    allow_new_observation: bool
    state: MapAdmissionState
    reason: str
    stable_frames: int
    stable_frames_required: int
    inlier_count: int
    correspondence_count: int
    inlier_ratio: float
    translation_step_m: Optional[float]
    rotation_step_deg: Optional[float]
    verified_relocalization: bool


class MapAdmissionGate:
    """Keep low-confidence recovery poses out of the persistent map.

    Normal tracking retains its established behaviour.  Once any visual frame
    fails, the gate requires a configurable number of consecutive, confident,
    motion-continuous successes before new map evidence is admitted again.

    RoMa's public ``reason == 'relocalized'`` result is already protected by
    its geometric verification.  Such a result may therefore seed a new
    recovery reference even when it is far from the last trusted pose.  The
    default three-frame stability requirement still prevents that first
    relocalised keyframe from being written to the map immediately.
    """

    def __init__(self, config: MapAdmissionConfig) -> None:
        self.config = config
        self._state = MapAdmissionState.STARTING
        self._trusted_pose: Optional[np.ndarray] = None
        self._recovery_reference_pose: Optional[np.ndarray] = None
        self._stable_frames = 0

    @property
    def state(self) -> MapAdmissionState:
        return self._state

    @property
    def stable_frames(self) -> int:
        return self._stable_frames

    def record_tracking_failure(self) -> None:
        """Close map admission after any unsuccessful visual frame."""
        if not self.config.enabled:
            return
        self._state = MapAdmissionState.RECOVERING
        self._stable_frames = 0
        self._recovery_reference_pose = None

    def observe_tracking_success(
        self,
        pose: np.ndarray,
        *,
        reason: str,
        inlier_count: int,
        correspondence_count: int,
    ) -> MapAdmissionDecision:
        """Observe a valid visual pose and decide map-evidence admission."""
        current_pose = validated_pose_matrix(pose, name="visual tracking pose")
        tracking_reason = str(reason).strip().lower()
        inliers = max(0, int(inlier_count))
        correspondences = max(0, int(correspondence_count))
        ratio = (
            float(inliers) / float(correspondences)
            if correspondences > 0
            else 0.0
        )
        verified_relocalization = tracking_reason == "relocalized"

        if not self.config.enabled:
            self._trusted_pose = current_pose
            self._recovery_reference_pose = None
            self._stable_frames = 0
            self._state = MapAdmissionState.TRACKING
            return self._decision(
                allow=True,
                reason="disabled",
                inliers=inliers,
                correspondences=correspondences,
                ratio=ratio,
                verified_relocalization=verified_relocalization,
            )

        # There is no older map pose to protect at initialisation.  This also
        # keeps the gate compatible with a different SLAM implementation that
        # does not use RoMa's literal "initialized" reason.
        if self._trusted_pose is None:
            self._trusted_pose = current_pose
            self._recovery_reference_pose = None
            self._stable_frames = 0
            self._state = MapAdmissionState.TRACKING
            return self._decision(
                allow=True,
                reason="initial_pose",
                inliers=inliers,
                correspondences=correspondences,
                ratio=ratio,
                verified_relocalization=verified_relocalization,
            )

        # A verified relocalisation is always re-checked through the recovery
        # gate, even if an upstream implementation did not expose its preceding
        # failure to this class.
        if verified_relocalization and self._state != MapAdmissionState.RECOVERING:
            self._state = MapAdmissionState.RECOVERING
            self._stable_frames = 0
            self._recovery_reference_pose = None

        if self._state != MapAdmissionState.RECOVERING:
            self._trusted_pose = current_pose
            self._stable_frames = 0
            return self._decision(
                allow=True,
                reason="tracking",
                inliers=inliers,
                correspondences=correspondences,
                ratio=ratio,
                verified_relocalization=False,
            )

        reference_pose = (
            self._recovery_reference_pose
            if self._recovery_reference_pose is not None
            else self._trusted_pose
        )
        translation_step_m, rotation_step_rad = pose_change_norm(
            reference_pose, current_pose
        )
        rotation_step_deg = float(np.rad2deg(rotation_step_rad))

        counts_are_consistent = (
            correspondences > 0 and inliers <= correspondences
        )
        confidence_ok = (
            counts_are_consistent
            and inliers >= self.config.min_recovery_inliers
            and ratio >= self.config.min_recovery_inlier_ratio
        )

        if verified_relocalization:
            # The visual algorithm has independently checked PnP support,
            # spatial coverage and reprojection error.  It may establish a new
            # continuity reference across a long outage, but remains blocked
            # until the configured confidence/stability checks complete.
            self._recovery_reference_pose = current_pose
            self._stable_frames = 1 if confidence_ok else 0
        else:
            motion_ok = (
                translation_step_m
                <= self.config.max_translation_step_m
                and rotation_step_deg
                <= self.config.max_rotation_step_deg
            )
            if not motion_ok:
                self._stable_frames = 0
                return self._decision(
                    allow=False,
                    reason="recovery_motion_discontinuity",
                    inliers=inliers,
                    correspondences=correspondences,
                    ratio=ratio,
                    translation_step_m=translation_step_m,
                    rotation_step_deg=rotation_step_deg,
                    verified_relocalization=False,
                )
            if confidence_ok:
                self._recovery_reference_pose = current_pose
                self._stable_frames += 1
            else:
                self._stable_frames = 0

        if not confidence_ok:
            return self._decision(
                allow=False,
                reason="recovery_low_confidence",
                inliers=inliers,
                correspondences=correspondences,
                ratio=ratio,
                translation_step_m=translation_step_m,
                rotation_step_deg=rotation_step_deg,
                verified_relocalization=verified_relocalization,
            )

        if self._stable_frames < self.config.stable_frames_required:
            return self._decision(
                allow=False,
                reason="recovery_stabilizing",
                inliers=inliers,
                correspondences=correspondences,
                ratio=ratio,
                translation_step_m=translation_step_m,
                rotation_step_deg=rotation_step_deg,
                verified_relocalization=verified_relocalization,
            )

        self._state = MapAdmissionState.TRACKING
        self._trusted_pose = current_pose
        self._recovery_reference_pose = None
        return self._decision(
            allow=True,
            reason="recovery_confirmed",
            inliers=inliers,
            correspondences=correspondences,
            ratio=ratio,
            translation_step_m=translation_step_m,
            rotation_step_deg=rotation_step_deg,
            verified_relocalization=verified_relocalization,
        )

    def _decision(
        self,
        *,
        allow: bool,
        reason: str,
        inliers: int,
        correspondences: int,
        ratio: float,
        verified_relocalization: bool,
        translation_step_m: Optional[float] = None,
        rotation_step_deg: Optional[float] = None,
    ) -> MapAdmissionDecision:
        return MapAdmissionDecision(
            allow_new_observation=bool(allow),
            state=self._state,
            reason=str(reason),
            stable_frames=int(self._stable_frames),
            stable_frames_required=int(
                self.config.stable_frames_required
            ),
            inlier_count=int(inliers),
            correspondence_count=int(correspondences),
            inlier_ratio=float(ratio),
            translation_step_m=(
                None
                if translation_step_m is None
                else float(translation_step_m)
            ),
            rotation_step_deg=(
                None
                if rotation_step_deg is None
                else float(rotation_step_deg)
            ),
            verified_relocalization=bool(verified_relocalization),
        )
