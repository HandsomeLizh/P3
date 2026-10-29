"""Normalise RoMa tracking output without ROS or map dependencies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import numpy as np


def visual_frontend_required(
    *,
    localization_mode: str,
    enable_lidar_mapping: bool,
    lidar_mapping_mode: str,
    lidar_fallback_to_stereo: bool,
) -> bool:
    """Return whether the selected mode consumes RoMa output.

    Direct LIO already owns both the pose and map geometry.  It needs the
    visual frontend only when stereo fallback is explicitly requested.  Every
    other supported combination still consumes visual pose, keyframes, or
    stereo geometry and therefore keeps the established frontend path.
    """

    return not (
        str(localization_mode).strip().lower() == "lio"
        and bool(enable_lidar_mapping)
        and str(lidar_mapping_mode).strip().lower() == "direct"
        and not bool(lidar_fallback_to_stereo)
    )


def process_visual_tracking(
    slam: object,
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    visual_required: bool,
    frame_index: int,
    geometry_revision: int,
    enabled: bool = True,
) -> Tuple[np.ndarray, object]:
    """Run one visual frame and return its pose and public tracking status.

    Pure-VO operation requires every visual contract to succeed and therefore
    preserves the original exception. External-localisation modes instead
    receive a synthetic failed status, allowing their independent pose source
    to continue while preventing visual map evidence from being accepted.
    """

    if not enabled:
        if visual_required:
            raise ValueError(
                "The visual frontend cannot be bypassed when visual pose "
                "output is required"
            )
        return np.empty((0, 0), dtype=np.float64), SimpleNamespace(
            success=False,
            attempted=False,
            frame_index=int(frame_index),
            pose_index=-1,
            keyframe_created=False,
            keyframe_id=None,
            geometry_corrected=False,
            geometry_revision=int(geometry_revision),
            reason="visual frontend intentionally bypassed",
            inlier_count=0,
            correspondence_count=0,
        )

    visual_exception = None
    try:
        raw_pose = np.asarray(
            slam.process_frame(left_image, right_image),
            dtype=np.float64,
        )
    except Exception as exception:
        if visual_required:
            raise
        visual_exception = exception
        raw_pose = np.empty((0, 0), dtype=np.float64)

    pose_is_valid = (
        raw_pose.shape == (4, 4) and np.isfinite(raw_pose).all()
    )
    if not pose_is_valid and visual_required:
        raise ValueError(
            f"Invalid pose returned by algorithm: {raw_pose}"
        )
    if not pose_is_valid and visual_exception is None:
        visual_exception = ValueError(
            "visual algorithm returned an invalid 4x4 pose"
        )

    status_getter = getattr(slam, "get_tracking_status", None)
    if visual_exception is None:
        if not callable(status_getter):
            status_exception = RuntimeError(
                "The configured SLAM class does not expose "
                "get_tracking_status()"
            )
            if visual_required:
                raise status_exception
            visual_exception = status_exception
        else:
            try:
                tracking_status = status_getter()
            except Exception as exception:
                if visual_required:
                    raise
                visual_exception = exception

    if visual_exception is not None:
        tracking_status = SimpleNamespace(
            success=False,
            frame_index=int(frame_index),
            pose_index=-1,
            keyframe_created=False,
            keyframe_id=None,
            geometry_corrected=False,
            geometry_revision=int(geometry_revision),
            reason=(
                "visual processing exception: "
                f"{type(visual_exception).__name__}: "
                f"{visual_exception}"
            ),
            inlier_count=0,
            correspondence_count=0,
        )

    return raw_pose, tracking_status
