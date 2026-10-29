"""Optional localisation sources and fusion backends.

The package deliberately has no ROS imports at module import time.  The
existing VO-only mapper can therefore start without GTSAM or an external LIO
node, while ROS adapters remain thin orchestration code in the mapper node.
"""

from .config import (
    FUSION_BACKENDS,
    LOCALIZATION_MODES,
    LocalizationConfiguration,
)
from .coordinator import (
    LocalizationCoordinator,
    LocalizationUnavailableError,
)
from .external_odometry import ExternalOdometryAdapter, FrameMismatchError
from .pose_buffer import PoseBufferQuery, TimedPoseBuffer
from .types import FusionUpdate, PoseEstimate, TimedPose

__all__ = [
    "ExternalOdometryAdapter",
    "FUSION_BACKENDS",
    "FrameMismatchError",
    "FusionUpdate",
    "LOCALIZATION_MODES",
    "LocalizationConfiguration",
    "LocalizationCoordinator",
    "LocalizationUnavailableError",
    "PoseBufferQuery",
    "PoseEstimate",
    "TimedPose",
    "TimedPoseBuffer",
]
