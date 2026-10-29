"""Online, out-of-process MPRF loop-candidate runtime.

The package intentionally has no ROS dependency.  The ROS 2 mapper talks to
it through a versioned HTTP/NPZ contract, while all metric acceptance remains
inside the mapper's LiDAR verifier.
"""

from .config import MprfRuntimeConfiguration
from .contracts import KeyframePacket

__all__ = ["KeyframePacket", "MprfRuntimeConfiguration"]

