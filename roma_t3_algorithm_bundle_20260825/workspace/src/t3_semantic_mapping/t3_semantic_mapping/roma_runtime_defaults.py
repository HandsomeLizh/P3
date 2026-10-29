"""Authoritative RoMa inference defaults shared by every runtime entry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any


ROMA_RUNTIME_DEFAULTS = MappingProxyType({
    "coarse_res": 280,
    "upsample_res": 280,
    "stereo_upsample_refinement": True,
    "temporal_upsample_refinement": False,
    "symmetric_matching": False,
    "use_custom_corr": True,
    "enable_feature_reuse": True,
    "batch_coarse_pairs": True,
    "enable_klt_temporal_tracking": False,
    "stereo_matches": 5000,
    "temporal_matches": 2000,
})


def launch_default(name: str) -> str:
    """Return a ROS launch-compatible spelling of a shared default."""
    value: Any = ROMA_RUNTIME_DEFAULTS[name]
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
