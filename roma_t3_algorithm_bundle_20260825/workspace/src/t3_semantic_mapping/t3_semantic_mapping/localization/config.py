from __future__ import annotations

from dataclasses import dataclass


LOCALIZATION_MODES = frozenset({"vo", "lio", "vo_lio"})
FUSION_BACKENDS = frozenset({"none", "gtsam_gp"})


@dataclass(frozen=True)
class LocalizationConfiguration:
    """Validated selection of localisation source and fusion backend.

    ``vo`` and ``lio`` are direct-source modes and intentionally do not load a
    factor-graph implementation.  ``vo_lio`` owns the local fused trajectory
    and therefore requires the explicit GTSAM backend.
    """

    mode: str = "vo"
    fusion_backend: str = "none"
    enable_gp_prior: bool = True
    allow_vo_only_fusion: bool = True

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        backend = str(self.fusion_backend).strip().lower()
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "fusion_backend", backend)
        object.__setattr__(self, "enable_gp_prior", bool(self.enable_gp_prior))
        object.__setattr__(
            self, "allow_vo_only_fusion", bool(self.allow_vo_only_fusion)
        )

        if mode not in LOCALIZATION_MODES:
            raise ValueError(
                "localization_mode must be one of "
                f"{sorted(LOCALIZATION_MODES)}, got {mode!r}"
            )
        if backend not in FUSION_BACKENDS:
            raise ValueError(
                "fusion_backend must be one of "
                f"{sorted(FUSION_BACKENDS)}, got {backend!r}"
            )
        if mode == "vo_lio" and backend != "gtsam_gp":
            raise ValueError(
                "localization_mode='vo_lio' requires "
                "fusion_backend='gtsam_gp'"
            )
        if mode != "vo_lio" and backend != "none":
            raise ValueError(
                f"fusion_backend={backend!r} is only valid with "
                "localization_mode='vo_lio'"
            )

    @property
    def requires_external_odometry(self) -> bool:
        return self.mode in {"lio", "vo_lio"}

    @property
    def requires_fusion_backend(self) -> bool:
        return self.mode == "vo_lio"
