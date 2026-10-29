"""Validated runtime configuration and bundled model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = 1
SOURCE_BUNDLE_LABEL = "fixed-20260816"
MODEL_SIZES = {
    "finetuned_dinov2_v3.pth": 346_402_202,
    "pretrained_salad.ckpt": 352_040_378,
    "retrained_salad.ckpt": 356_232_514,
    "sonata.pth": 434_008_287,
}


def default_modified_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MprfRuntimeConfiguration:
    """Configuration for retrieval and coarse multimodal pose inference."""

    mprf_source_root: Path = default_modified_root() / "third_party" / "MPRF"
    model_root: Path = default_modified_root() / "models"
    state_root: Path = default_modified_root() / "mprf_runtime" / "state"
    device: str = "cuda:0"
    use_amp: bool = True
    salad_image_size: int = 224
    dino_image_size: int = 224
    pose_image_size: int = 560
    min_keyframe_separation: int = 20
    min_time_separation_sec: float = 20.0
    retrieval_shortlist: int = 20
    refined_shortlist: int = 3
    min_salad_similarity: float = 0.45
    min_dino_similarity: float = 0.45
    salad_score_weight: float = 0.45
    dino_score_weight: float = 0.55
    max_pose_points: int = 2500
    pose_voxel_size_m: float = 0.05
    dino_feature_weight: float = 1.0
    sonata_feature_weight: float = 1.0
    match_mode: str = "mutual"
    match_min_similarity: float = 0.70
    match_min_margin: float = 0.015
    match_max_count: int = 1000
    ransac_iterations: int = 2000
    ransac_residual_m: float = 0.35
    ransac_min_inliers: int = 12
    ransac_min_inlier_ratio: float = 0.10
    ransac_max_rmse_m: float = 0.25
    ransac_min_spread_m: float = 0.30
    coarse_max_translation_m: float = 20.0
    coarse_max_rotation_deg: float = 120.0
    max_pose_cache_records: int = 8
    maximum_request_points: int = 100_000
    maximum_request_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("mprf_source_root", "model_root", "state_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        device = str(self.device).strip()
        if not device:
            raise ValueError("device cannot be empty")
        object.__setattr__(self, "device", device)
        for name in (
            "salad_image_size",
            "dino_image_size",
            "pose_image_size",
        ):
            value = int(getattr(self, name))
            if value <= 0 or value % 14:
                raise ValueError(f"{name} must be a positive multiple of 14")
            object.__setattr__(self, name, value)
        for name in (
            "min_keyframe_separation",
            "retrieval_shortlist",
            "refined_shortlist",
            "max_pose_points",
            "match_max_count",
            "ransac_iterations",
            "ransac_min_inliers",
            "max_pose_cache_records",
            "maximum_request_points",
            "maximum_request_bytes",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.refined_shortlist > self.retrieval_shortlist:
            raise ValueError("refined_shortlist cannot exceed retrieval_shortlist")
        for name in (
            "min_time_separation_sec",
            "pose_voxel_size_m",
            "match_min_margin",
            "ransac_residual_m",
            "ransac_max_rmse_m",
            "ransac_min_spread_m",
            "coarse_max_translation_m",
            "coarse_max_rotation_deg",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "min_salad_similarity",
            "min_dino_similarity",
            "match_min_similarity",
        ):
            value = float(getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
            object.__setattr__(self, name, value)
        ratio = float(self.ransac_min_inlier_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ransac_min_inlier_ratio must be in (0, 1]")
        object.__setattr__(self, "ransac_min_inlier_ratio", ratio)
        weights = (
            float(self.salad_score_weight),
            float(self.dino_score_weight),
            float(self.dino_feature_weight),
            float(self.sonata_feature_weight),
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("feature and retrieval weights cannot be negative")
        if weights[0] + weights[1] <= 0.0:
            raise ValueError("at least one retrieval score weight is required")
        if weights[2] + weights[3] <= 0.0:
            raise ValueError("at least one local feature weight is required")
        object.__setattr__(self, "salad_score_weight", weights[0])
        object.__setattr__(self, "dino_score_weight", weights[1])
        object.__setattr__(self, "dino_feature_weight", weights[2])
        object.__setattr__(self, "sonata_feature_weight", weights[3])
        match_mode = str(self.match_mode).strip().lower()
        if match_mode not in {"mutual", "hungarian"}:
            raise ValueError("match_mode must be mutual or hungarian")
        object.__setattr__(self, "match_mode", match_mode)
        object.__setattr__(self, "use_amp", bool(self.use_amp))

    @property
    def dino_weights(self) -> Path:
        return self.model_root / "mprf" / "Weights" / "finetuned_dinov2_v3.pth"

    @property
    def salad_weights(self) -> Path:
        return self.model_root / "mprf" / "Weights" / "pretrained_salad.ckpt"

    @property
    def sonata_weights(self) -> Path:
        return self.model_root / "sonata" / "sonata.pth"

    @property
    def dinov2_source(self) -> Path:
        return self.mprf_source_root / "Models" / "dinov2"

    @property
    def salad_source(self) -> Path:
        return self.mprf_source_root / "Models" / "salad"

    @property
    def sonata_source(self) -> Path:
        return self.mprf_source_root / "Models" / "sonata"

    def required_assets(self) -> tuple[Path, ...]:
        return (
            self.mprf_source_root / "README.md",
            self.dinov2_source / "hubconf.py",
            self.salad_source / "models" / "aggregators" / "salad.py",
            self.sonata_source / "sonata" / "model.py",
            self.dino_weights,
            self.salad_weights,
            self.sonata_weights,
        )

    def validate_assets(self) -> None:
        missing = [path for path in self.required_assets() if not path.is_file()]
        if missing:
            rendered = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"MPRF runtime assets are missing:\n{rendered}")
