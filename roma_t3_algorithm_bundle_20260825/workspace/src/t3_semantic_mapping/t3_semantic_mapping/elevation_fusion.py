"""Lightweight height-estimator settings, not a calibrated sensor model.

Variances are in m^2. One cell/batch is one correlated measurement; adding
more points to the same scan must not grant it unbounded confidence.
"""

from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class ElevationFusionConfig:
    fusion_mode: str = "mean"
    measurement_variance: float = 0.04
    lidar_measurement_variance: float = 0.01
    process_variance: float = 0.0001
    min_variance: float = 0.0004
    gate_sigma: float = 3.0
    gate_min_m: float = 0.30

    def __post_init__(self):
        mode = str(self.fusion_mode).strip().lower()
        if mode not in {"mean", "variance"}:
            raise ValueError("elevation_fusion_mode must be mean or variance")
        object.__setattr__(self, "fusion_mode", mode)
        for item in fields(self):
            if item.name == "fusion_mode":
                continue
            value = float(getattr(self, item.name))
            allow_zero = item.name in {"process_variance", "gate_min_m"}
            if not isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
                raise ValueError(f"Invalid elevation_{item.name}: {value}")
            object.__setattr__(self, item.name, value)
        if self.min_variance > min(
            self.measurement_variance, self.lidar_measurement_variance
        ):
            raise ValueError("elevation_min_variance exceeds measurement variance")

    def parameters(self):
        return {"elevation_" + item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_parameters(cls, parameters):
        return cls(**{
            item.name: parameters["elevation_" + item.name]
            for item in fields(cls) if "elevation_" + item.name in parameters
        })


ELEVATION_FUSION_DEFAULTS = ElevationFusionConfig().parameters()
