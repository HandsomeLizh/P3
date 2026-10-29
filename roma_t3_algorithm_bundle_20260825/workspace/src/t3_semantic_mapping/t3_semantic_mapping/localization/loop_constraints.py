"""Shared, ROS-independent contract for global loop closure.

The local fixed-lag estimator and every loop-candidate provider meet at this
small contract.  A provider may be the built-in LiDAR Scan Context detector or
an out-of-process foundation model.  Candidates are *never* graph factors:
they become :class:`LoopConstraint` objects only after metric LiDAR
verification.

Pose convention
---------------
``relative_pose_reference_query`` is ``T_reference_query``.  It maps a point
from the query camera frame into the reference camera frame and therefore
matches ``gtsam.Pose3.between(reference, query)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from .pose_math import validated_pose_matrix


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_pose(value: np.ndarray, *, name: str) -> np.ndarray:
    result = validated_pose_matrix(value, name=name)
    result.setflags(write=False)
    return result


def _readonly_sigmas(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(6).copy()
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("loop sigmas must contain six positive finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LoopCandidate:
    """Untrusted place-recognition output awaiting geometric verification."""

    reference_node: int
    query_node: int
    source: str
    score: float
    coarse_relative_pose: Optional[np.ndarray] = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reference = int(self.reference_node)
        query = int(self.query_node)
        if reference == query:
            raise ValueError("a loop candidate must connect two distinct nodes")
        source = str(self.source).strip()
        if not source:
            raise ValueError("loop candidate source cannot be empty")
        score = _finite_scalar(self.score, name="loop candidate score")
        coarse = self.coarse_relative_pose
        if coarse is not None:
            coarse = _readonly_pose(coarse, name="coarse loop pose")
        object.__setattr__(self, "reference_node", reference)
        object.__setattr__(self, "query_node", query)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "coarse_relative_pose", coarse)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


@dataclass(frozen=True)
class LoopConstraint:
    """Metric loop factor accepted by the LiDAR verification gate."""

    reference_node: int
    query_node: int
    relative_pose_reference_query: np.ndarray
    sigmas: np.ndarray
    source: str
    confidence: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reference = int(self.reference_node)
        query = int(self.query_node)
        if reference == query:
            raise ValueError("a loop constraint must connect distinct nodes")
        source = str(self.source).strip()
        if not source:
            raise ValueError("loop constraint source cannot be empty")
        confidence = _finite_scalar(
            self.confidence, name="loop constraint confidence"
        )
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("loop constraint confidence must be in [0, 1]")
        object.__setattr__(self, "reference_node", reference)
        object.__setattr__(self, "query_node", query)
        object.__setattr__(
            self,
            "relative_pose_reference_query",
            _readonly_pose(
                self.relative_pose_reference_query,
                name="verified loop pose",
            ),
        )
        object.__setattr__(self, "sigmas", _readonly_sigmas(self.sigmas))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def directed_key(self) -> tuple[int, int]:
        return self.reference_node, self.query_node

