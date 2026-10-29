from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Tuple

import numpy as np


TileKey = Tuple[int, int]


class MapContributionIndex:
    """Applied keyframe poses and the map tiles each one can modify.

    The index contains no map values and is cheap to clone for a candidate
    transaction.  Reverse entries let a corrected tile be rebuilt from every
    observation that contributes to it, preserving deterministic replay and
    every map layer.
    """

    def __init__(self) -> None:
        self._poses: dict[int, np.ndarray] = {}
        self._keyframe_tiles: dict[int, frozenset[TileKey]] = {}
        self._tile_keyframes: dict[TileKey, set[int]] = {}

    def __len__(self) -> int:
        return len(self._poses)

    @property
    def keyframe_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._poses))

    def clone(self) -> "MapContributionIndex":
        candidate = MapContributionIndex()
        candidate._poses = {
            keyframe_id: pose.copy()
            for keyframe_id, pose in self._poses.items()
        }
        candidate._keyframe_tiles = dict(self._keyframe_tiles)
        candidate._tile_keyframes = {
            key: set(keyframe_ids)
            for key, keyframe_ids in self._tile_keyframes.items()
        }
        return candidate

    @staticmethod
    def _validated_pose(pose_world_camera: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_world_camera, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid contribution-index pose: {pose.shape}")
        return pose.copy()

    @staticmethod
    def _normalized_tiles(tile_keys: Iterable[TileKey]) -> frozenset[TileKey]:
        return frozenset(
            (int(key[0]), int(key[1]))
            for key in tile_keys
        )

    def record(
        self,
        *,
        keyframe_id: int,
        pose_world_camera: np.ndarray,
        tile_keys: Iterable[TileKey],
    ) -> None:
        keyframe_id = int(keyframe_id)
        pose = self._validated_pose(pose_world_camera)
        normalized_tiles = self._normalized_tiles(tile_keys)

        previous_tiles = self._keyframe_tiles.get(keyframe_id, frozenset())
        for key in previous_tiles - normalized_tiles:
            contributors = self._tile_keyframes[key]
            contributors.remove(keyframe_id)
            if not contributors:
                del self._tile_keyframes[key]
        for key in normalized_tiles - previous_tiles:
            self._tile_keyframes.setdefault(key, set()).add(keyframe_id)

        self._poses[keyframe_id] = pose
        self._keyframe_tiles[keyframe_id] = normalized_tiles

    def record_many(
        self,
        *,
        poses_world_camera: Mapping[int, np.ndarray],
        tile_keys_by_keyframe: Mapping[int, Iterable[TileKey]],
    ) -> None:
        if set(poses_world_camera) != set(tile_keys_by_keyframe):
            raise ValueError(
                "Contribution-index pose and tile keyframe sets differ"
            )
        for keyframe_id in sorted(poses_world_camera):
            self.record(
                keyframe_id=keyframe_id,
                pose_world_camera=poses_world_camera[keyframe_id],
                tile_keys=tile_keys_by_keyframe[keyframe_id],
            )

    def is_complete_for(self, keyframe_ids: Iterable[int]) -> bool:
        expected = {int(keyframe_id) for keyframe_id in keyframe_ids}
        return expected == set(self._poses)

    def pose(self, keyframe_id: int) -> Optional[np.ndarray]:
        value = self._poses.get(int(keyframe_id))
        return None if value is None else value.copy()

    def tile_keys(self, keyframe_id: int) -> frozenset[TileKey]:
        return self._keyframe_tiles.get(int(keyframe_id), frozenset())

    def changed_keyframes(
        self,
        *,
        keyframe_ids: Iterable[int],
        current_poses: Mapping[int, np.ndarray],
    ) -> tuple[int, ...]:
        changed = []
        for keyframe_id in sorted(int(value) for value in keyframe_ids):
            current = self._validated_pose(current_poses[keyframe_id])
            applied = self._poses.get(keyframe_id)
            if applied is None or not np.array_equal(applied, current):
                changed.append(keyframe_id)
        return tuple(changed)

    def contributors(self, tile_keys: Iterable[TileKey]) -> tuple[int, ...]:
        result: set[int] = set()
        for key in self._normalized_tiles(tile_keys):
            result.update(self._tile_keyframes.get(key, ()))
        return tuple(sorted(result))

    def assert_consistent(self) -> None:
        if set(self._poses) != set(self._keyframe_tiles):
            raise RuntimeError("Contribution index has incomplete keyframes")
        expected_reverse: dict[TileKey, set[int]] = {}
        for keyframe_id, tile_keys in self._keyframe_tiles.items():
            for key in tile_keys:
                expected_reverse.setdefault(key, set()).add(keyframe_id)
        if expected_reverse != self._tile_keyframes:
            raise RuntimeError("Contribution index reverse entries are stale")
