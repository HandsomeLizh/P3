#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.map_contribution_index import MapContributionIndex
from t3_semantic_mapping.semantic_map_rebuilder import (
    KeyFrameObservationStore,
    SemanticKeyFrameObservation,
)
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


def _grid() -> TiledSemanticMapManager:
    return TiledSemanticMapManager(
        resolution=1.0,
        tile_cells=8,
        max_rays_per_update=100,
    )


def _pose(x: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = float(x)
    return result


def _observation(keyframe_id: int) -> SemanticKeyFrameObservation:
    points = np.array(
        [
            [0.5, 0.1, 1.0 + 0.01 * keyframe_id],
            [1.2, 0.2, 1.2 + 0.01 * keyframe_id],
            [1.8, 0.4, 0.8 + 0.01 * keyframe_id],
        ],
        dtype=np.float64,
    )
    labels = np.array(
        [1, 3 + keyframe_id % 2, 4],
        dtype=np.uint8,
    )
    return SemanticKeyFrameObservation(
        keyframe_id=keyframe_id,
        sensor_timestamp=float(keyframe_id),
        camera_points=points,
        semantic_labels=labels,
    )


def _assert_fusion_equal(
    test: unittest.TestCase,
    actual: TiledSemanticMapManager,
    expected: TiledSemanticMapManager,
) -> None:
    test.assertEqual(set(actual.tiles), set(expected.tiles))
    test.assertEqual(actual.known_cells, expected.known_cells)
    test.assertEqual(actual.surface_cells, expected.surface_cells)
    test.assertEqual(
        actual.total_accepted_points,
        expected.total_accepted_points,
    )
    for key in expected.tiles:
        actual_tile = actual.tiles[key]
        expected_tile = expected.tiles[key]
        for name, _dtype in expected._SNAPSHOT_TILE_FIELDS:
            np.testing.assert_array_equal(
                getattr(actual_tile, name),
                getattr(expected_tile, name),
                err_msg=f"tile={key}, layer={name}",
            )


class PartialCorrectedMapTest(unittest.TestCase):
    def setUp(self):
        self.rotation = np.eye(3, dtype=np.float64)
        self.offset = np.zeros(3, dtype=np.float64)
        self.store = KeyFrameObservationStore()
        self.poses = {}
        for keyframe_id in range(30):
            self.store.add(_observation(keyframe_id))
            # Six observations deliberately overlap near the origin; the
            # rest are isolated so the affected-subgraph path is exercised.
            x = float(keyframe_id) if keyframe_id < 6 else keyframe_id * 20.0
            self.poses[keyframe_id] = _pose(x)

        tile_keys = {}
        self.live, _ = self.store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv=self.poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
            tile_keys_by_keyframe=tile_keys,
        )
        self.index = MapContributionIndex()
        self.index.record_many(
            poses_world_camera=self.poses,
            tile_keys_by_keyframe=tile_keys,
        )
        self.live.mark_tiles_published(
            self.live.pending_dirty_tile_keys()
        )

    def _corrected_candidate(self, poses, pending=()):
        return self.store.build_corrected_candidate(
            live_grid=self.live,
            grid_factory=_grid,
            contribution_index=self.index,
            pending_observations=pending,
            keyframe_poses_cv=poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

    def test_local_rebuild_is_bit_exact_and_keeps_unaffected_tiles_shared(self):
        corrected_poses = dict(self.poses)
        corrected_poses[2] = _pose(17.0)
        old_unaffected = self.live.tiles[(50, 0)]

        candidate, statistics, candidate_index = (
            self._corrected_candidate(corrected_poses)
        )
        expected, _ = self.store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv=corrected_poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

        self.assertEqual(statistics["rebuild_strategy"], "partial")
        self.assertGreater(statistics["saved_fusion_operations"], 0)
        self.assertIs(candidate.tiles[(50, 0)], old_unaffected)
        self.assertLess(
            len(candidate.pending_dirty_tile_keys()),
            candidate.tile_count,
        )
        self.assertEqual(
            candidate_index.changed_keyframes(
                keyframe_ids=self.store.keyframe_ids,
                current_poses=corrected_poses,
            ),
            (),
        )
        _assert_fusion_equal(self, candidate, expected)

    def test_pending_observation_is_included_in_map_and_index(self):
        corrected_poses = dict(self.poses)
        corrected_poses[2] = _pose(17.0)
        pending = _observation(100)
        corrected_poses[100] = _pose(800.0)

        candidate, statistics, candidate_index = (
            self._corrected_candidate(corrected_poses, (pending,))
        )
        expected, _ = self.store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv=corrected_poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )
        self.store.fuse_observations(
            grid=expected,
            observations=(pending,),
            keyframe_poses_cv=corrected_poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

        self.assertEqual(statistics["rebuild_strategy"], "partial")
        self.assertTrue(
            candidate_index.is_complete_for(
                self.store.keyframe_ids + (100,)
            )
        )
        _assert_fusion_equal(self, candidate, expected)

    def test_large_correction_uses_original_full_rebuild_path(self):
        corrected_poses = dict(self.poses)
        for keyframe_id in range(16):
            corrected_poses[keyframe_id] = _pose(1000.0 + keyframe_id)

        candidate, statistics, candidate_index = (
            self._corrected_candidate(corrected_poses)
        )
        expected, _ = self.store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv=corrected_poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

        self.assertEqual(statistics["rebuild_strategy"], "full")
        self.assertEqual(
            statistics["rebuild_reason"],
            "changed_observation_cost",
        )
        self.assertTrue(
            candidate_index.is_complete_for(self.store.keyframe_ids)
        )
        _assert_fusion_equal(self, candidate, expected)

    def test_missing_index_is_rebuilt_by_safe_full_fallback(self):
        candidate, statistics, candidate_index = (
            self.store.build_corrected_candidate(
                live_grid=self.live,
                grid_factory=_grid,
                contribution_index=MapContributionIndex(),
                pending_observations=(),
                keyframe_poses_cv=self.poses,
                rotation_map_world=self.rotation,
                map_translation_offset=self.offset,
            )
        )

        self.assertEqual(statistics["rebuild_strategy"], "full")
        self.assertEqual(
            statistics["rebuild_reason"],
            "dependency_index_incomplete",
        )
        self.assertTrue(
            candidate_index.is_complete_for(self.store.keyframe_ids)
        )
        _assert_fusion_equal(self, candidate, self.live)

    def test_recovered_historical_observation_is_inserted_in_time_order(self):
        template = _observation(100)
        historical = SemanticKeyFrameObservation(
            keyframe_id=100,
            sensor_timestamp=4.5,
            camera_points=template.camera_points,
            semantic_labels=template.semantic_labels,
        )
        poses = dict(self.poses)
        poses[100] = _pose(4.5)

        candidate, statistics, candidate_index = (
            self._corrected_candidate(poses, (historical,))
        )
        expected, _ = self.store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv=poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
            additional_observations=(historical,),
        )

        self.assertEqual(statistics["rebuild_strategy"], "full")
        self.assertEqual(
            statistics["rebuild_reason"],
            "historical_pending_observation",
        )
        self.assertTrue(
            candidate_index.is_complete_for(
                self.store.keyframe_ids + (100,)
            )
        )
        _assert_fusion_equal(self, candidate, expected)


if __name__ == "__main__":
    unittest.main()
