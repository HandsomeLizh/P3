#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.semantic_grid import LayeredSemanticGridMap
from t3_semantic_mapping.tiled_semantic_map import (
    TiledSemanticMapManager,
    _bresenham_cells,
)
from t3_semantic_mapping.lidar_occupancy import (
    ENDPOINT_FREE,
    ENDPOINT_OCCUPIED,
    ENDPOINT_UNKNOWN,
)
from t3_semantic_mapping.semantic_map_rebuilder import (
    KeyFrameObservationStore,
    SemanticKeyFrameObservation,
)


def _manager(**kwargs) -> TiledSemanticMapManager:
    return TiledSemanticMapManager(
        resolution=1.0,
        tile_cells=4,
        max_rays_per_update=100,
        **kwargs,
    )


class TiledSemanticMapTest(unittest.TestCase):
    def test_sparse_elevation_fast_path_preserves_robust_statistics(self):
        grid = LayeredSemanticGridMap(
            resolution=1.0,
            length_x=2.0,
            length_y=2.0,
        )

        self.assertEqual(
            grid._update_elevation_cell(0, 0, np.array([2.0])),
            1,
        )
        self.assertEqual(
            grid._update_elevation_cell(0, 0, np.array([0.0, 4.0])),
            2,
        )
        self.assertEqual(grid.elevation_count[0, 0], 3)
        self.assertEqual(grid.elevation_mean[0, 0], 2.0)
        self.assertEqual(grid.elevation_M2[0, 0], 8.0)
        self.assertEqual(grid.elevation_min[0, 0], 0.0)
        self.assertEqual(grid.elevation_max[0, 0], 4.0)

        low_scale = LayeredSemanticGridMap(
            resolution=1.0,
            length_x=2.0,
            length_y=2.0,
            elevation_mad_scale=0.1,
            min_elevation_mad=0.01,
        )
        self.assertEqual(
            low_scale._update_elevation_cell(
                0,
                0,
                np.array([0.0, 4.0]),
            ),
            1,
        )
        self.assertEqual(low_scale.elevation_count[0, 0], 1)
        self.assertEqual(low_scale.elevation_mean[0, 0], 2.0)
        self.assertEqual(low_scale.elevation_M2[0, 0], 0.0)
        self.assertEqual(low_scale.elevation_min[0, 0], 2.0)
        self.assertEqual(low_scale.elevation_max[0, 0], 2.0)

    def test_stable_tile_grouping_preserves_every_ray_update(self):
        manager = _manager()
        points = np.array(
            [
                [6.2, 0.2, 1.0],
                [-5.8, 0.2, 1.0],
                [0.2, 5.2, 1.0],
                [0.2, -4.8, 1.0],
                [6.2, 0.2, 1.0],
            ]
        )
        labels = np.array([4, 4, 1, 4, 1], dtype=np.uint8)
        camera = np.array([0.2, 0.2, 0.0])

        manager.update(
            points_map=points,
            semantic_labels=labels,
            camera_position_map=camera,
        )

        expected: dict[tuple[int, int], np.float32] = {}
        camera_row = 0
        camera_column = 0
        for point, semantic_id in zip(points, labels):
            endpoint_row = int(np.floor(point[1]))
            endpoint_column = int(np.floor(point[0]))
            cells = _bresenham_cells(
                camera_column,
                camera_row,
                endpoint_column,
                endpoint_row,
            )
            for row, column in cells[:-1]:
                key = (int(row), int(column))
                expected[key] = np.float32(
                    expected.get(key, np.float32(0.0))
                    + np.float32(-0.4)
                )
            endpoint_key = (int(cells[-1, 0]), int(cells[-1, 1]))
            endpoint_increment = 0.85 if semantic_id == 4 else -0.4
            expected[endpoint_key] = np.float32(
                expected.get(endpoint_key, np.float32(0.0))
                + np.float32(endpoint_increment)
            )

        actual: dict[tuple[int, int], np.float32] = {}
        for tile_key, tile in manager.tiles.items():
            rows, columns = np.nonzero(tile.occupancy_observed)
            for local_row, local_column in zip(rows, columns):
                global_row = tile_key[1] * manager.tile_cells + int(local_row)
                global_column = (
                    tile_key[0] * manager.tile_cells + int(local_column)
                )
                actual[(global_row, global_column)] = (
                    tile.occupancy_log_odds[local_row, local_column]
                )

        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            self.assertEqual(actual[key], value)

    def test_window_geometry_changes_only_at_cell_boundaries(self):
        manager = _manager(origin_x=-2.0, origin_y=-2.0)

        first = manager.window_geometry(
            center_x=0.10,
            center_y=0.10,
            length_x=4.0,
            length_y=4.0,
        )
        same_cell = manager.window_geometry(
            center_x=0.99,
            center_y=0.99,
            length_x=4.0,
            length_y=4.0,
        )
        next_cell = manager.window_geometry(
            center_x=1.00,
            center_y=1.00,
            length_x=4.0,
            length_y=4.0,
        )

        self.assertEqual(first, same_cell)
        self.assertAlmostEqual(
            next_cell.origin_x - first.origin_x,
            manager.resolution,
        )
        self.assertAlmostEqual(
            next_cell.origin_y - first.origin_y,
            manager.resolution,
        )

    def test_rebuilder_accepts_tiled_manager_without_dense_arrays(self):
        manager = _manager()
        observation = SemanticKeyFrameObservation(
            keyframe_id=3,
            sensor_timestamp=12.5,
            camera_points=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
            semantic_labels=np.array([4], dtype=np.uint8),
        )
        candidate, statistics = KeyFrameObservationStore().build_incremental_candidate(
            live_grid=manager,
            observations=[observation],
            keyframe_poses_cv={3: np.eye(4, dtype=np.float64)},
            rotation_map_world=np.eye(3, dtype=np.float64),
            map_translation_offset=np.zeros(3, dtype=np.float64),
        )

        self.assertEqual(manager.known_cells, 0)
        self.assertGreaterEqual(candidate.surface_cells, 1)
        self.assertEqual(statistics["known_cells"], candidate.known_cells)

    def test_positive_and_negative_coordinates_create_expected_tiles(self):
        manager = _manager()
        statistics = manager.update(
            points_map=np.array(
                [
                    [0.2, 0.2, 1.0],
                    [4.2, 0.2, 2.0],
                    [-0.2, -0.2, 3.0],
                    [-4.2, 4.2, 4.0],
                ]
            ),
            semantic_labels=np.array([1, 3, 4, 1], dtype=np.uint8),
            camera_position_map=np.array([0.2, 0.2, 0.0]),
        )

        self.assertTrue({(0, 0), (1, 0), (-1, -1), (-2, 1)}.issubset(
            manager.tiles
        ))
        self.assertEqual(statistics["accepted_points"], 4)
        self.assertEqual(statistics["surface_cells"], 4)

    def test_occupancy_ray_crosses_tile_boundary(self):
        manager = _manager()
        manager.update(
            points_map=np.array([[6.2, 0.2, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([1.2, 0.2, 0.0]),
        )

        left = manager.tiles[(0, 0)]
        right = manager.tiles[(1, 0)]
        self.assertTrue(left.occupancy_observed[0, 1])
        self.assertTrue(left.occupancy_observed[0, 3])
        self.assertTrue(right.occupancy_observed[0, 0])
        self.assertTrue(right.occupancy_observed[0, 2])
        self.assertGreater(right.occupancy_log_odds[0, 2], 0.0)

    def test_unknown_lidar_obstacle_endpoint_crosses_tile_boundary(self):
        manager = _manager()
        manager.update(
            points_map=np.array([[6.2, 0.2, 1.0]]),
            semantic_labels=np.array([0], dtype=np.uint8),
            camera_position_map=np.array([1.2, 0.2, 0.0]),
            geometric_endpoint_states=np.array(
                [ENDPOINT_OCCUPIED], dtype=np.int8
            ),
        )

        endpoint_tile = manager.tiles[(1, 0)]
        self.assertTrue(endpoint_tile.occupancy_observed[0, 2])
        self.assertGreater(endpoint_tile.occupancy_log_odds[0, 2], 0.0)
        self.assertEqual(endpoint_tile.semantic_id_layer()[0, 2], 0)

    def test_ground_and_unclassified_unknown_endpoints_are_distinct(self):
        ground = _manager()
        ground.update(
            points_map=np.array([[3.2, 0.2, 0.0]]),
            semantic_labels=np.array([0], dtype=np.uint8),
            camera_position_map=np.array([0.2, 0.2, 0.5]),
            geometric_endpoint_states=np.array([ENDPOINT_FREE]),
        )
        ground_tile = ground.tiles[(0, 0)]
        self.assertTrue(ground_tile.occupancy_observed[0, 3])
        self.assertLess(ground_tile.occupancy_log_odds[0, 3], 0.0)

        unclassified = _manager()
        unclassified.update(
            points_map=np.array([[3.2, 0.2, 0.0]]),
            semantic_labels=np.array([0], dtype=np.uint8),
            camera_position_map=np.array([0.2, 0.2, 0.5]),
            geometric_endpoint_states=np.array([ENDPOINT_UNKNOWN]),
        )
        unknown_tile = unclassified.tiles[(0, 0)]
        self.assertFalse(unknown_tile.occupancy_observed[0, 3])

    def test_semantic_endpoint_state_takes_precedence_over_geometry(self):
        semantic_free = _manager()
        semantic_free.update(
            points_map=np.array([[3.2, 0.2, 0.0]]),
            semantic_labels=np.array([1], dtype=np.uint8),
            camera_position_map=np.array([0.2, 0.2, 0.5]),
            geometric_endpoint_states=np.array([ENDPOINT_OCCUPIED]),
        )
        self.assertLess(
            semantic_free.tiles[(0, 0)].occupancy_log_odds[0, 3],
            0.0,
        )

        semantic_obstacle = _manager()
        semantic_obstacle.update(
            points_map=np.array([[3.2, 0.2, 0.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([0.2, 0.2, 0.5]),
            geometric_endpoint_states=np.array([ENDPOINT_FREE]),
        )
        self.assertGreater(
            semantic_obstacle.tiles[(0, 0)].occupancy_log_odds[0, 3],
            0.0,
        )

    def test_invalid_geometric_endpoint_states_fail_before_map_mutation(self):
        manager = _manager()
        common = {
            "points_map": np.array([[3.2, 0.2, 0.0]]),
            "semantic_labels": np.array([0], dtype=np.uint8),
            "camera_position_map": np.array([0.2, 0.2, 0.5]),
        }
        with self.assertRaises(ValueError):
            manager.update(
                **common,
                geometric_endpoint_states=np.array([255]),
            )
        with self.assertRaises(ValueError):
            manager.update(
                **common,
                geometric_endpoint_states=np.array([], dtype=np.int8),
            )
        self.assertEqual(manager.tile_count, 0)

    def test_direct_lidar_rebuilder_classifies_unknown_wall(self):
        first, second = np.meshgrid(
            np.linspace(-1.0, 1.0, 9),
            np.linspace(-1.0, 1.0, 9),
            indexing="ij",
        )
        wall = np.column_stack(
            (
                np.full(first.size, 6.2),
                first.reshape(-1),
                second.reshape(-1),
            )
        )
        observation = SemanticKeyFrameObservation(
            keyframe_id=8,
            sensor_timestamp=8.0,
            camera_points=wall,
            semantic_labels=np.zeros(len(wall), dtype=np.uint8),
            require_positive_depth=False,
            source="lidar",
        )
        manager = _manager()

        statistics = KeyFrameObservationStore.fuse_observation(
            grid=manager,
            observation=observation,
            pose_world_camera=np.eye(4),
            rotation_map_world=np.eye(3),
            map_translation_offset=np.zeros(3),
        )

        self.assertGreater(statistics["lidar_obstacle_endpoints"], 70)
        self.assertEqual(statistics["lidar_ground_endpoints"], 0)
        self.assertTrue(
            any(
                np.any(tile.occupancy_log_odds > 0.0)
                for tile in manager.tiles.values()
            )
        )

    def test_clone_for_update_copies_only_touched_tiles(self):
        manager = _manager()
        manager.update(
            points_map=np.array([[1.2, 1.2, 1.0], [5.2, 1.2, 1.0]]),
            semantic_labels=np.array([1, 1], dtype=np.uint8),
            camera_position_map=np.array([1.2, 1.2, 0.0]),
        )
        manager.mark_tiles_published(manager.pending_dirty_tile_keys())
        original_left = manager.tiles[(0, 0)]
        original_right = manager.tiles[(1, 0)]

        candidate = manager.clone_for_update()
        candidate.update(
            points_map=np.array([[5.4, 1.4, 2.0]]),
            semantic_labels=np.array([3], dtype=np.uint8),
            camera_position_map=np.array([5.1, 1.1, 0.0]),
        )

        self.assertIs(candidate.tiles[(0, 0)], original_left)
        self.assertIsNot(candidate.tiles[(1, 0)], original_right)
        self.assertIs(manager.tiles[(1, 0)], original_right)

    def test_local_window_stays_fixed_while_global_map_grows(self):
        manager = _manager()
        manager.update(
            points_map=np.array([[1.2, 1.2, 1.0], [21.2, 1.2, 5.0]]),
            semantic_labels=np.array([1, 4], dtype=np.uint8),
            camera_position_map=np.array([1.2, 1.2, 0.0]),
        )

        near_origin = manager.extract_window(
            center_x=2.0,
            center_y=2.0,
            length_x=6.0,
            length_y=6.0,
        )
        near_far_point = manager.extract_window(
            center_x=21.0,
            center_y=1.0,
            length_x=6.0,
            length_y=6.0,
        )
        self.assertEqual(near_origin.elevation.shape, (6, 6))
        self.assertEqual(near_far_point.elevation.shape, (6, 6))
        self.assertEqual(np.count_nonzero(np.isfinite(near_origin.elevation)), 1)
        self.assertEqual(
            np.count_nonzero(np.isfinite(near_far_point.elevation)),
            1,
        )
        self.assertGreaterEqual(manager.tile_count, 2)

    def test_complete_tile_uses_existing_public_rectangle_contract(self):
        manager = _manager()
        manager.update(
            points_map=np.array([[4.2, -0.2, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([4.2, -0.2, 0.0]),
        )
        key = (1, -1)
        update = manager.tile_update(key, full_tile=True)
        metadata = manager.tile_update_metadata(key)
        geometry = update.window.geometry

        self.assertFalse(update.cleared)
        self.assertFalse(metadata.cleared)
        self.assertEqual(metadata.tile_revision, update.tile_revision)
        self.assertEqual(geometry.width, manager.tile_cells)
        self.assertEqual(geometry.height, manager.tile_cells)
        self.assertEqual(update.window.elevation.size, geometry.width * geometry.height)
        self.assertAlmostEqual(
            geometry.origin_x,
            key[0] * manager.tile_length,
        )
        self.assertAlmostEqual(
            geometry.origin_y,
            key[1] * manager.tile_length,
        )

    def test_rebuild_publishes_unknown_rectangle_for_disappeared_tile(self):
        previous = _manager()
        previous.update(
            points_map=np.array([[1.2, 1.2, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([1.2, 1.2, 0.0]),
        )
        previous.mark_tiles_published(previous.pending_dirty_tile_keys())

        rebuilt = _manager()
        rebuilt.update(
            points_map=np.array([[9.2, 1.2, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([9.2, 1.2, 0.0]),
        )
        rebuilt.mark_tiles_published(rebuilt.pending_dirty_tile_keys())
        rebuilt.mark_rebuild_dirty(previous)

        self.assertIn((0, 0), rebuilt.pending_dirty_tile_keys())
        clearing = rebuilt.tile_update((0, 0))
        clearing_metadata = rebuilt.tile_update_metadata((0, 0))
        self.assertTrue(clearing.cleared)
        self.assertTrue(clearing_metadata.cleared)
        self.assertEqual(
            clearing_metadata.tile_revision,
            clearing.tile_revision,
        )
        self.assertEqual(clearing.window.elevation.shape, (4, 4))
        self.assertTrue(np.all(clearing.window.occupancy == -1))
        self.assertTrue(np.all(np.isnan(clearing.window.elevation)))
        self.assertTrue(np.all(clearing.window.observation_count == 0))

        rebuilt.mark_tiles_published(rebuilt.pending_dirty_tile_keys())
        self.assertEqual(rebuilt.pending_dirty_tile_keys(), ())
        self.assertIn((0, 0), rebuilt.replay_tile_keys())
        replayed_clear = rebuilt.tile_update((0, 0), full_tile=True)
        self.assertTrue(np.all(replayed_clear.window.occupancy == -1))

        with tempfile.TemporaryDirectory() as directory:
            paths = rebuilt.save_snapshot(
                Path(directory),
                timestamp_text="13.500000000",
                frame_id="map",
            )
            restored = TiledSemanticMapManager.load_snapshot(paths["index"])
            self.assertIn((0, 0), restored.replay_tile_keys())
            restored_clear = restored.tile_update((0, 0), full_tile=True)
            self.assertTrue(np.all(restored_clear.window.occupancy == -1))
            self.assertEqual(
                restored_clear.tile_revision,
                clearing.tile_revision,
            )

    def test_reactivated_tile_revision_stays_newer_than_tombstone(self):
        previous = _manager()
        previous.update(
            points_map=np.array([[1.2, 1.2, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([1.2, 1.2, 0.0]),
        )

        rebuilt = _manager()
        rebuilt.mark_rebuild_dirty(previous)
        tombstone_revision = rebuilt.tile_update_metadata(
            (0, 0)
        ).tile_revision
        self.assertTrue(rebuilt.tile_update_metadata((0, 0)).cleared)

        rebuilt.update(
            points_map=np.array([[1.2, 1.2, 2.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([1.2, 1.2, 0.0]),
        )
        reactivated = rebuilt.tile_update_metadata((0, 0))

        self.assertFalse(reactivated.cleared)
        self.assertGreater(reactivated.tile_revision, tombstone_revision)

    def test_snapshot_round_trip_restores_every_fusion_layer(self):
        manager = _manager(
            lidar_ground_max_slope_deg=28.0,
            lidar_ground_neighbor_count=9,
            lidar_ground_max_neighbor_distance_m=0.55,
        )
        manager.update(
            points_map=np.array(
                [[-1.2, 0.2, 2.0], [5.2, 1.2, 3.0], [5.4, 1.4, 4.0]]
            ),
            semantic_labels=np.array([1, 4, 4], dtype=np.uint8),
            camera_position_map=np.array([-1.2, 0.2, 0.0]),
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = manager.save_snapshot(
                Path(directory),
                timestamp_text="12.500000000",
                frame_id="map",
            )
            restored = TiledSemanticMapManager.load_snapshot(paths["index"])

            self.assertEqual(set(restored.tiles), set(manager.tiles))
            self.assertEqual(restored.tile_revisions, manager.tile_revisions)
            self.assertEqual(restored.known_cells, manager.known_cells)
            self.assertEqual(
                restored.lidar_ground_config,
                manager.lidar_ground_config,
            )
            self.assertEqual(restored.pending_dirty_tile_keys(), ())
            self.assertEqual(
                restored.replay_tile_keys(), manager.replay_tile_keys()
            )
            for key in manager.tiles:
                expected = manager.tiles[key]
                actual = restored.tiles[key]
                np.testing.assert_array_equal(
                    actual.occupancy_log_odds, expected.occupancy_log_odds
                )
                np.testing.assert_array_equal(
                    actual.occupancy_observed, expected.occupancy_observed
                )
                np.testing.assert_array_equal(
                    actual.semantic_votes, expected.semantic_votes
                )
                np.testing.assert_array_equal(
                    actual.elevation_count, expected.elevation_count
                )
                np.testing.assert_array_equal(
                    actual.elevation_mean, expected.elevation_mean
                )
                np.testing.assert_array_equal(
                    actual.elevation_M2, expected.elevation_M2
                )
                np.testing.assert_array_equal(
                    actual.elevation_min, expected.elevation_min
                )
                np.testing.assert_array_equal(
                    actual.elevation_max, expected.elevation_max
                )

    def test_nonzero_lattice_origin_is_preserved(self):
        manager = _manager(origin_x=-2.0, origin_y=-2.0)
        manager.update(
            points_map=np.array([[-2.2, -2.2, 1.0], [-1.8, -1.8, 2.0]]),
            semantic_labels=np.array([1, 1], dtype=np.uint8),
            camera_position_map=np.array([-1.8, -1.8, 0.0]),
        )
        self.assertIn((-1, -1), manager.tiles)
        self.assertIn((0, 0), manager.tiles)


if __name__ == "__main__":
    unittest.main()
