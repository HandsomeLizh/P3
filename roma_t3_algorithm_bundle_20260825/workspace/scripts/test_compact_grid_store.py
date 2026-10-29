#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.compact_grid_store import (
    FORMAT_NAME,
    LiveCompactGlobalMapWriter,
    load_compact_metadata,
    load_compact_tile,
    save_compact_global_map,
)
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


class CompactGridStoreTest(unittest.TestCase):
    def _map(self):
        grid = TiledSemanticMapManager(
            resolution=0.1,
            tile_cells=32,
            max_rays_per_update=100,
        )
        grid.update(
            points_map=np.array(
                [[0.2, 0.2, 1.234], [3.4, 0.2, 2.345], [-0.2, -0.2, 0.5]]
            ),
            semantic_labels=np.array([1, 4, 3], dtype=np.uint8),
            camera_position_map=np.array([0.0, 0.0, 0.0]),
        )
        return grid

    def test_single_file_round_trip_preserves_compact_delivery_layers(self):
        grid = self._map()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_grid_map.sqlite3"
            saved = save_compact_global_map(
                grid,
                path,
                frame_id="map",
                timestamp_text="12.300000000",
            )
            self.assertEqual(saved, path)
            self.assertTrue(path.is_file())
            metadata = load_compact_metadata(path)
            self.assertEqual(metadata["format"], FORMAT_NAME)
            self.assertEqual(metadata["tile_count"], grid.tile_count)
            self.assertEqual(metadata["known_cells"], grid.known_cells)

            key = next(iter(grid.tiles))
            decoded = load_compact_tile(path, *key)
            source = grid.tiles[key]
            self.assertTrue(
                np.array_equal(decoded["occupancy"], source.occupancy_int8())
            )
            self.assertTrue(
                np.array_equal(decoded["semantic"], source.semantic_id_layer())
            )
            valid = np.isfinite(source.elevation_layer())
            np.testing.assert_allclose(
                decoded["elevation"][valid],
                source.elevation_layer()[valid],
                atol=0.0051,
            )
            with sqlite3.connect(str(path)) as connection:
                stored_tiles = connection.execute(
                    "SELECT COUNT(*) FROM tiles"
                ).fetchone()[0]
            self.assertEqual(stored_tiles, grid.tile_count)

    def test_overview_is_coarse_global_extent_and_keeps_obstacles(self):
        grid = self._map()
        geometry, occupancy = grid.extract_occupancy_overview(resolution=1.0)
        self.assertAlmostEqual(geometry.resolution, 1.0)
        self.assertEqual(occupancy.shape, (geometry.height, geometry.width))
        self.assertTrue(np.any(occupancy >= 50))
        self.assertLess(occupancy.size, grid.tile_count * grid.tile_cells ** 2)

    def test_live_writer_exposes_complete_transactional_revisions(self):
        empty = TiledSemanticMapManager(
            resolution=0.1,
            tile_cells=32,
            max_rays_per_update=100,
        )
        first = self._map()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_grid_map.sqlite3"
            writer = LiveCompactGlobalMapWriter(
                path,
                initial_grid=empty,
                frame_id="map",
            )
            try:
                writer.enqueue(
                    first.clone_for_update(),
                    map_revision=7,
                    timestamp_text="7.000000000",
                )
                writer.flush(7)
                metadata = load_compact_metadata(path)
                self.assertEqual(metadata["map_revision"], 7)
                self.assertEqual(metadata["tile_count"], first.tile_count)

                second = first.clone_for_update()
                second.update(
                    points_map=np.array([[9.2, 0.2, 3.0]]),
                    semantic_labels=np.array([4], dtype=np.uint8),
                    camera_position_map=np.array([9.0, 0.2, 0.0]),
                )
                writer.enqueue(
                    second.clone_for_update(),
                    map_revision=8,
                    timestamp_text="8.000000000",
                )
                writer.flush(8)
                metadata = load_compact_metadata(path)
                self.assertEqual(metadata["map_revision"], 8)
                self.assertEqual(metadata["tile_count"], second.tile_count)
                self.assertGreaterEqual(
                    writer.last_statistics["updated_tiles"], 1
                )
            finally:
                writer.close()

    def test_selected_local_layers_avoid_materialising_debug_layers(self):
        grid = self._map()
        window = grid.extract_window(
            center_x=0.0,
            center_y=0.0,
            length_x=6.0,
            length_y=6.0,
        )
        layers = window.layers(
            ("occupancy", "semantic_id", "elevation", "roughness")
        )
        self.assertEqual(
            tuple(layers),
            ("occupancy", "semantic_id", "elevation", "roughness"),
        )
        self.assertNotIn("color", layers)
        with self.assertRaises(ValueError):
            window.layers(("not_a_layer",))


if __name__ == "__main__":
    unittest.main()
