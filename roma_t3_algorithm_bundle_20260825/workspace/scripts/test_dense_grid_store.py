#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from t3_semantic_mapping.dense_grid_store import (
    FORMAT_NAME,
    LiveDenseGlobalMapWriter,
    save_dense_global_grid_map,
)
from t3_semantic_mapping.semantic_grid import GridGeometry
from t3_semantic_mapping.tiled_semantic_map import SemanticMapWindow


class DenseGridStoreTest(unittest.TestCase):
    def test_complete_map_is_directly_readable_without_pickle(self):
        geometry = GridGeometry(
            resolution=0.2,
            width=3,
            height=2,
            origin_x=-1.0,
            origin_y=2.0,
        )
        shape = (2, 3)
        window = SemanticMapWindow(
            geometry=geometry,
            class_names=("unknown", "ground"),
            occupancy=np.asarray([[-1, 0, 100], [20, 40, 60]], dtype=np.int8),
            semantic=np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8),
            semantic_confidence=np.full(shape, 0.75, dtype=np.float32),
            elevation=np.arange(6, dtype=np.float32).reshape(shape),
            elevation_variance=np.full(shape, 0.1, dtype=np.float32),
            height_range=np.full(shape, 0.2, dtype=np.float32),
            roughness=np.full(shape, 0.3, dtype=np.float32),
            observation_count=np.arange(6, dtype=np.uint32).reshape(shape),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_grid_map.npz"
            result = save_dense_global_grid_map(
                path,
                window,
                frame_id="map",
                map_revision=7,
                timestamp_text="123_456",
            )
            self.assertEqual(result, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["format"].item(), FORMAT_NAME)
                self.assertEqual(archive["frame_id"].item(), "map")
                self.assertEqual(int(archive["map_revision"]), 7)
                self.assertEqual(float(archive["resolution"]), 0.2)
                self.assertEqual(float(archive["origin_x"]), -1.0)
                self.assertEqual(float(archive["origin_y"]), 2.0)
                self.assertEqual(archive["occupancy"].shape, shape)
                self.assertEqual(archive["color_bgr"].shape, shape + (3,))
                np.testing.assert_array_equal(
                    archive["semantic_id"], window.semantic
                )

    def test_existing_file_is_replaced_and_no_temporary_file_remains(self):
        geometry = GridGeometry(1.0, 1, 1, 0.0, 0.0)
        window = SemanticMapWindow(
            geometry=geometry,
            class_names=("unknown",),
            occupancy=np.asarray([[100]], dtype=np.int8),
            semantic=np.zeros((1, 1), dtype=np.uint8),
            semantic_confidence=np.zeros((1, 1), dtype=np.float32),
            elevation=np.zeros((1, 1), dtype=np.float32),
            elevation_variance=np.zeros((1, 1), dtype=np.float32),
            height_range=np.zeros((1, 1), dtype=np.float32),
            roughness=np.zeros((1, 1), dtype=np.float32),
            observation_count=np.ones((1, 1), dtype=np.uint32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_grid_map.npz"
            path.write_bytes(b"old")
            save_dense_global_grid_map(
                path,
                window,
                frame_id="map",
                map_revision=1,
                timestamp_text="one",
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(int(archive["occupancy"][0, 0]), 100)
            self.assertEqual(list(Path(directory).glob(".*.tmp.npz")), [])

    def test_background_writer_persists_latest_revision(self):
        geometry = GridGeometry(1.0, 1, 1, 0.0, 0.0)

        def make_window(value):
            return SemanticMapWindow(
                geometry=geometry,
                class_names=("unknown",),
                occupancy=np.asarray([[value]], dtype=np.int8),
                semantic=np.zeros((1, 1), dtype=np.uint8),
                semantic_confidence=np.zeros((1, 1), dtype=np.float32),
                elevation=np.zeros((1, 1), dtype=np.float32),
                elevation_variance=np.zeros((1, 1), dtype=np.float32),
                height_range=np.zeros((1, 1), dtype=np.float32),
                roughness=np.zeros((1, 1), dtype=np.float32),
                observation_count=np.ones((1, 1), dtype=np.uint32),
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_grid_map.npz"
            writer = LiveDenseGlobalMapWriter(path, frame_id="map")
            try:
                writer.enqueue(
                    make_window(10), map_revision=1, timestamp_text="one"
                )
                writer.enqueue(
                    make_window(20), map_revision=2, timestamp_text="two"
                )
                writer.flush(2)
                self.assertEqual(writer.persisted_revision, 2)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(int(archive["map_revision"]), 2)
                    self.assertEqual(int(archive["occupancy"][0, 0]), 20)
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
