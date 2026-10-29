#!/usr/bin/env python3
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.semantic_grid import LayeredSemanticGridMap
from t3_semantic_mapping.semantic_map_rebuilder import (
    KeyFrameObservationStore,
    SemanticKeyFrameObservation,
)


def _grid() -> LayeredSemanticGridMap:
    return LayeredSemanticGridMap(
        resolution=1.0,
        length_x=10.0,
        length_y=10.0,
        max_rays_per_update=1200,
    )


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return pose


class _FailingGrid(LayeredSemanticGridMap):
    def update(self, **kwargs):
        del kwargs
        # Simulate a failure after a grid implementation has already mutated
        # some candidate state.
        self.elevation_count[0, 0] = 99
        raise RuntimeError("synthetic fusion failure")


class _RecordingStore(KeyFrameObservationStore):
    def __init__(self):
        super().__init__()
        self.fused_ids = []

    def fuse_observation(self, **kwargs):
        self.fused_ids.append(int(kwargs["observation"].keyframe_id))
        return super().fuse_observation(**kwargs)


class GridRebuildTest(unittest.TestCase):
    def setUp(self):
        self.rotation = np.eye(3, dtype=np.float64)
        self.offset = np.zeros(3, dtype=np.float64)
        self.observation = SemanticKeyFrameObservation(
            keyframe_id=7,
            sensor_timestamp=12.5,
            camera_points=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
            semantic_labels=np.array([4], dtype=np.uint8),
        )
        self.store = KeyFrameObservationStore()
        self.assertTrue(self.store.add(self.observation))

    def _rebuild(self, x: float):
        return self.store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv={7: _pose(x)},
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

    def test_corrected_pose_removes_old_cell_and_populates_new_cell(self):
        first, _ = self._rebuild(0.0)
        corrected, statistics = self._rebuild(2.0)

        old_row, old_column = first.xy_to_single_index(1.0, 0.0)
        new_row, new_column = corrected.xy_to_single_index(3.0, 0.0)
        self.assertGreater(first.elevation_count[old_row, old_column], 0)
        self.assertEqual(corrected.elevation_count[old_row, old_column], 0)
        self.assertGreater(corrected.elevation_count[new_row, new_column], 0)
        self.assertEqual(statistics["replayed_keyframes"], 1)

    def test_negative_direct_lidar_ids_replay_in_sensor_time_order(self):
        store = _RecordingStore()
        later = SemanticKeyFrameObservation(
            keyframe_id=-2,
            sensor_timestamp=2.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
        )
        earlier = SemanticKeyFrameObservation(
            keyframe_id=-1,
            sensor_timestamp=1.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([1], dtype=np.uint8),
        )
        # Insert order is intentionally unrelated to either ID or time.
        store.add_many((later, earlier))

        store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv={-1: _pose(0.0), -2: _pose(0.0)},
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

        self.assertEqual(store.keyframe_ids, (-2, -1))
        self.assertEqual(store.replay_keyframe_ids, (-1, -2))
        self.assertEqual(store.fused_ids, [-1, -2])

    def test_rebuild_is_deterministic_and_does_not_mutate_live_grid(self):
        live, _ = self._rebuild(0.0)
        live_before = live.elevation_count.copy()
        rebuilt_a, _ = self._rebuild(2.0)
        rebuilt_b, _ = self._rebuild(2.0)

        np.testing.assert_array_equal(live.elevation_count, live_before)
        np.testing.assert_array_equal(
            rebuilt_a.elevation_count, rebuilt_b.elevation_count
        )
        np.testing.assert_array_equal(
            rebuilt_a.semantic_votes, rebuilt_b.semantic_votes
        )
        np.testing.assert_allclose(
            rebuilt_a.occupancy_log_odds,
            rebuilt_b.occupancy_log_odds,
        )

    def test_complete_rebuild_rejects_any_missing_keyframe_pose(self):
        grid_factory_called = False

        def tracked_grid_factory():
            nonlocal grid_factory_called
            grid_factory_called = True
            return _grid()

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot rebuild a complete map.*keyframe.*7",
        ):
            self.store.rebuild(
                grid_factory=tracked_grid_factory,
                keyframe_poses_cv={},
                rotation_map_world=self.rotation,
                map_translation_offset=self.offset,
            )

        self.assertFalse(grid_factory_called)

    def test_keyframe_observation_is_inserted_exactly_once(self):
        duplicate = SemanticKeyFrameObservation(
            keyframe_id=7,
            sensor_timestamp=12.5,
            camera_points=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
            semantic_labels=np.array([4], dtype=np.uint8),
        )
        self.assertFalse(self.store.add(duplicate))
        self.assertEqual(len(self.store), 1)

        conflicting = SemanticKeyFrameObservation(
            keyframe_id=7,
            sensor_timestamp=12.5,
            camera_points=np.array([[2.0, 0.0, 1.0]], dtype=np.float32),
            semantic_labels=np.array([4], dtype=np.uint8),
        )
        with self.assertRaises(ValueError):
            self.store.add(conflicting)

    def test_batch_store_commit_is_all_or_nothing(self):
        new_observation = SemanticKeyFrameObservation(
            keyframe_id=8,
            sensor_timestamp=13.5,
            camera_points=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
            semantic_labels=np.array([1], dtype=np.uint8),
        )
        conflicting_existing = SemanticKeyFrameObservation(
            keyframe_id=7,
            sensor_timestamp=12.5,
            camera_points=np.array([[9.0, 0.0, 1.0]], dtype=np.float32),
            semantic_labels=np.array([4], dtype=np.uint8),
        )

        with self.assertRaises(ValueError):
            self.store.add_many((new_observation, conflicting_existing))

        self.assertEqual(len(self.store), 1)
        self.assertIsNone(self.store.get(8))

    def test_failed_incremental_candidate_never_mutates_live_grid(self):
        live = _FailingGrid(
            resolution=1.0,
            length_x=10.0,
            length_y=10.0,
        )
        before = live.elevation_count.copy()

        with self.assertRaisesRegex(RuntimeError, "synthetic fusion failure"):
            self.store.build_incremental_candidate(
                live_grid=live,
                observations=(self.observation,),
                keyframe_poses_cv={7: _pose(0.0)},
                rotation_map_world=self.rotation,
                map_translation_offset=self.offset,
            )

        np.testing.assert_array_equal(live.elevation_count, before)

    def test_successful_incremental_candidate_is_detached(self):
        live = _grid()
        candidate, statistics = self.store.build_incremental_candidate(
            live_grid=live,
            observations=(self.observation,),
            keyframe_poses_cv={7: _pose(0.0)},
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

        self.assertEqual(np.count_nonzero(live.elevation_count), 0)
        self.assertEqual(np.count_nonzero(candidate.elevation_count), 1)
        self.assertEqual(statistics["fused_keyframes"], 1)

    def test_observation_arrays_are_detached_and_read_only(self):
        source = np.array([[2.0, 1.0, 3.0]], dtype=np.float32)
        observation = SemanticKeyFrameObservation(
            keyframe_id=8,
            sensor_timestamp=13.5,
            camera_points=source,
            semantic_labels=np.array([1], dtype=np.uint8),
        )
        source[0, 0] = 99.0
        self.assertEqual(float(observation.camera_points[0, 0]), 2.0)
        with self.assertRaises(ValueError):
            observation.camera_points[0, 0] = 3.0

    def test_page_backed_archive_is_exact_reopenable_and_ram_bounded(self):
        observations = {
            keyframe_id: SemanticKeyFrameObservation(
                keyframe_id=keyframe_id,
                sensor_timestamp=100.0 + keyframe_id / 7.0,
                camera_points=np.array(
                    [
                        [np.pi + keyframe_id, -0.25, 1.125],
                        [0.125, np.e + keyframe_id, 2.75],
                    ],
                    dtype=np.float64,
                ),
                semantic_labels=np.array([1, 4], dtype=np.uint8),
                sensor_origin_camera=np.array(
                    [0.01, -0.02, 0.03], dtype=np.float64
                ),
                source="archive-test",
            )
            for keyframe_id in range(5)
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "observations.sqlite3"
            store = KeyFrameObservationStore(
                archive_path,
                memory_cache_size=2,
                compression_level=1,
            )
            self.assertEqual(
                store.add_many(observations.values()), tuple(range(5))
            )
            self.assertEqual(len(store), 5)
            self.assertLessEqual(store.resident_count, 2)
            self.assertEqual(store.point_count(3), 2)
            for keyframe_id, expected in observations.items():
                loaded = store.get(keyframe_id)
                self.assertIsNotNone(loaded)
                self.assertTrue(store.observations_equal(expected, loaded))
                self.assertLessEqual(store.resident_count, 2)
            store.close()

            reopened = KeyFrameObservationStore(
                archive_path,
                memory_cache_size=1,
                compression_level=0,
            )
            self.assertEqual(reopened.keyframe_ids, tuple(range(5)))
            for keyframe_id, expected in observations.items():
                loaded = reopened.get(keyframe_id)
                self.assertTrue(reopened.observations_equal(expected, loaded))
                self.assertLessEqual(reopened.resident_count, 1)
            reopened.close()

    def test_archive_transaction_rolls_back_whole_batch_on_disk_error(self):
        first = SemanticKeyFrameObservation(
            keyframe_id=8,
            sensor_timestamp=8.0,
            camera_points=np.array([[1.0, 0.0, 1.0]]),
            semantic_labels=np.array([1], dtype=np.uint8),
        )
        second = SemanticKeyFrameObservation(
            keyframe_id=9,
            sensor_timestamp=9.0,
            camera_points=np.array([[2.0, 0.0, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "atomic.sqlite3"
            store = KeyFrameObservationStore(
                archive_path, memory_cache_size=2
            )
            self.assertIsNotNone(store._connection)
            store._connection.execute(
                """
                CREATE TRIGGER reject_keyframe_nine
                BEFORE INSERT ON observations
                WHEN NEW.keyframe_id = 9
                BEGIN
                    SELECT RAISE(ABORT, 'synthetic disk failure');
                END
                """
            )
            with self.assertRaises(sqlite3.DatabaseError):
                store.add_many((first, second))
            self.assertEqual(len(store), 0)
            self.assertIsNone(store.get(8))
            self.assertIsNone(store.get(9))
            store.close()

            reopened = KeyFrameObservationStore(
                archive_path, memory_cache_size=1
            )
            self.assertEqual(len(reopened), 0)
            reopened.close()

    def test_disk_replay_produces_identical_complete_grid(self):
        observations = (
            SemanticKeyFrameObservation(
                keyframe_id=1,
                sensor_timestamp=1.0,
                camera_points=np.array(
                    [[1.2, 0.1, 1.5], [2.1, -0.4, 2.0]],
                    dtype=np.float64,
                ),
                semantic_labels=np.array([1, 4], dtype=np.uint8),
            ),
            SemanticKeyFrameObservation(
                keyframe_id=2,
                sensor_timestamp=2.0,
                camera_points=np.array(
                    [[0.4, 1.3, 1.2], [1.6, 1.4, 2.2]],
                    dtype=np.float64,
                ),
                semantic_labels=np.array([3, 1], dtype=np.uint8),
            ),
        )
        poses = {1: _pose(0.0), 2: _pose(1.0)}
        memory_store = KeyFrameObservationStore()
        memory_store.add_many(observations)
        expected, expected_statistics = memory_store.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv=poses,
            rotation_map_world=self.rotation,
            map_translation_offset=self.offset,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "exact.sqlite3"
            archived = KeyFrameObservationStore(
                archive_path, memory_cache_size=1
            )
            archived.add_many(observations)
            archived.close()
            archived = KeyFrameObservationStore(
                archive_path, memory_cache_size=1
            )
            actual, actual_statistics = archived.rebuild(
                grid_factory=_grid,
                keyframe_poses_cv=poses,
                rotation_map_world=self.rotation,
                map_translation_offset=self.offset,
            )
            archived.close()

        self.assertEqual(expected_statistics, actual_statistics)
        np.testing.assert_array_equal(
            expected.elevation_count, actual.elevation_count
        )
        np.testing.assert_array_equal(
            expected.semantic_votes, actual.semantic_votes
        )
        np.testing.assert_array_equal(
            expected.occupancy_log_odds, actual.occupancy_log_odds
        )
        np.testing.assert_array_equal(
            expected.elevation_mean, actual.elevation_mean
        )
        np.testing.assert_array_equal(
            expected.elevation_M2, actual.elevation_M2
        )


if __name__ == "__main__":
    unittest.main()
