#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
import tempfile
import types
import unittest
from typing import Optional

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
NODE_PATH = (
    SOURCE
    / "t3_semantic_mapping"
    / "roma_semantic_mapper_node.py"
)
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.semantic_grid import LayeredSemanticGridMap
from t3_semantic_mapping.map_contribution_index import MapContributionIndex
from t3_semantic_mapping.map_correction_worker import (
    MapCorrectionCompletion,
    MapCorrectionWorker,
)
from t3_semantic_mapping.semantic_map_rebuilder import (
    KeyFrameObservationStore,
    SemanticKeyFrameObservation,
)
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


@dataclass(frozen=True)
class _TestMapCorrectionRequest:
    geometry_revision: int
    base_map_revision: int
    live_grid: object
    contribution_index: object
    pending_observations: tuple
    keyframe_poses_cv: dict


def _load_transaction_methods():
    tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    node_class = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "RomaSemanticMapperNode"
    )
    wanted = {
        "_stage_keyframe_observation",
        "_pending_observation_snapshot",
        "_commit_map_candidate",
        "_rebuild_historical_map",
        "_start_background_map_correction",
        "_commit_background_map_correction",
        "_synchronize_background_map_correction",
        "_on_map_correction_timer",
        "_commit_pending_map_observations",
        "_synchronize_map_state",
        "_publish_map_if_changed",
        "_publish_path_if_due",
        "_update_map",
        "_record_missed_keyframe_observation",
        "_retry_missed_keyframe_observations",
        "_get_map_points_cv",
    }
    miniature = ast.ClassDef(
        name="TransactionMethods",
        bases=[],
        keywords=[],
        body=[
            item
            for item in node_class.body
            if isinstance(item, ast.FunctionDef) and item.name in wanted
        ],
        decorator_list=[],
    )
    future = next(
        item
        for item in tree.body
        if isinstance(item, ast.ImportFrom)
        and item.module == "__future__"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, miniature], type_ignores=[])
    )
    namespace = {
        "LayeredSemanticGridMap": LayeredSemanticGridMap,
        "Optional": Optional,
        "SemanticKeyFrameObservation": SemanticKeyFrameObservation,
        "MapCorrectionCompletion": MapCorrectionCompletion,
        "_MapCorrectionRequest": _TestMapCorrectionRequest,
        "copy": copy,
        "np": np,
        "time": time,
    }
    exec(compile(module, str(NODE_PATH), "exec"), namespace)
    return namespace["TransactionMethods"]


TransactionMethods = _load_transaction_methods()


def _grid() -> LayeredSemanticGridMap:
    return LayeredSemanticGridMap(
        resolution=1.0,
        length_x=10.0,
        length_y=10.0,
    )


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return pose


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass

    def fatal(self, _message):
        pass


class _Publisher:
    def __init__(self):
        self.count = 0

    def publish(self, _message):
        self.count += 1


class _FailOnceStore(KeyFrameObservationStore):
    def __init__(self):
        super().__init__()
        self.fail_incremental_once = False
        self.fail_rebuild_once = False

    def build_incremental_candidate(self, **kwargs):
        if self.fail_incremental_once:
            self.fail_incremental_once = False
            raise RuntimeError("synthetic incremental failure")
        return super().build_incremental_candidate(**kwargs)

    def rebuild(self, **kwargs):
        if self.fail_rebuild_once:
            self.fail_rebuild_once = False
            raise RuntimeError("synthetic rebuild failure")
        return super().rebuild(**kwargs)


class _Harness(TransactionMethods):
    def __init__(self):
        self._map_lock = threading.RLock()
        self.grid = _grid()
        self.map_observations = _FailOnceStore()
        self.pending_map_observations = {}
        self.rotation_map_world = np.eye(3, dtype=np.float64)
        self.map_translation_offset = np.zeros(3, dtype=np.float64)
        self.map_revision = 0
        self.publish_unchanged_map_outputs = False
        self._last_published_map_signature = None
        self.map_output_publish_count = 0
        self.map_output_skip_count = 0
        self.map_is_stale = False
        self.map_integrity_fault = None
        self.applied_geometry_revision = 0
        self.geometry_revision = 0
        self.keyframe_poses = {7: _pose(0.0)}
        self.publish_count = 0
        self.publish_attempt_count = 0
        self.publish_failures_remaining = 0
        self.frame_count = 0
        self.path_publish_stride = 3
        self.last_path_publish_frame_count = 0
        self.path_message = types.SimpleNamespace(poses=[object()])
        self.path_pub = _Publisher()
        self.logger = _Logger()

    def _get_keyframe_pose_state(self):
        return self.geometry_revision, {
            keyframe_id: pose.copy()
            for keyframe_id, pose in self.keyframe_poses.items()
        }

    def _make_empty_grid(self):
        return _grid()

    def _map_output_signature(self):
        return (int(self.map_revision),)

    def _publish_current_map_outputs(self, _header):
        self.publish_attempt_count += 1
        if self.publish_failures_remaining > 0:
            self.publish_failures_remaining -= 1
            raise RuntimeError("synthetic publication failure")
        self.publish_count += 1

    def get_logger(self):
        return self.logger


def _tiled_grid() -> TiledSemanticMapManager:
    return TiledSemanticMapManager(
        resolution=1.0,
        tile_cells=8,
        max_rays_per_update=100,
    )


class _TiledHarness(_Harness):
    def __init__(self):
        super().__init__()
        self.grid = _tiled_grid()
        self.map_contribution_index = MapContributionIndex()

    def _make_empty_grid(self):
        return _tiled_grid()


def _observation() -> SemanticKeyFrameObservation:
    return SemanticKeyFrameObservation(
        keyframe_id=7,
        sensor_timestamp=12.5,
        camera_points=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        semantic_labels=np.array([4], dtype=np.uint8),
    )


class _MissedObservationHarness(_Harness):
    """Fails observation building N times, then samples the given image."""

    def __init__(self, build_failures=1):
        super().__init__()
        self.missed_keyframe_observations = {}
        self.build_failures_remaining = int(build_failures)
        self.build_images = []
        self.build_lidar_messages = []

    def _make_keyframe_observation(self, *, keyframe_id, semantic_image,
                                   header, visual_image_bgr=None,
                                   lidar_message=None):
        self.build_images.append(np.array(semantic_image, copy=True))
        self.build_lidar_messages.append(lidar_message)
        if self.build_failures_remaining > 0:
            self.build_failures_remaining -= 1
            raise RuntimeError("synthetic observation build failure")
        return SemanticKeyFrameObservation(
            keyframe_id=int(keyframe_id),
            sensor_timestamp=12.5,
            camera_points=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
            semantic_labels=np.array(
                [semantic_image[0, 0]], dtype=np.uint8),
        )


class _EmptyObservationHarness(_MissedObservationHarness):
    def _make_keyframe_observation(self, **_kwargs):
        return None


def _keyframe_status():
    return types.SimpleNamespace(keyframe_created=True, keyframe_id=7)


def _plain_status():
    return types.SimpleNamespace(keyframe_created=False, keyframe_id=None)


class MissedObservationRetryTest(unittest.TestCase):
    def test_retry_keeps_the_exact_selected_lidar_message(self):
        harness = _MissedObservationHarness(build_failures=0)
        lidar_message = object()
        failure = RuntimeError("waiting for future LIO pose")
        failure.lidar_message = lidar_message

        harness._record_missed_keyframe_observation(
            keyframe_id=7,
            semantic_image=np.full((2, 2), 4, dtype=np.uint8),
            header=None,
            exception=failure,
        )
        harness._retry_missed_keyframe_observations()

        self.assertEqual(harness.build_lidar_messages, [lidar_message])
        self.assertFalse(harness.missed_keyframe_observations)
        self.assertIn(7, harness.pending_map_observations)

    def test_intentionally_empty_observation_is_not_committed_or_retried(self):
        harness = _EmptyObservationHarness(build_failures=0)

        harness._update_map(
            tracking_status=_keyframe_status(),
            semantic_image=np.zeros((2, 2), dtype=np.uint8),
            header=None,
        )

        self.assertFalse(harness.missed_keyframe_observations)
        self.assertFalse(harness.pending_map_observations)
        self.assertEqual(len(harness.map_observations), 0)
        self.assertEqual(harness.map_revision, 0)

        harness._update_map(
            tracking_status=_plain_status(),
            semantic_image=np.zeros((2, 2), dtype=np.uint8),
            header=None,
        )
        self.assertEqual(harness.publish_count, 1)
        self.assertEqual(harness.map_output_skip_count, 1)

    def test_intentional_admission_rejection_is_not_retried_as_failure(self):
        harness = _MissedObservationHarness(build_failures=100)

        harness._update_map(
            tracking_status=_keyframe_status(),
            semantic_image=np.full((2, 2), 4, dtype=np.uint8),
            header=None,
            admit_new_observation=False,
            admission_reason="synthetic recovery gate",
        )

        self.assertEqual(harness.build_images, [])
        self.assertFalse(harness.missed_keyframe_observations)
        self.assertFalse(harness.pending_map_observations)
        self.assertEqual(len(harness.map_observations), 0)
        self.assertEqual(harness.publish_count, 1)

    def test_build_failure_is_cached_and_retried_with_original_image(self):
        harness = _MissedObservationHarness(build_failures=2)
        keyframe_image = np.full((2, 2), 4, dtype=np.uint8)
        later_image = np.full((2, 2), 9, dtype=np.uint8)

        # 关键帧帧：构建失败两次（当帧一次 + 同帧重试一次），条目保留。
        with self.assertRaisesRegex(
            RuntimeError, "Refusing to publish an incomplete map"
        ):
            harness._update_map(
                tracking_status=_keyframe_status(),
                semantic_image=keyframe_image,
                header=None,
            )
        self.assertEqual(tuple(harness.missed_keyframe_observations), (7,))
        self.assertFalse(harness.pending_map_observations)
        self.assertEqual(len(harness.map_observations), 0)
        self.assertEqual(harness.publish_count, 0)

        # 后续普通帧：用缓存的关键帧同帧语义图重试成功并入图。
        harness._update_map(
            tracking_status=_plain_status(),
            semantic_image=later_image,
            header=None,
        )
        self.assertFalse(harness.missed_keyframe_observations)
        self.assertFalse(harness.pending_map_observations)
        self.assertEqual(len(harness.map_observations), 1)
        committed = harness.map_observations.get(7)
        # 语义必须来自关键帧当帧图像（值 4），不是重试帧图像（值 9）。
        np.testing.assert_array_equal(
            committed.semantic_labels, np.array([4], dtype=np.uint8))
        for image in harness.build_images:
            np.testing.assert_array_equal(image, keyframe_image)

    def test_persistently_failing_entry_is_retained_and_blocks_publication(self):
        harness = _MissedObservationHarness(build_failures=100)
        harness.missed_observation_warning_interval = 3
        with self.assertRaises(RuntimeError):
            harness._update_map(
                tracking_status=_keyframe_status(),
                semantic_image=np.zeros((2, 2), dtype=np.uint8),
                header=None,
            )
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                harness._update_map(
                    tracking_status=_plain_status(),
                    semantic_image=np.zeros((2, 2), dtype=np.uint8),
                    header=None,
                )
        self.assertEqual(tuple(harness.missed_keyframe_observations), (7,))
        self.assertGreaterEqual(
            harness.missed_keyframe_observations[7]["attempts"], 3
        )
        self.assertEqual(harness.publish_count, 0)

    def test_cache_capacity_freezes_map_without_dropping_retained_evidence(self):
        harness = _MissedObservationHarness(build_failures=100)
        harness.missed_observation_cache_limit = 1
        semantic_image = np.zeros((2, 2), dtype=np.uint8)
        harness._record_missed_keyframe_observation(
            keyframe_id=7,
            semantic_image=semantic_image,
            header=None,
            exception=RuntimeError("first"),
        )

        with self.assertRaisesRegex(RuntimeError, "publication is frozen"):
            harness._record_missed_keyframe_observation(
                keyframe_id=8,
                semantic_image=semantic_image,
                header=None,
                exception=RuntimeError("second"),
            )

        self.assertEqual(tuple(harness.missed_keyframe_observations), (7,))
        self.assertIn("publication is frozen", harness.map_integrity_fault)
        with self.assertRaisesRegex(RuntimeError, "publication is frozen"):
            harness._synchronize_map_state(header=None, publish=True)
        self.assertEqual(harness.publish_count, 0)


class MapTransactionTest(unittest.TestCase):
    def test_failed_append_is_retained_and_retried_exactly_once(self):
        harness = _Harness()
        observation = _observation()
        self.assertTrue(harness._stage_keyframe_observation(observation))
        original_grid = harness.grid
        original_counts = original_grid.elevation_count.copy()
        harness.map_observations.fail_incremental_once = True

        with self.assertRaisesRegex(
            RuntimeError, "synthetic incremental failure"
        ):
            harness._synchronize_map_state(header=None, publish=True)

        self.assertIs(harness.grid, original_grid)
        np.testing.assert_array_equal(
            harness.grid.elevation_count, original_counts
        )
        self.assertEqual(len(harness.map_observations), 0)
        self.assertEqual(tuple(harness.pending_map_observations), (7,))
        self.assertEqual(harness.publish_count, 0)

        statistics = harness._synchronize_map_state(
            header=None,
            publish=True,
        )

        self.assertEqual(statistics["fused_keyframes"], 1)
        self.assertEqual(len(harness.map_observations), 1)
        self.assertFalse(harness.pending_map_observations)
        self.assertIsNot(harness.grid, original_grid)
        self.assertEqual(np.count_nonzero(harness.grid.elevation_count), 1)
        self.assertEqual(harness.map_revision, 1)
        self.assertEqual(harness.publish_count, 1)

        # An ordinary later frame neither republishes the identical map nor
        # fuses the same keyframe a second time.
        harness._synchronize_map_state(header=None, publish=True)
        self.assertEqual(harness.map_revision, 1)
        self.assertEqual(harness.publish_count, 1)
        self.assertEqual(harness.map_output_skip_count, 1)
        self.assertEqual(
            int(np.sum(harness.grid.elevation_count)),
            1,
        )

    def test_path_stride_keeps_first_periodic_and_forced_final_updates(self):
        harness = _Harness()

        for frame_count in range(1, 6):
            harness.frame_count = frame_count
            harness._publish_path_if_due()

        self.assertEqual(harness.path_pub.count, 2)
        self.assertEqual(harness.last_path_publish_frame_count, 3)

        self.assertTrue(harness._publish_path_if_due(force=True))
        self.assertEqual(harness.path_pub.count, 3)
        self.assertEqual(harness.last_path_publish_frame_count, 5)

    def test_unchanged_map_is_not_republished(self):
        harness = _Harness()

        harness._synchronize_map_state(header=None, publish=True)
        harness._synchronize_map_state(header=None, publish=True)

        self.assertEqual(harness.map_revision, 0)
        self.assertEqual(harness.publish_attempt_count, 1)
        self.assertEqual(harness.publish_count, 1)
        self.assertEqual(harness.map_output_publish_count, 1)
        self.assertEqual(harness.map_output_skip_count, 1)

    def test_publication_failure_retries_without_refusing_evidence(self):
        harness = _Harness()
        harness._stage_keyframe_observation(_observation())
        harness.publish_failures_remaining = 1

        with self.assertRaisesRegex(
            RuntimeError, "synthetic publication failure"
        ):
            harness._synchronize_map_state(header=None, publish=True)

        self.assertEqual(harness.map_revision, 1)
        self.assertEqual(len(harness.map_observations), 1)
        self.assertFalse(harness.pending_map_observations)
        self.assertIsNone(harness._last_published_map_signature)
        self.assertEqual(harness.map_output_publish_count, 0)

        statistics = harness._synchronize_map_state(
            header=None,
            publish=True,
        )

        self.assertIsNone(statistics)
        self.assertEqual(harness.map_revision, 1)
        self.assertEqual(harness.publish_attempt_count, 2)
        self.assertEqual(harness.publish_count, 1)
        self.assertEqual(harness.map_output_publish_count, 1)
        self.assertEqual(
            int(np.sum(harness.grid.elevation_count)),
            1,
        )

    def test_failed_corrected_rebuild_keeps_old_map_then_retries(self):
        harness = _Harness()
        observation = _observation()
        harness.map_observations.add(observation)
        harness.grid, _ = harness.map_observations.rebuild(
            grid_factory=_grid,
            keyframe_poses_cv={7: _pose(0.0)},
            rotation_map_world=harness.rotation_map_world,
            map_translation_offset=harness.map_translation_offset,
        )
        old_grid = harness.grid
        old_row, old_column = old_grid.xy_to_single_index(1.0, 0.0)
        harness.geometry_revision = 1
        harness.keyframe_poses = {7: _pose(2.0)}
        harness.map_observations.fail_rebuild_once = True

        with self.assertRaisesRegex(RuntimeError, "synthetic rebuild failure"):
            harness._synchronize_map_state(header=None, publish=False)

        self.assertIs(harness.grid, old_grid)
        self.assertTrue(harness.map_is_stale)
        self.assertEqual(harness.applied_geometry_revision, 0)

        harness._synchronize_map_state(header=None, publish=False)

        new_row, new_column = harness.grid.xy_to_single_index(3.0, 0.0)
        self.assertEqual(
            harness.grid.elevation_count[old_row, old_column],
            0,
        )
        self.assertGreater(
            harness.grid.elevation_count[new_row, new_column],
            0,
        )
        self.assertFalse(harness.map_is_stale)
        self.assertEqual(harness.applied_geometry_revision, 1)


class TiledCorrectionTransactionTest(unittest.TestCase):
    @staticmethod
    def _batch_observation(keyframe_id: int):
        return SemanticKeyFrameObservation(
            keyframe_id=keyframe_id,
            sensor_timestamp=float(keyframe_id),
            camera_points=np.array(
                [[0.5, 0.1, 1.0], [1.5, 0.2, 1.2]],
                dtype=np.float64,
            ),
            semantic_labels=np.array([1, 4], dtype=np.uint8),
        )

    def _committed_harness(self):
        harness = _TiledHarness()
        harness.keyframe_poses = {}
        for keyframe_id in range(30):
            observation = self._batch_observation(keyframe_id)
            harness.pending_map_observations[keyframe_id] = observation
            x = float(keyframe_id) if keyframe_id < 6 else 20.0 * keyframe_id
            harness.keyframe_poses[keyframe_id] = _pose(x)
        harness._synchronize_map_state(header=None, publish=False)
        return harness

    def test_incremental_commit_builds_index_then_correction_is_partial(self):
        harness = self._committed_harness()
        self.assertTrue(
            harness.map_contribution_index.is_complete_for(
                harness.map_observations.keyframe_ids
            )
        )
        harness.geometry_revision = 1
        harness.keyframe_poses[2] = _pose(17.0)

        statistics = harness._synchronize_map_state(
            header=None,
            publish=False,
        )
        expected, _ = harness.map_observations.rebuild(
            grid_factory=_tiled_grid,
            keyframe_poses_cv=harness.keyframe_poses,
            rotation_map_world=harness.rotation_map_world,
            map_translation_offset=harness.map_translation_offset,
        )

        self.assertEqual(statistics["rebuild_strategy"], "partial")
        self.assertEqual(harness.applied_geometry_revision, 1)
        self.assertEqual(harness.map_revision, 2)
        self.assertEqual(set(harness.grid.tiles), set(expected.tiles))
        for key, expected_tile in expected.tiles.items():
            actual_tile = harness.grid.tiles[key]
            for name, _dtype in expected._SNAPSHOT_TILE_FIELDS:
                np.testing.assert_array_equal(
                    getattr(actual_tile, name),
                    getattr(expected_tile, name),
                )

    def test_stale_base_revision_cannot_replace_newer_live_map(self):
        harness = self._committed_harness()
        live_grid = harness.grid

        with self.assertRaisesRegex(RuntimeError, "Live map changed"):
            harness._commit_map_candidate(
                candidate=live_grid.clone_for_update(),
                pending_observations=(),
                geometry_revision=0,
                contribution_index=(
                    harness.map_contribution_index.clone()
                ),
                base_map_revision=harness.map_revision - 1,
            )

        self.assertIs(harness.grid, live_grid)
        self.assertEqual(harness.map_revision, 1)


class BackgroundCorrectionTransactionTest(unittest.TestCase):
    _batch_observation = staticmethod(
        TiledCorrectionTransactionTest._batch_observation
    )

    def _committed_harness(self):
        return TiledCorrectionTransactionTest._committed_harness(self)

    def _background_harness(self):
        harness = self._committed_harness()
        harness.map_correction_worker = MapCorrectionWorker()
        return harness

    def test_live_sync_returns_while_private_candidate_is_building(self):
        harness = self._background_harness()
        started = threading.Event()
        release = threading.Event()
        original_build = (
            harness.map_observations.build_corrected_candidate
        )

        def delayed_build(**kwargs):
            started.set()
            release.wait()
            return original_build(**kwargs)

        harness.map_observations.build_corrected_candidate = delayed_build
        harness.geometry_revision = 1
        harness.keyframe_poses[2] = _pose(17.0)
        live_grid = harness.grid
        try:
            pending = harness._synchronize_map_state(
                header=None,
                publish=False,
            )
            self.assertTrue(started.wait(timeout=1.0))
            self.assertEqual(
                pending["rebuild_strategy"],
                "background_pending",
            )
            self.assertIs(harness.grid, live_grid)
            self.assertEqual(harness.applied_geometry_revision, 0)

            new_observation = self._batch_observation(100)
            harness.pending_map_observations[100] = new_observation
            harness.keyframe_poses[100] = _pose(800.0)
            release.set()
            completed = harness._synchronize_map_state(
                header=None,
                publish=False,
                wait_for_correction=True,
            )

            self.assertEqual(completed["rebuild_strategy"], "partial")
            self.assertIsNotNone(harness.map_observations.get(100))
            self.assertTrue(
                harness.map_contribution_index.is_complete_for(
                    harness.map_observations.keyframe_ids
                )
            )
            self.assertFalse(harness.pending_map_observations)
        finally:
            release.set()
            harness.map_correction_worker.shutdown()

    def test_obsolete_geometry_result_is_discarded_and_rebuilt(self):
        harness = self._background_harness()
        started = threading.Event()
        release = threading.Event()
        original_build = (
            harness.map_observations.build_corrected_candidate
        )
        build_count = 0

        def delay_first_build(**kwargs):
            nonlocal build_count
            build_count += 1
            if build_count == 1:
                started.set()
                release.wait()
            return original_build(**kwargs)

        harness.map_observations.build_corrected_candidate = (
            delay_first_build
        )
        harness.geometry_revision = 1
        harness.keyframe_poses[2] = _pose(17.0)
        try:
            harness._synchronize_map_state(header=None, publish=False)
            self.assertTrue(started.wait(timeout=1.0))
            harness.geometry_revision = 2
            harness.keyframe_poses[2] = _pose(25.0)
            release.set()

            completed = harness._synchronize_map_state(
                header=None,
                publish=False,
                wait_for_correction=True,
            )

            self.assertGreaterEqual(build_count, 2)
            self.assertEqual(harness.applied_geometry_revision, 2)
            self.assertEqual(harness.map_revision, 2)
            self.assertIn(
                completed["rebuild_strategy"],
                ("partial", "full"),
            )
            self.assertEqual(
                harness.map_contribution_index.changed_keyframes(
                    keyframe_ids=harness.map_observations.keyframe_ids,
                    current_poses=harness.keyframe_poses,
                ),
                (),
            )
        finally:
            release.set()
            harness.map_correction_worker.shutdown()

    def test_late_recovered_observation_restarts_in_time_order(self):
        harness = self._background_harness()
        started = threading.Event()
        release = threading.Event()
        original_build = (
            harness.map_observations.build_corrected_candidate
        )
        build_count = 0

        def delay_first_build(**kwargs):
            nonlocal build_count
            build_count += 1
            if build_count == 1:
                started.set()
                release.wait()
            return original_build(**kwargs)

        harness.map_observations.build_corrected_candidate = (
            delay_first_build
        )
        harness.geometry_revision = 1
        harness.keyframe_poses[2] = _pose(17.0)
        try:
            harness._synchronize_map_state(header=None, publish=False)
            self.assertTrue(started.wait(timeout=1.0))
            template = self._batch_observation(100)
            historical = SemanticKeyFrameObservation(
                keyframe_id=100,
                sensor_timestamp=4.5,
                camera_points=template.camera_points,
                semantic_labels=template.semantic_labels,
            )
            harness.pending_map_observations[100] = historical
            harness.keyframe_poses[100] = _pose(4.5)
            release.set()

            completed = harness._synchronize_map_state(
                header=None,
                publish=False,
                wait_for_correction=True,
            )

            self.assertGreaterEqual(build_count, 2)
            self.assertEqual(completed["rebuild_strategy"], "full")
            self.assertEqual(
                completed["rebuild_reason"],
                "historical_pending_observation",
            )
            self.assertIsNotNone(harness.map_observations.get(100))
        finally:
            release.set()
            harness.map_correction_worker.shutdown()

    def test_background_failure_keeps_live_map_and_retry_succeeds(self):
        harness = self._background_harness()
        original_build = (
            harness.map_observations.build_corrected_candidate
        )
        failure_count = 0

        def fail_once(**kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count == 1:
                raise RuntimeError("synthetic background failure")
            return original_build(**kwargs)

        harness.map_observations.build_corrected_candidate = fail_once
        harness.geometry_revision = 1
        harness.keyframe_poses[2] = _pose(17.0)
        live_grid = harness.grid
        try:
            harness._synchronize_map_state(header=None, publish=False)
            with self.assertRaisesRegex(
                RuntimeError, "synthetic background failure"
            ):
                harness._synchronize_map_state(
                    header=None,
                    publish=False,
                    wait_for_correction=True,
                )
            self.assertIs(harness.grid, live_grid)
            self.assertTrue(harness.map_is_stale)

            completed = harness._synchronize_map_state(
                header=None,
                publish=False,
                wait_for_correction=True,
            )
            self.assertEqual(completed["rebuild_strategy"], "partial")
            self.assertFalse(harness.map_is_stale)
        finally:
            harness.map_correction_worker.shutdown()

    def test_ready_timer_commits_without_another_image(self):
        harness = self._background_harness()
        harness.geometry_revision = 1
        harness.keyframe_poses[2] = _pose(17.0)
        harness.last_header = object()
        try:
            harness._synchronize_map_state(header=None, publish=False)
            deadline = time.monotonic() + 2.0
            while (
                not harness.map_correction_worker.ready
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            self.assertTrue(harness.map_correction_worker.ready)

            harness._on_map_correction_timer()

            self.assertEqual(harness.applied_geometry_revision, 1)
            self.assertEqual(harness.map_revision, 2)
            self.assertFalse(harness.map_correction_worker.active)
            self.assertEqual(harness.publish_count, 1)
        finally:
            harness.map_correction_worker.shutdown()


class PointCloudArchiveStreamingTest(unittest.TestCase):
    def test_streaming_keeps_original_global_linspace_selection(self):
        observations = []
        for keyframe_id, count in ((2, 5), (7, 4), (11, 6)):
            points = np.column_stack(
                [
                    np.arange(count, dtype=np.float64) + keyframe_id,
                    np.full(count, keyframe_id * 0.1),
                    np.arange(count, dtype=np.float64) + 1.0,
                ]
            )
            observations.append(
                SemanticKeyFrameObservation(
                    keyframe_id=keyframe_id,
                    sensor_timestamp=float(keyframe_id),
                    camera_points=points,
                    semantic_labels=np.ones(count, dtype=np.uint8),
                )
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            harness = _Harness()
            harness.localization_mode = "lio"
            harness.pointcloud_max_points = 7
            harness.localization_keyframe_poses = {
                2: _pose(0.5),
                7: _pose(-1.0),
                11: _pose(3.0),
            }
            harness.map_observations = KeyFrameObservationStore(
                Path(temporary_directory) / "stream.sqlite3",
                memory_cache_size=1,
            )
            harness.map_observations.add_many(observations)

            complete_cloud = np.vstack(
                [
                    (
                        harness.localization_keyframe_poses[
                            observation.keyframe_id
                        ][:3, :3]
                        @ observation.camera_points.T
                    ).T
                    + harness.localization_keyframe_poses[
                        observation.keyframe_id
                    ][:3, 3]
                    for observation in observations
                ]
            )
            expected_indices = np.linspace(
                0,
                len(complete_cloud) - 1,
                harness.pointcloud_max_points,
                dtype=np.int64,
            )

            actual = harness._get_map_points_cv()

            np.testing.assert_array_equal(
                actual, complete_cloud[expected_indices]
            )
            self.assertLessEqual(
                harness.map_observations.resident_count, 1
            )
            harness.map_observations.close()


if __name__ == "__main__":
    unittest.main()
