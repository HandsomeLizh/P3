#!/usr/bin/env python3
from __future__ import annotations

import threading
import types
import unittest

import numpy as np
from std_msgs.msg import Header
from t3_interfaces.msg import SemanticMapBatch

from t3_semantic_mapping.map_transfer_protocol import MapTransferSession
from t3_semantic_mapping.roma_semantic_mapper_node import (
    RomaSemanticMapperNode,
)
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _Publisher:
    def __init__(self, label, events, *, subscriptions=0, fail_at=None):
        self.label = label
        self.topic_name = f"/test/{label}"
        self.events = events
        self.subscriptions = int(subscriptions)
        self.fail_at = fail_at
        self.attempts = 0

    def publish(self, message):
        self.attempts += 1
        if self.fail_at is not None and self.attempts == self.fail_at:
            raise RuntimeError(f"synthetic {self.label} failure")
        self.events.append((self.label, message))

    def get_subscription_count(self):
        return self.subscriptions


def _header() -> Header:
    header = Header()
    header.stamp.sec = 12
    header.stamp.nanosec = 500
    header.frame_id = "map"
    return header


def _grid(two_tiles=False) -> TiledSemanticMapManager:
    manager = TiledSemanticMapManager(
        resolution=1.0,
        tile_cells=4,
        max_rays_per_update=100,
        map_id="test-map",
    )
    points = [[1.2, 1.2, 1.0]]
    if two_tiles:
        points.append([5.2, 1.2, 1.5])
    manager.update(
        points_map=np.asarray(points, dtype=np.float64),
        semantic_labels=np.full(len(points), 4, dtype=np.uint8),
        camera_position_map=np.array([0.2, 1.2, 0.5]),
    )
    return manager


class _Harness:
    _publish_incremental_window_snapshot = (
        RomaSemanticMapperNode._publish_incremental_window_snapshot
    )
    _tile_transfer_plan = staticmethod(
        RomaSemanticMapperNode._tile_transfer_plan
    )
    _publish_map_batch_begin = (
        RomaSemanticMapperNode._publish_map_batch_begin
    )
    _publish_map_batch_complete = (
        RomaSemanticMapperNode._publish_map_batch_complete
    )
    _start_incremental_replay = RomaSemanticMapperNode._start_incremental_replay
    _request_incremental_delta = (
        RomaSemanticMapperNode._request_incremental_delta
    )
    _start_incremental_delta = (
        RomaSemanticMapperNode._start_incremental_delta
    )
    _finish_incremental_replay = RomaSemanticMapperNode._finish_incremental_replay
    _on_incremental_replay_timer = (
        RomaSemanticMapperNode._on_incremental_replay_timer
    )
    _discovered_incremental_subscription_gids = (
        RomaSemanticMapperNode._discovered_incremental_subscription_gids
    )
    _on_incremental_replay_service = (
        RomaSemanticMapperNode._on_incremental_replay_service
    )

    def __init__(self, *, two_tiles=False, payload_fail_at=None):
        self.events = []
        self.incremental_publish_mode = "full_tiles"
        self.incremental_map_pub = _Publisher(
            "payload",
            self.events,
            subscriptions=1,
            fail_at=payload_fail_at,
        )
        self.incremental_map_batch_pub = _Publisher(
            "batch",
            self.events,
        )
        self.grid = _grid(two_tiles=two_tiles)
        self._map_lock = threading.RLock()
        self.map_revision = 1
        self.map_message_update_id = 0
        self.map_frame = "map"
        self.last_header = _header()
        self.map_transfer_session = MapTransferSession(
            map_id=self.grid.map_id,
            session_id="test-session",
        )
        self._incremental_subscription_count = 1
        self.subscription_gids = [(1,)]
        self._incremental_subscription_gids = frozenset(
            self.subscription_gids
        )
        self._incremental_replay_state = None
        self._incremental_replay_requested = False
        self._incremental_delta_requested = False
        self.logger = _Logger()

    def get_logger(self):
        return self.logger

    def get_subscriptions_info_by_topic(self, _topic_name):
        return [
            types.SimpleNamespace(endpoint_gid=list(gid))
            for gid in self.subscription_gids
        ]


class MapTransferNodeTest(unittest.TestCase):
    def test_delta_is_streamed_one_frozen_tile_per_timer(self):
        harness = _Harness(two_tiles=True)

        harness._request_incremental_delta(harness.last_header)

        self.assertEqual(
            [label for label, _message in harness.events],
            ["batch"],
        )
        self.assertEqual(
            len(harness._incremental_replay_state.pending_keys),
            2,
        )
        harness._on_incremental_replay_timer()
        self.assertEqual(
            [label for label, _message in harness.events],
            ["batch", "payload"],
        )
        harness._on_incremental_replay_timer()

        labels = [label for label, _message in harness.events]
        self.assertEqual(labels, ["batch", "payload", "payload", "batch"])
        self.assertIsNone(harness._incremental_replay_state)
        self.assertEqual(harness.grid.pending_dirty_tile_keys(), ())
        begin = harness.events[0][1]
        complete = harness.events[-1][1]
        payloads = [
            message
            for label, message in harness.events
            if label == "payload"
        ]
        self.assertEqual(begin.phase, SemanticMapBatch.PHASE_BEGIN)
        self.assertEqual(complete.phase, SemanticMapBatch.PHASE_COMPLETE)
        self.assertEqual(
            list(complete.payload_update_id),
            [message.update_id for message in payloads],
        )
        self.assertEqual(complete.tile_count, len(payloads))
        self.assertFalse(complete.full_snapshot)
        self.assertFalse(harness.map_transfer_session.reset_pending)

    def test_completed_old_delta_does_not_clear_newer_tile_revision(self):
        harness = _Harness(two_tiles=False)
        harness._request_incremental_delta(harness.last_header)
        first_revision = harness.grid.tile_revisions[(0, 0)]

        candidate = harness.grid.clone_for_update()
        candidate.update(
            points_map=np.array([[1.2, 1.2, 2.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([0.2, 1.2, 0.5]),
        )
        harness.grid = candidate
        harness.map_revision += 1
        harness._request_incremental_delta(harness.last_header)
        harness._on_incremental_replay_timer()

        self.assertGreater(
            harness.grid.tile_revisions[(0, 0)],
            first_revision,
        )
        self.assertEqual(
            harness.grid.pending_dirty_tile_keys(),
            ((0, 0),),
        )
        self.assertTrue(harness._incremental_delta_requested)

    def test_clean_map_does_not_start_an_empty_delta_batch(self):
        harness = _Harness(two_tiles=True)
        harness.grid.mark_tiles_published(
            harness.grid.pending_dirty_tile_keys()
        )

        harness._request_incremental_delta(harness.last_header)

        self.assertEqual(harness.events, [])
        self.assertIsNone(harness._incremental_replay_state)

    def test_partial_payload_failure_never_emits_complete(self):
        harness = _Harness(two_tiles=True, payload_fail_at=2)

        harness._request_incremental_delta(harness.last_header)
        harness._on_incremental_replay_timer()
        harness._on_incremental_replay_timer()

        batches = [message for label, message in harness.events if label == "batch"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].phase, SemanticMapBatch.PHASE_BEGIN)
        self.assertTrue(harness.map_transfer_session.reset_pending)
        self.assertIsNone(harness._incremental_replay_state)
        self.assertTrue(harness._incremental_delta_requested)
        self.assertEqual(
            harness.grid.pending_dirty_tile_keys(),
            ((0, 0), (1, 0)),
        )

        while (
            harness._incremental_replay_state is not None
            or harness._incremental_delta_requested
        ):
            harness._on_incremental_replay_timer()

        batches = [
            message for label, message in harness.events if label == "batch"
        ]
        self.assertEqual(
            [message.phase for message in batches],
            [
                SemanticMapBatch.PHASE_BEGIN,
                SemanticMapBatch.PHASE_BEGIN,
                SemanticMapBatch.PHASE_COMPLETE,
            ],
        )
        self.assertTrue(batches[1].reset)
        self.assertEqual(harness.grid.pending_dirty_tile_keys(), ())

    def test_replay_streams_a_frozen_copy_on_write_snapshot(self):
        harness = _Harness(two_tiles=False)
        harness._start_incremental_replay()
        state = harness._incremental_replay_state
        self.assertIsNotNone(state)
        replay_revision = state.batch.map_revision

        candidate = harness.grid.clone_for_update()
        candidate.update(
            points_map=np.array([[1.2, 1.2, 2.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([0.2, 1.2, 0.5]),
        )
        harness.grid = candidate
        harness.map_revision += 1

        while harness._incremental_replay_state is not None:
            harness._on_incremental_replay_timer()

        batches = [message for label, message in harness.events if label == "batch"]
        payloads = [message for label, message in harness.events if label == "payload"]
        self.assertEqual(batches[0].phase, SemanticMapBatch.PHASE_BEGIN)
        self.assertEqual(batches[-1].phase, SemanticMapBatch.PHASE_COMPLETE)
        self.assertTrue(batches[-1].full_snapshot)
        self.assertEqual(batches[-1].map_revision, replay_revision)
        self.assertEqual(max(payloads[0].observation_count), 1)
        self.assertEqual(
            max(harness.grid.tiles[(0, 0)].elevation_count.reshape(-1)),
            2,
        )
        self.assertEqual(
            harness.grid.pending_dirty_tile_keys(),
            ((0, 0),),
        )

    def test_rebuild_tombstone_is_explicit_and_all_unknown(self):
        harness = _Harness()
        previous = harness.grid
        rebuilt = TiledSemanticMapManager(
            resolution=1.0,
            tile_cells=4,
            max_rays_per_update=100,
            map_id="test-map",
        )
        rebuilt.update(
            points_map=np.array([[9.2, 1.2, 1.0]]),
            semantic_labels=np.array([4], dtype=np.uint8),
            camera_position_map=np.array([8.2, 1.2, 0.5]),
        )
        rebuilt.mark_tiles_published(rebuilt.pending_dirty_tile_keys())
        rebuilt.mark_rebuild_dirty(previous)
        harness.grid = rebuilt
        harness.map_revision = 2

        harness._request_incremental_delta(harness.last_header)
        while harness._incremental_replay_state is not None:
            harness._on_incremental_replay_timer()

        complete = [
            message
            for label, message in harness.events
            if label == "batch"
            and message.phase == SemanticMapBatch.PHASE_COMPLETE
        ][0]
        payload = [
            message for label, message in harness.events if label == "payload"
        ]
        cleared_indices = [
            index
            for index, cleared in enumerate(complete.tile_cleared)
            if cleared
        ]
        self.assertEqual(len(cleared_indices), 1)
        cleared_index = cleared_indices[0]
        self.assertEqual(
            (complete.tile_x[cleared_index], complete.tile_y[cleared_index]),
            (0, 0),
        )
        payload_by_update_id = {
            message.update_id: message for message in payload
        }
        cleared_payload = payload_by_update_id[
            complete.payload_update_id[cleared_index]
        ]
        self.assertTrue(
            all(value == -1 for value in cleared_payload.occupancy)
        )
        self.assertTrue(
            all(value == 0 for value in cleared_payload.observation_count)
        )

    def test_manual_replay_requires_header_and_payload_consumer(self):
        harness = _Harness()
        response = types.SimpleNamespace(success=None, message="")
        harness.incremental_map_pub.subscriptions = 0
        harness._on_incremental_replay_service(None, response)
        self.assertFalse(response.success)
        self.assertFalse(harness._incremental_replay_requested)

        harness.incremental_map_pub.subscriptions = 1
        harness._on_incremental_replay_service(None, response)
        self.assertTrue(response.success)
        self.assertTrue(harness._incremental_replay_requested)

    def test_equal_count_subscriber_replacement_triggers_full_replay(self):
        harness = _Harness()
        harness.subscription_gids = [(2,)]

        harness._on_incremental_replay_timer()

        batches = [
            message
            for label, message in harness.events
            if label == "batch"
        ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].phase, SemanticMapBatch.PHASE_BEGIN)
        self.assertEqual(batches[1].phase, SemanticMapBatch.PHASE_COMPLETE)
        self.assertTrue(batches[1].full_snapshot)


if __name__ == "__main__":
    unittest.main()
