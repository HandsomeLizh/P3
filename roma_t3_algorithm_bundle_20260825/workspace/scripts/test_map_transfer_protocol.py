#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE / "src" / "t3_semantic_mapping"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from t3_semantic_mapping.map_transfer_protocol import (  # noqa: E402
    MapTransferBatchPlan,
    MapTransferSession,
    TileTransferPlan,
)


def _tile(x: int = 1, y: int = -2, revision: int = 3, cleared=False):
    return TileTransferPlan(
        tile_x=x,
        tile_y=y,
        tile_revision=revision,
        cleared=cleared,
    )


class MapTransferProtocolTest(unittest.TestCase):
    def test_first_completed_batch_resets_then_session_stays_stable(self):
        session = MapTransferSession(map_id="map", session_id="session-a")
        first = session.begin_batch(
            map_revision=4,
            full_snapshot=False,
            tiles=[_tile()],
        )
        self.assertTrue(first.reset)
        self.assertEqual(first.batch_id, 1)
        completed = session.complete_batch(first, [17])
        session.mark_complete(completed)

        second = session.begin_batch(
            map_revision=5,
            full_snapshot=False,
            tiles=[_tile(revision=4)],
        )
        self.assertFalse(second.reset)
        self.assertEqual(second.batch_id, 2)
        self.assertEqual(second.session_id, "session-a")

    def test_incomplete_attempt_does_not_consume_reset(self):
        session = MapTransferSession(map_id="map", session_id="session-a")
        abandoned = session.begin_batch(
            map_revision=1,
            full_snapshot=False,
            tiles=[_tile()],
        )
        retry = session.begin_batch(
            map_revision=1,
            full_snapshot=False,
            tiles=[_tile()],
        )
        self.assertTrue(abandoned.reset)
        self.assertTrue(retry.reset)
        self.assertEqual(retry.batch_id, abandoned.batch_id + 1)

    def test_empty_full_snapshot_is_a_valid_authoritative_reset(self):
        session = MapTransferSession(map_id="map", session_id="session-a")
        plan = session.begin_batch(
            map_revision=0,
            full_snapshot=True,
            tiles=[],
        )
        completed = session.complete_batch(plan, [])
        session.mark_complete(completed)
        self.assertEqual(plan.tiles, ())
        self.assertEqual(completed.payload_update_ids, ())
        self.assertFalse(session.reset_pending)

    def test_empty_delta_and_duplicate_tile_are_rejected(self):
        session = MapTransferSession(map_id="map", session_id="session-a")
        with self.assertRaises(ValueError):
            session.begin_batch(
                map_revision=1,
                full_snapshot=False,
                tiles=[],
            )
        with self.assertRaises(ValueError):
            session.begin_batch(
                map_revision=1,
                full_snapshot=False,
                tiles=[_tile(), _tile(revision=4)],
            )

    def test_completion_requires_one_unique_uint32_id_per_tile(self):
        session = MapTransferSession(map_id="map", session_id="session-a")
        plan = session.begin_batch(
            map_revision=1,
            full_snapshot=False,
            tiles=[_tile(1), _tile(2)],
        )
        for invalid in ([1], [1, 1], [1, 1 << 32], [1, 2.5]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    session.complete_batch(plan, invalid)

    def test_foreign_session_cannot_complete_or_acknowledge(self):
        first = MapTransferSession(map_id="map", session_id="session-a")
        second = MapTransferSession(map_id="map", session_id="session-b")
        plan = first.begin_batch(
            map_revision=1,
            full_snapshot=False,
            tiles=[_tile()],
        )
        completed = first.complete_batch(plan, [1])
        with self.assertRaises(ValueError):
            second.complete_batch(plan, [1])
        with self.assertRaises(ValueError):
            second.mark_complete(completed)

    def test_invalid_session_and_boolean_coercions_fail_fast(self):
        with self.assertRaises(ValueError):
            MapTransferSession(map_id="map", session_id="")
        session = MapTransferSession(map_id="map", session_id="session-a")
        with self.assertRaises(ValueError):
            session.begin_batch(
                map_revision=1,
                full_snapshot="yes",
                tiles=[_tile()],
            )
        with self.assertRaises(ValueError):
            TileTransferPlan(1, 2, 3, cleared=1)

    def test_integer_ranges_and_foreign_plan_values_are_validated(self):
        for kwargs in (
            {"tile_x": 1 << 31, "tile_y": 0, "tile_revision": 0},
            {"tile_x": 0, "tile_y": -(1 << 31) - 1, "tile_revision": 0},
            {"tile_x": 0, "tile_y": 0, "tile_revision": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TileTransferPlan(**kwargs)
        with self.assertRaises(ValueError):
            MapTransferBatchPlan(
                map_id="map",
                session_id="session",
                map_revision=0,
                batch_id=1,
                reset=True,
                full_snapshot=False,
                tiles=(object(),),
            )


if __name__ == "__main__":
    unittest.main()
