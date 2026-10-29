#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from std_msgs.msg import Header
from t3_interfaces.msg import SemanticMapBatch


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE / "src" / "t3_semantic_mapping"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from t3_semantic_mapping.map_transfer_message import (  # noqa: E402
    make_map_batch_begin,
    make_map_batch_complete,
)
from t3_semantic_mapping.map_transfer_protocol import (  # noqa: E402
    MapTransferSession,
    TileTransferPlan,
)


class MapTransferMessageTest(unittest.TestCase):
    def setUp(self):
        self.header = Header()
        self.header.stamp.sec = 12
        self.header.stamp.nanosec = 34
        self.header.frame_id = "map"
        self.session = MapTransferSession(
            map_id="task3-map",
            session_id="fixed-session",
        )

    def test_begin_announces_exact_tiles_without_payload_ids(self):
        plan = self.session.begin_batch(
            map_revision=7,
            full_snapshot=False,
            tiles=[
                TileTransferPlan(2, -1, 4, False),
                TileTransferPlan(3, -1, 9, True),
            ],
        )
        message = make_map_batch_begin(header=self.header, plan=plan)

        self.assertEqual(message.phase, SemanticMapBatch.PHASE_BEGIN)
        self.assertEqual(message.header.frame_id, "map")
        self.assertEqual(message.map_id, "task3-map")
        self.assertEqual(message.session_id, "fixed-session")
        self.assertEqual(message.map_revision, 7)
        self.assertEqual(message.batch_id, 1)
        self.assertTrue(message.reset)
        self.assertFalse(message.full_snapshot)
        self.assertEqual(message.tile_count, 2)
        self.assertEqual(list(message.tile_x), [2, 3])
        self.assertEqual(list(message.tile_y), [-1, -1])
        self.assertEqual(list(message.tile_revision), [4, 9])
        self.assertEqual(list(message.tile_cleared), [False, True])
        self.assertEqual(list(message.payload_update_id), [])

    def test_complete_correlates_every_legacy_payload(self):
        plan = self.session.begin_batch(
            map_revision=8,
            full_snapshot=True,
            tiles=[
                TileTransferPlan(-2, 5, 11, False),
                TileTransferPlan(-1, 5, 12, False),
            ],
        )
        completed = self.session.complete_batch(plan, [101, 102])
        message = make_map_batch_complete(
            header=self.header,
            batch=completed,
        )

        self.assertEqual(message.phase, SemanticMapBatch.PHASE_COMPLETE)
        self.assertTrue(message.full_snapshot)
        self.assertEqual(list(message.payload_update_id), [101, 102])
        self.assertEqual(message.tile_count, 2)

    def test_empty_full_snapshot_is_a_complete_reset_message(self):
        plan = self.session.begin_batch(
            map_revision=0,
            full_snapshot=True,
            tiles=[],
        )
        completed = self.session.complete_batch(plan, [])
        begin = make_map_batch_begin(header=self.header, plan=plan)
        end = make_map_batch_complete(header=self.header, batch=completed)

        self.assertTrue(begin.reset)
        self.assertTrue(begin.full_snapshot)
        self.assertEqual(begin.tile_count, 0)
        self.assertEqual(end.phase, SemanticMapBatch.PHASE_COMPLETE)
        self.assertEqual(list(end.payload_update_id), [])


if __name__ == "__main__":
    unittest.main()
