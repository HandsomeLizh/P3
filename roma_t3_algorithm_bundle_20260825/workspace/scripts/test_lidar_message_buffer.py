#!/usr/bin/env python3
"""Deterministic tests for the thread-safe LiDAR message buffer."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.lidar_message_buffer import LidarMessageBuffer


class LidarMessageBufferConfigurationTest(unittest.TestCase):
    def test_invalid_configuration_fails_fast(self):
        invalid = (
            {"max_size": 0},
            {"max_time_offset_sec": -0.1},
            {"max_time_offset_sec": float("nan")},
            {"wait_timeout_sec": -0.1},
            {"wait_timeout_sec": float("inf")},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LidarMessageBuffer(**kwargs)

    def test_nonpositive_message_or_target_timestamp_is_rejected(self):
        buffer = LidarMessageBuffer()
        with self.assertRaises(ValueError):
            buffer.add(0, "invalid")
        with self.assertRaises(ValueError):
            buffer.nearest(0)


class LidarMessageBufferSelectionTest(unittest.TestCase):
    def test_add_sorts_replaces_duplicates_and_keeps_newest_timestamps(self):
        buffer = LidarMessageBuffer(
            max_size=3,
            max_time_offset_sec=1.0,
            wait_timeout_sec=0.0,
        )
        buffer.add(300, "three-old")
        buffer.add(100, "one")
        buffer.add(200, "two")
        buffer.add(300, "three-new")
        buffer.add(400, "four")

        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.time_range_ns, (200, 400))
        self.assertEqual(buffer.nearest(300), ("three-new", 0.0))

    def test_waits_for_successor_before_selecting_true_nearest(self):
        target = 1_000_000_000
        buffer = LidarMessageBuffer(
            max_time_offset_sec=0.2,
            wait_timeout_sec=0.2,
        )
        buffer.add(900_000_000, "past")

        def publish_successor():
            time.sleep(0.02)
            buffer.add(1_040_000_000, "future")

        worker = threading.Thread(target=publish_successor)
        worker.start()
        message, offset = buffer.nearest(target)
        worker.join()

        self.assertEqual(message, "future")
        self.assertAlmostEqual(offset, 0.04)

    def test_equal_distance_prefers_predecessor_deterministically(self):
        buffer = LidarMessageBuffer(
            max_time_offset_sec=0.2,
            wait_timeout_sec=0.0,
        )
        buffer.add(900_000_000, "past")
        buffer.add(1_100_000_000, "future")
        message, offset = buffer.nearest(1_000_000_000)
        self.assertEqual(message, "past")
        self.assertAlmostEqual(offset, -0.1)

    def test_stream_tail_uses_valid_predecessor_after_timeout(self):
        buffer = LidarMessageBuffer(
            max_time_offset_sec=0.2,
            wait_timeout_sec=0.005,
        )
        buffer.add(900_000_000, "past")
        message, offset = buffer.nearest(1_000_000_000)
        self.assertEqual(message, "past")
        self.assertAlmostEqual(offset, -0.1)

    def test_out_of_gate_selection_is_explicitly_empty(self):
        buffer = LidarMessageBuffer(
            max_time_offset_sec=0.05,
            wait_timeout_sec=0.0,
        )
        buffer.add(900_000_000, "past")
        message, offset = buffer.nearest(1_000_000_000)
        self.assertIsNone(message)
        self.assertAlmostEqual(offset, -0.1)

    def test_empty_buffer_returns_infinite_offset(self):
        buffer = LidarMessageBuffer(wait_timeout_sec=0.0)
        message, offset = buffer.nearest(1_000_000_000)
        self.assertIsNone(message)
        self.assertEqual(offset, float("inf"))


if __name__ == "__main__":
    unittest.main()
