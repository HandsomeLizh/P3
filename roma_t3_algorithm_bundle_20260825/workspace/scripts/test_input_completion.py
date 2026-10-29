#!/usr/bin/env python3
"""Deterministic tests for mapper receipts and lossless playback ledgers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t1_dataset_sim"))
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t1_dataset_sim.playback_monitor import PlaybackMonitor
from message_filters import SimpleFilter
from t3_semantic_mapping.input_processing_monitor import (
    CallbackStageTimer,
    InputProcessingMonitor,
)
from t3_semantic_mapping.input_transport import TimestampSourceFilter


class TimestampSourceFilterTest(unittest.TestCase):
    def test_arrival_rewrite_preserves_source_stamp_for_files_only(self):
        source = SimpleFilter()
        wrapper = TimestampSourceFilter(
            source,
            timestamp_source="arrival",
            now_callback=lambda: SimpleNamespace(sec=0, nanosec=0),
        )
        delivered = []
        wrapper.registerCallback(delivered.append)
        message = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=123, nanosec=456)
            )
        )
        source.signalMessage(message)
        self.assertIs(delivered[0], message)
        self.assertEqual(message.header.stamp.sec, 0)
        self.assertEqual(
            wrapper.pop_source_stamp_ns(message), 123_000_000_456
        )
        self.assertIsNone(wrapper.pop_source_stamp_ns(message))


class CallbackStageTimerTest(unittest.TestCase):
    def test_switches_classify_the_complete_callback(self):
        timer = CallbackStageTimer(10.0)
        timer.switch("visual_tracking", 10.2)
        timer.switch("localization_and_pose", 11.0)
        timer.switch("map_update_and_publish", 11.1)
        timer.switch("callback_bookkeeping", 11.5)
        durations = timer.finish(11.6)

        self.assertAlmostEqual(durations["input_decode"], 0.2)
        self.assertAlmostEqual(durations["visual_tracking"], 0.8)
        self.assertAlmostEqual(durations["localization_and_pose"], 0.1)
        self.assertAlmostEqual(durations["map_update_and_publish"], 0.4)
        self.assertAlmostEqual(durations["callback_bookkeeping"], 0.1)
        self.assertAlmostEqual(sum(durations.values()), 1.6)

    def test_invalid_stage_state_and_time_fail_fast(self):
        with self.assertRaises(ValueError):
            CallbackStageTimer(-1.0)
        with self.assertRaises(ValueError):
            CallbackStageTimer(1.0, initial_stage="unknown")

        timer = CallbackStageTimer(1.0)
        with self.assertRaises(ValueError):
            timer.switch("visual_tracking", 0.9)
        with self.assertRaises(ValueError):
            timer.switch("unknown", 1.1)
        timer.finish(1.2)
        with self.assertRaises(RuntimeError):
            timer.finish(1.3)


class InputProcessingMonitorTest(unittest.TestCase):
    def test_complete_callbacks_preserve_outcome_and_stream_counts(self):
        monitor = InputProcessingMonitor()
        for stamp_ns in (1_000_000_000, 2_000_000_000):
            for source in ("left", "right", "semantic"):
                monitor.record_delivered(source, stamp_ns)

        monitor.record_synchronized(1_000_000_000, wall_time=10.0)
        monitor.record_processing_started()
        monitor.record_terminal(
            "completed", wall_time=11.0, duration_sec=1.0
        )
        monitor.record_completion_notification()
        monitor.record_synchronized(2_000_000_000, wall_time=12.0)
        monitor.record_terminal("stereo_skew_rejected", wall_time=12.0)
        monitor.record_completion_notification()

        summary = monitor.summary(
            processing_in_progress=False,
            valid_pose_count=1,
            tracking_or_processing_failure_count=0,
            map_revision=1,
        )
        self.assertEqual(summary["synchronized_callbacks"], 2)
        self.assertEqual(summary["processing_started"], 1)
        self.assertEqual(summary["terminal_callbacks"], 2)
        self.assertEqual(
            summary["terminal_outcomes"]["stereo_skew_rejected"], 1
        )
        self.assertTrue(summary["complete_input_consumption"])
        self.assertTrue(
            summary["invariants"][
                "all_synchronized_callbacks_terminated"
            ]
        )
        self.assertTrue(
            summary["invariants"][
                "one_completion_notification_per_synchronized_callback"
            ]
        )
        self.assertTrue(
            summary["invariants"][
                "processing_started_matches_non_rejected_callbacks"
            ]
        )
        self.assertTrue(
            summary["invariants"]["delivered_stream_counts_equal"]
        )
        self.assertTrue(
            summary["invariants"][
                "all_delivered_messages_synchronized"
            ]
        )
        self.assertFalse(
            summary["invariants"]["processing_in_progress"]
        )

    def test_missing_terminal_or_notification_is_explicitly_incomplete(self):
        monitor = InputProcessingMonitor()
        monitor.record_synchronized(100, wall_time=1.0)
        monitor.record_processing_started()
        summary = monitor.summary(
            processing_in_progress=True,
            valid_pose_count=0,
            tracking_or_processing_failure_count=0,
            map_revision=0,
        )
        self.assertFalse(summary["complete_input_consumption"])
        self.assertFalse(
            summary["invariants"][
                "all_synchronized_callbacks_terminated"
            ]
        )
        self.assertFalse(
            summary["invariants"][
                "one_completion_notification_per_synchronized_callback"
            ]
        )

    def test_duration_percentile_window_is_bounded(self):
        monitor = InputProcessingMonitor(duration_window_size=3)
        for index, duration in enumerate((1.0, 2.0, 3.0, 4.0)):
            monitor.record_synchronized(index, wall_time=float(index))
            monitor.record_processing_started()
            monitor.record_terminal(
                "completed",
                wall_time=float(index) + duration,
                duration_sec=duration,
            )
            monitor.record_completion_notification()
        durations = monitor.summary(
            processing_in_progress=False,
            valid_pose_count=4,
            tracking_or_processing_failure_count=0,
            map_revision=1,
        )["processing_duration_sec"]
        self.assertEqual(durations["count"], 4)
        self.assertEqual(durations["mean"], 2.5)
        self.assertEqual(durations["recent_window_count"], 3)
        self.assertEqual(durations["recent_p95"], 4.0)

    def test_stage_duration_summary_is_bounded_and_complete(self):
        monitor = InputProcessingMonitor(duration_window_size=2)
        samples = (
            {
                "input_decode": 0.1,
                "visual_tracking": 0.7,
                "localization_and_pose": 0.1,
                "map_update_and_publish": 0.3,
                "callback_bookkeeping": 0.1,
            },
            {
                "input_decode": 0.2,
                "visual_tracking": 0.8,
                "localization_and_pose": 0.1,
                "map_update_and_publish": 0.4,
                "callback_bookkeeping": 0.1,
            },
            {
                "input_decode": 0.3,
                "visual_tracking": 0.9,
                "localization_and_pose": 0.1,
                "map_update_and_publish": 0.5,
                "callback_bookkeeping": 0.1,
            },
        )
        for index, stages in enumerate(samples):
            duration = sum(stages.values())
            monitor.record_synchronized(index, wall_time=float(index))
            monitor.record_processing_started()
            monitor.record_terminal(
                "completed",
                wall_time=float(index) + duration,
                duration_sec=duration,
                stage_durations_sec=stages,
            )
            monitor.record_completion_notification()

        timing = monitor.summary(
            processing_in_progress=False,
            valid_pose_count=3,
            tracking_or_processing_failure_count=0,
            map_revision=2,
        )["processing_stage_duration_sec"]
        self.assertEqual(timing["classified_callback_count"], 3)
        visual = timing["stages"]["visual_tracking"]
        self.assertEqual(visual["count"], 3)
        self.assertAlmostEqual(visual["mean"], 0.8)
        self.assertAlmostEqual(visual["recent_p95"], 0.9)
        self.assertEqual(visual["recent_window_count"], 2)
        fractions = [
            stage["fraction_of_classified_duration"]
            for stage in timing["stages"].values()
        ]
        self.assertAlmostEqual(sum(fractions), 1.0)

    def test_incomplete_or_unknown_stage_accounting_fails_fast(self):
        monitor = InputProcessingMonitor()
        with self.assertRaisesRegex(ValueError, "total duration_sec"):
            monitor.record_terminal(
                "completed",
                wall_time=1.0,
                stage_durations_sec={"input_decode": 0.1},
            )
        with self.assertRaisesRegex(ValueError, "unknown processing stage"):
            monitor.record_terminal(
                "completed",
                wall_time=1.0,
                duration_sec=0.1,
                stage_durations_sec={"unknown": 0.1},
            )
        with self.assertRaisesRegex(ValueError, "complete callback"):
            monitor.record_terminal(
                "completed",
                wall_time=1.0,
                duration_sec=1.0,
                stage_durations_sec={"input_decode": 0.5},
            )
        self.assertEqual(monitor.terminal_count, 0)
        summary = monitor.summary(
            processing_in_progress=False,
            valid_pose_count=0,
            tracking_or_processing_failure_count=0,
            map_revision=0,
        )
        self.assertEqual(summary["processing_duration_sec"]["count"], 0)

    def test_valid_frame_metrics_are_exact_and_bounded(self):
        monitor = InputProcessingMonitor(duration_window_size=3)
        for index in range(12):
            is_cold_start = index == 0
            monitor.record_synchronized(index, wall_time=float(index))
            monitor.record_processing_started()
            monitor.record_terminal(
                "completed",
                wall_time=float(index) + 2.0,
                duration_sec=5.0 if is_cold_start else 2.0,
                sensor_stamp_ns=index,
                pose_publish_latency_sec=4.0 if is_cold_start else 1.5,
                map_update_latency_sec=0.3,
            )

        summary = monitor.summary(
            processing_in_progress=False,
            valid_pose_count=12,
            tracking_or_processing_failure_count=0,
            map_revision=12,
        )
        self.assertEqual(summary["schema_version"], 2)
        metrics = summary["test_metric_measurements"]
        self.assertEqual(metrics["valid_frame_count"], 12)
        self.assertEqual(metrics["pose_publish_latency_sec"]["max"], 4.0)
        self.assertEqual(metrics["valid_end_to_end_latency_sec"]["max"], 5.0)
        self.assertEqual(
            metrics["pose_publish_latency_sec"]["recent_window_count"], 3
        )
        sample = metrics["document_random_sample"]
        self.assertEqual(sample["source_valid_frame_count"], 12)
        self.assertEqual(sample["excluded_initial_valid_frame_count"], 1)
        self.assertEqual(sample["population_count"], 11)
        self.assertEqual(sample["sample_count"], 10)
        sampled_indices = {
            row["valid_frame_index"] for row in sample["samples"]
        }
        self.assertEqual(len(sampled_indices), 10)
        self.assertNotIn(1, sampled_indices)
        self.assertTrue(sampled_indices.issubset(set(range(2, 13))))

        self.assertTrue(
            all(
                row["pose_publish_latency_sec"] == 1.5
                for row in sample["samples"]
            )
        )
    def test_invalid_monitor_inputs_fail_fast(self):
        with self.assertRaises(ValueError):

            InputProcessingMonitor(duration_window_size=0)
        monitor = InputProcessingMonitor()
        with self.assertRaises(ValueError):
            monitor.record_delivered("depth", 1)
        with self.assertRaises(ValueError):
            monitor.record_terminal("unknown", wall_time=0.0)
        with self.assertRaisesRegex(ValueError, "recorded together"):
            monitor.record_terminal(
                "completed",
                wall_time=1.0,
                duration_sec=1.0,
                pose_publish_latency_sec=0.5,
            )


class PlaybackMonitorTest(unittest.TestCase):
    @staticmethod
    def _monitor() -> PlaybackMonitor:
        return PlaybackMonitor(
            expected_frame_count=2,
            expected_first_stamp_ns=1_000_000_000,
            expected_last_stamp_ns=2_000_000_000,
            requested_playback_rate=10.0,
            lossless_playback=True,
        )

    def test_exact_publication_and_acknowledgement_is_complete(self):
        monitor = self._monitor()
        monitor.record_frame_attempt(
            1_000_000_000,
            published=True,
            wall_time=10.0,
            schedule_lateness_sec=0.1,
        )
        self.assertTrue(monitor.acknowledge(1_000_000_000, 10.8))
        monitor.record_frame_attempt(
            2_000_000_000,
            published=True,
            wall_time=11.0,
            schedule_lateness_sec=0.9,
        )
        self.assertTrue(monitor.acknowledge(2_000_000_000, 11.8))
        monitor.mark_publication_finished()

        summary = monitor.summary()
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["complete_input_consumption"])
        self.assertEqual(summary["published_frame_count"], 2)
        self.assertEqual(summary["acknowledged_frame_count"], 2)
        self.assertAlmostEqual(summary["effective_playback_rate_x"], 1.0)
        self.assertAlmostEqual(
            summary["frame_schedule_lateness_sec"]["max"], 0.9
        )

    def test_missing_and_duplicate_acknowledgements_are_visible(self):
        monitor = self._monitor()
        for index, stamp_ns in enumerate((1_000_000_000, 2_000_000_000)):
            monitor.record_frame_attempt(
                stamp_ns,
                published=True,
                wall_time=10.0 + index,
                schedule_lateness_sec=0.0,
            )
        self.assertTrue(monitor.acknowledge(1_000_000_000, 10.5))
        self.assertFalse(monitor.acknowledge(1_000_000_000, 10.6))
        monitor.mark_publication_finished()

        summary = monitor.summary()
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["unacknowledged_frame_count"], 1)
        self.assertEqual(summary["unexpected_completion_notifications"], 1)
        self.assertEqual(
            summary["unacknowledged"],
            [{"stamp_ns": 2_000_000_000, "count": 1}],
        )

    def test_publish_failure_and_ack_timeout_are_not_complete(self):
        monitor = self._monitor()
        monitor.record_frame_attempt(
            1_000_000_000,
            published=False,
            wall_time=10.0,
            schedule_lateness_sec=0.0,
        )
        monitor.mark_failed("frame read failed")
        summary = monitor.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frame_publish_failure_count"], 1)
        self.assertEqual(summary["failure_reason"], "frame read failed")
        self.assertFalse(summary["complete_input_consumption"])

    def test_atomic_json_summaries_are_parseable(self):
        input_monitor = InputProcessingMonitor()
        input_payload = input_monitor.summary(
            processing_in_progress=False,
            valid_pose_count=0,
            tracking_or_processing_failure_count=0,
            map_revision=0,
        )
        playback_monitor = self._monitor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            playback_path = root / "playback.json"
            input_monitor.write_summary(input_path, input_payload)
            playback_monitor.write_summary(
                playback_path, playback_monitor.summary()
            )
            self.assertEqual(
                json.loads(input_path.read_text(encoding="utf-8")),
                input_payload,
            )
            self.assertEqual(
                json.loads(playback_path.read_text(encoding="utf-8")),
                playback_monitor.summary(),
            )


if __name__ == "__main__":
    unittest.main()
