#!/usr/bin/env python3
"""Unit tests for mapper input QoS and header-stamp synchronisation."""

from __future__ import annotations

import unittest

from message_filters import SimpleFilter
from builtin_interfaces.msg import Time
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image

from t3_semantic_mapping.input_transport import (
    InputTransportConfiguration,
    TimestampSourceFilter,
)


def _image(stamp_ns: int) -> Image:
    message = Image()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return message


class InputTransportTest(unittest.TestCase):
    def test_defaults_preserve_existing_transport(self):
        config = InputTransportConfiguration()
        self.assertEqual(config.sync_mode, "exact")
        self.assertEqual(config.sync_queue_size, 50)
        self.assertEqual(config.stereo_max_skew_sec, 0.0)
        self.assertEqual(config.stereo_qos.depth, 10)
        self.assertEqual(config.imu_qos.depth, 10)
        self.assertEqual(
            config.stereo_qos.reliability, ReliabilityPolicy.RELIABLE
        )
        self.assertEqual(
            config.semantic_qos.reliability, ReliabilityPolicy.RELIABLE
        )
        self.assertEqual(
            config.imu_qos.reliability, ReliabilityPolicy.RELIABLE
        )
        self.assertEqual(
            config.stereo_qos.durability, DurabilityPolicy.VOLATILE
        )

    def test_best_effort_is_selectable_per_source(self):
        config = InputTransportConfiguration(
            semantic_reliability=" BEST_EFFORT ",
            imu_reliability="best_effort",
        )
        self.assertEqual(config.stereo_reliability, "reliable")
        self.assertEqual(
            config.semantic_qos.reliability,
            ReliabilityPolicy.BEST_EFFORT,
        )
        self.assertEqual(
            config.imu_qos.reliability,
            ReliabilityPolicy.BEST_EFFORT,
        )

    def test_invalid_configuration_fails_fast(self):
        invalid_kwargs = (
            {"stereo_reliability": "sometimes"},
            {"sync_mode": "loose"},
            {"image_qos_depth": 0},
            {"imu_qos_depth": 0},
            {"sync_queue_size": 0},
            {"sync_slop_sec": -0.001},
            {"stereo_max_skew_sec": -0.001},
            {"sync_mode": "approximate", "sync_slop_sec": 0.0},
            {
                "sync_mode": "approximate",
                "sync_slop_sec": 0.01,
                "stereo_max_skew_sec": 0.02,
            },
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InputTransportConfiguration(**kwargs)

    def test_exact_mode_requires_identical_header_stamps(self):
        filters = [SimpleFilter(), SimpleFilter(), SimpleFilter()]
        synchronizer = InputTransportConfiguration().create_synchronizer(
            filters
        )
        callbacks = []
        synchronizer.registerCallback(lambda *messages: callbacks.append(messages))

        filters[0].signalMessage(_image(1_000_000_000))
        filters[1].signalMessage(_image(1_000_000_000))
        filters[2].signalMessage(_image(1_005_000_000))
        self.assertEqual(callbacks, [])

        filters[2].signalMessage(_image(1_000_000_000))
        self.assertEqual(len(callbacks), 1)

    def test_approximate_mode_accepts_only_within_slop(self):
        filters = [SimpleFilter(), SimpleFilter(), SimpleFilter()]
        config = InputTransportConfiguration(
            sync_mode="approximate", sync_slop_sec=0.01
        )
        synchronizer = config.create_synchronizer(filters)
        callbacks = []
        synchronizer.registerCallback(lambda *messages: callbacks.append(messages))

        filters[0].signalMessage(_image(2_000_000_000))
        filters[1].signalMessage(_image(2_004_000_000))
        filters[2].signalMessage(_image(2_009_000_000))
        self.assertEqual(len(callbacks), 1)

        filters[0].signalMessage(_image(3_000_000_000))
        filters[1].signalMessage(_image(3_001_000_000))
        filters[2].signalMessage(_image(3_011_000_000))
        self.assertEqual(len(callbacks), 1)

    def test_two_image_sync_supports_visual_only_mode(self):
        filters = [SimpleFilter(), SimpleFilter()]
        config = InputTransportConfiguration(
            sync_mode="approximate",
            sync_slop_sec=0.02,
            stereo_max_skew_sec=0.02,
        )
        synchronizer = config.create_synchronizer(filters)
        callbacks = []
        synchronizer.registerCallback(
            lambda *messages: callbacks.append(messages)
        )

        filters[0].signalMessage(_image(4_000_000_000))
        filters[1].signalMessage(_image(4_015_000_000))
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(len(callbacks[0]), 2)

    def test_arrival_timestamp_filter_is_explicit(self):
        source = SimpleFilter()
        arrival = Time(sec=9, nanosec=123)
        filtered = TimestampSourceFilter(
            source,
            timestamp_source="arrival",
            now_callback=lambda: arrival,
        )
        callbacks = []
        filtered.registerCallback(callbacks.append)

        source.signalMessage(_image(0))
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0].header.stamp, arrival)

        preserved_source = SimpleFilter()
        preserved = TimestampSourceFilter(
            preserved_source,
            timestamp_source="sensor",
            now_callback=lambda: arrival,
        )
        preserved_callbacks = []
        preserved.registerCallback(preserved_callbacks.append)
        preserved_source.signalMessage(_image(5_000_000_000))
        self.assertEqual(
            preserved_callbacks[0].header.stamp.sec,
            5,
        )

    def test_approximate_semantics_does_not_relax_stereo_by_default(self):
        config = InputTransportConfiguration(
            sync_mode="approximate", sync_slop_sec=0.02
        )
        self.assertTrue(config.accepts_stereo_skew(1_000, 1_000))
        self.assertFalse(config.accepts_stereo_skew(1_000, 1_001))

        relaxed = InputTransportConfiguration(
            sync_mode="approximate",
            sync_slop_sec=0.02,
            stereo_max_skew_sec=0.001,
        )
        self.assertTrue(
            relaxed.accepts_stereo_skew(1_000_000_000, 1_001_000_000)
        )
        self.assertFalse(
            relaxed.accepts_stereo_skew(1_000_000_000, 1_001_000_001)
        )


if __name__ == "__main__":
    unittest.main()
