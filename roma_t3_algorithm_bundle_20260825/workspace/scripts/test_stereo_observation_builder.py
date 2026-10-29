#!/usr/bin/env python3
"""Tests for mapper-independent stereo keyframe observation preparation."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t3_semantic_mapping.stereo_observation_builder import (
    StereoObservationBuilder,
    StereoObservationConfiguration,
)


class _SnapshotSource:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot

    def get_mapping_keyframe_snapshot(self, _keyframe_id):
        return self.snapshot


def _snapshot(keypoints, camera_points, *, is_bad=False):
    return SimpleNamespace(
        keypoints=np.asarray(keypoints, dtype=np.float64),
        camera_points=np.asarray(camera_points, dtype=np.float64),
        is_bad=bool(is_bad),
    )


def _semantic(height=80, width=100):
    image = np.zeros((height, width), dtype=np.uint8)
    image[40, 50] = 4
    image[40, 60] = 1
    return image


class StereoObservationConfigurationTest(unittest.TestCase):
    def test_defaults_and_normalisation(self):
        config = StereoObservationConfiguration(
            visible_max_points=7.9,
            assist_pixel_radius=2,
            assist_min_matches=3.9,
        )
        self.assertEqual(config.visible_max_points, 7)
        self.assertEqual(config.stereo_map_max_depth_m, 5.0)
        self.assertEqual(config.assist_pixel_radius, 2.0)
        self.assertEqual(config.assist_min_matches, 3)

    def test_invalid_assist_configuration_fails_fast(self):
        invalid = (
            {"stereo_map_max_depth_m": -1.0},
            {"stereo_map_max_depth_m": float("nan")},
            {"assist_pixel_radius": -1.0},
            {"assist_pixel_radius": float("nan")},
            {"assist_unmatched_stereo_max_depth_m": float("inf")},
            {"assist_replace_stereo_min_depth_m": float("nan")},
            {"assist_min_matches": 0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StereoObservationConfiguration(**kwargs)


class StereoComponentSamplingTest(unittest.TestCase):
    def test_missing_snapshot_api_snapshot_or_alignment_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not expose"):
            StereoObservationBuilder(object()).components(
                keyframe_id=1,
                semantic_image=np.zeros((2, 2), dtype=np.uint8),
            )
        with self.assertRaisesRegex(RuntimeError, "Missing keyframe"):
            StereoObservationBuilder(_SnapshotSource()).components(
                keyframe_id=2,
                semantic_image=np.zeros((2, 2), dtype=np.uint8),
            )
        bad = _SnapshotSource(_snapshot([[0, 0]], [[0, 0, 1]], is_bad=True))
        with self.assertRaisesRegex(RuntimeError, "already bad"):
            StereoObservationBuilder(bad).components(
                keyframe_id=3,
                semantic_image=np.zeros((2, 2), dtype=np.uint8),
            )
        unaligned = _SnapshotSource(
            _snapshot([[0, 0], [1, 1]], [[0, 0, 1]])
        )
        with self.assertRaisesRegex(ValueError, "not aligned"):
            StereoObservationBuilder(unaligned).components(
                keyframe_id=4,
                semantic_image=np.zeros((2, 2), dtype=np.uint8),
            )

    def test_nearest_depth_per_pixel_and_validity_rules_are_preserved(self):
        semantic = np.zeros((4, 4), dtype=np.uint8)
        semantic[1, 1] = 4
        semantic[2, 2] = 1
        source = _SnapshotSource(
            _snapshot(
                [[1.1, 1.1], [1.2, 1.2], [2.0, 2.0], [-1.0, 0.0]],
                [[0, 0, 4], [0, 0, 2], [1, 0, 3], [0, 0, 1]],
            )
        )
        keypoints, points, labels = StereoObservationBuilder(
            source
        ).components(keyframe_id=5, semantic_image=semantic)

        np.testing.assert_allclose(keypoints, [[1.2, 1.2], [2.0, 2.0]])
        np.testing.assert_allclose(points, [[0, 0, 2], [1, 0, 3]])
        np.testing.assert_array_equal(labels, [4, 1])

    def test_visible_point_cap_is_deterministic(self):
        source = _SnapshotSource(
            _snapshot(
                [[0, 0], [1, 0], [2, 0]],
                [[0, 0, 1], [0, 0, 2], [0, 0, 3]],
            )
        )
        builder = StereoObservationBuilder(
            source,
            StereoObservationConfiguration(visible_max_points=2),
        )
        keypoints, points, _ = builder.components(
            keyframe_id=6,
            semantic_image=np.zeros((2, 3), dtype=np.uint8),
        )
        np.testing.assert_allclose(keypoints, [[0, 0], [2, 0]])
        np.testing.assert_allclose(points[:, 2], [1, 3])


class StereoObservationCreationTest(unittest.TestCase):
    def test_plain_stereo_observation_preserves_timestamp_labels_and_source(self):
        source = _SnapshotSource(
            _snapshot([[50, 40]], [[0, 0, 3]])
        )
        observation = StereoObservationBuilder(source).make_stereo(
            keyframe_id=7,
            semantic_image=_semantic(),
            sensor_timestamp=12.5,
        )
        self.assertEqual(observation.keyframe_id, 7)
        self.assertEqual(observation.sensor_timestamp, 12.5)
        self.assertEqual(observation.source, "stereo")
        np.testing.assert_array_equal(observation.semantic_labels, [4])

    def test_plain_stereo_depth_gate_keeps_near_evidence_before_point_cap(self):
        semantic = np.array([[1, 2, 3, 4]], dtype=np.uint8)
        source = _SnapshotSource(
            _snapshot(
                [[0, 0], [1, 0], [2, 0], [3, 0]],
                [[0, 0, 2], [0, 0, 5], [0, 0, 8], [0, 0, 9]],
            )
        )
        builder = StereoObservationBuilder(
            source,
            StereoObservationConfiguration(visible_max_points=2),
        )

        observation = builder.make_stereo(
            keyframe_id=8,
            semantic_image=semantic,
            sensor_timestamp=8.0,
        )

        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.camera_points[:, 2], [2, 5])
        np.testing.assert_array_equal(observation.semantic_labels, [1, 2])

    def test_disabled_plain_stereo_depth_gate_preserves_far_evidence(self):
        source = _SnapshotSource(
            _snapshot([[50, 40], [60, 40]], [[0, 0, 3], [1, 0, 8]])
        )
        builder = StereoObservationBuilder(
            source,
            StereoObservationConfiguration(stereo_map_max_depth_m=0.0),
        )

        observation = builder.make_stereo(
            keyframe_id=8,
            semantic_image=_semantic(),
            sensor_timestamp=8.0,
        )

        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.camera_points[:, 2], [3, 8])

    def test_plain_stereo_without_reliable_map_evidence_returns_none(self):
        source = _SnapshotSource(_snapshot([[50, 40]], [[0, 0, 8]]))

        observation = StereoObservationBuilder(source).make_stereo(
            keyframe_id=8,
            semantic_image=_semantic(),
            sensor_timestamp=8.0,
        )

        self.assertIsNone(observation)

    def test_assist_fallback_keeps_only_configured_near_range(self):
        source = _SnapshotSource(
            _snapshot(
                [[50, 40], [60, 40]],
                [[0, 0, 3], [1, 0, 8]],
            )
        )
        observation, statistics = StereoObservationBuilder(source).make_fallback(
            keyframe_id=8,
            semantic_image=_semantic(),
            sensor_timestamp=8.0,
        )
        self.assertEqual(observation.source, "pointcloud_assisted_stereo_fallback")
        np.testing.assert_allclose(observation.camera_points, [[0, 0, 3]])
        np.testing.assert_array_equal(observation.semantic_labels, [4])
        self.assertEqual(statistics["input_stereo_points"], 2)
        self.assertEqual(statistics["output_points"], 1)

    def test_disabled_fallback_depth_gate_keeps_all_points(self):
        source = _SnapshotSource(
            _snapshot([[50, 40], [60, 40]], [[0, 0, 3], [1, 0, 8]])
        )
        config = StereoObservationConfiguration(
            assist_unmatched_stereo_max_depth_m=0.0
        )
        observation, _ = StereoObservationBuilder(
            source, config
        ).make_fallback(
            keyframe_id=9,
            semantic_image=_semantic(),
            sensor_timestamp=9.0,
        )
        self.assertEqual(len(observation.camera_points), 2)

    def test_empty_fallback_is_rejected(self):
        source = _SnapshotSource(_snapshot([[50, 40]], [[0, 0, 8]]))
        with self.assertRaisesRegex(RuntimeError, "no stereo points"):
            StereoObservationBuilder(source).make_fallback(
                keyframe_id=10,
                semantic_image=_semantic(),
                sensor_timestamp=10.0,
            )

    def test_pointcloud_assist_builds_aligned_observation_and_statistics(self):
        source = _SnapshotSource(_snapshot([[50, 40]], [[0, 0, 9]]))
        config = StereoObservationConfiguration(assist_min_matches=1)
        observation, statistics = StereoObservationBuilder(
            source, config
        ).make_assisted(
            keyframe_id=11,
            semantic_image=_semantic(),
            sensor_timestamp=11.0,
            pointcloud_camera_points=np.array([[0, 0, 2]]),
            camera_matrix=np.array(
                [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                dtype=np.float64,
            ),
            calibration_width=100,
            calibration_height=80,
        )
        self.assertEqual(observation.source, "pointcloud_assisted_stereo")
        np.testing.assert_allclose(observation.camera_points, [[0, 0, 2]])
        np.testing.assert_array_equal(observation.semantic_labels, [4])
        self.assertEqual(statistics["input_stereo_points"], 1)
        self.assertEqual(statistics["pointcloud_supported_points"], 1)

    def test_pointcloud_assist_enforces_minimum_support(self):
        source = _SnapshotSource(_snapshot([[50, 40]], [[0, 0, 9]]))
        builder = StereoObservationBuilder(
            source,
            StereoObservationConfiguration(assist_min_matches=2),
        )
        with self.assertRaisesRegex(RuntimeError, "matched only 1"):
            builder.make_assisted(
                keyframe_id=12,
                semantic_image=_semantic(),
                sensor_timestamp=12.0,
                pointcloud_camera_points=np.array([[0, 0, 2]]),
                camera_matrix=np.array(
                    [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                    dtype=np.float64,
                ),
                calibration_width=100,
                calibration_height=80,
            )


if __name__ == "__main__":
    unittest.main()
