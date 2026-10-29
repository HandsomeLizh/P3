#!/usr/bin/env python3
"""Wire-equivalence tests for large ROS primitive-sequence fast paths."""

from __future__ import annotations

import array
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, PointCloud2
from t3_interfaces.msg import IncrementalSemanticMap


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src" / "t1_dataset_sim"))
sys.path.insert(0, str(WORKSPACE / "src" / "t3_semantic_mapping"))

from t1_dataset_sim.message_utils import (  # noqa: E402
    image_to_message,
    nanoseconds_to_time,
    uint8_array_from_buffer,
)
from t3_semantic_mapping.grid_map_message import (  # noqa: E402
    float32_array,
    int8_array,
    uint8_array,
    uint32_array,
)


def _function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _attribute_assignment_calls(function) -> set[tuple[str, str]]:
    assignments = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, ast.Call
        ):
            continue
        if isinstance(node.value.func, ast.Name):
            call_name = node.value.func.id
        elif isinstance(node.value.func, ast.Attribute):
            call_name = node.value.func.attr
        else:
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                assignments.add((target.attr, call_name))
    return assignments


class DatasetTypedBufferTest(unittest.TestCase):
    def test_image_wire_bytes_match_legacy_sequence_assignment(self):
        image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        stamp = nanoseconds_to_time(12_345_678_901)
        actual = image_to_message(image, stamp, "left", "bgr8")

        expected = Image()
        expected.header.stamp = stamp
        expected.header.frame_id = "left"
        expected.height = 7
        expected.width = 11
        expected.encoding = "bgr8"
        expected.is_bigendian = 0
        expected.step = 33
        expected.data = image.tobytes()

        self.assertIsInstance(actual.data, array.array)
        self.assertEqual(actual.data.typecode, "B")
        self.assertEqual(actual.data.tobytes(), image.tobytes())
        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )

    def test_pointcloud_byte_buffer_matches_legacy_wire_value(self):
        payload = bytes(range(64))
        actual = PointCloud2()
        actual.data = uint8_array_from_buffer(payload)
        expected = PointCloud2()
        expected.data = payload

        self.assertEqual(actual.data.typecode, "B")
        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )

    def test_noncontiguous_raw_buffer_is_rejected(self):
        values = np.arange(20, dtype=np.uint8).reshape(4, 5)[:, ::2]
        self.assertFalse(values.flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            uint8_array_from_buffer(values)


class MapTypedBufferTest(unittest.TestCase):
    def test_helpers_preserve_dtype_order_values_and_nan(self):
        occupancy = np.array([[-1, 0], [55, 100]], dtype=np.int8)
        semantic = np.array([[1, 2], [3, 255]], dtype=np.uint8)
        counts = np.array([[0, 1], [2, 2 ** 32 - 1]], dtype=np.uint32)
        elevation = np.array([[np.nan, -1.25], [0.0, 3.5]], dtype=np.float32)

        self.assertEqual(int8_array(occupancy).typecode, "b")
        self.assertEqual(list(int8_array(occupancy)), [-1, 0, 55, 100])
        self.assertEqual(uint8_array(semantic).typecode, "B")
        self.assertEqual(list(uint8_array(semantic)), [1, 2, 3, 255])
        self.assertEqual(uint32_array(counts).typecode, "I")
        self.assertEqual(
            list(uint32_array(counts)), [0, 1, 2, 2 ** 32 - 1]
        )
        floats = np.asarray(float32_array(elevation), dtype=np.float32)
        np.testing.assert_array_equal(floats, elevation.reshape(-1))

    def test_occupancy_grid_wire_bytes_match_legacy_list_assignment(self):
        values = np.array([[-1, 0], [55, 100]], dtype=np.int8)
        actual = OccupancyGrid()
        actual.data = int8_array(values)
        expected = OccupancyGrid()
        expected.data = values.reshape(-1, order="C").tolist()

        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )

    def test_incremental_map_wire_bytes_match_legacy_lists(self):
        occupancy = np.array([[-1, 0], [55, 100]], dtype=np.int8)
        semantic = np.array([[1, 2], [3, 255]], dtype=np.uint8)
        counts = np.array([[0, 1], [2, 9]], dtype=np.uint32)

        actual = IncrementalSemanticMap()
        actual.occupancy = int8_array(occupancy)
        actual.semantic = uint8_array(semantic)
        actual.observation_count = uint32_array(counts)

        expected = IncrementalSemanticMap()
        expected.occupancy = occupancy.reshape(-1, order="C").tolist()
        expected.semantic = semantic.reshape(-1, order="C").tolist()
        expected.observation_count = counts.reshape(-1, order="C").tolist()

        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )

    def test_production_paths_use_typed_buffers(self):
        message_utils = WORKSPACE / (
            "src/t1_dataset_sim/t1_dataset_sim/message_utils.py"
        )
        lidar_io = WORKSPACE / (
            "src/t1_dataset_sim/t1_dataset_sim/lidar_io.py"
        )
        map_output_message = WORKSPACE / (
            "src/t3_semantic_mapping/t3_semantic_mapping/"
            "map_output_message.py"
        )

        self.assertIn(
            ("data", "uint8_array_from_buffer"),
            _attribute_assignment_calls(
                _function(message_utils, "image_to_message")
            ),
        )
        self.assertIn(
            ("data", "uint8_array_from_buffer"),
            _attribute_assignment_calls(
                _function(lidar_io, "read_binary_pcd_message")
            ),
        )
        self.assertIn(
            ("data", "int8_array"),
            _attribute_assignment_calls(
                _function(
                    map_output_message,
                    "make_occupancy_grid_message",
                )
            ),
        )
        incremental_calls = _attribute_assignment_calls(
            _function(
                map_output_message,
                "make_incremental_semantic_map_message",
            )
        )
        self.assertIn(("occupancy", "int8_array"), incremental_calls)
        self.assertIn(("semantic", "uint8_array"), incremental_calls)
        self.assertIn(
            ("observation_count", "uint32_array"), incremental_calls
        )


if __name__ == "__main__":
    unittest.main()
