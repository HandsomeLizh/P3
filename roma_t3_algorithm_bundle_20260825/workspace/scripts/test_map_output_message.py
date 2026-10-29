#!/usr/bin/env python3
"""Wire contracts for pure complete-rectangle map message constructors."""

from __future__ import annotations

from array import array
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.serialization import serialize_message
from std_msgs.msg import Header
from t3_interfaces.msg import IncrementalSemanticMap


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE / "src" / "t3_semantic_mapping"
MAPPER_PATH = PACKAGE / (
    "t3_semantic_mapping/roma_semantic_mapper_node.py"
)
sys.path.insert(0, str(PACKAGE))

from t3_semantic_mapping.map_output_message import (  # noqa: E402
    make_incremental_semantic_map_message,
    make_occupancy_grid_message,
)
from t3_semantic_mapping.semantic_grid import GridGeometry  # noqa: E402
from t3_semantic_mapping.tiled_semantic_map import (  # noqa: E402
    SemanticMapWindow,
)


def _header() -> Header:
    header = Header()
    header.stamp.sec = 17
    header.stamp.nanosec = 23
    header.frame_id = "map"
    return header


def _window() -> SemanticMapWindow:
    geometry = GridGeometry(
        resolution=0.1,
        width=3,
        height=2,
        origin_x=-1.25,
        origin_y=2.5,
    )
    return SemanticMapWindow(
        geometry=geometry,
        class_names=("unknown", "ground", "obstacle"),
        occupancy=np.array([[-1, 0, 25], [50, 75, 100]], dtype=np.int8),
        semantic=np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8),
        semantic_confidence=np.array(
            [[0.0, 0.5, 1.0], [0.75, 0.25, 0.0]], dtype=np.float32
        ),
        elevation=np.array(
            [[np.nan, -1.5, 0.0], [0.5, 2.0, 3.25]], dtype=np.float32
        ),
        elevation_variance=np.array(
            [[np.nan, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32
        ),
        height_range=np.array(
            [[np.nan, 0.2, 0.4], [0.6, 0.8, 1.0]], dtype=np.float32
        ),
        roughness=np.array(
            [[np.nan, 0.01, 0.02], [0.03, 0.04, 0.05]], dtype=np.float32
        ),
        observation_count=np.array(
            [[0, 1, 2], [3, 4, 2 ** 32 - 1]], dtype=np.uint32
        ),
    )


def _frozen_occupancy_reference(header, window) -> OccupancyGrid:
    geometry = window.geometry
    message = OccupancyGrid()
    message.header = header
    message.info.resolution = float(geometry.resolution)
    message.info.width = int(geometry.width)
    message.info.height = int(geometry.height)
    message.info.origin.position.x = float(geometry.origin_x)
    message.info.origin.position.y = float(geometry.origin_y)
    message.info.origin.position.z = 0.0
    message.info.origin.orientation.w = 1.0
    message.data = window.occupancy.reshape(-1).tolist()
    return message


def _frozen_incremental_reference(
    header,
    update_id: int,
    window,
) -> IncrementalSemanticMap:
    def typed(values, *, dtype, typecode):
        flat = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
        result = array(typecode)
        result.frombytes(memoryview(flat).cast("B"))
        return result

    geometry = window.geometry
    message = IncrementalSemanticMap()
    message.header = header
    message.update_id = int(update_id % (2 ** 32))
    message.resolution = float(geometry.resolution)
    message.width = int(geometry.width)
    message.height = int(geometry.height)
    message.origin.position.x = float(geometry.origin_x)
    message.origin.position.y = float(geometry.origin_y)
    message.origin.position.z = 0.0
    message.origin.orientation.w = 1.0
    message.occupancy = typed(
        window.occupancy, dtype=np.int8, typecode="b"
    )
    message.semantic = typed(
        window.semantic, dtype=np.uint8, typecode="B"
    )
    message.semantic_confidence = typed(
        window.semantic_confidence, dtype=np.float32, typecode="f"
    )
    message.elevation = typed(
        window.elevation, dtype=np.float32, typecode="f"
    )
    message.elevation_variance = typed(
        window.elevation_variance, dtype=np.float32, typecode="f"
    )
    message.height_range = typed(
        window.height_range, dtype=np.float32, typecode="f"
    )
    message.roughness = typed(
        window.roughness, dtype=np.float32, typecode="f"
    )
    message.observation_count = typed(
        window.observation_count, dtype=np.uint32, typecode="I"
    )
    message.class_names = list(window.class_names)
    return message


class MapOutputMessageTest(unittest.TestCase):
    def test_occupancy_message_is_byte_identical_to_frozen_reference(self):
        window = _window()
        actual = make_occupancy_grid_message(
            header=_header(),
            window=window,
        )
        expected = _frozen_occupancy_reference(_header(), window)
        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )
        self.assertEqual(actual.data.typecode, "b")

    def test_incremental_message_is_byte_identical_to_frozen_reference(self):
        window = _window()
        update_id = 2 ** 32 + 19
        actual = make_incremental_semantic_map_message(
            header=_header(),
            update_id=update_id,
            window=window,
        )
        expected = _frozen_incremental_reference(
            _header(), update_id, window
        )
        self.assertEqual(
            bytes(serialize_message(actual)),
            bytes(serialize_message(expected)),
        )
        self.assertEqual(actual.update_id, 19)
        self.assertEqual(actual.occupancy.typecode, "b")
        self.assertEqual(actual.semantic.typecode, "B")
        self.assertEqual(actual.semantic_confidence.typecode, "f")
        self.assertEqual(actual.observation_count.typecode, "I")

    def test_messages_own_detached_payload_buffers(self):
        window = _window()
        occupancy = make_occupancy_grid_message(
            header=_header(), window=window
        )
        incremental = make_incremental_semantic_map_message(
            header=_header(), update_id=1, window=window
        )
        occupancy_before = occupancy.data.tobytes()
        incremental_before = incremental.elevation.tobytes()

        window.occupancy.fill(100)
        window.elevation.fill(42.0)

        self.assertEqual(occupancy.data.tobytes(), occupancy_before)
        self.assertEqual(incremental.elevation.tobytes(), incremental_before)

    def test_mapper_delegates_without_changing_publish_ownership(self):
        tree = ast.parse(MAPPER_PATH.read_text(encoding="utf-8"))
        mapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RomaSemanticMapperNode"
        )
        methods = {
            node.name: node
            for node in mapper.body
            if isinstance(node, ast.FunctionDef)
        }

        cases = (
            (
                "_publish_map_outputs",
                "make_occupancy_grid_message",
                "OccupancyGrid",
                "occupancy_pub",
            ),
            (
                "_publish_incremental_window_snapshot",
                "make_incremental_semantic_map_message",
                "IncrementalSemanticMap",
                "incremental_map_pub",
            ),
        )
        for method_name, builder_name, old_constructor, publisher in cases:
            method = methods[method_name]
            named_calls = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
            ]
            builder = next(
                node for node in named_calls if node.func.id == builder_name
            )
            self.assertNotIn(
                old_constructor,
                {node.func.id for node in named_calls},
            )
            publish = next(
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "publish"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == publisher
            )
            self.assertLess(builder.lineno, publish.lineno)

    def test_disabling_debug_views_keeps_all_formal_map_outputs(self):
        from t3_semantic_mapping.roma_semantic_mapper_node import (
            RomaSemanticMapperNode,
        )

        class Publisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        mapper = RomaSemanticMapperNode.__new__(RomaSemanticMapperNode)
        mapper.publish_layer_images = False
        mapper.publish_map_pointcloud = False
        mapper.incremental_publish_mode = "fixed"
        mapper.map_frame = "map"
        mapper.map_message_update_id = 0
        mapper.occupancy_pub = Publisher()
        mapper.grid_map_pub = Publisher()
        mapper.incremental_map_pub = Publisher()

        mapper._publish_map_outputs(
            header=_header(),
            all_points_map=None,
            point_semantics=None,
            local_window=_window(),
        )

        self.assertEqual(len(mapper.occupancy_pub.messages), 1)
        self.assertEqual(len(mapper.grid_map_pub.messages), 1)
        self.assertEqual(len(mapper.incremental_map_pub.messages), 1)
        self.assertEqual(len(mapper.occupancy_pub.messages[0].data), 6)
        self.assertEqual(
            mapper.grid_map_pub.messages[0].layers,
            [
                "elevation",
                "elevation_variance",
                "occupancy",
                "semantic_id",
                "semantic_confidence",
                "observation_count",
                "height_range",
                "roughness",
                "color",
            ],
        )
        incremental = mapper.incremental_map_pub.messages[0]
        self.assertEqual(len(incremental.occupancy), 6)
        self.assertEqual(len(incremental.semantic), 6)
        self.assertEqual(len(incremental.semantic_confidence), 6)
        self.assertEqual(len(incremental.elevation), 6)
        self.assertEqual(len(incremental.elevation_variance), 6)
        self.assertEqual(len(incremental.height_range), 6)
        self.assertEqual(len(incremental.roughness), 6)
        self.assertEqual(len(incremental.observation_count), 6)


if __name__ == "__main__":
    unittest.main()
