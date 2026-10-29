#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import numpy as np
from grid_map_msgs.srv import GetGridMap
from std_msgs.msg import Header


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.roma_semantic_mapper_node import RomaSemanticMapperNode
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _grid():
    grid = TiledSemanticMapManager(
        resolution=0.1,
        tile_cells=32,
        max_rays_per_update=100,
    )
    grid.update(
        points_map=np.array([[0.2, 0.2, 1.0], [3.4, 0.2, 2.0]]),
        semantic_labels=np.array([1, 4], dtype=np.uint8),
        camera_position_map=np.array([0.0, 0.0, 0.0]),
    )
    return grid


class MapDeliveryInterfacesTest(unittest.TestCase):
    def _mapper(self):
        mapper = object.__new__(RomaSemanticMapperNode)
        mapper.grid = _grid()
        mapper._map_lock = threading.RLock()
        mapper.map_frame = "map"
        mapper.map_revision = mapper.grid.map_revision
        mapper.last_header = Header()
        mapper.last_header.frame_id = "map"
        mapper.global_overview_resolution = 1.0
        mapper._last_published_global_overview_revision = -1
        mapper.global_overview_pub = _Publisher()
        mapper.grid_map_query_max_cells = 1000000
        return mapper

    def test_global_overview_is_transient_payload_only_when_changed(self):
        mapper = self._mapper()
        self.assertTrue(mapper._publish_global_overview_if_changed())
        self.assertFalse(mapper._publish_global_overview_if_changed())
        message = mapper.global_overview_pub.messages[0]
        self.assertEqual(message.header.frame_id, "map")
        self.assertAlmostEqual(message.info.resolution, 1.0)
        self.assertEqual(len(message.data), message.info.width * message.info.height)

    def test_region_query_returns_only_requested_layers(self):
        mapper = self._mapper()
        request = GetGridMap.Request()
        request.frame_id = "map"
        request.position_x = 0.0
        request.position_y = 0.0
        request.length_x = 6.0
        request.length_y = 6.0
        request.layers = ["occupancy", "semantic_id", "elevation"]
        response = mapper._on_get_grid_map(request, GetGridMap.Response())
        self.assertEqual(
            response.map.layers,
            ["occupancy", "semantic_id", "elevation"],
        )
        self.assertEqual(response.map.header.frame_id, "map")
        self.assertEqual(len(response.map.data), 3)

    def test_region_query_rejects_unbounded_request(self):
        mapper = self._mapper()
        mapper.grid_map_query_max_cells = 100
        request = GetGridMap.Request()
        request.frame_id = "map"
        request.position_x = 0.0
        request.position_y = 0.0
        request.length_x = 20.0
        request.length_y = 20.0
        with self.assertRaises(ValueError):
            mapper._on_get_grid_map(request, GetGridMap.Response())


if __name__ == "__main__":
    unittest.main()
