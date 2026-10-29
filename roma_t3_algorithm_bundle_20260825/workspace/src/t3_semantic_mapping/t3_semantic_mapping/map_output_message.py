"""Pure constructors for the two stable rectangular map messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from t3_interfaces.msg import IncrementalSemanticMap

from .grid_map_message import (
    float32_array,
    int8_array,
    uint8_array,
    uint32_array,
)

if TYPE_CHECKING:
    from .semantic_grid import GridGeometry
    from .tiled_semantic_map import SemanticMapWindow


def make_occupancy_grid_message(
    *,
    header: Header,
    window: "SemanticMapWindow",
) -> OccupancyGrid:
    """Build the complete occupancy window without publishing side effects."""

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
    message.data = int8_array(window.occupancy_int8())
    return message


def make_occupancy_grid_from_array(
    *,
    header: Header,
    geometry: "GridGeometry",
    occupancy: np.ndarray,
) -> OccupancyGrid:
    """Build an OccupancyGrid from an already downsampled [Y, X] array."""

    values = np.asarray(occupancy, dtype=np.int8)
    expected_shape = (int(geometry.height), int(geometry.width))
    if values.shape != expected_shape:
        raise ValueError(
            f"occupancy shape {values.shape} does not match {expected_shape}"
        )
    message = OccupancyGrid()
    message.header = header
    message.info.resolution = float(geometry.resolution)
    message.info.width = int(geometry.width)
    message.info.height = int(geometry.height)
    message.info.origin.position.x = float(geometry.origin_x)
    message.info.origin.position.y = float(geometry.origin_y)
    message.info.origin.position.z = 0.0
    message.info.origin.orientation.w = 1.0
    message.data = int8_array(values)
    return message


def make_incremental_semantic_map_message(
    *,
    header: Header,
    update_id: int,
    window: "SemanticMapWindow",
) -> IncrementalSemanticMap:
    """Build one complete fixed/rolling/tile rectangle in the old schema."""

    geometry = window.geometry
    message = IncrementalSemanticMap()
    message.header = header
    message.update_id = int(update_id) % (2 ** 32)
    message.resolution = float(geometry.resolution)
    message.width = int(geometry.width)
    message.height = int(geometry.height)
    message.origin.position.x = float(geometry.origin_x)
    message.origin.position.y = float(geometry.origin_y)
    message.origin.position.z = 0.0
    message.origin.orientation.w = 1.0
    message.occupancy = int8_array(window.occupancy)
    message.semantic = uint8_array(window.semantic)
    message.semantic_confidence = float32_array(
        window.semantic_confidence
    )
    message.elevation = float32_array(window.elevation)
    message.elevation_variance = float32_array(
        window.elevation_variance
    )
    message.height_range = float32_array(window.height_range)
    message.roughness = float32_array(window.roughness)
    message.observation_count = uint32_array(window.observation_count)
    message.class_names = list(window.class_names)
    return message
