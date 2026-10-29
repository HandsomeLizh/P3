#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

workspace = Path(__file__).resolve().parents[1]
source = workspace / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(source))

from t3_semantic_mapping.semantic_grid import LayeredSemanticGridMap


grid = LayeredSemanticGridMap(
    resolution=0.1,
    length_x=10.0,
    length_y=10.0,
)

points = np.array(
    [
        [1.0, 0.0, -1.0],
        [1.1, 0.0, -0.95],
        [2.0, 1.0, -0.4],
        [2.1, 1.0, -0.35],
    ],
    dtype=np.float64,
)
labels = np.array([1, 1, 4, 4], dtype=np.uint8)

stats = grid.update(
    points_map=points,
    semantic_labels=labels,
    camera_position_map=np.array([0.0, 0.0, 0.0]),
)

assert stats["accepted_points"] == 4
assert np.count_nonzero(grid.elevation_count) >= 2
assert set(grid.all_layers()) == {
    "elevation",
    "elevation_variance",
    "occupancy",
    "semantic_id",
    "semantic_confidence",
    "observation_count",
    "height_range",
    "roughness",
    "color",
}
print("Grid core test: OK")
print(stats)
