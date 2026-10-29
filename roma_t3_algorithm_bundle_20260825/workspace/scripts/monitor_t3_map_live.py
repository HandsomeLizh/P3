#!/usr/bin/env python3
"""Continuously monitor the latest Task-3 2.5D semantic map.

This subscribes to:
    /Car/T3/semantic/incremental_map

Unlike inspect_grid_map.py, it does not exit after the first message and it
prints changing map content, including:
  * update_id and sensor timestamp
  * published rectangular map geometry
  * observed/known area inside that map snapshot
  * occupancy counts
  * semantic class counts
  * elevation, variance, height range and roughness statistics

Run while the mapper is active:
    source ~/t3_gridmap_v2_ws/scripts/enter_workspace.sh
    python ~/t3_gridmap_v2_ws/scripts/monitor_t3_map_live.py
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from t3_interfaces.msg import IncrementalSemanticMap


TOPIC = "/Car/T3/semantic/incremental_map"


def finite_stats(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no valid values"
    return (
        f"min={float(np.min(finite)):.3f}, "
        f"median={float(np.median(finite)):.3f}, "
        f"mean={float(np.mean(finite)):.3f}, "
        f"max={float(np.max(finite)):.3f}"
    )


class LiveMapMonitor(Node):
    def __init__(self) -> None:
        super().__init__("t3_live_map_monitor")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.last_update_id: Optional[int] = None
        self.subscription = self.create_subscription(
            IncrementalSemanticMap,
            TOPIC,
            self.callback,
            qos,
        )

        self.get_logger().info(f"Monitoring {TOPIC}")
        self.get_logger().info("Press Ctrl+C to stop.")

    def callback(self, message: IncrementalSemanticMap) -> None:
        update_id = int(message.update_id)
        if self.last_update_id == update_id:
            return
        self.last_update_id = update_id

        width = int(message.width)
        height = int(message.height)
        resolution = float(message.resolution)
        total_cells = width * height

        if total_cells <= 0:
            self.get_logger().warning("Received an empty map.")
            return

        expected = total_cells
        arrays = {
            "occupancy": np.asarray(message.occupancy, dtype=np.int16),
            "semantic": np.asarray(message.semantic, dtype=np.uint8),
            "semantic_confidence": np.asarray(
                message.semantic_confidence, dtype=np.float32
            ),
            "elevation": np.asarray(message.elevation, dtype=np.float32),
            "elevation_variance": np.asarray(
                message.elevation_variance, dtype=np.float32
            ),
            "height_range": np.asarray(message.height_range, dtype=np.float32),
            "roughness": np.asarray(message.roughness, dtype=np.float32),
            "observation_count": np.asarray(
                message.observation_count, dtype=np.uint32
            ),
        }

        invalid_lengths = {
            name: int(values.size)
            for name, values in arrays.items()
            if values.size != expected
        }
        if invalid_lengths:
            self.get_logger().error(
                f"Layer lengths do not match {expected}: {invalid_lengths}"
            )
            return

        for name in arrays:
            arrays[name] = arrays[name].reshape(height, width)

        occupancy = arrays["occupancy"]
        semantic = arrays["semantic"]
        confidence = arrays["semantic_confidence"]
        elevation = arrays["elevation"]
        variance = arrays["elevation_variance"]
        height_range = arrays["height_range"]
        roughness = arrays["roughness"]
        observations = arrays["observation_count"]

        known = np.isfinite(elevation) & (observations > 0)
        known_cells = int(np.count_nonzero(known))
        allocated_area_m2 = total_cells * resolution * resolution
        known_area_m2 = known_cells * resolution * resolution
        coverage_percent = (
            100.0 * known_cells / total_cells if total_cells else 0.0
        )

        origin_x = float(message.origin.position.x)
        origin_y = float(message.origin.position.y)

        if known_cells:
            rows, cols = np.where(known)
            min_row, max_row = int(rows.min()), int(rows.max())
            min_col, max_col = int(cols.min()), int(cols.max())
            observed_min_x = origin_x + min_col * resolution
            observed_max_x = origin_x + (max_col + 1) * resolution
            observed_min_y = origin_y + min_row * resolution
            observed_max_y = origin_y + (max_row + 1) * resolution
            observed_span_x = observed_max_x - observed_min_x
            observed_span_y = observed_max_y - observed_min_y
            observed_bbox = (
                f"X=[{observed_min_x:.2f}, {observed_max_x:.2f}] m, "
                f"Y=[{observed_min_y:.2f}, {observed_max_y:.2f}] m, "
                f"span={observed_span_x:.2f} x {observed_span_y:.2f} m"
            )
        else:
            observed_bbox = "no observed cells"

        unknown_occ = int(np.count_nonzero(occupancy < 0))
        free_occ = int(np.count_nonzero((occupancy >= 0) & (occupancy <= 35)))
        occupied_occ = int(np.count_nonzero(occupancy >= 65))
        uncertain_occ = int(
            np.count_nonzero((occupancy > 35) & (occupancy < 65))
        )

        class_names = list(message.class_names)
        semantic_lines = []
        if known_cells:
            ids, counts = np.unique(semantic[known], return_counts=True)
            for semantic_id, count in zip(ids.tolist(), counts.tolist()):
                name = (
                    class_names[semantic_id]
                    if 0 <= semantic_id < len(class_names)
                    else f"class_{semantic_id}"
                )
                semantic_lines.append(
                    f"    {semantic_id}: {name}: {count:,} cells"
                )
        else:
            semantic_lines.append("    no observed semantic cells")

        timestamp = (
            f"{int(message.header.stamp.sec)}."
            f"{int(message.header.stamp.nanosec):09d}"
        )

        print("\n" + "=" * 78)
        print(f"2.5D MAP UPDATE {update_id}")
        print("=" * 78)
        print(f"sensor timestamp:          {timestamp}")
        print(f"frame_id:                  {message.header.frame_id}")
        print(
            f"map snapshot geometry:     {width} x {height} cells, "
            f"{resolution:.3f} m/cell"
        )
        print(
            f"map snapshot extent:       "
            f"{width * resolution:.2f} x {height * resolution:.2f} m"
        )
        print(f"map snapshot area:         {allocated_area_m2:.2f} m^2")
        print(f"known/observed cells:      {known_cells:,} / {total_cells:,}")
        print(f"known/observed area:       {known_area_m2:.2f} m^2")
        print(f"snapshot coverage used:    {coverage_percent:.3f}%")
        print(f"observed bounding box:     {observed_bbox}")
        print(
            f"occupancy cells:           unknown={unknown_occ:,}, "
            f"free={free_occ:,}, uncertain={uncertain_occ:,}, "
            f"occupied={occupied_occ:,}"
        )
        print(f"elevation (m):             {finite_stats(elevation[known])}")
        print(f"elevation variance (m^2):  {finite_stats(variance[known])}")
        print(f"height range (m):          {finite_stats(height_range[known])}")
        print(f"roughness (m):             {finite_stats(roughness[known])}")
        if known_cells:
            print(
                "semantic confidence:       "
                + finite_stats(confidence[known])
            )
            print(
                "observations per cell:     "
                + finite_stats(observations[known].astype(np.float32))
            )
        print("semantic cells:")
        print("\n".join(semantic_lines))
        print("=" * 78)


def main() -> None:
    rclpy.init()
    node = LiveMapMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
