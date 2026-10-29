#!/usr/bin/env python3
"""Same-input CPU microbenchmark; deliberately never reads/plays ROS bags."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/t3_semantic_mapping"))
from t3_semantic_mapping.elevation_fusion import ElevationFusionConfig
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=12000)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cells < 100 or args.updates < 4:
        parser.error("Use at least 100 cells and 4 updates")
    rng = np.random.default_rng(20260915)
    side = int(np.ceil(np.sqrt(args.cells)))
    cells = np.arange(args.cells)
    xy = np.column_stack(((cells % side + .5) * .2, (cells // side + .5) * .2)) - side * .1
    truth = .03 * xy[:, 0] + .15 * np.sin(xy[:, 1] * .3)
    batches = []
    for _ in range(args.updates + 2):
        z = truth + rng.normal(0., .03, args.cells)
        # A repeatable mix of small measurement noise and isolated bad heights.
        bad = rng.random(args.cells) < .01
        z[bad] += rng.choice([-2., 2.], int(bad.sum()))
        batches.append(np.column_stack((xy, z)))
    maps = {mode: TiledSemanticMapManager(resolution=.2, tile_cells=256,
            elevation_fusion=ElevationFusionConfig(fusion_mode=mode))
            for mode in ("mean", "variance")}
    times = {mode: [] for mode in maps}
    rejected = {mode: 0 for mode in maps}
    # Initialize cleanly; this benchmark is not a test of first-frame accuracy.
    for target in maps.values():
        target.update_elevation_only(points_map=np.column_stack((xy, truth)))
    for index, points in enumerate(batches):
        # Alternate order to reduce systematic temperature/cache/load bias.
        for mode in (("mean", "variance") if index % 2 else ("variance", "mean")):
            start = time.perf_counter()
            candidate = maps[mode].clone_for_update()
            stats = candidate.update_elevation_only(points_map=points)
            elapsed = time.perf_counter() - start
            maps[mode] = candidate
            if index >= 2:
                times[mode].append(elapsed * 1000.)
                rejected[mode] += stats["elevation_rejected_points"]
    report = {
        "scope": "Synthetic LiDAR XY-cell updates + tile copy-on-write, not end-to-end VO/ROS/RViz",
        "platform": platform.platform(), "python": platform.python_version(),
        "cells_per_update": args.cells, "measured_updates_per_mode": args.updates,
        "tiles": maps["variance"].tile_count,
        "timing_ms": {mode: {"median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "samples": values}
            for mode, values in times.items()},
        "gate_rejected_points": rejected,
        "extra_filter_array_bytes": sum(tile.elevation_fused_mean.nbytes +
            tile.elevation_estimate_variance.nbytes for tile in maps["variance"].tiles.values()),
        "notes": "No ray cleanup, GPU, bags, ROS nodes or vehicle control. Fixed heuristic noise, not calibrated accuracy.",
    }
    report["median_overhead_ms"] = report["timing_ms"]["variance"]["median"] - report["timing_ms"]["mean"]["median"]
    report["median_overhead_percent"] = 100. * report["median_overhead_ms"] / report["timing_ms"]["mean"]["median"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
