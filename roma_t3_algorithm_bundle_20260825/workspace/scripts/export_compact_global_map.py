#!/usr/bin/env python3
"""Convert a saved fusion snapshot to the compact single-file map format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping.compact_grid_store import save_compact_global_map
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compression-level", type=int, default=6)
    args = parser.parse_args()

    grid = TiledSemanticMapManager.load_snapshot(args.snapshot_index)
    saved = save_compact_global_map(
        grid,
        args.output,
        frame_id="map",
        timestamp_text="converted_snapshot",
        compression_level=args.compression_level,
    )
    print(
        f"saved={saved} bytes={saved.stat().st_size} "
        f"tiles={grid.tile_count} known_cells={grid.known_cells} "
        f"revision={grid.map_revision}"
    )


if __name__ == "__main__":
    main()
