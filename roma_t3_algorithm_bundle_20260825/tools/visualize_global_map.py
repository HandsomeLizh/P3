#!/usr/bin/env python3
"""Render the persisted global elevation map with the estimated trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_tum_xy(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] < 4:
        raise ValueError(f"invalid TUM trajectory: {path}")
    return rows[:, 1:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_npz", type=Path)
    parser.add_argument("trajectory_tum", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()

    with np.load(args.map_npz) as data:
        elevation = np.asarray(data["elevation"], dtype=np.float32)
        resolution = float(data["resolution"])
        origin_x = float(data["origin_x"])
        origin_y = float(data["origin_y"])
        width = int(data["width"])
        height = int(data["height"])

    finite = np.isfinite(elevation)
    if not finite.any():
        raise ValueError("global map contains no finite elevation cells")
    vmin, vmax = np.nanpercentile(elevation, [1.0, 99.0])
    extent = (
        origin_x,
        origin_x + width * resolution,
        origin_y,
        origin_y + height * resolution,
    )
    trajectory = load_tum_xy(args.trajectory_tum)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_invalid(elevation),
        origin="lower",
        extent=extent,
        cmap="terrain",
        vmin=float(vmin),
        vmax=float(vmax),
        interpolation="nearest",
    )
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#d7191c", linewidth=1.8,
            label="VO trajectory")
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color="#1a9641", s=45,
               marker="o", label="Start", zorder=3)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="#2c3e75", s=55,
               marker="x", label="End", zorder=3)
    ax.set_title("LiDAR-assisted global elevation map (mapping only)")
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Elevation (m), clipped to 1st-99th percentile")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
