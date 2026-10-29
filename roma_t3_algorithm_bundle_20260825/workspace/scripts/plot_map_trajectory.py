#!/usr/bin/env python3
"""Render a map-frame TUM trajectory over a dense global GridMap NPZ."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot trajectory_map.tum over global_grid_map.npz without "
            "modifying either source file."
        )
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="ROMA output directory containing the NPZ and TUM files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "PNG output path; defaults to "
            "<run_directory>/map_trajectory_overlay.png."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Do not also save an editable SVG beside the PNG.",
    )
    return parser.parse_args()


def scalar(archive: np.lib.npyio.NpzFile, name: str) -> float:
    if name not in archive.files:
        raise KeyError(f"Missing NPZ scalar: {name}")
    return float(np.asarray(archive[name]).item())


def load_trajectory(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 8:
        raise ValueError(
            f"TUM trajectory must have 8 columns, got shape {rows.shape}"
        )
    if len(rows) < 2:
        raise ValueError("Trajectory must contain at least two poses")
    if not np.isfinite(rows).all():
        raise ValueError("Trajectory contains non-finite values")
    if np.any(np.diff(rows[:, 0]) <= 0.0):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    return rows


def display_limits(
    elevation: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    trajectory_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rows, columns = np.nonzero(np.isfinite(elevation))
    if len(rows):
        x_values = np.concatenate(
            (
                np.array(
                    [
                        origin_x + columns.min() * resolution,
                        origin_x + (columns.max() + 1) * resolution,
                    ]
                ),
                trajectory_xy[:, 0],
            )
        )
        y_values = np.concatenate(
            (
                np.array(
                    [
                        origin_y + rows.min() * resolution,
                        origin_y + (rows.max() + 1) * resolution,
                    ]
                ),
                trajectory_xy[:, 1],
            )
        )
    else:
        x_values = trajectory_xy[:, 0]
        y_values = trajectory_xy[:, 1]

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    span = max(x_max - x_min, y_max - y_min, resolution)
    padding = max(1.0, 0.07 * span)
    return (
        (x_min - padding, x_max + padding),
        (y_min - padding, y_max + padding),
    )


def robust_color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("The elevation layer has no finite observations")
    low, high = np.percentile(finite, (2.0, 98.0))
    low = float(low)
    high = float(high)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Could not derive finite elevation limits")
    if high <= low:
        delta = max(abs(low) * 0.02, 0.01)
        low -= delta
        high += delta
    return low, high


def render(run_directory: Path, output: Path, dpi: int, save_svg: bool) -> None:
    run_directory = run_directory.expanduser().resolve()
    map_path = run_directory / "global_grid_map.npz"
    trajectory_path = run_directory / "trajectory_map.tum"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)

    trajectory = load_trajectory(trajectory_path)
    trajectory_xy = trajectory[:, 1:3]
    path_length = float(
        np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=1).sum()
    )

    with np.load(map_path, allow_pickle=False) as archive:
        elevation = np.asarray(archive["elevation"], dtype=np.float32)
        observation_count = np.asarray(
            archive["observation_count"], dtype=np.float32
        )
        occupancy = np.asarray(archive["occupancy"], dtype=np.int16)
        resolution = scalar(archive, "resolution")
        origin_x = scalar(archive, "origin_x")
        origin_y = scalar(archive, "origin_y")
        frame_id = str(np.asarray(archive["frame_id"]).item())
        map_revision = int(np.asarray(archive["map_revision"]).item())

    if elevation.ndim != 2:
        raise ValueError(f"Elevation must be 2-D, got {elevation.shape}")
    if observation_count.shape != elevation.shape:
        raise ValueError("observation_count and elevation shapes differ")
    if occupancy.shape != elevation.shape:
        raise ValueError("occupancy and elevation shapes differ")
    if resolution <= 0.0:
        raise ValueError("Map resolution must be positive")

    height, width = elevation.shape
    extent = (
        origin_x,
        origin_x + width * resolution,
        origin_y,
        origin_y + height * resolution,
    )
    valid = np.isfinite(elevation)
    known_cells = int(np.count_nonzero(valid))
    if not known_cells:
        raise ValueError("Map has no finite elevation cells")

    confidence = np.zeros(elevation.shape, dtype=np.float32)
    if np.any(observation_count > 0.0):
        logged = np.log1p(observation_count)
        scale = float(np.percentile(logged[logged > 0.0], 95.0))
        if scale > 0.0:
            confidence = np.clip(logged / scale, 0.0, 1.0)
    alpha = np.where(valid, 0.58 + 0.40 * confidence, 0.0)

    vmin, vmax = robust_color_limits(elevation)
    x_limits, y_limits = display_limits(
        elevation,
        origin_x,
        origin_y,
        resolution,
        trajectory_xy,
    )

    background = "#0b1320"
    foreground = "#edf4ff"
    muted = "#a9b9cc"
    grid_color = "#4a5a70"
    trajectory_color = "#ff8a3d"
    start_color = "#55d98a"
    end_color = "#ff4d7d"

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "normal",
            "axes.labelcolor": foreground,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": foreground,
        }
    ):
        figure, axis = plt.subplots(figsize=(10.8, 9.2), constrained_layout=True)
        figure.patch.set_facecolor(background)
        axis.set_facecolor(background)

        colormap = plt.colormaps["terrain"].copy()
        colormap.set_bad((0.0, 0.0, 0.0, 0.0))
        image = axis.imshow(
            np.ma.masked_invalid(elevation),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=colormap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            alpha=alpha,
            zorder=1,
        )

        known_occupancy = occupancy >= 0
        if np.any(known_occupancy) and not np.all(known_occupancy):
            x_centers = origin_x + (
                np.arange(width, dtype=np.float64) + 0.5
            ) * resolution
            y_centers = origin_y + (
                np.arange(height, dtype=np.float64) + 0.5
            ) * resolution
            axis.contour(
                x_centers,
                y_centers,
                known_occupancy.astype(np.float32),
                levels=[0.5],
                colors=["#8dd8ff"],
                linewidths=0.7,
                alpha=0.32,
                zorder=2,
            )

        trajectory_line = axis.plot(
            trajectory_xy[:, 0],
            trajectory_xy[:, 1],
            color=trajectory_color,
            linewidth=2.8,
            solid_capstyle="round",
            zorder=6,
        )[0]
        trajectory_line.set_path_effects(
            [
                path_effects.Stroke(linewidth=6.5, foreground=background),
                path_effects.Normal(),
            ]
        )

        arrow_count = min(9, max(0, len(trajectory_xy) // 14))
        if arrow_count:
            indices = np.linspace(
                4,
                len(trajectory_xy) - 5,
                arrow_count,
                dtype=int,
            )
            derivatives = (
                trajectory_xy[np.minimum(indices + 2, len(trajectory_xy) - 1)]
                - trajectory_xy[np.maximum(indices - 2, 0)]
            )
            lengths = np.linalg.norm(derivatives, axis=1)
            usable = lengths > 1.0e-9
            directions = derivatives[usable] / lengths[usable, None]
            arrow_origins = trajectory_xy[indices[usable]]
            axis.quiver(
                arrow_origins[:, 0],
                arrow_origins[:, 1],
                directions[:, 0],
                directions[:, 1],
                angles="xy",
                scale_units="xy",
                scale=2.6,
                width=0.0045,
                headwidth=4.2,
                headlength=5.0,
                headaxislength=4.5,
                color=trajectory_color,
                zorder=7,
            )

        axis.scatter(
            trajectory_xy[0, 0],
            trajectory_xy[0, 1],
            s=110,
            marker="o",
            facecolor=start_color,
            edgecolor=background,
            linewidth=1.8,
            zorder=8,
        )
        axis.scatter(
            trajectory_xy[-1, 0],
            trajectory_xy[-1, 1],
            s=155,
            marker="*",
            facecolor=end_color,
            edgecolor=background,
            linewidth=1.5,
            zorder=8,
        )
        axis.annotate(
            "START",
            xy=trajectory_xy[0],
            xytext=(8, 9),
            textcoords="offset points",
            color=foreground,
            fontsize=9,
            zorder=9,
        )
        axis.annotate(
            "END",
            xy=trajectory_xy[-1],
            xytext=(8, -14),
            textcoords="offset points",
            color=foreground,
            fontsize=9,
            zorder=9,
        )

        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"{frame_id} X (m)")
        axis.set_ylabel(f"{frame_id} Y (m)")
        axis.grid(
            True,
            color=grid_color,
            linewidth=0.55,
            linestyle=":",
            alpha=0.45,
            zorder=0,
        )
        for spine in axis.spines.values():
            spine.set_color(grid_color)
            spine.set_linewidth(0.8)

        axis.set_title(
            "RoMa trajectory over global elevation map\n"
            f"{run_directory.name}  |  {len(trajectory)} poses  |  "
            f"{path_length:.2f} m planar path  |  "
            f"{known_cells:,} mapped cells  |  revision {map_revision}",
            color=foreground,
            pad=13,
        )

        colorbar = figure.colorbar(
            image,
            ax=axis,
            fraction=0.035,
            pad=0.025,
            shrink=0.86,
        )
        colorbar.set_label("Elevation Z (m)", color=foreground)
        colorbar.ax.tick_params(colors=muted)
        colorbar.outline.set_edgecolor(grid_color)

        legend_items = [
            Line2D(
                [0],
                [0],
                color=trajectory_color,
                linewidth=2.8,
                label="Trajectory",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=start_color,
                markeredgecolor=background,
                markersize=8,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=end_color,
                markeredgecolor=background,
                markersize=10,
                label="End",
            ),
        ]
        legend = axis.legend(
            handles=legend_items,
            loc="upper right",
            frameon=True,
            facecolor=background,
            edgecolor=grid_color,
            framealpha=0.88,
        )
        for label in legend.get_texts():
            label.set_color(foreground)

        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=max(72, int(dpi)),
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        if save_svg:
            figure.savefig(
                output.with_suffix(".svg"),
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
            )
        plt.close(figure)

    print(f"PNG: {output}")
    if save_svg:
        print(f"SVG: {output.with_suffix('.svg')}")
    print(
        f"poses={len(trajectory)}, path_length_xy_m={path_length:.6f}, "
        f"known_elevation_cells={known_cells}, resolution_m={resolution:.6f}"
    )


def main() -> int:
    args = parse_args()
    run_directory = args.run_directory.expanduser().resolve()
    output = (
        args.output
        if args.output is not None
        else run_directory / "map_trajectory_overlay.png"
    )
    render(
        run_directory,
        output,
        dpi=args.dpi,
        save_svg=not args.no_svg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
