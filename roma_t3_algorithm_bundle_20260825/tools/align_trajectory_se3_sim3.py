#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def umeyama(source, target, estimate_scale):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    guard = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        guard[-1, -1] = -1.0
    rotation = u @ guard @ vt
    if estimate_scale:
        variance = np.sum(source_centered**2) / len(source)
        scale = float(np.sum(singular_values * np.diag(guard)) / variance)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (source @ rotation.T) + translation
    return aligned, scale, rotation, translation


def path_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def error_summary(aligned, reference):
    error_3d = np.linalg.norm(aligned - reference, axis=1)
    error_xy = np.linalg.norm(aligned[:, :2] - reference[:, :2], axis=1)
    step_error = np.linalg.norm(
        np.diff(aligned, axis=0) - np.diff(reference, axis=0), axis=1
    )
    return {
        "rmse_3d_m": float(np.sqrt(np.mean(error_3d**2))),
        "mean_3d_m": float(np.mean(error_3d)),
        "median_3d_m": float(np.median(error_3d)),
        "p95_3d_m": float(np.percentile(error_3d, 95)),
        "max_3d_m": float(np.max(error_3d)),
        "terminal_3d_m": float(error_3d[-1]),
        "rmse_xy_m": float(np.sqrt(np.mean(error_xy**2))),
        "mean_step_translation_error_m": float(np.mean(step_error)),
        "rmse_step_translation_error_m": float(np.sqrt(np.mean(step_error**2))),
        "errors_3d": error_3d,
        "errors_xy": error_xy,
    }


def plot_alignment(output, title, elapsed, estimate, reference, aligned, summary, scale):
    estimate_color = "#d94841"
    reference_color = "#1877b5"
    aligned_color = "#238b57"
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=18)

    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], color=reference_color, lw=2.2, label="Ground truth")
    ax.plot(estimate[:, 0], estimate[:, 1], color=estimate_color, lw=2.0, label="Raw estimate")
    ax.set_title("Raw XY trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(reference[:, 0], reference[:, 1], color=reference_color, lw=2.2, label="Ground truth")
    ax.plot(aligned[:, 0], aligned[:, 1], color=aligned_color, lw=2.0, label=f"Aligned estimate (scale={scale:.6f})")
    ax.scatter(reference[0, 0], reference[0, 1], color=reference_color, marker="o", s=45)
    ax.scatter(reference[-1, 0], reference[-1, 1], color=reference_color, marker="X", s=65)
    ax.scatter(aligned[0, 0], aligned[0, 1], color=aligned_color, marker="o", s=45)
    ax.scatter(aligned[-1, 0], aligned[-1, 1], color=aligned_color, marker="X", s=65)
    ax.set_title("Aligned XY trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    raw_error = np.linalg.norm(estimate - reference, axis=1)
    ax.plot(elapsed, raw_error, color=estimate_color, lw=1.6, label="Raw 3D error")
    ax.plot(elapsed, summary["errors_3d"], color=aligned_color, lw=1.8, label="Aligned 3D error")
    ax.plot(elapsed, summary["errors_xy"], color=reference_color, lw=1.3, label="Aligned XY error")
    ax.set_title("Position error over runtime")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    relative_reference = reference - reference[0]
    relative_estimate = estimate - estimate[0]
    relative_aligned = aligned - aligned[0]
    ax.plot(elapsed, relative_reference[:, 2], color=reference_color, lw=2.0, label="Ground-truth relative Z")
    ax.plot(elapsed, relative_estimate[:, 2], color=estimate_color, lw=1.5, label="Raw relative Z")
    ax.plot(elapsed, relative_aligned[:, 2], color=aligned_color, lw=1.8, label="Aligned relative Z")
    ax.set_title("Relative elevation change")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Delta Z (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def serializable_summary(summary):
    return {key: value for key, value in summary.items() if not key.startswith("errors_")}


def transform_record(scale, rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "similarity_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    data = np.genfromtxt(args.trajectory_csv, delimiter=",", names=True)
    estimate = np.column_stack((data["estimate_x_m"], data["estimate_y_m"], data["estimate_z_m"]))
    reference = np.column_stack((data["reference_x_m"], data["reference_y_m"], data["reference_z_m"]))
    elapsed = data["elapsed_runtime_sec"]

    se3, se3_scale, se3_rotation, se3_translation = umeyama(estimate, reference, False)
    sim3, sim3_scale, sim3_rotation, sim3_translation = umeyama(estimate, reference, True)
    raw_summary = error_summary(estimate, reference)
    se3_summary = error_summary(se3, reference)
    sim3_summary = error_summary(sim3, reference)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_alignment(
        args.output_dir / "trajectory_alignment_se3.png",
        "RoMa trajectory aligned to ground truth with SE(3)",
        elapsed, estimate, reference, se3, se3_summary, se3_scale,
    )
    plot_alignment(
        args.output_dir / "trajectory_alignment_sim3.png",
        "RoMa trajectory aligned to ground truth with Sim(3)",
        elapsed, estimate, reference, sim3, sim3_summary, sim3_scale,
    )

    metrics = {
        "method": "Umeyama estimate-to-reference alignment",
        "sample_count": int(len(data)),
        "duration_sec": float(elapsed[-1] - elapsed[0]),
        "path_length_m": {
            "reference": path_length(reference),
            "raw_estimate": path_length(estimate),
            "se3_aligned": path_length(se3),
            "sim3_aligned": path_length(sim3),
        },
        "raw": serializable_summary(raw_summary),
        "se3": {
            **transform_record(se3_scale, se3_rotation, se3_translation),
            **serializable_summary(se3_summary),
        },
        "sim3": {
            **transform_record(sim3_scale, sim3_rotation, sim3_translation),
            **serializable_summary(sim3_summary),
        },
    }
    (args.output_dir / "trajectory_alignment_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "trajectory_alignment_se3_sim3.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "elapsed_runtime_sec", "estimate_x_m", "estimate_y_m", "estimate_z_m",
            "reference_x_m", "reference_y_m", "reference_z_m",
            "se3_x_m", "se3_y_m", "se3_z_m", "se3_error_3d_m",
            "sim3_x_m", "sim3_y_m", "sim3_z_m", "sim3_error_3d_m",
        ])
        writer.writerows(np.column_stack((
            elapsed, estimate, reference, se3, se3_summary["errors_3d"],
            sim3, sim3_summary["errors_3d"],
        )))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
