"""Subscriber-gated rendering of already-computed visual correspondences."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


_GREEN = (40, 220, 40)
_RED = (40, 40, 235)
_CYAN = (235, 210, 40)
_WHITE = (245, 245, 245)
_BLACK = (0, 0, 0)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        return cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Debug match image must be HxW or HxWx3: {value.shape}")
    return np.ascontiguousarray(value, dtype=np.uint8)


def _resize_panel(image: np.ndarray, width: int) -> tuple[np.ndarray, float, float]:
    source = _as_bgr(image)
    height_in, width_in = source.shape[:2]
    if height_in <= 0 or width_in <= 0:
        raise ValueError("Debug match image dimensions must be positive")
    height_out = max(1, int(round(height_in * width / float(width_in))))
    interpolation = cv2.INTER_AREA if width < width_in else cv2.INTER_LINEAR
    resized = cv2.resize(source, (width, height_out), interpolation=interpolation)
    return resized, width / float(width_in), height_out / float(height_in)


def _bounded_indices(mask: np.ndarray, maximum: int) -> np.ndarray:
    count = len(mask)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    good = np.flatnonzero(mask)
    bad = np.flatnonzero(~mask)
    good_budget = min(len(good), max(1, int(round(maximum * 0.7))))
    bad_budget = min(len(bad), maximum - good_budget)
    if good_budget + bad_budget < maximum:
        good_budget = min(len(good), maximum - bad_budget)

    def sample(indices: np.ndarray, budget: int) -> np.ndarray:
        if budget <= 0:
            return np.empty(0, dtype=np.int64)
        if len(indices) <= budget:
            return indices
        return indices[
            np.linspace(0, len(indices) - 1, budget, dtype=np.int64)
        ]

    return np.sort(np.concatenate((sample(good, good_budget), sample(bad, bad_budget))))


def _match_row(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    *,
    inlier_mask: Optional[np.ndarray],
    panel_width: int,
    maximum_lines: int,
    label: str,
    stereo: bool = False,
) -> np.ndarray:
    first_panel, first_sx, first_sy = _resize_panel(first_image, panel_width)
    second_panel, second_sx, second_sy = _resize_panel(second_image, panel_width)
    row_height = max(first_panel.shape[0], second_panel.shape[0])
    canvas = np.zeros((row_height, panel_width * 2, 3), dtype=np.uint8)
    canvas[: first_panel.shape[0], :panel_width] = first_panel
    canvas[: second_panel.shape[0], panel_width:] = second_panel

    first = np.asarray(first_points, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(second_points, dtype=np.float64).reshape(-1, 2)
    count = min(len(first), len(second))
    first = first[:count]
    second = second[:count]
    if inlier_mask is None:
        mask = np.ones(count, dtype=bool)
    else:
        mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if len(mask) < count:
            padded = np.zeros(count, dtype=bool)
            padded[: len(mask)] = mask
            mask = padded
        else:
            mask = mask[:count]

    finite = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    first = first[finite]
    second = second[finite]
    mask = mask[finite]
    selected = _bounded_indices(mask, maximum_lines)
    for index in selected:
        first_xy = (
            int(round(first[index, 0] * first_sx)),
            int(round(first[index, 1] * first_sy)),
        )
        second_xy = (
            panel_width + int(round(second[index, 0] * second_sx)),
            int(round(second[index, 1] * second_sy)),
        )
        color = _CYAN if stereo else (_GREEN if mask[index] else _RED)
        cv2.line(canvas, first_xy, second_xy, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, first_xy, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, second_xy, 2, color, -1, cv2.LINE_AA)

    cv2.line(canvas, (panel_width, 0), (panel_width, row_height - 1), _WHITE, 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 31), _BLACK, -1)
    cv2.putText(
        canvas,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _WHITE,
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_feature_match_debug(
    snapshot: dict,
    current_left: np.ndarray,
    current_right: np.ndarray,
    *,
    output_width: int = 1280,
    maximum_lines: int = 200,
) -> np.ndarray:
    """Render stereo plus temporal matches into one RViz-friendly BGR image."""
    width = max(320, int(output_width))
    lines = max(1, int(maximum_lines))
    panel_width = max(160, width // 2)

    stereo_left = np.asarray(snapshot["stereo_left_points"]).reshape(-1, 2)
    stereo_right = np.asarray(snapshot["stereo_right_points"]).reshape(-1, 2)
    stereo_count = min(len(stereo_left), len(stereo_right))
    stereo_row = _match_row(
        current_left,
        current_right,
        stereo_left,
        stereo_right,
        inlier_mask=None,
        panel_width=panel_width,
        maximum_lines=lines,
        label=f"STEREO  left | right   valid_3d={stereo_count}",
        stereo=True,
    )

    reference_image = snapshot.get("temporal_reference_image")
    temporal_reference = np.asarray(
        snapshot.get("temporal_reference_points", np.empty((0, 2)))
    ).reshape(-1, 2)
    temporal_current = np.asarray(
        snapshot.get("temporal_current_points", np.empty((0, 2)))
    ).reshape(-1, 2)
    temporal_count = min(len(temporal_reference), len(temporal_current))
    temporal_mask = np.asarray(
        snapshot.get("temporal_inlier_mask", np.zeros(temporal_count, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    temporal_mask = temporal_mask[:temporal_count]
    inliers = int(np.count_nonzero(temporal_mask))
    ratio = inliers / max(temporal_count, 1)
    source = str(snapshot.get("temporal_source", "none"))
    tracking = "OK" if bool(snapshot.get("tracking_success", False)) else "REJECTED"

    if reference_image is None or temporal_count == 0:
        temporal_row = np.zeros_like(stereo_row)
        label = "TEMPORAL  keyframe | current   waiting for reference"
        cv2.putText(
            temporal_row,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _WHITE,
            1,
            cv2.LINE_AA,
        )
    else:
        temporal_row = _match_row(
            reference_image,
            current_left,
            temporal_reference,
            temporal_current,
            inlier_mask=temporal_mask,
            panel_width=panel_width,
            maximum_lines=lines,
            label=(
                "TEMPORAL  keyframe | current   "
                f"source={source} matches={temporal_count} "
                f"inliers={inliers} ratio={ratio:.2f} tracking={tracking}"
            ),
        )

    cv2.line(stereo_row, (0, stereo_row.shape[0] - 1),
             (stereo_row.shape[1] - 1, stereo_row.shape[0] - 1), _WHITE, 1)
    return np.ascontiguousarray(np.vstack((stereo_row, temporal_row)))
