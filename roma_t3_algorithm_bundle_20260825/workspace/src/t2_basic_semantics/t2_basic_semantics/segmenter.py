from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CLASS_NAMES = [
    "unknown",
    "traversable_ground",
    "shadow",
    "bright_rock",
    "rough_obstacle",
]

# BGR display colours.
PALETTE_BGR = np.array(
    [
        [0, 0, 0],       # unknown
        [0, 180, 0],     # traversable ground
        [120, 80, 40],   # shadow
        [0, 220, 255],   # bright rock
        [0, 0, 220],     # rough obstacle
    ],
    dtype=np.uint8,
)


@dataclass
class BasicSemanticSegmenter:
    """A deterministic OpenCV baseline, not a substitute for Task 2 accuracy."""

    shadow_threshold: int = 52
    bright_threshold: int = 205
    edge_threshold: int = 45
    texture_threshold: float = 22.0

    def segment(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR HxWx3 image, got {None if bgr is None else bgr.shape}")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Local texture and edge strength.
        mean = cv2.blur(gray.astype(np.float32), (9, 9))
        mean_sq = cv2.blur(gray.astype(np.float32) ** 2, (9, 9))
        variance = np.maximum(mean_sq - mean ** 2, 0.0)
        texture = np.sqrt(variance)

        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gradient_x, gradient_y)

        labels = np.full(gray.shape, 1, dtype=np.uint8)  # default ground

        labels[gray < self.shadow_threshold] = 2

        bright = (
            (gray > self.bright_threshold)
            & (hsv[:, :, 1] < 80)
        )
        labels[bright] = 3

        rough = (
            (edge > self.edge_threshold)
            | (texture > self.texture_threshold)
        ) & (gray >= self.shadow_threshold)
        labels[rough] = 4

        # Extremely saturated or invalid-looking pixels remain unknown.
        unknown = (hsv[:, :, 1] > 245) & (hsv[:, :, 2] < 15)
        labels[unknown] = 0
        return labels

    @staticmethod
    def colorize(labels: np.ndarray) -> np.ndarray:
        clipped = np.clip(labels.astype(np.int32), 0, len(PALETTE_BGR) - 1)
        return PALETTE_BGR[clipped]
