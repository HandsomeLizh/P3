#!/usr/bin/env python3
"""Load all pinned models and optionally run one exact-revisit candidate."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from mprf_runtime.config import MprfRuntimeConfiguration
from mprf_runtime.contracts import KeyframePacket
from mprf_runtime.engine import MprfOnlineEngine


def packet(node_id: int, image: np.ndarray, points: np.ndarray) -> KeyframePacket:
    height, width = image.shape[:2]
    return KeyframePacket(
        session_id="model-smoke",
        node_id=node_id,
        timestamp_ns=(node_id + 1) * 1_000_000_000,
        image_bgr=image,
        points_sensor=points,
        camera_matrix=np.asarray(
            [
                [0.8 * width, 0.0, 0.5 * width],
                [0.0, 0.8 * width, 0.5 * height],
                [0.0, 0.0, 1.0],
            ]
        ),
        transform_camera_sensor=np.eye(4),
        calibration_width=width,
        calibration_height=height,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 224px/900-point pose inference instead of production settings",
    )
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="t3_mprf_smoke_") as temporary:
        configuration = MprfRuntimeConfiguration(
            state_root=Path(temporary),
            device=arguments.device,
            pose_image_size=224 if arguments.quick else 560,
            min_keyframe_separation=1,
            min_time_separation_sec=0.0,
            retrieval_shortlist=1,
            refined_shortlist=1,
            min_salad_similarity=0.1,
            min_dino_similarity=0.1,
            max_pose_points=900 if arguments.quick else 2500,
            match_min_similarity=0.70,
            match_min_margin=0.001,
            ransac_residual_m=0.05,
            ransac_min_inliers=10,
            ransac_min_inlier_ratio=0.05,
            ransac_max_rmse_m=0.03,
            ransac_min_spread_m=0.05,
        )
        engine = MprfOnlineEngine(configuration)
        print(engine.health())
        if arguments.load_only:
            return 0
        generator = np.random.default_rng(20260815)
        image = generator.integers(0, 256, (224, 320, 3), dtype=np.uint8)
        points = generator.uniform(
            [-2.0, -1.5, 3.0], [2.0, 1.5, 10.0], (1400, 3)
        ).astype(np.float32)
        first = engine.process(packet(0, image, points))
        second = engine.process(packet(1, image, points))
        print(first)
        print(second)
        if second["status"] != "candidate":
            raise RuntimeError(
                "exact-revisit smoke test did not produce an MPRF candidate"
            )
        pose = np.asarray(
            second["candidate"]["relative_pose_reference_query"]
        ).reshape(4, 4)
        if not np.allclose(pose, np.eye(4), atol=0.05):
            raise RuntimeError(f"exact-revisit coarse pose is not identity:\n{pose}")
    print("[OK] Full MPRF model smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
