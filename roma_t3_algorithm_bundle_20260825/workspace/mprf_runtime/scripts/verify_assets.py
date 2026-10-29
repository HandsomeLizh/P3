#!/usr/bin/env python3
"""Verify that copied source/model assets match the bundled file sizes."""

from __future__ import annotations

import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from mprf_runtime.config import (
    MODEL_SIZES,
    SOURCE_BUNDLE_LABEL,
    MprfRuntimeConfiguration,
)


def main() -> int:
    configuration = MprfRuntimeConfiguration()
    configuration.validate_assets()
    paths = {
        "finetuned_dinov2_v3.pth": configuration.dino_weights,
        "pretrained_salad.ckpt": configuration.salad_weights,
        "retrained_salad.ckpt": (
            configuration.model_root
            / "mprf"
            / "Weights"
            / "retrained_salad.ckpt"
        ),
        "sonata.pth": configuration.sonata_weights,
    }
    for name, expected in MODEL_SIZES.items():
        path = paths[name]
        actual = path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"model file size mismatch for {path}: {actual} != {expected}"
            )
        print(f"[OK] {name}: {actual} bytes")
    print(f"[OK] MPRF source bundle: {SOURCE_BUNDLE_LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
