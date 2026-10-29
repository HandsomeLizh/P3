#!/usr/bin/env python3
"""Guard the single-source contract for formal RoMa inference defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(PACKAGE_ROOT))

from t3_semantic_mapping.roma_runtime_defaults import (  # noqa: E402
    ROMA_RUNTIME_DEFAULTS,
)


class RomaRuntimeDefaultsContractTest(unittest.TestCase):
    def test_formal_values(self) -> None:
        self.assertEqual(ROMA_RUNTIME_DEFAULTS["coarse_res"], 280)
        self.assertEqual(ROMA_RUNTIME_DEFAULTS["upsample_res"], 280)
        self.assertTrue(
            ROMA_RUNTIME_DEFAULTS["stereo_upsample_refinement"]
        )
        self.assertFalse(
            ROMA_RUNTIME_DEFAULTS["temporal_upsample_refinement"]
        )
        self.assertFalse(ROMA_RUNTIME_DEFAULTS["symmetric_matching"])
        self.assertTrue(ROMA_RUNTIME_DEFAULTS["use_custom_corr"])
        self.assertTrue(ROMA_RUNTIME_DEFAULTS["enable_feature_reuse"])
        self.assertTrue(ROMA_RUNTIME_DEFAULTS["batch_coarse_pairs"])
        self.assertFalse(
            ROMA_RUNTIME_DEFAULTS["enable_klt_temporal_tracking"]
        )

    def test_launch_and_node_use_shared_values(self) -> None:
        launch_source = (
            PACKAGE_ROOT / "launch" / "t5_roma_vo.launch.py"
        ).read_text(encoding="utf-8")
        node_source = (
            PACKAGE_ROOT
            / "t3_semantic_mapping"
            / "roma_semantic_mapper_node.py"
        ).read_text(encoding="utf-8")
        for name in ROMA_RUNTIME_DEFAULTS:
            self.assertIn(
                f'ROMA_RUNTIME_DEFAULTS["{name}"]', node_source
            )
        for name in (
            "coarse_res",
            "upsample_res",
            "stereo_upsample_refinement",
            "temporal_upsample_refinement",
            "symmetric_matching",
            "use_custom_corr",
            "enable_feature_reuse",
            "batch_coarse_pairs",
            "enable_klt_temporal_tracking",
        ):
            self.assertIn(f'launch_default("{name}")', launch_source)

    def test_yaml_does_not_override_inference_defaults(self) -> None:
        yaml_source = (
            PACKAGE_ROOT / "config" / "semantic_map.yaml"
        ).read_text(encoding="utf-8")
        for name in ROMA_RUNTIME_DEFAULTS:
            active_lines = [
                line
                for line in yaml_source.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertFalse(
                any(line.lstrip().startswith(f"{name}:") for line in active_lines),
                f"semantic_map.yaml must not override {name}",
            )


if __name__ == "__main__":
    unittest.main()
