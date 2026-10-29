#!/usr/bin/env python3
"""Static guard for the compact T5 production output profile."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
NODE = WORKSPACE / (
    "src/t3_semantic_mapping/t3_semantic_mapping/"
    "roma_semantic_mapper_node.py"
)
LAUNCH = WORKSPACE / (
    "src/t3_semantic_mapping/launch/t5_roma_vo.launch.py"
)
FASTLIVO_LAUNCH = WORKSPACE / (
    "src/t3_semantic_mapping/launch/t5_fastlivo2.launch.py"
)
ROMA_RUNNER = WORKSPACE / "scripts/run_t5_roma_vo.sh"
FASTLIVO_RUNNER = WORKSPACE / "scripts/run_t5_fastlivo2.sh"


def _declared_parameter_defaults(path: Path) -> dict[str, object]:
    defaults: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "declare_parameter" or len(call.args) < 2:
            continue
        try:
            name = ast.literal_eval(call.args[0])
            value = ast.literal_eval(call.args[1])
        except (ValueError, TypeError):
            continue
        defaults[str(name)] = value
    return defaults


def _launch_defaults(path: Path) -> dict[str, object]:
    defaults: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "DeclareLaunchArgument" or not call.args:
            continue
        try:
            name = ast.literal_eval(call.args[0])
        except (ValueError, TypeError):
            continue
        for keyword in call.keywords:
            if keyword.arg != "default_value":
                continue
            try:
                defaults[str(name)] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
    return defaults


class ProductionOutputContractTest(unittest.TestCase):
    def test_production_output_defaults(self) -> None:
        node_defaults = _declared_parameter_defaults(NODE)
        self.assertFalse(node_defaults["publish_odometry_compat"])
        self.assertFalse(node_defaults["publish_grid_map_odom_alias"])
        self.assertFalse(node_defaults["publish_input_processed"])
        self.assertTrue(node_defaults["publish_feature_matches"])
        self.assertEqual(
            node_defaults["feature_matches_topic"],
            "/Car/T2/feature_matching/visualization",
        )
        self.assertEqual(
            node_defaults["feature_visualization_crop_width"], 1280
        )
        self.assertEqual(
            node_defaults["feature_visualization_crop_height"], 720
        )
        self.assertEqual(
            node_defaults["feature_visualization_width"], 640
        )
        self.assertEqual(
            node_defaults["feature_visualization_height"], 480
        )
        launch_defaults = _launch_defaults(LAUNCH)
        self.assertEqual(
            launch_defaults["enable_legacy_output_aliases"], "false"
        )
        self.assertEqual(
            launch_defaults["publish_input_processed"], "false"
        )
        self.assertEqual(
            launch_defaults["publish_feature_matches"], "true"
        )
        self.assertEqual(
            launch_defaults["feature_matches_topic"],
            "/Car/T2/feature_matching/visualization",
        )
        launch_source = LAUNCH.read_text(encoding="utf-8")
        self.assertIn(
            '"feature_matches_topic": LaunchConfiguration(', launch_source
        )
        self.assertNotIn("feature_visualization_left_topic", launch_source)
        self.assertNotIn("feature_visualization_right_topic", launch_source)

    def test_internal_artifacts_are_grouped(self) -> None:
        source = NODE.read_text(encoding="utf-8")
        self.assertIn(
            'self.internal_output_dir = self.output_dir / "_internal"',
            source,
        )
        self.assertIn(
            'self.internal_output_dir / "grid_map_snapshot"', source
        )
        self.assertIn(
            'self.internal_output_dir / "map_observation_archive"', source
        )

    def test_lite_map_mode_controls_full_map_products(self) -> None:
        for launch_path in (LAUNCH, FASTLIVO_LAUNCH):
            launch_defaults = _launch_defaults(launch_path)
            self.assertEqual(
                launch_defaults["full_global_map"], "true"
            )
            source = launch_path.read_text(encoding="utf-8")
            self.assertIn('"publish_global_grid_map": ParameterValue(', source)
            self.assertIn('"save_dense_global_grid_map": ParameterValue(', source)
            self.assertIn(
                'LaunchConfiguration("full_global_map")', source
            )

        for runner_path in (ROMA_RUNNER, FASTLIVO_RUNNER):
            source = runner_path.read_text(encoding="utf-8")
            self.assertIn("--lite-map", source)
            self.assertIn("--full-map", source)
            self.assertIn("full_global_map:=", source)


if __name__ == "__main__":
    unittest.main()
