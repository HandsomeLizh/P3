#!/usr/bin/env python3
"""Static wiring contracts for completion-aware dataset playback."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
DATASET_NODE = WORKSPACE / (
    "src/t1_dataset_sim/t1_dataset_sim/dataset_sensor_publisher.py"
)
MAPPER_NODE = WORKSPACE / (
    "src/t3_semantic_mapping/t3_semantic_mapping/"
    "roma_semantic_mapper_node.py"
)
PRACTICE_LAUNCHES = (
    WORKSPACE
    / "src/t3_semantic_mapping/launch/practice_live_precomputed.launch.py",
    WORKSPACE
    / "src/t3_semantic_mapping/launch/"
    "practice_live_online_semantics.launch.py",
)


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_node, name):
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _attribute_call_name(call):
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _declared_parameters(path, class_name):
    declarations = []
    for node in ast.walk(_class(_tree(path), class_name)):
        if (
            isinstance(node, ast.Call)
            and _attribute_call_name(node) == "declare_parameter"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            declarations.append(
                (node.args[0].value, node.args[1].value, node.lineno)
            )
    return declarations


class DatasetPlaybackContractTest(unittest.TestCase):
    def test_completion_parameters_are_unique_and_safe_by_default(self):
        declarations = _declared_parameters(
            DATASET_NODE, "DatasetSensorPublisher"
        )
        names = [name for name, _, _ in declarations]
        self.assertEqual(
            {
                name: count
                for name, count in Counter(names).items()
                if count != 1
            },
            {},
        )
        defaults = {name: value for name, value, _ in declarations}
        self.assertFalse(defaults["lossless_playback"])
        self.assertEqual(defaults["lossless_max_in_flight_frames"], 2)
        self.assertEqual(defaults["processing_ack_timeout_sec"], 120.0)
        self.assertEqual(defaults["max_frames"], 0)
        self.assertTrue(defaults["publish_imu"])
        self.assertIn(
            "max_frames cannot be negative",
            DATASET_NODE.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            defaults["input_processed_topic"],
            "/Car/T3/debug/input_processed",
        )

        mapper_defaults = dict(
            (name, value)
            for name, value, _ in _declared_parameters(
                MAPPER_NODE, "RomaSemanticMapperNode"
            )
        )
        self.assertEqual(
            mapper_defaults["input_processed_topic"],
            defaults["input_processed_topic"],
        )

    def test_practice_launches_enable_lossless_mode_and_persist_summary(self):
        for path in PRACTICE_LAUNCHES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("dataset_playback_launch_arguments()", source)
                self.assertIn(
                    "**dataset_playback_parameter_overrides()", source
                )
                self.assertIn("dataset_playback_summary.json", source)

    def test_raw_imu_replay_follows_existing_mapper_switch(self):
        dataset_source = DATASET_NODE.read_text(encoding="utf-8")
        self.assertIn(
            "self._load_imu() if self.publish_imu else []",
            dataset_source,
        )
        self.assertIn("if self.publish_imu", dataset_source)

        for path in PRACTICE_LAUNCHES:
            tree = _tree(path)
            mappings = [
                value
                for node in ast.walk(tree)
                if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant)
                and key.value == "publish_imu"
            ]
            with self.subTest(path=path.name):
                self.assertEqual(len(mappings), 1)
                mapping = mappings[0]
                self.assertIsInstance(mapping, ast.Call)
                self.assertEqual(_attribute_call_name(mapping), None)
                self.assertIsInstance(mapping.func, ast.Name)
                self.assertEqual(mapping.func.id, "LaunchConfiguration")
                self.assertEqual(len(mapping.args), 1)
                self.assertIsInstance(mapping.args[0], ast.Constant)
                self.assertEqual(mapping.args[0].value, "enable_imu")

    def test_playback_summary_names_filtered_input_scope(self):
        source = DATASET_NODE.read_text(encoding="utf-8")
        self.assertIn("complete_input_consumption_scope", source)
        self.assertIn("input_selection", source)
        self.assertIn("max_stereo_skew_ms", source)

    def test_frame_publication_has_explicit_success_or_failure_result(self):
        dataset = _class(_tree(DATASET_NODE), "DatasetSensorPublisher")
        publish_frame = _method(dataset, "_publish_frame")
        return_values = {
            node.value.value
            for node in ast.walk(publish_frame)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(return_values, {False, True})
        self.assertFalse(
            any(
                isinstance(node, ast.Return) and node.value is None
                for node in ast.walk(publish_frame)
            )
        )

    def test_timer_records_every_frame_attempt_and_fail_stops_lossless_error(self):
        dataset = _class(_tree(DATASET_NODE), "DatasetSensorPublisher")
        timer = _method(dataset, "_on_timer")
        call_names = [
            _attribute_call_name(node)
            for node in ast.walk(timer)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("_publish_frame", call_names)
        self.assertIn("record_frame_attempt", call_names)
        self.assertIn("_fail_playback", call_names)
        self.assertIn("_mapper_is_ready", call_names)

        readiness = _method(dataset, "_mapper_is_ready")
        readiness_calls = [
            _attribute_call_name(node)
            for node in ast.walk(readiness)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("count_publishers", readiness_calls)
        self.assertIn("count_subscribers", readiness_calls)

    def test_shutdown_writes_summary_before_destroying_node(self):
        tree = _tree(DATASET_NODE)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [
            (_attribute_call_name(node), node.lineno)
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
        ]
        close_line = next(
            line for name, line in calls if name == "close_resources"
        )
        destroy_line = next(
            line for name, line in calls if name == "destroy_node"
        )
        self.assertLess(close_line, destroy_line)


if __name__ == "__main__":
    unittest.main()
