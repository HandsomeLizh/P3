#!/usr/bin/env python3
"""Static contracts for mapper parameters and ROS endpoint ownership."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import unittest

import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
NODE_PATH = WORKSPACE / (
    "src/t3_semantic_mapping/t3_semantic_mapping/"
    "roma_semantic_mapper_node.py"
)
LAUNCH_OPTIONS_PATH = WORKSPACE / (
    "src/t3_semantic_mapping/t3_semantic_mapping/launch_options.py"
)
CONFIG_PATH = WORKSPACE / (
    "src/t3_semantic_mapping/config/semantic_map.yaml"
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal(expression: ast.AST, constants: dict[str, object]):
    if isinstance(expression, ast.Name) and expression.id in constants:
        return constants[expression.id]
    try:
        return ast.literal_eval(expression)
    except (ValueError, TypeError):
        # Ubuntu 20.04's deployment Python is 3.8 and has no ast.unparse().
        # Dynamic path defaults do not overlap the YAML/shared launch values
        # compared below, so a stable sentinel is sufficient.
        return "<dynamic>"


def _module_constants(tree: ast.Module) -> dict[str, object]:
    constants = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            pass
    return constants


def _mapper_class(tree: ast.Module) -> ast.ClassDef:
    return next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.ClassDef)
        and statement.name == "RomaSemanticMapperNode"
    )


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        statement
        for statement in class_node.body
        if isinstance(statement, ast.FunctionDef) and statement.name == name
    )


def _attribute_call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _node_parameter_calls():
    tree = _tree(NODE_PATH)
    constants = _module_constants(tree)
    mapper = _mapper_class(tree)
    declarations = []
    reads = []
    for node in ast.walk(mapper):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        call_name = _attribute_call_name(node)
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(
            first.value, str
        ):
            continue
        if call_name == "declare_parameter":
            declarations.append(
                (
                    first.value,
                    _literal(node.args[1], constants),
                    node.lineno,
                )
            )
        elif call_name == "get_parameter":
            reads.append((first.value, node.lineno))
    return declarations, reads


def _normalise_launch_default(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.startswith("["):
        return ast.literal_eval(stripped)
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return value


def _launch_argument_calls():
    calls = []
    for node in ast.walk(_tree(LAUNCH_OPTIONS_PATH)):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "DeclareLaunchArgument"
            or not node.args
            or not isinstance(node.args[0], ast.Constant)
        ):
            continue
        default = None
        for keyword in node.keywords:
            if keyword.arg == "default_value":
                default = ast.literal_eval(keyword.value)
                break
        calls.append((node.args[0].value, default, node.lineno))
    return calls


def _returned_dict_keys(function: ast.FunctionDef) -> set[str]:
    return_statement = next(
        node for node in ast.walk(function) if isinstance(node, ast.Return)
    )
    if not isinstance(return_statement.value, ast.Dict):
        raise AssertionError(f"{function.name} must directly return a dict")
    return {
        key.value
        for key in return_statement.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _self_attribute_name(expression: ast.AST) -> str | None:
    if (
        isinstance(expression, ast.Attribute)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "self"
    ):
        return expression.attr
    return None


class MapperWiringContractTest(unittest.TestCase):
    def test_mprf_results_are_polled_after_sensor_input_stops(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        interface_method = _method(mapper, "_create_ros_interfaces")
        timer_callback = _method(mapper, "_on_mprf_result_timer")

        timer_assignments = [
            node
            for node in ast.walk(interface_method)
            if isinstance(node, ast.Assign)
            and any(
                _self_attribute_name(target) == "mprf_result_timer"
                for target in node.targets
            )
        ]
        self.assertEqual(len(timer_assignments), 1)
        self.assertIsInstance(timer_assignments[0].value, ast.IfExp)
        self.assertIn("mprf_client", ast.dump(timer_assignments[0].value.test))

        callback_dump = ast.dump(timer_callback)
        self.assertIn("_poll_mprf_loop_results", callback_dump)
        self.assertIn("_synchronize_map_state", callback_dump)
        self.assertIn("_rebuild_path_and_trajectory", callback_dump)

    def test_direct_lio_bypass_is_wired_without_a_new_runtime_switch(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        initializer = _method(mapper, "__init__")
        create_slam = _method(mapper, "_create_slam")
        callback = _method(mapper, "_on_synchronized_input")
        direct_update = _method(
            mapper, "_update_map_without_vo_tracking"
        )

        policy_calls = [
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "visual_frontend_required"
        ]
        self.assertEqual(len(policy_calls), 1)
        policy_keywords = {item.arg for item in policy_calls[0].keywords}
        self.assertEqual(
            policy_keywords,
            {
                "localization_mode",
                "enable_lidar_mapping",
                "lidar_mapping_mode",
                "lidar_fallback_to_stereo",
            },
        )

        bypass_guards = [
            node
            for node in ast.walk(create_slam)
            if isinstance(node, ast.If)
            and "visual_frontend_enabled" in ast.dump(node.test)
            and any(
                isinstance(child, ast.Return)
                and isinstance(child.value, ast.Tuple)
                and any(
                    isinstance(item, ast.Constant) and item.value is None
                    for item in child.value.elts
                )
                for statement in node.body
                for child in ast.walk(statement)
            )
        ]
        self.assertEqual(len(bypass_guards), 1)
        algorithm_imports = [
            node
            for node in ast.walk(create_slam)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
        ]
        self.assertTrue(algorithm_imports)
        self.assertLess(
            bypass_guards[0].lineno,
            min(node.lineno for node in algorithm_imports),
        )

        tracking_calls = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "process_visual_tracking"
        ]
        self.assertEqual(len(tracking_calls), 1)
        enabled = next(
            item.value
            for item in tracking_calls[0].keywords
            if item.arg == "enabled"
        )
        self.assertEqual(
            ast.dump(enabled),
            ast.dump(
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="visual_frontend_enabled",
                    ctx=ast.Load(),
                )
            ),
        )
        self.assertIn("visual_tracking_attempted", ast.dump(callback))
        self.assertTrue(
            any(
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr == "frame_count"
                for node in ast.walk(direct_update)
            )
        )

    def test_node_parameters_are_declared_once_and_all_are_consumed(self):
        declarations, reads = _node_parameter_calls()
        names = [name for name, _, _ in declarations]
        counts = Counter(names)
        self.assertEqual(
            {name: count for name, count in counts.items() if count != 1},
            {},
        )
        declared = set(names)
        read = {name for name, _ in reads}
        self.assertEqual(read - declared, set())
        self.assertEqual(declared - read, set())

    def test_shared_launch_defaults_match_direct_node_defaults(self):
        declarations, _ = _node_parameter_calls()
        node_defaults = {name: value for name, value, _ in declarations}
        launch_calls = _launch_argument_calls()
        launch_names = [name for name, _, _ in launch_calls]
        self.assertEqual(
            [name for name, count in Counter(launch_names).items() if count > 1],
            [],
        )
        for name, launch_default, _ in launch_calls:
            if name not in node_defaults:
                continue
            with self.subTest(parameter=name):
                self.assertEqual(
                    _normalise_launch_default(launch_default),
                    node_defaults[name],
                )

    def test_mapper_launch_override_keys_are_declared_node_parameters(self):
        declarations, _ = _node_parameter_calls()
        declared = {name for name, _, _ in declarations}
        tree = _tree(LAUNCH_OPTIONS_PATH)
        mapper_override_functions = {
            "input_topic_parameter_overrides",
            "input_parameter_overrides",
            "localization_parameter_overrides",
            "mapping_parameter_overrides",
        }
        keys = set()
        keys_by_function = {}
        for function in tree.body:
            if (
                isinstance(function, ast.FunctionDef)
                and function.name in mapper_override_functions
            ):
                function_keys = _returned_dict_keys(function)
                keys_by_function[function.name] = function_keys
                keys.update(function_keys)
        self.assertEqual(keys - declared, set())
        self.assertIn(
            "enable_imu", keys_by_function["input_parameter_overrides"]
        )
        self.assertNotIn(
            "enable_imu",
            keys_by_function["input_topic_parameter_overrides"],
        )
        self.assertIn(
            "input_processed_topic",
            keys_by_function["input_topic_parameter_overrides"],
        )

    def test_parameter_file_defaults_match_direct_node_defaults(self):
        declarations, _ = _node_parameter_calls()
        node_defaults = {name: value for name, value, _ in declarations}
        parameters = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
            "roma_semantic_mapper_node"
        ]["ros__parameters"]
        self.assertEqual(set(parameters) - set(node_defaults), set())
        for name, configured_default in parameters.items():
            with self.subTest(parameter=name):
                self.assertEqual(
                    _normalise_launch_default(configured_default),
                    node_defaults[name],
                )

    def test_launch_switches_are_not_shadowed_by_parameter_file(self):
        parameters = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
            "roma_semantic_mapper_node"
        ]["ros__parameters"]
        launch_parameters = {
            name
            for name, _, _ in _launch_argument_calls()
            if name not in {"publish_lio_odometry", "lio_publish_stride"}
        }
        self.assertEqual(set(parameters) & launch_parameters, set())

    def test_all_rclpy_endpoints_are_centralised_and_owned(self):
        tree = _tree(NODE_PATH)
        mapper = _mapper_class(tree)
        interface_method = _method(mapper, "_create_ros_interfaces")
        endpoint_names = {
            "create_publisher",
            "create_subscription",
            "create_timer",
            "create_service",
        }
        parent = {}
        for node in ast.walk(mapper):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        calls = [
            node
            for node in ast.walk(mapper)
            if isinstance(node, ast.Call)
            and _attribute_call_name(node) in endpoint_names
        ]
        self.assertGreater(len(calls), 0)
        for call in calls:
            with self.subTest(line=call.lineno):
                self.assertGreaterEqual(call.lineno, interface_method.lineno)
                self.assertLessEqual(call.lineno, interface_method.end_lineno)
                ancestor = parent[call]
                while not isinstance(ancestor, (ast.Assign, ast.AnnAssign)):
                    ancestor = parent[ancestor]
                targets = (
                    ancestor.targets
                    if isinstance(ancestor, ast.Assign)
                    else [ancestor.target]
                )
                self.assertTrue(
                    any(_self_attribute_name(target) for target in targets)
                )

    def test_disabled_imu_has_neither_subscription_nor_callback_group(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        interface_method = _method(mapper, "_create_ros_interfaces")
        assignments = {}
        for node in ast.walk(interface_method):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = _self_attribute_name(node.targets[0])
            if name is not None:
                assignments[name] = node.value
        for name in ("imu_callback_group", "imu_subscription"):
            with self.subTest(attribute=name):
                expression = assignments[name]
                self.assertIsInstance(expression, ast.IfExp)
                self.assertEqual(_self_attribute_name(expression.test), "enable_imu")
                self.assertIsInstance(expression.orelse, ast.Constant)
                self.assertIsNone(expression.orelse.value)

    def test_primary_sync_callback_precedes_passive_diagnostics(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        interface_method = _method(mapper, "_create_ros_interfaces")
        registrations = []
        for node in ast.walk(interface_method):
            if (
                not isinstance(node, ast.Call)
                or _attribute_call_name(node) != "registerCallback"
                or not node.args
            ):
                continue
            callback_name = _self_attribute_name(node.args[0])
            registrations.append((node.lineno, callback_name))
        primary_line = next(
            line for line, name in registrations if name == "_on_synchronized_input"
        )
        diagnostic_line = next(
            line for line, name in registrations if name == "_record_image_input"
        )
        self.assertLess(primary_line, diagnostic_line)

    def test_every_synchronized_callback_exit_publishes_one_receipt(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        callback = _method(mapper, "_on_synchronized_input")
        completion_calls = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and _attribute_call_name(node) == "_finish_synchronized_input"
        ]
        outcomes = {
            keyword.value.value
            for call in completion_calls
            for keyword in call.keywords
            if keyword.arg == "outcome"
            and isinstance(keyword.value, ast.Constant)
        }
        self.assertEqual(len(completion_calls), 3)
        self.assertEqual(
            outcomes,
            {"stereo_skew_rejected", "reentrant_rejected"},
        )
        processing_finally = next(
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Call)
                and _attribute_call_name(child)
                == "_finish_synchronized_input"
                for statement in node.finalbody
                for child in ast.walk(statement)
            )
        )
        self.assertGreater(len(processing_finally.finalbody), 0)

    def test_accepted_callback_classifies_each_runtime_stage(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        callback = _method(mapper, "_on_synchronized_input")
        switches = sorted(
            (
                node.lineno,
                node.args[0].value,
            )
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and _attribute_call_name(node) == "switch"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
        self.assertEqual(
            [stage for _, stage in switches],
            [
                "visual_tracking",
                "localization_and_pose",
                "map_update_and_publish",
                "callback_bookkeeping",
            ],
        )
        finish_calls = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and _attribute_call_name(node) == "finish"
        ]
        self.assertEqual(len(finish_calls), 1)
        completion = next(
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and _attribute_call_name(node) == "_finish_synchronized_input"
            and any(
                keyword.arg == "stage_durations_sec"
                for keyword in node.keywords
            )
        )
        timing_keyword = next(
            keyword
            for keyword in completion.keywords
            if keyword.arg == "stage_durations_sec"
        )
        self.assertIsInstance(timing_keyword.value, ast.Name)
        self.assertEqual(timing_keyword.value.id, "stage_durations")

    def test_stage_diagnostics_cannot_suppress_completion_receipt(self):
        mapper = _mapper_class(_tree(NODE_PATH))
        callback = _method(mapper, "_on_synchronized_input")
        finish_try = next(
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Call)
                and _attribute_call_name(child) == "finish"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        self.assertTrue(
            any(
                isinstance(handler.type, ast.Name)
                and handler.type.id == "Exception"
                for handler in finish_try.handlers
            )
        )

        completion = _method(mapper, "_finish_synchronized_input")
        accounting_try = next(
            node
            for node in ast.walk(completion)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Call)
                and _attribute_call_name(child) == "record_terminal"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        self.assertTrue(
            any(
                isinstance(handler.type, ast.Name)
                and handler.type.id == "ValueError"
                and any(
                    isinstance(child, ast.Call)
                    and _attribute_call_name(child) == "record_terminal"
                    for statement in handler.body
                    for child in ast.walk(statement)
                )
                for handler in accounting_try.handlers
            )
        )
        receipt_publish = next(
            node
            for node in ast.walk(completion)
            if isinstance(node, ast.Call)
            and _attribute_call_name(node) == "publish"
        )
        self.assertGreater(receipt_publish.lineno, accounting_try.lineno)


if __name__ == "__main__":
    unittest.main()
