"""Dependency-free regression checks for the RoMa/ROS tracking contract."""

import ast
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROMA_PATH = WORKSPACE / "third_party" / "RoMa" / "roma_vo.py"
MAPPER_PATH = (
    WORKSPACE
    / "src"
    / "t3_semantic_mapping"
    / "t3_semantic_mapping"
    / "roma_semantic_mapper_node.py"
)


def _method(tree, class_name, method_name):
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _is_status_call(statement):
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_set_tracking_status"
        for node in ast.walk(statement)
    )


class TrackingFailureGuardStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roma_tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
        cls.mapper_tree = ast.parse(MAPPER_PATH.read_text(encoding="utf-8"))

    def test_public_tracking_contract_exists(self):
        classes = {
            node.name: node
            for node in self.roma_tree.body
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("TrackingStatus", classes)
        slam_methods = {
            node.name
            for node in classes["StereoSLAMv2"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("get_tracking_status", slam_methods)
        self.assertIn("_set_tracking_status", slam_methods)

    def test_tracking_status_setter_records_failure(self):
        tracking_status = next(
            node
            for node in self.roma_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrackingStatus"
        )
        setter = _method(
            self.roma_tree,
            "StereoSLAMv2",
            "_set_tracking_status",
        )
        getter = _method(
            self.roma_tree,
            "StereoSLAMv2",
            "get_tracking_status",
        )
        miniature_slam = ast.ClassDef(
            name="StereoSLAMv2",
            bases=[],
            keywords=[],
            body=[setter, getter],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="dataclasses",
                        names=[ast.alias(name="dataclass")],
                        level=0,
                    ),
                    ast.ImportFrom(
                        module="typing",
                        names=[ast.alias(name="Optional")],
                        level=0,
                    ),
                    tracking_status,
                    miniature_slam,
                ],
                type_ignores=[],
            )
        )
        namespace = {}
        exec(compile(module, str(ROMA_PATH), "exec"), namespace)

        slam = namespace["StereoSLAMv2"]()
        slam.frame_count = 7
        slam._set_tracking_status(
            success=False,
            reason="pnp_failed",
            inlier_count=3,
            correspondence_count=12,
        )
        status = slam.get_tracking_status()

        self.assertFalse(status.success)
        self.assertFalse(status.pose_updated)
        self.assertEqual(status.frame_index, 7)
        self.assertEqual(status.reason, "pnp_failed")
        self.assertEqual(status.inlier_count, 3)
        self.assertEqual(status.correspondence_count, 12)

    def test_every_tracking_failure_routes_through_guard(self):
        process = _method(self.roma_tree, "StereoSLAMv2", "process_frame")
        guarded_returns = 0
        for node in ast.walk(process):
            if not isinstance(node, ast.Return):
                continue
            expression = node.value
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "_handle_tracking_failure"
            ):
                guarded_returns += 1
        self.assertGreaterEqual(guarded_returns, 5)

        guard = _method(
            self.roma_tree,
            "StereoSLAMv2",
            "_handle_tracking_failure",
        )
        self.assertTrue(_is_status_call(guard))
        self.assertTrue(any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "poses"
            for node in ast.walk(guard)
        ))

    def test_mapper_guard_precedes_map_update(self):
        """VO 跟踪失败帧不得成为视觉地图证据。

        节点引入 LiDAR/LIO 定位模式后，守卫从直接判断
        ``tracking_status.success`` 变成先推导 ``vo_tracking_success``，
        再据其分流。本测试按行为断言：守卫变量必须来自
        ``tracking_status.success``；纯 VO 模式下失败帧必须 return；
        ``_update_map``（视觉证据融合）只允许出现在成功分支且位于守卫之后。
        """
        callback = _method(
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_on_synchronized_input",
        )

        derivations = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "vo_tracking_success"
                for target in node.targets
            )
            and "tracking_status" in ast.dump(node.value)
            and "success" in ast.dump(node.value)
        ]
        self.assertEqual(len(derivations), 1)

        failure_guards = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "vo_tracking_success"
        ]
        self.assertEqual(len(failure_guards), 1)
        guard = failure_guards[0]
        vo_mode_returns = [
            inner
            for inner in ast.walk(guard)
            if isinstance(inner, ast.If)
            and "localization_mode" in ast.dump(inner.test)
            and "'vo'" in ast.dump(inner.test)
            and any(
                isinstance(statement, ast.Return)
                for statement in inner.body
            )
        ]
        self.assertTrue(vo_mode_returns)

        map_update_lines = [
            node.lineno
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_update_map"
        ]
        self.assertEqual(len(map_update_lines), 1)
        self.assertLess(guard.lineno, map_update_lines[0])

        success_branches = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "vo_tracking_success"
        ]
        self.assertTrue(any(
            any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "_update_map"
                for statement in branch.body
                for inner in ast.walk(statement)
            )
            for branch in success_branches
        ))

    def test_recovery_admission_gate_precedes_visual_map_update(self):
        initializer = _method(
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "__init__",
        )
        callback = _method(
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_on_synchronized_input",
        )
        admission_configs = [
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MapAdmissionConfig"
        ]
        self.assertEqual(len(admission_configs), 1)
        enabled_values = [
            keyword.value
            for keyword in admission_configs[0].keywords
            if keyword.arg == "enabled"
        ]
        self.assertEqual(len(enabled_values), 1)
        enabled_dump = ast.dump(enabled_values[0])
        self.assertIn("map_admission_requested", enabled_dump)
        self.assertIn("localization_mode", enabled_dump)
        self.assertIn("'vo'", enabled_dump)

        calls = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        failure_calls = [
            node
            for node in calls
            if node.func.attr == "record_tracking_failure"
        ]
        admission_calls = [
            node
            for node in calls
            if node.func.attr == "observe_tracking_success"
        ]
        map_calls = [
            node for node in calls if node.func.attr == "_update_map"
        ]

        self.assertEqual(len(failure_calls), 2)
        self.assertEqual(len(admission_calls), 1)
        self.assertEqual(len(map_calls), 1)
        self.assertLess(admission_calls[0].lineno, map_calls[0].lineno)
        admission_guards = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.If)
            and "vo_tracking_success" in ast.dump(node.test)
            and "localization_mode" in ast.dump(node.test)
            and any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "observe_tracking_success"
                for statement in node.body
                for inner in ast.walk(statement)
            )
        ]
        self.assertEqual(len(admission_guards), 1)
        keyword_names = {item.arg for item in map_calls[0].keywords}
        self.assertIn("admit_new_observation", keyword_names)

        update = _method(
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_update_map",
        )
        argument_names = {item.arg for item in update.args.kwonlyargs}
        self.assertIn("admit_new_observation", argument_names)
        update_dump = ast.dump(update)
        self.assertIn("admit_new_observation", update_dump)
        self.assertIn("_record_missed_keyframe_observation", update_dump)

    def test_exception_before_visual_admission_closes_vo_gate(self):
        callback = _method(
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_on_synchronized_input",
        )
        callback_dump = ast.dump(callback)

        self.assertIn("visual_admission_observed", callback_dump)
        false_initializers = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "visual_admission_observed"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        ]
        true_assignments = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "visual_admission_observed"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ]
        self.assertEqual(len(false_initializers), 1)
        self.assertEqual(len(true_assignments), 2)

        guarded_exception_handlers = [
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.ExceptHandler)
            and any(
                isinstance(statement, ast.If)
                and "localization_mode" in ast.dump(statement.test)
                and "visual_admission_observed" in ast.dump(statement.test)
                and any(
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "record_tracking_failure"
                    for inner in ast.walk(statement)
                )
                for statement in node.body
            )
        ]
        self.assertEqual(len(guarded_exception_handlers), 1)


if __name__ == "__main__":
    unittest.main()
