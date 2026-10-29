"""Dependency-free structural checks for tracking-loss relocalization."""

import ast
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROMA_PATH = WORKSPACE / "third_party" / "RoMa" / "roma_vo.py"


def _class(tree, name):
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_node, name):
    return next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class RelocalizationContractStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROMA_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.slam = _class(cls.tree, "StereoSLAMv2")
        cls.relocalizer = _class(cls.tree, "Relocalizer")
        cls.loop = _class(cls.tree, "LoopClosing")

    def _source(self, node):
        return ast.get_source_segment(self.source, node)

    def test_database_is_independent_of_optional_backend(self):
        constructor = _method(self.slam, "__init__")
        relocalizer_calls = [
            node for node in ast.walk(constructor)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Relocalizer"
        ]
        self.assertEqual(len(relocalizer_calls), 1)
        backend_branches = [
            node for node in ast.walk(constructor)
            if isinstance(node, ast.If)
            and "enable_backend" in ast.dump(node.test)
        ]
        self.assertTrue(backend_branches)
        self.assertFalse(any(
            relocalizer_calls[0] in ast.walk(branch)
            for branch in backend_branches
        ))

    def test_query_reuses_full_loop_geometry_without_pose_prior(self):
        query = self._source(_method(self.relocalizer, "query"))
        self.assertIn("_mutual_ratio_matches", query)
        self.assertIn("self.geometry._verify", query)
        self.assertNotIn("min_interval", query)
        self.assertNotIn("candidate.pose", query)
        self.assertNotIn("current.pose", query)

        verify = self._source(_method(self.loop, "_verify"))
        self.assertIn("loop_feature_indices", verify)
        self.assertIn("current_feature_indices", verify)
        self.assertIn("shared_inliers", verify)

    def test_every_keyframe_registers_detached_geometry(self):
        create = self._source(_method(self.slam, "_create_keyframe"))
        self.assertIn("make_keyframe_geometry_snapshot", create)
        self.assertIn("relocalizer.add_snapshot", create)
        self.assertLess(
            create.index("make_keyframe_geometry_snapshot"),
            create.index("self.keyframes.append"),
        )

    def test_failed_frontend_paths_attempt_recovery(self):
        process = _method(self.slam, "process_frame")
        recovery_calls = [
            node for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_handle_tracking_failure"
        ]
        self.assertGreaterEqual(len(recovery_calls), 5)
        disabled = [
            call for call in recovery_calls
            if any(
                keyword.arg == "allow_relocalization"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in call.keywords
            )
        ]
        # Only the branch with too little current stereo geometry is unable to
        # satisfy the existing reciprocal-PnP minimum.
        self.assertEqual(len(disabled), 1)

    def test_acceptance_composes_live_pose_and_forces_new_anchor(self):
        attempt = self._source(_method(self.slam, "_try_relocalize"))
        commit = self._source(_method(self.slam, "_commit_relocalized_frame"))
        self.assertIn(
            "reference.pose.copy() @ constraint.T_loop_current", attempt)
        self.assertIn("self._commit_relocalized_frame", attempt)
        self.assertIn("self.poses.append", commit)
        self.assertIn("self._create_keyframe", commit)
        self.assertIn("self._submit_keyframe_to_backend", commit)
        self.assertIn('reason="relocalized"', commit)
        self.assertIn('self.tracking_state = "TRACKING"', commit)

    def test_rejection_only_updates_tracking_state_and_status(self):
        guard = self._source(_method(self.slam, "_handle_tracking_failure"))
        self.assertIn("self._try_relocalize", guard)
        self.assertIn("self._set_tracking_status", guard)
        self.assertIn("return self.poses[-1]", guard)
        self.assertNotIn("self.poses.append", guard)
        self.assertNotIn("self.keyframes.append", guard)
        self.assertNotIn("self.global_mps", guard)


if __name__ == "__main__":
    unittest.main()
