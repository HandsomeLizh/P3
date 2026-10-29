"""Dependency-free structural checks for the loop-closing repair."""

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


class LoopClosingContractStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROMA_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.loop = _class(cls.tree, "LoopClosing")
        cls.backend = _class(cls.tree, "BackendOptimizer")
        cls.slam = _class(cls.tree, "StereoSLAMv2")

    def test_candidate_search_has_no_estimated_position_gate(self):
        detect = ast.get_source_segment(
            self.source, _method(self.loop, "_detect"))
        self.assertIn("_mutual_ratio_matches", detect)
        self.assertNotIn("pose[:3", detect)
        self.assertNotIn("np.linalg.norm", detect)

    def test_geometry_produces_independent_relative_measurement(self):
        verify = ast.get_source_segment(
            self.source, _method(self.loop, "_verify"))
        self.assertIn("_estimate_pnp", verify)
        self.assertIn("T_loop_current = np.linalg.inv(T_current_loop)", verify)
        self.assertNotIn("loop.pose", verify)
        self.assertNotIn("current.pose", verify)

    def test_worker_uses_snapshots_and_result_queue(self):
        add_keyframe = ast.get_source_segment(
            self.source, _method(self.backend, "add_keyframe"))
        callback = ast.get_source_segment(
            self.source, _method(self.backend, "_on_loop_result"))
        self.assertIn("_make_loop_snapshot", add_keyframe)
        self.assertIn("self.loop_results.put", callback)
        self.assertNotIn(".pose =", callback)

    def test_main_thread_commits_loop_corrections(self):
        apply_result = ast.get_source_segment(
            self.source, _method(self.slam, "_apply_pending_loop_results"))
        process = ast.get_source_segment(
            self.source, _method(self.slam, "process_frame"))
        self.assertIn("consume_loop_results", apply_result)
        self.assertIn("for mp in self.global_mps.values()", apply_result)
        self.assertIn("self._sync_pose_history_from_keyframes", apply_result)
        self.assertIn("self._apply_pending_loop_results()", process)


if __name__ == "__main__":
    unittest.main()
