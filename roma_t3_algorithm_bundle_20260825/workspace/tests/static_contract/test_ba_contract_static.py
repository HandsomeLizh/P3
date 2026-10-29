"""Dependency-free structural regression checks for the repaired stereo BA."""

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


class StereoBAContractStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROMA_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.keyframe = _class(cls.tree, "KeyFrame")
        cls.ba = _class(cls.tree, "SlidingWindowBA")
        cls.slam = _class(cls.tree, "StereoSLAMv2")

    def test_keyframe_keeps_aligned_right_observations(self):
        fields = {
            node.target.id
            for node in self.keyframe.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        }
        self.assertIn("right_keypoints", fields)

    def test_window_eviction_does_not_destroy_history(self):
        method = _method(self.ba, "_marginalize")
        text = ast.get_source_segment(self.source, method)
        self.assertNotIn("is_bad =", text)
        self.assertNotIn("remove_observation", text)
        self.assertIn("self.keyframes.pop(0)", text)

    def test_joint_ba_has_metric_baseline_and_fixed_boundary(self):
        constructor = ast.get_source_segment(
            self.source, _method(self.ba, "__init__"))
        optimize = ast.get_source_segment(
            self.source, _method(self.ba, "optimize"))
        self.assertIn("baseline", constructor)
        self.assertIn("pc[stereo, 0] - self.baseline", optimize)
        self.assertIn("n_free_poses = n_poses - 1", optimize)
        self.assertIn("for pi in range(1, n_poses)", optimize)

    def test_ba_writeback_is_guarded_and_current_pose_is_returned(self):
        optimize = ast.get_source_segment(
            self.source, _method(self.ba, "optimize"))
        process = ast.get_source_segment(
            self.source, _method(self.slam, "process_frame"))
        submit = ast.get_source_segment(
            self.source, _method(self.slam, "_submit_keyframe_to_backend"))
        self.assertIn("cost_not_improved", optimize)
        self.assertIn("self.last_accepted = True", optimize)
        self.assertIn("if self.sw_ba.last_accepted", submit)
        self.assertIn("return kf.pose.copy()", submit)
        self.assertIn("T_wc = self._submit_keyframe_to_backend(kf)", process)


if __name__ == "__main__":
    unittest.main()
