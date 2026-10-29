import ast
from pathlib import Path

import numpy as np


ROMA_PATH = Path(__file__).resolve().parents[1] / "roma_vo.py"


def _load_policy():
    tree = ast.parse(ROMA_PATH.read_text(encoding="utf-8"))
    slam = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StereoSLAMv2"
    )
    policy = next(
        node
        for node in slam.body
        if isinstance(node, ast.FunctionDef) and node.name == "_need_keyframe"
    )
    miniature = ast.ClassDef(
        name="StereoSLAMv2",
        bases=[],
        keywords=[],
        body=[policy],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[miniature], type_ignores=[])
    )
    namespace = {"np": np}
    exec(compile(module, str(ROMA_PATH), "exec"), namespace)
    return namespace["StereoSLAMv2"]


StereoSLAMv2 = _load_policy()


def _slam(
    frame_count,
    last_keyframe,
    interval=5,
    min_gap=2,
    min_inliers=25,
    interval_from_last=True,
):
    slam = StereoSLAMv2()
    slam.frame_count = frame_count
    slam._last_kf_frame_idx = last_keyframe
    slam.keyframe_interval = interval
    slam.keyframe_min_gap = min_gap
    slam.keyframe_min_inliers = min_inliers
    slam.keyframe_interval_from_last = interval_from_last
    return slam


def test_minimum_gap_prevents_consecutive_keyframes():
    slam = _slam(frame_count=10, last_keyframe=9)
    transform = np.eye(4)
    transform[0, 3] = 1.0
    assert slam._need_keyframe(transform, n_inliers=1) is False


def test_interval_is_measured_from_last_keyframe():
    slam = _slam(frame_count=9, last_keyframe=4)
    assert slam._need_keyframe(np.eye(4), n_inliers=100) is True


def test_legacy_interval_keeps_absolute_frame_modulo():
    slam = _slam(
        frame_count=10,
        last_keyframe=8,
        min_gap=1,
        interval_from_last=False,
    )
    assert slam._need_keyframe(np.eye(4), n_inliers=100) is True


def test_configured_low_inlier_trigger_is_respected():
    slam = _slam(frame_count=4, last_keyframe=1)
    assert slam._need_keyframe(np.eye(4), n_inliers=30) is False
    assert slam._need_keyframe(np.eye(4), n_inliers=20) is True
