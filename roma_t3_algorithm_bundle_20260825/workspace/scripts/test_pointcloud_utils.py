#!/usr/bin/env python3
"""Wire and sampling contracts for the vectorized debug PointCloud2 path."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(PACKAGE))

from t3_semantic_mapping.coordinate_utils import (  # noqa: E402
    points_opencv_to_map,
)
from t3_semantic_mapping.pointcloud_utils import (  # noqa: E402
    finite_point_sample_indices,
    make_semantic_point_cloud,
)


def _header() -> Header:
    header = Header()
    header.stamp.sec = 17
    header.stamp.nanosec = 23
    header.frame_id = "map"
    return header


def _fields():
    return [
        PointField(
            name=name,
            offset=4 * index,
            datatype=PointField.FLOAT32,
            count=1,
        )
        for index, name in enumerate(("x", "y", "z", "semantic"))
    ]


def _legacy_message(*, header, points_xyz, semantic_ids, max_points):
    """Frozen reference for the replaced tuple/struct-pack implementation."""

    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    semantics = np.asarray(semantic_ids, dtype=np.uint8).reshape(-1)
    if len(points) != len(semantics):
        raise ValueError("point/semantic mismatch")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    semantics = semantics[finite]
    if max_points > 0 and len(points) > max_points:
        selected = np.linspace(
            0,
            len(points) - 1,
            max_points,
            dtype=np.int64,
        )
        points = points[selected]
        semantics = semantics[selected]
    rows = [
        (float(point[0]), float(point[1]), float(point[2]), float(label))
        for point, label in zip(points, semantics)
    ]
    cloud = point_cloud2.create_cloud(header, _fields(), rows)
    cloud.is_dense = False
    return cloud


class SemanticPointCloudMessageTest(unittest.TestCase):
    def test_sensor_msgs_py_is_test_only_dependency(self):
        manifest = ET.parse(PACKAGE / "package.xml").getroot()
        declarations = [
            (element.tag, (element.text or "").strip())
            for element in manifest
            if element.tag.endswith("depend")
        ]
        self.assertEqual(
            [
                tag
                for tag, dependency in declarations
                if dependency == "sensor_msgs_py"
            ],
            ["test_depend"],
        )

        production_imports = []
        module_dir = PACKAGE / "t3_semantic_mapping"
        for module_path in module_dir.rglob("*.py"):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    production_imports.extend(
                        alias.name for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    production_imports.append(node.module)
        self.assertNotIn("sensor_msgs_py", production_imports)

    def test_vectorized_message_is_byte_identical_to_legacy_cdr(self):
        rng = np.random.default_rng(20260815)
        cases = []
        for count, maximum in ((0, 0), (7, 0), (101, 13)):
            points = rng.normal(size=(count, 3)).astype(np.float64)
            semantics = rng.integers(0, 5, size=count, dtype=np.uint8)
            if count:
                points[count // 2, 1] = np.nan
            cases.append((points, semantics, maximum))

        for points, semantics, maximum in cases:
            with self.subTest(count=len(points), maximum=maximum):
                expected = _legacy_message(
                    header=_header(),
                    points_xyz=points,
                    semantic_ids=semantics,
                    max_points=maximum,
                )
                actual = make_semantic_point_cloud(
                    header=_header(),
                    points_xyz=points,
                    semantic_ids=semantics,
                    max_points=maximum,
                )
                self.assertEqual(
                    serialize_message(actual), serialize_message(expected)
                )
                self.assertEqual(actual.data.typecode, "B")
                self.assertEqual(actual.point_step, 16)
                self.assertEqual(actual.row_step, 16 * actual.width)

    def test_finite_global_linspace_indices_preserve_endpoints(self):
        points = np.arange(30, dtype=np.float64).reshape(10, 3)
        points[[1, 8], 0] = np.nan
        actual = finite_point_sample_indices(points, 4)
        finite = np.array([0, 2, 3, 4, 5, 6, 7, 9])
        expected = finite[np.linspace(0, 7, 4, dtype=np.int64)]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            finite_point_sample_indices(points, 0), finite
        )

    def test_posttransform_sampling_preserves_complete_output_message(self):
        rng = np.random.default_rng(7)
        points_cv = rng.normal(size=(1000, 3)).astype(np.float64)
        rotation = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        offset = np.array([2.0, -3.0, 0.5], dtype=np.float64)
        points_map = points_opencv_to_map(points_cv, rotation, offset)
        semantics = (
            np.arange(len(points_cv), dtype=np.uint32) % 5
        ).astype(np.uint8)
        expected = _legacy_message(
            header=_header(),
            points_xyz=points_map,
            semantic_ids=semantics,
            max_points=50,
        )

        selected = finite_point_sample_indices(points_map, 50)
        sampled_map = points_map[selected]
        actual = make_semantic_point_cloud(
            header=_header(),
            points_xyz=sampled_map,
            semantic_ids=semantics[selected],
            max_points=50,
        )
        self.assertEqual(
            serialize_message(actual), serialize_message(expected)
        )

    def test_point_semantic_length_mismatch_still_fails(self):
        with self.assertRaisesRegex(ValueError, "Point and semantic"):
            make_semantic_point_cloud(
                header=_header(),
                points_xyz=np.zeros((2, 3)),
                semantic_ids=np.zeros(1, dtype=np.uint8),
            )

    def test_mapper_samples_after_transform_before_semantic_lookup(self):
        path = PACKAGE / "t3_semantic_mapping/roma_semantic_mapper_node.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RomaSemanticMapperNode"
        )
        method = next(
            node
            for node in mapper.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_publish_current_map_outputs"
        )
        calls = [
            node.func.id
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ]
        self.assertIn("finite_point_sample_indices", calls)
        calls_by_line = sorted(
            (node.lineno, node.func.attr)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"semantic_ids_at"}
        )
        sample_call = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "finite_point_sample_indices"
        )
        self.assertLess(sample_call.lineno, calls_by_line[0][0])


if __name__ == "__main__":
    unittest.main()
