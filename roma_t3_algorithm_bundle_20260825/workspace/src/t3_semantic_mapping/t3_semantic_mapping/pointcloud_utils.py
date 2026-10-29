from __future__ import annotations

from array import array

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def make_xyz_point_cloud(*, header, points_xyz: np.ndarray) -> PointCloud2:
    """Pack metric XYZ without assigning unobserved semantic labels."""
    points = np.ascontiguousarray(points_xyz, dtype="<f4").reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    payload = array("B")
    if points.size:
        payload.frombytes(points.tobytes())
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=[
            PointField(name=name, offset=4 * index,
                       datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(("x", "y", "z"))
        ],
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(points),
        data=payload,
        is_dense=True,
    )


def finite_point_sample_indices(
    points_xyz: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """Return the established finite/global-linspace debug-cloud indices."""

    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    if int(max_points) > 0 and len(indices) > int(max_points):
        selected = np.linspace(
            0,
            len(indices) - 1,
            int(max_points),
            dtype=np.int64,
        )
        indices = indices[selected]
    return indices


def make_semantic_point_cloud(
    *,
    header,
    points_xyz: np.ndarray,
    semantic_ids: np.ndarray,
    max_points: int = 50000,
) -> PointCloud2:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    semantics = np.asarray(semantic_ids, dtype=np.uint8).reshape(-1)

    if len(points) != len(semantics):
        raise ValueError(
            f"Point and semantic counts differ: {len(points)} != {len(semantics)}"
        )

    indices = finite_point_sample_indices(points, max_points)
    points = points[indices]
    semantics = semantics[indices]

    fields = [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="semantic",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]

    # The previous sensor_msgs_py helper built a Python tuple for every point,
    # packed every tuple separately, then assigned a generic bytes sequence.
    # A contiguous little-endian Nx4 float32 matrix is the identical declared
    # PointCloud2 row layout and lets the ROS message use its typed-buffer path.
    rows = np.empty((len(points), 4), dtype="<f4")
    rows[:, :3] = points
    rows[:, 3] = semantics
    payload = array("B")
    if rows.size:
        payload.frombytes(memoryview(rows).cast("B"))

    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = len(rows)
    cloud.is_dense = False
    cloud.is_bigendian = False
    cloud.fields = fields
    cloud.point_step = 16
    cloud.row_step = 16 * len(rows)
    cloud.data = payload
    return cloud
