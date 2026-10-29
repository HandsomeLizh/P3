from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField

from .message_utils import uint8_array_from_buffer


@dataclass(frozen=True)
class OrganizedLidarRecord:
    timestamp_ns: int
    path: Path
    sequence: int
    frame_id: str
    point_count: int


def load_organized_lidar_records(
    sequence_dir: Path,
    *,
    first_timestamp_ns: Optional[int] = None,
    last_timestamp_ns: Optional[int] = None,
    stride: int = 1,
) -> List[OrganizedLidarRecord]:
    """Load raw LiDAR frame metadata without reading PCD payloads."""
    if int(stride) <= 0:
        raise ValueError("LiDAR publish stride must be positive")
    sequence_dir = Path(sequence_dir)
    lidar_dir = sequence_dir / "lidar_raw"
    metadata_path = lidar_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Organized raw LiDAR metadata is missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = []
    for ordinal, entry in enumerate(metadata.get("frames", [])):
        if ordinal % int(stride) != 0:
            continue
        timestamp_ns = int(entry["header_time_ns"])
        if first_timestamp_ns is not None and timestamp_ns < int(
            first_timestamp_ns
        ):
            continue
        if last_timestamp_ns is not None and timestamp_ns > int(
            last_timestamp_ns
        ):
            continue
        path = lidar_dir / str(entry["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"Raw LiDAR PCD is missing: {path}")
        records.append(
            OrganizedLidarRecord(
                timestamp_ns=timestamp_ns,
                path=path,
                sequence=int(entry.get("seq", ordinal)),
                frame_id=str(entry.get("frame_id", "velodyne")),
                point_count=int(entry.get("point_count", 0)),
            )
        )
    if not records:
        raise RuntimeError(
            f"No raw LiDAR frames selected from {metadata_path}"
        )
    return records


_PCD_DATATYPES = {
    ("I", 1): PointField.INT8,
    ("U", 1): PointField.UINT8,
    ("I", 2): PointField.INT16,
    ("U", 2): PointField.UINT16,
    ("I", 4): PointField.INT32,
    ("U", 4): PointField.UINT32,
    ("F", 4): PointField.FLOAT32,
    ("F", 8): PointField.FLOAT64,
}


def read_binary_pcd_message(
    path: Path,
    *,
    stamp: Time,
    frame_id: str,
    max_points: int = 0,
) -> PointCloud2:
    """Read a PCD v0.7 binary payload into PointCloud2.

    ``max_points=0`` preserves the complete binary payload.  A positive cap
    performs deterministic, layout-preserving sampling for dataset playback;
    every retained record still contains all original PCD fields.
    """
    if int(max_points) < 0:
        raise ValueError("max_points cannot be negative")
    path = Path(path)
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header is incomplete: {path}")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exception:
                raise ValueError(f"PCD header is not ASCII: {path}") from exception
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        payload = stream.read()

    data_kind = header.get("DATA", [""])[0].lower()
    if data_kind != "binary":
        raise ValueError(
            f"Only PCD DATA binary is supported, got {data_kind!r}: {path}"
        )
    names = header.get("FIELDS", [])
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = [value.upper() for value in header.get("TYPE", [])]
    counts = [int(value) for value in header.get("COUNT", [])]
    if not counts and names:
        counts = [1] * len(names)
    if not (len(names) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"PCD field metadata lengths differ: {path}")
    width = int(header.get("WIDTH", ["0"])[0])
    height = int(header.get("HEIGHT", ["1"])[0])
    points = int(header.get("POINTS", [str(width * height)])[0])
    if width <= 0 or height <= 0 or points != width * height:
        raise ValueError(
            f"Invalid PCD geometry width={width}, height={height}, "
            f"points={points}: {path}"
        )

    fields = []
    offset = 0
    for name, size, scalar_type, count in zip(names, sizes, types, counts):
        datatype = _PCD_DATATYPES.get((scalar_type, size))
        if datatype is None:
            raise ValueError(
                f"Unsupported PCD scalar TYPE={scalar_type}, SIZE={size}: {path}"
            )
        field = PointField()
        field.name = str(name)
        field.offset = int(offset)
        field.datatype = int(datatype)
        field.count = int(count)
        fields.append(field)
        offset += int(size) * int(count)
    point_step = int(offset)
    row_step = point_step * width
    expected_bytes = row_step * height
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PCD payload size differs: {len(payload)} != {expected_bytes}: {path}"
        )
    if 0 < int(max_points) < points:
        records = np.frombuffer(payload, dtype=np.uint8).reshape(
            points, point_step
        )
        indices = np.linspace(
            0, points - 1, int(max_points), dtype=np.int64
        )
        payload = records[indices].tobytes(order="C")
        width = int(max_points)
        height = 1
        points = width
        row_step = point_step * width

    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = str(frame_id)
    message.height = height
    message.width = width
    message.fields = fields
    message.is_bigendian = False
    message.point_step = point_step
    message.row_step = row_step
    message.data = uint8_array_from_buffer(payload)
    message.is_dense = True
    return message
