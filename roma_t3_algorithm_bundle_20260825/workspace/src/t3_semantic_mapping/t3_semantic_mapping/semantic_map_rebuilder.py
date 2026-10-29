from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
from typing import Optional
import zlib

import numpy as np

from .coordinate_utils import points_opencv_to_map
from .lidar_occupancy import (
    DEFAULT_LIDAR_GROUND_CONFIG,
    ENDPOINT_FREE,
    ENDPOINT_OCCUPIED,
    ENDPOINT_UNKNOWN,
    classify_lidar_endpoint_occupancy,
)
from .map_contribution_index import MapContributionIndex
from .semantic_grid import LayeredSemanticGridMap


@dataclass(frozen=True)
class SemanticKeyFrameObservation:
    """Immutable semantic geometry in one keyframe camera frame.

    Stereo observations require positive optical depth.  A 360-degree LiDAR
    observation is already transformed into this same camera frame but may
    legitimately contain points behind the optical plane, so it opts out of
    that one filter while retaining finite-point validation.
    """

    keyframe_id: int
    sensor_timestamp: float
    camera_points: np.ndarray
    semantic_labels: np.ndarray
    sensor_origin_camera: Optional[np.ndarray] = None
    require_positive_depth: bool = True
    source: str = "stereo"

    def __post_init__(self) -> None:
        sensor_timestamp = float(self.sensor_timestamp)
        if not np.isfinite(sensor_timestamp):
            raise ValueError("sensor_timestamp must be finite")
        points = np.asarray(self.camera_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(self.semantic_labels, dtype=np.uint8).reshape(-1)
        if len(points) != len(labels):
            raise ValueError(
                "camera_points and semantic_labels differ: "
                f"{len(points)} != {len(labels)}"
            )
        origin = (
            np.zeros(3, dtype=np.float64)
            if self.sensor_origin_camera is None
            else np.asarray(
                self.sensor_origin_camera, dtype=np.float64
            ).reshape(3)
        )
        if not np.isfinite(origin).all():
            raise ValueError("sensor_origin_camera must be finite")
        finite = np.isfinite(points).all(axis=1)
        if bool(self.require_positive_depth):
            finite &= points[:, 2] > 0.1
        points = points[finite].copy()
        labels = labels[finite].copy()
        origin = origin.copy()
        points.setflags(write=False)
        labels.setflags(write=False)
        origin.setflags(write=False)
        object.__setattr__(self, "keyframe_id", int(self.keyframe_id))
        object.__setattr__(self, "sensor_timestamp", sensor_timestamp)
        object.__setattr__(self, "camera_points", points)
        object.__setattr__(self, "semantic_labels", labels)
        object.__setattr__(self, "sensor_origin_camera", origin)
        object.__setattr__(
            self, "require_positive_depth", bool(self.require_positive_depth)
        )
        object.__setattr__(self, "source", str(self.source))


class KeyFrameObservationStore:
    """Replayable semantic provenance with an optional bounded RAM cache.

    With ``archive_path=None`` this retains the original in-memory behaviour,
    which keeps lightweight unit-test and library callers source-compatible.
    A configured archive uses one transactional SQLite file as a page-backed
    store and keeps only the most recently used observations in RAM.  The
    database transaction is committed before the live index/cache changes, so
    a write failure cannot create a partially visible observation batch.
    """

    _ARCHIVE_SCHEMA_VERSION = 1

    def __init__(
        self,
        archive_path: Optional[Path] = None,
        *,
        memory_cache_size: Optional[int] = None,
        compression_level: int = 1,
    ) -> None:
        if memory_cache_size is not None and int(memory_cache_size) <= 0:
            raise ValueError("memory_cache_size must be positive when set")
        if archive_path is None and memory_cache_size is not None:
            raise ValueError(
                "A bounded memory cache requires archive_path so evicted "
                "observations remain replayable"
            )
        compression_level = int(compression_level)
        if compression_level < 0 or compression_level > 9:
            raise ValueError("compression_level must be between 0 and 9")

        self._lock = threading.RLock()
        self._observations: OrderedDict[
            int, SemanticKeyFrameObservation
        ] = OrderedDict()
        self._keyframe_ids: set[int] = set()
        self._point_counts: dict[int, int] = {}
        self._sensor_timestamps: dict[int, float] = {}
        self._memory_cache_size = (
            int(memory_cache_size)
            if memory_cache_size is not None
            else None
        )
        self._compression_level = compression_level
        self._archive_path = (
            Path(archive_path).expanduser()
            if archive_path is not None
            else None
        )
        self._connection: Optional[sqlite3.Connection] = None
        self._closed = False

        if self._archive_path is not None:
            self._open_archive()

    @property
    def archive_path(self) -> Optional[Path]:
        return self._archive_path

    @property
    def resident_count(self) -> int:
        """Number of full observations currently resident in process RAM."""
        with self._lock:
            return len(self._observations)

    @property
    def archive_size_bytes(self) -> int:
        """Current SQLite main/WAL byte count for operator diagnostics."""
        if self._archive_path is None:
            return 0
        paths = (
            self._archive_path,
            Path(f"{self._archive_path}-wal"),
        )
        return sum(path.stat().st_size for path in paths if path.exists())

    def _open_archive(self) -> None:
        assert self._archive_path is not None
        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(self._archive_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            # Completeness is preferred over write throughput: once add_many
            # returns, SQLite has synchronised the committed provenance batch.
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=30000")
            # Bound SQLite's own page cache as well as the Python observation
            # cache. A negative cache_size value is interpreted as KiB.
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS archive_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            version_row = connection.execute(
                "SELECT value FROM archive_metadata WHERE key='schema_version'"
            ).fetchone()
            if version_row is None:
                connection.execute(
                    "INSERT INTO archive_metadata(key, value) VALUES (?, ?)",
                    ("schema_version", str(self._ARCHIVE_SCHEMA_VERSION)),
                )
            elif int(version_row[0]) != self._ARCHIVE_SCHEMA_VERSION:
                raise RuntimeError(
                    "Unsupported semantic observation archive schema "
                    f"{version_row[0]!r}; expected "
                    f"{self._ARCHIVE_SCHEMA_VERSION}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    keyframe_id INTEGER PRIMARY KEY,
                    sensor_timestamp REAL NOT NULL,
                    point_count INTEGER NOT NULL,
                    camera_points BLOB NOT NULL,
                    semantic_labels BLOB NOT NULL,
                    sensor_origin_camera BLOB NOT NULL,
                    require_positive_depth INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    compressed INTEGER NOT NULL
                )
                """
            )
            rows = connection.execute(
                "SELECT keyframe_id, point_count, sensor_timestamp "
                "FROM observations"
            ).fetchall()
        except Exception:
            connection.close()
            raise

        self._connection = connection
        self._keyframe_ids = {int(row[0]) for row in rows}
        self._point_counts = {
            int(row[0]): int(row[1]) for row in rows
        }
        self._sensor_timestamps = {
            int(row[0]): float(row[2]) for row in rows
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("KeyFrameObservationStore is closed")

    def __len__(self) -> int:
        with self._lock:
            return len(self._keyframe_ids)

    @property
    def keyframe_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._keyframe_ids))

    @property
    def replay_keyframe_ids(self) -> tuple[int, ...]:
        """IDs in deterministic sensor-time order for nonlinear fusion."""
        with self._lock:
            return tuple(
                sorted(
                    self._keyframe_ids,
                    key=lambda keyframe_id: (
                        self._sensor_timestamps[keyframe_id],
                        keyframe_id,
                    ),
                )
            )

    @staticmethod
    def observation_order_key(
        observation: SemanticKeyFrameObservation,
    ) -> tuple[float, int]:
        return (
            float(observation.sensor_timestamp),
            int(observation.keyframe_id),
        )

    @property
    def latest_archive_order_key(self) -> Optional[tuple[float, int]]:
        with self._lock:
            if not self._keyframe_ids:
                return None
            return max(
                (
                    self._sensor_timestamps[keyframe_id],
                    keyframe_id,
                )
                for keyframe_id in self._keyframe_ids
            )

    def observations_follow_archive(
        self,
        observations: Iterable[SemanticKeyFrameObservation],
    ) -> bool:
        """Whether a batch can be appended without changing replay order."""
        batch = tuple(observations)
        if not batch:
            return True
        latest = self.latest_archive_order_key
        if latest is None:
            return True
        earliest = min(
            self.observation_order_key(observation)
            for observation in batch
        )
        return earliest > latest

    def point_count(self, keyframe_id: int) -> Optional[int]:
        """Return archived point count without loading the point arrays."""
        with self._lock:
            return self._point_counts.get(int(keyframe_id))

    def transformed_point_sample(
        self,
        *,
        keyframe_poses_cv: Mapping[int, np.ndarray],
        max_points: int,
    ) -> np.ndarray:
        """Return the original global-linspace debug point-cloud sample.

        This does not affect map evidence. Selecting global indices before
        loading observations reproduces the old post-concatenation
        ``np.linspace`` cap while keeping peak RAM proportional to the output
        limit and the configured LRU cache, rather than total run duration.
        """
        max_points = int(max_points)
        with self._lock:
            self._ensure_open()
            entries = []
            for keyframe_id in sorted(self._keyframe_ids):
                source_pose = keyframe_poses_cv.get(keyframe_id)
                if source_pose is None:
                    continue
                pose = np.asarray(source_pose, dtype=np.float64)
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError(
                        f"Invalid debug-cloud pose for keyframe {keyframe_id}"
                    )
                point_count = self._point_counts.get(keyframe_id)
                if point_count is None:
                    raise RuntimeError(
                        "Observation archive has incomplete point-count index"
                    )
                entries.append((keyframe_id, pose.copy(), int(point_count)))

            total_points = sum(entry[2] for entry in entries)
            selected_global_indices = None
            if max_points > 0 and total_points > max_points:
                selected_global_indices = np.linspace(
                    0,
                    total_points - 1,
                    max_points,
                    dtype=np.int64,
                )

            clouds = []
            global_offset = 0
            for keyframe_id, pose, point_count in entries:
                if selected_global_indices is None:
                    local_indices = None
                else:
                    first = int(
                        np.searchsorted(
                            selected_global_indices,
                            global_offset,
                            side="left",
                        )
                    )
                    last = int(
                        np.searchsorted(
                            selected_global_indices,
                            global_offset + point_count,
                            side="left",
                        )
                    )
                    local_indices = (
                        selected_global_indices[first:last] - global_offset
                    )
                global_offset += point_count
                if local_indices is not None and len(local_indices) == 0:
                    continue

                observation = self._get_locked(keyframe_id)
                if observation is None:
                    raise RuntimeError(
                        f"Observation archive lost keyframe {keyframe_id}"
                    )
                if len(observation.camera_points) != point_count:
                    raise RuntimeError(
                        f"Observation point count changed for keyframe "
                        f"{keyframe_id}"
                    )
                local_points = observation.camera_points.astype(
                    np.float64, copy=False
                )
                if local_indices is not None:
                    local_points = local_points[local_indices]
                clouds.append(
                    (pose[:3, :3] @ local_points.T).T + pose[:3, 3]
                )

            return (
                np.vstack(clouds)
                if clouds
                else np.empty((0, 3), dtype=np.float64)
            )

    def get(self, keyframe_id: int) -> SemanticKeyFrameObservation | None:
        with self._lock:
            self._ensure_open()
            return self._get_locked(int(keyframe_id))

    def _get_locked(
        self, keyframe_id: int
    ) -> SemanticKeyFrameObservation | None:
        observation = self._observations.get(keyframe_id)
        if observation is not None:
            self._observations.move_to_end(keyframe_id)
            return observation
        if keyframe_id not in self._keyframe_ids:
            return None
        if self._connection is None:
            raise RuntimeError(
                f"Observation index contains {keyframe_id} without an archive"
            )
        row = self._connection.execute(
            """
            SELECT sensor_timestamp, point_count, camera_points,
                   semantic_labels, sensor_origin_camera,
                   require_positive_depth, source, compressed
            FROM observations WHERE keyframe_id=?
            """,
            (keyframe_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"Archive index is missing keyframe {keyframe_id}"
            )
        observation = self._observation_from_row(keyframe_id, row)
        self._cache_locked(observation)
        return observation

    @staticmethod
    def _decode_blob(blob: bytes, *, compressed: bool) -> bytes:
        try:
            return zlib.decompress(blob) if compressed else bytes(blob)
        except zlib.error as exception:
            raise RuntimeError(
                "Corrupt compressed semantic observation blob"
            ) from exception

    def _observation_from_row(
        self, keyframe_id: int, row
    ) -> SemanticKeyFrameObservation:
        point_count = int(row[1])
        compressed = bool(row[7])
        points_raw = self._decode_blob(row[2], compressed=compressed)
        labels_raw = self._decode_blob(row[3], compressed=compressed)
        origin_raw = self._decode_blob(row[4], compressed=compressed)
        expected_point_bytes = point_count * 3 * np.dtype("<f8").itemsize
        expected_label_bytes = point_count * np.dtype(np.uint8).itemsize
        expected_origin_bytes = 3 * np.dtype("<f8").itemsize
        if (
            len(points_raw) != expected_point_bytes
            or len(labels_raw) != expected_label_bytes
            or len(origin_raw) != expected_origin_bytes
        ):
            raise RuntimeError(
                f"Corrupt semantic observation sizes for keyframe {keyframe_id}"
            )
        points = np.frombuffer(points_raw, dtype="<f8").reshape(-1, 3)
        labels = np.frombuffer(labels_raw, dtype=np.uint8)
        origin = np.frombuffer(origin_raw, dtype="<f8")
        observation = SemanticKeyFrameObservation(
            keyframe_id=keyframe_id,
            sensor_timestamp=float(row[0]),
            camera_points=points,
            semantic_labels=labels,
            sensor_origin_camera=origin,
            require_positive_depth=bool(row[5]),
            source=str(row[6]),
        )
        if len(observation.camera_points) != point_count:
            raise RuntimeError(
                f"Corrupt semantic observation values for keyframe {keyframe_id}"
            )
        return observation

    def _encode_blob(self, array: np.ndarray, dtype) -> bytes:
        raw = np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")
        if self._compression_level > 0:
            return zlib.compress(raw, self._compression_level)
        return raw

    def _archive_values(
        self, observation: SemanticKeyFrameObservation
    ) -> tuple:
        return (
            int(observation.keyframe_id),
            float(observation.sensor_timestamp),
            int(len(observation.camera_points)),
            sqlite3.Binary(
                self._encode_blob(observation.camera_points, "<f8")
            ),
            sqlite3.Binary(
                self._encode_blob(observation.semantic_labels, np.uint8)
            ),
            sqlite3.Binary(
                self._encode_blob(observation.sensor_origin_camera, "<f8")
            ),
            int(bool(observation.require_positive_depth)),
            str(observation.source),
            int(self._compression_level > 0),
        )

    def _cache_locked(
        self, observation: SemanticKeyFrameObservation
    ) -> None:
        keyframe_id = int(observation.keyframe_id)
        self._observations[keyframe_id] = observation
        self._observations.move_to_end(keyframe_id)
        if self._memory_cache_size is not None:
            while len(self._observations) > self._memory_cache_size:
                self._observations.popitem(last=False)

    @staticmethod
    def observations_equal(
        first: SemanticKeyFrameObservation,
        second: SemanticKeyFrameObservation,
    ) -> bool:
        return (
            first.keyframe_id == second.keyframe_id
            and first.sensor_timestamp == second.sensor_timestamp
            and np.array_equal(first.camera_points, second.camera_points)
            and np.array_equal(first.semantic_labels, second.semantic_labels)
            and np.array_equal(
                first.sensor_origin_camera, second.sensor_origin_camera
            )
            and first.require_positive_depth == second.require_positive_depth
            and first.source == second.source
        )

    def add(self, observation: SemanticKeyFrameObservation) -> bool:
        """Insert exactly once; return False only for an identical replay."""
        return bool(self.add_many((observation,)))

    def add_many(
        self,
        observations: Iterable[SemanticKeyFrameObservation],
    ) -> tuple[int, ...]:
        """Validate a batch completely, then commit all new observations.

        A conflict in any item leaves the store unchanged. Identical replays
        are accepted but are not reported as newly inserted IDs.
        """
        batch: dict[int, SemanticKeyFrameObservation] = {}
        for observation in observations:
            keyframe_id = int(observation.keyframe_id)
            previous = batch.get(keyframe_id)
            if previous is not None and not self.observations_equal(
                previous, observation
            ):
                raise ValueError(
                    f"Batch contains conflicting keyframe {keyframe_id}"
                )
            batch[keyframe_id] = observation

        with self._lock:
            self._ensure_open()
            additions: dict[int, SemanticKeyFrameObservation] = {}
            for keyframe_id, observation in batch.items():
                existing = self._get_locked(keyframe_id)
                if existing is None:
                    additions[keyframe_id] = observation
                    continue
                if not self.observations_equal(existing, observation):
                    raise ValueError(
                        f"Keyframe {keyframe_id} already has a different "
                        "observation"
                    )

            if not additions:
                return ()

            # SQLite gives the disk-backed path the same all-or-nothing batch
            # semantics as the original dictionary implementation. Encoding
            # stays inside the transaction so injected serialization or I/O
            # failures roll back every row in the batch.
            if self._connection is not None:
                try:
                    self._connection.execute("BEGIN IMMEDIATE")
                    for keyframe_id in sorted(additions):
                        self._connection.execute(
                            """
                            INSERT INTO observations(
                                keyframe_id, sensor_timestamp, point_count,
                                camera_points, semantic_labels,
                                sensor_origin_camera, require_positive_depth,
                                source, compressed
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            self._archive_values(additions[keyframe_id]),
                        )
                    self._connection.execute("COMMIT")
                except Exception:
                    if self._connection.in_transaction:
                        self._connection.execute("ROLLBACK")
                    raise

            # No live mutation happens until every item has passed validation
            # and, when enabled, the durable transaction has committed.
            for keyframe_id in sorted(additions):
                observation = additions[keyframe_id]
                self._keyframe_ids.add(keyframe_id)
                self._point_counts[keyframe_id] = len(
                    observation.camera_points
                )
                self._sensor_timestamps[keyframe_id] = float(
                    observation.sensor_timestamp
                )
                self._cache_locked(observation)
            return tuple(sorted(additions))

    def close(self) -> None:
        """Checkpoint and close the optional page-backed archive."""
        with self._lock:
            if self._closed:
                return
            if self._connection is not None:
                try:
                    self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    self._connection.close()
                    self._connection = None
            self._observations.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        del exception_type, exception, traceback
        self.close()

    @staticmethod
    def fuse_observation(
        *,
        grid: LayeredSemanticGridMap,
        observation: SemanticKeyFrameObservation,
        pose_world_camera: np.ndarray,
        rotation_map_world: np.ndarray,
        map_translation_offset: np.ndarray,
    ) -> dict[str, float]:
        pose = np.asarray(pose_world_camera, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"Invalid pose for keyframe {observation.keyframe_id}: "
                f"{pose.shape}"
            )

        local_points = observation.camera_points.astype(np.float64, copy=False)
        points_world = (
            pose[:3, :3] @ local_points.T
        ).T + pose[:3, 3]
        sensor_origin_world = (
            pose[:3, :3] @ observation.sensor_origin_camera
            + pose[:3, 3]
        )
        points_map = points_opencv_to_map(
            points_world,
            rotation_map_world,
            map_translation_offset,
        )
        sensor_origin_map = points_opencv_to_map(
            sensor_origin_world.reshape(1, 3),
            rotation_map_world,
            map_translation_offset,
        )[0]
        endpoint_states = None
        if observation.source == "lidar":
            endpoint_states = classify_lidar_endpoint_occupancy(
                points_map,
                sensor_origin_map,
                config=getattr(
                    grid,
                    "lidar_ground_config",
                    DEFAULT_LIDAR_GROUND_CONFIG,
                ),
            )
        statistics = grid.update(
            points_map=points_map,
            semantic_labels=observation.semantic_labels,
            camera_position_map=sensor_origin_map,
            geometric_endpoint_states=endpoint_states,
        )
        if endpoint_states is not None:
            statistics = dict(statistics)
            statistics.update(
                {
                    "lidar_ground_endpoints": int(
                        np.count_nonzero(endpoint_states == ENDPOINT_FREE)
                    ),
                    "lidar_obstacle_endpoints": int(
                        np.count_nonzero(endpoint_states == ENDPOINT_OCCUPIED)
                    ),
                    "lidar_unclassified_endpoints": int(
                        np.count_nonzero(endpoint_states == ENDPOINT_UNKNOWN)
                    ),
                }
            )
        return statistics

    @staticmethod
    def _capture_updated_tiles(
        *,
        grid: LayeredSemanticGridMap,
        keyframe_id: int,
        destination: Optional[dict[int, tuple]],
    ) -> None:
        if destination is None:
            return
        tile_keys = getattr(grid, "last_updated_tile_keys", None)
        if tile_keys is None:
            raise RuntimeError(
                "Tile contribution capture requires a tiled map manager"
            )
        destination[int(keyframe_id)] = tuple(
            sorted((int(key[0]), int(key[1])) for key in tile_keys)
        )

    @staticmethod
    def _require_live_poses(
        *,
        keyframe_ids: Iterable[int],
        keyframe_poses_cv: Mapping[int, np.ndarray],
        operation: str,
    ) -> tuple[int, ...]:
        ordered_ids = tuple(sorted({int(value) for value in keyframe_ids}))
        missing_keyframes = [
            keyframe_id
            for keyframe_id in ordered_ids
            if keyframe_poses_cv.get(keyframe_id) is None
        ]
        if missing_keyframes:
            preview = ", ".join(
                str(keyframe_id) for keyframe_id in missing_keyframes[:8]
            )
            suffix = "..." if len(missing_keyframes) > 8 else ""
            raise RuntimeError(
                f"Cannot {operation}: no live pose for "
                f"{len(missing_keyframes)} keyframe(s): "
                f"{preview}{suffix}"
            )
        return ordered_ids

    def _replay_ordered_ids(
        self, keyframe_ids: Iterable[int]
    ) -> tuple[int, ...]:
        selected_ids = {int(value) for value in keyframe_ids}
        with self._lock:
            unknown_ids = selected_ids - self._keyframe_ids
            if unknown_ids:
                raise RuntimeError(
                    "Cannot order unknown archived keyframe(s): "
                    f"{tuple(sorted(unknown_ids))}"
                )
            return tuple(
                sorted(
                    selected_ids,
                    key=lambda keyframe_id: (
                        self._sensor_timestamps[keyframe_id],
                        keyframe_id,
                    ),
                )
            )

    def replay_keyframes(
        self,
        *,
        grid_factory: Callable[[], LayeredSemanticGridMap],
        keyframe_ids: Iterable[int],
        keyframe_poses_cv: Mapping[int, np.ndarray],
        rotation_map_world: np.ndarray,
        map_translation_offset: np.ndarray,
        tile_keys_by_keyframe: Optional[dict[int, tuple]] = None,
    ) -> tuple[LayeredSemanticGridMap, dict[str, float]]:
        """Replay selected archived observations in sensor-time order."""
        required_ids = self._require_live_poses(
            keyframe_ids=keyframe_ids,
            keyframe_poses_cv=keyframe_poses_cv,
            operation="replay corrected map keyframes",
        )
        ordered_ids = self._replay_ordered_ids(required_ids)
        candidate = grid_factory()
        accepted_points = 0
        visible_points = 0
        latest_statistics: dict[str, float] = {
            "accepted_points": 0,
            "visible_points": 0,
            "known_cells": 0,
            "known_area_m2": 0.0,
            "elevation_min": float("nan"),
            "elevation_max": float("nan"),
        }
        for keyframe_id in ordered_ids:
            observation = self.get(keyframe_id)
            if observation is None:
                raise RuntimeError(
                    f"Archive lost indexed keyframe {keyframe_id}"
                )
            latest_statistics = self.fuse_observation(
                grid=candidate,
                observation=observation,
                pose_world_camera=keyframe_poses_cv[keyframe_id],
                rotation_map_world=rotation_map_world,
                map_translation_offset=map_translation_offset,
            )
            self._capture_updated_tiles(
                grid=candidate,
                keyframe_id=keyframe_id,
                destination=tile_keys_by_keyframe,
            )
            accepted_points += int(
                latest_statistics.get("accepted_points", 0)
            )
            visible_points += int(
                latest_statistics.get("visible_points", 0)
            )

        result = dict(latest_statistics)
        result["accepted_points"] = int(accepted_points)
        result["visible_points"] = int(visible_points)
        result["replayed_keyframes"] = int(len(ordered_ids))
        result["missing_keyframes"] = 0
        return candidate, result

    def fuse_observations(
        self,
        *,
        grid: LayeredSemanticGridMap,
        observations: Iterable[SemanticKeyFrameObservation],
        keyframe_poses_cv: Mapping[int, np.ndarray],
        rotation_map_world: np.ndarray,
        map_translation_offset: np.ndarray,
        tile_keys_by_keyframe: Optional[dict[int, tuple]] = None,
    ) -> dict[str, float]:
        """Fuse a deterministic batch into a grid owned by the caller."""
        ordered = sorted(
            observations,
            key=lambda item: (item.sensor_timestamp, item.keyframe_id),
        )
        accepted_points = 0
        visible_points = 0
        known_cells = getattr(grid, "known_cells", None)
        if known_cells is None:
            known_cells = int(np.count_nonzero(grid.elevation_count))
        latest_statistics: dict[str, float] = {
            "accepted_points": 0,
            "visible_points": 0,
            "known_cells": int(known_cells),
        }
        seen_ids: set[int] = set()
        for observation in ordered:
            keyframe_id = int(observation.keyframe_id)
            if keyframe_id in seen_ids:
                raise ValueError(
                    f"Duplicate keyframe {keyframe_id} in fusion batch"
                )
            seen_ids.add(keyframe_id)
            pose = keyframe_poses_cv.get(keyframe_id)
            if pose is None:
                raise RuntimeError(
                    f"No live pose for keyframe {keyframe_id}"
                )
            latest_statistics = self.fuse_observation(
                grid=grid,
                observation=observation,
                pose_world_camera=pose,
                rotation_map_world=rotation_map_world,
                map_translation_offset=map_translation_offset,
            )
            self._capture_updated_tiles(
                grid=grid,
                keyframe_id=keyframe_id,
                destination=tile_keys_by_keyframe,
            )
            accepted_points += int(
                latest_statistics.get("accepted_points", 0)
            )
            visible_points += int(
                latest_statistics.get("visible_points", 0)
            )

        result = dict(latest_statistics)
        result["accepted_points"] = int(accepted_points)
        result["visible_points"] = int(visible_points)
        result["fused_keyframes"] = int(len(ordered))
        return result

    def build_incremental_candidate(
        self,
        *,
        live_grid: LayeredSemanticGridMap,
        observations: Iterable[SemanticKeyFrameObservation],
        keyframe_poses_cv: Mapping[int, np.ndarray],
        rotation_map_world: np.ndarray,
        map_translation_offset: np.ndarray,
        tile_keys_by_keyframe: Optional[dict[int, tuple]] = None,
    ) -> tuple[LayeredSemanticGridMap, dict[str, float]]:
        """Apply a batch to a detached grid, never to the published grid."""
        clone_for_update = getattr(live_grid, "clone_for_update", None)
        candidate = (
            clone_for_update()
            if callable(clone_for_update)
            else deepcopy(live_grid)
        )
        statistics = self.fuse_observations(
            grid=candidate,
            observations=observations,
            keyframe_poses_cv=keyframe_poses_cv,
            rotation_map_world=rotation_map_world,
            map_translation_offset=map_translation_offset,
            tile_keys_by_keyframe=tile_keys_by_keyframe,
        )
        return candidate, statistics

    def rebuild(
        self,
        *,
        grid_factory: Callable[[], LayeredSemanticGridMap],
        keyframe_poses_cv: Mapping[int, np.ndarray],
        rotation_map_world: np.ndarray,
        map_translation_offset: np.ndarray,
        tile_keys_by_keyframe: Optional[dict[int, tuple]] = None,
        additional_observations: Iterable[
            SemanticKeyFrameObservation
        ] = (),
    ) -> tuple[LayeredSemanticGridMap, dict[str, float]]:
        """Build a complete candidate map without mutating the live map."""
        additional = tuple(additional_observations)
        additional_by_id = {
            int(observation.keyframe_id): observation
            for observation in additional
        }
        if len(additional_by_id) != len(additional):
            raise ValueError(
                "Duplicate keyframe in additional rebuild observations"
            )
        with self._lock:
            self._ensure_open()
            archived_items = tuple(
                (
                    self._sensor_timestamps[keyframe_id],
                    keyframe_id,
                    None,
                )
                for keyframe_id in self._keyframe_ids
            )
        archived_ids = {item[1] for item in archived_items}
        overlap = archived_ids & set(additional_by_id)
        if overlap:
            raise ValueError(
                "Additional rebuild observation is already archived: "
                f"{tuple(sorted(overlap))}"
            )
        replay_items = list(archived_items)
        replay_items.extend(
            (
                float(observation.sensor_timestamp),
                int(observation.keyframe_id),
                observation,
            )
            for observation in additional
        )
        replay_items.sort(key=lambda item: (item[0], item[1]))
        keyframe_ids = tuple(item[1] for item in replay_items)
        missing_keyframes = [
            keyframe_id
            for keyframe_id in keyframe_ids
            if keyframe_poses_cv.get(keyframe_id) is None
        ]
        if missing_keyframes:
            preview = ", ".join(
                str(keyframe_id) for keyframe_id in missing_keyframes[:8]
            )
            suffix = "..." if len(missing_keyframes) > 8 else ""
            raise RuntimeError(
                "Cannot rebuild a complete map: no live pose for "
                f"{len(missing_keyframes)} archived keyframe(s): "
                f"{preview}{suffix}"
            )

        candidate = grid_factory()
        accepted_points = 0
        replayed_keyframes = 0
        latest_statistics: dict[str, float] = {
            "accepted_points": 0,
            "known_cells": 0,
            "known_area_m2": 0.0,
            "elevation_min": float("nan"),
            "elevation_max": float("nan"),
        }

        for _timestamp, keyframe_id, additional_observation in replay_items:
            pose = keyframe_poses_cv[keyframe_id]
            observation = (
                additional_observation
                if additional_observation is not None
                else self.get(keyframe_id)
            )
            if observation is None:
                raise RuntimeError(
                    f"Archive lost indexed keyframe {keyframe_id}"
                )
            latest_statistics = self.fuse_observation(
                grid=candidate,
                observation=observation,
                pose_world_camera=pose,
                rotation_map_world=rotation_map_world,
                map_translation_offset=map_translation_offset,
            )
            self._capture_updated_tiles(
                grid=candidate,
                keyframe_id=keyframe_id,
                destination=tile_keys_by_keyframe,
            )
            accepted_points += int(
                latest_statistics.get("accepted_points", 0)
            )
            replayed_keyframes += 1

        result = dict(latest_statistics)
        result["accepted_points"] = int(accepted_points)
        result["replayed_keyframes"] = int(replayed_keyframes)
        result["missing_keyframes"] = 0
        return candidate, result

    def build_corrected_candidate(
        self,
        *,
        live_grid: LayeredSemanticGridMap,
        grid_factory: Callable[[], LayeredSemanticGridMap],
        contribution_index: Optional[MapContributionIndex],
        pending_observations: Iterable[SemanticKeyFrameObservation],
        keyframe_poses_cv: Mapping[int, np.ndarray],
        rotation_map_world: np.ndarray,
        map_translation_offset: np.ndarray,
    ) -> tuple[
        LayeredSemanticGridMap,
        dict,
        Optional[MapContributionIndex],
    ]:
        """Build an exact corrected candidate with a bounded fast path.

        A local correction rebuilds only the union of each changed
        keyframe's old and new tile footprints.  Every archived observation
        that can contribute to those tiles is replayed in deterministic
        sensor-time order. If the dependency index is unavailable or that
        work would cost at least as much as a full replay, this method
        automatically uses the original complete rebuild path.
        """
        committed_ids = self.keyframe_ids

        def replay_cost(keyframe_ids: Iterable[int]) -> int:
            total = 0
            for keyframe_id in keyframe_ids:
                point_count = self.point_count(keyframe_id)
                if point_count is None:
                    raise RuntimeError(
                        "No archived point count for keyframe "
                        f"{keyframe_id}"
                    )
                total += max(1, int(point_count))
            return total

        full_replay_cost = replay_cost(committed_ids)
        pending = tuple(
            sorted(pending_observations, key=lambda item: item.keyframe_id)
        )
        pending_ids = tuple(int(item.keyframe_id) for item in pending)
        if len(set(pending_ids)) != len(pending_ids):
            raise ValueError("Duplicate keyframe in pending correction batch")
        overlap = set(committed_ids) & set(pending_ids)
        if overlap:
            raise ValueError(
                "Pending correction batch already exists in archive: "
                f"{tuple(sorted(overlap))}"
            )
        self._require_live_poses(
            keyframe_ids=committed_ids + pending_ids,
            keyframe_poses_cv=keyframe_poses_cv,
            operation="build a complete corrected map",
        )

        index_ready = contribution_index is not None
        full_reason = "dependency_index_unavailable"
        if index_ready:
            try:
                contribution_index.assert_consistent()
                index_ready = contribution_index.is_complete_for(
                    committed_ids
                )
            except RuntimeError:
                index_ready = False
                full_reason = "dependency_index_inconsistent"
            else:
                if not index_ready:
                    full_reason = "dependency_index_incomplete"

        partial_capable = all(
            callable(getattr(live_grid, method, None))
            for method in (
                "clone_for_update",
                "replace_rebuilt_tiles",
                "mark_rebuild_dirty",
            )
        )
        if not partial_capable:
            index_ready = False
            full_reason = "map_manager_has_no_partial_rebuild"

        changed_ids: tuple[int, ...] = ()
        if index_ready:
            assert contribution_index is not None
            changed_ids = contribution_index.changed_keyframes(
                keyframe_ids=committed_ids,
                current_poses=keyframe_poses_cv,
            )

        def add_pending(
            candidate: LayeredSemanticGridMap,
            candidate_index: Optional[MapContributionIndex],
            tile_keys_by_keyframe: Optional[dict[int, tuple]],
        ) -> dict:
            if not pending:
                return {
                    "accepted_points": 0,
                    "visible_points": 0,
                    "fused_keyframes": 0,
                }
            statistics = self.fuse_observations(
                grid=candidate,
                observations=pending,
                keyframe_poses_cv=keyframe_poses_cv,
                rotation_map_world=rotation_map_world,
                map_translation_offset=map_translation_offset,
                tile_keys_by_keyframe=tile_keys_by_keyframe,
            )
            if candidate_index is not None:
                assert tile_keys_by_keyframe is not None
                candidate_index.record_many(
                    poses_world_camera={
                        keyframe_id: keyframe_poses_cv[keyframe_id]
                        for keyframe_id in pending_ids
                    },
                    tile_keys_by_keyframe=tile_keys_by_keyframe,
                )
            return statistics

        def finish_statistics(
            candidate: LayeredSemanticGridMap,
            statistics: dict,
            *,
            strategy: str,
            reason: str,
            affected_tile_count: int,
            replayed_keyframes: int,
            footprint_keyframes: int,
            estimated_replay_points: int,
        ) -> dict:
            result = dict(statistics)
            result.update(
                {
                    "rebuild_strategy": strategy,
                    "rebuild_reason": reason,
                    "changed_keyframes": int(len(changed_ids)),
                    "affected_tiles": int(affected_tile_count),
                    "replayed_keyframes": int(replayed_keyframes),
                    "footprint_keyframes": int(footprint_keyframes),
                    "fusion_operations": int(
                        replayed_keyframes + footprint_keyframes
                    ),
                    "total_keyframes": int(
                        len(committed_ids) + len(pending)
                    ),
                    "saved_fusion_operations": int(
                        max(
                            0,
                            len(committed_ids)
                            - replayed_keyframes
                            - footprint_keyframes
                            + len(pending),
                        )
                    ),
                    "estimated_replay_points": int(
                        estimated_replay_points
                    ),
                    "saved_estimated_replay_points": int(
                        max(
                            0,
                            full_replay_cost
                            - estimated_replay_points,
                        )
                    ),
                }
            )
            total_accepted = getattr(
                candidate, "total_accepted_points", None
            )
            if total_accepted is not None:
                result["accepted_points"] = int(total_accepted)
            return result

        def full_candidate(reason: str):
            tile_keys = {} if contribution_index is not None else None
            candidate, statistics = self.rebuild(
                grid_factory=grid_factory,
                keyframe_poses_cv=keyframe_poses_cv,
                rotation_map_world=rotation_map_world,
                map_translation_offset=map_translation_offset,
                tile_keys_by_keyframe=tile_keys,
                additional_observations=pending,
            )
            candidate_index = (
                MapContributionIndex()
                if contribution_index is not None
                else None
            )
            if candidate_index is not None:
                assert tile_keys is not None
                candidate_index.record_many(
                    poses_world_camera={
                        keyframe_id: keyframe_poses_cv[keyframe_id]
                        for keyframe_id in committed_ids + pending_ids
                    },
                    tile_keys_by_keyframe=tile_keys,
                )
            mark_rebuild_dirty = getattr(
                candidate, "mark_rebuild_dirty", None
            )
            if callable(mark_rebuild_dirty):
                mark_rebuild_dirty(live_grid)
            if candidate_index is not None:
                candidate_index.assert_consistent()
            affected = len(
                set(getattr(live_grid, "tiles", ()))
                | set(getattr(candidate, "tiles", ()))
            )
            return (
                candidate,
                finish_statistics(
                    candidate,
                    statistics,
                    strategy="full",
                    reason=reason,
                    affected_tile_count=affected,
                    replayed_keyframes=len(committed_ids) + len(pending),
                    footprint_keyframes=0,
                    estimated_replay_points=full_replay_cost,
                ),
                candidate_index,
            )

        if not index_ready:
            return full_candidate(full_reason)

        assert contribution_index is not None
        if pending and not self.observations_follow_archive(pending):
            return full_candidate("historical_pending_observation")
        if not changed_ids:
            candidate = live_grid.clone_for_update()
            candidate_index = contribution_index.clone()
            pending_tile_keys: dict[int, tuple] = {}
            statistics = add_pending(
                candidate,
                candidate_index,
                pending_tile_keys,
            )
            candidate_index.assert_consistent()
            return (
                candidate,
                finish_statistics(
                    candidate,
                    statistics,
                    strategy="geometry_noop",
                    reason="no_applied_keyframe_pose_changed",
                    affected_tile_count=0,
                    replayed_keyframes=len(pending),
                    footprint_keyframes=0,
                    estimated_replay_points=0,
                ),
                candidate_index,
            )

        changed_replay_cost = replay_cost(changed_ids)
        if 2 * changed_replay_cost >= full_replay_cost:
            return full_candidate("changed_observation_cost")

        changed_tile_keys: dict[int, tuple] = {}
        self.replay_keyframes(
            grid_factory=grid_factory,
            keyframe_ids=changed_ids,
            keyframe_poses_cv=keyframe_poses_cv,
            rotation_map_world=rotation_map_world,
            map_translation_offset=map_translation_offset,
            tile_keys_by_keyframe=changed_tile_keys,
        )
        affected_tiles = set()
        for keyframe_id in changed_ids:
            affected_tiles.update(
                contribution_index.tile_keys(keyframe_id)
            )
            affected_tiles.update(changed_tile_keys[keyframe_id])

        contributor_ids = set(
            contribution_index.contributors(affected_tiles)
        )
        contributor_ids.update(changed_ids)
        ordered_contributors = tuple(sorted(contributor_ids))
        partial_replay_cost = changed_replay_cost + replay_cost(
            ordered_contributors
        )
        if partial_replay_cost >= full_replay_cost:
            return full_candidate("partial_replay_not_cheaper")

        rebuilt_tile_keys: dict[int, tuple] = {}
        rebuilt, statistics = self.replay_keyframes(
            grid_factory=grid_factory,
            keyframe_ids=ordered_contributors,
            keyframe_poses_cv=keyframe_poses_cv,
            rotation_map_world=rotation_map_world,
            map_translation_offset=map_translation_offset,
            tile_keys_by_keyframe=rebuilt_tile_keys,
        )
        for keyframe_id in ordered_contributors:
            expected = (
                frozenset(changed_tile_keys[keyframe_id])
                if keyframe_id in changed_tile_keys
                else contribution_index.tile_keys(keyframe_id)
            )
            if frozenset(rebuilt_tile_keys[keyframe_id]) != expected:
                return full_candidate("dependency_footprint_changed")

        candidate = live_grid.clone_for_update()
        candidate.replace_rebuilt_tiles(
            rebuilt=rebuilt,
            tile_keys=affected_tiles,
        )
        candidate.mark_rebuild_dirty(
            live_grid,
            tile_keys=affected_tiles,
        )

        candidate_index = contribution_index.clone()
        candidate_index.record_many(
            poses_world_camera={
                keyframe_id: keyframe_poses_cv[keyframe_id]
                for keyframe_id in ordered_contributors
            },
            tile_keys_by_keyframe=rebuilt_tile_keys,
        )
        pending_tile_keys = {}
        add_pending(candidate, candidate_index, pending_tile_keys)
        candidate_index.assert_consistent()

        return (
            candidate,
            finish_statistics(
                candidate,
                statistics,
                strategy="partial",
                reason="affected_tile_dependency_replay",
                affected_tile_count=len(affected_tiles),
                replayed_keyframes=(
                    len(ordered_contributors) + len(pending)
                ),
                footprint_keyframes=len(changed_ids),
                estimated_replay_points=partial_replay_cost,
            ),
            candidate_index,
        )
