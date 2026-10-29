"""Crash-safe, lossless keyframe storage with a descriptor-only RAM index."""

from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PROTOCOL_VERSION
from .contracts import KeyframePacket, validated_session_id
from .geometry import normalized_vector


@dataclass(frozen=True)
class StoredKeyframe:
    session_id: str
    node_id: int
    timestamp_ns: int
    salad_descriptor: np.ndarray
    dino_descriptor: np.ndarray
    path: Path

    def __post_init__(self) -> None:
        # The complete index stays resident, but half precision limits it to
        # about 21 KiB/keyframe (SALAD + DINO). Scoring promotes one bounded
        # chunk at a time to float32, so no candidate is discarded.
        salad = normalized_vector(self.salad_descriptor).astype(
            np.float16, copy=True
        )
        dino = normalized_vector(self.dino_descriptor).astype(
            np.float16, copy=True
        )
        salad.setflags(write=False)
        dino.setflags(write=False)
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "salad_descriptor", salad)
        object.__setattr__(self, "dino_descriptor", dino)
        object.__setattr__(self, "path", Path(self.path).resolve())


class SessionKeyframeStore:
    """One acquisition session; payloads grow on disk, descriptors in RAM."""

    def __init__(self, root: Path, session_id: str) -> None:
        self.session_id = validated_session_id(session_id)
        self.directory = Path(root).resolve() / self.session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self._records: list[StoredKeyframe] = []
        self._by_node: dict[int, StoredKeyframe] = {}
        self._load_existing()

    def _record_path(self, node_id: int) -> Path:
        return self.directory / f"keyframe_{int(node_id):020d}.npz"

    def _read_record(self, path: Path) -> StoredKeyframe:
        with np.load(path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["protocol_version"]).reshape(()))
            if version != PROTOCOL_VERSION:
                raise ValueError(
                    f"stored protocol {version} is not supported by {PROTOCOL_VERSION}"
                )
            session = str(np.asarray(payload["session_id"]).reshape(()))
            if session != self.session_id:
                raise ValueError(
                    f"stored session {session!r} does not match {self.session_id!r}"
                )
            return StoredKeyframe(
                session_id=session,
                node_id=int(np.asarray(payload["node_id"]).reshape(())),
                timestamp_ns=int(
                    np.asarray(payload["timestamp_ns"]).reshape(())
                ),
                salad_descriptor=payload["salad_descriptor"],
                dino_descriptor=payload["dino_descriptor"],
                path=path,
            )

    def _load_existing(self) -> None:
        records = []
        for path in sorted(self.directory.glob("keyframe_*.npz")):
            records.append(self._read_record(path))
        records.sort(key=lambda record: record.node_id)
        previous_timestamp = None
        for record in records:
            if record.node_id in self._by_node:
                raise RuntimeError(
                    f"duplicate stored MPRF node {record.node_id}"
                )
            if previous_timestamp is not None and record.timestamp_ns <= previous_timestamp:
                raise RuntimeError("stored MPRF timestamps are not strictly increasing")
            self._records.append(record)
            self._by_node[record.node_id] = record
            previous_timestamp = record.timestamp_ns

    def add(
        self,
        packet: KeyframePacket,
        *,
        salad_descriptor: np.ndarray,
        dino_descriptor: np.ndarray,
        response_payload: dict[str, object] | None = None,
    ) -> StoredKeyframe:
        if packet.session_id != self.session_id:
            raise ValueError("packet belongs to a different MPRF session")
        duplicate = self._by_node.get(packet.node_id)
        if duplicate is not None:
            if duplicate.timestamp_ns != packet.timestamp_ns:
                raise ValueError("duplicate node has a conflicting timestamp")
            return duplicate
        if self._records:
            latest = self._records[-1]
            if packet.node_id <= latest.node_id:
                raise ValueError("MPRF node IDs must be appended in increasing order")
            if packet.timestamp_ns <= latest.timestamp_ns:
                raise ValueError("MPRF timestamps must be strictly increasing")
        salad = normalized_vector(salad_descriptor)
        dino = normalized_vector(dino_descriptor)
        response_json = ""
        if response_payload is not None:
            response_json = json.dumps(
                response_payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        destination = self._record_path(packet.node_id)
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=self.directory,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        try:
            with temporary_file:
                np.savez_compressed(
                    temporary_file,
                    protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int32),
                    session_id=np.asarray(packet.session_id),
                    node_id=np.asarray(packet.node_id, dtype=np.int64),
                    timestamp_ns=np.asarray(packet.timestamp_ns, dtype=np.int64),
                    image_bgr=packet.image_bgr,
                    points_sensor=packet.points_sensor,
                    camera_matrix=packet.camera_matrix,
                    transform_camera_sensor=packet.transform_camera_sensor,
                    calibration_size=np.asarray(
                        [packet.calibration_width, packet.calibration_height],
                        dtype=np.int32,
                    ),
                    # Float16 keeps a long-run RAM/disk descriptor index small;
                    # cosine scoring always promotes it back to float32.
                    salad_descriptor=salad.astype(np.float16),
                    dino_descriptor=dino.astype(np.float16),
                    response_json=np.asarray(response_json),
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
        record = StoredKeyframe(
            session_id=packet.session_id,
            node_id=packet.node_id,
            timestamp_ns=packet.timestamp_ns,
            salad_descriptor=salad,
            dino_descriptor=dino,
            path=destination,
        )
        self._records.append(record)
        self._by_node[record.node_id] = record
        return record

    def load_response(
        self, record: StoredKeyframe
    ) -> dict[str, object] | None:
        with np.load(record.path, allow_pickle=False) as payload:
            if "response_json" not in payload.files:
                return None
            encoded = str(np.asarray(payload["response_json"]).reshape(()))
        if not encoded:
            return None
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise ValueError("stored MPRF response is not a JSON object")
        return value

    def load_packet(self, record: StoredKeyframe) -> KeyframePacket:
        if record.session_id != self.session_id:
            raise ValueError("stored record belongs to a different session")
        with np.load(record.path, allow_pickle=False) as payload:
            calibration = np.asarray(
                payload["calibration_size"], dtype=np.int64
            ).reshape(2)
            return KeyframePacket(
                session_id=str(np.asarray(payload["session_id"]).reshape(())),
                node_id=int(np.asarray(payload["node_id"]).reshape(())),
                timestamp_ns=int(
                    np.asarray(payload["timestamp_ns"]).reshape(())
                ),
                image_bgr=payload["image_bgr"],
                points_sensor=payload["points_sensor"],
                camera_matrix=payload["camera_matrix"],
                transform_camera_sensor=payload["transform_camera_sensor"],
                calibration_width=int(calibration[0]),
                calibration_height=int(calibration[1]),
            )

    def get(self, node_id: int) -> StoredKeyframe | None:
        return self._by_node.get(int(node_id))

    def records(self) -> tuple[StoredKeyframe, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)
