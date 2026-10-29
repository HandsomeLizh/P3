#!/usr/bin/env python3
"""Validate one committed full_tiles batch without retaining map arrays."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from t3_interfaces.msg import IncrementalSemanticMap, SemanticMapBatch


@dataclass(frozen=True)
class PayloadSummary:
    update_id: int
    stamp: tuple[int, int]
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    all_unknown: bool


def _stamp(message) -> tuple[int, int]:
    return int(message.header.stamp.sec), int(message.header.stamp.nanosec)


def _batch_signature(message: SemanticMapBatch) -> tuple:
    return (
        message.map_id,
        message.session_id,
        int(message.map_revision),
        int(message.batch_id),
        bool(message.reset),
        bool(message.full_snapshot),
        int(message.tile_count),
        tuple(message.tile_x),
        tuple(message.tile_y),
        tuple(message.tile_revision),
        tuple(message.tile_cleared),
        _stamp(message),
        message.header.frame_id,
    )


class TransferChecker(Node):
    def __init__(self, *, payload_topic: str, batch_topic: str,
                 require_full_snapshot: bool) -> None:
        super().__init__("t3_full_tile_transfer_checker")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.require_full_snapshot = bool(require_full_snapshot)
        self.payloads: dict[tuple[tuple[int, int], int], PayloadSummary] = {}
        self.begins: dict[tuple[str, int], SemanticMapBatch] = {}
        self.completes: dict[tuple[str, int], SemanticMapBatch] = {}
        self.result = None
        self.error = None
        self.create_subscription(
            IncrementalSemanticMap,
            payload_topic,
            self._on_payload,
            qos,
        )
        self.create_subscription(
            SemanticMapBatch,
            batch_topic,
            self._on_batch,
            qos,
        )

    def _on_payload(self, message: IncrementalSemanticMap) -> None:
        try:
            cell_count = int(message.width) * int(message.height)
            arrays = (
                message.occupancy,
                message.semantic,
                message.semantic_confidence,
                message.elevation,
                message.elevation_variance,
                message.height_range,
                message.roughness,
                message.observation_count,
            )
            if cell_count <= 0 or any(len(values) != cell_count for values in arrays):
                raise ValueError(
                    f"payload {message.update_id} has inconsistent layer lengths"
                )
            all_unknown = (
                all(value == -1 for value in message.occupancy)
                and all(value == 0 for value in message.semantic)
                and all(value == 0 for value in message.observation_count)
                and all(math.isnan(value) for value in message.elevation)
            )
            key = (_stamp(message), int(message.update_id))
            self.payloads[key] = PayloadSummary(
                update_id=int(message.update_id),
                stamp=key[0],
                width=int(message.width),
                height=int(message.height),
                resolution=float(message.resolution),
                origin_x=float(message.origin.position.x),
                origin_y=float(message.origin.position.y),
                all_unknown=all_unknown,
            )
            self._try_finish()
        except Exception as exception:
            self.error = str(exception)

    def _on_batch(self, message: SemanticMapBatch) -> None:
        try:
            count = int(message.tile_count)
            tile_arrays = (
                message.tile_x,
                message.tile_y,
                message.tile_revision,
                message.tile_cleared,
            )
            if any(len(values) != count for values in tile_arrays):
                raise ValueError("batch tile metadata lengths are inconsistent")
            key = (message.session_id, int(message.batch_id))
            if message.phase == SemanticMapBatch.PHASE_BEGIN:
                if message.payload_update_id:
                    raise ValueError("BEGIN must not contain payload update IDs")
                self.begins[key] = message
            elif message.phase == SemanticMapBatch.PHASE_COMPLETE:
                if len(message.payload_update_id) != count:
                    raise ValueError(
                        "COMPLETE payload IDs do not match tile_count"
                    )
                if len(set(message.payload_update_id)) != count:
                    raise ValueError("COMPLETE payload IDs are not unique")
                self.completes[key] = message
            else:
                raise ValueError(f"unknown batch phase {message.phase}")
            self._try_finish()
        except Exception as exception:
            self.error = str(exception)

    def _try_finish(self) -> None:
        if self.result is not None or self.error is not None:
            return
        for key, complete in tuple(self.completes.items()):
            if self.require_full_snapshot and not complete.full_snapshot:
                continue
            begin = self.begins.get(key)
            if begin is None:
                continue
            if _batch_signature(begin) != _batch_signature(complete):
                self.error = "BEGIN and COMPLETE metadata differ"
                return
            stamp = _stamp(complete)
            payloads = [
                self.payloads.get((stamp, int(update_id)))
                for update_id in complete.payload_update_id
            ]
            if any(payload is None for payload in payloads):
                continue
            cleared = list(complete.tile_cleared)
            for is_cleared, payload in zip(cleared, payloads):
                if is_cleared and not payload.all_unknown:
                    self.error = "cleared tile payload is not all unknown"
                    return
            self.result = {
                "map_id": complete.map_id,
                "session_id": complete.session_id,
                "map_revision": int(complete.map_revision),
                "batch_id": int(complete.batch_id),
                "reset": bool(complete.reset),
                "full_snapshot": bool(complete.full_snapshot),
                "tile_count": int(complete.tile_count),
                "cleared_tile_count": sum(cleared),
                "payload_update_ids": list(complete.payload_update_id),
            }
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload-topic",
        default="/Car/T3/semantic/incremental_map",
    )
    parser.add_argument(
        "--batch-topic",
        default="/Car/T3/semantic/map_batch",
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--require-full-snapshot", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = TransferChecker(
        payload_topic=args.payload_topic,
        batch_topic=args.batch_topic,
        require_full_snapshot=args.require_full_snapshot,
    )
    deadline = time.monotonic() + max(0.1, args.timeout_sec)
    try:
        while (
            rclpy.ok()
            and node.result is None
            and node.error is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.error is not None:
            raise RuntimeError(node.error)
        if node.result is None:
            raise TimeoutError("no complete matching map batch before timeout")
        print(json.dumps(node.result, ensure_ascii=False, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
