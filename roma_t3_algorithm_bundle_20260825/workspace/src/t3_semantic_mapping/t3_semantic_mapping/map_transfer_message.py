from __future__ import annotations

from std_msgs.msg import Header
from t3_interfaces.msg import SemanticMapBatch

from .map_transfer_protocol import (
    CompletedMapTransferBatch,
    MapTransferBatchPlan,
)


def _base_message(
    *,
    header: Header,
    plan: MapTransferBatchPlan,
) -> SemanticMapBatch:
    message = SemanticMapBatch()
    message.header = header
    message.map_id = plan.map_id
    message.session_id = plan.session_id
    message.map_revision = plan.map_revision
    message.batch_id = plan.batch_id
    message.reset = plan.reset
    message.full_snapshot = plan.full_snapshot
    message.tile_count = len(plan.tiles)
    message.tile_x = [tile.tile_x for tile in plan.tiles]
    message.tile_y = [tile.tile_y for tile in plan.tiles]
    message.tile_revision = [tile.tile_revision for tile in plan.tiles]
    message.tile_cleared = [tile.cleared for tile in plan.tiles]
    return message


def make_map_batch_begin(
    *,
    header: Header,
    plan: MapTransferBatchPlan,
) -> SemanticMapBatch:
    message = _base_message(header=header, plan=plan)
    message.phase = SemanticMapBatch.PHASE_BEGIN
    message.payload_update_id = []
    return message


def make_map_batch_complete(
    *,
    header: Header,
    batch: CompletedMapTransferBatch,
) -> SemanticMapBatch:
    message = _base_message(header=header, plan=batch.plan)
    message.phase = SemanticMapBatch.PHASE_COMPLETE
    message.payload_update_id = list(batch.payload_update_ids)
    return message
