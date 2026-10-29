from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Optional, Sequence
from uuid import uuid4


_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _integer(
    value: int,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be in [{minimum}, {maximum}], got {result}"
        )
    return result


@dataclass(frozen=True)
class TileTransferPlan:
    """Metadata for one complete authoritative tile payload."""

    tile_x: int
    tile_y: int
    tile_revision: int
    cleared: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tile_x",
            _integer(
                self.tile_x,
                name="tile_x",
                minimum=_INT32_MIN,
                maximum=_INT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "tile_y",
            _integer(
                self.tile_y,
                name="tile_y",
                minimum=_INT32_MIN,
                maximum=_INT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "tile_revision",
            _integer(
                self.tile_revision,
                name="tile_revision",
                minimum=0,
                maximum=_UINT64_MAX,
            ),
        )
        if not isinstance(self.cleared, bool):
            raise ValueError("cleared must be bool")

    @property
    def key(self) -> tuple[int, int]:
        return self.tile_x, self.tile_y


@dataclass(frozen=True)
class MapTransferBatchPlan:
    """One delta or frozen full-map transfer before payload publication."""

    map_id: str
    session_id: str
    map_revision: int
    batch_id: int
    reset: bool
    full_snapshot: bool
    tiles: tuple[TileTransferPlan, ...]

    def __post_init__(self) -> None:
        map_id = str(self.map_id).strip()
        session_id = str(self.session_id).strip()
        if not map_id:
            raise ValueError("map_id must not be empty")
        if not session_id:
            raise ValueError("session_id must not be empty")
        object.__setattr__(self, "map_id", map_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(
            self,
            "map_revision",
            _integer(
                self.map_revision,
                name="map_revision",
                minimum=0,
                maximum=_UINT64_MAX,
            ),
        )
        object.__setattr__(
            self,
            "batch_id",
            _integer(
                self.batch_id,
                name="batch_id",
                minimum=1,
                maximum=_UINT64_MAX,
            ),
        )
        if not isinstance(self.reset, bool):
            raise ValueError("reset must be bool")
        if not isinstance(self.full_snapshot, bool):
            raise ValueError("full_snapshot must be bool")
        tiles = tuple(self.tiles)
        if any(not isinstance(tile, TileTransferPlan) for tile in tiles):
            raise ValueError("tiles must contain TileTransferPlan values")
        keys = [tile.key for tile in tiles]
        if len(keys) != len(set(keys)):
            raise ValueError("one map-transfer batch cannot repeat a tile")
        if not tiles and not self.full_snapshot:
            raise ValueError("an empty delta batch is not meaningful")
        object.__setattr__(self, "tiles", tiles)


@dataclass(frozen=True)
class CompletedMapTransferBatch:
    """A batch whose legacy payload update IDs are fully known."""

    plan: MapTransferBatchPlan
    payload_update_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MapTransferBatchPlan):
            raise ValueError("plan must be a MapTransferBatchPlan")
        values = tuple(
            _integer(
                value,
                name="payload_update_id",
                minimum=0,
                maximum=_UINT32_MAX,
            )
            for value in self.payload_update_ids
        )
        if len(values) != len(self.plan.tiles):
            raise ValueError(
                "payload update IDs must match the planned tile count"
            )
        if len(values) != len(set(values)):
            raise ValueError("payload update IDs must be unique in one batch")
        object.__setattr__(self, "payload_update_ids", values)


class MapTransferSession:
    """Allocate unambiguous full-tile batches without owning ROS state."""

    def __init__(
        self,
        *,
        map_id: str,
        session_id: Optional[str] = None,
    ) -> None:
        normalized_map_id = str(map_id).strip()
        normalized_session_id = str(
            uuid4().hex if session_id is None else session_id
        ).strip()
        if not normalized_map_id:
            raise ValueError("map_id must not be empty")
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")
        self.map_id = normalized_map_id
        self.session_id = normalized_session_id
        self._next_batch_id = 1
        self._reset_pending = True

    @property
    def reset_pending(self) -> bool:
        return self._reset_pending

    def begin_batch(
        self,
        *,
        map_revision: int,
        full_snapshot: bool,
        tiles: Iterable[TileTransferPlan],
    ) -> MapTransferBatchPlan:
        if not isinstance(full_snapshot, bool):
            raise ValueError("full_snapshot must be bool")
        batch_id = self._next_batch_id
        if batch_id > _UINT64_MAX:
            raise OverflowError("semantic-map batch_id exhausted uint64")
        self._next_batch_id += 1
        return MapTransferBatchPlan(
            map_id=self.map_id,
            session_id=self.session_id,
            map_revision=map_revision,
            batch_id=batch_id,
            reset=self._reset_pending,
            full_snapshot=full_snapshot,
            tiles=tuple(tiles),
        )

    def complete_batch(
        self,
        plan: MapTransferBatchPlan,
        payload_update_ids: Sequence[int],
    ) -> CompletedMapTransferBatch:
        self._validate_ownership(plan)
        return CompletedMapTransferBatch(
            plan=plan,
            payload_update_ids=tuple(payload_update_ids),
        )

    def mark_complete(self, batch: CompletedMapTransferBatch) -> None:
        if not isinstance(batch, CompletedMapTransferBatch):
            raise ValueError("batch must be a CompletedMapTransferBatch")
        self._validate_ownership(batch.plan)
        self._reset_pending = False

    def _validate_ownership(self, plan: MapTransferBatchPlan) -> None:
        if not isinstance(plan, MapTransferBatchPlan):
            raise ValueError("plan must be a MapTransferBatchPlan")
        if plan.map_id != self.map_id or plan.session_id != self.session_id:
            raise ValueError("map-transfer batch belongs to another session")
