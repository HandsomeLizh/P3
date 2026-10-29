from __future__ import annotations

import ast
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
MAPPER = (
    WORKSPACE
    / "src"
    / "t3_semantic_mapping"
    / "t3_semantic_mapping"
    / "roma_semantic_mapper_node.py"
)
OLD_MESSAGE = (
    WORKSPACE
    / "src"
    / "t3_interfaces"
    / "msg"
    / "IncrementalSemanticMap.msg"
)
BACKUP_OLD_MESSAGE = (
    WORKSPACE.parents[1]
    / "backup"
    / "t3_gridmap_v2_ws"
    / "src"
    / "t3_interfaces"
    / "msg"
    / "IncrementalSemanticMap.msg"
)
NEW_MESSAGE = OLD_MESSAGE.with_name("SemanticMapBatch.msg")
CMAKE = OLD_MESSAGE.parents[1] / "CMakeLists.txt"
LAUNCH_OPTIONS = MAPPER.with_name("launch_options.py")


def _method_source(source: str, tree: ast.AST, name: str) -> str:
    class_node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "RomaSemanticMapperNode"
    )
    method = next(
        item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, method) or ""


class MapTransferContractStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAPPER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_legacy_incremental_payload_is_byte_identical(self):
        self.assertEqual(
            OLD_MESSAGE.read_bytes(),
            BACKUP_OLD_MESSAGE.read_bytes(),
        )

    def test_batch_schema_carries_every_completeness_key(self):
        schema = NEW_MESSAGE.read_text(encoding="utf-8")
        for declaration in (
            "string map_id",
            "string session_id",
            "uint64 map_revision",
            "uint64 batch_id",
            "uint8 phase",
            "bool reset",
            "bool full_snapshot",
            "int32[] tile_x",
            "int32[] tile_y",
            "uint64[] tile_revision",
            "bool[] tile_cleared",
            "uint32[] payload_update_id",
        ):
            self.assertIn(declaration, schema)
        self.assertIn('"msg/SemanticMapBatch.msg"', CMAKE.read_text())

    def test_streamed_delta_has_begin_payload_complete_commit_order(self):
        class_node = next(
            item
            for item in self.tree.body
            if isinstance(item, ast.ClassDef)
            and item.name == "RomaSemanticMapperNode"
        )
        method_names = {
            item.name
            for item in class_node.body
            if isinstance(item, ast.FunctionDef)
        }
        self.assertNotIn("_publish_incremental_tile_updates", method_names)

        start = _method_source(
            self.source,
            self.tree,
            "_start_incremental_delta",
        )
        timer = _method_source(
            self.source,
            self.tree,
            "_on_incremental_replay_timer",
        )
        finish = _method_source(
            self.source,
            self.tree,
            "_finish_incremental_replay",
        )
        self.assertLess(
            start.index("self._publish_map_batch_begin"),
            start.index("self._incremental_replay_state ="),
        )
        payload = timer.index("self._publish_incremental_window_snapshot")
        recorded = timer.index("state.payload_update_ids.append")
        completed = timer.rindex("self._finish_incremental_replay()")
        self.assertLess(payload, recorded)
        self.assertLess(recorded, completed)
        self.assertIn("except Exception", timer[payload:])
        self.assertIn("self._incremental_replay_state = None", timer[payload:])
        self.assertLess(
            finish.index("self.map_transfer_session.complete_batch"),
            finish.index("self._publish_map_batch_complete"),
        )
        self.assertLess(
            finish.index("self._publish_map_batch_complete"),
            finish.index("self.map_transfer_session.mark_complete"),
        )

    def test_late_subscriber_replay_uses_frozen_copy_on_write_snapshot(self):
        start = _method_source(
            self.source,
            self.tree,
            "_start_incremental_replay",
        )
        timer = _method_source(
            self.source,
            self.tree,
            "_on_incremental_replay_timer",
        )
        self.assertIn("snapshot = self.grid.clone_for_update()", start)
        self.assertIn("snapshot.replay_tile_keys()", start)
        self.assertIn("state.snapshot.tile_update", timer)
        self.assertIn("_finish_incremental_replay", timer)

    def test_live_delta_serialization_is_streamed_outside_map_callback(self):
        publish = _method_source(
            self.source,
            self.tree,
            "_publish_current_map_outputs",
        )
        start = _method_source(
            self.source,
            self.tree,
            "_start_incremental_delta",
        )
        finish = _method_source(
            self.source,
            self.tree,
            "_finish_incremental_replay",
        )
        self.assertNotIn("tile_update(", publish)
        self.assertIn("snapshot = self.grid.clone_for_update()", start)
        self.assertIn("snapshot.pending_dirty_tile_keys()", start)
        self.assertIn("full_snapshot=False", start)
        self.assertIn("mark_tiles_published_at_revisions", finish)

    def test_companion_endpoint_is_full_tiles_only_and_replayable(self):
        interfaces = _method_source(
            self.source,
            self.tree,
            "_create_ros_interfaces",
        )
        self.assertIn("SemanticMapBatch", interfaces)
        self.assertIn('== "full_tiles"', interfaces)
        self.assertIn("incremental_map_replay_service", interfaces)
        self.assertIn("_on_incremental_replay_service", interfaces)

    def test_all_launches_share_batch_topic_and_replay_service_options(self):
        shared = LAUNCH_OPTIONS.read_text(encoding="utf-8")
        for name in (
            "incremental_map_batch_topic",
            "incremental_map_replay_service",
        ):
            self.assertEqual(shared.count(f'"{name}"'), 3)
        for name in (
            "mapper_only.launch.py",
            "practice_live_precomputed.launch.py",
            "practice_live_online_semantics.launch.py",
        ):
            launch = (MAPPER.parents[1] / "launch" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn("mapping_launch_arguments()", launch)
            self.assertIn("**mapping_parameter_overrides()", launch)


if __name__ == "__main__":
    unittest.main()
