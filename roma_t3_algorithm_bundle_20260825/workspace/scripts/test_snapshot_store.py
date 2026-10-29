#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing
import os
import signal
import tempfile
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src" / "t3_semantic_mapping"
sys.path.insert(0, str(SOURCE))

from t3_semantic_mapping import snapshot_store
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


def _manager(point_x: float, *, height: float = 1.0) -> TiledSemanticMapManager:
    manager = TiledSemanticMapManager(
        resolution=1.0,
        tile_cells=4,
        max_rays_per_update=20,
    )
    manager.update(
        points_map=np.array([[point_x, 0.2, height]], dtype=np.float64),
        semantic_labels=np.array([4], dtype=np.uint8),
        camera_position_map=np.array([point_x, 0.2, 0.0]),
    )
    return manager


def _save(
    manager: TiledSemanticMapManager,
    directory: Path,
    sequence: int,
):
    return manager.save_snapshot(
        directory,
        timestamp_text=f"{sequence}.000000000",
        frame_id="map",
    )


def _version_names(directory: Path):
    versions = directory / snapshot_store.VERSIONS_DIRECTORY_NAME
    if not versions.is_dir():
        return set()
    return {
        path.name
        for path in versions.iterdir()
        if path.name.startswith(snapshot_store.VERSION_PREFIX)
    }


def _assert_no_atomic_work(test: unittest.TestCase, directory: Path) -> None:
    versions = directory / snapshot_store.VERSIONS_DIRECTORY_NAME
    if versions.is_dir():
        test.assertFalse(
            any(
                path.name.startswith(snapshot_store.STAGING_PREFIX)
                for path in versions.iterdir()
            )
        )
    test.assertFalse(
        any(
            path.name.startswith(snapshot_store.INDEX_TEMP_PREFIX)
            for path in directory.iterdir()
        )
    )


def _assert_manager_equal(
    test: unittest.TestCase,
    actual: TiledSemanticMapManager,
    expected: TiledSemanticMapManager,
) -> None:
    test.assertEqual(set(actual.tiles), set(expected.tiles))
    test.assertEqual(actual.tile_revisions, expected.tile_revisions)
    test.assertEqual(actual.map_revision, expected.map_revision)
    test.assertEqual(actual.total_accepted_points, expected.total_accepted_points)
    test.assertEqual(actual.replay_tile_keys(), expected.replay_tile_keys())
    for key, expected_tile in expected.tiles.items():
        actual_tile = actual.tiles[key]
        for name, _ in expected._SNAPSHOT_TILE_FIELDS:
            np.testing.assert_array_equal(
                getattr(actual_tile, name),
                getattr(expected_tile, name),
            )


def _sigint_save_worker(directory_text: str, ready) -> None:
    manager = _manager(8.2)

    def interrupted_write(path, **_arrays):
        Path(path).write_bytes(b"partial-npz")
        ready.set()
        time.sleep(30.0)

    try:
        with mock.patch.object(
            np,
            "savez_compressed",
            side_effect=interrupted_write,
        ):
            _save(manager, Path(directory_text), 2)
    except KeyboardInterrupt:
        return
    raise RuntimeError("SIGINT test worker completed without interruption")


class AtomicSnapshotStoreTest(unittest.TestCase):
    def test_committed_root_manifest_loads_complete_immutable_version(self):
        manager = _manager(-0.2, height=2.0)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = _save(manager, directory, 1)

            self.assertEqual(paths["index"], directory / "map_index.yaml")
            metadata = yaml.safe_load(
                paths["index"].read_text(encoding="utf-8")
            )
            version = metadata["snapshot_version"]
            self.assertEqual(
                metadata["snapshot_storage"],
                snapshot_store.ATOMIC_SNAPSHOT_STORAGE,
            )
            self.assertEqual(
                metadata["snapshot_storage_version"],
                snapshot_store.ATOMIC_SNAPSHOT_STORAGE_VERSION,
            )
            self.assertEqual(_version_names(directory), {version})
            self.assertEqual(
                paths["tiles"],
                directory
                / snapshot_store.VERSIONS_DIRECTORY_NAME
                / version
                / "tiles",
            )
            for entry in metadata["tiles"]:
                relative = Path(entry["file"])
                self.assertEqual(relative.parts[0], ".versions")
                self.assertEqual(relative.parts[1], version)
                self.assertTrue((directory / relative).is_file())

            internal_index = (
                directory
                / snapshot_store.VERSIONS_DIRECTORY_NAME
                / version
                / "map_index.yaml"
            )
            internal = yaml.safe_load(internal_index.read_text(encoding="utf-8"))
            self.assertTrue(
                all(
                    Path(entry["file"]).parts[0] == "tiles"
                    for entry in internal["tiles"]
                )
            )
            restored = TiledSemanticMapManager.load_snapshot(paths["index"])
            _assert_manager_equal(self, restored, manager)
            _assert_no_atomic_work(self, directory)

    def test_repeated_save_retains_current_and_previous_without_stale_tiles(self):
        first = _manager(0.2)
        second = _manager(8.2)
        third = _manager(-8.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _save(first, directory, 1)
            _save(second, directory, 2)
            paths = _save(third, directory, 3)

            metadata = yaml.safe_load(
                paths["index"].read_text(encoding="utf-8")
            )
            self.assertEqual(len(_version_names(directory)), 2)
            self.assertIn(metadata["snapshot_version"], _version_names(directory))
            restored = TiledSemanticMapManager.load_snapshot(paths["index"])
            _assert_manager_equal(self, restored, third)
            self.assertNotEqual(set(restored.tiles), set(first.tiles))
            self.assertNotEqual(set(restored.tiles), set(second.tiles))
            _assert_no_atomic_work(self, directory)

    def test_write_error_leaves_previous_manifest_and_version_unchanged(self):
        committed = _manager(0.2)
        replacement = _manager(8.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_paths = _save(committed, directory, 1)
            old_index = first_paths["index"].read_bytes()
            old_versions = _version_names(directory)

            with mock.patch.object(
                np,
                "savez_compressed",
                side_effect=OSError("injected tile write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    _save(replacement, directory, 2)

            self.assertEqual(first_paths["index"].read_bytes(), old_index)
            self.assertEqual(_version_names(directory), old_versions)
            restored = TiledSemanticMapManager.load_snapshot(first_paths["index"])
            _assert_manager_equal(self, restored, committed)
            _assert_no_atomic_work(self, directory)

    def test_keyboard_interrupt_cleans_partial_staging_and_keeps_old_map(self):
        committed = _manager(0.2)
        replacement = _manager(8.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_paths = _save(committed, directory, 1)
            old_index = first_paths["index"].read_bytes()
            old_versions = _version_names(directory)

            def interrupted(path, **_arrays):
                Path(path).write_bytes(b"partial-npz")
                raise KeyboardInterrupt("injected interrupt")

            with mock.patch.object(
                np,
                "savez_compressed",
                side_effect=interrupted,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    _save(replacement, directory, 2)

            self.assertEqual(first_paths["index"].read_bytes(), old_index)
            self.assertEqual(_version_names(directory), old_versions)
            restored = TiledSemanticMapManager.load_snapshot(first_paths["index"])
            _assert_manager_equal(self, restored, committed)
            _assert_no_atomic_work(self, directory)

    def test_real_sigint_keeps_previous_committed_snapshot(self):
        committed = _manager(0.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_paths = _save(committed, directory, 1)
            old_index = first_paths["index"].read_bytes()
            old_versions = _version_names(directory)
            context = multiprocessing.get_context("fork")
            ready = context.Event()
            process = context.Process(
                target=_sigint_save_worker,
                args=(str(directory), ready),
            )
            process.start()
            try:
                self.assertTrue(ready.wait(timeout=5.0))
                os.kill(process.pid, signal.SIGINT)
                process.join(timeout=10.0)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)

            self.assertEqual(first_paths["index"].read_bytes(), old_index)
            self.assertEqual(_version_names(directory), old_versions)
            _assert_manager_equal(
                self,
                TiledSemanticMapManager.load_snapshot(first_paths["index"]),
                committed,
            )
            _assert_no_atomic_work(self, directory)

    def test_manifest_replace_failure_removes_uncommitted_final_version(self):
        committed = _manager(0.2)
        replacement = _manager(8.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _save(_manager(-8.2), directory, 0)
            first_paths = _save(committed, directory, 1)
            old_index = first_paths["index"].read_bytes()
            old_versions = _version_names(directory)
            self.assertEqual(len(old_versions), 2)
            real_replace = snapshot_store.os.replace

            def fail_root_replace(source, destination):
                if Path(destination) == directory / "map_index.yaml":
                    raise OSError("injected manifest replace failure")
                return real_replace(source, destination)

            with mock.patch.object(
                snapshot_store.os,
                "replace",
                side_effect=fail_root_replace,
            ):
                with self.assertRaisesRegex(OSError, "manifest replace"):
                    _save(replacement, directory, 2)

            self.assertEqual(first_paths["index"].read_bytes(), old_index)
            self.assertEqual(_version_names(directory), old_versions)
            restored = TiledSemanticMapManager.load_snapshot(first_paths["index"])
            _assert_manager_equal(self, restored, committed)
            _assert_no_atomic_work(self, directory)

    def test_post_commit_cleanup_failure_keeps_new_map_and_retries_later(self):
        first = _manager(-8.2)
        second = _manager(0.2)
        third = _manager(8.2)
        fourth = _manager(12.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _save(first, directory, 1)
            _save(second, directory, 2)

            with mock.patch.object(
                snapshot_store,
                "_prepare_retention",
                side_effect=OSError("injected cleanup failure"),
            ):
                with self.assertWarnsRegex(RuntimeWarning, "cleanup"):
                    paths = _save(third, directory, 3)

            _assert_manager_equal(
                self,
                TiledSemanticMapManager.load_snapshot(paths["index"]),
                third,
            )
            self.assertEqual(len(_version_names(directory)), 3)

            paths = _save(fourth, directory, 4)
            _assert_manager_equal(
                self,
                TiledSemanticMapManager.load_snapshot(paths["index"]),
                fourth,
            )
            self.assertEqual(len(_version_names(directory)), 2)
            _assert_no_atomic_work(self, directory)

    def test_validation_failure_never_changes_committed_snapshot(self):
        committed = _manager(0.2)
        replacement = _manager(8.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_paths = _save(committed, directory, 1)
            old_index = first_paths["index"].read_bytes()
            old_versions = _version_names(directory)

            with mock.patch.object(
                replacement,
                "_validate_snapshot_directory",
                side_effect=ValueError("injected validation failure"),
            ):
                with self.assertRaisesRegex(ValueError, "validation failure"):
                    _save(replacement, directory, 2)

            self.assertEqual(first_paths["index"].read_bytes(), old_index)
            self.assertEqual(_version_names(directory), old_versions)
            _assert_no_atomic_work(self, directory)

    def test_corrupted_tile_payload_is_rejected_before_commit(self):
        committed = _manager(0.2)
        replacement = _manager(8.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_paths = _save(committed, directory, 1)
            old_index = first_paths["index"].read_bytes()
            old_versions = _version_names(directory)
            real_savez = np.savez_compressed

            def write_corrupted(path, **arrays):
                changed = dict(arrays)
                changed["elevation_mean"] = arrays["elevation_mean"].copy()
                changed["elevation_mean"].flat[0] += 1.0
                return real_savez(path, **changed)

            with mock.patch.object(
                np,
                "savez_compressed",
                side_effect=write_corrupted,
            ):
                with self.assertRaisesRegex(ValueError, "differs"):
                    _save(replacement, directory, 2)

            self.assertEqual(first_paths["index"].read_bytes(), old_index)
            self.assertEqual(_version_names(directory), old_versions)
            _assert_manager_equal(
                self,
                TiledSemanticMapManager.load_snapshot(first_paths["index"]),
                committed,
            )
            _assert_no_atomic_work(self, directory)

    def test_legacy_snapshot_migrates_without_invalidating_open_reader(self):
        manager = _manager(0.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            legacy = manager._write_snapshot_directory(
                directory,
                timestamp_text="0.000000000",
                frame_id="map",
            )
            legacy_index_text = legacy["index"].read_text(encoding="utf-8")
            legacy_files = tuple((directory / "tiles").glob("*.npz"))
            _assert_manager_equal(
                self,
                TiledSemanticMapManager.load_snapshot(legacy["index"]),
                manager,
            )

            _save(manager, directory, 1)
            self.assertTrue((directory / "tiles").is_dir())
            self.assertTrue(all(path.is_file() for path in legacy_files))
            # A reader that already parsed the legacy manifest can still use
            # its retained flat payloads during the first migration cycle.
            parsed_legacy = yaml.safe_load(legacy_index_text)
            self.assertTrue(
                all(
                    (directory / entry["file"]).is_file()
                    for entry in parsed_legacy["tiles"]
                )
            )

            _save(manager, directory, 2)
            self.assertFalse((directory / "tiles").exists())
            self.assertEqual(len(_version_names(directory)), 2)
            _assert_manager_equal(
                self,
                TiledSemanticMapManager.load_snapshot(
                    directory / "map_index.yaml"
                ),
                manager,
            )

    def test_only_owned_abandoned_work_is_cleaned(self):
        manager = _manager(0.2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            versions = directory / snapshot_store.VERSIONS_DIRECTORY_NAME
            versions.mkdir(parents=True)
            abandoned = versions / f"{snapshot_store.STAGING_PREFIX}old"
            abandoned.mkdir()
            (abandoned / "partial").write_text("x", encoding="utf-8")
            temporary_index = (
                directory / f"{snapshot_store.INDEX_TEMP_PREFIX}old.tmp"
            )
            temporary_index.write_text("partial", encoding="utf-8")
            user_directory = versions / "notes"
            user_directory.mkdir()
            user_file = directory / "user.tmp"
            user_file.write_text("keep", encoding="utf-8")

            _save(manager, directory, 1)

            self.assertFalse(abandoned.exists())
            self.assertFalse(temporary_index.exists())
            self.assertTrue(user_directory.is_dir())
            self.assertEqual(user_file.read_text(encoding="utf-8"), "keep")
            _assert_no_atomic_work(self, directory)


if __name__ == "__main__":
    unittest.main()
