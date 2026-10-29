#!/usr/bin/env python3
"""Pure mapping tests: no ROS node, bag playback, GPU or vehicle commands."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/t3_semantic_mapping"))
from t3_semantic_mapping.elevation_fusion import ElevationFusionConfig
from t3_semantic_mapping.semantic_grid import LayeredSemanticGridMap
from t3_semantic_mapping.tiled_semantic_map import TiledSemanticMapManager


def grid(mode="variance", **settings):
    return LayeredSemanticGridMap(
        resolution=1., length_x=4., length_y=4., origin_x=0., origin_y=0.,
        elevation_fusion=ElevationFusionConfig(fusion_mode=mode, **settings))


def manager(mode="variance", **settings):
    return TiledSemanticMapManager(
        resolution=1., tile_cells=4, max_rays_per_update=0,
        elevation_fusion=ElevationFusionConfig(fusion_mode=mode, **settings))


def lidar(target, z):
    return target.update_elevation_only(points_map=np.array([[.5, .5, z]]))


def stereo(target, z, count=1):
    return target.update(
        points_map=np.tile([.5, .5, z], (count, 1)),
        semantic_labels=np.ones(count, dtype=np.uint8),
        camera_position_map=np.zeros(3))


class ElevationFusionTest(unittest.TestCase):
    def test_settings_round_trip_and_validation(self):
        config = ElevationFusionConfig(fusion_mode="variance")
        self.assertEqual(config, ElevationFusionConfig.from_parameters(config.parameters()))
        self.assertEqual(ElevationFusionConfig.from_parameters({}).fusion_mode, "mean")
        for settings in (
            {"fusion_mode": "unknown"}, {"measurement_variance": 0},
            {"lidar_measurement_variance": -1}, {"min_variance": .05},
            {"gate_sigma": 0}, {"gate_min_m": -1},
            {"process_variance": float("nan")}, {"measurement_variance": float("inf")},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ElevationFusionConfig(**settings)

    def test_mean_mode_keeps_original_sample_statistics(self):
        target = grid("mean")
        for value in [0., .2, .4, 5.]:
            lidar(target, value)
        values = np.array([0., .2, .4, 5.])
        self.assertAlmostEqual(target.elevation_layer()[0, 0], values.mean(), places=6)
        self.assertAlmostEqual(target.elevation_variance_layer()[0, 0], values.var(ddof=1), places=6)
        self.assertIsNone(target.elevation_fused_mean)
        self.assertIsNone(target.elevation_estimate_variance)

    def test_stereo_lidar_weighted_update_and_distinct_roughness(self):
        target = grid()
        stereo(target, .2)
        lidar(target, 0.)
        expected = .2 * .01 / (.0401 + .01)
        self.assertAlmostEqual(target.elevation_layer()[0, 0], expected, places=7)
        self.assertAlmostEqual(target.elevation_estimate_variance[0, 0], .0401 * .01 / .0501, places=8)
        self.assertAlmostEqual(target.elevation_mean[0, 0], .1)
        self.assertAlmostEqual(target.elevation_variance_layer()[0, 0], .02, places=7)
        self.assertAlmostEqual(target.height_range_layer()[0, 0], .2, places=7)

    def test_sample_density_does_not_dominate_fused_height(self):
        sparse, dense = grid(), grid()
        stereo(sparse, .2)
        stereo(dense, .2, count=1000)
        lidar(sparse, 0.)
        lidar(dense, 0.)
        np.testing.assert_array_equal(sparse.elevation_layer(), dense.elevation_layer())
        np.testing.assert_array_equal(sparse.elevation_estimate_variance, dense.elevation_estimate_variance)
        self.assertEqual(dense.elevation_count[0, 0], 1001)

    def test_spatial_scatter_reduces_measurement_confidence(self):
        target = grid()
        target._update_elevation_cell(0, 0, np.array([-.2, .2]))
        self.assertAlmostEqual(target.elevation_estimate_variance[0, 0], .08, places=7)

    def test_mad_filter_still_rejects_intra_scan_outlier(self):
        target = grid()
        accepted = target._update_elevation_cell(0, 0, np.array([1., 1., 1., 99.]))
        self.assertEqual(accepted, 3)
        self.assertEqual(target.elevation_layer()[0, 0], 1.)

    def test_gate_rejects_both_directions_without_changing_elevation_state(self):
        target = manager()
        lidar(target, .1)
        tile = target.tiles[(0, 0)]
        names = [name for name, _ in target._snapshot_tile_fields()]
        before = {name: getattr(tile, name).copy() for name in names}
        for value in (3., -3., 3.):
            stats = lidar(target, value)
            self.assertEqual(stats["accepted_points"], 0)
            self.assertEqual(stats["elevation_rejected_points"], 1)
            for name in names:
                np.testing.assert_array_equal(getattr(tile, name), before[name])
        self.assertEqual(lidar(target, .11)["elevation_rejected_points"], 0)

    def test_unknown_cell_and_negative_tile_can_initialize_at_any_height(self):
        target = manager()
        stats = target.update_elevation_only(points_map=np.array([
            [.5, .5, 10.], [-.5, -.5, -10.], [5.5, .5, 20.]]))
        self.assertEqual(stats["accepted_points"], 3)
        self.assertEqual(stats["elevation_min"], -10.)
        self.assertEqual(stats["elevation_max"], 20.)
        self.assertEqual(target.surface_cells, 3)

    def test_confidence_floor_allows_later_small_correction(self):
        target = grid()
        for _ in range(600):
            lidar(target, 0.)
        self.assertGreaterEqual(float(target.elevation_estimate_variance[0, 0]), .0004 - 1e-9)
        self.assertEqual(lidar(target, .1)["accepted_points"], 1)
        self.assertGreater(target.elevation_layer()[0, 0], .005)

    def test_gate_minimum_is_in_metres(self):
        target = grid(measurement_variance=.0004, lidar_measurement_variance=.0004,
                      process_variance=0.)
        lidar(target, 0.)
        self.assertEqual(lidar(target, .29)["accepted_points"], 1)
        self.assertEqual(lidar(target, 1.)["accepted_points"], 0)

    def test_nonfinite_input_is_ignored(self):
        target = manager()
        result = target.update_elevation_only(points_map=np.array([
            [0., 0., np.nan], [0., np.inf, 0.], [.5, .5, 0.]]))
        self.assertEqual(result["accepted_points"], 1)
        self.assertEqual(target.surface_cells, 1)

    def test_elevation_only_cow_and_later_semantic_promotion(self):
        original = manager()
        stereo(original, .2)
        before = {name: getattr(original.tiles[(0, 0)], name).copy()
                  for name, _ in original._snapshot_tile_fields()}
        candidate = original.clone_for_update()
        lidar(candidate, 0.)
        self.assertIs(candidate.tiles[(0, 0)].semantic_votes, original.tiles[(0, 0)].semantic_votes)
        for name, _ in original._FILTER_TILE_FIELDS:
            self.assertFalse(np.shares_memory(getattr(candidate.tiles[(0, 0)], name),
                                             getattr(original.tiles[(0, 0)], name)))
        stereo(candidate, .1)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(original.tiles[(0, 0)], name), value)

    def test_snapshot_restores_filter_and_continues_identically(self):
        target = manager(process_variance=.0002)
        stereo(target, .2)
        lidar(target, 0.)
        with tempfile.TemporaryDirectory() as directory:
            paths = target.save_snapshot(Path(directory), timestamp_text="test", frame_id="map")
            metadata = yaml.safe_load(paths["index"].read_text())
            self.assertEqual(metadata["format_version"], 2)
            restored = TiledSemanticMapManager.load_snapshot(paths["index"])
            for z in [.1, 2., -.1]:
                self.assertEqual(lidar(target, z), lidar(restored, z))
            for name, _ in target._snapshot_tile_fields():
                np.testing.assert_array_equal(getattr(target.tiles[(0, 0)], name),
                                              getattr(restored.tiles[(0, 0)], name))

    def test_original_v1_snapshot_without_filter_parameters_loads_as_mean(self):
        target = manager("mean")
        lidar(target, .2)
        with tempfile.TemporaryDirectory() as directory:
            paths = target._write_snapshot_directory(Path(directory), timestamp_text="v1", frame_id="map")
            metadata = yaml.safe_load(paths["index"].read_text())
            self.assertEqual(metadata["format_version"], 1)
            for name in ElevationFusionConfig().parameters():
                metadata["fusion"].pop(name)
            paths["index"].write_text(yaml.safe_dump(metadata))
            restored = TiledSemanticMapManager.load_snapshot(paths["index"])
            lidar(restored, 4.)
            self.assertAlmostEqual(restored.tiles[(0, 0)].elevation_layer()[0, 0], 2.1, places=6)

    def test_corrupt_filter_state_cannot_silently_resume(self):
        for corruption in ("missing", "invalid"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                target = manager()
                lidar(target, 0.)
                paths = target._write_snapshot_directory(Path(directory), timestamp_text="v2", frame_id="map")
                metadata = yaml.safe_load(paths["index"].read_text())
                tile_path = Path(directory) / metadata["tiles"][0]["file"]
                with np.load(tile_path) as saved:
                    values = {key: saved[key].copy() for key in saved.files}
                if corruption == "missing":
                    values.pop("elevation_estimate_variance")
                else:
                    values["elevation_estimate_variance"][0, 0] = np.nan
                np.savez_compressed(tile_path, **values)
                with self.assertRaises(ValueError):
                    TiledSemanticMapManager.load_snapshot(paths["index"])

    def test_layer_contract_and_memory_cost(self):
        old, new = grid("mean"), grid()
        self.assertEqual(set(old.all_layers()), set(new.all_layers()))
        self.assertEqual(new.elevation_fused_mean.nbytes + new.elevation_estimate_variance.nbytes,
                         8 * new.elevation_count.size)
        self.assertTrue(np.isnan(new.elevation_layer()).all())

    def test_weighted_height_reaches_dense_compact_and_ros_outputs(self):
        from std_msgs.msg import Header
        from t3_semantic_mapping.compact_grid_store import save_compact_global_map, load_compact_tile
        from t3_semantic_mapping.dense_grid_store import save_dense_global_grid_map
        from t3_semantic_mapping.grid_map_message import make_grid_map_message
        target = manager()
        stereo(target, .2)
        lidar(target, 0.)
        window = target.extract_complete_window()
        expected = .2 * .01 / .0501
        self.assertAlmostEqual(window.elevation[0, 0], expected, places=7)
        message = make_grid_map_message(header=Header(frame_id="map"),
            geometry=window.geometry, layers={"elevation": window.elevation,
                                             "roughness": window.roughness})
        values = np.asarray(message.data[0].data)
        self.assertEqual(message.layers, ["elevation", "roughness"])
        self.assertAlmostEqual(values[np.isfinite(values)][0], expected, places=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_dense_global_grid_map(path / "dense.npz", window, frame_id="map",
                                       map_revision=target.map_revision, timestamp_text="test")
            with np.load(path / "dense.npz") as saved:
                self.assertAlmostEqual(saved["elevation"][0, 0], expected, places=7)
            save_compact_global_map(target, path / "compact.sqlite3",
                                    frame_id="map", timestamp_text="test")
            decoded = load_compact_tile(path / "compact.sqlite3", 0, 0)
            self.assertAlmostEqual(decoded["elevation"][0, 0], expected, delta=.0051)

    def test_demo_launch_parameter_wiring_without_starting_nodes(self):
        from launch import LaunchContext
        from launch.actions import DeclareLaunchArgument
        bundle = Path(__file__).resolve().parents[2]
        launch_path = bundle / "workspace/src/t3_semantic_mapping/launch/t3_envx_demo.launch.py"
        spec = importlib.util.spec_from_file_location("height_test_launch", launch_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode in ("mean", "variance"):
            context = LaunchContext()
            context.environment["T3_ELEVATION_FUSION"] = mode
            context.environment["T3_BUNDLE_ROOT"] = str(bundle)
            context.environment["T3_OUTPUT_DIR"] = str(bundle / "integration_demo/tmp/unused_launch_test")
            for entity in module.generate_launch_description().entities:
                if isinstance(entity, DeclareLaunchArgument):
                    entity.execute(context)
            # _nodes only creates descriptions; mock Node as an extra guard.
            with patch.object(module, "Node", side_effect=lambda **kwargs: kwargs):
                descriptions = module._nodes(context)
            parameters = descriptions[1]["parameters"][0]
            config = ElevationFusionConfig.from_parameters(parameters)
            self.assertEqual(config.fusion_mode, mode)
            self.assertEqual(parameters["localization_mode"], "vo")
            self.assertFalse(parameters["enable_lidar_mapping"])
            self.assertTrue(parameters["publish_registered_lidar"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
