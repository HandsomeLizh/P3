import json
from pathlib import Path

import numpy as np
import yaml

from t1_dataset_sim.dataset_formats import (
    ORGANIZED_FORMAT,
    StereoRectifier,
    detect_dataset_format,
    load_calibration,
    load_organized_imu_records,
    load_organized_stereo_records,
)
from t1_dataset_sim.fastlio_io import load_organized_fastlio_odometry


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_organized_timestamps_pair_by_index_and_reject_skew(tmp_path):
    for side in ("left", "right"):
        for index in range(2):
            _touch(tmp_path / "images" / side / f"{index:06d}.png")

    table = tmp_path / "images" / "timestamps.tsv"
    table.write_text(
        "side\tframe_index\ttimestamp_ns\tfilename\n"
        "left\t000000\t1000000000\timages/left/000000.png\n"
        "left\t000001\t2000000000\timages/left/000001.png\n"
        "right\t000000\t1005000000\timages/right/000000.png\n"
        "right\t000001\t2050000000\timages/right/000001.png\n",
        encoding="utf-8",
    )

    assert detect_dataset_format(tmp_path) == ORGANIZED_FORMAT
    records, stats = load_organized_stereo_records(
        tmp_path, max_abs_skew_ns=20_000_000
    )

    assert len(records) == 1
    assert records[0].timestamp_ns == 1_000_000_000
    assert records[0].skew_ns == 5_000_000
    assert stats.shared_rows == 2
    assert stats.eligible_rows == 1
    assert stats.paired_rows == 1
    assert stats.skipped_for_skew == 1
    assert stats.skipped_for_max_frames == 0
    assert stats.max_abs_skew_ns == 50_000_000
    assert stats.selection_summary(
        pairing_method="frame_index",
        configured_max_abs_skew_ns=20_000_000,
        max_frames=0,
    ) == {
        "schema_version": 1,
        "pairing_method": "frame_index",
        "eligibility_definition": (
            "shared pair whose absolute left/right timestamp skew does not "
            "exceed configured_max_stereo_skew_ms"
        ),
        "raw_left_row_count": 2,
        "raw_right_row_count": 2,
        "shared_candidate_pair_count": 2,
        "stereo_skew_eligible_pair_count": 1,
        "selected_frame_count": 1,
        "excluded_by_stereo_skew_count": 1,
        "excluded_by_max_frames_count": 0,
        "missing_left_count": 0,
        "missing_right_count": 0,
        "configured_max_stereo_skew_ms": 20.0,
        "observed_max_abs_stereo_skew_ms": 50.0,
        "configured_max_frames": 0,
        "all_timestamp_rows_have_partner": True,
        "all_shared_candidates_pass_stereo_skew": False,
        "all_stereo_skew_eligible_pairs_selected": True,
    }


def test_organized_pairing_stats_cover_full_table_when_selection_is_limited(
    tmp_path,
):
    for side in ("left", "right"):
        for index in range(3):
            _touch(tmp_path / "images" / side / f"{index:06d}.png")

    table = tmp_path / "images" / "timestamps.tsv"
    table.write_text(
        "side\tframe_index\ttimestamp_ns\tfilename\n"
        "left\t000000\t1000000000\timages/left/000000.png\n"
        "left\t000001\t2000000000\timages/left/000001.png\n"
        "left\t000002\t3000000000\timages/left/000002.png\n"
        "right\t000000\t1000000000\timages/right/000000.png\n"
        "right\t000001\t2005000000\timages/right/000001.png\n"
        "right\t000002\t3100000000\timages/right/000002.png\n",
        encoding="utf-8",
    )

    records, stats = load_organized_stereo_records(
        tmp_path,
        max_abs_skew_ns=20_000_000,
        max_frames=1,
    )

    assert [record.frame_index for record in records] == ["000000"]
    assert stats.shared_rows == 3
    assert stats.eligible_rows == 2
    assert stats.paired_rows == 1
    assert stats.skipped_for_skew == 1
    assert stats.skipped_for_max_frames == 1
    assert stats.max_abs_skew_ns == 100_000_000


def test_organized_json_calibration_builds_metric_rectifier(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    intrinsics = {
        "Cam_L": {
            "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0] * 5,
            "resolution": [48, 64],
        },
        "Cam_R": {
            "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0] * 5,
            "resolution": [48, 64],
        },
    }
    extrinsics = {
        "Cam_R_to_Cam_L": {
            "R": np.eye(3).tolist(),
            "T": [[-200.0], [0.0], [0.0]],
        }
    }
    (calibration_dir / "intrinsics.json").write_text(
        json.dumps(intrinsics), encoding="utf-8"
    )
    (calibration_dir / "extrinsics.json").write_text(
        json.dumps(extrinsics), encoding="utf-8"
    )

    calibration = load_calibration(tmp_path)
    assert calibration["baseline"] == 0.2
    assert calibration["camera"]["width"] == 64
    assert calibration["camera"]["height"] == 48

    rectifier = StereoRectifier(calibration)
    left = np.zeros((48, 64, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    rectified_left, rectified_right = rectifier.rectify(left, right)
    assert rectified_left.shape == left.shape
    assert rectified_right.shape == right.shape
    labels = np.zeros((48, 64), dtype=np.uint8)
    assert rectifier.rectify_left(labels, interpolation=0).shape == labels.shape
    assert rectifier.camera_parameters(64, 48)["fx"] > 0.0


def test_organized_imu_preserves_rad_per_second_units(tmp_path):
    path = tmp_path / "imu.csv"
    path.write_text(
        "record_time_ns,header_time_ns,angular_velocity_x,"
        "angular_velocity_y,angular_velocity_z\n"
        "1001,1000,1.0,-2.0,3.0\n",
        encoding="utf-8",
    )

    records = load_organized_imu_records(path)
    assert len(records) == 1
    assert records[0].timestamp_ns == 1000
    np.testing.assert_allclose(records[0].gyro_rad_s, [1.0, -2.0, 3.0])


def test_legacy_yaml_calibration_remains_supported(tmp_path):
    expected = {
        "camera": {"fx": 100.0, "fy": 101.0, "cx": 32.0, "cy": 24.0},
        "baseline": 0.2,
    }
    path = tmp_path / "calib.yaml"
    path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    loaded = load_calibration(path)
    assert loaded == expected
    assert load_calibration(tmp_path) == expected
    assert StereoRectifier(loaded).enabled is False


def test_raw_stereo_yaml_builds_rectification_and_map_alignment(tmp_path):
    right_to_left = np.eye(4, dtype=np.float64)
    right_to_left[0, 3] = 0.25
    raw = {
        "raw_stereo": {
            "width": 64,
            "height": 48,
            "left": {
                "k": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
                "d": [0.0] * 5,
            },
            "right": {
                "k": [[101.0, 0.0, 32.0], [0.0, 101.0, 24.0], [0.0, 0.0, 1.0]],
                "d": [0.0] * 5,
            },
            "transform_right_to_left": right_to_left.tolist(),
        },
        "mounting": {"transform_vehicle_left": np.eye(4).tolist()},
    }
    path = tmp_path / "calib.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    calibration = load_calibration(path)
    assert calibration["baseline"] == 0.25
    assert calibration["rectification"]["enabled"] is True
    assert calibration["rectification"]["source_undistorted"] is False
    rotation = np.asarray(
        calibration["map_alignment"]["rotation_camera_to_map"]
    )
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)

    rectifier = StereoRectifier(calibration)
    left = np.zeros((48, 64, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    assert rectifier.rectify(left, right)[0].shape == left.shape


def test_fastlio_replay_prefers_sensor_time_filters_and_normalizes(tmp_path):
    trajectory_dir = tmp_path / "lidar_fastlio"
    trajectory_dir.mkdir()
    trajectory = trajectory_dir / "trajectory.npz"
    np.savez(
        trajectory,
        record_time_ns=np.array([9000, 9001, 9002], dtype=np.int64),
        header_time_ns=np.array([100, 200, 300], dtype=np.int64),
        position=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        orientation=np.array(
            [[0.0, 0.0, 0.0, 2.0]] * 3, dtype=np.float64
        ),
    )

    records = load_organized_fastlio_odometry(
        tmp_path,
        first_timestamp_ns=150,
        last_timestamp_ns=300,
        stride=2,
    )

    assert [record.timestamp_ns for record in records] == [200]
    np.testing.assert_allclose(
        records[0].orientation_xyzw, [0.0, 0.0, 0.0, 1.0]
    )
    assert records[0].pose_covariance.shape == (6, 6)


def test_fastlio_replay_rejects_nonmonotonic_sensor_timestamps(tmp_path):
    trajectory_dir = tmp_path / "lidar_fastlio"
    trajectory_dir.mkdir()
    np.savez(
        trajectory_dir / "trajectory.npz",
        header_time_ns=np.array([200, 100], dtype=np.int64),
        position=np.zeros((2, 3), dtype=np.float64),
        orientation=np.array(
            [[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float64
        ),
    )

    try:
        load_organized_fastlio_odometry(tmp_path)
    except ValueError as exception:
        assert "not increasing" in str(exception)
    else:
        raise AssertionError("non-monotonic FAST-LIO timestamps were accepted")
