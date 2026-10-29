"""Read-only map/trajectory inspection, with results written beside P3 maps.

Reference poses, when requested, are read from SQLite, never published to ROS.
Bag receive time is an approximate association clock, not exposure ground truth.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
from evaluate_trajectory_accuracy import _load_tum, _rigid_alignment, _statistics, _path_length


def readonly_database(path):
    return sqlite3.connect('file:' + quote(str(path.resolve())) + '?mode=ro&immutable=1', uri=True)


def reference_positions(bag, time_bounds=None):
    import yaml
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import deserialize_message
    metadata = yaml.safe_load((bag / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    rows = []
    file_times = {item['path']: item for item in metadata.get('files', [])}
    nanoseconds = None if time_bounds is None else (int((time_bounds[0]-2)*1e9), int((time_bounds[1]+2)*1e9))
    for relative in metadata['relative_file_paths']:
        if nanoseconds is not None and relative in file_times:
            item = file_times[relative]
            first = item['starting_time']['nanoseconds_since_epoch']
            last = first + item['duration']['nanoseconds']
            if last < nanoseconds[0] or first > nanoseconds[1]:
                continue
        with readonly_database(bag / relative) as connection:
            topic = connection.execute("SELECT id FROM topics WHERE name='/car/pose'").fetchone()
            if topic is None:
                continue
            query = 'SELECT timestamp,data FROM messages WHERE topic_id=?'
            parameters = topic
            if nanoseconds is not None:
                query += ' AND timestamp BETWEEN ? AND ?'
                parameters += nanoseconds
            for timestamp, data in connection.execute(query + ' ORDER BY timestamp', parameters):
                message = deserialize_message(data, PoseStamped)
                p = message.pose.position
                rows.append((timestamp * 1e-9, p.x, p.y, p.z))
    array = np.asarray(rows, dtype=float).reshape(-1, 4)
    if len(array) < 3 or not np.isfinite(array).all():
        raise ValueError('Insufficient or nonfinite reference poses')
    array = array[np.argsort(array[:, 0])]
    _, selected = np.unique(array[:, 0], return_index=True)
    return array[selected]


def compare_reference(output, times, positions, bag, max_gap=1.0):
    reference = reference_positions(bag, (times[0], times[-1]))
    indices = np.searchsorted(reference[:, 0], times)
    selected = np.flatnonzero((indices > 0) & (indices < len(reference)))
    brackets = indices[selected]
    gaps = reference[brackets, 0] - reference[brackets - 1, 0]
    selected = selected[gaps <= max_gap]
    gaps = gaps[gaps <= max_gap]
    if len(selected) < 3:
        raise ValueError(f'Too few reference associations within {max_gap}-second brackets')
    reference_sample = np.column_stack([
        np.interp(times[selected], reference[:, 0], reference[:, axis]) for axis in (1, 2, 3)])
    rotation, translation, scale = _rigid_alignment(positions[selected], reference_sample)
    aligned = positions[selected] @ rotation.T + translation
    errors = np.linalg.norm(aligned - reference_sample, axis=1)
    stats = _statistics(errors)
    distance = _path_length(reference_sample)
    singular = np.linalg.svd(reference_sample - reference_sample.mean(axis=0), compute_uv=False)
    result = {
        'source': str(bag), 'topic': '/car/pose', 'association': 'interpolated bag receive time',
        'matched_poses': len(selected), 'available_estimates': len(times),
        'maximum_reference_bracket_sec': float(gaps.max()),
        'maximum_permitted_reference_bracket_sec': max_gap,
        'fixed_scale_se3_position_error_m': stats,
        'reference_sampled_path_m': distance,
        'estimate_matched_path_m': _path_length(positions[selected]),
        'rmse_over_reference_path_percent': 100 * stats['rmse'] / distance if distance > 0 else None,
        'successive_displacement_error_m': _statistics(np.linalg.norm(np.diff(aligned, axis=0) - np.diff(reference_sample, axis=0), axis=1)),
        'scale_diagnostic_not_applied': float(scale),
        'trajectory_spatial_singular_values_m': singular.tolist(),
        'qualification': 'Approximate receive-time association; fixed-scale SE(3), no scale correction. Diagnostic, not exposure-time accuracy certification.'}
    with (output / 'trajectory_accuracy_samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sensor_time', 'estimate_aligned_x', 'estimate_aligned_y', 'estimate_aligned_z',
                         'reference_x', 'reference_y', 'reference_z', 'position_error_m'])
        writer.writerows(np.column_stack([times[selected], aligned, reference_sample, errors]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    origin = reference_sample[0]
    axes[0].plot(reference_sample[:, 0]-origin[0], reference_sample[:, 1]-origin[1], color='#477aa5', label='UE reference')
    axes[0].plot(aligned[:, 0]-origin[0], aligned[:, 1]-origin[1], color='#db7840', label='Visual estimate (SE3, scale=1)')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Reference X offset / m')
    axes[0].set_ylabel('Reference Y offset / m')
    axes[0].legend()
    axes[1].plot(times[selected]-times[selected][0], errors, color='#db7840')
    axes[1].set_xlabel('Sensor time / s')
    axes[1].set_ylabel('Position error / m')
    axes[1].set_title(f"RMSE {stats['rmse']:.3f} m; p95 {stats['p95']:.3f} m")
    fig.savefig(output / 'trajectory_accuracy.png', dpi=130)
    plt.close(fig)
    (output / 'trajectory_accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def inspect(output, bag=None):
    if not output.resolve().is_relative_to(Path('/home/yanfa/P3')):
        raise ValueError('Inspection output must be in /home/yanfa/P3')
    times, positions, rotations = _load_tum(output / 'trajectory_map.tum')
    result = {'output': str(output), 'trajectory': {
        'pose_count': len(times), 'sensor_duration_sec': float(times[-1] - times[0]),
        'path_length_m': _path_length(positions),
        'maximum_step_m': float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(times) > 1 else 0.,
        'all_finite': True, 'strictly_increasing_timestamps': True}}
    pcd = output / 'lidar_map.pcd'
    fields = {}
    with pcd.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('PCD binary header missing')
            parts = line.decode('ascii').strip().split()
            if parts and not parts[0].startswith('#'):
                fields[parts[0]] = parts[1:]
            if line == b'DATA binary\n':
                offset = stream.tell()
                break
    if fields['FIELDS'] != ['x', 'y', 'z'] or fields['TYPE'] != ['F', 'F', 'F'] or fields['SIZE'] != ['4', '4', '4']:
        raise ValueError('Expected packed float32 XYZ PCD')
    count = int(fields['POINTS'][0])
    if count <= 0 or pcd.stat().st_size != offset + count * 12:
        raise ValueError('PCD count/file size mismatch')
    cloud = np.memmap(pcd, dtype='<f4', mode='r', offset=offset, shape=(count, 3))
    bounds_min, bounds_max = np.full(3, np.inf), np.full(3, -np.inf)
    for start in range(0, count, 8192):
        block = cloud[start:start + 8192]
        if not np.isfinite(block).all():
            raise ValueError('PCD contains nonfinite coordinates')
        bounds_min = np.minimum(bounds_min, block.min(axis=0))
        bounds_max = np.maximum(bounds_max, block.max(axis=0))
    with readonly_database(output / 'lidar_map.sqlite3') as connection:
        database_count = connection.execute('SELECT COUNT(*) FROM voxels').fetchone()[0]
        database_check = connection.execute('PRAGMA quick_check').fetchone()[0]
    if database_count != count or database_check != 'ok':
        raise ValueError('Final PCD and SQLite disagree or database check failed')
    result['cloud'] = {'full_points': count, 'database_points': database_count,
                       'database_quick_check': database_check, 'all_finite': True,
                       'min_xyz': bounds_min.tolist(), 'max_xyz': bounds_max.tolist(),
                       'pcd_bytes': pcd.stat().st_size}
    with np.load(output / 'global_grid_map.npz', allow_pickle=False) as archive:
        elevation = archive['elevation']
        occupancy = archive['occupancy']
        resolution = float(archive['resolution'])
        origin = [float(archive['origin_x']), float(archive['origin_y'])]
        known = np.isfinite(elevation)
        if not known.any() or not ((occupancy >= -1) & (occupancy <= 100)).all():
            raise ValueError('Invalid elevation/occupancy map')
        result['grid'] = {'frame_id': str(archive['frame_id']), 'resolution': resolution,
                          'shape': list(elevation.shape), 'known_height_cells': int(known.sum()),
                          'known_area_m2': float(known.sum() * resolution**2),
                          'elevation_percentiles_m': np.percentile(elevation[known], [0, 5, 50, 95, 100]).tolist()}
    if bag:
        result['reference_diagnostic'] = compare_reference(output, times, positions, bag)
    with (output / 'pose_processing_time.csv').open() as stream:
        timings = [float(row['processing_time_sec']) for row in csv.DictReader(stream)]
    result['pose_processing_sec'] = _statistics(timings)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    sample = cloud[::max(1, int(np.ceil(count / 100000)))]
    axes[0].scatter(sample[:, 0], sample[:, 1], c=sample[:, 2], s=0.3, cmap='terrain', rasterized=True)
    axes[0].plot(positions[:, 0], positions[:, 1], color='#d02030', linewidth=1, label='Visual body trajectory')
    axes[0].legend(loc='best')
    axes[0].set_title(f'VO-registered LiDAR (preview of {count:,} full voxels)')
    height, width = elevation.shape
    picture = axes[1].imshow(elevation, origin='lower', cmap='terrain',
                             extent=[origin[0], origin[0] + width * resolution,
                                     origin[1], origin[1] + height * resolution])
    axes[1].plot(positions[:, 0], positions[:, 1], color='#d02030', linewidth=1)
    axes[1].set_title('Elevation map (unknown cells transparent)')
    fig.colorbar(picture, ax=axes[1], label='Elevation / m')
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('Map X / m')
        ax.set_ylabel('Map Y / m')
    fig.savefig(output / 'map_quality_overview.png', dpi=140)
    plt.close(fig)
    result['file_checks_passed'] = True
    (output / 'saved_map_assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bag', type=Path)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    if args.reference_only:
        if not args.bag or not args.output.resolve().is_relative_to(Path('/home/yanfa/P3')):
            parser.error('--reference-only requires --bag and a P3 output folder')
        times, positions, _ = _load_tum(args.output / 'trajectory_map.tum')
        print(json.dumps(compare_reference(args.output, times, positions, args.bag), indent=2))
    else:
        inspect(args.output, args.bag)
