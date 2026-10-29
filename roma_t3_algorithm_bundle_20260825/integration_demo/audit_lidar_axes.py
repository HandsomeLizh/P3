"""Read-only bag diagnostic: fixed visual poses, two explicit sensor-axis hypotheses.

No ROS publications, ICP fitting, calibration overwrite or existing-map rewrite.
The second transform only converts a ROS FLU input into the FRD sensor axes
assumed by the measured mounting matrix. This is a hypothesis test, not a new
calibration or an absolute map-accuracy certification.
"""
import argparse
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from t3_semantic_mapping.lidar_mapping import pointcloud2_xyz_array


ROOT = Path(__file__).resolve().parents[1]


def open_readonly(path):
    return sqlite3.connect('file:' + quote(str(path.resolve())) + '?mode=ro&immutable=1', uri=True)


def raw_at(bag, metadata, timestamp):
    target = int(timestamp * 1e9)
    candidates = []
    for item in metadata['files']:
        start = item['starting_time']['nanoseconds_since_epoch']
        end = start + item['duration']['nanoseconds']
        if end < target-1_500_000_000 or start > target+1_500_000_000:
            continue
        with open_readonly(bag / item['path']) as con:
            topic = con.execute("SELECT id FROM topics WHERE name='/Car/T5/OS1/points'").fetchone()
            if topic is None:
                continue
            candidates.extend(con.execute(
                'SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ?',
                (topic[0], target-1_500_000_000, target+1_500_000_000)).fetchall())
    if not candidates:
        raise ValueError('No nearby LiDAR observation')
    stamp, data = min(candidates, key=lambda item: abs(item[0]-target))
    message = deserialize_message(data, PointCloud2)
    points = pointcloud2_xyz_array(message)
    radius = np.linalg.norm(points, axis=1)
    points = points[np.isfinite(points).all(axis=1) & (radius > .5) & (radius < 35.)]
    _, selected = np.unique(np.floor(points/.2).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(selected)], (stamp-target)*1e-9


def world_points(raw, mounting, row):
    base = raw @ mounting[:3, :3].T + mounting[:3, 3]
    return base @ Rotation.from_quat(row[4:8]).as_matrix().T + row[1:4]


def overlap_residual(reference, candidate):
    # Compare to locally planar neighborhoods, without moving either scan.
    # Euclidean nearest-point distance alone confounds beam sampling and alignment.
    sample = candidate[::max(1, int(np.ceil(len(candidate)/6000)))]
    distances, indices = cKDTree(reference).query(sample, k=12, workers=1)
    neighbors = reference[indices]
    center = neighbors.mean(axis=1)
    centered = neighbors-center[:, None, :]
    covariance = np.einsum('nki,nkj->nij', centered, centered)/12
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    mask = ((distances[:, 0] < 1.5) & (distances[:, -1] < 2.) &
            (eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12) < .02) &
            (eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-12) > .05))
    residual = np.abs(np.einsum('ni,ni->n', sample-center, eigenvectors[:, :, 0]))[mask]
    return {'qualified_points': len(residual),
            'median_abs_plane_distance_m': float(np.median(residual)) if len(residual) else None,
            'p95_abs_plane_distance_m': float(np.percentile(residual, 95)) if len(residual) else None,
            'qualification': 'Local planar overlap consistency, not absolute accuracy; samples differ between hypotheses.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path('/home/yanfa/P3')):
        raise ValueError('Diagnostic outputs must remain in P3')
    rows = np.loadtxt(args.output/'trajectory_map.tum', comments='#', ndmin=2)
    selected = sorted(set([0, 1, 2, 3, 4, 5, min(10, len(rows)-1), min(20, len(rows)-1),
                           min(40, len(rows)-1), len(rows)-1]))
    selected = [i for i in selected if i < len(rows)]
    metadata = yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    # Reconstruct the pre-fix baseline explicitly so rerunning this diagnostic
    # after deployment does not accidentally apply the input conversion twice.
    original = yaml.safe_load((ROOT/'workspace/src/t3_semantic_mapping/config/t5_rgb10_rgb11.yaml').read_text())
    current = np.diag([1., -1., -1., 1.]) @ np.asarray(original['mounting']['transform_vehicle_lidar'])
    hypotheses = {'current': current, 'input_FLU_to_calibrated_FRD': current @ np.diag([1., -1., -1., 1.])}
    clouds = {name: {} for name in hypotheses}
    result = {'source_bag': str(args.bag), 'source_trajectory': str(args.output/'trajectory_map.tum'),
              'modified_existing_map_or_calibration': False,
              'method': 'Same recorded visual trajectory for both hypotheses. No ICP, no ground-truth pose injection.',
              'candidate_transform_base_sensor': hypotheses['input_FLU_to_calibrated_FRD'].tolist(),
              'frames': [], 'overlap_to_first_frame': {}}
    for i in selected:
        raw, offset = raw_at(args.bag, metadata, rows[i, 0])
        item = {'frame_index': i, 'bag_receive_offset_sec': offset, 'nearby_raw_points': len(raw)}
        for name, transform in hypotheses.items():
            clouds[name][i] = world_points(raw, transform, rows[i])
            if i == 0:
                near = clouds[name][i][np.linalg.norm(clouds[name][i][:, :2], axis=1) < 5]
                item[name+'_near_z_percentiles_m'] = np.percentile(near[:, 2], [10, 50, 90]).tolist()
        result['frames'].append(item)
        print('Read frame', i, flush=True)
    for name, group in clouds.items():
        result['overlap_to_first_frame'][name] = {
            str(i): overlap_residual(group[0], group[i]) for i in selected if i != 0}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for col, (name, group) in enumerate(clouds.items()):
        for row_index, frames in enumerate([selected[:6], [selected[0], selected[-3], selected[-2], selected[-1]]]):
            for index in frames:
                p = group[index]
                p = p[(np.abs(p[:, 1]) < 2.) & (p[:, 0] > -5) & (p[:, 0] < 35)]
                axes[row_index, col].scatter(p[:, 0], p[:, 2], s=1, label=f'frame {index}', alpha=.6)
            axes[row_index, col].plot(rows[:, 1], rows[:, 3], color='black', linewidth=1, label='visual body path')
            axes[row_index, col].set_xlabel('Map X / m')
            axes[row_index, col].set_ylabel('Map Z / m')
            axes[row_index, col].set_title(name + (' / initial frames' if row_index == 0 else ' / later overlap'))
            axes[row_index, col].legend(markerscale=3, fontsize=8)
            axes[row_index, col].grid(alpha=.2)
    report = args.output/'lidar_axis_diagnostic.json'
    report.write_text(json.dumps(result, indent=2)+'\n')
    fig.savefig(args.output/'lidar_axis_diagnostic.png', dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
