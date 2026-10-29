"""Read-only coarse terrain-route screening from indexed reference pose samples."""
import argparse
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote
import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message


def summarize(bag):
    metadata = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    start = metadata['starting_time']['nanoseconds_since_epoch'] * 1e-9
    samples = []
    for i, relative in enumerate(metadata['relative_file_paths']):
        path = bag/relative
        connection = sqlite3.connect('file:'+quote(str(path.resolve()))+'?mode=ro&immutable=1', uri=True)
        try:
            topic = connection.execute("SELECT id FROM topics WHERE name='/car/pose'").fetchone()
            if topic is None:
                continue
            first = connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1', topic).fetchone()
            last = connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1', topic).fetchone()
            if first is None or last is None:
                continue
            mid = connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp>=? ORDER BY timestamp LIMIT 1',
                                     (topic[0], (first[0]+last[0])//2)).fetchone()
            for row in (first, mid, last):
                if row is None:
                    continue
                message = deserialize_message(row[1], PoseStamped)
                p = message.pose.position
                samples.append([row[0]*1e-9-start, p.x, p.y, p.z])
        finally:
            connection.close()
        if i%100 == 0:
            print(f'{bag.name}: screened {i+1}/{len(metadata["relative_file_paths"])} chunks', flush=True)
    data = np.unique(np.asarray(samples, dtype=float), axis=0)
    data = data[np.isfinite(data).all(axis=1)]
    candidates = []
    for row in data:
        window = data[(data[:,0] >= row[0]) & (data[:,0] <= row[0]+240)]
        if len(window)<8 or window[-1,0]-window[0,0]<180:
            continue
        candidates.append({'start_seconds': float(window[0,0]), 'duration_seconds': 240,
                           'sampled_height_span_m': float(np.ptp(window[:,3])),
                           'min_height_m': float(window[:,3].min()), 'max_height_m': float(window[:,3].max()),
                           'reference_samples': len(window)})
    selected = []
    for item in sorted(candidates, key=lambda x:x['sampled_height_span_m'], reverse=True):
        if all(abs(item['start_seconds']-prior['start_seconds'])>240 for prior in selected):
            selected.append(item)
            if len(selected)==5:
                break
    return {'bag': str(bag), 'sample_count': len(data), 'sampled_height_span_m': float(np.ptp(data[:,3])),
            'candidates': selected, 'reference_sample_rows': data.tolist(),
            'qualification': 'Coarse read-only reference screening: first/middle/last reference pose of each SQLite chunk. Candidate windows still require full sensor replay and visual validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path('/home/yanfa/P3')):
        raise ValueError('Reports must stay inside P3')
    result = summarize(args.bag)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reference_sample_rows'}, indent=2))
