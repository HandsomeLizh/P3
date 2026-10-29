# Map delivery interfaces

Runtime baseline: 2026-08-21. The rolling local map is expressed in `odom`;
complete global products and persistent files are expressed in `map`.

The mapper keeps its complete high-resolution global state sparse internally
and publishes a complete dense snapshot whenever the map revision changes.

## Complete real-time global map

- Topic: `/Car/T3/mapping/global_grid_map`
- Type: `grid_map_msgs/msg/GridMap`
- Frame: `map`
- Geometry: tile-aligned bounding rectangle of all allocated global-map tiles
- Resolution: `0.2 m/cell`
- Layers: `elevation`
- Trigger: once per changed map revision
- QoS: reliable, transient-local, keep-last depth 1

Late subscribers immediately receive the newest complete elevation snapshot.
The full occupancy, semantic, quality, and elevation layers remain available in
the live NPZ/SQLite files. Message size grows with the area of the tile-aligned
global bounding rectangle, so the local map and global overview remain
available for consumers that do not need high-resolution elevation.

## Real-time local map

- Topic: `/Car/T3/mapping/grid_map`
- Type: `grid_map_msgs/msg/GridMap`
- Frame: `odom`
- Geometry: rolling `64 m x 64 m`, `0.2 m/cell` (`320 x 320` cells)
- Layers: `occupancy`, `semantic_id`, `elevation`, `roughness`
- QoS: reliable, transient-local, keep-last depth 1

## Global overview

- Topic: `/Car/T3/mapping/global_overview`
- Type: `nav_msgs/msg/OccupancyGrid`
- Frame: `map`
- Resolution: `1.0 m/cell`
- Period: `20 s`, and only when the map revision changed
- QoS: reliable, transient-local, keep-last depth 1

## Regional high-resolution query

- Service: `/Car/T3/mapping/get_grid_map`
- Type: `grid_map_msgs/srv/GetGridMap`
- Maximum default response geometry: 1,000,000 cells

The caller supplies the map-frame centre, requested length and optional layer
names. An empty layer list requests every available layer. Runtime consumers
should normally request only the layers they use.

Example:

```bash
ros2 service call /Car/T3/mapping/get_grid_map \
  grid_map_msgs/srv/GetGridMap \
  "{frame_id: map, position_x: 0.0, position_y: 0.0, length_x: 30.0, length_y: 30.0, layers: [occupancy, semantic_id, elevation]}"
```

## Persistent global map

- Save service: `/Car/T3/mapping/save_grid_map`
- Type: `std_srvs/srv/Trigger`
- Live revision topic: `/Car/T3/mapping/global_map_revision`
- Revision type: `std_msgs/msg/UInt64`
- Compact delivery file: `<output_dir>/global_grid_map.sqlite3`
- Direct dense matrix file: `<output_dir>/global_grid_map.npz`
- Lossless reconstruction snapshot: `<output_dir>/_internal/grid_map_snapshot/map_index.yaml`

Standard launch scripts create `<output_dir>` below
`$HOME/P3/output/` with a timestamped run name. An explicit
`output_dir:=...` launch argument still overrides this default.

The SQLite file is updated during mapping by a background writer. Each map
revision changes only affected compressed blocks in one SQLite transaction, so
concurrent readers see either the previous complete revision or the next
complete revision, never a partially updated map. The revision topic advances
only after the matching database transaction commits.

The SQLite file stores finalized map layers as compressed, indexed 256-cell
blocks. Blocks are internal storage only and are not exposed as ROS topics.
Elevation is quantized to centimetres; other quality layers use documented
fixed-point encodings in the database metadata. The lossless snapshot remains
the authoritative source when fusion reconstruction is required.

The NPZ file contains the current complete tile-aligned rectangle as directly
readable two-dimensional arrays. A coalescing background writer atomically
replaces the file when map revisions change, without waiting in the localisation
callback. Manual save and orderly shutdown wait for the newest revision.

Existing snapshots can be converted with:

```bash
python3 scripts/export_compact_global_map.py \
  <snapshot>/map_index.yaml global_grid_map.sqlite3
```

## Feature-match debug image

- Topic: `/Car/T2/feature_matching/visualization`
- Type: `sensor_msgs/msg/Image`
- QoS: best-effort, volatile, keep-last depth 1

Rendering is subscriber-gated and reuses matches already computed by the visual
frontend; it does not run an additional RoMa inference. P3 publishes the image,
while the topic remains part of the Task 2 feature-matching delivery interface.
