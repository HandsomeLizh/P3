# Architecture

> Runtime baseline: 2026-08-21. Generic node declarations remain conservative compatibility fallbacks; the production T5 contract is defined by `t5_roma_vo.launch.py` or `t5_fastlivo2.launch.py` and their launch overrides.

```text
batch_organized / rosbag / real-time ROS 2 sensors
                  |
                  +---- stereo images + optional IMU -> RoMa StereoSLAMv2 -> VO pose
                  |                                      |
                  |                                      +---- visual keyframes / stereo evidence
                  |
                  +---- raw LiDAR + IMU ----------> external FAST-LIO2 ---- /Odometry
                  |                                      |
                  |                                      +---- LIO pose
                  |
                  +---- LiDAR + IMU + RGB10 ------> FAST-LIVO2 ---- /aft_mapped_to_init
                  |
                  +---- optional Task 2 mask: /task2/seg_mask
                                                         |
                          +------------------------------+------------------+
                          | localisation selector                           |
                          |                                                  |
                          | vo      : direct VO (default)                   |
                          | lio     : direct external LIO                    |
                          | vo_lio  : GTSAM fixed-lag VO/LIO fusion          |
                          +------------------------------+------------------+
                                                         |
                                             selected base_link pose
                                                         |
                   +-----------------------+----------------------+-------------------+
                   |                       |                                          |
             stereo geometry       PointCloud2-assisted stereo          direct PointCloud2
                (default)             depth (optional)              geometry + local surface
                                                                            occupancy
                                  time/frame/extrinsic/deskew contract gate
                   +-----------------------+----------------------+-------------------+
                                                         |
                                             3D points + semantics
                                                         |
                                      sparse tiled global 2.5D fusion
                                                         |
             +----------------+----------------+----------+-----------------+
             |                |                |                            |
          Odometry       local GridMap       global outputs/files      debug products
```

## Compatibility boundary

`localization_mode=vo` and `fusion_backend=none` are the defaults. In that
configuration no external odometry is subscribed, GTSAM is not imported, and
the established RoMa behaviour is retained. All three localisation modes feed
the same receiver-facing Task 3 topics and message types.

`third_party/RoMa/` is treated as a frozen localisation dependency for this
architecture pass. Cleanup and feature work are confined to the ROS 2 adapter,
2.5D mapping, LiDAR assistance, provenance archive, launch and deployment
layers. The adapter may consume RoMa's existing public state, but it must not
silently change RoMa keyframe ownership or algorithm behaviour.

Machine-specific launch locations are injectable through `T3_ALGORITHM_ROOT`,
`T3_DATA_DIR` and `T3_OUTPUT_DIR`; explicit launch arguments retain precedence
and the previous defaults remain compatibility fallbacks.

Workspace activation has one idempotent owner in `scripts/enter_workspace.sh`.
The current deployment deliberately uses ROS system Python `/usr/bin/python3`;
it loads ROS 2, the colcon install space, the bundled RoMa source and local model
cache without activating a Python virtual environment. Wrapper scripts source
the same file and therefore share one dependency environment.

## Input transport and formal-message adapters

The mapper keeps a stable standard-message boundary: stereo and optional semantic
`sensor_msgs/Image`, `sensor_msgs/Imu`, optional `PointCloud2`, and optional
external `Odometry`. `input_transport.py` owns validation of image/IMU DDS
reliability, independent image/IMU history depths, and exact/approximate
header-stamp synchronization. Direct-node declarations retain exact/Reliable
fallbacks, but production T5 RoMa overrides stereo input to Best Effort depth 1,
approximate synchronization with 30 ms slop, and a separate 10 ms left/right
geometry gate. Semantics are disabled by default and, when enabled, use the
Reliable `/task2/seg_mask` interface.

Task 1's formal multimodal custom message and Task 2's
`SceneSemanticModel.msg` are not defined completely in the available interface
documents. Once their real packages are delivered, thin boundary adapters
must unpack them into the standard topics above. Custom field parsing must not
be added to the mapper callback or guessed from message names. Startup logs and
passive receipt diagnostics expose effective topics, QoS, timestamps, and
stalls without copying image payloads; diagnostic callbacks are registered
after the authoritative synchronizer and cannot preempt it.

The synchronized image boundary has an optional completion contract for
completion-paced offline tools. When `publish_input_processed=true`, every
synchronized callback publishes the original acquisition stamp as a
Reliable/Volatile `std_msgs/msg/Header` on
`/Car/T3/debug/input_processed`. A receipt means
"consumed with an explicit outcome", not "valid pose": tracking failure,
localisation unavailability, exceptions and deliberate rejection remain
separate bounded counters in `input_processing_summary.json`.  The two
dataset practice launchers use this receipt to wait for the real mapper ROS
graph, keep at most two image frames in flight, fail-stop on timeout, and
write `dataset_playback_summary.json`.  The standalone dataset publisher
keeps its old non-lossless default. It also keeps `publish_imu=true` by
default. The two practice launchers map that gate to the existing mapper
`enable_imu` switch: when a saved-LIO/direct run explicitly disables the
mapper's unused visual-IMU path, the dataset process skips IMU file loading,
publisher creation and event replay as well. This is a dataset-wrapper
ownership rule only; an independently running FAST-LIO2 process still owns
and consumes its raw IMU stream. No worker, latest-frame slot or image
payload copy was added; the RoMa/map callback remains serial.  Real sensors
and standard `ros2 bag play` cannot be retroactively throttled by this local
dataset contract, so their source counts and sustained compute rate still
require separate integration evidence.

Dataset completeness is defined over an explicit selection pipeline, not over
an ambiguous directory file count. `StereoPairingStats` records raw left/right
rows, shared pairing candidates, pairs eligible under the configured stereo
exposure-skew gate, and frames finally selected after `max_frames`. The playback
summary names its scope as
`selected_frame_count_after_pairing_and_configured_filters` and embeds all four
counts plus exclusion reasons. Pair eligibility is computed by the same
`batch_organized` loader used by trajectory evaluation. This is diagnostic
only: it does not widen the default 20 ms stereo gate, change publication, or
add a second pairing policy. In the current `2025-06-26-23-01-34` evidence the
pipeline is 157 shared candidates -> 34 skew-eligible pairs -> 34 selected
pairs, so 34/34 consumption is complete while 157-frame stereo coverage is not.

At ROS 2 Python primitive-sequence boundaries, the dataset simulator and map
publisher materialise contiguous typed arrays (`uint8`, `int8`, `uint32` and
`float32`) instead of assigning generic Python byte/list sequences. This avoids
generated setter loops over every image, cloud or grid element. It adds no
queue, cache or retained payload, and changes neither the `.msg` definitions nor
the serialized CDR bytes. A real Task 1 driver should use the equivalent typed
buffer path; the mapper does not require a private transport protocol.

The stable rectangular map schemas have one pure construction boundary in
`map_output_message.py`. It converts a detached `SemanticMapWindow` into either
an `OccupancyGrid` or a complete eight-layer `IncrementalSemanticMap`, using the
same typed buffers. The functions own no publisher, counter, queue, cache or
map state. The node still owns update-id progression, output-mode selection,
publisher calls and their order; `full_tiles` session/commit handling remains
in the transfer path. This boundary therefore changes neither receiver-facing
topics and message types nor partial-failure/retry semantics.

`sensor_msgs_py` is intentionally a test-only dependency. Production debug
cloud construction is owned by `pointcloud_utils.py`; the external helper is
kept only to build an independent frozen legacy CDR reference. The full
development installer still installs it so the regression suite is runnable,
while the package manifest no longer presents it as a runtime requirement.

Formal pose transport has the same narrow boundary in
`pose_output_message.py`. `OdometryOutputModel` remains the stateful owner of
child-frame pose, velocity and covariance values; two pure functions only
encode those values into `Odometry` and its matching `TransformStamped`. The
node still owns the publisher, TF enable switch and broadcaster, Path history,
TUM files, correction rebuilds and all output ordering. Frozen CDR tests and a
live same-stamp comparison require Odometry and TF to remain identical to the
previous wire representation.

The remaining large mapper methods are retained deliberately after a
state-ownership review. Initialization, the serial synchronized callback,
shutdown ordering, Path history and TUM files share lifecycle and correction
timing; moving them behind a new stateful manager would add ownership without
removing work from the hot path. Four proven-unused internal convenience APIs
were removed instead. `RosImuBuffer.get_delta_R_cam()` is explicitly retained:
the frozen RoMa frontend calls it dynamically for the PnP rotation initial
guess even though there is no self-owned call site.

The serial mapper callback is profiled with five fixed stages: input decode,
visual tracking, localisation/pose publication, map update/publication, and
terminal bookkeeping. Each stage keeps only a bounded 2048-duration window;
no sensor payload or dynamic stage key is retained. Stage-accounting failure
falls back to the established total-duration ledger before the authoritative
completion receipt is published. This is additive diagnostics, not a ROS
interface or scheduling change. Current x86 evidence places about 82--87% of
classified callback time in the frozen RoMa visual-tracking dependency.

The optional map debug `PointCloud2` preserves the established finite-point
and global-linspace selection after conversion to the map frame. Selection is
performed before semantic lookup, and the selected `x/y/z/semantic` rows are
materialised as one contiguous little-endian float32 matrix rather than Python
tuples. Frozen-reference tests require byte-identical CDR output, so this path
cannot silently change point count, order, fields, or values. GridMap,
IncrementalSemanticMap, tile sessions, and snapshots do not depend on this
debug-cloud optimization.

FAST-LIO2 is deliberately a separate ROS 2 process with an isolated build area inside
`third_party/fastlio2/`; it is shipped under the same top-level `t3_gridmap_v2_ws/`.
The selected implementation is the community-maintained
[`Ericsii/FAST_LIO_ROS2`](https://github.com/Ericsii/FAST_LIO_ROS2), pinned
locally at commit `2fffc570a25d0df172720bac034fbdb6a13d2162`. The official organisation
also hosts a [`ROS2` branch](https://github.com/hku-mars/FAST_LIO/tree/ROS2), but
that branch's README identifies Ericsiii as its fork maintainer; it is fetched
locally as `hku-mars/ROS2` for comparison rather than treated as an independent
official-native implementation.
The only coupling is the standard `nav_msgs/msg/Odometry` boundary. This keeps
the LIO implementation replaceable and prevents its build dependencies and
licence from leaking into the Task 3 package.

LiDAR map geometry has a second, explicit standard-message boundary owned by
`lidar_temporal_alignment.py`. It does not infer whether a cloud has already
been deskewed. `none` is the bit-compatible path; `upstream_deskewed` treats
the cloud as one rigid scan at its header time; `per_point` requires an
explicit scalar field/unit whose value is relative to `header.stamp`. The two
corrected modes require exact/interpolated external poses with valid samples
on both sides. Nearest-pose fallback is forbidden at this boundary.

For a point acquired at `t_i`, the component returns coordinates in the same
sensor axes at image time `t_ref` using

```text
T_Sref_Si = inverse(T_C_S) * inverse(T_W_C(t_ref))
             * T_W_C(t_i) * T_C_S
```

The cloud-header sensor origin is transformed by the same chain and remains
the direct-LiDAR occupancy-ray origin. `upstream_deskewed` deliberately ignores
all point-time-like fields, preventing a FAST-LIO2 body cloud from being
deskewed twice.

`lidar_message_buffer.py` separately owns the concurrent message-state part:
timestamp ordering, duplicate replacement, bounded retention, successor
waiting, nearest selection and stream-tail timeout. It retains PointCloud2
objects by reference and creates no worker thread. The ROS node validates the
incoming frame/header and delegates one `add()`/`nearest()` operation; it no
longer owns a parallel deque/Condition implementation. The buffer waits until
the stream reaches image time before choosing the true predecessor/successor
nearest cloud; only at a stream tail may a valid past predecessor be accepted
after timeout.

`stereo_observation_builder.py` owns the read-only visual observation
boundary: visible keyframe sampling, plain stereo observations,
PointCloud2-assisted stereo observations, and the configured near-stereo
fallback. It reads one immutable RoMa keyframe snapshot and returns one
immutable map observation. The ROS node retains cloud timing/extrinsics, mode
routing, retry/fail-stop, map transactions, logging, and publication; there is
no parallel observation implementation or new queue/thread.

`visual_tracking_adapter.py` is the single mutable visual-frontend call
boundary. It invokes RoMa once, validates the returned 4x4 pose, reads the
public tracking status, and converts visual exceptions into a deterministic
failed status only when external localisation is allowed to continue. It owns
no history or ROS/map state. Pure-VO failure admission, external pose queries,
localisation selection, transactions, and publication remain visibly ordered
in the synchronized node callback.

Mapper runtime switches are declared by the node and exposed once through
`launch_options.py`; switch values are deliberately not duplicated in
`semantic_map.yaml`. This prevents parameter-file precedence from defeating an
explicit launch override. In particular, `enable_imu=false` creates neither an
IMU subscription nor an IMU callback group, while the default `true` keeps the
established topic and behaviour. A static wiring contract checks parameter
declaration/use, launch/default parity, endpoint ownership, and callback
registration order without adding a runtime configuration registry.

## Localisation modes

| Mode | Backend | Pose authority | Failure behaviour |
|---|---|---|---|
| `vo` | `none` | RoMa + gyro rotation initial guess | Existing visual behaviour |
| `lio` | `none` | External `/Odometry` | Reject a frame if no time-valid LIO pose exists; RoMa failure does not kill LIO |
| `vo_lio` | `gtsam_gp` | Local fixed-lag factor graph | Robust relative VO/LIO factors; optional VO-only continuation during short LIO dropout |

The GTSAM backend stores a `Pose3` and six-dimensional body-tangent velocity
at each support state. Its optional continuous-time WNOA prior uses

```text
Q(dt) = [[dt^3/3 Qc, dt^2/2 Qc],
         [dt^2/2 Qc, dt Qc]]
```

which follows the Gaussian-process prior form used by gnssFGO. It is a local
fixed-lag fusion smoother, not itself a global loop-closure/pose-graph
replacement. A separate, opt-in global layer is described below; the two
estimators have deliberately different ownership and do not feed corrected
states back into each other.
The audited reference clone is `/opt/reference/gnssFGO`, ROS 2 branch
commit `012b95296ac20b800ecb3c36e3c95229df05ae51`; its complete GNSS framework is
not linked into this workspace. Raw IMU is intentionally not added again
because FAST-LIO2 has already
consumed it. The existing RoMa IMU path remains limited to angular-velocity
integration for a visual PnP rotation initial guess.

RoMa is not loaded for the one combination that consumes none of its output:
`lio + direct PointCloud2 + lidar_fallback_to_stereo=false`. That path keeps
the synchronized images because camera semantics are still projected onto the
cloud, but selects map keyframes from accepted LIO poses (the first valid pose,
then each count divisible by `keyframe_interval`) and does not import or run
the visual model. Every other combination still needs visual pose, visual
keyframes, stereo geometry,
or an explicit stereo fallback and therefore keeps the established RoMa path.
External modes continue to isolate genuine visual failures from the LIO pose
stream.

## Optional global loop-closure layer

Global correction is disabled by default and is accepted only for the external
`lio` and `vo_lio` modes with validated LiDAR geometry. The candidate source is
selectable as isolated MPRF, local Scan Context, or both. MPRF owns retrieval
only: SALAD/DINOv2/SONATA can propose at most three historical candidates, but
no appearance result can directly become a graph factor. Every candidate must
pass bidirectional metric LiDAR ICP, overlap/uniqueness/RMSE/cycle-consistency
gates, and the configured temporal confirmation count.

Accepted constraints enter a project-owned robust iSAM2 pose graph. Its
odometry edges are frozen local increments and its loop edges use bounded
measurement uncertainty plus a Huber kernel. The resulting `global <- local`
correction is applied when producing the global Path, TUM trajectory, and map
keyframe poses. It is never fed back into FAST-LIO2, RoMa, or the local
fixed-lag smoother; this prevents the same relative measurement from being
counted twice.

MPRF model inference runs in a separate Python/CUDA process behind a strict
NPZ/JSON protocol. The ROS client uses a bounded latest-first queue, keeps local
odometry non-blocking, and polls completions from the mapper's mutually
exclusive callback group. A timer continues polling after image input stops,
so a final result cannot remain stranded merely because the robot parked.
Service timeout, malformed output, queue pressure, or process failure increments
diagnostics but cannot stop the local pose/map path.

Only a small geometry LRU is retained in RAM; historical loop geometry and all
map observations remain recoverable from bounded/persistent storage. A global
geometry revision rebuilds Path/TUM and replays every accepted map observation
into a private candidate map. The prior complete map remains live until that
candidate is validated and atomically switched. Full implementation and
deployment details are in `课题三MPRF闭环实现与部署说明.md`.

## External frame contract

The adapter validates both frame IDs and sensor timestamps. The default
FAST-LIO2 contract is:

```text
/Odometry: T_camera_init_body
header.frame_id = camera_init
child_frame_id  = body
```

The `body` child is the IMU/body pose, not automatically the LiDAR pose. With
the column-vector convention `p_target = T_target_source p_source`, Task 3 uses:

```text
T_camera_child = T_camera_lidar * T_lidar_child
T_parent_camera = T_parent_child * inverse(T_camera_child)
```

`T_camera_lidar` comes from `LiDAR_to_Cam_L.RT` plus left-image
rectification. `T_lidar_child` must be the inverse of FAST-LIO2's
`T_body_lidar` extrinsic. External modes refuse to start until the matrix is
explicitly supplied and `lio_child_to_lidar_confirmed=true`; identity is never
silently assumed. Pose interpolation is allowed only when both bracketing
samples pass the configured 0.15 s gate, so an outage is not bridged by a
plausible-looking but stale pose.

External odometry covariance is currently not consumed because its child-frame
covariance has not yet been transformed with the complete SE(3) adjoint and
lever arm. The fusion backend therefore uses explicit configurable VO/LIO
measurement sigmas.

## Receiver-facing Odometry contract

The visual state is estimated at the left camera, but the production T5 launch
supplies an explicitly confirmed `T_camera_base_link`. The receiver-facing
contract is therefore `map -> odom -> base_link`: formal Odometry is
`odom -> base_link`, Path and global map products use `map`, and the rolling
local GridMap uses `odom`. Changing only a frame string remains a startup
error. The transformed child pose feeds Odometry, TF, Path, and
`trajectory_map.tum`, while map evidence and `trajectory_opencv.tum` retain the
camera state. The production profile publishes only
`/Car/T3/localization/odometry`; historical duplicate aliases are available
only when `enable_legacy_output_aliases=true`.

Twist is always expressed in `child_frame_id`, as required by
`nav_msgs/Odometry`. Direct VO/LIO use a zero-look-ahead backward pose
difference; the GP fusion velocity takes precedence when available. The first
sample, a non-monotonic/long timestamp gap, or a geometry correction resets the
difference. Missing calibrated pose/twist posteriors are represented by a
large finite configurable covariance rather than an all-zero false-certainty
matrix. Internal rotation-first tangent covariance is reordered to ROS
`[x,y,z,rx,ry,rz]` at this single output boundary.

## Map evidence and publication

PointCloud2 map input is a separate switch from LIO localisation and must pass
frame, timestamp, temporal-contract and extrinsic gates. Corrected clouds are
expressed at the image/keyframe time before semantic projection or map fusion,
so the archived observation and its selected keyframe pose share one reference
time. Missing point time, missing bracketing poses, an unsafe time magnitude,
or an inconsistent calibration rejects the observation and enters the existing
retry/fail-stop completeness path unless the operator explicitly enabled the
stereo fallback. `direct` mode preserves the established
path: its calibrated sensor origin supplies occupancy rays, points visible in
the left camera receive image semantics, and 360-degree geometry outside the
camera view remains semantic-unknown. For direct-LiDAR observations only, a
local k-neighbour surface estimate also supplies an independent endpoint state:
reliable upward ground is free, reliable non-ground is occupied, and sparse,
linear, or rough support remains unknown. Explicit image semantic free/obstacle
classes take precedence, so geometric occupancy never changes the semantic
label. The observation source is archived, and corrected-map rebuilds apply
the same classifier and snapshot-stored thresholds as online fusion.

`assist` mode projects a nearest-surface z-buffer into the rectified image,
preserves reliable near stereo depth, corrects only supported far stereo rays,
and rejects unsupported far map points. Its occupancy origin remains the left
camera, and it does not modify RoMa tracking/PnP depth or enable direct-LiDAR
endpoint classification. When
external LIO is selected, map-axis alignment uses the same calibrated
LiDAR-to-camera rotation even if raw LiDAR geometry is disabled. Inconsistent
rotations between the LIO and raw-LiDAR calibration paths are rejected.

Pure-stereo map observations apply `stereo_map_max_depth_m` before the shared
point-count cap; the default is an inclusive optical-depth limit of 5 m. This
selection is downstream of RoMa, so localisation retains the original far
features. A keyframe with no trusted map point is an intentional no-evidence
skip: it creates neither a retry entry nor an empty map revision. A zero limit
is the explicit compatibility escape hatch. LiDAR direct range and assisted
PointCloud2 correction remain independently owned policies.

Every valid selected pose triggers the live pose output. Dense map products are
published on a real map revision (or a configured rolling-window cell change),
instead of reconstructing identical payloads as a heartbeat. New map evidence
is archived and fused once per mapping keyframe. A runtime contribution index
records each applied keyframe pose, every tile touched by its endpoints and
occupancy rays, and the reverse tile-to-keyframe dependency. A local backend,
BA, loop, or external-geometry correction replays every contributor to the
exact old/new affected-tile union. Missing/inconsistent dependencies, a large
correction, a changed dependency footprint, or work comparable to a full
replay falls back to a complete rebuild rather than accepting an approximate
map.

Corrected candidates are built by one single-flight worker while the last
complete map remains live. The ROS callback validates the captured map
revision, geometry revision, and exact pose snapshot before atomically
switching the candidate map, contribution index, archived pending evidence,
and applied revision. Obsolete and failed candidates cannot mutate live state
and remain retryable. Explicit save and shutdown boundaries wait for active
correction work, while ordinary image callbacks do not.

Committed observation provenance is stored losslessly as float64/uint8 data in
a transactional SQLite archive under the run output directory. The in-process
LRU holds 64 observations by default, but eviction never removes archive rows;
rebuild streams selected or complete observations in deterministic
`(sensor_timestamp, keyframe_id)` order. A failed
multi-observation disk transaction is rolled back before either the live
provenance index or the published map can change. Historical VO keyframes keep
their existing RoMa ownership semantics; the 64-entry bound applies only to the
mapper's archived map observations, not to RoMa's internal memory. These mapper
storage changes do not subsample GridMap evidence or any published map layer.

Completeness is fail-stop rather than best-effort. A corrected rebuild is
rejected if any archived keyframe lacks a live pose. Likewise, an unavailable
selected LiDAR observation is retained for retry together with the exact
`PointCloud2` instance originally selected; a retry may not silently reselect a
newer cloud after the live DDS history has advanced. While any observation is
unresolved, no newer map revision is published. If the bounded retry-image and
cloud-reference cache is exhausted, the node freezes map output explicitly
instead of evicting old evidence. Setting `lidar_fallback_to_stereo=true` is
the explicit operator choice for continuity with stereo evidence.

At the current 0.2 m resolution, the global map creates sparse 256-cell
(51.2 m) tiles on demand. The production T5 interface publishes a rolling
64 m x 64 m local GridMap in `odom`, a revision-triggered complete global
elevation GridMap in `map`, and a 1.0 m global OccupancyGrid overview. The
legacy `IncrementalSemanticMap`, `SemanticMapBatch`, and full-tile replay
implementation remains available for compatibility testing but is disabled by
the production T5 configuration (`publish_incremental_map=false`) and is not a
new receiver-facing primary interface. Complete multi-layer state is delivered
through the live NPZ/SQLite files and bounded regional query service.

## Snapshot commit boundary

`TiledSemanticMapManager` owns map-format serialization and exact per-layer
validation. `snapshot_store.py` separately owns filesystem transactions. It
writes a private version under `.versions/.staging-*`, fsyncs and validates
that version, renames it to an immutable `snapshot-*` directory, then replaces
the root `map_index.yaml` as the only commit point. A failed write, validation,
rename, or Python signal leaves the prior manifest and payloads untouched.

The current and immediate previous committed versions are retained. Legacy
flat `tiles/` snapshots remain readable and are kept through the first
versioned migration save. This persistence protocol does not change ROS 2
topics or the normal fusion/publication path. One mapper is the authoritative
writer for a given output directory; concurrent mapper processes must use
separate `output_dir` values.

Snapshot compression and validation still run synchronously while holding the
map lock. This is separate from the background corrected-candidate worker and
remains a long-map performance item; the lock preserves one coherent version
and cannot silently omit accepted evidence.
