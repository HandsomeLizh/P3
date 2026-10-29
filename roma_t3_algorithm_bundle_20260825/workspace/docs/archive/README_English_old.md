# Task 3 Semantic Elevation GridMap V2 Workspace

完整中文使用与集成文档：
[课题三定位与2.5D地图使用说明.md](课题三定位与2.5D地图使用说明.md)

新机器环境、依赖、下载、编译与上车验收：
[课题三新机器部署与FAST-LIO2编译说明.md](课题三新机器部署与FAST-LIO2编译说明.md)

复制整个工程前生成 Orin/x86 离线镜像和一键安装包：
[课题三离线部署教程](offline_deploy/README_离线部署.md)

当前数据集上的定位、深度、地图和实时性量化结果：
[课题三定位与2.5D地图精度测试报告.md](课题三定位与2.5D地图精度测试报告.md)

可选 MPRF 闭环的算法边界、模型/源码、服务部署、ROS 2 开关和真实序列结果：
[课题三MPRF闭环实现与部署说明.md](课题三MPRF闭环实现与部署说明.md)

已确认问题、处理顺序和逐项 review 结论：
[课题三定位与2.5D地图待处理问题记录.md](课题三定位与2.5D地图待处理问题记录.md)

当前代码、架构、模式、固定接口、完整输出和外部验收边界的最终复核：
[最终整体架构、接口与运行 review 报告](accuracy_results/final_audit_20260815/最终整体架构接口运行review报告.md)

逐条对应两份项目文档、区分本地通过与正式外部验收的完成性矩阵：
[项目要求完成性矩阵](accuracy_results/final_audit_20260815/项目要求完成性矩阵.md)

复制到新机器后用于重建 6 个自有包并精确复现 365 项回归、静态检查和接口哈希的
统一入口：`bash scripts/run_quality_gates.sh`。

课题一/二实时自定义消息的薄适配层、字段配置和联调顺序：
[课题一课题二接口快速适配模板说明_fixed.md](课题一课题二接口快速适配模板说明_fixed.md)

LiDAR `direct` 未知语义终点的占据修正、数据集统计和可复查快照：
[P0-6 LiDAR direct 占据终点修正与测试报告](accuracy_results/p0_6_lidar_endpoint_occupancy_20260815/P0-6_LiDAR_direct占据终点修正与测试报告.md)

纯双目远深度污染、5 m 地图证据门、同观测 A/B 和 ROS 2 完整性 review：
[P0-7 纯双目可信深度门与 review 报告](accuracy_results/p0_7_stereo_depth_gate_20260815/P0-7_纯双目可信深度门与review报告.md)

`full_tiles` 会话/批次/reset/完整重放协议与 ROS 2 证据：
[P0-5 全局瓦片批次协议与测试报告](accuracy_results/p0_5_full_tile_protocol_20260815/P0-5_全局瓦片批次协议与测试报告.md)

快照原子提交、上一版回退、中断/失败注入和 ROS 2 保存证据：
[P1-2 地图快照原子保存与测试报告](accuracy_results/p1_2_atomic_snapshot_20260815/P1-2_地图快照原子保存与测试报告.md)

后端/回环修正的精确局部重建、后台候选图、原子切换和流式瓦片证据：
[P1-1 后端修正局部重建与后台发布测试报告](accuracy_results/p1_1_map_correction_20260815/P1-1_后端修正局部重建与后台发布测试报告.md)

LiDAR 真正最近帧、FAST-LIO2 已去畸变契约、逐点时间补偿和诚实精度对照：
[P1-3 LiDAR 时间对齐与运动补偿测试报告](accuracy_results/p1_3_lidar_temporal_20260815/P1-3_LiDAR时间对齐与运动补偿测试报告.md)

P2 第一处结构拆分、LiDAR 缓存等价性、逐层地图 A/B 和微基准：
[P2-1 LiDAR 消息缓存组件拆分与测试报告](accuracy_results/p2_1_lidar_buffer_20260815/P2-1_LiDAR消息缓存组件拆分与测试报告.md)

P2 第二处结构拆分、双目/assist 观测等价性、三条 ROS 路径和性能 review：
[P2-2 双目与点云辅助观测构建拆分测试报告](accuracy_results/p2_2_stereo_observation_20260815/P2-2_双目与点云辅助观测构建拆分测试报告.md)

P2 第三处结构拆分、视觉异常契约、VO/LIO 回放和逐层严格 A/B：
[P2-3 视觉前端适配器拆分与测试报告](accuracy_results/p2_3_visual_tracking_adapter_20260815/P2-3_视觉前端适配器拆分与测试报告.md)

P2 第四步参数/端点审查、IMU 开关覆盖修正和真实 ROS 图对照：
[P2-4 参数初始化与 ROS 端点审查测试报告](accuracy_results/p2_4_initialization_wiring_20260815/P2-4_参数初始化与ROS端点审查测试报告.md)

P2 第五步同步输入完整性、无损回放背压和真实处理倍率：
[P2-5 输入完整性与无损回放测试报告](accuracy_results/p2_5_input_backpressure_20260815/P2-5_输入完整性与无损回放测试报告.md)

P2 第六步 ROS 2 大数组构造瓶颈、线协议等价性和真实 A/B：
[P2-6 ROS 大数组 typed-buffer 优化与测试报告](accuracy_results/p2_6_throughput_profile_20260815/P2-6_ROS大数组typed-buffer优化与测试报告.md)

P2 第七步有界回调分项、瓶颈定位和调试 PointCloud2 线协议等价优化：
[P2-7 运行分项与调试点云优化测试报告](accuracy_results/p2_7_runtime_stage_profile_20260815/P2-7_运行分项与调试点云优化测试报告.md)

P2 第八步正式矩形地图消息纯构造边界、CDR 等价和真实 ROS 2 完整性验证：
[P2-8 正式地图消息构造边界拆分测试报告](accuracy_results/p2_8_map_message_boundary_20260815/P2-8_正式地图消息构造边界拆分测试报告.md)

P2 第九步 `sensor_msgs_py` 真实用途、最小运行依赖和完整测试环境边界：
[P2-9 sensor_msgs_py 测试依赖边界报告](accuracy_results/p2_9_test_dependency_boundary_20260815/P2-9_sensor_msgs_py测试依赖边界报告.md)

P2 第十步正式 Odometry/TF 纯消息构造、CDR 等价和三路实时对账：
[P2-10 Odometry 与 TF 消息构造边界报告](accuracy_results/p2_10_pose_message_boundary_20260815/P2-10_Odometry与TF消息构造边界报告.md)

P2 第十一步内部死 API 清理、RoMa 动态接口核对与高耦合边界复审：
[P2-11 内部死接口清理与高耦合边界复审报告](accuracy_results/p2_11_internal_api_cleanup_20260815/P2-11_内部死接口清理与高耦合边界复审报告.md)

P2 第十二步工作区重复激活、显式 venv 拒绝与包装脚本启动修正：
[P2-12 工作区重复激活与包装脚本启动修正报告](accuracy_results/p2_12_workspace_activation_20260815/P2-12_工作区重复激活与包装脚本启动修正报告.md)

P2 第十三步目标序列完整性、分项性能、精度有效性与输入选择语义：
[P2-13 目标序列完整性性能精度与输入选择修正报告](accuracy_results/p2_13_full_target_vo_20260815/P2-13_目标序列完整性性能精度与输入选择修正报告.md)

P2 第十四步同步主序列 231 帧全量无损完整性、稠密性能与精度：
[P2-14 同步主序列全量完整性性能精度报告](accuracy_results/p2_14_sync_full_vo_20260815/P2-14_同步主序列全量完整性性能精度报告.md)

P2 第十五步分层调试图/全局调试点云 2×2 A/B、正式地图无损对账和嵌入式建议：
[P2-15 派生调试输出性能与信息完整性报告](accuracy_results/p2_15_debug_output_ab_20260815/P2-15_派生调试输出性能与信息完整性报告.md)

P2 第十六步原子 2.5D 融合热点、八数组逐字节等价和真实 ROS 2 完整性 review：
[P2-16 原子地图融合等价优化与 review 报告](accuracy_results/p2_16_atomic_fusion_20260815/P2-16_原子地图融合等价优化与review报告.md)

P2 第十七步清理只被测试调用的同步瓦片发布旁路，并将契约改为覆盖真实流式路径：
[P2-17 单一流式瓦片发布路径清理与 review 报告](accuracy_results/p2_17_single_tile_stream_20260815/P2-17_单一流式瓦片发布路径清理与review报告.md)

P2 第十八步让纯 `lio + direct` 不再加载或逐帧运行无消费者的 RoMa，同时保持
其余模式和正式接口不变：
[P2-18 LIO direct 视觉旁路与 review 报告](accuracy_results/p2_18_lio_direct_bypass_20260815/P2-18_LIO_direct视觉旁路与review报告.md)

P2 第二十步在最长完全同步序列上验证 direct LIO 的 231/231 无损输入、完整输出、
持久化和性能，并定位到数据集包装器仍串行回放无消费者原始 IMU：
[P2-20 direct LIO 同步主序列全量稳定性性能报告](accuracy_results/p2_20_full_sync_lio_direct_20260815/P2-20_direct_LIO同步主序列全量稳定性性能报告.md)

P2 第二十一步用默认兼容薄开关跳过该精确配置下无消费者的原始 IMU 回放，完成
同序列完整性、SQLite 逐字节、轨迹时间契约和性能 A/B：
[P2-21 direct LIO 无用 IMU 回放清理与 review 报告](accuracy_results/p2_21_skip_unused_imu_20260815/P2-21_direct_LIO无用IMU回放清理与review报告.md)

课题一/二输入边界、QoS、时间同步和联调材料清单：
[课题三ROS2输入接口联调说明.md](课题三ROS2输入接口联调说明.md)

Odometry 坐标、速度、协方差及车体外参安全开关：
[课题三Odometry位姿接口说明.md](课题三Odometry位姿接口说明.md)

当前状态是“代码回归与 x86 启动通过，可继续联调”，不是“精度验收通过”。
最新代码级回归为 365/365 通过（静态契约 43、冻结 RoMa 数值 29、
自研 ROS/定位/地图层 267、隔离 MPRF 运行时 11、数据解析 7、
课题一/二适配字段契约 8）；短序列定位、地图、LIO 接口和
点云辅助深度已经实测，
标准 Image/Imu 输入已支持可配置 QoS、exact/approximate 同步和停滞诊断，
数据集联合 launch 默认使用就绪门、2 帧在途窗口和逐帧处理回执，
能显式证明发布/同步/终止是否完整。P2-5 当时量到约 0.23～0.25×；消除
ROS 2 Python 大数组逐元素检查，并在不改点、顺序和 CDR 的前提下向量化调试点云后，
短段 x86 实测约 0.42～0.46× 传感器时间。P2-14 又用完全同步的
`23-42-20` 跑完 231/231 帧，无损完整性通过，但稠密全量实测只有
0.285×；回调 P95/最大为 2.240/2.767 s，35 个超过 2 s 的回调全部是地图修订帧。
该长度下 RoMa 和地图段分别占分类时间的 65.71%/33.49%。P2-8 又将 `OccupancyGrid` 与
`IncrementalSemanticMap` 的逐字段构造收敛到两个纯函数，但 publisher、话题、
消息类型、发布顺序和 CDR 字节不变。P2-9 又将仅用于冻结参考测试的
`sensor_msgs_py` 改为 `test_depend`，生产节点在禁止导入它时仍可正常导入。
正式 Odometry/TF 的逐字段构造也已收敛到两个纯函数，但速度模型、
publisher、TF 开关、Path 和 TUM 所有权仍在节点内；真实 3 帧中 Odometry/TF
同时间戳逐分量完全一致。P2-11 又在不增加新抽象的前提下删除 4 个
零消费者的内部 API，同时确认 `get_delta_R_cam()` 是冻结 RoMa 真实调用的
动态边界，必须保留。高耦合初始化、主回调、关闭顺序和 Path/TUM 所有权经复审
不做风格性大拆分。P2-12 修正了“先 source、再运行自带激活的包装脚本”
在默认 venv 缺失时的二次激活误判；对用户显式错误路径的拒绝仍保留。
目标序列 `2025-06-26-23-01-34` 进一步证明原始左/右各 157 行中只有
34 对通过固定 20 ms 双目时差门限；这 34 对已全部发布/回执/同步/终止，
但不能冒充 157 帧完整双目运行。现在回放总结会显式区分原始、候选、时差合格和
最终选中数，轨迹评估也会报告时间空洞及采样路径比例。
同步主序列的新全量轨迹为 231 个位姿，固定尺度 ATE RMSE 为 0.125 m，
9.803 m 上 `RMSE/D=1.276%`；它高于 1% 且距离远小于 1 km，不作精度验收通过结论。
在同序列前 45 帧的 2×2 A/B 中，关闭 5 张派生调试图和 RoMa 全局调试
PointCloud2 将提交后发布尾段从平均 0.367 s/修订降到 0.046 s/修订，
同时仍完整发布 OccupancyGrid、9 层 GridMap 和 8 层增量地图。因此嵌入式/正式运行
建议显式设两个调试开关为 `false`，但为了旧端兼容尚未改参数默认值。
继续对自研原子融合链做等价优化后，同一批 18 条保存观测的八组瓦片数组
逐字节一致，隔离回放融合耗时下降 57.0%。新的 45 帧 ROS 2 运行为
45/45 无损，地图更新+发布总时间下降 37.1%，回调 P95/最大值为
1.333/1.945 s。但真实消费仍只有 0.492×，冻结 RoMa 视觉段占 87.4%，
所以仍不标记为 1× 实时通过。
P2-17 又删除了生产零调用的同步 `full_tiles` 旁路，现在节点级测试和
安装态都只走“冻结快照→每定时器一瓦片→COMPLETE”的唯一路径；6/6 帧后晚加入
接收端已收到 `reset=true` 的 3 瓦片完整重放，接口与默认 `fixed` 均未改。
P0-7 现已把纯双目默认地图证据限定为相机 `Z<=5 m`，但不向 RoMa
定位回写。同 90 条观测/同位姿重放中，旧图表面 X 延伸到 103.85 m，
新默认为 8.45 m，融合时间降低 63.9%。45 帧安装态运行为 45/45 无损、
0 失败，SQLite 中最大存档深度 4.99974 m，晚加入仍收到原 9 层 GridMap。
P2-18 进一步让 `lio + direct + lidar_fallback_to_stereo=false` 不再加载或
逐帧运行 RoMa；其余模式不变。同一 6 帧基线的两份 LIO 轨迹逐字节一致，
地图仍为 2 次修订/4 瓦片，平均回调由 1.290 s 降到 0.369 s。首次地图提交
仍出现 1.865 s 尖峰，因此只确认消除了无消费者的视觉成本，不宣称稳定实时。
P2-20 随后以最长完全同步序列全量运行 direct LIO：231/231 输入明确终态、
47 次地图修订和 194,290 个归档点完整，但联合数据集链只有 0.504×，根因是
单线程包装器还串行回放 mapper 不消费的 23,923 条原始 IMU。P2-21 只让 practice
数据集 launch 的原始 IMU 回放跟随既有 `enable_imu`，默认仍为 `true`，不改实时
ROS/FAST-LIO2 的 IMU 接口。同序列关闭无消费者回放后仍为 231/231、47 次修订，
SQLite 47 行及 BLOB 逐字节相同；联合倍率达到 1.164×，回调均值/P95/最大为
0.109/0.568/0.994 s，峰值 RSS 下降 14.23%。小量栅格差异来自恢复正确的 LIO
时间插值，不是删点；该内部参考也不能替代独立真值精度。
因此 direct LIO 的本机数据集链已证明超过 1×，需要视觉的模式仍未达到 1×；
两类路径都还需要在真实持续输入和 Orin/Humble 上复验。
但课题一/二正式自定义 `.msg` 仍需上游提供后再做薄适配；
`D>=1 km`、`T>300 s`、Orin 实时性及独立真值 LIO 精度也仍需实车完成。

最终集成目标是接口协调文档中的 AGX Orin 64G（aarch64、JetPack 5.1.1/
Ubuntu 20.04、ROS 2 Humble、Cyclone DDS）。当前 FAST-LIO2 成功编译是 x86_64
可行性验证，不是 ARM 发布包；Orin 必须使用集成方的 Humble 黄金镜像并原生重编。

This is a ROS 2 Humble workspace for the Task 3 standalone validation and
integration-preparation workflow. The current machine has also been built and
tested with ROS 2 Rolling.

It preserves the proven pipeline by default:

- stereo images on independent ROS topics;
- exact sensor timestamps from image filenames;
- independent IMU topic with timestamp-based gyro integration;
- RoMa/StereoSLAMv2 loaded from the bundled, frozen
  `third_party/RoMa/` selected by `scripts/enter_workspace.sh`;
- precomputed or online basic semantic labels;
- real-time Odometry, Path and semantic point cloud.

Localisation is now selected explicitly without changing those public ROS 2
interfaces:

| `localization_mode` | `fusion_backend` | Pose source |
|---|---|---|
| `vo` (default) | `none` (default) | Existing RoMa VO |
| `lio` | `none` | External FAST-LIO2-compatible `nav_msgs/msg/Odometry` |
| `vo_lio` | `gtsam_gp` | GTSAM fixed-lag VO/LIO fusion with optional WNOA GP prior |

The external LIO node remains an independent, replaceable ROS 2 process. The
audited local candidate is the **community-maintained port**:
[`Ericsii/FAST_LIO_ROS2`](https://github.com/Ericsii/FAST_LIO_ROS2),
pinned at commit `2fffc570a25d0df172720bac034fbdb6a13d2162` under the portable,
source-only directory `third_party/fastlio2`. It has been
rebuilt directly from that bundled source and launch/interface checked on the
current machine without algorithm source patches. The official organisation now hosts a
[`hku-mars/FAST_LIO` ROS2 branch](https://github.com/hku-mars/FAST_LIO/tree/ROS2),
but that branch itself identifies Ericsiii as the ROS 2 fork maintainer. It is
available locally as the `hku-mars/ROS2` comparison remote. Task 3 depends only
on standard Odometry, so either implementation can be substituted.

It additionally publishes a formal multi-layer 2.5D map:

```text
/Car/T3/mapping/grid_map
grid_map_msgs/msg/GridMap
```

Layers:

- `elevation`
- `elevation_variance`
- `occupancy`
- `semantic_id`
- `semantic_confidence`
- `observation_count`
- `height_range`
- `roughness`
- `color`

The project-specific output remains available as:

```text
/Car/T3/semantic/incremental_map
t3_interfaces/msg/IncrementalSemanticMap
```

The existing ROS 2 topics and `IncrementalSemanticMap` definition are
unchanged. Explicit `full_tiles` additionally exposes a companion
`/Car/T3/semantic/map_batch` transaction topic and full-replay Trigger service;
neither endpoint exists in the default `fixed` mode. Internally, the global
map grows as sparse 25.6 m tiles (256 cells at 0.10 m). The standard
`GridMap`, occupancy and debug-image topics remain a dense 60 m window. The
default `fixed` mode reproduces the original fixed 600x600 geometry
with origin (-30, -30) m, so existing consumers do not need a new tile or
moving-window interpretation.

Map payloads are now published on actual map revisions by default, while
Odometry remains per valid pose. All map topics keep transient-local QoS;
`fixed`/`rolling_window` retain the latest complete rectangle and `full_tiles`
actively streams a frozen complete replay to a new payload endpoint. Set
`publish_unchanged_map_outputs:=true` only for a receiver that explicitly
depends on the previous per-pose duplicate-map cadence.

Backend/BA/loop geometry corrections now build a detached candidate through an
exact keyframe-to-tile dependency index. Local corrections replay only all
contributors to the affected tile union; unsafe or non-cheaper cases fall back
to a complete replay. One background worker keeps the last complete map live,
and revision/pose checks gate the atomic switch. Normal and full-replay
`full_tiles` batches both stream one frozen tile per timer callback.

Pure-VO mapping now has an independent post-loss admission gate. A failed
visual frame closes new map-evidence admission; recovery must satisfy the
configured inlier, pose-continuity and consecutive-frame checks before a new
keyframe is persisted. Odometry publication is intentionally unchanged, and
external/fused LIO pose modes bypass this visual-pose gate. All controls use
the `map_admission_*` launch/ROS parameter family.

Pure-stereo geometry also has an independent map-evidence depth gate. The
default `stereo_map_max_depth_m=5.0` is applied before the map point cap, while
the original keyframe remains untouched for localisation. Set it to `0.0` only
to reproduce the previous unbounded map-observation behaviour. Rejected far
stereo estimates remain unknown map space; PointCloud2 `direct` and `assist`
keep their own range/correction policies.

PointCloud2 input can be enabled explicitly for the 2.5D map. This switch is
independent of the selected pose source and does not change any output topic or
message type. Two modes are available:

- `direct`: keep the established 360-degree point-cloud geometry path and
  classify reliable local ground/non-ground endpoints as independent
  free/occupied evidence while preserving image semantics;
- `assist`: retain reliable near stereo depth, use projected point-cloud depth
  to correct supported far stereo rays, and reject unsupported far map points.

Map-geometry mode and temporal mode are independent. `lidar_temporal_mode`
has three explicit contracts: `none` keeps the original behaviour,
`upstream_deskewed` trusts FAST-LIO2 point deskew and only aligns the rigid
cloud to the image time, and `per_point` reads an explicitly configured
header-relative point-time field and uses bracketed external LIO poses. The
last two modes fail closed when pose coverage, timing, or calibration is
invalid; they never substitute an unbracketed nearest pose. The ordered cloud
buffer is now an independent ROS-free component. It waits for the first
successor before selecting the true nearest predecessor/successor rather than
accepting the first past cloud inside the gate; the ROS node no longer owns a
second copy of that ordering/waiting state machine.

Direct mode remains the default whenever the optional input is enabled:

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=direct \
  lidar_topic:=/Car/T1/fusion/lidar/points \
  lidar_extrinsics_path:=/path/to/extrinsics.json \
  lidar_expected_frame_id:=velodyne \
  lidar_temporal_mode:=none
```

To assist stereo map depth instead:

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=assist \
  lidar_topic:=/Car/T1/fusion/lidar/points \
  lidar_extrinsics_path:=/path/to/extrinsics.json \
  lidar_expected_frame_id:=velodyne \
  lidar_fallback_to_stereo:=true
```

The feature is off by default. Assisted depth changes map evidence only; it
does not replace RoMa VO depth/PnP or alter the public outputs. See the Chinese
integration document for timestamp gates, frame conventions, fallback
behaviour, tuning parameters and measured proxy-depth results. With the shown
fallback enabled, a missing cloud retains only stereo points inside the
assisted-mode reliable-depth gate instead of restoring unsupported far points.

For deterministic interface testing of the target recording's saved estimator
output (not ground truth):

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_target_lio_replay.sh
```

For rosbag or a real robot, run FAST-LIO2 from raw LiDAR + IMU and let it publish
`/Odometry`. Do not replay the saved trajectory as an accuracy reference. Before
selecting `lio` or `vo_lio`, explicitly provide and confirm `T_lidar_child`;
the usual FAST-LIO2 child frame `body` is the IMU/body frame, not necessarily
the LiDAR frame.

For the recommended production geometry path, consume FAST-LIO2's internally
IMU-deskewed body cloud and still correct its whole-frame offset to the image
time:

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  localization_mode:=lio fusion_backend:=none \
  lio_child_to_lidar_confirmed:=true \
  lio_child_to_lidar_transform:="[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]" \
  enable_lidar_mapping:=true \
  lidar_topic:=/cloud_registered_body \
  lidar_expected_frame_id:=body \
  lidar_extrinsics_path:=/path/to/body_to_left_camera_extrinsics.json \
  lidar_temporal_mode:=upstream_deskewed
```

The target replay script uses `per_point/time/sec` because that archive
contains raw PCD scans with signed offsets of roughly -0.10 to +0.00064 s.
Saved 10 Hz LIO poses are not equivalent to FAST-LIO2's high-rate IMU deskew,
so that replay is an interface/diagnostic test, not a production accuracy
recommendation.

Two alternatives are retained. `rolling_window` keeps the complete 60 m map
message but centres it near the robot. `full_tiles` publishes complete changed
tiles and surrounds the unchanged legacy payloads with a companion
`SemanticMapBatch` BEGIN/COMPLETE transaction. It also replays a frozen full
map to newly matched subscribers. Enable either only after the receiver agrees
to its geometry and batch semantics:

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  incremental_publish_mode:=rolling_window

ros2 launch t3_semantic_mapping mapper_only.launch.py \
  incremental_publish_mode:=full_tiles
```

## Quick start (x86 development baseline)

AGX Orin 不要直接执行下面的通用 apt 流程；先按上述中文部署文档做
JetPack/Humble/PyTorch 兼容预检。
目标机上可用 `scripts/check_deployment_platform.sh orin` 检查平台，用
`scripts/check_roma_inference.py` 执行一对真实图像的 GPU 冒烟测试。

```bash
cd ~/t3_gridmap_v2_ws
bash scripts/install_prereqs.sh
bash scripts/build.sh
source scripts/enter_workspace.sh
bash scripts/check_environment.sh
```

`enter_workspace.sh` 可在同一终端重复 source；各个 `run_*`/check/save
包装脚本会自行再激活，不会把缺失的自动默认 venv 误当成用户配置。

Generate local validation semantics once if needed:

```bash
bash scripts/generate_semantics.sh
```

Run two frames:

```bash
bash scripts/run_live_precomputed.sh \
  max_frames:=2 \
  playback_rate:=1.0 \
  start_delay_sec:=0.0
```

数据集联合 launch 默认 `lossless_playback=true`；`start_delay_sec`
只是额外的最短等待，发布器还会等 mapper 三路订阅和处理回执端点
真实出现。回放结束后联合检查 `dataset_playback_summary.json` 和
`input_processing_summary.json`，不要只看“已播完”日志。

Inspect the formal GridMap in another terminal:

```bash
source ~/t3_gridmap_v2_ws/scripts/enter_workspace.sh
ros2 topic echo /Car/T3/mapping/grid_map --once --field layers
python3 ~/t3_gridmap_v2_ws/scripts/inspect_grid_map.py
```

Save the latest map while the mapper is running:

```bash
bash ~/t3_gridmap_v2_ws/scripts/save_grid_map_now.sh
```

Snapshots are written as a global index plus restorable per-tile files under:

```text
~/t3_gridmap_v2_ws/output/grid_map_snapshot/
```

Always load the root `map_index.yaml`. Tile payloads live in immutable
`.versions/snapshot-*` directories; the root index is replaced only after all
layers have been written, fsynced, and validated. The current and immediately
previous complete versions are retained, so an interrupted later save leaves
the last committed map usable.

## Important coordinate note

`map` uses X forward, Y left, Z up. Pure-vision mode applies the configured
30-degree downward camera pitch. Optional LiDAR mode instead aligns to the
calibrated REP-103 sensor frame by default. Z=0 is the first camera optical
centre, not the ground plane. Set `map_z_offset_m` in
`src/t3_semantic_mapping/config/semantic_map.yaml` to the measured camera
height if ground should be near zero.

This basic semantic segmentation is only a Task 3 standalone substitute. In
system integration, the semantic source should be replaced by the agreed
Task 2 output while keeping the timestamp and coordinate contracts.
