# P3 工作区说明（context.md）

更新：2026-10-29。机器：Orin `yanfa@192.168.100.216`。课题三（T3）定位与建图。

## 1. 目录

```
/home/yanfa/P3/
├── context.md                          本文件
├── lidar_visual_fusion/                主工程（ROS 2 Humble colcon 工作区，git 仓库）
└── roma_t3_algorithm_bundle_20260825/  运行依赖，不要移动：UI 窗口 t3_visual_window、RViz 插件、UE 采集运行时 envx_runtime

/home/yanfa/P3_bak/                     备份与旧输出（见第 6 节），目录结构与 P3 相同
```

### lidar_visual_fusion 内部

| 路径 | 内容 |
|---|---|
| `start_live.sh` | 实时建图入口（建图 + 传感器中继 + UI） |
| `start_simulation_sources.sh` | 启动或复用 UE 采集、车辆桥接、人工控制窗口（domain 10） |
| `stop.sh` / `status.sh` | 停止 / 查看本工作区的运行 |
| `build.sh` | colcon 编译，并记录源码指纹 `build_manifest.json` |
| `save_map.sh` / `export_map.sh` | 保存 / 导出地图 |
| `start.sh`、`start_bag.sh`、`start_long_bag.sh`、`start_short_bag.sh`、`start_voxelmap_short.sh` | 离线 bag 运行入口（`start.sh` 是其余脚本的公共底层） |
| `src/t3_lidar_visual_fusion/` | Python 节点：传感器适配、视觉里程计、门控选源、建图 |
| `src/t3_voxelmap/` | C++ LiDAR 里程计 `voxelmap_node.cpp`（可选 IMU：`include/optional_imu.hpp`） |
| `src/t3_interfaces/`、`src/grid_map_msgs/` | 消息定义 |
| `src/vins/`、`src/camera_models/` | VINS-Fusion，主配置下不启动（二进制仍在构建指纹内） |
| `src/point_lio/` | 已退役，不启动 |
| `scripts/` | `control.py`（启停、指纹检查）、`generate_config.py`（profile → 各节点参数和 `ekf.yaml`）、`fusion.launch.py`、`live_sensor_relay.py`（domain 10→57）、`replay_bag.py`、`env.sh`、`p3_visuals.sh`（UI）、`sim_imu/`（轮速、模拟 IMU、回放、评分） |
| `visual/visual_monitor.py` | UI 右侧的 Tk 高程监视器 |
| `config/` | 运行配置（profile），主配置 `simulation_lidar_360_wheel.yaml` |
| `tests/`、`test_data/` | 单元/集成测试；test_data 只留测试实际引用的输入 |
| `docs/` | 历史开发报告（其中提到的部分旧脚本已移到 P3_bak） |
| `vendor/`、`deps/` | XFeat、LightGlue、VINS-Fusion 源码；私有 ROS 依赖（含 robot_localization） |
| `build/`、`install/` | 编译产物 |
| `results/`、`logs/` | 每次运行的输出和日志（清理后为空） |

### 主要 Python 模块（`src/t3_lidar_visual_fusion/t3_lidar_visual_fusion/`）

| 节点 | 文件 | 作用 |
|---|---|---|
| sensor_adapter | `sensor_adapter.py` | 把相机、LiDAR、IMU 消息统一成 `/fusion/*` |
| learned_odometry | `learned_odometry.py` + `learned_matching.py`、`learned_tracker.py`、`stereo_geometry.py` | XFeat + LighterGlue 双目视觉里程计 |
| adaptive_guard | `adaptive_guard.py` + `output_continuity.py`、`visual_continuity.py`、`motion_consistency.py` | 输入门控、选源、输出连续性检查、轮速门控 |
| terrain_mapper | `terrain_mapper.py` + `probabilistic_elevation.py`、`height_bias.py`、`disk_map.py`、`bounded_cloud.py` | 高程图与点云建图 |
| odometry_guard | `odometry_guard.py` | 旧版门控，`adaptive_source_selection: false` 时才用 |
| telemetry_motion | `telemetry_motion.py` | 遥测速度进 EKF，主配置关闭 |

## 2. 工作原理

```
domain 10: UE 采集(约 1 Hz 一批: 双目 RGB、OS1 点云、meta)   P4 车辆接口(/car/telemetry 轮速 20 Hz)
               │ live_sensor_relay                              │ sim_imu/wheel_speed_relay.py
domain 57:     ▼                                                ▼
          sensor_adapter ──► learned_odometry ──(机体速度)──► EKF ◄──(6 自由度绝对位姿)── voxelmap_node
                    └──────► voxelmap_node（视觉增量作配准初值）                               ▲
                                                   │                     （可选）模拟 IMU /fusion/imu
                                                   ▼
                     adaptive_guard：门控、选源、连续性检查、轮速上限 ──► 正式位姿（约 1 Hz）
                                                   ▼
                     terrain_mapper：只用 LiDAR，360°、10 m 内 ──► 局部高程图（给 P4）、全局图、UI
```

- **雷达为主，视觉为辅**。EKF 的 `odom0` 是 VoxelMap 的绝对位姿（x y z roll pitch yaw），所以位置和高度由雷达决定。`odom1` 是视觉，只给机体速度（`visual_constraint_mode: body_twist`）。
- **视觉的另外两个用处**：不开 IMU 时，视觉给雷达配准提供初值，雷达在弱几何方向上也拿它当运动参考；雷达持续不可用时，输出切到视觉备份轨迹（接在最近一次融合位姿上）。
- **门控**：LiDAR 配准失败、协方差过大或与视觉运动不一致的帧，不进 EKF。输出位姿相对上一帧的跳变超过速度上限的，不发布。
- **建图**：每帧 LiDAR 按时间戳在正式位姿缓存里插值，然后放进 0.2 m 的概率高程栅格。新扫描和旧地图高度大面积冲突时，这一帧不入图。
- **频率**：整条链路约 1 Hz，由仿真采集决定。EKF 虽配置为 30 Hz，但只在测量时刻发布。
- **构建约束**：改源码后必须 `./build.sh`，否则 `control.py` 检测到源码指纹不一致会拒绝启动。同一个工作区同一时间只能有一个运行（`run_state.json`）。
- **UI**：Qt 外壳 `t3_visual_window`（在 roma 包里），左侧是 RViz（`config/p3_visual_window.rviz`），右侧嵌入 `visual/visual_monitor.py`。

## 3. 启动

详见桌面 `P3轮速版_启动说明.md`。话题、输入输出、测试结果见桌面 `P3_216_交接文档.md`。

```bash
# 终端 A
bash /home/yanfa/P4/scripts/start_p4_vehicle_interface.sh
cd /home/yanfa/P3/lidar_visual_fusion && ./stop.sh && ./start_simulation_sources.sh --host 192.168.10.23
# 终端 B
cd /home/yanfa/P3/lidar_visual_fusion
scripts/sim_imu/start_live_debug.sh --imu off    # 轮速、无 IMU（主用）；--imu on 为轮速 + 模拟 IMU
```

回放和评分命令见 `scripts/sim_imu/README_CN.md`。

## 4. 我做的修改（2026-10-28～10-29，已编译，未提交 git）

新参数的默认值都等于旧行为，只有新增的 profile 才启用。

| 文件 | 修改 | 作用 |
|---|---|---|
| `stereo_geometry.py` | `mapping_frustum_weights` 增加 `omnidirectional` | 建图点云从"只保留相机视锥内"改成 360°，半径不变 |
| `terrain_mapper.py` | 传入 `omnidirectional`；360° 时不再单独抽稀预览点云 | UI 显示 360° 点云 |
| `visual/visual_monitor.py` | 地图标注显示"360°，半径 X m" | 显示用 |
| `output_continuity.py` | `check()` 增加 `speed_limit`、`vertical_rate_limit` 参数，以及 `correction_vertical` | 外部速度上限只会收紧原来的静态门限 |
| `adaptive_guard.py` | 订阅 `/fusion/wheel_speed`，算速度上限；增加 `lidar_failure_grace_sec`；状态 JSON 增加 `wheel_speed`；新增拒绝计数 `wheel_speed_gate` | 防止高度突跳 |
| `visual_continuity.py` | 增加 `max_reference_age` | 视觉备份定期重新锚定 |
| `tests/test_output_continuity.py`、`tests/test_visual_continuity.py` | 新增对应测试 | — |
| `config/simulation_lidar_360.yaml` | 新增：360°、10 m | 旧行为基线 |
| `config/simulation_lidar_360_wheel.yaml` | 新增：360 配置加轮速门控、4 s 宽限、30 s 重锚、`imu_mode: "off"` | **主配置** |
| `config/simulation_lidar_360_simimu.yaml` | 新增：wheel 配置加模拟 IMU | 可选 |
| `scripts/sim_imu/` | 新增：`gt_imu.py`、`sim_imu_live.py`、`sim_imu_player.py`、`wheel_speed_relay.py`、`wheel_speed_player.py`、`start_live_debug.sh`、`replay_*.sh`、`evaluate_against_truth.py` 等 | 真值加噪声生成模拟 IMU、轮速中继、一键实时/回放、真值评分 |

修改的原因：
- **高度突跳**：LiDAR 单帧配准失败后，输出立即切到从开机起从未重新锚定的视觉备份轨迹，而输出连续性门限放行了约 1.25 m/s 的跳变。对应三处修复：失败后 4 s 宽限期内不切源、视觉备份至少每 30 s 重锚、按轮速限制每一步的位移。
- **轮速上限**：最大轮转速 × 0.1281 m × 1.2 + 0.03 m/s。垂直方向按 20° 坡度换算。没有新鲜轮速数据时，退回原来的静态门限。
- **模拟 IMU 不是真实传感器**，用它得到的结果必须标注为仿真实验。

已验证：
- 单元测试通过，编译通过，构建指纹检查通过。
- 10-29 10:30–12:03 用主配置实时跑过一次，结果在 `P3_bak/lidar_visual_fusion/results/live_mapping_20261029_103007_42583`。

未完成：
- 主配置还没做端到端回放回归。
- 10-29 那次实时运行发现，正式输出的 z 会相对 LiDAR 原始里程计平滑漂移：93 分钟后 −0.74 m，而 LiDAR 原始高度误差在 4 mm 以内。原因未查，详见交接文档第 10 节。
- 开 IMU 时停车会发散，需要在 VoxelMap 里加零速约束。

## 5. 注意事项

- `config/simulation_lidar_camera_fov.yaml` 相对 git 有一处**不是我做的**未提交改动：9-23 改成按距离裁切；10-28 15:21 把 cx/cy 改成图像中心 1224/1024。主配置的 `camera_k` 也是 1224/1024（git 原值是 1230.95/990.69）。
- `git status` 会显示 14 个旧入口脚本被删除（`D`），是这次清理移走的，可从 P3_bak 移回或用 `git checkout -- <文件>` 恢复。
- 有 3 个历史对比测试依赖已移走的 `archive/`：`test_optional_imu_ros`、`test_simulation_lidar_basis`、`test_stereo_transport_equivalence`。
- `run_state.json` 仍指向上一次运行的旧路径。不影响使用，下次启动时会被覆盖。

## 6. 本次清理（2026-10-29）

全部是移动，没有删除源码，只删了 `__pycache__` 和 `.pytest_cache` 缓存。完整清单在 `/home/yanfa/P3_bak/MOVED.txt`，恢复方法是把同名路径移回 P3。

| 移到 P3_bak 的内容 | 说明 |
|---|---|
| `lidar_visual_fusion/results`（107 G）、`logs`、`log` | 以往全部运行输出和日志 |
| `lidar_visual_fusion/archive`、`src/bak_t3_lidar_visual_fusion`、各处 `*.bak_*` 和 `*.before_revert_*`、`config/bak0923-*` | 旧版本备份 |
| 14 个旧入口脚本：`start_roma_*`、`start_visual_only_*`、`start_xfeat_*`、`start_learned_fusion.sh`、`test_fusion_bag.sh` 等，以及 `visual_only_live_state.json` | RoMa、纯视觉、早期 bag 演示，当前流程不用 |
| `test_data` 里的候选代码快照和旧报告脚本 | 测试没有引用 |
| P3 根目录下的其他内容：8 月的脚本和文档、`data1223`、`mooncapture_bags`、`test_rosbag`、`rosbag2_*`、`t3_gridmap_v2_delivery`、`imu_sim_staging`、`P3_backup_files` 等 | 旧 RoMa 流程和数据，当前工程不引用 |
