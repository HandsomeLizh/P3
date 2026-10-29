# 课题三定位与2.5D栅格地图

> 当前代码基线：2026-08-21。默认主线为T5双目RoMa VO；FAST-LIO2和FAST-LIVO2是独立可选入口，尚未完成正式指标验收。

## 1. 项目功能

项目读取双目图像以及可选的语义、IMU和LiDAR输入，输出统一车辆位姿、滚动局部GridMap、完整全局高程GridMap、全局占据总览，并在运行中保存NPZ和SQLite完整地图。

默认RoMa链路：

```text
RGB10 + RGB11 -> RoMa双目/时间匹配 -> VO -> odom/base_link位姿
                                            -> 2.5D地图
```

可选FAST-LIVO2链路：

```text
Ouster点云 + Ouster IMU + RGB10 -> FAST-LIVO2 -> 统一位姿与地图接口
```

## 2. 当前默认参数

| 项目 | 当前值 |
|---|---|
| RoMa粗匹配/精修分辨率 | `280 / 280` |
| 双目精修 | 开启 |
| 时间匹配精修 | 关闭 |
| 双向对称匹配 | 关闭 |
| CUDA自定义相关层 | 开启 |
| 特征复用 | 开启 |
| 双目/时间粗匹配合批 | 开启 |
| KLT时间跟踪 | 关闭 |
| IMU旋转初值 | 关闭 |
| 外部后端/全局回环 | 关闭 |
| 语义Mask | 关闭，按需接入 `/task2/seg_mask` |
| 局部地图 | `64 m × 64 m`，`0.2 m/cell`，滚动窗口 |
| 全局高程话题 | 地图版本变化时发布，只含 `elevation` |
| 全局总览 | `1.0 m/cell`，地图变化后最快每20秒发布 |

RoMa推理默认值只有一个来源：`roma_runtime_defaults.py`。节点直接运行和
`t5_roma_vo.launch.py` 均引用它；`semantic_map.yaml` 不再重复保存算法分辨率或匹配开关，因此不会再出现节点、YAML与正式入口默认值不一致的问题。

## 3. 编译和环境检查

```bash
cd ~/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws
bash scripts/build.sh
source scripts/enter_workspace.sh
bash scripts/check_environment.sh
```

工程固定使用ROS 2 Humble和 `/usr/bin/python3`。`build.sh` 只扫描 `src/`，不会把 `third_party/` 当作colcon包误编译。

## 4. 启动RoMa主线

不带语义、显式关闭调试匹配图：

```bash
bash scripts/run_t5_roma_vo.sh \
  --no-feature-matches
```

接入课题二语义Mask并显示匹配图：

```bash
bash scripts/run_t5_roma_vo.sh \
  --feature-matches \
  enable_semantics:=true \
  semantic_topic:=/task2/seg_mask
```

标准启动脚本会自动保存到
`$HOME/P3/output/roma_YYYYMMDD_HHMMSS/`。如需覆盖，可额外传入
`output_dir:=/指定/目录`。

大范围仿真使用轻量全局模式：

```bash
bash scripts/run_t5_roma_vo.sh --lite-map
```

该开关关闭完整高分辨率 `/Car/T3/mapping/global_grid_map` 和稠密
`global_grid_map.npz`，但继续发布64 m滚动局部图、1 m全局总览，
并继续保存稀疏完整地图 `global_grid_map.sqlite3`。普通实车启动默认
保留完整全局GridMap和NPZ，便于当前实时可视化。

当前launch的 `publish_feature_matches` 默认是 `false`，普通启动不会创建特征匹配发布器。需要调试时显式传入 `--feature-matches`；开启后也只有存在订阅者时才渲染。

IMU默认关闭。`enable_imu:=true` 仅给PnP提供陀螺旋转初值，不是完整视觉惯性融合：

```bash
bash scripts/run_t5_roma_vo.sh enable_imu:=true
```

## 5. 可选定位入口

FAST-LIO2（LiDAR+IMU独立定位）：

```bash
bash scripts/run_t5_ouster_lio.sh --check
bash scripts/run_t5_ouster_lio.sh
```

FAST-LIVO2（LiDAR+IMU+RGB10，可同时接课题三地图）：

```bash
bash scripts/run_t5_fastlivo2.sh --check
bash scripts/run_t5_fastlivo2.sh
```

注意：交付包带有FAST-LIVO2固定源码归档，但当前准备脚本不会自动编译它。全新机器需要先建立独立FAST-LIVO2安装空间并设置 `FASTLIVO_WS`；`--check`通过才表示真正可运行。

只运行FAST-LIVO2、不启动课题三地图：

```bash
bash scripts/run_t5_fastlivo2.sh --no-gridmap
```

`--no-gridmap` 不是无相机模式；当前FAST-LIVO2配置仍为 `img_en=1`，需要RGB10。视觉完全不可用时应使用FAST-LIO2。三种定位入口一次只能启动一个，避免重复发布TF、位姿和地图。

## 6. 当前输入

| 数据 | T5话题 | 默认状态 |
|---|---|---|
| RGB10左图 | `/Car/T5/Cam_Left/image_raw/color` | RoMa/FAST-LIVO2使用 |
| RGB11右图 | `/Car/T5/Cam_Right/image_raw/color` | RoMa和带地图FAST-LIVO2使用 |
| 语义Mask | `/task2/seg_mask` | RoMa默认关闭；带地图FAST-LIVO2当前要求存在 |
| WIT IMU | `/Car/T5/wit/imu` | RoMa默认关闭 |
| Ouster点云 | `/Car/T5/OS1/points` | FAST-LIO2/FAST-LIVO2使用 |
| Ouster IMU | `/Car/T5/OS1/imu` | FAST-LIO2/FAST-LIVO2使用 |

RoMa使用Best Effort、depth=1接收双目图像，近似同步队列50，默认同步窗口30 ms、左右目最终门限10 ms。正式实时输入必须使用有效传感器采集时间戳。

## 7. 坐标系和对外接口

```text
map -> odom -> base_link
```

| 输出 | 类型 | 坐标语义 |
|---|---|---|
| `/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry` | `odom -> base_link`，正式位姿 |
| `/Car/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap` | `odom`中的64 m滚动局部图 |
| `/Car/T3/mapping/global_grid_map` | `grid_map_msgs/msg/GridMap` | `map`中的完整全局高程；`--lite-map`时关闭 |
| `/Car/T3/mapping/global_overview` | `nav_msgs/msg/OccupancyGrid` | `map`中的低分辨率全局占据图 |
| `/Car/T3/mapping/global_map_revision` | `std_msgs/msg/UInt64` | 已完整落盘的地图版本 |
| `/Car/T3/debug/trajectory` | `nav_msgs/msg/Path` | `map`中的累计车体轨迹 |

默认不再发布重复的 `/Car/T3/semantic/current_pose`、
`/Car/T3/mapping/grid_map_odom` 和离线逐帧回执
`/Car/T3/debug/input_processed`。旧接收端临时联调时可传入
`enable_legacy_output_aliases:=true`；只有完成回执驱动的离线工具才使用
`publish_input_processed:=true`。

标定原始车体系为前—右—下（FRD），ROS `base_link` 为前—左—上（FLU），相机光学系为右—下—前。RoMa内部估计RGB10光心，再用固定外参转换成 `base_link` 对外发布。

当前 `map` 原点仍由定位后端初始化决定；代码没有强制让第一帧 `base_link` 的平移数值恰好为零。若测试要求初始车体原点为 `(0,0,0)`，需要增加初始化重基准。

## 8. 地图保存

默认输出目录由 `enter_workspace.sh` 设置为：

```text
$HOME/P3/output/roma_YYYYMMDD_HHMMSS
```

主要文件：

```text
trajectory_map.tum
global_grid_map.npz                 # 默认实车模式；大范围模式不生成
global_grid_map.sqlite3
input_processing_summary.json
_internal/trajectory_opencv.tum
_internal/grid_map_snapshot/map_index.yaml
_internal/map_observation_archive/observations_*.sqlite3
```

根目录只放对外交付地图、正式轨迹和运行摘要；`_internal/` 中的文件用于无损重建、内存分页和算法诊断，其他课题无需读取。

默认实车模式下NPZ和SQLite都在运行中由后台线程更新；大范围模式只更新SQLite。正常 `Ctrl+C` 退出会等待最新地图版本落盘；不要用 `kill -9`。运行中需要强制保存时：

```bash
bash scripts/save_grid_map_now.sh
```

## 9. RViz

```bash
bash scripts/run_rviz.sh
bash scripts/run_rviz.sh --feature-matches
```

RViz固定坐标系为 `map`。第二条命令才会订阅匹配图并触发其渲染。

## 10. 文档入口

- 当前交接和指标：[../deliverables/课题三项目交接与测试操作手册.md](../deliverables/课题三项目交接与测试操作手册.md)
- 输入接口：[课题三ROS2输入接口联调说明.md](课题三ROS2输入接口联调说明.md)
- 位姿接口：[课题三Odometry位姿接口说明.md](课题三Odometry位姿接口说明.md)
- 地图接口：[docs/课题三地图与位姿接口使用说明.md](docs/课题三地图与位姿接口使用说明.md)
- 架构：[ARCHITECTURE.md](ARCHITECTURE.md)
- 历史材料：`docs/archive/`，不得当作当前默认配置使用。
