# Orin 数字仿真演示

工作机：`yanfa@192.168.100.216`，Ubuntu 22.04.5 / ARM64 / ROS 2 Humble，Orin GPU。
主代码目录：`/home/yanfa/P3/roma_t3_algorithm_bundle_20260825`。

日常操作请优先阅读 [建图使用说明](建图使用说明.md)；修正记录见 [实时仿真启动说明](实时仿真启动说明.md)，详细接口见 [双流程运行与接口联调](双流程运行与接口联调.md)。
本轮已修正 LiDAR 输入轴向转换；以前生成的点云及含 LiDAR 的高程地图需要从空地图重建，不能沿用旧文件作为修正后结果。

## 一键启动及屏幕显示

推荐把建图和车辆控制分开启动：

```bash
/home/yanfa/P3/start_live_mapping.sh 192.168.10.22
# 确认图像、位姿更新，并将旧车辆 GUI 速度归零、关闭后，再启用控制。
/home/yanfa/P3/start_live_control.sh 192.168.10.22
```

第一步只连接数采，不播放或录制 bag，不启动车辆控制；第二步由用户在 GUI 中操作，不自动执行路线。IP 改变时替换上述地址。
下面保留兼容入口的说明；带 IP 的旧 `start_demo.sh` 会同时连接车辆通信，不作为本次推荐入口。

在 Orin 上执行（从 SSH 执行也会在 Orin 自己连接的显示屏上打开 RViz）：

```bash
/home/yanfa/P3/start_demo.sh
```

该命令启动视觉定位、高程建图、LiDAR 点云拼接、网页指标界面和 RViz。
RViz 自动选择本机 `DISPLAY=:0` 和当前 GDM 的 Xauthority，自动加载全部显示配置。
图像数据未到达之前显示空地图是正常的。

如果已经知道 UE 仿真电脑的 IP，可一次启动所有采集和车辆通信组件：

```bash
/home/yanfa/P3/start_demo.sh UE电脑的IP
```

算法已经启动后连接 UE：

```bash
ROOT=/home/yanfa/P3/roma_t3_algorithm_bundle_20260825
python3 "$ROOT/integration_demo/democtl.py" sensors --host UE电脑的IP
```

端口沿用 Env_X 原配置：数采 **6665**，车辆通信 **6668**。两个端口先检查可达性。
采集、车辆通信及 GUI 使用 P3 项目内 `envx_runtime` 的私有副本，不修改原 `/home/yanfa/Env_X`。
默认数采 0.25 Hz，原始图像不重复落盘；这些程序与算法、RViz 统一使用 `ROS_DOMAIN_ID=10` 和 Cyclone DDS。
0.25 Hz 是依据 Orin 上的短测保守设置，新实时入口允许降低频率，例如 `--capture-hz 0.2`，不接受高于 0.25 Hz 的值。
每次启动生成独立输出目录，不能把验证数据和实时数据接在同一条轨迹后面。

关闭 RViz 后重新打开，无须手动配置图层：

```bash
/home/yanfa/P3/run_rviz.sh
```

打开原有车辆控制 GUI（显示在 Orin 屏幕上）：

```bash
python3 /home/yanfa/P3/roma_t3_algorithm_bundle_20260825/integration_demo/democtl.py gui
```

网页指标：`http://192.168.100.216:8765`。

## 正常停止

先在车辆 GUI 中停车并确认停稳，再停止通信及算法：

```bash
/home/yanfa/P3/stop_demo.sh
```

脚本依次停止 GUI、采集、车辆通信、算法、网页和 RViz，等待地图保存。
进程使用 PID 与启动时间联合检查，避免误停复用同一 PID 的其他任务。
已有旧版 RoMa 程序运行时，新入口拒绝重复启动，防止两套程序同时发布车辆 TF。

## 输出接口

| 话题 | 类型 | 内容 |
| --- | --- | --- |
| `/T3/semantic/current_pose` | `nav_msgs/msg/Odometry` | 视觉定位得到的 `odom → base_link` 位姿 |
| `/T3/semantic/trajectory` | `nav_msgs/msg/Path` | `map` 坐标系中的车体轨迹 |
| `/T3/semantic/incremental_map` | `t3_interfaces/msg/IncrementalSemanticMap` | 更新后的 64×64 m 滚动窗口；包含高程及语义字段 |
| `/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap` | 局部高程栅格 |
| `/T3/mapping/global_grid_map` | `grid_map_msgs/msg/GridMap` | 完整全局高程栅格 |
| `/T3/mapping/lidar_incremental` | `sensor_msgs/msg/PointCloud2` | 视觉位姿配准后的新增 LiDAR 观测帧，可包含已观察过的体素 |
| `/T3/mapping/lidar_map` | `sensor_msgs/msg/PointCloud2` | 累计 LiDAR 三维地图的限量预览；单窗口显示再抽样至最多 5 万点，不影响完整地图保存 |
| `/T3/mapping/lidar_status` | `std_msgs/msg/String` | JSON：有效关键帧、体素数量、保存路径 |

LiDAR 使用已有独立外参、0.2 m 三维体素、0.5–80 m 范围过滤。
视觉定位及其建图质量检查通过后，才用该次位姿转换 LiDAR。高程融合取每个 XY 栅格的代表高度；
三维点云分支保留不同 Z 的结构，不会把立面或上下层压成一张高度平面。
点云不参与位姿估计。回环、外部里程计和后端位姿修正均关闭。
`/T3/mapping/lidar_keyframe` 是算法与点云存储器之间的内部已配准观测接口。

当前按本次任务先使用双目视觉与 LiDAR，语义 Mask 默认未启用，不能把默认标签当作已验证的语义识别结果。
外部课题二语义可通过 `democtl.py start --enable-semantics --semantic-topic 实际Mask话题` 接入，
需要匹配图像内容、尺寸和采集时间；该分支尚未在本次现场数据上验证。
当前输入直接使用 Env_X 的 `/Car/T5/Cam_Left/image_raw/color`、
`/Car/T5/Cam_Right/image_raw/color`、`/Car/T5/OS1/points`。

## 文件结果

每次运行目录：`output_demo/envx_YYYYMMDD_HHMMSS/`。
当前目录和 PID 记录在 `integration_demo/run_state.json`。

- `trajectory_map.tum`：视觉车体轨迹。
- `global_grid_map.npz`、`global_grid_map.sqlite3`：全局高程栅格。
- `lidar_map.sqlite3`：完整体素点云，逐关键帧事务提交，不受网页/RViz 预览点数限制。
- `lidar_map.pcd`：完整二进制 XYZ 点云，后台每 20 秒导出，正常停止时再次导出最新版本。
- `lidar_map_status.json`：点云状态。
- `logs/`：各组件日志。
- `realtime_results/`：已有指标界面保存的 CSV/JSON。

## 已完成的 Orin 测试与当前边界

以下为早期功能测试记录，不代表本轮轴向修正后的建图精度；最新交互测试及验证边界见《实时仿真启动说明》。修正后未再次运行历史 bag；随后已完成短时实时控制、采集、定位、建图与显示联调。

2026-09-15，在这台 Orin 上完成：

1. `t3_semantic_mapping` 原生编译；9 项点云/ROS 消息测试通过。
2. 使用 Orin 自带的 `Env_X/SaveImages_K0` 记录数据，依次运行前 12 批双目和 LiDAR。
3. 前 8 批全部收到处理回执和视觉位姿，视觉移动约 2.97 m；全部要求输出话题有非空数据。
4. 增加 4 批检查实际屏幕显示，RViz 显示 `Global Status: Ok`。
5. 累计 8 个 LiDAR 建图关键帧、301478 个三维体素；预览限制不影响完整 PCD/SQLite 保存。

详见 `recorded_verification.json`、`visualization_verification.json`、`rviz_screen.png`。
这是记录数据的功能验证，不是实时仿真精度验收。还需要 UE 电脑联网后的 IP，
再确认传感器时间戳、图像/点云时间差和车辆运动下的持续运行。
月球车和机器狗的物理传感器尚未在这次环境中接入，不能视为实机验收完成。

## 版本差异及清理

核对范围为源码、脚本、运行模块、配置及仪表盘：

- `/home/yanfa/roma_t3_algorithm_bundle_20260825` 的 227 个共有文件与本地交付包相同。
- P3 下的同名交付包这 227 个文件也相同，另有 native_live、实时 launch 和缩小图像标定等 7 个文件。
- P3 旧工作区的 207 个共有文件相同，6 个文件不同：`compact_grid_store.py`、
  `tiled_semantic_map.py`、`roma_semantic_mapper_node.py`、`play_bag_with_record_stamps.py`、
  `test_evaluate_all_metrics.py`、`evaluate_all_metrics.py`。较新交付包包含地图持久化并行处理和实时指标等更新。
- 本次修改原文件之前保存了 `integration_demo/before_demo_20260915_180014.tar.gz`。
- 删除顶层损坏 DINO 权重和完全相同的 RoMa 重复权重，原路径改为指向当前完整模型的软链接，
  回收 1486391964 字节。完整模型仍可从运行目录复制恢复旧路径。
- 清理 P3 的 Python/pytest 可重建缓存，以及本次 RViz 插件下载缓存。
- 历史 rosbag、原始采集、旧源码和备份保留。`integration_demo/rviz_plugins` 是当前显示所需运行库。

RViz 缺少的官方高程地图插件已解包在用户目录的 `rviz_plugins/opt/ros/humble`，
启动脚本自动添加插件和共享库路径；未修改全局系统安装。
