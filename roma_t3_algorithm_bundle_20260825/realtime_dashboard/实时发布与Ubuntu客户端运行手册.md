# T3 实时发布与 Ubuntu 可视化客户端运行手册

## 1. 已部署文件

Ubuntu GPU 容器：`t3-x86-gpu`

```text
/home/yanfa/P3/realtime_dashboard/t3_realtime_dashboard.py
/home/yanfa/P3/realtime_dashboard/run_realtime_dashboard.sh
/home/yanfa/P3/realtime_dashboard/t5_roma_vo_bag_parallel_24p9.launch.py
/home/yanfa/P3/realtime_dashboard/modified_source/roma_semantic_mapper_node.py
```

mapper 源码和当前安装空间已同步更新：

```text
/home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/src/t3_semantic_mapping/t3_semantic_mapping/roma_semantic_mapper_node.py
/home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/install_gpu/t3_semantic_mapping/lib/python3.10/site-packages/t3_semantic_mapping/roma_semantic_mapper_node.py
```

专用 launch 的源码和安装空间也已同步。原文件保留了后缀为
`.before_realtime_dashboard_20260825` 的备份。

## 2. ROS 接口

| 信息 | ROS 话题 | 类型 |
|---|---|---|
| 位姿估计 | `/Car/T3/localization/odometry` | `nav_msgs/Odometry` |
| 位姿真值 | `/car/pose` | `geometry_msgs/PoseStamped` |
| 每帧精确耗时 | `/Car/T3/metrics/frame_timing` | `std_msgs/String`，JSON |
| 实时对齐位姿和误差 | `/Car/T3/metrics/live_pose_error` | `std_msgs/String`，JSON |
| 每 500 s 姿态稳定度 | `/Car/T3/metrics/orientation_stability` | `std_msgs/String`，JSON |
| 地图范围和存储指标 | `/Car/T3/metrics/map_stats` | `std_msgs/String`，JSON |
| RViz 全局高程地图 | `/Car/T3/mapping/global_grid_map` | `grid_map_msgs/GridMap` |

`frame_timing` 由 mapper 在每帧结束时直接发布，字段包括传感器时间戳、
位姿处理时间、地图更新时间和端到端时间，因此不是客户端接收时间的近似值。

客户端对“截至当前时刻已经收到”的估计和真值位置做累计刚体 SE3 对齐，不改变
尺度、不使用未来样本；实时位置误差为对齐后估计位置与最近真值位置之间的欧氏
距离。这与正式 ATE 的刚体对齐口径一致。真值匹配最大时间差默认为 0.2 s。

姿态稳定度按客户端单调墙钟每累计 500 s 发布一次，不使用加速回放后的传感器
时间冒充实际运行时间。计算时分别求窗口起止处的真值和估计 Roll/Pitch/Yaw
变化，取姿态误差变化三轴绝对值的最大值，再除以固定 500 s，单位为度每秒。
界面同时显示三轴差值和稳定度。

地图有效面积计算为：

```text
有效面积(m2) = elevation 层有限值栅格数 * 0.2 * 0.2
单位面积存储量(GB/km2) = 全部地图文件字节数 / 1e9 / 有效面积(km2)
```

## 3. 自动保存的测试结果

客户端从启动开始逐条追加 CSV，并且每 2 s 原子更新 JSON 和 Markdown 汇总。
网页最多绘制最近 600 点，但以下 CSV 保存全部实时样本，不会随网页窗口截断：

| 文件 | 对应需求 |
|---|---|
| `realtime_pose_samples.csv` | 位置/姿态真值、估计值、位置误差 |
| `orientation_stability_500s.csv` | 每个完整 500 s 姿态稳定度；不足 500 s 时只有表头 |
| `pose_processing_time.csv` | 每帧位姿处理传感器时间戳、单调时钟端点和耗时 |
| `map_update_time.csv` | 每帧地图更新时间戳、单调时钟端点和耗时 |
| `map_stats_history.csv` | 0.2 m 分辨率、栅格数、有效面积和矩形范围历史 |
| `requirements_test_results.json` | 机器可读的需求指标、阈值、结果和状态 |
| `需求指标测试结果.md` | 可直接交接的逐项测试结论 |
| `dashboard_state_cache.json` | 网页最后状态，用同一输出目录重启时恢复 |

每次运行必须给 mapper 和客户端传入同一个全新输出目录。客户端收到数据后会自动
创建这些文件；正常按 `Ctrl+C` 结束时还会再强制写一次最终汇总。

## 4. 运行前检查

在 Windows PowerShell 中连接 Ubuntu，密码为 `111111`：

```powershell
ssh yanfa@192.168.2.2
```

在 Ubuntu 中检查旧进程。不要同时启动两个 mapper 或两个回放器：

```bash
docker exec t3-x86-gpu pgrep -af roma_semantic_mapper_node || true
docker exec t3-x86-gpu pgrep -af t3_realtime_dashboard.py || true
docker exec t3-x86-bagtest-20260824-235238 \
  pgrep -af play_stereo_lidar_rosbag || true
```

如有旧进程，应回到对应终端按 `Ctrl+C` 正常结束。

## 5. 终端 1：启动 mapper

```bash
OUT_NAME=roma_realtime_20260824_235238_$(date +%Y%m%d_%H%M%S)
echo "$OUT_NAME"

docker exec -it t3-x86-gpu env \
  ROS_DOMAIN_ID=228 \
  T3_OUTPUT_DIR="/home/yanfa/P3/output/$OUT_NAME" \
  T3_LAUNCH_FILE=t5_roma_vo_bag_parallel_24p9.launch.py \
  T3_CALIB_PATH=/home/yanfa/P3/output/calibration_spreadsheets_parallel_24p9.yaml \
  T3_REFERENCE_TYPE=pose \
  T3_REFERENCE_TOPIC=/car/pose \
  T3_SYNC_SLOP=0.07 \
  T3_STEREO_MAX_SKEW=0.07 \
  /home/yanfa/P3/bagtest_workflow/01_run_roma.sh \
  enable_pointcloud_fusion:=true \
  pointcloud_fusion_config:=/home/yanfa/P3/output/pointcloud_extrinsic_os1_mapping_only.yaml \
  publish_realtime_metrics:=true
```

必须等待终端显示 `Mapper ready` 和 `point-cloud elevation fusion` 后再回放。

## 6. 终端 2：启动实时客户端

先从终端 1 输出中取得同一个输出目录，例如：

```bash
OUT=/home/yanfa/P3/output/roma_realtime_20260824_235238_20260825_153000

docker exec -it t3-x86-gpu \
  /home/yanfa/P3/realtime_dashboard/run_realtime_dashboard.sh "$OUT"
```

默认参数已经是 500 s 稳定度周期、HTTP 端口 8765、ROS domain 228。
客户端会把最后一屏状态保存为输出目录中的 `dashboard_state_cache.json`；用同一
输出目录重启客户端时会恢复显示。每次正式试验仍应使用新的输出目录。
浏览器打开：

```text
http://192.168.2.2:8765
```

健康检查：

```bash
curl http://127.0.0.1:8765/healthz
```

应返回 `{"status":"ok"}`。

## 7. 终端 3：回放 588 帧数据

```bash
docker exec -it t3-x86-bagtest-20260824-235238 bash -lc '
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=228
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  python3 /tmp/play_stereo_lidar_rosbag_with_scaled_header.py /bag \
    --rate 7 \
    --header-time-scale 1000000 \
    --lidar-lead-sec 0.5 \
    --reference-type pose_stamped \
    --reference-topic /car/pose
'
```

结束时必须看到 `published_pairs=588`、`lidar_messages=588`。回放结束后继续等待
mapper 显示 `Frame 588`，再在终端 1 按 `Ctrl+C`，以便完整落盘。

## 8. 终端 4：RViz 显示全局地图

Ubuntu 必须有图形桌面或正确的 X11 转发环境：

```bash
docker exec -it t3-x86-gpu bash -lc '
  export ROS_DOMAIN_ID=228
  cd /home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws
  ./scripts/run_rviz.sh
'
```

现有 RViz 配置已经订阅 `/Car/T3/mapping/global_grid_map`，Fixed Frame 为 `map`，
显示项名称为 `Complete Global Elevation GridMap`。

## 9. 检查实时话题

```bash
docker exec -it t3-x86-gpu bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/install_gpu/setup.bash
  export ROS_DOMAIN_ID=228
  ros2 topic hz /Car/T3/metrics/frame_timing
'
```

查看单条指标：

```bash
docker exec -it t3-x86-gpu bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/install_gpu/setup.bash
  export ROS_DOMAIN_ID=228
  ros2 topic echo --once /Car/T3/metrics/map_stats
'
```

## 10. 常见问题

| 现象 | 检查方法 |
|---|---|
| 网页打不开 | 检查客户端终端是否运行、`curl /healthz` 是否成功、8765 端口是否被防火墙拦截 |
| 网页只有地图没有位姿 | 客户端应在 bag 回放前启动；位姿和耗时话题不是历史回放缓存 |
| 没有 500 s 稳定度 | 必须累计满 500 s 客户端实际运行时间；加速回放的传感器时间不计入实际运行时长 |
| 地图面积很小 | 检查 mapper 命令是否同时传入 `enable_pointcloud_fusion:=true` 和点云配置 |
| RViz 没有地图 | 检查 Fixed Frame 为 `map`，并确认全局地图话题存在且有发布者 |
| 单位面积存储量为 `--` | 需要先收到至少一张具有有效 elevation 栅格的全局地图 |
