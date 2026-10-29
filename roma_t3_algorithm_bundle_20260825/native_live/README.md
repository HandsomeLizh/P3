# Ubuntu 原生实时运行

本目录用于在 Ubuntu/Jetson 主机上直接运行算法，不启动 Docker，不播放 rosbag。算法启动后会等待 ROS 2 实时话题。

## 数据发布约定

| 数据 | 默认话题 | ROS 2 类型 |
|---|---|---|
| 左目图像 | `/Car/T5/Cam_Left/image_raw/color` | `sensor_msgs/msg/Image` |
| 右目图像 | `/Car/T5/Cam_Right/image_raw/color` | `sensor_msgs/msg/Image` |
| 位姿真值 | `/car/pose` | `geometry_msgs/msg/PoseStamped` |
| OS1 点云 | `/Car/T5/OS1/points` | `sensor_msgs/msg/PointCloud2` |

发布端和算法端必须使用相同的 `ROS_DOMAIN_ID=10`。左右图像必须使用真实采集时间填写 `header.stamp`，两幅图的时间差默认不得超过 `0.07 s`。

点云只提供地图高程证据，不参与定位定姿。相机参数使用 `config/calibration_spreadsheets_parallel_24p9.yaml`，点云外参使用 `config/pointcloud_extrinsic_os1_mapping_only.yaml`。

## 启动

SSH 登录 Jetson 后只执行：

```bash
/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/native_live/run_native_live.sh
```

脚本输出 `Native live pipeline is ready and waiting for data.` 后，再启动实时数据发布端。网页仪表盘默认为 `http://192.168.100.10:8765`。

## 状态和停止

```bash
/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/native_live/status_native_live.sh
/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/native_live/stop_native_live.sh
```

停止时必须使用 `stop_native_live.sh`，以便算法完成地图和清单文件的落盘。

## 结果位置

每次运行会新建：

```text
/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/output_native/roma_live_YYYYMMDD_HHMMSS/
```

当前运行的结果路径写在：

```text
/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/output_native/current_output.txt
```

其中 `logs/` 保存算法、真值记录器和仪表盘日志，`realtime_results/` 保存与测试指标对应的 CSV/JSON 结果，`_internal/reference_pose.tum` 保存实时真值轨迹。

## 修改话题（仅在发布端话题不同时）

```bash
export T3_LEFT_TOPIC=/your/left/image
export T3_RIGHT_TOPIC=/your/right/image
export T3_REFERENCE_TOPIC=/your/ground_truth
export T3_LIDAR_TOPIC=/your/lidar/points
/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/native_live/run_native_live.sh
```

注意：`T3_LIDAR_TOPIC` 目前只用于显示启动约定；如果真实点云话题不同，还需同步修改点云 YAML 的 `topic`。
