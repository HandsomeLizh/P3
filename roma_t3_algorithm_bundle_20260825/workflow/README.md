# T3 RoMa 双容器 Bag 测试

## 容器职责

- `t3-x86-gpu`：运行 RoMa、保存地图与轨迹、执行指标评估。
- `t3-x86-bagtest-fixed`：只读访问 `/bag`，发布左右目图像与所选真值话题。
- 两个容器统一使用 `ROS_DOMAIN_ID=228` 和 CycloneDDS。

## 启动顺序

### 1. GPU 容器先启动 RoMa

```bash
/home/yanfa/P3/bagtest_workflow/01_run_roma.sh
```

实物 bag 不传 `T3_CALIB_PATH`，继续使用 launch 中的实物标定。仿真 bag
显式传入拟合标定文件：

```bash
T3_CALIB_PATH=/home/yanfa/P3/output/t5_sim_estimated_20260823_214305.yaml \
T3_REFERENCE_TYPE=pose \
T3_REFERENCE_TOPIC=/car/pose \
T3_OUTPUT_DIR=/home/yanfa/P3/output/roma_20260823_214305 \
  /home/yanfa/P3/bagtest_workflow/01_run_roma.sh
```

默认配置：

- `install_gpu`
- CUDA GPU 0
- 原始 `sensor_msgs/Image.header.stamp`
- 左右目近似同步窗口 70 ms
- 双目 QoS `reliable/depth=10`
- 不使用 `/clock`

### 2. Bag 容器再播放数据

```bash
T3_BAG_RATE=2.0 /home/yanfa/P3/bagtest_workflow/02_play_rosbag.sh
```

实物 bag 默认保持 `Header.stamp` 不变。`20260823_214305` 的仿真 Header
相对时间被压缩约 1,000,000 倍；使用缩放播放器恢复时间跨度，并给每对左右图
写入同一个时间戳：

```bash
T3_HEADER_TIME_SCALE=1000000 \
T3_BAG_RATE=10 \
T3_REFERENCE_TYPE=pose \
T3_REFERENCE_TOPIC=/car/pose \
  /home/yanfa/P3/bagtest_workflow/02_play_rosbag.sh
```

短测或倍率调优时可增加 `T3_MAX_PAIRS=32`。仅当 ≥95% 的同步回调完成且
无重入拒绝/时间偏差拒绝时，才提高倍率；否则降低 `T3_BAG_RATE`。缩放模式
会保留 `/car/pose` 自带的正常 Header 时间戳，只对左右目图像的相对时间乘以
`T3_HEADER_TIME_SCALE`。当前仿真 bag 本身不含 `/Car/T5/pq_data`。

仿真模式只发布：

- `/Car/T5/Cam_Left/image_raw/color`
- `/Car/T5/Cam_Right/image_raw/color`
- `/car/pose`（`geometry_msgs/msg/PoseStamped`）

实物 bag 保持默认值 `T3_REFERENCE_TYPE=pq`；若不需要真值记录，可使用
`T3_REFERENCE_TYPE=none`。

播放器发现右目话题已经存在发布者时会拒绝启动，避免同名话题冲突。

### 3. 停止顺序

先停止 rosbag 播放器，再停止 RoMa。RoMa 收到一次 SIGINT 后需要等待最终地图快照保存完成。

### 4. 评估

```bash
/home/yanfa/P3/bagtest_workflow/03_evaluate.sh OUTPUT_DIR
```

仿真运行会把 `/car/pose` 自动记录为 `_internal/reference_pose.tum`，评估脚本
会自动选用它。实物评估使用从 bag SQLite `messages.timestamp` 提取的
`pq_reference.tum`。
原始 PQ 是 `Float64MultiArray`，没有 Header；回放时被动记录的系统墙钟不能作为图像传感器时间参考。

PQ 姿态字段默认使用 `--pq-rpy-signs -1 1 1`：2026-08-24 bag 中的
PQ roll 极性与 ROMA、WIT 两个独立来源相反，因此先修正 roll 字段极性再转换为
四元数。该选项只用于 PQ 来源；显式传入的通用 `--reference-tum` 不会被改写。
如果后续数据源已统一坐标约定，可显式传入 `--pq-rpy-signs 1 1 1`。

## 兼容入口

原入口仍然保留：

- GPU：`/home/yanfa/P3/run_roma_bagtest.sh`
- Bag：`/home/yanfa/P3/play_bag_for_roma.sh`
