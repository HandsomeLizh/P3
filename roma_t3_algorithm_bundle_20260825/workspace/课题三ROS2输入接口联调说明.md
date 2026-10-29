# 课题三ROS 2输入接口联调说明

> 更新日期：2026-08-21。本文以当前T5启动入口为准；节点直接执行时的通用回退参数不等于T5正式默认值。

## 1. 当前T5输入总览

| 数据 | 当前话题 | 类型 | RoMa默认状态 | FAST-LIVO2状态 |
|---|---|---|---|---|
| RGB10左图 | `/Car/T5/Cam_Left/image_raw/color` | `sensor_msgs/msg/Image` | 必需 | 必需 |
| RGB11右图 | `/Car/T5/Cam_Right/image_raw/color` | `sensor_msgs/msg/Image` | 必需 | 带课题三地图时必需 |
| 语义Mask | `/task2/seg_mask` | `sensor_msgs/msg/Image` | 默认关闭，显式开启 | 带课题三地图时当前必需 |
| WIT IMU | `/Car/T5/wit/imu` | `sensor_msgs/msg/Imu` | 默认关闭 | 不使用 |
| Ouster点云 | `/Car/T5/OS1/points` | `sensor_msgs/msg/PointCloud2` | 不使用 | 必需 |
| Ouster IMU | `/Car/T5/OS1/imu` | `sensor_msgs/msg/Imu` | 不使用 | 必需 |
| FAST-LIVO2里程计 | `/aft_mapped_to_init` | `nav_msgs/msg/Odometry` | 不使用 | 地图适配层使用 |

## 2. RoMa双目输入契约

当前T5 launch最终参数：

```text
left/right reliability = best_effort
image_input_qos_depth = 1
image_sync_mode = approximate
image_sync_queue_size = 50
image_sync_slop_sec = 0.03
stereo_sync_max_skew_sec = 0.01
image_timestamp_source = sensor
```

要求：

- 图像编码必须能由 `cv_bridge` 转换为BGR图像；
- 左右图必须使用传感器采集时间戳，不能使用算法发布墙钟；
- 左右图分辨率和时间顺序必须稳定；
- 候选组可以在30 ms同步窗口内形成，但左右目最终时间差超过10 ms仍会拒绝；
- Best Effort和depth=1用于实时优先，处理不过来时不无限积压旧帧。

检查：

```bash
ros2 topic info --verbose /Car/T5/Cam_Left/image_raw/color
ros2 topic info --verbose /Car/T5/Cam_Right/image_raw/color
ros2 topic hz /Car/T5/Cam_Left/image_raw/color
ros2 topic hz /Car/T5/Cam_Right/image_raw/color
```

## 3. 语义Mask接口

当前正式语义话题为：

```text
/task2/seg_mask
sensor_msgs/msg/Image
```

推荐契约：

- 单通道整数类别ID，优先 `mono8`；
- 尺寸与RGB10一致，或由双方明确冻结唯一缩放规则；
- `Header.stamp`继承对应RGB10采集时刻；
- `frame_id`和类别表固定；
- QoS当前接收端为Reliable。

RoMa默认不订阅语义。启用：

```bash
bash scripts/run_t5_roma_vo.sh \
  enable_semantics:=true \
  semantic_topic:=/task2/seg_mask
```

开发自测时可临时启用内置基础语义发布器：

```bash
bash scripts/run_t5_roma_vo.sh \
  enable_semantics:=true \
  launch_basic_semantics:=true
```

内置基础语义节点只用于联调，不得替代课题二正式算法输出。

## 4. IMU接口

RoMa入口默认：

```text
topic=/Car/T5/wit/imu
enable_imu=false
reliability=reliable
depth=10
```

启用后只利用陀螺仪积分给PnP提供旋转初值，不是完整视觉惯性融合：

```bash
bash scripts/run_t5_roma_vo.sh enable_imu:=true
```

启用前必须确认时间戳、角速度单位rad/s、轴向、IMU到RGB10固定外参以及静止零偏。`/Car/T5/wit/angle` 是设备融合后的角度输出，不是 `sensor_msgs/Imu` 原始测量，不能直接替代当前IMU输入。

FAST-LIO2和FAST-LIVO2使用 `/Car/T5/OS1/imu`，与RoMa的WIT IMU不是同一接口。

## 5. LiDAR和外部定位输入

FAST-LIO2入口：

```text
/Car/T5/OS1/points
/Car/T5/OS1/imu
output: /Odometry
```

FAST-LIVO2入口：

```text
/Car/T5/OS1/points
/Car/T5/OS1/imu
/Car/T5/Cam_Left/image_raw/color
output: /aft_mapped_to_init
```

检查入口：

```bash
bash scripts/run_t5_ouster_lio.sh --check
bash scripts/run_t5_fastlivo2.sh --check
```

点云必须确认：

- `header.stamp`的时钟域；
- `frame_id`是否为当前标定对应的 `sensor_frame`；
- x/y/z字段及每点时间字段；
- 驱动是否已经转换坐标或去畸变；
- Ouster点云、Ouster IMU和相机是否处于可对齐时基。

没有这些信息时，话题“能收到”不代表可以用于精度测试。

## 6. 历史损坏bag兼容

部分历史bag的图像 `Header.stamp` 为零或来自不一致时钟。只做功能复现时，可使用：

```bash
bash scripts/run_t5_roma_vo.sh \
  use_sim_time:=true \
  image_timestamp_source:=arrival \
  image_sync_slop_sec:=0.03 \
  stereo_sync_max_skew_sec:=0.03
```

另一终端原速播放：

```bash
ros2 bag play <bag_path> --clock
```

该模式使用约10 ms粒度的回放时钟重建到达时间，只适合功能、性能和地图调试，不能作为正式定位精度或定姿稳定度时间同步证据。

## 7. 处理回执和诊断

生产模式默认不发布逐帧处理回执。只有完成回执驱动的离线数据工具才启用：

```bash
bash scripts/run_t5_roma_vo.sh publish_input_processed:=true
```

```text
/Car/T3/debug/input_processed
std_msgs/msg/Header
```

启用后，该回执在每个同步回调终止时发布，表示该组输入已有明确处理结果，不表示一定产生了有效位姿或地图修订。

运行结束后检查：

```text
<output_dir>/input_processing_summary.json
```

重点字段包括各输入到达数、同步组数、处理开始/结束数、有效位姿数、失败分类和分阶段耗时。

## 8. 联调检查清单

1. 启动定位程序后再启动数据源或bag；
2. 核对实际话题名、类型、发布者数和QoS；
3. 记录每路频率、分辨率、编码、`frame_id`和时间戳范围；
4. 核对左右目时间差和语义Mask对左图的时间差；
5. LiDAR模式核对点云与IMU时钟以及外参方向；
6. 检查最终处理摘要；仅在离线完成回执模式下检查 `/Car/T3/debug/input_processed`；
7. 正式精度测试不得启用事后放宽同步门限、挑帧或损坏时间戳兼容模式。

## 9. 自定义消息边界

当前mapper稳定边界仍使用ROS标准 `Image`、`Imu`、`PointCloud2`和 `Odometry`。课题一/二若最终提供自定义消息，应由薄适配节点转换到上述标准边界；在没有真实 `.msg`定义、QoS、时间和坐标契约前，不应把外部字段解析直接写进mapper主回调。
