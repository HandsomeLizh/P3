# 课题三Odometry位姿接口说明

> 更新日期：2026-08-21。本文描述当前T5 launch的实际对外接口，不沿用早期“直接发布左相机光心”的旧约定。

## 1. 对外话题

| 话题 | 类型 | 定位 |
|---|---|---|
| `/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry` | 正式接口 |

默认只发布 `/Car/T3/localization/odometry`。尚未迁移的旧接收端可临时用
`enable_legacy_output_aliases:=true` 恢复 `/Car/T3/semantic/current_pose`，新代码不得依赖该别名。

## 2. 坐标契约

```text
map -> odom -> base_link
```

当前消息字段为：

```text
header.frame_id = odom
child_frame_id  = base_link
pose.pose       = T_odom_base_link
```

同时由 `/tf` 发布：

```text
map -> odom
odom -> base_link
```

RoMa内部估计坐标仍为 `car_stereo_left_optical_frame`，但T5 launch已经提供经过确认的 `T_camera_base_link`，因此位姿、速度、协方差、Path和 `trajectory_map.tum` 都转换为 `base_link`，不是仅修改帧名。

FAST-LIVO2原生的 `camera_init -> aft_mapped` 也只在适配层内部使用，接入课题三地图节点后转换为同一 `map/odom/base_link` 接口。

## 3. 字段语义

| 字段 | 当前语义 |
|---|---|
| `header.stamp` | 本次有效估计对应的传感器时间戳 |
| `header.frame_id` | `odom` |
| `child_frame_id` | `base_link` |
| `pose.pose.position` | `base_link`原点在 `odom` 中的位置，单位m |
| `pose.pose.orientation` | `base_link`到 `odom` 的四元数，字段顺序x、y、z、w |
| `twist.twist.linear` | 表达在 `base_link` 中的线速度，单位m/s |
| `twist.twist.angular` | 表达在 `base_link` 中的角速度，单位rad/s |
| `pose.covariance` | ROS顺序 `[x,y,z,roll,pitch,yaw]` 的6×6行主序协方差 |
| `twist.covariance` | 同一ROS顺序的速度协方差 |

没有可信后验协方差时，程序使用大有限值表示“未知”，不会用全零矩阵冒充零误差。

## 4. 不同定位入口

| 入口 | 位姿来源 | 速度来源 | 当前状态 |
|---|---|---|---|
| RoMa `vo + none` | 双目/时间匹配与PnP | 相邻有效 `base_link` 位姿后向差分 | 默认主线 |
| FAST-LIO2 | `/Odometry`或其独立输出 | FAST-LIO2/适配层 | 可选，独立启动 |
| FAST-LIVO2 | `/aft_mapped_to_init` | 外部源或后向差分 | 已接入，未正式验收 |
| `vo_lio + gtsam_gp` | 局部因子图 | 后端六维速度状态 | 非默认实验链路 |

一次运行只能选择一个主定位入口，禁止RoMa和FAST-LIVO2同时向同一课题三命名空间发布位姿与TF。

## 5. 速度规则

当定位源没有直接给出速度时，程序使用相邻有效位姿的无滤波后向差分。以下情况速度置零并保留高不确定度：

- 第一条位姿；
- 时间戳不递增；
- 相邻时间差超过 `odometry_velocity_max_interval_sec=1.0`；
- 全局几何修正后需要重置差分历史。

该速度是估计轨迹的离散导数，不是轮速真值。控制课题使用前仍需评估噪声和延迟。

## 6. 原点说明

对外child已经是 `base_link`，但这不等于第一帧车体平移数值一定为零。当前 `map` 原点由定位后端的初始化坐标决定：RoMa从第一帧相机状态初始化，FAST-LIVO2从 `camera_init` 初始化。固定外参改变被描述的物理点，但不会自动重新定义世界原点。

如果正式接口要求：

```text
第一帧 base_link = (0, 0, 0, identity)
```

需要在初始化后增加一次固定重基准变换，并使位姿、TF、Path、地图和落盘轨迹共同使用该变换，不能只改Odometry。

## 7. 检查命令

```bash
ros2 topic echo --once /Car/T3/localization/odometry
ros2 topic hz /Car/T3/localization/odometry
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_link
```

应检查：

1. `header.stamp`递增且对应传感器采集时刻；
2. `header.frame_id=odom`、`child_frame_id=base_link`；
3. 同一时刻Odometry与 `odom -> base_link` TF一致；
4. `/Car/T3/debug/trajectory` 位于 `map`；
5. 语义地图和位姿通过 `map -> odom` 能正确叠加。

## 8. 正式验收边界

当前接口和外参转换已经实现，但不等于定位精度或定姿稳定度已经达标。正式测试仍需时间戳有效的独立真值、固定外参、规定里程和五次重复试验。具体计算见 `../deliverables/课题三项目交接与测试操作手册.md`。
