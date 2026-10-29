# 轮速门控、可选模拟 IMU 与调试数据录制（sim_imu）

**模拟 IMU 是用 UE 真值加噪声生成的，不是物理传感器。用它得到的结果必须标注为"模拟 IMU 仿真实验"。** 实车标准 `sensor_msgs/Imu` 接口不受影响。

## 三个配置

| 配置 | 轮速门控 | IMU | 说明 |
|---|---|---|---|
| `config/simulation_lidar_360_wheel.yaml` | 开 | 关（`imu_mode: "off"`） | **主配置** |
| `config/simulation_lidar_360_simimu.yaml` | 开 | 模拟 IMU | 可选，在主配置基础上加 IMU |
| `config/simulation_lidar_360.yaml` | 关 | 关 | 旧行为，作对比基线 |

轮速版的改动（只在前两个配置里生效，其他 profile 行为不变）：
- `lidar_failure_grace_sec: 4.0`：LiDAR 单帧配准失败后 4 s 内不切到视觉备份（此前 0.14–1.1 m 的高度跳变都是这个原因）。
- `visual_continuity.max_reference_age: 30`：视觉备份至少每 30 s 重新锚定到融合位姿，不再沿用开机时的参考。
- `wheel_speed`：轮速给出车速上限（最大轮转速 × 0.1281 m × 1.2 + 0.03 m/s），输出位姿的水平和垂直变化超出就拒绝；没有新鲜轮速时退回原来的静态门限。

## 实时运行：一条命令代替终端 B

```bash
cd /home/yanfa/P3/lidar_visual_fusion
scripts/sim_imu/start_live_debug.sh --imu off    # 主配置：轮速、无 IMU
scripts/sim_imu/start_live_debug.sh --imu on     # 轮速 + 模拟 IMU
```

- 终端 A（采集与人工控制）和 P4 的操作不变。**P4 车辆接口必须在运行**，`/car/telemetry` 来自它。
- 脚本会依次：在 domain 10 启动真值录制和轮速中继（始终开启）、模拟 IMU（`--imu on` 时）；以对应配置和 `--record-input` 启动 `start_live.sh`；按 Ctrl+C 后自动停掉这些辅助进程、关闭真值 bag，在运行目录写入 `debug_session.json`（指向真值）和 `truth_eval.json`（真值评分）。
- `--imu on` 时，建图启动后请**静止约 5 s**，让 IMU 完成重力初始化。
- 可选参数：`--noise none`（无噪声 IMU）、`--no-eval`（跳过评分）。
- 检查轮速是否生效：`fusion_status.json` 里的 `wheel_speed.fresh` 应为 `true`，`samples` 持续增长。

## 回放对比（无界面，独立 domain 91）

```bash
S=scripts/sim_imu
B=/home/yanfa/Env_X/InterFace/bags/20260918_142301
$S/replay_sensor_bag.sh $B --imu off                 # 主配置（轮速默认开）
$S/replay_sensor_bag.sh $B --imu on                  # 轮速 + IMU
$S/replay_sensor_bag.sh $B --imu off --wheel off     # 旧行为基线
$S/replay_live_session.sh results/live_mapping_<run> --imu off   # start_live_debug.sh 录的会话
```

结果在 `results/replay_*/truth_eval.json`。回放时 IMU 直接发布到 `/fusion/imu`，轮速发布到 `/fusion/wheel_speed`，只回放传感器话题。同一工作区同一时间只能有一个 P3 运行实例。

## 原理

- 输入：`/car/pose`（20 Hz 真值，时间戳是 Orin 接收时间）、`/sensor/meta`（每批采集瞬间的位姿）、`/sensor/batch_status`（同一批的传感器时间戳 `stamp_ns`）。
- 时间对齐：真值流时间 − 传感器时间戳 = 0.5–0.8 s，随会话变化并缓慢漂移。在线做法是把每帧 meta 位姿与真值流匹配，用 60 s 滑动窗口做鲁棒拟合；离线时用整段数据拟合。车速约 0.03 m/s 以下时无法匹配，暂用先验值 0.65 s。
- 输出：base_link 系（FLU），100 Hz。陀螺取真值姿态的导数，比力 = Rᵀ(a + g)，g = 9.81（与 VoxelMap 一致）。噪声默认 `mems`（白噪声加零偏随机游走）。
- 实时链路中，IMU 同时直接发布到 domain 57 的 `/fusion/imu`。原因是中继和 Python sensor_adapter 用 5 条的 best-effort 队列接收 IMU，实测会丢掉约 40%，VoxelMap 会因此报 `imu_gap`。
- 没有比例因子误差、安装误差、温漂和振动；车速很低，加速度计主要提供重力方向。

## 评分（evaluate_against_truth.py）

- `origin_aligned`：只用起始时刻的真值位姿对齐，不做拟合，反映累计漂移（俯仰漂移会表现为高度误差）。
- `se3_aligned`：全程刚体拟合，给出 ATE。
- 同时评估正式输出和 VoxelMap 原始里程计，并统计高度跳变、IMU 使用情况和入图丢帧。

## 验证

`python3 scripts/sim_imu/test_gt_imu.py`：8 项测试，覆盖解析运动、静止时重力、偏移恢复（含抖动和重复旧样本）、meta 坐标转换、噪声统计、偏移模型的限速和重启。
