# 课题三 MPRF 闭环实现与部署说明（fixed）

更新时间：2026-08-16

## 1. 当前结论

MPRF 已作为一个**默认关闭、进程隔离、可替换的闭环候选源**接入课题三。
它不是直接改定位结果的黑盒模型，也不会把一次外观相似直接写成因子：

```text
MPRF 外观/多模态召回
        ↓ 最多 3 个候选
本进程双向 LiDAR ICP 几何复核
        ↓ 每个 query 只保留最佳几何结果
相邻重访的时序一致性确认（默认 2 次）
        ↓
独立的鲁棒 iSAM2 全局位姿图
        ↓
Path/轨迹重写 + 2.5D 地图完整重建并原子切换
```

本地 VO、LIO 或 VO/LIO 局部估计器始终保留自己的原始轨迹。全局图只消费
冻结的局部增量，不把修正结果再次喂回局部估计器，因此不会重复计算同一测量。
MPRF 服务超时、退出、返回坏包或候选被拒绝时，实时 Odometry 和现有地图链继续
工作。正式 ROS 2 输入/输出话题和两个课题三 `.msg` 均未修改。

当前适用范围是：

- `localization_mode=lio`；
- `localization_mode=vo_lio`；
- 同时启用 LiDAR 地图几何，以便候选必须经过米制点云复核。

纯 `vo` 仍使用冻结 RoMa 自带的回环开关，不接这个新全局层。节点会拒绝
`vo + enable_global_loop_closure=true`，避免两套回环所有权混在一起。

## 2. 为什么不是“完全照搬 MPRF”

项目只固定保存上游 MPRF 模型定义和权重；ROS 接口、服务协议、存储、候选验证、
后端和地图重建均由本项目自己实现：

1. **SALAD 全局检索**：在全部满足时间/关键帧间隔的历史记录中分块做精确余弦
   top-k，不截断历史数据库；
2. **DINOv2 重排**：用第二种视觉描述子降低单一外观特征误召回；
3. **DINOv2 + SONATA 局部特征**：尝试估计粗略
   `T_reference_query`；
4. 如果局部 RANSAC 不能给出可信位姿，仍返回显式的
   `candidate_mode=retrieval_only`，由 ROS 侧从局部里程计相对位姿初始化 ICP；
5. ROS 侧对同一 query 的最多 3 个候选逐个做完整几何复核，只让置信度最高的
   已验证结果进入时序确认；
6. 只有确认后的约束才进入本项目轻量 iSAM2 图，不运行 gnssFGO，也不链接其源码。

第 4 步是这次真实数据验证后增加的召回兜底。目标序列中 SALAD/DINO 能找到真实
重访，但 MPRF 局部跨模态 RANSAC 并非每次都能解出粗位姿。如果直接丢弃，系统很
安全但召回为零；保留为 `retrieval_only` 后，错误外观候选仍必须通过同一套 LiDAR
门限，所以没有降低闭环准入标准。

## 3. 几何与后端安全门

默认门限由 `LidarLoopConfiguration` 固定并可在 ROS 参数中调整主要项目：

| 检查 | 默认值/行为 |
|---|---|
| 子图 | 当前及邻近 3 个 LiDAR 关键帧 |
| 体素/点数 | 0.25 m，最多 5000 点 |
| 候选时间间隔 | 至少 20 s |
| 候选节点间隔 | 至少 20 个输出节点 |
| ICP | 70% trimmed point-to-point，最多 30 次 |
| 最大对应距离 | 1.0 m |
| 最少内点 | 120 |
| 最小 overlap | 0.25 |
| 最小唯一目标点比例 | 0.50 |
| 最大 RMSE | 0.30 m |
| 正反向循环误差 | 0.20 m / 3 deg |
| 相对初值最大修正 | 4.0 m / 25 deg |
| 时序确认 | 默认 2 次相邻且修正一致的重访 |

ICP 同时从 query→reference 和 reference→query 运行。约束噪声由实际 RMSE、
overlap 和子图特征半径计算，并做上下界裁剪。全局图包含：

- 第一个节点的强先验；
- 冻结局部轨迹的连续 `BetweenFactorPose3`；
- 通过全部门限的闭环 `BetweenFactorPose3`；
- 闭环因子外包 Huber 鲁棒核；
- iSAM2 每次闭环后额外完成一次重线性化传播。

全局修正只生成 `global <- local` 校正层。所有历史 Path/TUM 和地图关键帧位姿
从不可变局部轨迹重新计算；2.5D 地图从无损 SQLite 观测档案重放到私有候选图，
成功后一次原子替换。重建期间继续对外保留上一张完整地图。

## 4. 三种候选模式

| `global_loop_candidate_source` | 候选来源 | 是否需要 MPRF 服务 | 最终准入门 |
|---|---|---:|---|
| `mprf` | SALAD + DINOv2，局部 DINO/SONATA 可选粗位姿 | 是 | LiDAR ICP + 时序确认 |
| `lidar` | 本进程 Scan Context | 否 | 同一 LiDAR ICP + 时序确认 |
| `hybrid` | 两者同时提供候选 | 是 | 同一 LiDAR ICP + 时序确认 |

推荐先使用 `mprf`。若现场网络或伴随 GPU 不稳定，`lidar` 是完全本地、依赖更少的
退化模式；重复地形特别严重时可试 `hybrid`，但要重新统计误召回、耗时和闭环因子。

`enable_global_loop_closure=false` 是默认值。关闭时不创建全局图、不创建 MPRF
客户端和轮询定时器，原有 VO/LIO/地图功能不受影响。

## 5. 目录与固定资产

发布时整体保留唯一的 `t3_gridmap_v2_ws/` 顶层目录：

```text
t3_gridmap_v2_ws/
├── src/                       # ROS 2 客户端、LiDAR 安全门、iSAM2、地图重建
├── mprf_runtime/              # 自研 HTTP 服务、协议、存储、模型适配和测试
├── third_party/MPRF/          # 固定上游源码，只读
└── models/
    ├── mprf/Weights/          # 微调 DINOv2、预训练/重训练 SALAD
    └── sonata/sonata.pth      # SONATA 权重
```

固定版本：

| 资产 | 固定版本 |
|---|---|
| DLR-RM/MPRF | 交付内置版本（fixed-20260816） |
| SONATA 源码 | 交付内置版本（fixed-20260816） |
| PyTorch | 2.5.0 + CUDA 12.4（MPRF 独立环境） |
| Python | 3.10.18（MPRF 独立环境） |

校验脚本只核对权重是否存在及其固定字节数。执行：

```bash
cd /opt/t3_gridmap_v2_ws
python3 mprf_runtime/scripts/verify_assets.py
```

当前正式服务读取：

- `finetuned_dinov2_v3.pth`；
- `pretrained_salad.ckpt`；
- `sonata.pth`。

`retrained_salad.ckpt` 只作为后续 A/B 资产保留，当前不会与结构含义不同的预训练
checkpoint 混用。上游 MPRF 是 MIT 许可；各模型权重可能有独立许可，外发或商用前
仍需逐项复核。

## 6. 为什么 MPRF 必须独立进程

主 ROS 2 目标环境是 Ubuntu 20.04 / ROS 2 Humble / Python 3.8，最终平台是
JetPack 5.1.1 的 AGX Orin。MPRF/SONATA 发布环境需要 Python 3.10、PyTorch 2.5、
CUDA 12.4、spconv 和 torch-scatter，不能安全塞进同一个 ROS Python 环境。

因此采用两个进程：

```text
AGX Orin / ROS 2 主机                 x86_64 CUDA 12.4 伴随 GPU
---------------------                --------------------------
FAST-LIO2 + mapper  -- HTTP/NPZ -->  MPRF 单 GPU 串行服务
本地 LiDAR ICP/iSAM2 <-- JSON 候选 --  DINO/SALAD/SONATA
```

推荐正式部署也保持此边界。当前没有宣称 CUDA 12.4 的 x86 wheel 能在 JetPack 5.1
直接运行；若一定要单机 Orin 部署，必须为 aarch64/JetPack 重新选择兼容的 Torch、
spconv、torch-scatter 并重新做模型和实时性验收。

## 7. 在 x86 伴随机创建 MPRF 环境

生成的 `.venv` 与机器架构、Python 和 CUDA ABI 绑定，不随源码复制。新机器安装
`uv` 后执行：

```bash
cd /path/to/t3_gridmap_v2_ws/mprf_runtime
bash scripts/create_environment.sh
```

脚本会创建 Python 3.10.18 环境并安装固定版本。随后先做校验和模型烟测：

```bash
cd /path/to/t3_gridmap_v2_ws/mprf_runtime
python3 scripts/verify_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_models.py \
  --device cuda:0
```

`--quick` 只用于快速诊断；正式配置是 560 px 局部位姿图像和最多 2500 个位姿点，
部署验收应至少完整跑一次不带 `--quick` 的命令。

## 8. 启动服务

### 8.1 同机回环地址

默认只监听 loopback，不需要令牌：

```bash
cd /path/to/t3_gridmap_v2_ws/mprf_runtime
CUDA_VISIBLE_DEVICES=0 bash scripts/run_service.sh \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --state-root /path/to/persistent/mprf_state
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8765/health
```

应看到 `status=ready`、`backend=dinov2_salad_sonata`、正确的源码 revision、
`cuda_available=true` 和实际 GPU 名称。

### 8.2 远程伴随 GPU

非 loopback 监听时服务会强制要求 bearer token。两端设置同名环境变量：

```bash
export MPRF_SERVICE_TOKEN='<由部署方生成的随机长令牌>'
CUDA_VISIBLE_DEVICES=0 bash scripts/run_service.sh \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda:0 \
  --state-root /path/to/persistent/mprf_state
```

ROS 主机也设置 `MPRF_SERVICE_TOKEN`，再将
`mprf_endpoint:=http://<伴随机IP>:8765` 传给 launch。HTTP 本身不提供 TLS；只能在
受信内网/VLAN 或隧道内使用，不应直接暴露到公网。

服务 API 是本项目内部进程协议：

- `GET /health`；
- `POST /v1/keyframes`，`Content-Type: application/x-npz`；
- `POST /v1/reset`。

请求限制为 128 MiB、最多 100000 个点；包内 session、节点顺序、时间戳、图像、
内参和刚体外参都做严格校验，NumPy 读取禁用 pickle。GPU 推理用单一信号量串行，
避免多个 HTTP 线程同时耗尽显存。

## 9. ROS 2 启动

先构建主工作区：

```bash
cd /path/to/t3_gridmap_v2_ws
bash scripts/build.sh
source scripts/enter_workspace.sh
```

### 9.1 实时 LIO + direct LiDAR + MPRF

FAST-LIO2 先独立发布 `/Odometry` 和点云。随后启动 mapper：

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  use_sim_time:=false \
  data_dir:=/path/to/calibration \
  output_dir:=/path/to/output \
  localization_mode:=lio \
  fusion_backend:=none \
  enable_imu:=false \
  lio_odometry_topic:=/Odometry \
  lio_expected_parent_frame:=camera_init \
  lio_expected_child_frame:=body \
  lio_child_to_lidar_transform:="[<16个T_lidar_child行主序值>]" \
  lio_child_to_lidar_confirmed:=true \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=direct \
  lidar_fallback_to_stereo:=false \
  enable_global_loop_closure:=true \
  global_loop_candidate_source:=mprf \
  global_loop_confirmation_count:=2 \
  mprf_endpoint:=http://127.0.0.1:8765 \
  publish_layer_images:=false \
  publish_map_pointcloud:=false
```

这里的 `enable_imu=false` 只关闭 mapper 给 RoMa 用的 IMU 路径；FAST-LIO2 仍必须
自己消费原始 LiDAR 和 IMU。相机图像与课题二语义图仍用于点云语义投影和 MPRF。

### 9.2 VO/LIO 局部融合 + MPRF

将上面的定位参数改为：

```text
localization_mode:=vo_lio
fusion_backend:=gtsam_gp
enable_imu:=true
```

局部 10 s fixed-lag 图与独立全局 iSAM2 图职责不同：前者融合连续 VO/LIO，后者只
处理已验证闭环。全局结果不会作为新 LIO/VO 测量回灌。

### 9.3 当前数据集复现

服务启动后：

```bash
cd /path/to/t3_gridmap_v2_ws
bash scripts/run_target_lio_replay.sh \
  /data/batch_organized/2025-06-26-23-01-34 \
  enable_global_loop_closure:=true \
  global_loop_candidate_source:=mprf \
  global_loop_confirmation_count:=2 \
  mprf_endpoint:=http://127.0.0.1:8765 \
  max_frames:=157 \
  max_stereo_skew_ms:=200.0 \
  playback_rate:=100.0 \
  start_delay_sec:=5.0 \
  output_dir:=/path/to/output
```

`200 ms` 只用于复现这个左右目采集时差较大的数据集，并且仍会明确排除 20 对；
不能照搬到实时双目几何。正式系统应保持硬件同步和项目默认门限。

## 10. 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `enable_global_loop_closure` | `false` | 总开关 |
| `global_loop_candidate_source` | `mprf` | `mprf` / `lidar` / `hybrid` |
| `global_loop_confirmation_count` | `2` | 几何结果需要一次或两次一致确认 |
| `global_loop_geometry_cache_records` | `16` | ROS 侧 RAM 中保留的点云记录；其余从磁盘读 |
| `mprf_endpoint` | `http://127.0.0.1:8765` | 服务地址 |
| `mprf_request_timeout_sec` | `20.0` | 单请求超时 |
| `mprf_queue_size` | `8` | 有界最新优先队列；满时丢最旧待处理候选，不阻塞定位 |
| `mprf_transient_retries` | `2` | 429/连接临时错误重试次数 |
| `mprf_poll_period_sec` | `0.10` | 停车后仍应用完成结果的串行轮询周期 |
| `mprf_auth_token_env` | `MPRF_SERVICE_TOKEN` | bearer token 所在环境变量名 |

主候选与 ICP 门限也都有 `global_loop_*` 参数。不要在没有独立真值和误闭环统计时
只为“多出几个闭环”而放宽 overlap、唯一性、RMSE、循环误差或修正量门限。

## 11. 长时间运行与故障隔离

- MPRF 服务将每个 session 的原图、点云、标定、描述子和响应原子保存为 NPZ；
- 全部历史检索描述子以 float16 常驻，每关键帧约 21 KiB；评分按 256 条分块，
  不一次物化完整高维矩阵；
- DINO/SONATA 局部特征 RAM LRU 默认只保留 8 条；
- ROS 侧 LiDAR 闭环几何 RAM 默认 16 条，完整下采样点云按关键帧落盘；
- ROS→MPRF 队列默认 8，始终有界且最新优先，模型慢时不会拖住 Odometry；
- 每个 mapper 进程使用新的随机 session ID，重启不会与旧节点号冲突；
- 一个服务实例建议只服务一个正在运行的机器人；多机器人应使用独立端口/state；
- 服务响应做协议版本、session、query 节点、时间戳、分数、候选顺序和刚体矩阵校验；
- 远程服务失败只增加诊断计数，不会关停局部定位和建图；
- MPRF 结果由默认互斥回调组中的 0.1 s 定时器应用，机器人停止出图后也不会把
  最后的闭环困到 shutdown 阶段；
- 逐点 LiDAR 去畸变等待未来 LIO 位姿时，失败事务会有界保留该关键帧**原来选中的
  PointCloud2**，重试不会换成另一帧或因 20 帧实时缓存淘汰而永久冻结地图。

正常关闭会写：

```text
output_dir/input_processing_summary.json
output_dir/dataset_playback_summary.json        # 数据集 launch 才有
output_dir/trajectory_*.tum
output_dir/map_observation_archive/*.sqlite3
output_dir/loop_closure_archive/run_*/*.npz
output_dir/grid_map_snapshot/
```

`input_processing_summary.json.global_loop` 会记录图节点数、已接受闭环因子数、错误数，
以及 MPRF submitted/dropped/completed/failed 计数。若 `dropped>0`，说明服务吞吐跟不
上候选产生速度；先检查 GPU/网络，再考虑提高有界队列，不能让主定位线程同步等待。

## 12. 2026-08-16 实测结果

硬件：x86_64 工作站，NVIDIA RTX 4090；MPRF 使用 CUDA 12.4 / Torch 2.5.0。

### 12.1 模型与进程边界

- DINOv2、SALAD、SONATA 全部按生产配置成功加载；
- 完整 560 px / 2500 点模型烟测通过；
- 标准库异步客户端→HTTP NPZ→独立 GPU 服务→严格 JSON 解析通过；
- 固定源码包和 4 个模型权重的文件大小检查全部通过。

### 12.2 真实目标序列

数据：`2025-06-26-23-01-34`。

- 原始左右目各 157 行；按本次显式 `200 ms` 复现门限排除 20 对，选择 137 对；
- 137/137 全部发布、同步、终止并收到 mapper 回执；
- `tracking_or_processing_failure_count=0`；
- LIO + direct LiDAR 地图生成 28 个关键帧、4 个全局瓦片；
- MPRF 服务收到并持久化 28/28 个候选关键帧，队列扩大到 8 后本次无遗漏；
- 早期重复外观候选均被 LiDAR 几何复核拒绝；
- `q99` 首次通过几何门并等待确认；`q104` 与其一致，随后一次接受 2 条闭环因子；
- iSAM2 修正后，28 个地图关键帧在后台完整重放并原子切到节点地图 revision 29；
- 后续 `q109/q114/q119/q124/q134` 的不合格候选继续被拒绝；`q129` 只有单次
  几何通过，因没有第二次一致确认而未加入因子；
- 节点仍在线时保存服务返回 `success=True`，快照包含 4 个完整瓦片。

另一个不经过 ROS launch 的完整进程边界复核使用 23 个实际关键帧，得到 12 次候选
响应、5 次几何通过和 2 条最终因子，结论与 ROS 运行一致。

这些结果证明：候选有召回、误候选能被挡住、确认后的因子能驱动后端和完整地图
重建。它**不是绝对精度验收**：该数据的保存 FAST-LIO 轨迹是内部参考，不是独立
真值；序列也不到 300 s/1 km。ATE、RPE、闭环后绝对误差和月面误闭环率仍需用
独立真值、长序列与实车数据验收。

## 13. 状态判断与排错

日志状态含义：

| 状态 | 含义 |
|---|---|
| `geometry_rejected` | 外观候选未通过米制几何门，不加因子 |
| `awaiting_temporal_confirmation` | 本次几何通过，但默认还需第二次一致重访 |
| `accepted` | 确认完成，约束已加入全局图 |
| `unknown_query_node` | MPRF 响应对应的 LiDAR 几何尚未注册，结果丢弃 |

常用检查：

```bash
curl -fsS http://127.0.0.1:8765/health
rg "MPRF shortlist|Atomically switched GridMap" ~/.ros/log/python3_*.log
python3 -m json.tool /path/to/output/input_processing_summary.json
ros2 service call /Car/T3/mapping/save_grid_map std_srvs/srv/Trigger '{}'
```

常见问题：

| 现象 | 优先检查 |
|---|---|
| 服务起不来 | `verify_assets.py`、CUDA 可见性、Torch/CUDA 版本、显存 |
| 非本机地址被拒绝 | 两端是否设置同一个 token 环境变量 |
| 有候选但全被拒绝 | 先检查 LiDAR/相机外参、时间、点字段和局部轨迹；不要先放宽 ICP |
| 长时间没有候选 | 健康页 requests/database_records、图像是否有内容、20 s/20 节点间隔 |
| `dropped>0` | GPU 推理速度、网络、关键帧频率；必要时在内存预算内调队列 |
| 闭环后地图不变 | 查 `accepted_loop_factor_count`、geometry revision 和完整重放日志 |
| MPRF 挂掉 | 可直接保持本地 LIO/VO-LIO 运行，或切 `candidate_source:=lidar` |

## 14. 回归入口与仍未完成的外部验收

统一质量门：

```bash
cd /path/to/t3_gridmap_v2_ws
bash scripts/run_quality_gates.sh
```

当前结果：6/6 自有 ROS 2 包构建成功，365/365 测试通过；120 个 Python 文件通过
pyflakes 和编译，shell 入口、MPRF 固定资产、正式消息哈希与冻结 RoMa 哈希通过。

仍需外部完成：

- AGX Orin 64G / JetPack 5.1.1 / ROS 2 Humble 原生编译与持续实时性；
- 实际课题一、课题二正式自定义消息的薄适配；
- 实车相机–LiDAR–IMU 外参和硬件时间同步；
- 大于 300 s、接近或超过 1 km、含真实月面重复地形的闭环评估；
- 独立真值下的 ATE/RPE、闭环前后地图一致性、误闭环率和漏闭环率；
- 课题四对 `fixed` 或 `full_tiles` 的最终接收模式确认。

没有这些外部证据前，应表述为“实现、回归和当前 x86 数据验证通过”，不能写成
“项目精度验收通过”。
