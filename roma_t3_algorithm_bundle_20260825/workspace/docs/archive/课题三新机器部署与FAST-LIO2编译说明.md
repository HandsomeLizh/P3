# 课题三新机器部署与 FAST-LIO2 编译说明

> 适用工程：单一 `t3_gridmap_v2_ws`；RoMa、FAST-LIO2 和 MPRF 均已集成在
> 该工作空间的 `third_party/`、`models/` 和 `mprf_runtime/` 中  
> 开发/复现基线：Ubuntu 22.04 x86_64 + ROS 2 Humble + Cyclone DDS  
> 项目集成目标：AGX Orin 64G + JetPack 5.1.1 + Ubuntu 20.04 aarch64  
> 本机编译验证：2026-08-14

如果准备现在复制工程并希望目标机不联网，优先按
[`offline_deploy/README_离线部署.md`](offline_deploy/README_离线部署.md) 操作。
其中已经把 Focal/arm64 本地 APT 仓库、Humble 固定源码、GTSAM、
Jetson PyTorch 与 CMake 隐式依赖一并封存，可在 Orin 上原生断网编译。
本文件保留标定、接口与验收细节。

## 1. 结论和部署边界

对外交付只有一个顶层 ROS 2 工作空间 `t3_gridmap_v2_ws/`：

```text
课题三工作区 t3_gridmap_v2_ws
  ├─ 原有 RoMa VO（默认）
  ├─ 可选直接 LIO
  ├─ 可选 VO/LIO + GTSAM 局部因子图
  └─ 2.5D 栅格地图及课题接口

FAST-LIO2 固定源码目录 `third_party/fastlio2`
  ├─ Livox-SDK2
  ├─ livox_ros_driver2
  └─ FAST_LIO_ROS2 + ikd-Tree

可选 MPRF 伴随 GPU 进程 mprf_runtime
  ├─ 固定 DINOv2 / SALAD / SONATA 模型
  ├─ 只产生闭环候选
  └─ ROS 主进程负责 LiDAR 复核与轻量 iSAM2
```

两者只通过 ROS 2 标准接口连接：

```text
FAST-LIO2  /Odometry  nav_msgs/msg/Odometry
       └──────────────────────────────► 课题三定位适配层
PointCloud2              sensor_msgs/msg/PointCloud2
       └──────────────────────────────► 课题三直接建图/辅助深度
```

第三方源码虽然随同一工作空间交付，但不放入自研 `src/`：FAST-LIO2 使用
内部隔离构建目录和独立 ROS 2 进程，MPRF 使用独立 Python 环境和伴随进程。
因此不会改变课题三原来的 VO 默认行为，也不会把 GPL 上游源码与自研包混编或加载到同一进程。
课题三默认仍是 `localization_mode=vo`、`fusion_backend=none`；只有显式选择
`lio` 或 `vo_lio` 时才使用 FAST-LIO2。

## 2. 两套机器环境必须区分

### 2.1 项目接口文档锁定的最终平台

`课题五-项目各课题接口协调V2.docx` 明确写的是：

| 项目 | 锁定值 |
|---|---|
| 计算平台 | NVIDIA AGX Orin 64G |
| 系统/架构 | Ubuntu 20.04 LTS、aarch64/Arm64 |
| JetPack | 5.1.1 |
| 编程语言 | 系统 Python 3、C++ |
| ROS/DDS | ROS 2 Humble、Cyclone DDS |
| 图像环境 | OpenCV 4 |

因此最终交付不能只用 x86 工作站结果代替，FAST-LIO2、RoMa 和课题三
必须在 Orin 本机重新编译和测时。

### 2.2 锁定平台存在的现实版本冲突

这不是单纯执行几条 `apt` 命令就能解决的环境：

1. ROS 2 Humble 官方 deb 包面向 Ubuntu 22.04 Jammy，而 JetPack 5.1.1 是
   Ubuntu 20.04 Focal。
2. JetPack 5.1.1 官方栈为 CUDA 11.4、Python 3.8；NVIDIA 对 JetPack 5.1.x
   列出的 GPU PyTorch 最新系列为 2.1.0a。
3. 当前 `t3_gridmap_v2_ws/third_party/RoMa/pyproject.toml` 声明 Python `>=3.9`、PyTorch
   `>=2.5.1`，与上述 JetPack 5.1.1 官方 wheel 不一致。

所以，不能在裸 Focal 系统上套用 Jammy 的
`apt install ros-humble-*` 命令。本工程现在提供了可追溯的原生方案：
PCL/系统依赖从 Focal/arm64 本地 APT 仓库安装，Humble 从固定源码编译，
PyTorch/torchvision 使用固定 Jetson 文件。RoMa/PyTorch 仍必须做 Orin 实机
兼容测试；在测试完成前，不能宣称整套 VO 已可在 JetPack 5.1.1
稳定实时部署。

如果集成方允许升级到 Ubuntu 22.04/JetPack 6 系列，则可以统一 Python 3.10、
Humble 和较新 PyTorch，但这是项目环境基线变更，必须由接口协调方书面确认。

### 2.3 开发机/纯软件复现环境

| 项目 | 必需/建议 | 说明 |
|---|---|---|
| Ubuntu 22.04 x86_64 | 推荐基线 | 与 Humble 官方 deb 和 Livox Humble 说明相符 |
| ROS 2 Humble Desktop | 必需 | 不建议新机器继续使用本机的早期 Rolling |
| Cyclone DDS | 必需 | 本工程脚本统一设置 `rmw_cyclonedds_cpp` |
| GCC/G++、CMake、colcon | 必需 | Ubuntu 22.04 apt 版本即可 |
| PCL、Eigen、Boost、APR | FAST-LIO2 必需 | 均由 apt 脚本安装 |
| Python 3.10 | x86 Humble 必需 | 必须与 Humble `rclpy`/`cv_bridge` ABI 一致 |
| NVIDIA GPU | VO 实时运行建议 | FAST-LIO2 本身只用 CPU；RoMa 才需要 CUDA |
| 32 GB 内存、12 GB 以上独立显存 | x86 建议 | 不是接口硬限制，长序列和 RoMa 会明显受益 |
| 50 GB 以上空闲系统盘 | 工程建议 | 不含 rosbag/数据集、长期地图快照和无损观测存档 |

x86 Humble 不要让 Conda 的另一套 Python 抢在 ROS Python 前面。应用
Python 3.10 并用 `--system-site-packages` 创建虚拟环境，使 apt 安装的
`rclpy`、`cv_bridge` 和消息包仍然可见。

FAST-LIO2 不要求 GPU。若只启用 LIO 进程，可以不安装 CUDA；但当前课题三
mapper 仍以相机/语义帧为触发，完整 VO/双目路径需要可用的 PyTorch 和 RoMa。

长时运行时，mapper 默认把全部关键帧地图证据以 `float64` 无损写入
`<output_dir>/map_observation_archive/*.sqlite3`，RAM 只缓存最近 64 帧。
SQLite、WAL 和 zlib 都来自 Python/系统标准库，不增加第三方部署依赖。
这解决的是内存无上限增长，不会丢弃历史证据；因此磁盘容量仍需按
实际关键帧频率和点数做 1 h / 8 h 压力测试，并监控 `output_dir` 剩余空间。
地图快照另外保留当前和紧邻上一个完整版本，写入期间还会有一份
staging；因此部署容量要按“稳态两份、短时三份压缩快照”预留，不能只按
单份 `grid_map_snapshot` 估算。根 `map_index.yaml` 只在新版本逐层校验通过后原子切换。
磁盘写满、历史位姿缺失或所选 LiDAR 关键帧证据无法构建时，mapper 会保留
上一版完整地图并冻结新地图发布。部署监控应把该状态视为故障，不能通过
删除 pending 证据恢复“表面正常”的输出。

## 3. 新机器需要带走或下载的内容

发布包只保留一个顶层工作空间：

```text
t3_gridmap_v2_ws/
├── src/                       # 本 ROS 2 工作空间
├── scripts/
├── offline_deploy/
├── models/
├── mprf_runtime/
└── third_party/
    ├── RoMa/
    ├── fastlio2/
    ├── MPRF/
    └── fused-local-corr/
```

Orin 发布目录建议使用纯 ASCII 绝对路径，例如 `/opt/t3_gridmap_v2_ws`或
`/data/t3_gridmap_v2_ws`。脚本从工作空间自身定位 RoMa、FAST-LIO2、模型和
离线资产，不依赖它的上级目录名。

`t3_gridmap_v2_ws/third_party/RoMa` 不能简单用上游 RoMa 仓库替代，因为这里的 `roma_vo.py` 含本项目的
双目 VO、关键帧、BA、回环和跟踪状态接口。迁移时应将它与课题三工程一起复制。
从 x86 开发机只复制源码、配置、模型权重和标定文件；不要复制
`build/`、`install/`、`log/`、`build_*`、`install_*` 或 Python 虚拟环境。这些
文件含 x86_64 二进制、本机绝对路径和 ABI 信息，必须在 Orin 上重建。
内置 `third_party/fastlio2` 已故意排除这些编译产物，只保留可迁移的
固定源码。主工作空间编译只扫描 `src/`，FAST-LIO2 由专用脚本隔离构建。

工程内已包含下列固定源码；原生离线脚本会先验证提交，任何源码缺失都会
立即停止，不会在目标机上尝试联网补齐，避免上游更新造成不可重复编译：

| 组件 | 来源 | 固定提交 |
|---|---|---|
| Livox-SDK2 | `Livox-SDK/Livox-SDK2` | `08f523c930b2f0ba1e98a6afaa8d7476bf479908` |
| Livox ROS Driver 2 | `Livox-SDK/livox_ros_driver2` | `4a1def929e5b59c7a8122d19fce6efba581ce9f7` |
| FAST-LIO2 ROS 2 移植 | `Ericsii/FAST_LIO_ROS2` | `2fffc570a25d0df172720bac034fbdb6a13d2162` |
| ikd-Tree 子模块 | `hku-mars/ikd-Tree` | `e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4` |
| MPRF 模型源码 | `DLR-RM/MPRF` | `606938ded985a2b5beb29274403123a760a637fd` |

`hku-mars/FAST_LIO` 官方组织目前也托管 `ROS2` 分支，但该分支 README 自身注明
ROS 2 fork 维护者是 Ericsiii。本工程因此把当前已审核且较新的社区工作树作为固定
候选，同时只依赖其标准 `/Odometry`，以后替换实现不需要修改课题三消息接口。

上表中的 Livox 驱动是当前该 FAST-LIO2 移植的编译依赖，不表示实车必须
使用 Livox。如果实车是 Velodyne、Ouster 或其他雷达，FAST-LIO2 仍可以保留
`livox_ros_driver2` 作为编译依赖，运行时则启动实际型号的驱动和对应 YAML。
在雷达型号尚未确认前，本文的 Mid-360 命令只能作为 Livox 分支示例。

## 4. 安装 ROS 2 和课题三系统依赖

### 4.1 先做平台预检

脚本只读，不会安装或改动系统：

```bash
cd <T3_WS>
bash scripts/check_deployment_platform.sh orin   # AGX Orin 集成机
bash scripts/check_deployment_platform.sh dev    # x86 开发机
```

`[ERROR]` 表示当前不能按目标平台发布；`[WARN]` 表示仍需集成方确认。
空间过时的 ROS 环境变量也会干扰检查，建议在新终端中执行。

### 4.2 x86 Ubuntu 22.04 开发机

按 ROS 2 官方 Humble Ubuntu deb 文档配置软件源并安装 Desktop：

```bash
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

然后安装课题三依赖：

```bash
cd <T3_WS>
export ROS_DISTRO=humble
bash scripts/install_prereqs.sh
```

该脚本会安装编译工具、NumPy/OpenCV/SciPy/Pandas、ROS 消息、`cv_bridge`、
`message_filters`、GridMap 可视化、rosbag2、Cyclone DDS 等。工程内已带
`grid_map_msgs` 的标准消息副本，因此消息接口不依赖另一份源码工作区。

### 4.3 AGX Orin 集成机

不要在裸 Ubuntu 20.04/JetPack 5.1.1 上执行上面的 Jammy
`ros-humble-*` apt 步骤。本工程两个依赖脚本已加保护：检测到
Focal + Humble 时会主动停止，避免从错误软件源混装系统。

先刷好 JetPack 5.1.1 / L4T 35.3.1，然后使用已随工程带走的原生离线包：

```bash
cd <T3_WS>/offline_deploy
bash scripts/verify_bundle.sh
bash scripts/check_native_target.sh orin
bash scripts/install_native_orin.sh
source scripts/enter_native_environment.sh
bash scripts/check_native_runtime.sh
```

详细的本地 APT、源码编译、组件开关和断网校验见
`../offline_deploy/README_离线部署.md`。

## 5. 建立 RoMa Python 环境

### 5.1 x86 Humble：创建与 ROS 相同 Python ABI 的环境

```bash
python3 -m venv --system-site-packages "$HOME/venvs/ros2_roma_cv4"
source "$HOME/venvs/ros2_roma_cv4/bin/activate"
python -m pip install --upgrade pip wheel
```

确认这里必须是 Ubuntu 22.04/Humble 对应的 Python 3.10：

```bash
python --version
python -c "import rclpy, cv2; print(rclpy.__file__, cv2.__version__)"
```

### 5.2 x86 Humble：安装 PyTorch 和本项目 RoMa

先根据新机器显卡驱动，在 PyTorch 官方 `Start Locally` 页面选择 Linux、Pip、
Python 和合适的 CUDA 平台，执行页面给出的安装命令。不要同时混装 Conda PyTorch
和 pip PyTorch。

再安装项目所带 RoMa：

```bash
ROMA_ROOT=<T3_WS>/third_party/RoMa

python -m pip install \
  -c "$ROMA_ROOT/constraints-ros2-humble.txt" \
  -e "$ROMA_ROOT"
```

`fused-local-corr` 是可选加速项，不是首次部署的必要条件；先让普通实现稳定运行，
再根据当前 PyTorch/CUDA 版本单独编译它，可减少 CUDA 扩展 ABI 冲突。

验证：

```bash
python - <<'PY'
import torch
import romatch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("RoMa:", romatch.__file__)
PY
```

首次运行 `roma_outdoor` 还会下载模型权重。正式机器人离线运行前，应提前启动一次并
确认权重已缓存，不能等到现场无网络时再下载。
本项目 `roma_outdoor` 实际需要的两个文件是：

```text
~/.cache/torch/hub/checkpoints/roma_outdoor.pth
~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
```

离线迁移时可以把这两个文件连同目录复制到新机器，然后在断网条件下
实际初始化一次 RoMa，以确认缓存文件和当前代码版本匹配。本机已用文件大小分别为
445,647,516 字节和 1,217,586,395 字节。

### 5.3 AGX Orin：RoMa/PyTorch 不能直接照搬 x86 命令

PyTorch `Start Locally` 页面的通用 x86 CUDA wheel 不适用于 Jetson。Orin 必须使用
NVIDIA 按 JetPack 版本发布的 aarch64 wheel，且 PyTorch、torchvision、CUDA、
Python 必须是同一个兼容组合。

当前 `t3_gridmap_v2_ws/third_party/RoMa/pyproject.toml` 声明 Python `>=3.9` 和 PyTorch
`>=2.5.1`，而 JetPack 5.1.1 官方栈是 Python 3.8/CUDA 11.4，NVIDIA 兼容表中
JetPack 5.1.x 的较新 PyTorch 组合停在 2.1.0a 系列。因此不能通过随意改低
`pyproject.toml` 版本号来宣称完成兼容。

当前 `offline_deploy` 已经固定 NVIDIA JetPack 5.1.1 的 aarch64
PyTorch 2.1 wheel、torchvision 0.16.1 源码和 RoMa 所需本地模型。正确顺序是：

1. 刷好完整的 JetPack 5.1.1，并复制本工程的离线目录；
2. 在 Orin 上记录 `python3 --version`、PyTorch/torchvision/CUDA 版本；
3. 运行 `offline_deploy/scripts/install_native_orin.sh`，不要用 RoMa 的在线高版本依赖覆盖固定环境；
4. 对当前 RoMa 源码做一对图像的模型初始化和推理试验；
5. 再用目标数据的短序列检查显存、帧率和精度；
6. 通过后记录模型文件名和字节数作为发布基线。

断网安装命令为：

```bash
cd <T3_WS>/offline_deploy
bash scripts/install_native_orin.sh
source scripts/enter_native_environment.sh
T3_CHECK_ROMA_MODEL=1 bash scripts/check_native_runtime.sh
```

然后用项目自带的真实一对图像检查：

```bash
cd <T3_WS>
python3 scripts/check_roma_inference.py \
  /path/to/batch_organized/2025-06-26-23-01-34 \
  --roma-root <T3_WS>/third_party/RoMa \
  --pair stereo
```

该脚本使用与 mapper 相同的 `RoMaMatcher`、`coarse_res=560`和
`upsample_res=560`，输出解释器/架构、PyTorch/CUDA、初始化时间、匹配时间、
返回匹配数和峰值 CUDA 显存。它不会在 CUDA 失效时静默退回 CPU，因此可以
暴露错误 wheel 或显存不足。必须用 ROS 对应 Python，不要使用终端中残留的 Conda Python。

这一步需要真实 Orin 环境，无法在当前 x86 机器上验证 Jetson GPU wheel。
如果旧 RoMa 在 PyTorch 2.1/Python 3.8 上不兼容，可以先以 `localization_mode=lio`
完成定位和 ROS 接口验收，但相机驱动、语义输入和 2.5D mapper 仍必须按实际
运行路径验证；不能把这当作完整 VO 发布。

### 5.4 可选 GTSAM

只有 `localization_mode=vo_lio fusion_backend=gtsam_gp` 需要 GTSAM；默认 VO 和
直接 LIO 都不需要。可先尝试与 Python 3.10 匹配的稳定 4.2 wheel：

```bash
python -m pip install "gtsam==4.2"
python - <<'PY'
import gtsam
import gtsam_unstable
print("GTSAM and fixed-lag bindings: OK")
PY
```

第二个导入必须成功，因为本工程 fixed-lag smoother 位于 `gtsam_unstable`。若目标
平台没有合适 wheel，应从 GTSAM 官方稳定 4.2 源码为 ROS 所用的同一套
Python 环境编译，不要换用另一套 Python。Orin 首次上车先保持
`fusion_backend=none`；FAST-LIO2 和基本接口达标后，再单独编译、测时并开启 GTSAM。

### 5.5 可选 MPRF 闭环环境

MPRF 的发布栈是 Python 3.10 / PyTorch 2.5.0 / CUDA 12.4，与 JetPack 5.1.1 的
Python 3.8 / CUDA 11.4 不兼容。推荐在 x86 CUDA 伴随机创建独立环境，Orin 只运行
ROS 2 客户端、本地 LiDAR 安全门和 iSAM2：

```bash
cd <T3_WS>/mprf_runtime
bash scripts/create_environment.sh
python3 scripts/verify_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_models.py --device cuda:0
```

`.venv` 是生成物，不随工程复制。服务、远程令牌、ROS launch、模型哈希、长时间
存储和真实序列结果统一见 `课题三MPRF闭环实现与部署说明.md`。MPRF 默认关闭；
不部署伴随机时，原 VO/LIO 与地图功能不变，也可选择完全本地的
`global_loop_candidate_source:=lidar`。

## 6. 编译课题三工作区

### 6.1 x86 开发机

```bash
cd <T3_WS>
export ROS_DISTRO=humble
export ROS_VENV="$HOME/venvs/ros2_roma_cv4"

bash scripts/build.sh
source scripts/enter_workspace.sh
bash scripts/check_environment.sh
```

`enter_workspace.sh` 支持同一终端重复 source，也支持先 source 后再调用
内部会自行激活的 `run_*`/check/save 包装脚本。自动默认 venv 不存在时
不会导出假 `ROS_VENV`；用户显式配置的缺失路径仍会立即拒绝。

### 6.2 AGX Orin 集成机

从全新源码目录开始，确保没有携带 x86 的 `build/install/log`。如果集成镜像的
Humble 不在 `/opt/ros/humble`，通过 `ROS_SETUP_FILE` 指向真实的 underlay：

```bash
cd <T3_WS>
export ROS_SETUP_FILE=<INTEGRATION_HUMBLE_SETUP>/setup.bash
export ROS_PYTHON_EXECUTABLE=/usr/bin/python3
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

bash scripts/check_deployment_platform.sh orin
bash scripts/build.sh
source scripts/enter_workspace.sh
bash scripts/check_environment.sh
```

`check_environment.sh` 会导入 RoMa/PyTorch；如果第 5.3 节的 Orin 兼容组合还未
确定，它在这里失败是一个未完成的发布项，不应通过屏蔽检查来改成
“通过”。C++/ROS 包可先编译，但完整 VO 发布必须等该检查成功。
`build.sh` 会把架构、系统、ROS、Python 和安装路径写入
`T3_BUILD_INFO.txt`；`enter_workspace.sh` 会拒绝加载不同架构的已编译工作区。
工作区路径含非 ASCII 字符时，构建中间目录会放到 `/tmp`，但 `install/`
使用自包含拷贝安装，不会再链接到可被重启/清理的临时 build 目录。

`check_environment.sh` 应打印 `Environment check: OK`，且以下包都应来自当前
`t3_gridmap_v2_ws/install`：

```text
t1_dataset_sim
t2_basic_semantics
t3_interfaces
t3_semantic_mapping
grid_map_msgs
t3_upstream_adapters
```

构建后运行统一质量门禁：

```bash
bash scripts/run_quality_gates.sh
```

该脚本只有一个测试范围所有者，会依次：

1. 增量重建全部 6 个自有 ROS 2 包；
2. 精确运行顶层静态契约 43 项；
3. 只运行冻结 RoMa 的 6 个数值文件，共 29 项，不会误收集上游模型/benchmark 测试；
4. 自动发现并运行自研 ROS、定位、地图和工具层 267 项；
5. 运行隔离 MPRF 协议、存储和几何测试 11 项；
6. 运行数据格式 7 项；
7. 对当前 120 个 Python 文件执行 `pyflakes` 和隔离缓存的编译检查；
8. 检查所有 shell 入口、MPRF 固定资产、两个正式消息和冻结 RoMa 哈希。

当前完整结果为 **365/365**，其中新增的 8 项用于固定课题一/二适配器的
字段路径和类型契约。编译缓存写到独立 `/tmp` 临时目录并在退出时删除，
不会在源码树留下 `__pycache__`。如只需单独定位某一组件，仍可直接运行对应的
`python3 scripts/test_<component>.py`；完整交付判断必须执行统一门禁，不能凭手工
挑选的一部分测试声称通过。

`sensor_msgs_py` 现为 `t3_semantic_mapping` 的测试依赖，不是运行依赖。
`install_prereqs.sh` 仍安装 `ros-${ROS_DISTRO}-sensor-msgs-py`，是为了使新机器
能运行完整冻结参考测试；实际 mapper 运行不会导入该包。

新机器的数据集联合 launch 默认启用就绪门、最多 2 帧在途和逐帧处理回执，
不再依靠猜测 RoMa 加载时间。每次部署冒烟后应检查输出目录中的
`dataset_playback_summary.json` 和 `input_processing_summary.json`。P2-5 基线只量到
约 0.23～0.25×，typed-buffer 与调试点云等价优化后，当前 x86 仍只有约
0.42～0.46× 传感器时间。`input_processing_summary.json` 中五段耗时显示冻结
RoMa 视觉跟踪占约 82%～87%，
不得把“无损降速完成”当成 Orin 实时通过；课题一实车 ROS 2 驱动也应以 typed
buffer 填充大 payload，避免重新引入逐元素 setter 开销。
需要视觉的路径在当前 x86 仍未达 1×。保存 LIO 的 direct 数据集路径在显式
`enable_imu=false`、不回放 mapper 无消费者的 23,923 条原始 IMU 后，最长同步
序列已完成 231/231 帧、47 次地图修订和 47 条/194,290 点逐字节相同的原始归档，
联合倍率为 1.164×、回调最大 0.994 s。该开关只影响本项目的数据集 publisher/
mapper；真实 FAST-LIO2 仍必须消费它自己的原始 IMU。两类路径都仍需目标机
长时频率和输入对账验收。

还必须检查 `dataset_playback_summary.json.input_selection`。
`complete_input_consumption=true` 的范围是配对、双目时差门限和 `max_frames`
之后的最终选中帧，不是原始目录行数。目标序列实际为 157 共享候选→
34 个 20 ms 时差合格对。它可验证输入拒绝与终止账本，不可作为 157 帧完整
双目精度证据。

统一脚本内部设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 并显式列出冻结 RoMa 的六个
文件，是为了避免开发机上的 ROS pytest 插件或同目录 benchmark/model 测试扩大
收集范围；这不是跳过项目测试，而是固定已经审查过的 29 项数值回归边界。

## 7. 下载并编译 FAST-LIO2

### 7.1 安装系统依赖

x86 Ubuntu 22.04 开发机可以直接执行：

```bash
cd <T3_WS>
export ROS_DISTRO=humble
bash scripts/install_fastlio2_prereqs.sh
```

Orin Focal 必须使用集成镜像批准的依赖源。在执行一键编译前，至少确认
`cmake`、`colcon`、PCL、Eigen、Boost、APR、libpcap、libpng、libusb 及 ROS 包 `pcl_ros`、
`pcl_conversions`、`rclcpp_components` 可用。不要为了缺一个包而混入 Jammy
二进制包。

### 7.2 一键隔离编译 FAST-LIO2

整个 `t3_gridmap_v2_ws` 解压完成后，可先在无 ROS 环境下检查四个内置仓库是否完整：

```bash
cd <T3_WS>
bash scripts/check_fastlio2_sources.sh
```

然后编译：

```bash
FASTLIO_WS=<T3_WS>/third_party/fastlio2

bash scripts/setup_fastlio2_workspace.sh "$FASTLIO_WS"
```

该命令在 x86 和 Orin 上都必须各自原生执行。在 Orin 的定制 Humble 路径下，
使用：

```bash
export ROS_DISTRO=humble
export ROS_SETUP_FILE=<INTEGRATION_HUMBLE_SETUP>/setup.bash
export ROS_PYTHON_EXECUTABLE=/usr/bin/python3
export FASTLIO_BUILD_JOBS=4
bash scripts/setup_fastlio2_workspace.sh "$FASTLIO_WS"
```

脚本会执行以下操作：

1. 验证内置源码并切到上表的固定提交；若源码确实缺失才从上游下载；
2. 初始化固定版本的 ikd-Tree 子模块；
3. 将 Livox-SDK2 编译到 `$FASTLIO_WS/local`，不写 `/usr/local`；
4. 编译 `livox_ros_driver2` 和 `fast_lio`；
5. 检查 ROS 包、CustomMsg 和动态库；
6. 写出 `$FASTLIO_WS/FASTLIO2_BUILD_INFO.txt`，记录架构、系统、ROS
   underlay、安装路径和所有固定提交。

Livox-SDK2 和 FAST-LIO 上游声明支持 ARM，但本次实际编译机是 x86_64。当前
固定的 ROS 2 移植在非 x86 分支上将 `MP_PROC_NUM=1`，不开启其 ikd-Tree
OpenMP 多线程分支。这不改变 ROS 接口，但可能影响 Orin 实时性；先保持上游
代码不动做功能/精度基线，再根据实机 CPU 占用和点云频率决定是否优化。
`enter_fastlio2_workspace.sh` 还会同时检查记录架构和实际 ELF，
如果把 x86-64 安装目录误复制到 aarch64，它会直接拒绝加载。

加载环境：

```bash
export FASTLIO_WS=<T3_WS>/third_party/fastlio2
source <T3_WS>/scripts/enter_fastlio2_workspace.sh
```

验收：

```bash
ros2 pkg prefix livox_ros_driver2
ros2 pkg prefix fast_lio
ros2 pkg executables fast_lio
ros2 interface show livox_ros_driver2/msg/CustomMsg
ros2 launch fast_lio mapping.launch.py --show-args
ros2 launch livox_ros_driver2 msg_MID360_launch.py --show-args
```

## 8. 实车前必须配置的项目

### 8.0 先确认雷达型号和消息字段

- Livox/Mid-360：按下文使用 `livox_ros_driver2/msg/CustomMsg` 和
  `mid360.yaml`。
- Velodyne：启动设备对应的 ROS 2 驱动，使用 `velodyne.yaml`，并核对
  `ring` 和每点 `time` 字段。
- Ouster：启动设备对应的 ROS 2 驱动，使用 `ouster64.yaml`，并核对
  时间单位和线数。

不能只把话题名改对就认为完成适配；`preprocess.lidar_type`、每点时间字段、
时间单位和扫描线数也必须与实际消息一致。

### 8.1 Livox 网络

根据实际雷达修改：

```text
$FASTLIO_WS/src/livox_ros_driver2/config/MID360_config.json
```

至少确认雷达 IP、主机有线网卡 IP 和端口。主机网卡应与雷达处于同一网段；不要把
示例里的 `host_ip` 原样带到另一台机器。配置完成后可先只启动驱动，确认
`/livox/lidar` 和 `/livox/imu` 有稳定频率与递增时间戳。

### 8.2 FAST-LIO2 参数

以 Mid-360 为例，检查：

```text
$FASTLIO_WS/src/FAST_LIO_ROS2/config/mid360.yaml
```

重点参数：

| 参数 | 要求 |
|---|---|
| `common.lid_topic` | 与驱动 CustomMsg 话题一致，通常 `/livox/lidar` |
| `common.imu_topic` | 与 IMU 话题一致，通常 `/livox/imu` |
| `preprocess.lidar_type` | 与实际型号和消息格式一致 |
| `preprocess.timestamp_unit` | 与点内时间字段单位一致 |
| `mapping.extrinsic_T/R` | `T_body_lidar`，即 LiDAR 点变换到 IMU/body |
| `mapping.extrinsic_est_en` | 有可靠标定时建议 `false` |
| `publish.scan_bodyframe_pub_en` | 课题三要使用 body 点云时设为 `true` |

LiDAR 和 IMU 必须使用同一时钟域并硬件同步。FAST-LIO2 的软件时间同步只能作为无法
硬同步时的退路，不能视为同等精度。点云若缺少每点时间，运动去畸变会失效。

### 8.3 三组外参不要混淆

```text
T_body_lidar   FAST-LIO2 内部使用，LiDAR -> IMU/body
T_camera_lidar 课题三语义投影使用，LiDAR -> 左相机
T_lidar_child  课题三 LIO 位姿适配使用，FAST-LIO child -> LiDAR
```

常规 FAST-LIO2 的 child 是 `body`，所以：

```text
T_lidar_child = inverse(T_body_lidar)
```

必须独立标定相机—LiDAR 和 LiDAR—IMU，并用静态场景检查投影。不能因为三者都描述
“位移”就省略旋转或直接把一个矩阵复制给另一个参数。

## 9. 启动顺序

### 9.1 终端 1：Livox 驱动

```bash
export FASTLIO_WS=<T3_WS>/third_party/fastlio2
source <T3_WS>/scripts/enter_fastlio2_workspace.sh

ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

FAST-LIO2 的 Livox 路径需要带每点相对时间的 `livox_ros_driver2/msg/CustomMsg`，因此
应启动 `msg_*` launch，而不是只输出普通 `PointCloud2` 的 RViz launch。

### 9.2 终端 2：FAST-LIO2

```bash
export FASTLIO_WS=<T3_WS>/third_party/fastlio2
source <T3_WS>/scripts/enter_fastlio2_workspace.sh

ros2 launch fast_lio mapping.launch.py \
  config_file:=mid360.yaml \
  rviz:=false
```

确认输出：

```bash
ros2 topic info /Odometry --verbose
ros2 topic hz /Odometry
ros2 topic echo /Odometry --once
```

本移植发布：

```text
/Odometry              nav_msgs/msg/Odometry
header.frame_id         camera_init
child_frame_id          body
/cloud_registered_body  sensor_msgs/msg/PointCloud2
```

### 9.3 终端 3：课题三直接使用 LIO

先使用 LIO 定位、保持原双目地图点来源，是最简单且接口歧义最少的第一阶段：

```bash
cd <T3_WS>
source scripts/enter_workspace.sh

ros2 launch t3_semantic_mapping mapper_only.launch.py \
  data_dir:=/path/to/calibration_or_sequence \
  algorithm_root:=<T3_WS>/third_party/RoMa \
  output_dir:=/path/to/output \
  localization_mode:=lio \
  fusion_backend:=none \
  lio_odometry_topic:=/Odometry \
  lio_expected_parent_frame:=camera_init \
  lio_expected_child_frame:=body \
  lio_child_to_lidar_transform:="[<T_lidar_body 的 16 个行主序数值>]" \
  lio_child_to_lidar_confirmed:=true \
  enable_lidar_mapping:=false
```

### 9.4 同时用 LiDAR 点更新 2.5D 地图

Livox 的 `msg_*` launch 给 FAST-LIO2 提供的是 CustomMsg，而课题三地图输入类型是
标准 `sensor_msgs/msg/PointCloud2`，不能把 `/livox/lidar` CustomMsg 直接接到
课题三 PointCloud2 订阅。

无需再起第二个雷达驱动，可以使用 FAST-LIO2 已去畸变的 body-frame 点云。
这是高级用法：需要另外准备一份符合本工程 JSON schema 的 body→原始左相机外参：

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  ... \
  localization_mode:=lio \
  fusion_backend:=none \
  lio_odometry_topic:=/Odometry \
  lio_expected_parent_frame:=camera_init \
  lio_expected_child_frame:=body \
  lio_child_to_lidar_transform:="[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]" \
  lio_child_to_lidar_confirmed:=true \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=direct \
  lidar_topic:=/cloud_registered_body \
  lidar_expected_frame_id:=body \
  lidar_extrinsics_path:=/path/to/body_to_left_camera_extrinsics.json \
  lidar_temporal_mode:=upstream_deskewed
```

`upstream_deskewed` 是这条生产路径的必要显式契约：FAST-LIO2 源码会用
高频 IMU 将每个点去畸变到 `lidar_end_time`，然后在
`/cloud_registered_body` 发布 `feats_undistort`。课题三不再解析该点云中的
`curvature`/类时间字段，避免二次 deskew；它只用严格包围的 LIO 位姿
将整帧点云从 `lidar_end_time` 变换到图像关键帧时刻。

此时课题三把输入传感器帧视为 `body`，所以传入的语义投影外参必须是
`T_camera_body`，而不是未经组合的 `T_camera_lidar`。JSON 仍必须包含
`LiDAR_to_Cam_L.RT`，但由于此时输入帧是 `body`，该 `RT` 保存的实际含义是
`T_Cam_L_Body`（body 点变换到未矫正左相机）；节点会再组合左相机矫正旋转。
child 到输入点云帧为同一 `body`，所以 `lio_child_to_lidar_transform` 是单位阵。
在这份 body 外参没有通过静态投影检查前，保持 `enable_lidar_mapping=false`。若仍要输入真正的原始
LiDAR-frame PointCloud2，则需增加经过验证的 CustomMsg→PointCloud2 转换节点，并
恢复 `T_camera_lidar` 与 `T_lidar_body`。

如果不需要 360° 点云直接建图，而是想用去畸变点云校正双目远距离地图深度，
可将同一输入切换为：

```bash
ros2 launch t3_semantic_mapping mapper_only.launch.py \
  ... \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=assist \
  lidar_topic:=/cloud_registered_body \
  lidar_expected_frame_id:=body \
  lidar_extrinsics_path:=/path/to/body_to_left_camera_extrinsics.json \
  lidar_temporal_mode:=upstream_deskewed \
  pointcloud_assist_max_points_per_keyframe:=20000 \
  lidar_fallback_to_stereo:=true
```

`assist` 默认保留 5 m 内的双目地图点，只替换有点云投影支持的远点，并丢弃
无支持远点。它不改 RoMa VO/PnP 深度，也不保留相机视野外的 360° 点云。
上述命令开启回退后，点云短时缺帧也只融合 5 m 内双目点，不会把无支持远点重新放回地图。
这两种模式的输出话题和消息类型完全相同；只有内部地图几何证据不同。

若输入不是 FAST-LIO2 已去畸变点云，而是标准原始 `PointCloud2`，
只有上游书面确认标量字段满足
`header.stamp + point_offset = point acquisition time` 时才能使用：

```text
lidar_temporal_mode=per_point
lidar_point_time_field=time
lidar_point_time_unit=sec        # 按实际驱动改为 ms/us/ns
lidar_point_time_max_abs_sec=0.20
lidar_deskew_time_bin_sec=0.001
```

`per_point` 使用外部 LIO 位姿插值，不等价于 FAST-LIO2 内部高频 IMU
运动传播。缺字段、单位错误、位姿不包围或外参不一致时节点会拒绝观测，
不会自动退成未补偿点云。详细证据见
`accuracy_results/p1_3_lidar_temporal_20260815/`。

### 9.5 可选 VO/LIO 因子图

GTSAM 验证完成后才启用：

```bash
localization_mode:=vo_lio fusion_backend:=gtsam_gp
```

这一层融合 VO 与 LIO 的位姿约束及 WNOA 连续时间先验。FAST-LIO2 已经内部紧耦合
IMU，因此课题三因子图不会重复加入同一份原始 IMU 因子。

## 10. 与其他课题保持不变的输出

增加 FAST-LIO2 不改变课题三对外发布类型：

```text
/Car/T3/mapping/grid_map
  grid_map_msgs/msg/GridMap

/Car/T3/semantic/incremental_map
  t3_interfaces/msg/IncrementalSemanticMap
```

默认 `incremental_publish_mode=fixed`，仍发布原有 60 m × 60 m 固定范围解释；内部
地图使用稀疏瓦片继续增长。未与接收方确认前，不要切换 `rolling_window` 或
`full_tiles`，因此迁移 FAST-LIO2 不会顺带改变课题四看到的地图消息语义。

双方确认并显式启用 `full_tiles` 后，还需同时使用：

```text
/Car/T3/semantic/map_batch
  t3_interfaces/msg/SemanticMapBatch

/Car/T3/mapping/replay_full_map
  std_srvs/srv/Trigger
```

这两个端点在默认 `fixed` 下不会创建。接收端必须等 BEGIN/COMPLETE 与所有
瓦片载荷到齐后再原子提交，详细状态机见使用说明第 8.2/16 节。

完整话题、图层、坐标和增量地图约定见
`课题三定位与2.5D地图使用说明.md`。

## 11. rosbag 和数据集测试

### 11.1 正式 rosbag

bag 至少需要实际使用的图像、IMU、语义和 LiDAR 话题。推荐启动顺序为：驱动/回放
所需解码环境 → FAST-LIO2 → 课题三 mapper → `ros2 bag play`。所有节点必须使用
同一个 `ROS_DOMAIN_ID`、RMW 实现和 `/clock` 策略。

若 bag 已保存 `/Odometry`，可先用它验证课题三接口；最终精度测试仍应从同一 bag
中的原始 LiDAR + IMU 重新运行 FAST-LIO2，避免把保存轨迹误当真值。

### 11.2 当前目标序列的确定性接口回放

```bash
cd <T3_WS>
CUDA_VISIBLE_DEVICES=0 bash scripts/run_target_lio_replay.sh \
  /path/to/batch_organized/2025-06-26-23-01-34
```

该命令回放数据集中已保存的 FAST-LIO 估计，只验证时间、帧、地图和接口，不评价
FAST-LIO 自身精度。该序列原始 PCD 保留了约 `-0.10~+0.00064 s`
的 header-relative `time`，脚本已显式选择 `per_point/time/sec`。
2026-08-15 的 6 帧 ROS 2 实跑得到 5 条有效 LIO 位姿、2 个 LiDAR 关键帧、
revision 2 和 4 个可重载瓦片，但保存的约 10 Hz 轨迹不是高频 IMU deskew
替代品，仍只作确定性接口测试。

同一序列的双目时间戳并不同步：157 个共享索引中只有 34 对通过默认
20 ms 门限，最大合格输入空洞为 50.312 s。不要在新机器上把门限放宽到数百毫秒来
“找回”其他帧；完整 VO/地图精度验收应换用硬件同步的 rosbag 或实时 ROS 2 输入。

## 12. x86 本机实际编译结果

2026-08-14 已在当前 x86_64 旧机器直接从随工程携带的固定源码完成一次
隔离重编译，没有修改 FAST-LIO2 算法源码：

```text
主机：Ubuntu 20.04.6 + ROS 2 Rolling（早期版本）
GCC：9.4.0
PCL：1.10
Eigen：3.3.7
固定源码：t3_gridmap_v2_ws/third_party/fastlio2
隔离构建产物：/tmp/fastlio2_bundled_source_validation
固定提交检查：Livox-SDK2、livox_ros_driver2、FAST_LIO_ROS2、ikd-Tree，4/4 通过
Livox-SDK2：编译和用户目录安装成功
ROS 2 包：livox_ros_driver2、fast_lio，2/2 编译成功
```

启动级检查结果：

```text
节点：/laser_mapping
/Odometry 类型：nav_msgs/msg/Odometry
/Odometry 发布端：1
QoS：Reliable、Keep Last、Depth 20
/cloud_registered_body：sensor_msgs/msg/PointCloud2
动态库：无 not found
SIGINT 退出：重建线程正常终止，节点正常退出
```

隔离构建把 `build/install/log/local/build_sdk2` 全部定向到源码目录外，验证了
复制 `t3_gridmap_v2_ws` 后确实可以只靠其中的固定源码重新生成产物。上面的隔离产物
只是当前机器的复验目录，不属于需要复制到新机器的工程内容。

同一台 x86 机器上，使用系统 Python 3.8.10、PyTorch 2.2.0+cu121、
RTX 4090 和目标序列 `2025-06-26-23-01-34` 首帧双目图像运行
`check_roma_inference.py` 已成功：返回 2,000 组匹配，模型初始化 7.476 s，
单次匹配 0.949 s，PyTorch 峰值显存约 2,790 MiB。这仅是当前 x86 GPU
的功能/性能冒烟结果，不能外推为 Orin 时延或兼容结论。

旧主机的 PCL 配置还打印了 pcap/png/libusb 可选 IO 特性关闭警告，不影响本次
FAST-LIO2 节点和 `/Odometry` 检查。新机器依赖脚本已显式列入对应开发包；
如果 Orin 仍有该警告，使用 PCL pcap 离线读取前必须单独验证。

当前机器的旧 ament 不能读取 CMake 3.28/3.29 新格式的 export 文件，所以本机验证
临时使用官方 CMake 3.16.9，并通过脚本的 `FASTLIO_*_BASE` 参数将复验产物放到
独立目录。这是旧 Rolling 主机的工具链兼容措施，不是 FAST-LIO2 算法修改；
x86 Humble 开发机可按第 7 节直接构建，Orin 则必须先满足第 4.3 节的集成环境。

本机没有连接实物 Livox，因此上述结果只证明 x86 上“下载、编译、动态链接、
launch 和 ROS 接口正常”。它不证明 aarch64 可编译、Orin 实时性、Jetson RoMa
兼容性或实车轨迹精度。这些项目必须在目标 Orin 和实物传感器上按第 13 节验收。

## 13. 上车验收清单

按顺序检查，前一项失败不要继续跑长序列：

1. `check_deployment_platform.sh orin` 无错误，记录 aarch64、JetPack/L4T、
   Humble、Cyclone DDS 和 Python 实际版本；
2. 用 `tegrastats` 而不是 x86 的 `nvidia-smi` 监视 Orin；记录
   `nvpmodel -q`、功耗模式、温度和是否开启锁频；
3. 需要 VO 时，`torch.cuda.is_available()` 为 `True`，并完成实际 RoMa 一对图像推理；
4. 六个课题三自有 ROS 包都来自 Orin 本机构建的当前工作区；
5. `livox_ros_driver2` 和 `fast_lio` 都来自 Orin 本机构建的独立 FAST-LIO2 工作区；
6. 雷达和 IMU 频率稳定，时间戳单调且同步，点云具有正确每点时间；
7. FAST-LIO2 静止初始化完成，静止时位姿不持续明显漂移；
8. `/Odometry` 的 frame 严格为期望的 `camera_init -> body`；
9. 课题三日志中 LIO 接收数持续增长，拒绝数不增长；
10. 相机—LiDAR 语义投影在近、中、远距离均对齐；
11. 课题三启动日志显示 `temporal=upstream_deskewed`，并在主动断开
    LIO 包围样本时拒绝点云观测，不用 nearest 位姿继续发布假精确地图；
12. 直线、原地旋转、闭环返回起点三种小场景分别测试；
13. 在固定功耗模式下统计端到端位姿和地图更新延迟，不只统计内部瓦片提交；
14. 位姿处理和地图更新分别使用不少于 100 组有效输入、不少于 10 次
    独立重复，统计端到端平均/中位数/P95/最大值并检查 `<=2 s`；
15. 定姿稳定度使用 `T>300 s`，定姿和定位精度分别不少于 5 次独立重复；
    定位路程 `D>=1 km`，用独立真值和 SE(3) 对齐评价，闭环误差只作辅助诊断；
16. 地图有效覆盖 `>=0.5 km²`，存储量以可完整恢复的必要数据计算
    `<=3 GB/km²`，存储测试不少于 3 次独立重复。
17. 长时路线中重复调用地图保存，确认根 `map_index.yaml` 每次都可加载、
    `.versions` 只保留当前/上一版，并对保存中 SIGINT 做一次故障演练。

## 14. 常见故障

| 现象 | 原因与处理 |
|---|---|
| Orin 上找不到 `ros-humble-*` | 官方 Humble deb 是 Jammy 包；使用 `offline_deploy` 中的 Focal 本地 APT + Humble 固定源码，不要混入 Jammy 包 |
| x86 编译结果在 Orin 不能运行 | 架构和 ABI 不同；只复制源码和数据，在 aarch64 上重新编译 |
| Jetson 无法安装 RoMa 声明的 PyTorch | JetPack 5.1.1 与当前 RoMa 元数据存在版本冲突；使用项目 wheelhouse 并先做实机兼容试验 |
| `No module named catkin_pkg` | Conda/错误 Python 被 CMake 选中；清理终端并用 ROS 对应 `/usr/bin/python3` 重编 |
| `Failed to find exported target names` | 旧 Rolling ament 与新 CMake 不兼容；正式机改用 Humble，本机兼容构建使用 CMake 3.16.9 |
| `liblivox_lidar_sdk_shared.so: not found` | 未加载用户目录 SDK；用 `source scripts/enter_fastlio2_workspace.sh` |
| 找不到 `pcl_ros` | 运行 `install_fastlio2_prereqs.sh`，不要只装系统 `libpcl-dev` |
| FAST-LIO2 收不到 Livox 点云 | 启动了错误 launch、IP 不同网段或话题不一致；FAST-LIO2 Livox 路径应接 CustomMsg |
| `/Odometry` 不发布 | 检查 LiDAR/IMU 时间、每点时间、初始化静止状态、配置中的型号和话题 |
| 课题三拒绝全部 LIO | 检查 parent/child frame、时间偏差以及 `lio_child_to_lidar_confirmed` |
| 精确点云模式持续拒绝观测 | 查 `header.stamp` 定义、点时间字段/单位、LIO 是否严格包围整个扫描，以及 mapping/LIO 是否使用同一 `T_camera_sensor`；不要改回 `none` 掩盖契约错误 |
| LiDAR 语义投影错位 | 外参方向、输入点云 frame 或相机整流矩阵错误；不要用 `T_camera_lidar` 代替 `T_camera_body` |
| 同机能发现、跨机发现不了 | 两机 `ROS_DOMAIN_ID`、RMW、网卡/组播或防火墙不一致 |
| RoMa 首次启动很慢/失败 | 提前下载模型权重，确认 CUDA wheel 与驱动匹配，并避免混入旧 CUDA 扩展 |

## 15. 第三方来源与许可证注意

- Livox-SDK2 和 Livox ROS Driver 2 使用各自上游许可证；驱动官方说明其定位偏测试和
  调试工具，正式产品仍应完成压力、断线恢复和异常包测试。
- FAST-LIO_ROS2 根目录许可证为 GPL-2.0，而其 `package.xml` 标注 BSD，两者不一致。
  在许可证问题确认前按更严格的 GPL-2.0 对待，并维持当前独立进程/隔离构建边界。
- RoMa 上游主体为 MIT，DINOv2 部分另有 Apache 2.0 声明；本项目修改文件也应随
  工程一起保留来源和许可证。
- 固定 MPRF 上游仓库为 MIT；DINOv2、SALAD、SONATA 代码与模型权重仍需分别按其
  上游条款复核，尤其在工程外发或商用前。MPRF 服务与 ROS 主进程保持独立环境。

上游参考：

- <https://github.com/ros2/ros2_documentation/blob/humble/source/Installation/Ubuntu-Install-Debs.rst>
- <https://developer.nvidia.com/embedded/jetpack-sdk-511>
- <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>
- <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html>
- <https://github.com/hku-mars/FAST_LIO/tree/ROS2>
- <https://github.com/Ericsii/FAST_LIO_ROS2>
- <https://github.com/Livox-SDK/Livox-SDK2>
- <https://github.com/Livox-SDK/livox_ros_driver2>
- <https://github.com/Parskatt/RoMa>
- <https://github.com/borglab/gtsam>
- <https://github.com/DLR-RM/MPRF>
- <https://pytorch.org/get-started/locally/>
