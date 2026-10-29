# 课题三 Orin 原生安装主线（已有 ROS 2）

更新日期：2026-08-17  
交付目录：`t3_gridmap_v2_delivery`  
工作空间：`t3_gridmap_v2_ws`（名称不能修改）  
目标平台：AGX Orin 64G、JetPack 5.1.1、Ubuntu 20.04 aarch64、ROS 2 Humble

## 1. 使用方法

本文只保留一条安装主线：

```text
加载已有 ROS 2
-> 连接随包源码和模型
-> 在 Orin 编译完整 FAST-LIO2 后端（默认 VO 不会启用它）
-> 配置 T1/T2
-> 选择工作空间外的 venv 或 Conda 编译课题三
-> 启动默认 RoMa VO 模式
```

执行顺序：

- 推荐：第 2 节 → 第 3 节 → 第 4 节 → 第 5A 节 → 第 6 节；
- 使用 Conda：第 2 节 → 第 3 节 → 第 4 节 → 第 5B 节 → 第 6 节。

本文假设 ROS 2 Humble 已经安装。工程代码的真实默认值是
`localization_mode:=vo` 和 `fusion_backend:=none`，即启用 RoMa VO，不是纯 LIO。
因此默认主线必须先准备与 JetPack 匹配的 PyTorch、torchvision 和 RoMa
运行依赖；GTSAM 和 MPRF 仍是可选项。

## 2. 公共步骤：加载 ROS、准备交付包

先把整个交付目录复制到 Orin 的 Linux 文件系统。以下假设位置为：

```text
/opt/t3_gridmap_v2_delivery
```

先把所有需要长期保留的 `export` 变量安装成用户配置，这一步只做一次：

```bash
mkdir -p "$HOME/.config/t3_gridmap_v2"

if [ ! -f "$HOME/.config/t3_gridmap_v2/environment.sh" ]; then
  install -m 0644 \
    /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/t3_environment.sh.example \
    "$HOME/.config/t3_gridmap_v2/environment.sh"
fi

nano "$HOME/.config/t3_gridmap_v2/environment.sh"
```

在编辑器中一次性确认：

- `ROS_SETUP_FILE`：已有 Humble 的 `install/setup.bash`；
- `T3_DELIVERY`：交付目录，其下工作空间名必须仍为 `t3_gridmap_v2_ws`；
- `T3_CALIB_DIR`：真实双目标定目录；
- `T3_PROFILE`：`standard` / `t1` / `t2` / `t1_t2`；
- `T3_DEFAULT_ENV_MODE`：推荐 `venv`，使用 Conda 时改为 `conda`；
- `ROS_DOMAIN_ID`：与课题一、二、四的机器一致；
- 使用 LiDAR/LIO 时再填 `T3_LIDAR_*`、`T3_FASTLIO_*`、`T3_LIO_*`。

保存退出后，当前首次安装终端执行：

```bash
source "$HOME/.config/t3_gridmap_v2/environment.sh"
source "$ROS_SETUP_FILE"

echo "ROS_DISTRO=$ROS_DISTRO"
command -v ros2
command -v colcon
```

`ROS_DISTRO` 必须是 `humble`。

检查本项目需要的 ROS 包：

```bash
for package_name in \
  ament_cmake \
  ament_cmake_auto \
  cv_bridge \
  geometry_msgs \
  launch_ros \
  message_filters \
  nav_msgs \
  pcl_conversions \
  pcl_ros \
  rclcpp_components \
  rclpy \
  rmw_cyclonedds_cpp \
  sensor_msgs \
  std_msgs \
  std_srvs \
  tf2_ros; do
  ros2 pkg prefix "$package_name" >/dev/null 2>&1 \
    && echo "[OK] $package_name" \
    || echo "[MISSING] $package_name"
done

/usr/bin/python3 -c \
  'import rclpy, cv2, numpy, scipy, yaml; print("[OK] ROS system Python")'
```

如果全部显示 `[OK]`，不需要运行任何依赖安装脚本，继续执行：

```bash
cd "$T3_DELIVERY"
bash prepare_t3_gridmap_v2_ws.sh

cd "$T3_WS"
T3_EXPECT_MODEL_PROFILE=full bash scripts/verify_model_profile.sh
```

准备后会得到两个相对链接：

```text
t3_gridmap_v2_ws/third_party
  -> ../t3_gridmap_v2_ws_dependencies/runtime_sources/third_party

t3_gridmap_v2_ws/models
  -> ../t3_gridmap_v2_ws_dependencies/runtime_assets/models
```

如果有 `[MISSING]`，先看第 8 节，不要继续编译。

## 3. 公共步骤：在 Orin 编译 FAST-LIO2

即使 ROS、PCL、Eigen、Boost 已经安装，FAST-LIO2 自带源码仍需在每台新 Orin 上
原生编译一次。这是把完整后端准备好。统一环境脚本会默认加载
FAST-LIO2 overlay，但默认 RoMa VO 启动参数不会启动或使用它；只有切换到
`lio` / `vo_lio`，或点云确实来自 FAST-LIO2 时才会消费它的输出。

依次执行：

```bash
cd "$T3_WS"
source "$ROS_SETUP_FILE"

bash scripts/check_fastlio2_sources.sh

FASTLIO_OFFLINE=1 \
FASTLIO_BUILD_JOBS=4 \
bash scripts/setup_fastlio2_workspace.sh

source "$T3_WS/scripts/enter_fastlio2_workspace.sh"

ros2 pkg prefix fast_lio
ros2 pkg prefix livox_ros_driver2
```

这一步编译：

- Livox-SDK2；
- livox_ros_driver2；
- FAST_LIO_ROS2；
- ikd-Tree。

编译结果保存在依赖目录的独立 `fastlio2/{build_sdk2,build,local,install,log}`，
不会进入主 `src/`。

这一步只做一次。完成整个工程编译后，以后新终端直接使用
第 7 节的统一环境脚本；下面这条只在需要单独诊断 FAST-LIO2 时使用：

```bash
source "$T3_WS/scripts/enter_fastlio2_workspace.sh"
```

## 4. 公共步骤：配置 T1/T2

正式消息包放到：

```text
t3_gridmap_v2_ws/src/vendor_interfaces/<正式接口包>/
```

修改：

```text
t3_gridmap_v2_ws/src/t3_upstream_adapters/config/interface_mapping.yaml
```

必须替换真实的消息类型、字段、话题、QoS、类别映射和所有 `CHANGE_ME_*`。

选择启动 profile：

- 上游已经直接发布标准话题：`standard`；
- 只启用课题一适配器：`t1`；
- 只启用课题二适配器：`t2`；
- 同时启用课题一、二适配器：`t1_t2`。

如果正式接口尚未配置，先使用 `standard`，不要启动带占位符的适配器。

## 5A. 推荐环境：工作空间外的 venv

这一版保留系统 ROS Python 包，同时把 RoMa/PyTorch 等 Python 库放在
`$HOME/t3_gridmap_v2_runtime/ros_venv`，不写入代码工作空间。

依次执行：

```bash
# 只在第一次配置时创建；已经存在就不要重建。
if [ ! -f "$ROS_VENV/bin/activate" ]; then
  /usr/bin/python3 -m venv --system-site-packages "$ROS_VENV"
fi

source "$ROS_SETUP_FILE"
source "$ROS_VENV/bin/activate"
unset ROS_PYTHON_EXECUTABLE

# 这一环境应已装好 Orin 兼容的 RoMa/PyTorch 运行依赖；
# 第 9 节说明随包候选缓存和兼容性限制。
PYTHONPATH="$T3_WS/third_party/RoMa${PYTHONPATH:+:$PYTHONPATH}" \
python -c \
  'import rclpy, cv2, torch, torchvision, romatch; from cv_bridge import CvBridge; print("[OK] RoMa + ROS Python environment")'

cd "$T3_WS"
bash scripts/build.sh
source scripts/enter_t3_environment.sh venv

ros2 pkg prefix t3_interfaces
ros2 pkg prefix t3_semantic_mapping
ros2 pkg prefix t3_upstream_adapters
ros2 pkg prefix grid_map_msgs
```

四个路径都应位于当前 `t3_gridmap_v2_ws/install`。

## 5B. Conda 环境版

这一版假设 Orin 已经安装 Conda 或 Miniforge。Conda 的 Python 小版本必须与构建
现有 ROS 2 的系统 Python 完全一致，通常为 3.8。

先加载 Conda 并创建工作空间外的环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

ROS_PYTHON_VERSION="$(/usr/bin/python3 -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

conda create -y \
  -p "$T3_CONDA_ENV" \
  "python=$ROS_PYTHON_VERSION" \
  pip

conda activate "$T3_CONDA_ENV"
```

让相同 Python 小版本的 Conda 环境看到 Ubuntu 系统 Python 包：

```bash
CONDA_SITE_PACKAGES="$(python -c \
  'import site; print(site.getsitepackages()[0])')"

/usr/bin/python3 - <<'PY' > \
  "$CONDA_SITE_PACKAGES/ubuntu_system_dist_packages.pth"
import sys

for path in sys.path:
    if "dist-packages" in path:
        print(path)
PY
```

再加载现有 ROS，并明确告诉项目脚本使用 Conda Python：

```bash
source "$ROS_SETUP_FILE"

unset ROS_VENV

python -c \
  'import rclpy, cv2, numpy, scipy, yaml; from cv_bridge import CvBridge; print("[OK] Conda + ROS Python")'
python -m colcon --help >/dev/null

cd "$T3_WS"
ROS_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" bash scripts/build.sh
source scripts/enter_t3_environment.sh conda

python -c \
  'import rclpy, cv2, numpy, scipy, yaml, torch, torchvision; from cv_bridge import CvBridge; print("[OK] installed RoMa environment")'

ros2 pkg prefix t3_interfaces
ros2 pkg prefix t3_semantic_mapping
ros2 pkg prefix t3_upstream_adapters
ros2 pkg prefix grid_map_msgs
```

Conda 模式不使用 `ROS_VENV`。统一入口脚本会在激活 Conda 后自动设置
`ROS_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"`，不需要写入持久配置。

如果 `import rclpy` 或 `import cv_bridge` 失败，说明 Conda Python 与现有 ROS 的
Python ABI 不一致；不要继续混装，改用第 5A 节的 venv。

Jetson 上不要直接安装普通 x86 CUDA PyTorch。需要 RoMa 时只能使用与 JetPack、
Python、CUDA 和 aarch64 匹配的 Jetson wheel。

## 6. 公共步骤：启动默认 RoMa VO

### 6.1 默认：RoMa VO 定位 + 双目几何建图

先准备输出目录和现场参数：

```bash
mkdir -p "$T3_RUNTIME_ROOT/output"

echo "T3_PROFILE=$T3_PROFILE"
echo "T3_CALIB_DIR=$T3_CALIB_DIR"
```

如果这两项不对，只修改
`$HOME/.config/t3_gridmap_v2/environment.sh`，不要再在每个终端重复 `export`。

启动：

```bash
cd "$T3_WS"

bash scripts/run_t1_t2_adapter_template.sh \
  --profile "$T3_PROFILE" \
  localization_mode:=vo \
  fusion_backend:=none \
  enable_lidar_mapping:=false \
  enable_global_loop_closure:=false \
  data_dir:="$T3_CALIB_DIR" \
  output_dir:="$T3_RUNTIME_ROOT/output"
```

默认 VO 需要左图、右图、语义图和双目标定，会初始化 RoMa；不消费
`/Odometry`，也不要求 LiDAR 外参。只有显式改成 `lio` 或 `vo_lio` 时，
才需要加载 FAST-LIO2 并配置对应外参、frame 和 `/Odometry`。

### 6.2 切换定位来源和建图几何来源

这是两组独立开关：

| 作用 | 参数 | 可选值 |
|---|---|---|
| 决定位姿从哪里来 | `localization_mode` | `vo` / `lio` / `vo_lio` |
| 是否用点云做地图几何 | `enable_lidar_mapping` | `false` / `true` |
| 点云怎么参与建图 | `lidar_mapping_mode` | `direct` / `assist` |

`direct` 表示点云直接生成 2.5D 地图几何；`assist` 表示保留双目几何，
只用时间对齐的点云修正较远的双目深度。

默认 RoMa VO + 双目建图直接使用第 6.1 节的完整启动命令。下面其他模式也都给出
完整命令，不再单独列出无法执行的参数片段。
命令中的 topic、frame、外参路径和 16 个数的变换矩阵全部从
`$HOME/.config/t3_gridmap_v2/environment.sh` 读取；先填好该文件，下面代码块
就可直接执行。

RoMa VO 定位 + LiDAR 直接建图：

```bash
cd "$T3_WS"
bash scripts/run_t1_t2_adapter_template.sh \
  --profile "$T3_PROFILE" \
  localization_mode:=vo \
  fusion_backend:=none \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=direct \
  lidar_topic:="$T3_LIDAR_TOPIC" \
  lidar_extrinsics_path:="$T3_LIDAR_EXTRINSICS" \
  lidar_expected_frame_id:="$T3_LIDAR_FRAME" \
  lidar_temporal_mode:=none \
  lidar_fallback_to_stereo:=false \
  enable_global_loop_closure:=false \
  data_dir:="$T3_CALIB_DIR" \
  output_dir:="$T3_RUNTIME_ROOT/output"
```

这个 `none` 是兼容模式，不做扫描内运动补偿，可用于先联调，不应直接当作
高速运动精度验收配置。

RoMa VO 定位 + LiDAR 辅助双目深度：

```bash
cd "$T3_WS"
bash scripts/run_t1_t2_adapter_template.sh \
  --profile "$T3_PROFILE" \
  localization_mode:=vo \
  fusion_backend:=none \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=assist \
  lidar_topic:="$T3_LIDAR_TOPIC" \
  lidar_extrinsics_path:="$T3_LIDAR_EXTRINSICS" \
  lidar_expected_frame_id:="$T3_LIDAR_FRAME" \
  lidar_temporal_mode:=none \
  lidar_fallback_to_stereo:=true \
  enable_global_loop_closure:=false \
  data_dir:="$T3_CALIB_DIR" \
  output_dir:="$T3_RUNTIME_ROOT/output"
```

纯 LIO 定位 + LiDAR 直接建图：
首先在另一个终端启动已经按现场 LiDAR/IMU 修改好配置的 FAST-LIO2：

```bash
source /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/enter_t3_environment.sh

ros2 launch fast_lio mapping.launch.py \
  config_path:="$T3_FASTLIO_CONFIG_DIR" \
  config_file:="$T3_FASTLIO_CONFIG_FILE" \
  rviz:=false
```

用户配置文件中的 `T3_FASTLIO_CONFIG_FILE` 要与现场雷达一致，并确认
FAST-LIO2 正在 `T3_LIO_ODOMETRY_TOPIC` 发布里程计；上面的终端保持运行。
然后在新终端启动课题三：

```bash
source /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/enter_t3_environment.sh

cd "$T3_WS"
bash scripts/run_t1_t2_adapter_template.sh \
  --profile "$T3_PROFILE" \
  localization_mode:=lio \
  fusion_backend:=none \
  enable_imu:=false \
  lio_odometry_topic:="$T3_LIO_ODOMETRY_TOPIC" \
  lio_expected_parent_frame:="$T3_LIO_PARENT_FRAME" \
  lio_expected_child_frame:="$T3_LIO_CHILD_FRAME" \
  lio_child_to_lidar_confirmed:=true \
  lio_child_to_lidar_transform:="$T3_LIO_CHILD_TO_LIDAR" \
  enable_lidar_mapping:=true \
  lidar_mapping_mode:=direct \
  lidar_topic:="$T3_LIDAR_TOPIC" \
  lidar_extrinsics_path:="$T3_LIDAR_EXTRINSICS" \
  lidar_expected_frame_id:="$T3_LIDAR_FRAME" \
  lidar_temporal_mode:=upstream_deskewed \
  lidar_fallback_to_stereo:=false \
  enable_global_loop_closure:=false \
  data_dir:="$T3_CALIB_DIR" \
  output_dir:="$T3_RUNTIME_ROOT/output"
```

这组参数还必须与现场 FAST-LIO2 的 frame、LiDAR/IMU 外参一致。课题三中
`enable_imu:=false` 只是关闭 RoMa 使用的 IMU 订阅；FAST-LIO2 自己仍需要原始 IMU。

VO/LIO 融合：

```bash
source /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/enter_t3_environment.sh

cd "$T3_WS"
bash scripts/run_t1_t2_adapter_template.sh \
  --profile "$T3_PROFILE" \
  localization_mode:=vo_lio \
  fusion_backend:=gtsam_gp \
  lio_odometry_topic:="$T3_LIO_ODOMETRY_TOPIC" \
  lio_expected_parent_frame:="$T3_LIO_PARENT_FRAME" \
  lio_expected_child_frame:="$T3_LIO_CHILD_FRAME" \
  lio_child_to_lidar_confirmed:=true \
  lio_child_to_lidar_transform:="$T3_LIO_CHILD_TO_LIDAR" \
  lidar_extrinsics_path:="$T3_LIDAR_EXTRINSICS" \
  enable_lidar_mapping:=false \
  enable_global_loop_closure:=false \
  data_dir:="$T3_CALIB_DIR" \
  output_dir:="$T3_RUNTIME_ROOT/output"
```

这一模式要求当前 Python 环境已安装 `gtsam` 和 `gtsam_unstable`。
`vo` 和 `lio` 必须使用 `fusion_backend:=none`；`vo_lio` 必须使用
`fusion_backend:=gtsam_gp`。

LiDAR 外参文件必须是 JSON，不是 YAML。其中至少要有如下结构，
`RT` 是行主序 4×4 齐次矩阵，表示把 LiDAR 点变换到原始左相机：

```json
{
  "LiDAR_to_Cam_L": {
    "RT": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]
  }
}
```

上面单位矩阵只展示文件格式，不是可用的实车外参。

### 6.3 切换对外地图范围

另外还有一个只改变“怎么向课题四发地图”的参数，它不会改变位姿来源或
双目/LiDAR 融合算法：

| `incremental_publish_mode` | 行为 |
|---|---|
| `fixed` | 默认，固定原点的 60 m × 60 m 完整地图 |
| `rolling_window` | 跟随机器人移动的 60 m × 60 m 局部窗口 |
| `full_tiles` | 按完整瓦片发布全局地图，课题四必须同时处理批次元数据 |

例如改成滚动窗口：

```bash
cd "$T3_WS"
bash scripts/run_t1_t2_adapter_template.sh \
  --profile "$T3_PROFILE" \
  localization_mode:=vo \
  fusion_backend:=none \
  enable_lidar_mapping:=false \
  incremental_publish_mode:=rolling_window \
  enable_global_loop_closure:=false \
  data_dir:="$T3_CALIB_DIR" \
  output_dir:="$T3_RUNTIME_ROOT/output"
```

这些值不建议用 `ros2 param set` 在运行中修改，因为节点在启动时已经根据模式
创建了不同的订阅、缓存和后端对象。应停止当前节点，修改启动命令后重新启动。

### 6.4 输出 topic 和消息格式

给课题四/下游使用的主地图输出是：

| Topic | ROS 2 类型 | 用途 |
|---|---|---|
| `/Car/T3/semantic/incremental_map` | `t3_interfaces/msg/IncrementalSemanticMap` | 正式多层 2.5D 语义地图 |
| `/Car/T3/semantic/current_pose` | `nav_msgs/msg/Odometry` | 当前位姿和速度 |
| `/Car/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap` | 标准 GridMap 形式的同一张地图 |
| `/Car/T3/mapping/occupancy_grid` | `nav_msgs/msg/OccupancyGrid` | 标准单层占据栅格 |
| `/Car/T3/mapping/map_points` | `sensor_msgs/msg/PointCloud2` | 带 `semantic` 字段的有上限调试点样本 |

另外还有以下可视化/调试输出：

| Topic | ROS 2 类型 | 格式 |
|---|---|---|
| `/Car/T3/debug/trajectory` | `nav_msgs/msg/Path` | `map` 坐标系轨迹 |
| `/Car/T3/mapping/semantic_grid` | `sensor_msgs/msg/Image` | `mono8` 语义 ID |
| `/Car/T3/mapping/semantic_grid_color` | `sensor_msgs/msg/Image` | `bgr8` 语义伪彩色 |
| `/Car/T3/mapping/elevation_grid` | `sensor_msgs/msg/Image` | `32FC1` 高程 |
| `/Car/T3/mapping/elevation_variance_grid` | `sensor_msgs/msg/Image` | `32FC1` 高程方差 |
| `/Car/T3/mapping/elevation_grid_color` | `sensor_msgs/msg/Image` | `bgr8` 高程伪彩色 |
| `/Car/T3/debug/input_processed` | `std_msgs/msg/Header` | 三图同步回调已结束的回执，不等于位姿成功 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 默认 `map -> car_stereo_left_optical_frame` |

`IncrementalSemanticMap` 的核心字段是：

```text
header                         # stamp + frame_id，默认 frame_id=map
update_id                      # uint32 更新序号
resolution, width, height      # 默认 0.10 m，fixed 为 600 x 600
origin                         # 栅格左下角在 map 坐标系中的位姿
occupancy[]                    # int8：-1 未知，0 空闲，100 占据
semantic[]                     # uint8 语义类 ID
semantic_confidence[]          # float32，0..1
elevation[]                    # float32 高程，未观测为 NaN
elevation_variance[]           # float32 高程方差，未观测为 NaN
height_range[]                 # float32 单元内高度范围，未观测为 NaN
roughness[]                    # float32 粗糙度，未观测为 NaN
observation_count[]            # uint32 融合进该单元的观测数
class_names[]                  # ID 到类别名的映射
```

八个栅格数组的长度都必须是 `width * height`，并按行主序展平：
`index = row * width + column`。默认类别映射是
`0 unknown`、`1 traversable_ground`、`2 shadow`、`3 bright_rock`、
`4 rough_obstacle`。

`fixed` 和 `rolling_window` 只需订阅主地图 topic；`full_tiles` 模式下，
每条主地图消息是一块 256×256 瓦片，下游还必须同时订阅
`/Car/T3/semantic/map_batch [t3_interfaces/msg/SemanticMapBatch]`，按
`BEGIN -> N 条瓦片 -> COMPLETE` 完整提交。
主地图默认 QoS 为 `Reliable + Transient Local`；下游如果要在晚于 mapper
启动时仍然收到最新地图，订阅端也应使用 `Transient Local`。

环境和 mapper 启动后，可直接执行：

```bash
ros2 topic type /Car/T3/semantic/incremental_map
ros2 interface show t3_interfaces/msg/IncrementalSemanticMap
ros2 topic echo /Car/T3/semantic/incremental_map --once --field header
ros2 topic echo /Car/T3/semantic/incremental_map --once --field width
ros2 topic echo /Car/T3/semantic/current_pose --once
```

切换 VO/LIO/LiDAR 建图模式不会改变这些 topic 和消息类型。

## 7. 以后每次开机怎么进入环境

已新增统一入口脚本：

```bash
source /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/enter_t3_environment.sh
```

不带参数时，脚本使用配置文件中的 `T3_DEFAULT_ENV_MODE`，
模板默认是 `$HOME/t3_gridmap_v2_runtime/ros_venv`。它会加载已有 Humble、
FAST-LIO2 overlay、课题三
`install/setup.bash`、内置 RoMa 源码和模型，最后进入 `t3_gridmap_v2_ws`。
它会先自动读取 `$HOME/.config/t3_gridmap_v2/environment.sh`，因此无需重复
`export ROS_SETUP_FILE`、`T3_PROFILE`、`T3_CALIB_DIR` 或 LiDAR 变量。

Conda 机器推荐在一次性配置中写 `T3_DEFAULT_ENV_MODE=conda`，以后也用上面
同一条无参命令。临时覆盖配置时才写：

```bash
source /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/enter_t3_environment.sh conda
```

如果只想进入 RoMa 环境、明确不加载 FAST-LIO2，才使用：

```bash
source /opt/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/enter_t3_environment.sh \
  --roma-only
```

注意：默认加载 FAST-LIO2 只表示 ROS 环境能找到它的包和库，不会自动启动
LiDAR 驱动、FAST-LIO2 节点或 LiDAR 建图。实际建图模式仍由第 6.2 节的
`localization_mode`、`enable_lidar_mapping` 和 `lidar_mapping_mode` 决定。

如果 Humble 是源码编译的，只需把用户配置文件中的 `ROS_SETUP_FILE`
改成它的真实 `install/setup.bash`，不要在每个终端再手动 `export`。

必须使用 `source`，不能写成 `bash enter_t3_environment.sh`，否则子进程结束后
环境变量不会留在当前终端。

## 8. 只有缺依赖时才执行

如果第 2 节全部显示 `[OK]`，跳过本节。

普通 Ubuntu 20.04 开发库可以直接安装：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  pkg-config \
  libapr1-dev \
  libboost-all-dev \
  libeigen3-dev \
  libjpeg-dev \
  libopenblas-dev \
  libpcap-dev \
  libpcl-dev \
  libpng-dev \
  libusb-1.0-0-dev \
  zlib1g-dev \
  python3-dev \
  python3-pip \
  python3-venv \
  python3-numpy \
  python3-opencv \
  python3-scipy \
  python3-yaml
```

ROS 2 Humble 的公开二进制包面向 Ubuntu 22.04 Jammy，不能把 Jammy 软件源混入
JetPack 5.1.1 的 Ubuntu 20.04 Focal。

只有团队确实提供 Focal/arm64/Humble 软件源时，才可以执行：

```bash
cd "$T3_WS"

T3_ALLOW_FOCAL_HUMBLE_APT=1 \
bash scripts/install_prereqs.sh

T3_ALLOW_FOCAL_HUMBLE_APT=1 \
bash scripts/install_fastlio2_prereqs.sh
```

`T3_ALLOW_FOCAL_HUMBLE_APT=1` 只是解除安全拦截，不会自动创建兼容软件源。

没有团队软件源时，缺少的 ROS 包应在现有 Humble 源码工作区或单独 ROS overlay
中补编译，不要把 ROS 2 源码放进 `t3_gridmap_v2_ws/src`。

## 9. RoMa 是默认主线，GTSAM 和 MPRF 是可选增强

- RoMa VO：默认必须项，需要 Jetson PyTorch、torchvision 和 RoMa 运行依赖；
- LiDAR 全局回环：需要 `gtsam`；
- `vo_lio + gtsam_gp`：需要 `gtsam` 和 `gtsam_unstable`；
- MPRF：在 x86_64 NVIDIA 伴随机运行，Orin 只运行客户端和 LiDAR 几何复核。

如果金镜环境已经安装了兼容的 RoMa/PyTorch，不需要解压候选缓存。
否则才解压到工作空间之外：

```bash
mkdir -p "$T3_RUNTIME_ROOT/orin_build_cache"

tar -xzf "$T3_DEPS/optional_orin_build_cache.tar.gz" \
  -C "$T3_RUNTIME_ROOT/orin_build_cache"
```

当前 RoMa 元数据要求 Python 3.9 以上、PyTorch 2.5.1 以上，而 JetPack 5.1.1 随包
候选 wheel 是 Python 3.8/PyTorch 2.1。这一组合需要真实 Orin 兼容确认；
未通过 RoMa 初始化和一对双目图像推理前，不能将默认 VO 视为部署完成。

MPRF 详细说明见：

```text
t3_gridmap_v2_ws/课题三MPRF闭环实现与部署说明.md
```

## 10. 最终检查

- 外层目录是 `t3_gridmap_v2_delivery`；
- 工作空间名称仍是 `t3_gridmap_v2_ws`；
- `third_party` 和 `models` 相对链接有效；
- 模型 profile 显示 `full`；
- FAST-LIO2 已在当前 Orin 原生编译；
- venv 与 Conda 二选一，不同时混用；
- T1/T2 占位符已替换，或明确使用 `standard`；
- 默认启动参数是 `localization_mode:=vo`、`fusion_backend:=none`、
  `enable_lidar_mapping:=false`；
- RoMa 已在真实 Orin 上完成模型初始化和双目推理确认；
- 运行输出位于工作空间之外；
- 未启用的 GTSAM、MPRF 和可选缓存保持未安装状态。
