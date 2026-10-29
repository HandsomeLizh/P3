# T3 当前算法代码与配置交付包

## 1. 交付范围

本目录保存 2026-08-25 当前实际运行的 RoMa 双目视觉定位、点云辅助建图、
实时指标客户端和离线指标评估代码。模型权重和本次采用的相机/点云配置一并保存，
因此代码和配置不依赖原工作目录中的散落文件。

不包含 rosbag、历史运行结果、`build/`、`install/`、`log/`、Python 缓存和 Git
缓存；这些不属于算法源码或配置。完整文件列表见 `metadata/file_manifest.tsv`，
SHA-256 校验值见 `metadata/MANIFEST.sha256`。

## 2. 目录说明

| 目录 | 内容 |
|---|---|
| `workspace/src/` | ROS 2 消息、适配器和 T3 定位建图软件包源码 |
| `workspace/scripts/` | 运行后处理、轨迹对齐、全指标评估及测试脚本 |
| `workspace/mprf_runtime/` | 工程自带运行时 Python 模块和配置 |
| `workspace/docs/`、`tests/`、`patches/` | 设计文档、测试和补丁记录 |
| `third_party/RoMa/` | 当前 RoMa 匹配器源码的实体副本，不是符号链接 |
| `models/torch/hub/checkpoints/` | 当前 RoMa 使用的 DINOv2 和 RoMa 权重 |
| `realtime_dashboard/` | 实时 ROS 2 客户端、网页可视化、运行脚本和中文手册 |
| `workflow/` | 当前 rosbag 流程的启动、真值记录和评估脚本 |
| `config/` | 当前冻结 YAML 参数和两份原始 XLSX 标定文件 |
| `tools/` | 本次使用的 rosbag 发布、轨迹对齐和地图可视化工具 |
| `metadata/` | 环境信息、文件清单和 SHA-256 校验值 |

## 3. 当前冻结参数

- 相机参数：`config/calibration_spreadsheets_parallel_24p9.yaml`
- 点云仅参与建图的外参：`config/pointcloud_extrinsic_os1_mapping_only.yaml`
- 双目光轴平行，整体俯仰角 -24.9 度；相机模型无畸变。
- 当前地图分辨率为 0.2 m。
- 点云融合只扩大地图，不进入视觉定位和定姿求解。
- ROS domain：`228`。
- 定姿稳定度：完整周期以 500 s 为分母；不足 500 s 时以当前实际运行时长为分母。

## 4. 使用入口

详细的容器、文件放置、逐条运行命令、实时话题、结果文件和 RViz 操作见：

`realtime_dashboard/实时发布与Ubuntu客户端运行手册.md`

当前流程入口为：

- 算法启动：`workflow/01_run_roma.sh`
- 实时客户端：`realtime_dashboard/run_realtime_dashboard.sh`
- 离线全指标评估：`workflow/03_evaluate.sh`
- 评估器源码：`workspace/scripts/evaluate_all_metrics.py`

运行前需要 Ubuntu 22.04、ROS 2 Humble、CUDA/PyTorch 环境以及本工程依赖。
容器内编译时必须使用与机器架构一致的工作区；本次 x86 GPU 容器使用
`install_gpu`，不要误用已有的 ARM `install` 目录。本交付包不保存编译产物，
应在目标环境重新执行 `colcon build`。

## 5. 完整性检查

在本目录执行：

```bash
sha256sum -c metadata/MANIFEST.sha256
```

全部显示 `OK` 后再进行交接或部署。`metadata/environment.txt` 记录了打包时的
系统、ROS、Python、PyTorch、CUDA、GPU 和容器信息。
