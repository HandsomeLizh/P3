# P3 课题三：定位与建图

工作区说明见 [`context.md`](context.md)。

| 目录 | 内容 |
|---|---|
| `lidar_visual_fusion/` | 主工程（ROS 2 Humble colcon 工作区）：LiDAR + 双目视觉融合定位与高程建图 |
| `roma_t3_algorithm_bundle_20260825/` | RoMa 双目视觉定位交付包：源码、配置、工具、UE 采集运行时源码 |

## 未包含的内容

本仓库只保存算法源码、配置、文档和测试数据。以下内容没有上传，需要另外获取：

- 模型权重：DINOv2、RoMa、ALIKED、SuperPoint、LightGlue 的 `.pth` 文件
- rosbag 录制数据、运行输出、日志
- `build/`、`install/` 编译产物
- `lidar_visual_fusion/deps/`：Orin（arm64）离线安装用的 `.deb`、`.whl` 包和 apt 缓存
- `lidar_visual_fusion/vendor/` 下的第三方源码，按下表从上游克隆

## 第三方依赖（`lidar_visual_fusion/vendor/`）

| 目录 | 上游仓库 | 使用的提交 | 备注 |
|---|---|---|---|
| `ALIKED-weights-source` | https://github.com/Shiaoming/ALIKED | `683d7c6` | |
| `LightGlue` | https://github.com/cvg/LightGlue | `eb42fee` | |
| `XFeat` | https://github.com/verlab/accelerated_features | `e92685f` | 本地有 1 处未提交修改 |
| `VoxelMap` | https://github.com/hku-mars/VoxelMap | `d787ee8` | |
| `PV-LIO` | https://github.com/HViktorTsoi/PV-LIO | `494c27f` | |
| `VINS-Fusion` | https://github.com/leohaijunli/VINS-Fusion-ROS2-humble-arm | `ff167b7` | 本地有 4 处未提交修改；主配置不启动 |
| `point_lio` | https://github.com/dfloreaa/point_lio_ros2 | `a8e2d0d` | 本地有 10 处未提交修改；已退役 |
| `capture_overlap_source` | 自有代码 | | 已包含在本仓库 |
| `sensor_input` | | | 只有 `capture_overlap_source` 编译出的 arm64 可执行文件，未上传 |

`roma_t3_algorithm_bundle_20260825/third_party/RoMa` 是 RoMa 源码的完整副本，已包含在本仓库中。

## 复原步骤

```bash
git clone git@github.com:HandsomeLizh/P3.git ~/P3
cd ~/P3/lidar_visual_fusion/vendor

# 1. 按上表克隆第三方库并切到对应提交，例如：
git clone https://github.com/verlab/accelerated_features.git XFeat && git -C XFeat checkout e92685f

# 2. 打上本地修改（XFeat、VINS-Fusion、point_lio）
git -C XFeat       apply ~/P3/restore/vendor_patches/XFeat.patch
git -C VINS-Fusion apply ~/P3/restore/vendor_patches/VINS-Fusion.patch
git -C point_lio   apply ~/P3/restore/vendor_patches/point_lio.patch

# 3. 放回权重后校验（路径相对于仓库根目录）
cd ~/P3 && sha256sum -c restore/weights.sha256

# 4. 编译
cd ~/P3/lidar_visual_fusion && ./build.sh
```

- `restore/vendor_patches/`：第三方库的本地修改，已验证能在上表提交上完整还原。
- `restore/weights.sha256`：原机器上全部权重文件的路径和 SHA-256。XFeat、ALIKED 权重随各自上游仓库提供；
  RoMa / DINOv2 权重会在首次运行时由 torch hub 自动下载；LightGlue / SuperPoint 权重来自 LightGlue 的 GitHub release。
- 系统依赖（ROS 2 Humble、ceres、kornia 等）在 arm64 上可用 apt / pip 安装，所需包名见原机器 `lidar_visual_fusion/deps/`。
