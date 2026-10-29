# 新机器部署入口

完整的中文环境要求、RoMa/PyTorch 安装、课题三编译、FAST-LIO2 下载
编译、标定与上车验收步骤见：

[课题三新机器部署与FAST-LIO2编译说明.md](课题三新机器部署与FAST-LIO2编译说明.md)

以下仅保留 x86 Ubuntu 22.04/Humble 开发机的快速检查命令。
AGX Orin 目标为 aarch64 + JetPack 5.1.1/Ubuntu 20.04，必须先按上述完整文档
获取集成方的 Humble 黄金镜像/wheelhouse，不要直接执行下面的通用 apt 流程。

发布包只有一个顶层工作空间，环境脚本会自动发现其中的固定 RoMa 依赖：

```text
<root>/t3_gridmap_v2_ws/
  ├── src/                              # 唯一主 colcon 源码目录
  └── third_party/
      ├── RoMa/roma_vo.py              # 默认 VO
      └── fastlio2/                    # 可选，内部隔离构建
```

An ABI-compatible virtual environment is optional; without one the scripts use
the ROS system Python. Dataset paths are supplied explicitly at launch time.

Do not source the old `~/t3_semantic_practice_ws` overlay in the same terminal.

```bash
cd /path/to/t3_gridmap_v2_ws
bash scripts/install_prereqs.sh
bash scripts/build.sh
source scripts/enter_workspace.sh
bash scripts/check_environment.sh
bash scripts/run_quality_gates.sh
```

The activation script is repeatable. It is safe to source it before calling a
`run_*` wrapper, even though that wrapper sources the same file again. A missing
automatic default venv uses the ROS system Python; an explicitly configured
missing `ROS_VENV` remains an error.

Verify package origins:

```bash
ros2 pkg prefix t1_dataset_sim
ros2 pkg prefix t2_basic_semantics
ros2 pkg prefix t3_interfaces
ros2 pkg prefix t3_semantic_mapping
ros2 pkg prefix grid_map_msgs
ros2 pkg prefix t3_upstream_adapters
```

All six paths should point into `<T3_WS>/install`.

## Run a batch_organized sequence

The launcher auto-detects the organized layout, reads `images/timestamps.tsv`
and `imu/imu.csv`, converts the JSON stereo calibration, and rectifies both
cameras before publishing:

```bash
bash scripts/run_batch_organized.sh \
  /data/batch_organized \
  max_frames:=20
```

Passing the batch root selects its first sequence; pass a sequence directory
directly to choose a different one.

Stereo pairs whose exposure times differ by more than 20 ms are skipped by
default.  The limit can be changed explicitly with
`max_stereo_skew_ms:=VALUE`; increasing it can reduce geometric accuracy on a
moving platform.

After every joint dataset run, inspect both JSON ledgers under `output_dir`.
`dataset_playback_summary.json.input_selection` distinguishes raw left/right
rows, shared candidates, stereo-skew-eligible pairs, and final frames selected
after `max_frames`. Its `complete_input_consumption` applies only to the
explicit `complete_input_consumption_scope`, not automatically to every raw
row in the directory. `input_processing_summary.json` then separates terminal
callbacks from valid poses and map revisions. The current target sequence
`2025-06-26-23-01-34` is a useful warning example: 157 shared rows contain only
34 pairs within 20 ms; all 34 were consumed, but this is not a 157-frame stereo
accuracy run. Keep the geometry guard and obtain synchronized rosbag/live input
instead of increasing it to absorb 100--300 ms exposure offsets.
