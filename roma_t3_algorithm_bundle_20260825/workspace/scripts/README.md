# scripts目录使用说明

生产运行只需要下面几个入口：

```text
build.sh                 编译工作空间
enter_workspace.sh       加载ROS、工作空间和RoMa路径
check_environment.sh     只读检查当前系统环境
run_t5_roma_vo.sh        默认RoMa定位与建图入口
run_rviz.sh              RViz地图和轨迹显示
save_grid_map_now.sh     运行中强制保存当前地图
audit_p3_layout.sh       只读统计P3目录，供清理前审计
```

可选定位支路：

```text
run_t5_fastlivo2.sh
run_t5_ouster_lio.sh
enter_fastlivo2_workspace.sh
enter_fastlio2_workspace.sh
setup_fastlio2_workspace.sh
```

文件名以 `evaluate_`、`inspect_`、`monitor_`、`play_`、`export_`、
`check_` 或 `test_` 开头的脚本属于离线评测、诊断或开发测试，不是日常启动入口。
`run_batch_organized.sh` 和 `run_target_lio_replay.sh` 仅用于历史数据复测。

普通实车运行默认发布完整全局高程GridMap并保存NPZ。大范围仿真使用：

```bash
bash scripts/run_t5_roma_vo.sh --lite-map
```

FAST-LIVO2入口也支持同一 `--lite-map` 开关。此模式只关闭完整全局
GridMap和稠密NPZ，局部GridMap、低分辨率全局总览、SQLite完整地图、
位姿和轨迹均保持启用。`--full-map` 可显式恢复默认完整输出。

项目现在固定使用系统Python和现有ROS 2 Humble环境，不再维护虚拟环境模式、
通用apt安装入口或多套环境激活脚本。新接手人员不要自行升级NumPy、SciPy、
OpenCV、PyTorch或CUDA组件；先运行 `check_environment.sh` 核对当前机器。

SSH端清理前先运行：

```bash
bash ~/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/audit_p3_layout.sh ~/P3 \
  | tee ~/P3/p3_layout_audit.txt
```

该脚本只读取目录和文件信息，不移动、不覆盖、不删除任何内容。

## 精确时间戳bag回放与全指标报告

历史bag的双目图像Header时间戳损坏时，使用数据库记录时间回放。需要同时
重发PQ和两套IMU时增加 `--all-metrics-topics`：

```bash
python3 /home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/play_bag_with_record_stamps.py \
  /path/to/bag \
  --rate 0.5 \
  --all-metrics-topics
```

运行结束并保存地图后，一条命令生成全部指标报告：

```bash
python3 /home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/scripts/evaluate_all_metrics.py \
  /home/yanfa/P3/output/本次输出目录 \
  --bag /path/to/bag
```

默认分别以平均值判断定位处理时间和地图更新时间是否不超过2秒；如测试组
要求P95或最大值，增加 `--time-statistic recent_p95` 或
`--time-statistic max`。报告写入运行目录：

```text
all_metrics_report.json
all_metrics_report.md
trajectory_comparison.csv        仅有参考轨迹时生成
pq_reference.tum                 PQ有有效消息时生成
pq_reference_metadata.json       记录PQ字段和坐标假设
```

PQ默认按 `counter,x,y,z,roll,pitch,yaw` 解码，并采用bag数据库记录时间。
正式验收前必须由PQ提供方确认字段、单位、坐标系和参考系统精度。若bag只是
注册了PQ话题但消息数为0，工具会将定位精度和定姿稳定度明确标为
`unavailable`，不会用视觉自漂或IMU数据冒充真值。
