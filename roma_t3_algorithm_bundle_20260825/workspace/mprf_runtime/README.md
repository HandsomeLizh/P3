# MPRF 隔离运行时

这里是课题三自研的 MPRF 在线候选服务，不是完整复制上游程序。它负责固定模型
加载、在线检索/重排、受限协议和持久化；LiDAR 几何复核、时序确认、全局 iSAM2
以及地图重建仍在 ROS 2 工作区内完成。

完整算法边界、参数、远程部署、ROS 2 启动方式和真实序列结果见：

- [`../课题三MPRF闭环实现与部署说明.md`](../课题三MPRF闭环实现与部署说明.md)

新机器上的最短准备流程：

```bash
cd /path/to/t3_gridmap_v2_ws/mprf_runtime
python3 scripts/verify_assets.py
bash scripts/create_environment.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_models.py --device cuda:0
bash scripts/run_tests.sh
```

启动本机服务：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_service.sh \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --state-root /path/to/persistent/mprf_state
```

`.venv/`、`state/`、`*.egg-info/` 和 Python 缓存是机器生成物，不应随源码复制。
必须复制 `mprf_runtime/` 源码、`../third_party/MPRF/` 固定源码和
`../models/` 权重；权重与源码版本由 `scripts/verify_assets.py` 校验。
