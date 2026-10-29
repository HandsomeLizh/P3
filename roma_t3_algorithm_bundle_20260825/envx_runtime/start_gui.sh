#!/bin/bash
# ============================================================
#  月球车控制 GUI 启动脚本
#  (lunar_car_gui)
# ============================================================
#
#  功能:
#    - tkinter GUI 界面，滑条控制线速度和曲率
#    - 发布 /car/cmd_vel (Twist) 速度控制指令与 /car/goal_point 目标点
#    - 订阅 /car/telemetry 显示遥测数据 (位置/速度/轮速)
#    - 订阅栅格地图与规划轨迹并在画布上显示, 鼠标点击选中目标点, 按钮发布
#
#  用法:
#    ./start_lunar_car_gui.sh
#
#  前提:
#    - 需要图形桌面环境 (X11/Wayland)
#    - lunar_car_ctrl 节点需已启动
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"

# ---- 检查编译产物，首次运行自动编译 ----
if [ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    echo "[INFO] 首次运行，正在编译..."
    cd "${WORKSPACE_DIR}"
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --packages-select lunar_car_gui
    if [ $? -ne 0 ]; then
        echo "[ERROR] 编译失败！"
        exit 1
    fi
fi

# ---- 加载 ROS2 环境和工作空间 ----
source /opt/ros/humble/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"

# ---- 自动检测 DISPLAY (无 DISPLAY 时尝试设置 :0) ----
if [ -z "${DISPLAY}" ]; then
    if [ -e /tmp/.X11-unix/X0 ]; then
        export DISPLAY=:0
        echo "[INFO] 自动设置 DISPLAY=:0"
    else
        echo "[ERROR] 未找到图形显示，请确认桌面环境已启动"
        exit 1
    fi
fi

# ---- 地图/轨迹话题配置 (可通过环境变量 GUI_MAP_TOPIC / GUI_PATH_TOPIC 覆盖) ----
MAP_TOPIC="${GUI_MAP_TOPIC:-/map}"
PATH_TOPIC="${GUI_PATH_TOPIC:-/plan}"

echo "========================================"
echo "  月球车控制 GUI"
echo "========================================"
echo ""
echo "  发布: /car/cmd_vel (Twist), /car/goal_point (PoseStamped)"
echo "  订阅: /car/telemetry, ${MAP_TOPIC} (OccupancyGrid), ${PATH_TOPIC} (Path)"
echo ""
# ---- 启动 GUI 节点 ----
ros2 run lunar_car_gui lunar_car_gui \
    --ros-args -p map_topic:="${MAP_TOPIC}" -p path_topic:="${PATH_TOPIC}"
