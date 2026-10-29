#!/bin/bash
# ============================================================
#  月球车遥测与控制节点 启动脚本
#  (lunar_car_ctrl)
# ============================================================
#
#  功能:
#    - TCP 持久连接 UE 服务器
#    - 订阅 /car/cmd_vel，周期性发送 4W4S 动力学控制指令
#    - 接收 UE 遥测反馈，发布 /car/telemetry、/car/odom、/car/pose
#
#  用法:
#    ./start_lunar_car.sh [UE主机IP] [端口] [字节序]
#
#  示例:
#    ./start_lunar_car.sh                              # 默认 192.168.1.21:6668, 小端
#    ./start_lunar_car.sh 192.168.1.100 7011           # 指定 IP 和端口, 小端
#    ./start_lunar_car.sh 192.168.1.100 7011 true      # 指定 IP、端口、大端序
#
#  可配置参数 (通过 ros2 param set 动态调整):
#    wheel_base      轴距 (m)                   默认: 0.815
#    track_width     轮距 (m)                   默认: 0.622
#    wheel_radius    车轮半径 (m)               默认: 0.12 (轮速 rad/s ↔ 线速度 m/s 换算)
#    max_wheel_speed 单轮最大轮速 (rad/s)       默认: 10.0
#    control_period  控制周期 (s)               默认: 0.05 (20Hz)
#    big_endian      通信字节序 (见下方说明)    默认: false (小端)
#
#  字节序说明 (big_endian):
#    控制与 UE 服务器 TCP 通信时的数据字节序。
#    false = 小端序 (Little-Endian, x86/x64 架构默认)
#    true  = 大端序 (Big-Endian, 网络字节序)
#    需与 UE 服务端保持一致，否则解析数据会错乱。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"

# ---- 默认参数 (位置参数覆盖) ----
UE_HOST="${1:-192.168.10.22}"   # UE 服务器 IP
UE_PORT="${2:-6668}"           # UE 服务器端口
BIG_ENDIAN="${3:-false}"       # 通信字节序: false=小端(默认), true=大端

# ---- 检查编译产物，首次运行自动编译 ----
if [ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    echo "[INFO] 首次运行，正在编译..."
    cd "${WORKSPACE_DIR}"
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --packages-select lunar_car_ctrl
    if [ $? -ne 0 ]; then
        echo "[ERROR] 编译失败！"
        exit 1
    fi
fi

# ---- 加载 ROS2 环境和工作空间 ----
source /opt/ros/humble/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"

echo "========================================"
echo "  月球车遥测与控制节点"
echo "  UE 地址: ${UE_HOST}:${UE_PORT}"
echo "  字节序:   ${BIG_ENDIAN} ($( [ "${BIG_ENDIAN}" = "true" ] && echo '大端' || echo '小端' ))"
echo "========================================"
echo ""
echo "  订阅: /car/cmd_vel          (Twist)"
echo "  发布: /car/telemetry        (VehicleTelemetry)"
echo "        /car/odom             (Odometry)"
echo "        /car/pose             (PoseStamped)"
echo "        /car/telemetry_raw    (ByteMultiArray)"
echo ""
echo "  按 Ctrl+C 退出"
echo "========================================"

# ---- 启动控制节点 (TCP 连接参数) ----
ros2 run lunar_car_ctrl lunar_car_node \
    --ros-args \
    -p tcp_host:="${UE_HOST}" \
    -p tcp_port:=${UE_PORT} \
    -p big_endian:=${BIG_ENDIAN}
