#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# lunar_car_gui_node.py — 月球车控制 GUI 节点
#
# 功能:
#   订阅: /car/telemetry     (VehicleTelemetry) — 遥测数据显示
#         (位姿 RH/LH、车体速度、四轮轮速与转角)
#         /car/wheel_commands (WheelCommands)  — 实际发送的四轮指令 (左手系)
#         <map_topic>  (OccupancyGrid, 默认 /map)  — 栅格地图显示
#         <path_topic> (Path, 默认 /plan)          — 规划轨迹显示
#   发布: /car/cmd_vel  (Twist)  — 速度控制指令 (10Hz 持续发布)
#         /car/set_mode (String) — 控制模式切换 ("park"/"auto")
#         /car/goal_point (PoseStamped) — 目标点 (界面输入或鼠标点击地图)
#         /sensor/set_exposure (Float32) — 曝光时间设置 (ms, RGB 2/3 曝光,
#           sensor_capture 节点按偏差换算后下发给全部 RGB 相机)
#         /sensor/set_light_intensity (Float32) — 灯光亮度设置 (GUI 限 10~30)
#
# 界面布局 (左地图画布 + 右控制/遥测栏):
#   左栏: 地图画布 (栅格地图 + 规划轨迹 + 车位姿 + 目标点标记)
#         鼠标左键点击选中目标点 (不自动发布), 拖拽平移, 滚轮缩放
#   右栏: 控制 (模式、线速度/曲率上下两行滑条、紧急停车)、
#         目标点输入、相机设置 (曝光时间/灯光亮度)、位姿/速度遥测、车轮与指令
#
# 说明:
#   - lunar_car_ctrl 自动驾驶模式要求 1s 内收到 cmd_vel,
#     因此 GUI 以 10Hz 持续发布当前滑条值 (可通过复选框关闭)。
#   - 线速度滑条范围 ±0.1 m/s 与节点 max_linear_speed 默认值一致;
#   - 曲率滑条范围 ±2 (1/m)，发布时转换为角速度: ω = v × κ。
#   - 地图以 numpy 采样渲染为 PPM 图像 (支持任意缩放/平移),
#     轨迹/车位姿/目标点以矢量图层叠加绘制。
# ============================================================

import math
import threading
import time
import signal
import tkinter as tk

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

try:
    from PIL import Image as PILImage     # 可选: 地图高质量缩放 (不可用时回退最近邻)
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from geometry_msgs.msg import PoseStamped, Twist
from lunar_car_ctrl.msg import VehicleTelemetry, WheelCommands
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, Float32, String
from rclpy.signals import SignalHandlerOptions


class LunarCarGuiNode(Node):
    """ROS2 节点: 订阅遥测/地图/轨迹、按固定频率发布速度指令与模式切换"""

    def __init__(self):
        super().__init__('lunar_car_gui_node')

        # ---- 可配置话题名 (栅格地图与规划轨迹由外部规划器发布) ----
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('path_topic', '/plan')
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value

        # ---- 数据缓存 (GUI 线程与回调线程共享, 需加锁) ----
        self._lock = threading.Lock()
        self.latest_telemetry = None      # 最新遥测消息
        self.last_telemetry_time = None   # 最新遥测的接收时间 (node 时钟)
        self.latest_wheel_cmd = None      # 最新发送的四轮指令 (左手系)
        self.latest_grid = None           # 最新栅格地图 (OccupancyGrid)
        self.last_grid_time = None        # 最新地图的接收时间
        self.latest_path = None           # 最新规划轨迹 (Path)
        self.last_path_time = None        # 最新轨迹的接收时间
        self.linear_cmd = 0.0             # 当前线速度指令 (m/s)
        self.curvature_cmd = 0.0           # 当前曲率指令 (1/m)
        self.publish_enabled = True       # 是否持续发布 cmd_vel
        self.requested_mode = 'park'      # 最近请求的控制模式
        self.external_control_until = 0.0
        self.external_control_active = False
        self.declare_parameter('external_control_lease_supported', True)

        # ---- 订阅: 遥测数据与实际发送的四轮指令 ----
        self.telemetry_sub = self.create_subscription(
            VehicleTelemetry, '/car/telemetry',
            self.on_telemetry, 10)
        self.wheel_cmd_sub = self.create_subscription(
            WheelCommands, '/car/wheel_commands',
            self.on_wheel_commands, 10)

        # ---- 订阅: 栅格地图与规划轨迹 (地图渲染与目标点设置) ----
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_grid, 10)
        self.path_sub = self.create_subscription(
            Path, self.path_topic, self.on_path, 10)

        # ---- 发布: 速度指令、模式切换与目标点 ----
        self.cmd_vel_pub = self.create_publisher(Twist, '/car/cmd_vel', 10)
        self.create_subscription(Bool, '/car/external_control_lease', self.on_external_control, 10)
        self.create_subscription(Twist, '/car/cmd_vel', self.on_external_command, 10)
        self.create_subscription(String, '/car/set_mode', self.on_mode_display, 10)
        self.mode_pub = self.create_publisher(String, '/car/set_mode', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/car/goal_point', 10)

        # ---- 发布: 曝光时间与灯光亮度设置 (由 sensor_capture 节点转发到相机) ----
        self.exposure_pub = self.create_publisher(Float32, '/Car/T1/camera_exposure', 10)
        self.light_pub = self.create_publisher(Float32, '/sensor/set_light_intensity', 10)

        # ---- 定时器: 10Hz 持续发布 cmd_vel ----
        # (自动驾驶模式要求 1s 内收到 cmd_vel, 否则节点自动切停车)
        self.create_timer(0.1, self.publish_cmd_vel)

    # ============ 遥测回调 ============
    def on_telemetry(self, msg):
        with self._lock:
            self.latest_telemetry = msg
            self.last_telemetry_time = self.get_clock().now()

    # ============ 四轮指令回调 (左手系, 实际发送给车端的值) ============
    def on_wheel_commands(self, msg):
        with self._lock:
            self.latest_wheel_cmd = msg

    # ============ 栅格地图 / 规划轨迹回调 ============
    def on_grid(self, msg):
        with self._lock:
            self.latest_grid = msg
            self.last_grid_time = self.get_clock().now()

    def on_path(self, msg):
        with self._lock:
            self.latest_path = msg
            self.last_path_time = self.get_clock().now()

    # ============ 10Hz 发布速度指令 (曲率 → 角速度) ============
    def publish_cmd_vel(self):
        with self._lock:
            if time.monotonic() < self.external_control_until:
                return  # The fixed-route controller owns cmd_vel while its lease is fresh.
            released = self.external_control_active
            if released:
                self.external_control_active = False
                self.linear_cmd = 0.0
                self.curvature_cmd = 0.0
            if not self.publish_enabled:
                if not released:
                    return
            twist = Twist()
            twist.linear.x = self.linear_cmd
            twist.angular.z = self.linear_cmd * self.curvature_cmd  # ω = v × κ
        if released:
            self.send_mode('park')  # Includes loss of the controller heartbeat.
        self.cmd_vel_pub.publish(twist)

    def on_external_control(self, message):
        with self._lock:
            self.external_control_until = time.monotonic() + 1.5 if message.data else 0.0
            if message.data:
                self.external_control_active = True

    def on_external_command(self, message):
        with self._lock:
            if self.external_control_active:
                self.linear_cmd = message.linear.x
                self.curvature_cmd = message.angular.z / message.linear.x if abs(message.linear.x) > 1e-6 else 0.0

    def on_mode_display(self, message):
        with self._lock:
            self.requested_mode = message.data

    # ============ 发布模式切换 ============
    def send_mode(self, mode):
        with self._lock:
            self.requested_mode = mode
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.get_logger().info('请求切换控制模式: %s' % mode)

    # ============ 发布目标点 (单次, 右手系坐标) ============
    def send_goal_point(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0   # 目标点不指定航向, 使用单位四元数
        self.goal_pub.publish(msg)
        self.get_logger().info('发布目标点: x=%.3f, y=%.3f' % (float(x), float(y)))

    # ============ 发布曝光时间设置 (单次, 单位毫秒, 为 RGB 2/3 曝光) ============
    def send_exposure(self, exposure_ms):
        msg = Float32()
        msg.data = float(exposure_ms)
        self.exposure_pub.publish(msg)
        self.get_logger().info('发布曝光设置: %.3f ms' % float(exposure_ms))

    # ============ 发布灯光亮度设置 (单次, GUI 限 10~30) ============
    def send_light_intensity(self, intensity):
        msg = Float32()
        msg.data = float(intensity)
        self.light_pub.publish(msg)
        self.get_logger().info('发布灯光亮度: %.3f' % float(intensity))

    # ============ 设置速度指令 (由滑条回调调用) ============
    def set_command(self, linear, curvature):
        with self._lock:
            self.linear_cmd = float(linear)
            self.curvature_cmd = float(curvature)

    # ============ 紧急停车: 归零指令并发布一次停车模式 ============
    def emergency_stop(self):
        with self._lock:
            self.linear_cmd = 0.0
            self.curvature_cmd = 0.0
        zero = Twist()
        self.cmd_vel_pub.publish(zero)
        self.send_mode('park')


class LunarCarGui:
    """tkinter 界面: 左地图画布 + 右控制/遥测栏"""

    # 滑条范围 (与控制节点限幅约定一致)
    LINEAR_RANGE = 0.1      # ±0.1 m/s
    CURVATURE_RANGE = 2.0   # ±2.0 (1/m)

    # 灯光亮度允许输入范围
    LIGHT_MIN = 10.0
    LIGHT_MAX = 30.0

    # 地图渲染颜色 (RGB)
    COLOR_BG = 0x9E          # 未知区域/背景灰
    COLOR_FREE = 0xEB        # 自由区域 (近白)
    COLOR_OCC = 0x2E         # 占据栅格 (深灰)

    # 视图缩放限制 (像素/米)
    SCALE_MIN = 0.5
    SCALE_MAX = 500.0

    def __init__(self, node):
        self.node = node

        self.root = tk.Tk()
        self.root.title('月球车控制 GUI')
        self.root.geometry('1280x760')
        self.root.minsize(1000, 620)

        # ---- 地图视图状态 (像素/米 与视图中心世界坐标) ----
        self._view_scale = 30.0
        self._view_center = (0.0, 0.0)
        self._map_key = None          # 已渲染地图图像的签名, 变化时重绘
        self._map_photo = None        # PhotoImage 需持有引用防止 GC
        self._map_image_id = None
        self._fitted_stamp = None     # 已自适应视图的地图时间戳
        self._goal_xy = None          # 当前目标点 (世界坐标)
        self._drag_start = None       # 鼠标拖拽起点 (像素)
        self._drag_moved = False

        self._build_content_area()

        # ---- 启动界面刷新 (10Hz) ----
        self._update_display()

    # ========================================================
    #  控制区 (右栏顶部): 模式 + 上下两行滑条 + 紧急停车
    # ========================================================
    def _build_control_area(self, parent):
        ctrl = tk.LabelFrame(parent, text='控制', padx=8, pady=4)
        ctrl.pack(fill=tk.X, pady=(0, 4))

        # ---- 第 1 行: 模式切换 + 状态 ----
        row1 = tk.Frame(ctrl)
        row1.pack(fill=tk.X)

        tk.Label(row1, text='模式:').pack(side=tk.LEFT)
        tk.Button(row1, text='停车', width=8,
                  command=lambda: self.node.send_mode('park')).pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text='自动驾驶', width=8,
                  command=lambda: self.node.send_mode('auto')).pack(side=tk.LEFT, padx=2)
        self.mode_label = tk.Label(row1, text='请求: -', width=10, anchor='w')
        self.mode_label.pack(side=tk.LEFT, padx=6)
        self.conn_label = tk.Label(row1, text='遥测: 未接收', anchor='e')
        self.conn_label.pack(side=tk.RIGHT)

        # ---- 第 2 行: 线速度滑条 ----
        row2 = tk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=2)

        tk.Label(row2, text='线速度 (m/s)').pack(side=tk.LEFT)
        self.linear_scale = tk.Scale(
            row2, from_=-self.LINEAR_RANGE, to=self.LINEAR_RANGE,
            resolution=0.001, orient=tk.HORIZONTAL, length=320,
            showvalue=True, command=self._on_scale_change)
        self.linear_scale.pack(side=tk.LEFT, padx=4)

        # ---- 第 3 行: 曲率滑条 ----
        row3 = tk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=2)

        tk.Label(row3, text='曲率 (1/m)').pack(side=tk.LEFT)
        self.curvature_scale = tk.Scale(
            row3, from_=-self.CURVATURE_RANGE, to=self.CURVATURE_RANGE,
            resolution=0.01, orient=tk.HORIZONTAL, length=320,
            showvalue=True, command=self._on_scale_change)
        self.curvature_scale.pack(side=tk.LEFT, padx=4)

        # ---- 第 4 行: 指令显示 + 发布开关 ----
        row4 = tk.Frame(ctrl)
        row4.pack(fill=tk.X)

        self.cmd_label = tk.Label(row4, text='指令: v=0.000 m/s, κ=0.00 1/m', anchor='w')
        self.cmd_label.pack(side=tk.LEFT)

        self.pub_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row4, text='持续发布 cmd_vel (10Hz)', variable=self.pub_var,
                       command=self._on_pub_toggle).pack(side=tk.LEFT, padx=6)

        # ---- 第 5 行: 归零滑条 + 紧急停车 ----
        row5 = tk.Frame(ctrl)
        row5.pack(fill=tk.X)

        tk.Button(row5, text='紧急停车', width=10, bg='#c0392b', fg='white',
                  command=self._on_emergency_stop).pack(side=tk.RIGHT, padx=2)
        tk.Button(row5, text='归零滑条', width=8,
                  command=self._reset_scales).pack(side=tk.RIGHT, padx=2)

    # ========================================================
    #  内容区: 左栏地图画布, 右栏控制/目标点/遥测
    # ========================================================
    def _build_content_area(self):
        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        content.columnconfigure(0, weight=5)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        self._build_map_area(content)
        self._build_right_panel(content)

    # ---- 左栏: 地图画布 (栅格地图 + 轨迹 + 车位姿 + 目标点) ----
    def _build_map_area(self, parent):
        frame = tk.LabelFrame(parent, text='地图与轨迹', padx=4, pady=4)
        frame.grid(row=0, column=0, sticky='nsew', padx=4)

        # 工具行: 数据状态 + 自适应按钮
        toolbar = tk.Frame(frame)
        toolbar.pack(fill=tk.X)
        self.map_status_label = tk.Label(toolbar, text='地图: 未接收', anchor='w')
        self.map_status_label.pack(side=tk.LEFT)
        tk.Label(toolbar, text='左键: 选中目标点 | 拖拽: 平移 | 滚轮: 缩放',
                 fg='gray').pack(side=tk.LEFT, padx=12)
        tk.Button(toolbar, text='适应地图', width=8,
                  command=self._fit_view).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(frame, bg='#404040', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # 鼠标交互: 点击设目标点 (无拖动时)、拖拽平移、滚轮缩放
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_press)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_release)
        self.canvas.bind('<Button-4>', lambda e: self._on_zoom(e, 1.25))
        self.canvas.bind('<Button-5>', lambda e: self._on_zoom(e, 0.8))
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind('<Configure>', lambda _e: setattr(self, '_map_key', None))

    # ---- 右栏: 控制 + 目标点输入 + 遥测显示 ----
    def _build_right_panel(self, parent):
        right = tk.Frame(parent)
        right.grid(row=0, column=1, sticky='nsew', padx=4)

        # ---- 控制区 ----
        self._build_control_area(right)

        # ---- 目标点发布 ----
        goal = tk.LabelFrame(right, text='目标点 (/car/goal_point)', padx=8, pady=4)
        goal.pack(fill=tk.X, pady=(0, 4))

        row = tk.Frame(goal)
        row.pack(fill=tk.X)
        tk.Label(row, text='X (m)').pack(side=tk.LEFT)
        self.goal_x_entry = tk.Entry(row, width=9)
        self.goal_x_entry.insert(0, '0.0')
        self.goal_x_entry.pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(row, text='Y (m)').pack(side=tk.LEFT)
        self.goal_y_entry = tk.Entry(row, width=9)
        self.goal_y_entry.insert(0, '0.0')
        self.goal_y_entry.pack(side=tk.LEFT, padx=(2, 8))
        tk.Button(row, text='发布', width=6,
                  command=self._on_publish_goal).pack(side=tk.LEFT, padx=4)

        self.goal_label = tk.Label(goal, text='未发布 (点击地图可选中目标点)',
                                   anchor='w', justify=tk.LEFT)
        self.goal_label.pack(fill=tk.X, pady=(2, 0))

        # ---- 相机设置: 曝光时间 + 灯光亮度 (由 sensor_capture 节点转发到相机) ----
        camera = tk.LabelFrame(right, text='相机设置 (/sensor)', padx=8, pady=4)
        camera.pack(fill=tk.X, pady=(0, 4))

        # 曝光行: 输入框 (无范围限制, RGB 2/3 曝光, RGB 0/1 由采集节点按偏差换算) + 发送按钮
        exp_row = tk.Frame(camera)
        exp_row.pack(fill=tk.X)
        tk.Label(exp_row, text='曝光 (ms)').pack(side=tk.LEFT)
        self.exposure_entry = tk.Entry(exp_row, width=9)
        self.exposure_entry.insert(0, '400.0')
        self.exposure_entry.pack(side=tk.LEFT, padx=(2, 8))
        tk.Button(exp_row, text='发送', width=6,
                  command=self._on_send_exposure).pack(side=tk.LEFT, padx=4)

        # 灯光行: 输入框 (限 10~30, 默认 20) + 发送按钮
        light_row = tk.Frame(camera)
        light_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(light_row, text='灯光亮度').pack(side=tk.LEFT)
        self.light_entry = tk.Entry(light_row, width=9)
        self.light_entry.insert(0, '20')
        self.light_entry.pack(side=tk.LEFT, padx=(2, 8))
        tk.Button(light_row, text='发送', width=6,
                  command=self._on_send_light).pack(side=tk.LEFT, padx=4)

        self.camera_label = tk.Label(camera, text='未发送', anchor='w')
        self.camera_label.pack(fill=tk.X, pady=(2, 0))

        # ---- 遥测: 位姿与速度 ----
        pose = tk.LabelFrame(right, text='遥测: 位姿与速度', padx=8, pady=4)
        pose.pack(fill=tk.X, pady=4)
        mono = ('Monospace', 10)
        self.pose_rh_label = tk.Label(pose, justify=tk.LEFT, anchor='w', font=mono)
        self.pose_rh_label.pack(fill=tk.X, pady=2)
        self.pose_lh_label = tk.Label(pose, justify=tk.LEFT, anchor='w', font=mono)
        self.pose_lh_label.pack(fill=tk.X, pady=2)
        self.vel_label = tk.Label(pose, justify=tk.LEFT, anchor='w', font=mono)
        self.vel_label.pack(fill=tk.X, pady=2)

        # ---- 遥测: 车轮与发送指令 ----
        wheel = tk.LabelFrame(right, text='遥测: 车轮与指令', padx=8, pady=4)
        wheel.pack(fill=tk.X, pady=4)
        self.wheel_label = tk.Label(wheel, justify=tk.LEFT, anchor='w', font=mono)
        self.wheel_label.pack(fill=tk.X, pady=2)
        self.cmd_wheel_label = tk.Label(wheel, justify=tk.LEFT, anchor='w', font=mono)
        self.cmd_wheel_label.pack(fill=tk.X, pady=2)
        self.status_label = tk.Label(wheel, justify=tk.LEFT, anchor='w', font=mono)
        self.status_label.pack(fill=tk.X, pady=2)

        # 初始占位文本
        placeholder = '等待遥测数据...'
        for lbl in (self.pose_rh_label, self.pose_lh_label,
                    self.vel_label, self.wheel_label):
            lbl.config(text=placeholder)
        self.cmd_wheel_label.config(text='发送指令 (左手系): 等待数据...')
        self.status_label.config(text='')

    # ========================================================
    #  坐标变换: 世界坐标 (右手系, y 向上) ↔ 画布像素 (y 向下)
    # ========================================================
    def _world_to_pixel(self, wx, wy):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cx, cy = self._view_center
        px = w / 2.0 + (wx - cx) * self._view_scale
        py = h / 2.0 - (wy - cy) * self._view_scale
        return px, py

    def _pixel_to_world(self, px, py):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cx, cy = self._view_center
        wx = cx + (px - w / 2.0) / self._view_scale
        wy = cy - (py - h / 2.0) / self._view_scale
        return wx, wy

    def _fit_view(self):
        """视图自适应: 完整显示当前栅格地图 (无地图时回到原点默认缩放)"""
        with self.node._lock:
            grid = self.node.latest_grid
        w = max(self.canvas.winfo_width(), 2)
        h = max(self.canvas.winfo_height(), 2)
        if grid is None:
            self._view_center = (0.0, 0.0)
            self._view_scale = 30.0
        else:
            res = grid.info.resolution
            map_w = grid.info.width * res
            map_h = grid.info.height * res
            ox = grid.info.origin.position.x
            oy = grid.info.origin.position.y
            self._view_center = (ox + map_w / 2.0, oy + map_h / 2.0)
            if map_w > 0 and map_h > 0:
                self._view_scale = 0.95 * min(w / map_w, h / map_h)
        self._map_key = None   # 视图变化, 强制重绘底图

    # ========================================================
    #  鼠标交互
    # ========================================================
    def _on_mouse_press(self, event):
        self._drag_start = (event.x, event.y, self._view_center)
        self._drag_moved = False

    def _on_mouse_drag(self, event):
        """拖拽平移视图 (移动超过阈值才视为拖拽, 否则松开时视为点击)"""
        if self._drag_start is None:
            return
        sx, sy, (cx, cy) = self._drag_start
        dx = event.x - sx
        dy = event.y - sy
        if abs(dx) > 4 or abs(dy) > 4:
            self._drag_moved = True
        self._view_center = (cx - dx / self._view_scale,
                             cy + dy / self._view_scale)
        self._map_key = None

    def _on_mouse_release(self, event):
        """未发生拖拽的点击: 选中目标点 (仅填充输入框与标记, 不自动发布, 需点击发布按钮)"""
        if self._drag_start is None:
            return
        moved = self._drag_moved
        self._drag_start = None
        if moved:
            return
        wx, wy = self._pixel_to_world(event.x, event.y)
        self._goal_xy = (wx, wy)
        self.goal_x_entry.delete(0, tk.END)
        self.goal_x_entry.insert(0, '%.2f' % wx)
        self.goal_y_entry.delete(0, tk.END)
        self.goal_y_entry.insert(0, '%.2f' % wy)
        self.goal_label.config(text='已选中: x=%.3f, y=%.3f (点发布按钮生效)' % (wx, wy),
                               fg='#2980b9')

    def _on_zoom(self, event, factor):
        """以鼠标位置为中心缩放"""
        w = max(self.canvas.winfo_width(), 2)
        h = max(self.canvas.winfo_height(), 2)
        wx, wy = self._pixel_to_world(event.x, event.y)
        self._view_scale = max(self.SCALE_MIN,
                               min(self.SCALE_MAX, self._view_scale * factor))
        # 保持鼠标下的世界点不动: 反解新的视图中心
        self._view_center = (wx - (event.x - w / 2.0) / self._view_scale,
                             wy + (event.y - h / 2.0) / self._view_scale)
        self._map_key = None

    def _on_mousewheel(self, event):
        """Windows/macOS 滚轮 (Linux 使用 Button-4/5)"""
        self._on_zoom(event, 1.25 if event.delta > 0 else 0.8)

    # ========================================================
    #  界面事件回调
    # ========================================================
    def _on_scale_change(self, _value=None):
        """滑条变化: 更新节点的速度指令缓存 (由定时器发布)"""
        self.node.set_command(self.linear_scale.get(), self.curvature_scale.get())

    def _on_pub_toggle(self):
        with self.node._lock:
            self.node.publish_enabled = self.pub_var.get()

    def _reset_scales(self):
        self.linear_scale.set(0.0)
        self.curvature_scale.set(0.0)
        self.node.set_command(0.0, 0.0)

    def _on_emergency_stop(self):
        """紧急停车: 归零滑条 + 发布零指令 + 切换停车模式"""
        self._reset_scales()
        self.node.emergency_stop()

    def _on_publish_goal(self):
        """读取界面 x/y 输入并发布目标点; 输入非法时提示而不发布"""
        try:
            x = float(self.goal_x_entry.get())
            y = float(self.goal_y_entry.get())
        except ValueError:
            self.goal_label.config(text='输入非法, 请输入数字', fg='#c0392b')
            return
        self._goal_xy = (x, y)
        self.node.send_goal_point(x, y)
        self.goal_label.config(text='已发布: x=%.3f, y=%.3f' % (x, y), fg='#27ae60')

    def _on_send_exposure(self):
        """读取曝光时间输入并发布 (无范围限制, 仅拒绝负值; 为 RGB 2/3 曝光, RGB 0/1 由采集节点按偏差换算)"""
        try:
            exposure_ms = float(self.exposure_entry.get())
        except ValueError:
            self.camera_label.config(text='曝光输入非法, 请输入数字', fg='#c0392b')
            return
        if exposure_ms < 0.0:
            self.camera_label.config(text='曝光必须 ≥ 0', fg='#c0392b')
            return
        self.node.send_exposure(exposure_ms)
        self.camera_label.config(text='已发送曝光: %.1f ms' % exposure_ms, fg='#27ae60')

    def _on_send_light(self):
        """读取灯光亮度输入并发布 (限 10~30, 超范围拒绝)"""
        try:
            intensity = float(self.light_entry.get())
        except ValueError:
            self.camera_label.config(text='灯光输入非法, 请输入数字', fg='#c0392b')
            return
        if not (self.LIGHT_MIN <= intensity <= self.LIGHT_MAX):
            self.camera_label.config(
                text='灯光亮度超出范围, 必须 %.0f~%.0f' % (self.LIGHT_MIN, self.LIGHT_MAX),
                fg='#c0392b')
            return
        self.node.send_light_intensity(intensity)
        self.camera_label.config(text='已发送灯光亮度: %.1f' % intensity, fg='#27ae60')

    # ========================================================
    #  地图底图渲染 (numpy 栅格着色 → 缩放 → PPM bytes → PhotoImage)
    # ========================================================
    def _grid_to_rgb(self, grid):
        """栅格数据 → (H, W, 3) RGB 数组 (未知/地图外为灰, 自由近白, 占据深灰)"""
        data = np.asarray(grid.data, dtype=np.int16)   # -1 未知, 0~100 占据概率
        gray = (self.COLOR_FREE
                - (self.COLOR_FREE - self.COLOR_OCC) * np.clip(data, 0, 100) / 100.0)
        rgb = np.stack([gray, gray, gray], axis=-1)
        unknown = data < 0
        rgb[unknown] = self.COLOR_BG
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _render_map_image(self, grid, w, h):
        """将栅格地图按当前视图渲染为画布大小的 RGB 图像"""
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        gw = grid.info.width
        gh = grid.info.height
        rgb_grid = self._grid_to_rgb(grid)
        cx, cy = self._view_center

        if _HAS_PIL:
            # PIL 双三次缩放: 栅格图 → 完整地图尺寸 → 按视图裁剪到画布范围, 质量好且快
            # 上下翻转: 栅格第 0 行对应最小 y (地图底部), 而图像第 0 行在顶部
            scale_w = max(1, int(round(gw * res * self._view_scale)))
            scale_h = max(1, int(round(gh * res * self._view_scale)))
            img = PILImage.fromarray(
                np.ascontiguousarray(rgb_grid[::-1])).resize(
                (scale_w, scale_h), PILImage.BICUBIC)
            left = int(round((cx - ox) * self._view_scale - w / 2.0))
            top = int(round((oy + gh * res - cy) * self._view_scale - h / 2.0))
            canvas_img = PILImage.new('RGB', (w, h), (self.COLOR_BG,) * 3)
            canvas_img.paste(img, (-left, -top))
            ppm = b'P6\n%d %d\n255\n' % (w, h) + canvas_img.tobytes()
        else:
            # 回退方案: 逐像素最近邻采样 (每像素计算栅格行列)
            xs = cx + (np.arange(w) - w / 2.0) / self._view_scale
            ys = cy - (np.arange(h) - h / 2.0) / self._view_scale
            cols = ((xs - ox) / res).astype(np.int64)[np.newaxis, :]
            rows = ((ys - oy) / res).astype(np.int64)[:, np.newaxis]
            cols = np.broadcast_to(cols, (h, w))
            rows = np.broadcast_to(rows, (h, w))
            valid = (cols >= 0) & (cols < gw) & (rows >= 0) & (rows < gh)
            rgb = np.full((h, w, 3), self.COLOR_BG, dtype=np.uint8)
            rgb[valid] = rgb_grid[rows[valid], cols[valid]]
            ppm = b'P6\n%d %d\n255\n' % (w, h) + rgb.tobytes()

        # Tk PhotoImage 需传入原始 P6 bytes (base64/P3 字符串在 Tk 8.6 解析失败)
        return tk.PhotoImage(data=ppm)

    def _grid_signature(self, grid):
        """地图数据身份标识: 时间戳 + 尺寸 + 分辨率 + 原点"""
        stamp = grid.header.stamp
        info = grid.info
        return (stamp.sec, stamp.nanosec, info.width, info.height,
                info.resolution, info.origin.position.x, info.origin.position.y)

    # ========================================================
    #  画布刷新: 底图 + 矢量图层 (轨迹/车位姿/目标点)
    # ========================================================
    def _update_canvas(self, grid, path, tel):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        # ---- 新地图到达时自动适应视图 ----
        if grid is not None:
            stamp = (grid.header.stamp.sec, grid.header.stamp.nanosec)
            if self._fitted_stamp != stamp:
                self._fitted_stamp = stamp
                self._fit_view()

        # ---- 底图: 视图/地图/画布尺寸变化时重新渲染 ----
        key = (None if grid is None else self._grid_signature(grid),
               round(self._view_scale, 4), self._view_center, w, h)
        if key != self._map_key:
            self._map_key = key
            if grid is not None:
                self._map_photo = self._render_map_image(grid, w, h)
            else:
                self._map_photo = None
            if self._map_image_id is not None:
                self.canvas.delete(self._map_image_id)
                self._map_image_id = None
            if self._map_photo is not None:
                self._map_image_id = self.canvas.create_image(
                    0, 0, anchor=tk.NW, image=self._map_photo, tags='map')

        # ---- 矢量图层 ----
        self.canvas.delete('overlay')

        # 规划轨迹 (蓝色折线, 点数过多时抽稀)
        if path is not None and len(path.poses) > 1:
            poses = path.poses
            step = max(1, len(poses) // 1500)
            pts = []
            for i in range(0, len(poses), step):
                p = poses[i].pose.position
                pts.extend(self._world_to_pixel(p.x, p.y))
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill='#2980b9', width=2,
                                        tags='overlay')

        # 车位姿 (右手系遥测, 橙色箭头)
        if tel is not None:
            yaw = self._quat_to_yaw(tel.qx, tel.qy, tel.qz, tel.qw)
            length, half_w = 0.6, 0.3     # 箭头尺寸 (米)
            cos_a, sin_a = math.cos(yaw), math.sin(yaw)
            tip = (tel.x + length * cos_a, tel.y + length * sin_a)
            left = (tel.x - half_w * sin_a, tel.y + half_w * cos_a)
            right = (tel.x + half_w * sin_a, tel.y - half_w * cos_a)
            poly = []
            for wx, wy in (tip, left, right):
                poly.extend(self._world_to_pixel(wx, wy))
            self.canvas.create_polygon(*poly, fill='#e67e22', outline='black',
                                       tags='overlay')

        # 目标点标记 (红色十字 + 圆圈)
        if self._goal_xy is not None:
            gx, gy = self._world_to_pixel(*self._goal_xy)
            r = 8.0
            self.canvas.create_oval(gx - r, gy - r, gx + r, gy + r,
                                    outline='#e74c3c', width=2, tags='overlay')
            self.canvas.create_line(gx - r * 1.6, gy, gx + r * 1.6, gy,
                                    fill='#e74c3c', width=2, tags='overlay')
            self.canvas.create_line(gx, gy - r * 1.6, gx, gy + r * 1.6,
                                    fill='#e74c3c', width=2, tags='overlay')

    # ========================================================
    #  周期刷新显示 (10Hz)
    # ========================================================
    def _update_display(self):
        with self.node._lock:
            tel = self.node.latest_telemetry
            recv_time = self.node.last_telemetry_time
            wheel_cmd = self.node.latest_wheel_cmd
            grid = self.node.latest_grid
            grid_time = self.node.last_grid_time
            path = self.node.latest_path
            path_time = self.node.last_path_time
            mode = self.node.requested_mode
            external = self.node.external_control_active
            linear = self.node.linear_cmd
            curvature = self.node.curvature_cmd

        # ---- 状态行 ----
        self.mode_label.config(text='脚本: %s' % mode if external else '请求: %s' % mode)
        self.cmd_label.config(text='指令: v=%.3f m/s, κ=%.2f 1/m' % (linear, curvature))

        # ---- 地图/轨迹接收状态 ----
        now = self.node.get_clock().now()
        self.map_status_label.config(
            text='地图(%s): %s   轨迹(%s): %s' % (
                self.node.map_topic, self._age_text(grid_time, now),
                self.node.path_topic, self._age_text(path_time, now)),
            fg='#27ae60' if grid_time is not None else 'gray')

        # ---- 画布 ----
        self._update_canvas(grid, path, tel)

        if tel is None or recv_time is None:
            self.conn_label.config(text='遥测: 未接收', fg='gray')
        else:
            age = (now - recv_time).nanoseconds * 1e-9
            if age < 1.0:
                self.conn_label.config(text='遥测: %.2fs 前' % age, fg='#27ae60')
            else:
                self.conn_label.config(text='遥测: %.1fs 前 (超时)' % age, fg='#c0392b')

            # ---- 位姿与速度 ----
            yaw = self._quat_to_yaw(tel.qx, tel.qy, tel.qz, tel.qw)
            self.pose_rh_label.config(text=(
                '位姿 (右手系)\n'
                '  位置: x=%8.3f  y=%8.3f  z=%8.3f  m\n'
                '  航向: %7.2f°   四元数: (% .3f, % .3f, % .3f, % .3f)'
                % (tel.x, tel.y, tel.z, math.degrees(yaw),
                   tel.qx, tel.qy, tel.qz, tel.qw)))
            self.pose_lh_label.config(text=(
                '位姿 (左手系原始)\n'
                '  位置: x=%8.3f  y=%8.3f  z=%8.3f  m'
                % (tel.x_lh, tel.y_lh, tel.z_lh)))
            self.vel_label.config(text=(
                '车体速度 (右手系)\n'
                '  线速度: vx=%7.3f  vy=%7.3f  vz=%7.3f  m/s\n'
                '  角速度: wx=%7.3f  wy=%7.3f  wz=%7.3f  rad/s'
                % (tel.vx, tel.vy, tel.vz, tel.wx, tel.wy, tel.wz)))

            # ---- 车轮 (轮速 rad/s, 转角显示为度) ----
            self.wheel_label.config(text=(
                '车轮 (FL / FR / RL / RR)\n'
                '  轮速 rad/s: %7.2f %7.2f %7.2f %7.2f\n'
                '  转角 deg  : %7.1f %7.1f %7.1f %7.1f'
                % (tel.wheel_speed_fl, tel.wheel_speed_fr,
                   tel.wheel_speed_rl, tel.wheel_speed_rr,
                   math.degrees(tel.steer_angle_fl), math.degrees(tel.steer_angle_fr),
                   math.degrees(tel.steer_angle_rl), math.degrees(tel.steer_angle_rr))))

        # ---- 实际发送的四轮指令 (左手系) ----
        if wheel_cmd is not None:
            self.cmd_wheel_label.config(text=(
                '发送指令 (左手系)\n'
                '  轮速 rad/s: %7.2f %7.2f %7.2f %7.2f\n'
                '  转角 deg  : %7.1f %7.1f %7.1f %7.1f'
                % (wheel_cmd.wheel_speed[0], wheel_cmd.wheel_speed[1],
                   wheel_cmd.wheel_speed[2], wheel_cmd.wheel_speed[3],
                   math.degrees(wheel_cmd.steer_angle[0]), math.degrees(wheel_cmd.steer_angle[1]),
                   math.degrees(wheel_cmd.steer_angle[2]), math.degrees(wheel_cmd.steer_angle[3]))))

        # 100ms 后再次刷新
        self.root.after(100, self._update_display)

    @staticmethod
    def _age_text(recv_time, now):
        """接收时间 → 状态文本"""
        if recv_time is None:
            return '未接收'
        age = (now - recv_time).nanoseconds * 1e-9
        return '%.1fs 前' % age

    @staticmethod
    def _quat_to_yaw(qx, qy, qz, qw):
        """四元数 → 航向角 (绕 Z 轴, 右手系, 正值 = 向左转)"""
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny, cosy)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = LunarCarGuiNode()

    # 后台线程驱动 ROS2 回调 (订阅/定时器), 主线程运行 tkinter
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        gui = LunarCarGui(node)
        closing = threading.Event()
        def handle_signal(_signum, _frame):
            closing.set()
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        def poll_close():
            if closing.is_set():
                gui.root.destroy()
            else:
                gui.root.after(100, poll_close)
        gui.root.after(100, poll_close)
        gui.run()
    finally:
        if rclpy.ok():
            node.emergency_stop()
            time.sleep(0.15)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
