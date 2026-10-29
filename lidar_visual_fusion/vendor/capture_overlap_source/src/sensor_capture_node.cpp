/**
 * @file sensor_capture_node.cpp
 * @brief ROS2 传感器数据采集节点实现 (C++)
 *
 * 架构:
 *   captureLoop (采集线程):
 *     周期性发送采集指令 → 启动 receiveBatch
 *   receiveBatch:
 *     启动 receiveThread (接收线程) + processThread (处理线程)
 *   receiveThread:
 *     TCP 读取 → 解析文件头 → 提取数据 → 入队 (快速，内存操作)
 *   processThread:
 *     出队 → 磁盘保存 + PNG 解码 + ROS 发布 (耗时，不阻塞接收)
 *
 * 时间戳策略:
 *   legacy_header_us: 保留原包头微秒换算，仅供旧协议兼容。
 *   meta_relative: 同批次使用 meta.json 时间，首批对齐 Orin 请求时刻。
 */

#include "sensor_capture_ros2/sensor_capture_node.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/resource.h>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <set>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace sensor_capture_ros2
{

// ============================================================
//  工具函数 (静态)
// ============================================================

/** @brief 传感器类型 → 名称字符串 (用于 topic key 生成) */
static std::string getSensorNameStatic(uint8_t sensor_type)
{
    switch (sensor_type) {
        case SENSOR_RGB:       return "rgb";
        case SENSOR_TOF:       return "tof";
        case SENSOR_CPU_LIDAR: return "cpulidar";
        case SENSOR_EVENT:     return "event";
        default:               return "sensor_" + std::to_string(sensor_type);
    }
}

/** @brief 在数据中搜索魔数/签名，返回偏移位置 */
static size_t findInData(const std::vector<uint8_t> & data,
                         const std::vector<uint8_t> & pattern, size_t start = 0)
{
    if (pattern.empty() || data.size() < pattern.size()) return std::string::npos;
    auto it = std::search(data.begin() + start, data.end(),
                          pattern.begin(), pattern.end());
    return (it == data.end()) ? std::string::npos : std::distance(data.begin(), it);
}

/** @brief 将 float 按位重新解释为 uint32 (用于协议序列化与校验和计算) */
static uint32_t floatToBits(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

/**
 * @brief 将 CV_16UC1 图像线性拉伸到 0~65535 全量程 (原地归一化)
 * 保持 16 位深度不变，便于归一化后仍以 mono16 PNG 保存
 */
static void normalizeImage16U(cv::Mat & img)
{
    double min_val = 0.0, max_val = 0.0;
    cv::minMaxLoc(img, &min_val, &max_val);
    if (max_val <= min_val) return;  // 图像无有效数据，保持原样
    cv::normalize(img, img, 0, 65535, cv::NORM_MINMAX);
}

/** @brief 将 UE 返回的 UTF-16LE 文件名解码为 UTF-8 */
static std::string decodeUtf16LE(const char * raw, size_t max_bytes)
{
    std::string result;
    for (size_t i = 0; i + 1 < max_bytes; i += 2) {
        uint16_t ch = static_cast<uint8_t>(raw[i]) |
                      (static_cast<uint8_t>(raw[i + 1]) << 8);
        if (ch == 0) break;
        if (ch < 0x80) {
            result += static_cast<char>(ch);
        } else if (ch < 0x800) {
            result += static_cast<char>(0xC0 | (ch >> 6));
            result += static_cast<char>(0x80 | (ch & 0x3F));
        } else {
            result += static_cast<char>(0xE0 | (ch >> 12));
            result += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (ch & 0x3F));
        }
    }
    return result;
}

// ============================================================
//  构造函数 — 初始化参数、创建发布者/订阅者、可选自动启动
// ============================================================

SensorCaptureNode::SensorCaptureNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("sensor_capture_node", options),
  sock_fd_(-1),
  running_(false),
  start_time_(this->now()),
  batch_request_stamp_(this->now()),
  batch_publish_stamp_(this->now()),
  save_data_(true),
  batch_done_(false),
  batch_error_(false)
{
    // ---- 声明参数 (均可通过 launch 或 ros2 param set 覆盖) ----
    this->declare_parameter<std::string>("tcp_host", "192.168.1.22");
    this->declare_parameter<int>("tcp_port", 6665);
    this->declare_parameter<std::string>("save_dir", "/home/yanfa/Env_X/SaveImages");
    this->declare_parameter<int>("batch_count", 600);
    this->declare_parameter<double>("timeout", 120.0);
    this->declare_parameter<double>("batch_interval", 1);
    this->declare_parameter<double>("capture_freq", 2.0);  // 采集指令发送频率 Hz
    this->declare_parameter<std::string>("timestamp_mode", "legacy_header_us");
    this->declare_parameter<bool>("big_endian", false);
    this->declare_parameter<bool>("auto_start", true);
    this->declare_parameter<bool>("save_data", true);
    this->declare_parameter<bool>("publish_thumbnail", false);
    this->declare_parameter<int>("thumb_width", 320);
    this->declare_parameter<int>("thumb_height", 240);
    this->declare_parameter<double>("tof_fov_x", 67.0);
    this->declare_parameter<double>("tof_fov_y", 51.0);
    this->declare_parameter<double>("tof2_fov_x", 55.7);  // tof_2/tof_3 水平视场角 (度)
    this->declare_parameter<double>("tof2_fov_y", 47.7);  // tof_2/tof_3 垂直视场角 (度)
    this->declare_parameter<double>("tof_depth_scale", 0.01);   // 深度原始值→米 (UE 输出为 cm)
    this->declare_parameter<double>("tof_max_range", 9.5);       // TOF 有效最大量程 (米), 超出则剔除
    this->declare_parameter<double>("lidar_scale", 0.01);        // LiDAR 坐标原始值→米 (UE 输出为 cm)
    this->declare_parameter<bool>("normalize_save", false);  // 保存深度图/强度图前是否归一化
    this->declare_parameter<int>("exposure_camera_id", 0xFF);  // (已废弃，保留参数兼容)
    this->declare_parameter<double>("exposure_offset_ms", 600.0);  // 曝光偏差: RGB 0/1 曝光 = RGB 2/3 曝光 + 该值 (ms)

    // 声明可配置的输出 Topic 映射
    // 格式: "key:=topic_name"
    // 通过参数可覆盖，只保留需要的 topic 即可
    this->declare_parameter<std::vector<std::string>>("output_topics", {
        // ---- 彩色相机 (Cam_Left / Cam_Right) ----
        "rgb_0:=/Car/T5/Cam_Left/image_raw/color",
        "rgb_1:=/Car/T5/Cam_Right/image_raw/color",
        // ---- 低曝光相机 (Cam_Left / Cam_Right) ----
        "rgb_2:=/Car/T5/Cam_Left/image_raw/low_exposure",
        "rgb_3:=/Car/T5/Cam_Right/image_raw/low_exposure",
        // ---- ToF 深度相机 (TOF_Left / TOF_Right / TOF_Left2 / TOF_Right2) ----
        // 强度图不再发布 (仅保存磁盘); tof_2/tof_3 本身无强度数据, 仅有点云与深度图发布条目
        "tof_0:=/Car/T5/TOF_Left/points",
        "tof_0_depth:=/Car/T5/TOF_Left/image_raw/depth",
        "tof_1:=/Car/T5/TOF_Right/points",
        "tof_1_depth:=/Car/T5/TOF_Right/image_raw/depth",

        "tof_2:=/Car/T5/TOF_Left2/points",
        "tof_2_depth:=/Car/T5/TOF_Left2/image_raw/depth",
        "tof_3:=/Car/T5/TOF_Right2/points",
        "tof_3_depth:=/Car/T5/TOF_Right2/image_raw/depth",

        
       // ---- 事件相机 (event_camera_left / event_camera_right) ----
        "event_0:=/Car/T5/event_camera_left/cd_events",
        "event_1:=/Car/T5/event_camera_right/cd_events",
       
        // ---- 激光雷达 (OS1) ----
        "cpulidar_0:=/Car/T5/OS1/points",
        
        "galaxy_set:=/galaxy_node/set_camera_params",
    });

    // ---- 读取参数到成员变量 ----
    tcp_host_ = this->get_parameter("tcp_host").as_string();
    tcp_port_ = this->get_parameter("tcp_port").as_int();
    save_dir_ = this->get_parameter("save_dir").as_string();
    batch_count_ = this->get_parameter("batch_count").as_int();
    timeout_ = this->get_parameter("timeout").as_double();
    batch_interval_ = this->get_parameter("batch_interval").as_double();
    capture_freq_ = this->get_parameter("capture_freq").as_double();
    timestamp_mode_ = this->get_parameter("timestamp_mode").as_string();
    if (timestamp_mode_ != "legacy_header_us" && timestamp_mode_ != "meta_relative") {
        throw std::invalid_argument("timestamp_mode must be legacy_header_us or meta_relative");
    }
    batchWaitSeconds(capture_freq_, batch_interval_, 0.0);  // validate before starting threads
    RCLCPP_INFO(this->get_logger(), "Timestamp mode: %s", timestamp_mode_.c_str());
    big_endian_ = this->get_parameter("big_endian").as_bool();
    save_data_ = this->get_parameter("save_data").as_bool();
    publish_thumbnail_ = this->get_parameter("publish_thumbnail").as_bool();
    thumb_width_ = this->get_parameter("thumb_width").as_int();
    thumb_height_ = this->get_parameter("thumb_height").as_int();
    tof_fov_x_deg_ = this->get_parameter("tof_fov_x").as_double();
    tof_fov_y_deg_ = this->get_parameter("tof_fov_y").as_double();
    tof2_fov_x_deg_ = this->get_parameter("tof2_fov_x").as_double();
    tof2_fov_y_deg_ = this->get_parameter("tof2_fov_y").as_double();
    tof_depth_scale_ = this->get_parameter("tof_depth_scale").as_double();
    tof_max_range_ = this->get_parameter("tof_max_range").as_double();
    lidar_scale_ = this->get_parameter("lidar_scale").as_double();
    normalize_save_ = this->get_parameter("normalize_save").as_bool();
    exposure_camera_id_ = this->get_parameter("exposure_camera_id").as_int();
    exposure_offset_ms_ = this->get_parameter("exposure_offset_ms").as_double();
    topic_mapping_ = this->get_parameter("output_topics").as_string_array();

    configurePublication();

    // ---- 根据 output_topics 动态创建发布者 ----
    setupPublishers();

    // ---- 元数据/状态 发布者 ----
    auto qos10 = rclcpp::QoS(1).transient_local();
    meta_pub_ = this->create_publisher<std_msgs::msg::String>("/sensor/meta", qos10);
    batch_status_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/sensor/batch_status", rclcpp::QoS(1));

    // ---- 外部触发采集订阅 (发送 Empty 消息启动采集) ----
    sub_start_capture_ = this->create_subscription<std_msgs::msg::Empty>(
        "/sensor/start_capture", rclcpp::QoS(1),
        std::bind(&SensorCaptureNode::onStartCaptureTrigger, this,
                  std::placeholders::_1));

    // ---- 曝光时间设置订阅 (Float32, 单位毫秒, 值为 RGB 2/3 曝光, 加增量后转发给 RGB 0/1) ----
    sub_set_exposure_ = this->create_subscription<std_msgs::msg::Float32>(
        "/Car/T1/camera_exposure", rclcpp::QoS(1),
        std::bind(&SensorCaptureNode::onSetExposure, this,
                  std::placeholders::_1));

    // ---- 灯光亮度设置订阅 (Float32, 0.0 ~ 1.0 或更大) ----
    sub_set_light_intensity_ = this->create_subscription<std_msgs::msg::Float32>(
        "/sensor/set_light_intensity", rclcpp::QoS(1),
        std::bind(&SensorCaptureNode::onSetLightIntensity, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
        "节点初始化完成 | TCP: %s:%d | 保存目录: %s | 批次: %d",
        tcp_host_.c_str(), tcp_port_, save_dir_.c_str(), batch_count_);
    RCLCPP_INFO(this->get_logger(),
        "缩略图: %s (%dx%d) | TOF FOV: %.1f°×%.1f° | 缩略图发布者数: %zu",
        publish_thumbnail_ ? "开启" : "关闭",
        thumb_width_, thumb_height_,
        tof_fov_x_deg_, tof_fov_y_deg_,
        thumb_publishers_.size());

    // ---- 自动启动采集 (如果配置) ----
    if (this->get_parameter("auto_start").as_bool()) {
        startCapture();
    }
}

SensorCaptureNode::~SensorCaptureNode()
{
    running_ = false;
    publication_queue_.close(true);
    queue_cv_.notify_all();
    // 先关闭 socket 解除 recv() 阻塞，否则 join() 会死锁
    if (sock_fd_ >= 0) {
        shutdown(sock_fd_, SHUT_RDWR);
    }
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (publication_thread_.joinable()) publication_thread_.join();
    disconnectFromServer();
}

// ============================================================
//  发布者初始化 — 根据 output_topics 参数动态创建
//  解析 "key:=topic" 格式，自动推断消息类型
// ============================================================

void SensorCaptureNode::setupPublishers()
{
    // 图像也使用 reliable QoS:
    // best_effort 发布大图像 (单帧 ~7MB) 时, DDS 分片后任一片丢失即整帧丢弃,
    // 且批次集中发布时订阅端接收缓冲区易溢出, 实测导致订阅者几乎收不到图像。
    // reliable 发布者同时兼容 reliable 和 best_effort 订阅者 (RViz 两种配置均可收)。
    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

    // 点云使用 reliable QoS:
    // 1) reliable 发布者同时兼容 Reliable 和 Best Effort 订阅者 (RViz 两种配置均可收)
    // 2) TOF 点云单帧约 4.9MB，best_effort 下大消息易丢包，reliable 保证可靠传输
    auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

    // 根据参数 output_topics 动态创建发布者
    // 参数格式: "key:=topic_name"，通过 topic 名称自动推断消息类型
    for (const auto & entry : topic_mapping_) {
        // 解析 "key:=topic" 格式
        auto sep_pos = entry.find(":=");
        if (sep_pos == std::string::npos) {
            RCLCPP_WARN(this->get_logger(), "无效的 output_topics 条目 (缺少 ':='): %s",
                        entry.c_str());
            continue;
        }
        std::string key = entry.substr(0, sep_pos);
        std::string topic = entry.substr(sep_pos + 2);

        if (key.empty() || topic.empty()) {
            RCLCPP_WARN(this->get_logger(), "无效的 output_topics 条目 (key 或 topic 为空): %s",
                        entry.c_str());
            continue;
        }

        std::string msg_type = inferMsgType(topic);

        PublisherInfo info;
        info.msg_type = msg_type;

        if (msg_type == "image") {
            info.publisher = this->create_publisher<sensor_msgs::msg::Image>(
                topic, sensor_qos);
        } else if (msg_type == "pointcloud") {
            info.publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                topic, cloud_qos);
        } else if (msg_type == "camera_info") {
            info.publisher = this->create_publisher<sensor_msgs::msg::CameraInfo>(
                topic, sensor_qos);
        } else if (msg_type == "event_array") {
            info.publisher = this->create_publisher<sjcam_msgs::msg::EventArray>(
                topic, sensor_qos);
        } else {
            info.publisher = this->create_publisher<std_msgs::msg::ByteMultiArray>(
                topic, sensor_qos);
        }

        publishers_[key] = info;
        RCLCPP_INFO(this->get_logger(), "创建发布者: %s -> %s (%s)",
                    key.c_str(), topic.c_str(), msg_type.c_str());

        // 缩略图发布者: 仅对 RGB 图像创建，topic 追加 /thumb
        if (publish_thumbnail_ && msg_type == "image" &&
            key.size() > 4 && key.substr(0, 4) == "rgb_") {
            std::string thumb_topic = topic + "/thumb";
            auto thumb_pub = this->create_publisher<sensor_msgs::msg::Image>(
                thumb_topic, sensor_qos);
            thumb_publishers_[key] = thumb_pub;
            RCLCPP_INFO(this->get_logger(), "创建缩略图发布者: %s -> %s (%dx%d)",
                        key.c_str(), thumb_topic.c_str(), thumb_width_, thumb_height_);
        }

        // TOF 深度图/强度图发布者: 根据 key 后缀分类存储
        if (msg_type == "image" && key.size() > 4 && key.substr(0, 4) == "tof_") {
            if (key.rfind("_depth") == key.size() - 6) {
                std::string base = key.substr(0, key.size() - 6);  // e.g. "tof_0"
                tof_depth_publishers_[base] =
                    this->create_publisher<sensor_msgs::msg::Image>(topic, sensor_qos);
                RCLCPP_INFO(this->get_logger(), "创建 TOF 深度图发布者: %s -> %s",
                            key.c_str(), topic.c_str());
            } else if (key.rfind("_intensity") == key.size() - 10) {
                std::string base = key.substr(0, key.size() - 10);  // e.g. "tof_0"
                tof_intensity_publishers_[base] =
                    this->create_publisher<sensor_msgs::msg::Image>(topic, sensor_qos);
                RCLCPP_INFO(this->get_logger(), "创建 TOF 强度图发布者: %s -> %s",
                            key.c_str(), topic.c_str());
            }
        }

        // 事件相机累积图像发布者: topic 追加 /image
        if (msg_type == "event_array") {
            std::string img_topic = topic + "/image";
            event_image_publishers_[key] =
                this->create_publisher<sensor_msgs::msg::Image>(img_topic, sensor_qos);
            RCLCPP_INFO(this->get_logger(), "创建事件图像发布者: %s -> %s",
                        key.c_str(), img_topic.c_str());
        }
    }
}

/**
 * @brief 根据 topic 名称推断消息类型
 *
 * 匹配规则 (按优先级):
 *   /image_raw, /image              → "image"
 *   /cd_events, /events             → "event_array"
 *   /points, /scan                  → "pointcloud"
 *   /camera_info                    → "camera_info"
 *   其他                            → "byte_array"
 */
std::string SensorCaptureNode::inferMsgType(const std::string & topic)
{
    if (topic.find("/image_raw") != std::string::npos ||
        topic.find("/image") != std::string::npos)
    {
        return "image";
    }
    if (topic.find("/cd_events") != std::string::npos ||
        topic.find("/events") != std::string::npos)
    {
        return "event_array";
    }
    if (topic.find("/points") != std::string::npos ||
        topic.find("/scan") != std::string::npos)
    {
        return "pointcloud";
    }
    if (topic.find("/camera_info") != std::string::npos) {
        return "camera_info";
    }
    return "byte_array";
}

// ============================================================
//  Topic 触发回调
// ============================================================

void SensorCaptureNode::onStartCaptureTrigger(
    const std_msgs::msg::Empty::SharedPtr /*msg*/)
{
    if (running_) {
        RCLCPP_WARN(this->get_logger(), "采集已在运行中，忽略重复触发");
    } else {
        RCLCPP_INFO(this->get_logger(), "收到触发消息，开始采集");
        startCapture();
    }
}

/**
 * @brief 曝光时间设置回调: 接收值为 RGB 2/3 的曝光时间，交由 applyExposure 下发
 */
void SensorCaptureNode::onSetExposure(
    const std_msgs::msg::Float32::SharedPtr msg)
{
    applyExposure(msg->data, "订阅消息");
}

/**
 * @brief 下发曝光设置到所有 RGB 相机 (共 4 条指令, 逐相机单独发送):
 *        RGB 2/3 直接使用传入值; RGB 0/1 = 传入值 + exposure_offset_ms_ 偏差
 *
 * 注意: 每条指令独立成包发送，指令间留出短暂间隔，确保接收端逐条解析，
 *       避免连续小包被内核合并到同一 TCP 段中。
 */
void SensorCaptureNode::applyExposure(float exposure_23_ms, const std::string & source)
{
    if (exposure_23_ms < 0.0f) {
        RCLCPP_WARN(this->get_logger(), "曝光时间无效 (%.3f ms, 必须 ≥ 0)，已忽略 [%s]",
                    exposure_23_ms, source.c_str());
        return;
    }

    RCLCPP_INFO(this->get_logger(),
        "下发曝光设置 [%s]: RGB 2/3 = %.3f ms, 偏差 = %.3f ms",
        source.c_str(), exposure_23_ms, exposure_offset_ms_);

    // RGB 0/1 (彩色相机) 曝光 = 传入曝光 + 偏差 (负值钳制为 0)
    float exposure_01 = exposure_23_ms + static_cast<float>(exposure_offset_ms_);
    if (exposure_01 < 0.0f) {
        RCLCPP_WARN(this->get_logger(),
            "RGB 0/1 曝光计算结果为负 (%.3f ms)，已钳制为 0", exposure_01);
        exposure_01 = 0.0f;
    }

    // 四个相机逐条单独发送 (不合并)，每条之间短暂等待，确保接收端逐条接收 (共 4 条指令)
    struct ExposureTarget { uint8_t camera_id; float exposure_ms; };
    const ExposureTarget targets[4] = {
        {2, exposure_23_ms}, {3, exposure_23_ms},
        {0, exposure_01},    {1, exposure_01},
    };

    for (int i = 0; i < 4; ++i) {
        auto packet = buildSetExposure(SENSOR_RGB, targets[i].camera_id, targets[i].exposure_ms);
        if (sendCommand(packet, "曝光设置指令")) {
            RCLCPP_INFO(this->get_logger(),
                "已发送曝光设置: camera_id=%d, exposure=%.3f ms",
                static_cast<int>(targets[i].camera_id), targets[i].exposure_ms);
        }
        // 指令间短暂间隔，确保每条指令作为独立数据包到达接收端而非被合并发送
        if (i < 3) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

/** @brief 灯光亮度设置回调: 构建灯光亮度指令包并通过 TCP 发送 */
void SensorCaptureNode::onSetLightIntensity(
    const std_msgs::msg::Float32::SharedPtr msg)
{
    float intensity = msg->data;
    if (intensity < 0.0f) {
        RCLCPP_WARN(this->get_logger(), "灯光亮度无效 (%.3f, 必须 ≥ 0)，已忽略",
                    intensity);
        return;
    }

    auto packet = buildSetLightIntensity(intensity);
    if (sendCommand(packet, "灯光亮度设置指令")) {
        RCLCPP_INFO(this->get_logger(), "已发送灯光亮度设置: intensity=%.3f", intensity);
    }
}

// ============================================================
//  采集主逻辑
//  startCapture: 启动采集线程
//  captureLoop:  循环发送采集指令 + 接收数据
// ============================================================

/** @brief 启动采集 (防重复，创建采集线程) */
void SensorCaptureNode::startCapture()
{
    if (running_) {
        RCLCPP_WARN(this->get_logger(), "采集已在运行中，忽略重复启动");
        return;
    }
    if (capture_thread_.joinable()) capture_thread_.join();
    if (publication_thread_.joinable()) publication_thread_.join();
    publication_queue_.reset();
    running_ = true;
    if (parallel_publication_) publication_thread_ = std::thread(&SensorCaptureNode::publicationLoop, this);
    capture_thread_ = std::thread(&SensorCaptureNode::captureLoop, this);
    RCLCPP_INFO(this->get_logger(), "采集线程已启动");
}

/**
 * @brief 采集主循环 (在独立线程中运行, 启动时提升线程优先级)
 *
 * 每个批次:
 *   1. 检查连接，必要时重连
 *   2. 逐个发送传感器采集指令 (间隔由 capture_freq_ 控制)
 *   3. 调用 receiveBatch 接收并处理数据
 *   4. 批次间隔等待
 */
void SensorCaptureNode::captureLoop()
{
    // ---- 提升线程优先级: 保证采集指令发送不被其他进程/线程抢占延迟 ----
    {
        sched_param param{};
        int policy = SCHED_RR;
        // 实时优先级取中值，避免占用最高优先级影响系统关键任务
        param.sched_priority =
            (sched_get_priority_min(policy) + sched_get_priority_max(policy)) / 2;
        if (pthread_setschedparam(pthread_self(), policy, &param) == 0) {
            RCLCPP_INFO(this->get_logger(),
                "采集线程已提升为实时调度 (SCHED_RR, 优先级=%d)",
                param.sched_priority);
        } else {
            // 无实时调度权限 (需 CAP_SYS_NICE/root) 时，退化为提高 nice 值
            errno = 0;
            if (setpriority(PRIO_PROCESS, 0, -15) == 0) {
                RCLCPP_INFO(this->get_logger(),
                    "实时调度不可用，采集线程 nice 值已提升为 -15");
            } else {
                RCLCPP_WARN(this->get_logger(),
                    "采集线程优先级提升失败 (可用 sudo 运行或配置 CAP_SYS_NICE 启用实时调度)");
            }
        }
    }

    fs::create_directories(save_dir_);
    int success_count = 0;

    // 与 main.py 一致: 每批次发送一条通配符采集指令 (0xFF, 0xFF)
    try {
        for (int batch_index = 1; batch_index <= batch_count_; ++batch_index) {
            if (!running_) break;

            // 如果连接断开，尝试重新连接
            if (sock_fd_ < 0) {
                RCLCPP_WARN(this->get_logger(), "连接已断开，尝试重连...");
                if (!connectToServer()) {
                    RCLCPP_ERROR(this->get_logger(), "重连失败，1s 后重试");
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    --batch_index;  // 重试当前批次
                    continue;
                }
            }

            RCLCPP_INFO(this->get_logger(), "===== 批次 %d/%d =====",
                        batch_index, batch_count_);

            // 发送通配符采集命令 (与 main.py build_start_capture 一致)
            const auto request_started = std::chrono::steady_clock::now();
            batch_request_steady_ = request_started;
            batch_request_stamp_ = this->now();
            auto packet = buildStartCapture(0xFF, 0xFF);
            bool send_ok = true;
            {
                // 与订阅回调的 sendCommand 互斥，避免并发写入导致数据交错
                std::lock_guard<std::mutex> lock(send_mutex_);
                ssize_t ret = send(sock_fd_, packet.data(), packet.size(), MSG_NOSIGNAL);
                if (ret < 0) {
                    RCLCPP_ERROR(this->get_logger(), "发送失败，断开连接");
                    close(sock_fd_);
                    sock_fd_ = -1;
                    send_ok = false;
                }
            }

            if (!send_ok) {
                --batch_index;  // 重试当前批次
                std::this_thread::sleep_for(std::chrono::seconds(1));
                RCLCPP_INFO(this->get_logger(), "发送失败, CF...");
                continue;
            }

            RCLCPP_INFO(this->get_logger(), "已发送通配符采集命令 (0xFF, 0xFF)，等待数据...");

            if (receiveBatch(batch_index)) {
                ++success_count;
                RCLCPP_INFO(this->get_logger(), "批次 %d/%d 完成",
                            batch_index, batch_count_);

                // Honor capture_freq as a maximum request rate, retaining the configured minimum gap.
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - request_started).count();
                const double wait_seconds = batchWaitSeconds(capture_freq_, batch_interval_, elapsed);
                if (batch_index < batch_count_ && wait_seconds > 0.0) {
                    RCLCPP_INFO(this->get_logger(), "等待 %.1fs 后进入下一批次...",
                                wait_seconds);
                    // 分段 sleep 以便及时响应 running_ = false
                    double waited = 0.0;
                    while (running_ && waited < wait_seconds) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        waited += 0.1;
                    }
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "批次 %d/%d 失败，断开重连",
                             batch_index, batch_count_);
                if (sock_fd_ >= 0) {
                    close(sock_fd_);
                    sock_fd_ = -1;
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
                --batch_index;  // 重试当前批次
                continue;
            }
        }
    } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "采集异常: %s", e.what());
    }

    disconnectFromServer();
    publication_queue_.close(!rclcpp::ok());
    if (publication_thread_.joinable()) publication_thread_.join();
    running_ = false;

    // 发布完成状态
    auto status_msg = std_msgs::msg::String();
    status_msg.data = "{\"status\":\"finished\",\"success_count\":" +
                      std::to_string(success_count) + ",\"total_count\":" +
                      std::to_string(batch_count_) + "}";
    if (rclcpp::ok()) batch_status_pub_->publish(status_msg);

    RCLCPP_INFO(this->get_logger(), "采集结束: %d/%d 批次成功",
                success_count, batch_count_);
}

// ============================================================
//  TCP 连接管理
//  持久连接，断开后自动重连
//  TCP 缓冲区 50MB，避免大数据量阻塞
// ============================================================

/** @brief 建立到 UE 服务器的 TCP 持久连接 */
bool SensorCaptureNode::connectToServer()
{
    RCLCPP_INFO(this->get_logger(), "正在连接 %s:%d ...",
                tcp_host_.c_str(), tcp_port_);

    sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        RCLCPP_ERROR(this->get_logger(), "创建 socket 失败");
        return false;
    }

    // 增大 TCP 收发缓冲区到 50MB，避免大数据量时阻塞
    int buf_size = 50 * 1024 * 1024;
    setsockopt(sock_fd_, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));
    setsockopt(sock_fd_, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));

    // 禁用 Nagle 算法: 每次 send() 立即发出，避免多条设置指令被合并到同一 TCP 段
    int no_delay = 1;
    setsockopt(sock_fd_, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));

    // 设置超时
    struct timeval tv;
    tv.tv_sec = static_cast<long>(timeout_);
    tv.tv_usec = 0;
    setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port_);
    inet_pton(AF_INET, tcp_host_.c_str(), &addr.sin_addr);

    if (connect(sock_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        RCLCPP_ERROR(this->get_logger(), "连接失败: %s:%d",
                     tcp_host_.c_str(), tcp_port_);
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "已连接到 UE: %s:%d",
                tcp_host_.c_str(), tcp_port_);
    last_completed_batch_id_.reset();
    return true;
}

void SensorCaptureNode::disconnectFromServer()
{
    if (sock_fd_ >= 0) {
        close(sock_fd_);
        sock_fd_ = -1;
        RCLCPP_INFO(this->get_logger(), "TCP 连接已关闭");
    }
}

// ============================================================
//  协议实现
//  采集指令: 包头(MAGIC+CMD+SIZE+CKSUM) + 载荷(HEAD+类型+ID+CKSUM)
// ============================================================

/** @brief 构建单个传感器采集指令包 */
std::vector<uint8_t> SensorCaptureNode::buildStartCapture(uint8_t sensor_type, uint8_t camera_id)
{
    // 构建内部命令
    std::vector<uint8_t> cmd;

    // payload: HEAD_CAPTURE_REQUEST + sensor_type + camera_id + checksum
    pushU32(cmd, HEAD_CAPTURE_REQUEST);
    cmd.push_back(sensor_type);
    cmd.push_back(camera_id);
    uint8_t checksum = static_cast<uint8_t>(
        (HEAD_CAPTURE_REQUEST + sensor_type + camera_id) & 0xFF);
    cmd.push_back(checksum);

    // 外层包头: MAGIC + CMD_START_CAPTURE + payload_size + header_checksum
    uint32_t payload_size = static_cast<uint32_t>(cmd.size());
    uint32_t header_checksum = MAGIC ^ CMD_START_CAPTURE ^ payload_size;

    std::vector<uint8_t> packet;
    pushU32(packet, MAGIC);
    pushU32(packet, CMD_START_CAPTURE);
    pushU32(packet, payload_size);
    pushU32(packet, header_checksum);
    packet.insert(packet.end(), cmd.begin(), cmd.end());

    return packet;
}

/** @brief 构建曝光时间设置指令包 (对应 UE 的 FSetExposureCommand) */
std::vector<uint8_t> SensorCaptureNode::buildSetExposure(
    uint8_t sensor_type, uint8_t camera_id, float exposure_ms)
{
    // 构建内部命令: HEAD + SensorType + CameraID + ExposureMs(float) + Checksum
    std::vector<uint8_t> cmd;

    pushU32(cmd, HEAD_SET_EXPOSURE);
    cmd.push_back(sensor_type);
    cmd.push_back(camera_id);

    // 曝光时间: float 按位转为 uint32 后，按配置的字节序写入 (与 UE *(uint32*)&ExposureMs 一致)
    uint32_t exposure_bits = floatToBits(exposure_ms);
    pushU32(cmd, exposure_bits);

    // 校验和: (Head + SensorType + CameraID + *(uint32*)&ExposureMs) & 0xFF
    uint8_t checksum = static_cast<uint8_t>(
        (HEAD_SET_EXPOSURE + sensor_type + camera_id + exposure_bits) & 0xFF);
    cmd.push_back(checksum);

    // 外层包头: MAGIC + CMD_SET_EXPOSURE + payload_size + header_checksum
    uint32_t payload_size = static_cast<uint32_t>(cmd.size());
    uint32_t header_checksum = MAGIC ^ CMD_SET_EXPOSURE ^ payload_size;

    std::vector<uint8_t> packet;
    pushU32(packet, MAGIC);
    pushU32(packet, CMD_SET_EXPOSURE);
    pushU32(packet, payload_size);
    pushU32(packet, header_checksum);
    packet.insert(packet.end(), cmd.begin(), cmd.end());

    return packet;
}

/** @brief 构建灯光亮度设置指令包 (对应 UE 的 FSetLightIntensityCommand) */
std::vector<uint8_t> SensorCaptureNode::buildSetLightIntensity(float intensity)
{
    // 构建内部命令: HEAD + Intensity(float) + Checksum
    std::vector<uint8_t> cmd;

    pushU32(cmd, HEAD_SET_LIGHT_INTENSITY);

    // 亮度值: float 按位转为 uint32 后，按配置的字节序写入 (与 UE *(uint32*)&Intensity 一致)
    uint32_t intensity_bits = floatToBits(intensity);
    pushU32(cmd, intensity_bits);

    // 校验和: (Head + *(uint32*)&Intensity) & 0xFF
    uint8_t checksum = static_cast<uint8_t>(
        (HEAD_SET_LIGHT_INTENSITY + intensity_bits) & 0xFF);
    cmd.push_back(checksum);

    // 外层包头: MAGIC + CMD_SET_LIGHT_INTENSITY + payload_size + header_checksum
    uint32_t payload_size = static_cast<uint32_t>(cmd.size());
    uint32_t header_checksum = MAGIC ^ CMD_SET_LIGHT_INTENSITY ^ payload_size;

    std::vector<uint8_t> packet;
    pushU32(packet, MAGIC);
    pushU32(packet, CMD_SET_LIGHT_INTENSITY);
    pushU32(packet, payload_size);
    pushU32(packet, header_checksum);
    packet.insert(packet.end(), cmd.begin(), cmd.end());

    return packet;
}

/** @brief 线程安全发送指令包: 未连接时自动尝试连接，发送失败时关闭连接 */
bool SensorCaptureNode::sendCommand(const std::vector<uint8_t> & packet,
                                    const std::string & desc)
{
    std::lock_guard<std::mutex> lock(send_mutex_);

    // 当前无连接时先尝试连接 (采集未运行时也能下发设置指令)
    if (sock_fd_ < 0) {
        RCLCPP_INFO(this->get_logger(), "TCP 未连接，尝试连接以发送 %s ...",
                    desc.c_str());
        if (!connectToServer()) {
            RCLCPP_ERROR(this->get_logger(), "连接失败，无法发送 %s", desc.c_str());
            return false;
        }
    }

    ssize_t ret = send(sock_fd_, packet.data(), packet.size(), MSG_NOSIGNAL);
    if (ret < 0) {
        RCLCPP_ERROR(this->get_logger(), "发送 %s 失败，断开连接", desc.c_str());
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }
    return true;
}

/** @brief 按配置的字节序读取 uint32 (big_endian_ 或 little_endian) */
uint32_t SensorCaptureNode::readU32(const uint8_t * p) const
{
    if (big_endian_) {
        // 大端序: 高位字节在前
        return (static_cast<uint32_t>(p[0]) << 24)
             | (static_cast<uint32_t>(p[1]) << 16)
             | (static_cast<uint32_t>(p[2]) << 8)
             |  static_cast<uint32_t>(p[3]);
    } else {
        // 小端序: 低位字节在前
        return static_cast<uint32_t>(p[0])
             | (static_cast<uint32_t>(p[1]) << 8)
             | (static_cast<uint32_t>(p[2]) << 16)
             | (static_cast<uint32_t>(p[3]) << 24);
    }
}

/** @brief 按配置的字节序写入 uint32 (big_endian_ 或 little_endian) */
void SensorCaptureNode::pushU32(std::vector<uint8_t> & buf, uint32_t v) const
{
    if (big_endian_) {
        // 大端序: 高位字节在前
        buf.push_back((v >> 24) & 0xFF);
        buf.push_back((v >> 16) & 0xFF);
        buf.push_back((v >> 8) & 0xFF);
        buf.push_back(v & 0xFF);
    } else {
        // 小端序: 低位字节在前
        buf.push_back(v & 0xFF);
        buf.push_back((v >> 8) & 0xFF);
        buf.push_back((v >> 16) & 0xFF);
        buf.push_back((v >> 24) & 0xFF);
    }
}

/** @brief 循环 recv 直到精确接收 size 字节 */
bool SensorCaptureNode::recvExact(std::vector<uint8_t> & buffer, size_t size)
{
    buffer.resize(size);
    size_t received = 0;
    while (received < size) {
        ssize_t n = recv(sock_fd_, buffer.data() + received, size - received, 0);
        if (n <= 0) return false;
        received += n;
    }
    return true;
}

bool SensorCaptureNode::recvPacket(uint32_t & command, std::vector<uint8_t> & payload)
{
    std::vector<uint8_t> header_buf;
    if (!recvExact(header_buf, 16)) return false;

    // 使用 readU32 按配置的字节序解析包头
    uint32_t magic = readU32(header_buf.data());
    command = readU32(header_buf.data() + 4);
    uint32_t payload_size = readU32(header_buf.data() + 8);
    uint32_t checksum = readU32(header_buf.data() + 12);

    if (magic != MAGIC) return false;

    uint32_t calc_checksum = MAGIC ^ command ^ payload_size;
    if (checksum != calc_checksum) return false;

    // Bound allocation before trusting a TCP size field (also applies to legacy mode).
    if (payload_size > publication_batch_bytes_ || payload_size < 4) return false;
    if (!recvExact(payload, payload_size)) return false;
    return true;
}

bool SensorCaptureNode::parseFileHeader(
    const std::vector<uint8_t> & data, SensorFileHeader & header)
{
    if (data.size() < 28) return false;

    const uint8_t * p = data.data();

    // 使用 readU32 按配置的字节序解析文件头
    header.head = readU32(p);
    if (header.head != HEAD_SENSOR_DATA) return false;

    header.batch_id = readU32(p + 4);
    header.total_count = readU32(p + 8);
    header.sequence = readU32(p + 12);
    header.file_size = readU32(p + 16);
    header.timestamp = readU32(p + 20);
    header.sensor_type = p[24];
    header.camera_id = p[25];
    header.flags = p[26];
    header.reserved = p[27];

    // 解析 UTF-16LE 文件名 (偏移 28, 最多 128 字节)
    size_t name_bytes = std::min<size_t>(128, data.size() - 28);
    header.file_name = decodeUtf16LE(
        reinterpret_cast<const char *>(p + 28), name_bytes);

    if (header.file_name.empty()) {
        std::string sname = getSensorNameStatic(header.sensor_type);
        header.file_name = sname + "_" + std::to_string(header.camera_id) +
                           "_" + std::to_string(header.sequence) + ".bin";
    }

    return true;
}

size_t SensorCaptureNode::findPayloadStart(
    const std::vector<uint8_t> & data, const std::string & file_name)
{
    // meta.json → 找 '{'
    if (file_name == "meta.json") {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] == '{') return i;
        }
        return 0;
    }

    // PNG → 找 PNG 魔数
    if (file_name.size() >= 4 && file_name.substr(file_name.size() - 4) == ".png") {
        size_t pos = findInData(data, PNG_MAGIC);
        if (pos != std::string::npos) return pos;
        for (size_t off : {92, 156}) {
            if (off + 8 <= data.size() &&
                std::equal(PNG_MAGIC.begin(), PNG_MAGIC.end(), data.begin() + off))
                return off;
        }
        return 0;
    }

    // NPY → 找 NPY 魔数
    if (file_name.size() >= 4 && file_name.substr(file_name.size() - 4) == ".npy") {
        size_t pos = findInData(data, NPY_MAGIC);
        if (pos != std::string::npos) return pos;
        for (size_t off : {92, 156}) {
            if (off + 6 <= data.size() &&
                std::equal(NPY_MAGIC.begin(), NPY_MAGIC.end(), data.begin() + off))
                return off;
        }
        return 0;
    }

    return 156;
}

std::vector<uint8_t> SensorCaptureNode::extractFileData(
    const std::vector<uint8_t> & payload, const SensorFileHeader & header)
{
    size_t start = findPayloadStart(payload, header.file_name);

    // 修正偏移
    bool is_png = header.file_name.size() >= 4 &&
                  header.file_name.substr(header.file_name.size() - 4) == ".png";
    bool is_npy = header.file_name.size() >= 4 &&
                  header.file_name.substr(header.file_name.size() - 4) == ".npy";

    if (is_png && start + 8 <= payload.size()) {
        if (!std::equal(PNG_MAGIC.begin(), PNG_MAGIC.end(), payload.begin() + start)) {
            if (payload.size() >= 164 &&
                std::equal(PNG_MAGIC.begin(), PNG_MAGIC.end(), payload.begin() + 156))
                start = 156;
        }
    }

    if (is_npy && start + 6 <= payload.size()) {
        if (!std::equal(NPY_MAGIC.begin(), NPY_MAGIC.end(), payload.begin() + start)) {
            if (payload.size() >= 162 &&
                std::equal(NPY_MAGIC.begin(), NPY_MAGIC.end(), payload.begin() + 156))
                start = 156;
        }
    }

    size_t end = std::min(start + header.file_size, payload.size());
    if (start >= payload.size()) return {};
    return std::vector<uint8_t>(payload.begin() + start, payload.begin() + end);
}

// ============================================================
//  批次接收 (多线程流水线)
// ============================================================

bool SensorCaptureNode::receiveBatch(int batch_index)
{
    // 计算批次文件夹
    char buf[64];
    snprintf(buf, sizeof(buf), "batch_%04d", batch_index);
    std::string batch_folder = save_dir_ + "/" + buf;

    // 清空队列和标志
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!data_queue_.empty()) data_queue_.pop();
    }
    batch_done_ = false;
    batch_error_ = false;

    // 启动接收线程和处理线程
    auto fail = [this](const std::exception& error) {
        RCLCPP_ERROR(get_logger(), "Capture worker exception: %s", error.what());
        batch_error_ = true;
        batch_done_ = true;
        if (sock_fd_ >= 0) shutdown(sock_fd_, SHUT_RDWR);
        queue_cv_.notify_all();
    };
    receive_thread_ = std::thread([this, batch_index, batch_folder, fail] {
        try { receiveThread(batch_index, batch_folder); } catch (const std::exception& error) { fail(error); }
    });
    process_thread_ = std::thread([this, batch_folder, fail] {
        try { processThread(batch_folder); } catch (const std::exception& error) { fail(error); }
    });

    // 等待两个线程完成
    receive_thread_.join();
    process_thread_.join();

    bool success = batch_done_.load() && !batch_error_.load();
    if (success) {
        RCLCPP_INFO(this->get_logger(), "批次 %d 处理完成", batch_index);
    } else if (!success && !running_) {
        RCLCPP_ERROR(this->get_logger(), "批次 %d 被中断", batch_index);
    }
    return success;
}

// ------------------------------------------------------------
//  接收线程: TCP 读取 → 解析 → 入队
// ------------------------------------------------------------
void SensorCaptureNode::receiveThread(int batch_index,
                                      const std::string & batch_folder)
{
    auto start_time = std::chrono::steady_clock::now();
    std::set<uint32_t> received_sequences;
    std::optional<uint32_t> active_batch;
    uint32_t expected_count = 0;
    size_t received_bytes = 0;
    bool received_meta_end = false;
    const bool strict_batch = timestamp_mode_ == "meta_relative";
    auto finish_batch = [&]() {
        if (strict_batch && received_meta_end && expected_count > 0 &&
            received_sequences.size() == expected_count) {
            last_completed_batch_id_ = active_batch;
            batch_done_ = true;
            queue_cv_.notify_all();
            return true;
        }
        return false;
    };

    while (running_ && !batch_done_) {
        // 超时检查
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed > timeout_) {
            RCLCPP_ERROR(this->get_logger(), "批次 %d 接收超时", batch_index);
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            return;
        }

        uint32_t cmd;
        std::vector<uint8_t> payload;
        if (!recvPacket(cmd, payload)) {
            RCLCPP_ERROR(this->get_logger(), "批次 %d 连接中断", batch_index);
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            return;
        }

        received_bytes += payload.size();
        if (received_bytes > 2 * publication_batch_bytes_) {
            RCLCPP_ERROR(get_logger(), "Wire batch exceeded memory budget");
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            return;
        }
        if (cmd != CMD_DATA) continue;
        if (payload.size() < 4) continue;

        uint32_t head = readU32(payload.data());
        if (head == HEAD_FEEDBACK) continue;
        if (head != HEAD_SENSOR_DATA) continue;

        SensorFileHeader header;
        if (!parseFileHeader(payload, header)) continue;
        if (payload.size() < 156 || header.file_size > payload.size() - 156 ||
            (header.file_name == "meta.json" && header.file_size > 1024 * 1024)) {
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            return;
        }

        if (strict_batch) {
            if (last_completed_batch_id_ && header.batch_id == *last_completed_batch_id_) continue;
            if (!active_batch) active_batch = header.batch_id;
            if (header.batch_id != *active_batch || header.total_count == 0 || header.total_count > 128 ||
                (expected_count && expected_count != header.total_count)) {
                RCLCPP_ERROR(this->get_logger(), "Inconsistent UE batch boundary; discard and reconnect");
                batch_error_ = true;
                batch_done_ = true;
                queue_cv_.notify_all();
                return;
            }
            expected_count = header.total_count;
        }

        // ---- meta.json: 提取原始数据入队 ----
        if (header.file_name == "meta.json") {
            size_t start_off = findPayloadStart(payload, "meta.json");
            std::vector<uint8_t> meta_data(
                payload.begin() + start_off,
                payload.begin() + std::min(start_off + header.file_size, payload.size()));

            BatchItem item;
            item.header = header;
            item.batch_folder = batch_folder;
            item.batch_index = batch_index;
            item.is_meta = true;
            item.meta_raw = std::move(meta_data);

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                data_queue_.push(std::move(item));
            }
            queue_cv_.notify_one();

            RCLCPP_INFO(this->get_logger(), "收到 meta.json (size=%u bytes)",
                        header.file_size);

            if (strict_batch) {
                received_meta_end = (header.flags & 0x01) != 0;
                if (finish_batch()) return;
                continue;
            }
            if (header.flags & 0x01) {
                batch_done_ = true;
                queue_cv_.notify_one();
                return;
            }
            continue;
        }

        // ---- 传感器数据: 去重 → 提取 → 入队 ----
        if (received_sequences.count(header.sequence)) continue;
        received_sequences.insert(header.sequence);

        // 提取文件数据
        std::vector<uint8_t> file_data;
        bool is_npy = header.file_name.size() >= 4 &&
                      header.file_name.substr(header.file_name.size() - 4) == ".npy";

        if (is_npy) {
            size_t npy_pos = findInData(payload, NPY_MAGIC);
            if (npy_pos != std::string::npos) {
                size_t end = std::min(npy_pos + header.file_size, payload.size());
                file_data.assign(payload.begin() + npy_pos, payload.begin() + end);
            } else {
                size_t end = std::min<size_t>(156 + header.file_size, payload.size());
                file_data.assign(payload.begin() + 156, payload.begin() + end);
            }
        } else {
            file_data = extractFileData(payload, header);
        }

        // PNG 校验
        bool is_png = header.file_name.size() >= 4 &&
                      header.file_name.substr(header.file_name.size() - 4) == ".png";
        if (is_png && file_data.size() >= PNG_MAGIC.size() &&
            !std::equal(PNG_MAGIC.begin(), PNG_MAGIC.end(), file_data.begin()))
        {
            if (payload.size() >= 164 &&
                std::equal(PNG_MAGIC.begin(), PNG_MAGIC.end(), payload.begin() + 156))
            {
                size_t end = std::min<size_t>(156 + header.file_size, payload.size());
                file_data.assign(payload.begin() + 156, payload.begin() + end);
            }
        }

        // 入队
        {
            BatchItem item;
            item.header = header;
            item.file_data = std::move(file_data);
            item.batch_folder = batch_folder;
            item.batch_index = batch_index;
            item.is_meta = false;

            std::lock_guard<std::mutex> lock(queue_mutex_);
            data_queue_.push(std::move(item));
        }
        queue_cv_.notify_one();

        if (strict_batch) {
            if (finish_batch()) return;
            continue;
        }
        if (header.flags & 0x01) {
            batch_done_ = true;
            queue_cv_.notify_one();
            return;
        }
    }

    // 循环退出但未被标记完成 → 超时或中断
    if (!batch_done_) {
        batch_error_ = true;
        batch_done_ = true;
        queue_cv_.notify_all();
    }
}

// ------------------------------------------------------------
//  处理线程: 出队 → 保存 → 批次完成后统一发布
//  阶段 1: 接收期间，出队并保存磁盘，缓存到 batch_items_
//  阶段 2: 接收完成后，遍历 batch_items_ 统一发布
// ------------------------------------------------------------
void SensorCaptureNode::processThread(const std::string & batch_folder)
{
    // 如果需要保存数据，提前创建目录
    if (save_data_) {
        fs::create_directories(batch_folder);
    }

    // 清空批次缓存和计数器
    batch_items_.clear();
    batch_recv_counts_.clear();

    int processed = 0;
    double decode_seconds = 0.0;
    size_t retained_bytes = 0;
    std::string batch_metadata;

    // ---- 阶段 1: 接收期间，保存磁盘 + 缓存数据 ----
    while (true) {
        BatchItem item;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // Completion/running flags are atomic and can change without
            // queue_mutex_; periodically recheck so an unlucky notification
            // race cannot leave shutdown or a completed batch stuck forever.
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !data_queue_.empty() ||
                       batch_done_.load() || !running_;
            });

            if (!data_queue_.empty()) {
                item = std::move(data_queue_.front());
                data_queue_.pop();
            } else if (batch_done_.load() || !running_) {
                // 队列为空且批次完成 → 进入阶段 2
                break;
            } else {
                continue;
            }
        }

        try {
            if (item.is_meta) {
                batch_metadata.assign(item.meta_raw.begin(), item.meta_raw.end());
                // ---- 保存 meta.json ----
                if (save_data_) {
                    std::string file_path = item.batch_folder + "/meta.json";
                    std::ofstream ofs(file_path, std::ios::binary);
                    ofs.write(reinterpret_cast<const char *>(item.meta_raw.data()),
                              item.meta_raw.size());
                    ofs.close();
                }
                // Original meta is published by the complete-batch publication worker.
            } else {
                // ---- 保存传感器数据到磁盘 ----
                if (save_data_) {
                    uint8_t stype_save = item.header.sensor_type;
                    if (stype_save == SENSOR_TOF || stype_save == SENSOR_CPU_LIDAR) {
                        // TOF/激光雷达: 保存为 PCD 格式 (不保存原始 npy/二进制数据)
                        std::string base = item.header.file_name;
                        size_t dot_pos = base.rfind('.');
                        if (dot_pos != std::string::npos) base = base.substr(0, dot_pos);

                        sensor_msgs::msg::PointCloud2 cloud;
                        bool built = (stype_save == SENSOR_TOF)
                            ? buildTofPointCloud(item.file_data,
                                                 getTofConfig(item.header.camera_id), cloud)
                            : buildLidarPointCloud(item.file_data, cloud);
                        if (built && !savePcdFile(item.batch_folder + "/" + base + ".pcd", cloud)) {
                            RCLCPP_WARN(this->get_logger(), "PCD 保存失败: %s", base.c_str());
                        }

                        // TOF 额外保存深度图和强度图 (mono16 PNG)
                        if (stype_save == SENSOR_TOF) {
                            TofCamConfig tof_cfg = getTofConfig(item.header.camera_id);
                            cv::Mat depth_img;
                            if (buildTofDepthImage(item.file_data, tof_cfg, depth_img)) {
                                // 四个 TOF 的深度图保存前一律归一化 (不受 normalize_save 控制)
                                normalizeImage16U(depth_img);
                                cv::imwrite(item.batch_folder + "/" + base + "_depth.png", depth_img);
                            }
                            // 强度图仅 tof_0/tof_1 有 (双平面布局), 保留保存功能, 不再发布
                            if (tof_cfg.has_intensity) {
                                cv::Mat intensity_img;
                                if (buildTofIntensityImage(item.file_data, tof_cfg.width, tof_cfg.height, intensity_img)) {
                                    if (normalize_save_) normalizeImage16U(intensity_img);
                                    cv::imwrite(item.batch_folder + "/" + base + "_intensity.png", intensity_img);
                                }
                            }
                        }
                    } else {
                        std::string file_path = item.batch_folder + "/" + item.header.file_name;
                        std::ofstream ofs(file_path, std::ios::binary);
                        ofs.write(reinterpret_cast<const char *>(item.file_data.data()),
                                  item.file_data.size());
                        ofs.close();
                    }
                }

                // 缓存到批次，等接收完成后统一发布
                uint8_t stype = item.header.sensor_type;
                batch_recv_counts_[stype]++;
                total_recv_counts_[stype]++;
                const std::string key = makePublisherKey(stype, item.header.camera_id);
                const bool selected = publishers_.count(key) || tof_depth_publishers_.count(key);
                if (selected) {
                    if (stype == SENSOR_RGB && publishers_.count(key)) {
                        // Decode only selected images while receiveThread reads
                        // the remaining files. Publication still waits for the
                        // complete batch and its validated common meta stamp.
                        // Check PNG dimensions before allowing the decoder to allocate memory.
                        if (item.file_data.size() < 24) throw std::runtime_error("Truncated RGB PNG");
                        auto png_u32 = [&](size_t at) {
                            uint32_t value = 0;
                            for (size_t i = at; i < at + 4; ++i) value = (value << 8) | item.file_data[i];
                            return value;
                        };
                        const uint64_t pixels = uint64_t(png_u32(16)) * png_u32(20);
                        if (!pixels || pixels > publication_batch_bytes_ / 8)
                            throw std::runtime_error("Decoded RGB exceeds memory budget");
                        const auto decoding = std::chrono::steady_clock::now();
                        item.prepared_image = cv::imdecode(cv::Mat(item.file_data, false), cv::IMREAD_UNCHANGED);
                        decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - decoding).count();
                        if (item.prepared_image.empty()) throw std::runtime_error("RGB PNG decode failed");
                        if (!item.prepared_image.empty()) std::vector<uint8_t>().swap(item.file_data);
                    }
                    retained_bytes += item.file_data.capacity() + item.prepared_image.total() * item.prepared_image.elemSize();
                    if (retained_bytes > publication_batch_bytes_) throw std::runtime_error("Decoded batch exceeds memory budget");
                    batch_items_.push_back(std::move(item));
                } // Unselected payloads are released immediately after optional saving.
            }
            ++processed;
        } catch (const std::exception & e) {
            RCLCPP_ERROR(this->get_logger(), "处理数据异常: %s", e.what());
            // Never allow a failed left image to masquerade as a complete batch.
            batch_error_ = true;
            batch_done_ = true;
            queue_cv_.notify_all();
            if (sock_fd_ >= 0) shutdown(sock_fd_, SHUT_RDWR);
        }
    }

    // ---- 阶段 2: 批次接收完成，统一发布所有传感器数据 ----
    if (!batch_items_.empty()) {
        if (batch_error_ || !running_) {
            batch_items_.clear();
            return;  // Never publish a partial or interrupted sensor batch.
        }
        if (timestamp_mode_ == "meta_relative") {
            try {
                const int64_t stamp = batch_clock_.map(metaTimestampNs(batch_metadata),
                    batch_request_stamp_.nanoseconds(), this->now().nanoseconds());
                batch_publish_stamp_ = rclcpp::Time(stamp, this->get_clock()->get_clock_type());
                RCLCPP_INFO(this->get_logger(), "UE batch %u: aligned batch stamp %.9f, raw header %u",
                    batch_items_.front().header.batch_id, batch_publish_stamp_.seconds(),
                    batch_items_.front().header.timestamp);
            } catch (const std::exception & error) {
                RCLCPP_ERROR(this->get_logger(), "Dropping sensor batch: %s", error.what());
                batch_items_.clear();
                return;
            }
        }
        enqueuePublication(std::move(batch_metadata), processed, decode_seconds);
    }

    batch_items_.clear();
}

// ============================================================
//  数据发布
//  publishSensorData: 根据传感器类型分发到具体发布函数
//  publishImage:      PNG 解码 → sensor_msgs/Image
//  publishPointCloud: 原始字节 → sensor_msgs/PointCloud2
//  publishEventArray: 13字节/事件 → sjcam_msgs/EventArray
//  publishByteArray:  原始字节 → std_msgs/ByteMultiArray
// ============================================================

/** @brief 获取传感器名称 (成员函数版本) */
std::string SensorCaptureNode::getSensorName(uint8_t sensor_type)
{
    return getSensorNameStatic(sensor_type);
}

/** @brief 生成发布者 key: "sensor_name_camera_id" (如 "rgb_0", "event_1") */
std::string SensorCaptureNode::makePublisherKey(uint8_t sensor_type, uint8_t camera_id)
{
    return getSensorNameStatic(sensor_type) + "_" + std::to_string(camera_id);
}

/**
 * @brief 根据传感器类型分发数据到对应的发布函数
 *
 * 时间戳计算: stamp = start_time_ + header.timestamp × 1μs
 * start_time_ 为节点构造时记录的基准时间，header.timestamp 为 UE 返回的时间增量
 */
void SensorCaptureNode::publishSensorData(
    const SensorFileHeader & header,
    const std::vector<uint8_t> & file_data,
    const std::string & /*file_name*/, const rclcpp::Time & batch_stamp, const cv::Mat & prepared)
{
    uint8_t stype = header.sensor_type;
    uint8_t cam_id = header.camera_id;
    std::string base_key = makePublisherKey(stype, cam_id);

    // 获取发布的 topic 名称
    std::string topic_name;
    auto pub_it = publishers_.find(base_key);
    if (pub_it != publishers_.end()) {
        topic_name = pub_it->second.publisher->get_topic_name();
    }
    if (pub_it == publishers_.end() && tof_depth_publishers_.find(base_key) == tof_depth_publishers_.end()) {
        return;  // Explicitly unconfigured sensors are not decoded or published.
    }

    // TOF 额外显示深度图 topic (强度图已取消发布)
    if (stype == SENSOR_TOF) {
        auto dit = tof_depth_publishers_.find(base_key);
        if (dit != tof_depth_publishers_.end())
            topic_name += std::string(" + ") + dit->second->get_topic_name();
    }

    RCLCPP_DEBUG(this->get_logger(),
        "发布: type=0x%02X cam_id=%u key=\"%s\" -> topic=\"%s\" (%zu bytes)",
        stype, cam_id, base_key.c_str(), topic_name.c_str(), file_data.size());

    // 时间戳 = 软件启动基准时间 + UE 返回的传感器时间增量（header.timestamp 单位: 微秒）
    const rclcpp::Time stamp = timestamp_mode_ == "meta_relative" ? batch_stamp :
        start_time_ + rclcpp::Duration::from_seconds(static_cast<double>(header.timestamp) * 1e-6);

    try {
        if (stype == SENSOR_RGB) {
            publishImage(base_key, file_data, stamp, prepared);
        } else if (stype == SENSOR_TOF) {
            TofCamConfig cfg = getTofConfig(cam_id);
            if (pub_it != publishers_.end()) publishTOFPointCloud(base_key, file_data, cfg, stamp);
            publishTOFDepthImage(base_key, file_data, cfg, stamp);
            // 强度图不再发布: tof_0/1 仅保存磁盘, tof_2/3 无强度数据
        } else if (stype == SENSOR_EVENT) {
            publishEventArray(base_key, file_data, stamp);
        } else if (stype == SENSOR_CPU_LIDAR) {
            publishPointCloud(base_key, file_data, stamp);
        } else {
            RCLCPP_WARN(this->get_logger(),
                "未知传感器类型: 0x%02X cam_id=%u, key=%s — 数据丢弃",
                stype, cam_id, base_key.c_str());
        }
    } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "发布 %s 失败: %s", base_key.c_str(), e.what());
        throw; // Complete-batch worker records failure; do not report success.
    }
}

/** @brief PNG 解码发布为 sensor_msgs/Image */
void SensorCaptureNode::publishImage(
    const std::string & key,
    const std::vector<uint8_t> & data, const rclcpp::Time & stamp, const cv::Mat & prepared)
{
    auto it = publishers_.find(key);
    if (it == publishers_.end()) {
        // RCLCPP_WARN(this->get_logger(),
        //     "图像发布: 未找到 key=\"%s\" 的发布者 (数据 %zu bytes) — 检查 output_topics 配置",
        //     key.c_str(), data.size());
        return;
    }

    // PNG → cv::Mat → sensor_msgs/Image
    cv::Mat decoded = prepared.empty() ? cv::imdecode(
        cv::Mat(data, false), cv::IMREAD_UNCHANGED) : prepared;
    if (decoded.empty()) {
        RCLCPP_WARN(this->get_logger(), "图像解码失败: %s", key.c_str());
        return;
    }

    // 转换为 ROS2 Image 消息
    std::string encoding;
    switch (decoded.channels()) {
        case 1: encoding = "mono8"; break;
        case 3: encoding = "bgr8"; break;
        case 4: encoding = "bgra8"; break;
        default: encoding = "bgr8"; break;
    }

    auto msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), encoding, decoded).toImageMsg();
    msg->header.stamp = stamp;
    msg->header.frame_id = key;

    auto pub = std::dynamic_pointer_cast<
        rclcpp::Publisher<sensor_msgs::msg::Image>>(it->second.publisher);
    if (pub) pub->publish(*msg);

    // 发布缩略图
    publishThumbnail(key, decoded, stamp);
}

/** @brief 将图像缩放到 thumb_width_ x thumb_height_ 后发布轻量版 Image */
void SensorCaptureNode::publishThumbnail(
    const std::string & key,
    const cv::Mat & decoded, const rclcpp::Time & stamp)
{
    if (!publish_thumbnail_) return;

    auto it = thumb_publishers_.find(key);
    if (it == thumb_publishers_.end()) return;

    // 缩放到目标尺寸
    cv::Mat thumb;
    cv::resize(decoded, thumb, cv::Size(thumb_width_, thumb_height_));

    // 确定编码格式
    std::string encoding;
    switch (thumb.channels()) {
        case 1: encoding = "mono8"; break;
        case 3: encoding = "bgr8"; break;
        case 4: encoding = "bgra8"; break;
        default: encoding = "bgr8"; break;
    }

    auto msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), encoding, thumb).toImageMsg();
    msg->header.stamp = stamp;
    msg->header.frame_id = key;
    it->second->publish(*msg);
}

/**
 * @brief 将 PointCloud2 保存为 PCD (binary) 文件
 *
 * PCD v0.7 格式: 文本头 + 二进制点数据。
 * 头部字段描述根据 cloud.fields 生成 (当前均为 float32)。
 * @param path 保存路径 (.pcd)
 * @param cloud 点云消息 (使用 fields/width/height/data)
 * @return 保存成功返回 true
 */
bool SensorCaptureNode::savePcdFile(
    const std::string & path, const sensor_msgs::msg::PointCloud2 & cloud)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
    ofs << "VERSION 0.7\n";
    ofs << "FIELDS";
    for (const auto & f : cloud.fields) ofs << ' ' << f.name;
    ofs << "\nSIZE";
    for (size_t i = 0; i < cloud.fields.size(); ++i) ofs << " 4";   // float32
    ofs << "\nTYPE";
    for (size_t i = 0; i < cloud.fields.size(); ++i) ofs << " F";   // float
    ofs << "\nCOUNT";
    for (const auto & f : cloud.fields) ofs << ' ' << f.count;
    ofs << "\nWIDTH " << cloud.width
        << "\nHEIGHT " << cloud.height
        << "\nVIEWPOINT 0 0 0 1 0 0 0"
        << "\nPOINTS " << static_cast<size_t>(cloud.width) * cloud.height
        << "\nDATA binary\n";

    ofs.write(reinterpret_cast<const char *>(cloud.data.data()),
              static_cast<std::streamsize>(cloud.data.size()));
    ofs.close();
    return !ofs.fail();
}

/**
 * @brief 获取 TOF 相机配置 (按 camera_id 区分)
 *   camera_id 0/1 → 640×480, FOV 67°×51°, [深度平面][强度平面]
 *   camera_id 2/3 → 2448×2048, FOV 55.7°×47.7°, 仅深度平面 (无强度数据)
 */
TofCamConfig SensorCaptureNode::getTofConfig(uint8_t camera_id) const
{
    if (camera_id >= 2) {
        // tof_2/tof_3: 高分辨率深度相机, 无强度平面
        return {2448, 2048, tof2_fov_x_deg_, tof2_fov_y_deg_, false};
    }
    // tof_0/tof_1: 标准 ToF, 深度 + 强度双平面
    return {640, 480, tof_fov_x_deg_, tof_fov_y_deg_, true};
}

/**
 * @brief 解析 TOF 数据构建 PointCloud2 (基于视场角计算 3D 坐标)
 *
 * 数据格式: 先全部深度，再全部强度 (非交错)
 *   [uint16 depth × 640×480] [uint16 intensity × 640×480]
 *   小端序，行优先排列
 *
 * 分辨率: 640×480 (TOF 传感器固定分辨率)
 * 视场角: tof_fov_x_deg_ (水平) × tof_fov_y_deg_ (垂直)
 *
 * 针孔相机模型 (径向距离→三轴投影):
 *   fx = width  / (2 × tan(FOV_x / 2))
 *   fy = height / (2 × tan(FOV_y / 2))
 *   dx = (col - cx) / fx
 *   dy = (row - cy) / fy
 *   cos_θ = 1 / √(1 + dx² + dy²)
 *   r = depth × tof_depth_scale_   (原始深度 cm → 米, 默认 0.01; r > tof_max_range_ 的点剔除)
 *   Z = r × cos_θ   (径向距离投影到光轴)
 *   X = Z × dx
 *   Y = Z × dy
 *
 * 点云字段 (均为 float32, 每点 16 字节): x, y, z, intensity
 * header (stamp/frame_id) 由调用方设置, 便于发布与保存复用。
 * @return 数据充足返回 true, 否则 false (msg 内容无效)
 */
bool SensorCaptureNode::buildTofPointCloud(
    const std::vector<uint8_t> & data,
    const TofCamConfig & cfg, sensor_msgs::msg::PointCloud2 & msg)
{
    const size_t WIDTH = cfg.width;
    const size_t HEIGHT = cfg.height;
    size_t total_pixels = WIDTH * HEIGHT;
    size_t plane_size = total_pixels * 2;  // 每个通道: width×height×2 bytes (uint16)

    // 数据布局: 含强度时 [深度平面][强度平面], 否则仅 [深度平面]
    size_t required = cfg.has_intensity ? plane_size * 2 : plane_size;
    if (data.size() < required) {
        return false;
    }

    const uint8_t * depth_ptr = data.data();
    const uint8_t * intensity_ptr =
        cfg.has_intensity ? (data.data() + plane_size) : nullptr;

    // 根据视场角计算焦距 (像素单位)
    double fov_x_rad = cfg.fov_x_deg * M_PI / 180.0;
    double fov_y_rad = cfg.fov_y_deg * M_PI / 180.0;
    double fx = WIDTH  / (2.0 * std::tan(fov_x_rad / 2.0));
    double fy = HEIGHT / (2.0 * std::tan(fov_y_rad / 2.0));
    double cx = WIDTH / 2.0;
    double cy = HEIGHT / 2.0;

    constexpr uint32_t FLOAT32 = sensor_msgs::msg::PointField::FLOAT32;
    sensor_msgs::msg::PointField px, py, pz, pi;
    px.name = "x";          px.offset = 0;  px.datatype = FLOAT32; px.count = 1;
    py.name = "y";          py.offset = 4;  py.datatype = FLOAT32; py.count = 1;
    pz.name = "z";          pz.offset = 8;  pz.datatype = FLOAT32; pz.count = 1;
    pi.name = "intensity";  pi.offset = 12; pi.datatype = FLOAT32; pi.count = 1;
    msg.fields = {px, py, pz, pi};

    msg.point_step = 16;
    msg.is_bigendian = false;
    msg.is_dense = true;
    msg.data.resize(total_pixels * msg.point_step);  // 先按最大容量分配，剔除无效点后收缩

    // 解析顺序数据并投影到 3D (径向距离超过 tof_max_range_ 的点无效, 剔除)
    size_t valid_count = 0;
    for (size_t row = 0; row < HEIGHT; ++row) {
        for (size_t col = 0; col < WIDTH; ++col) {
            size_t pixel_idx = row * WIDTH + col;
            size_t byte_offset = pixel_idx * 2;

            // 无强度平面的相机 (tof_2/tof_3) intensity 填 0
            uint16_t intensity = 0;
            if (intensity_ptr != nullptr) {
                intensity = static_cast<uint16_t>(intensity_ptr[byte_offset])
                          | (static_cast<uint16_t>(intensity_ptr[byte_offset + 1]) << 8);
            }
            uint16_t depth = static_cast<uint16_t>(depth_ptr[byte_offset])
                           | (static_cast<uint16_t>(depth_ptr[byte_offset + 1]) << 8);

            // 径向距离 (米, 原始深度 × 换算系数), 超过有效量程的点剔除
            double r = depth * tof_depth_scale_;
            if (r > tof_max_range_) continue;

            // 径向距离 → 三轴投影
            double dx = (static_cast<double>(col) - cx) / fx;
            double dy = (static_cast<double>(row) - cy) / fy;
            double cos_theta = 1.0 / std::sqrt(1.0 + dx * dx + dy * dy);
            float z = static_cast<float>(r * cos_theta);
            float x = static_cast<float>(z * dx);
            float y = static_cast<float>(z * dy);
            float intens = static_cast<float>(intensity);

            size_t dst = valid_count * msg.point_step;
            std::memcpy(&msg.data[dst + 0],  &x, 4);
            std::memcpy(&msg.data[dst + 4],  &y, 4);
            std::memcpy(&msg.data[dst + 8],  &z, 4);
            std::memcpy(&msg.data[dst + 12], &intens, 4);
            valid_count++;
        }
    }

    // 剔除无效点后收缩为无组织点云 (height=1)
    msg.data.resize(valid_count * msg.point_step);
    msg.width = valid_count;
    msg.height = 1;
    msg.row_step = msg.point_step * valid_count;
    return true;
}

/** @brief 解析 TOF 数据并发布为 PointCloud2 (按相机配置的分辨率/视场角计算 3D 坐标) */
void SensorCaptureNode::publishTOFPointCloud(
    const std::string & key,
    const std::vector<uint8_t> & data,
    const TofCamConfig & cfg, const rclcpp::Time & stamp)
{
    auto it = publishers_.find(key);
    if (it == publishers_.end()) {
        RCLCPP_WARN(this->get_logger(),
            "TOF 点云: 未找到 key=\"%s\" 的发布者 — 检查 output_topics 配置", key.c_str());
        return;
    }

    sensor_msgs::msg::PointCloud2 msg;
    if (!buildTofPointCloud(data, cfg, msg)) {
        size_t plane_size = cfg.width * cfg.height * 2;
        size_t required = cfg.has_intensity ? plane_size * 2 : plane_size;
        RCLCPP_WARN(this->get_logger(),
            "TOF 点云: 数据不足 (%zu bytes, 需要 %zu bytes)",
            data.size(), required);
        return;
    }
    msg.header.stamp = stamp;
    msg.header.frame_id = "sensors";   // 点云统一坐标系

    auto pub = std::dynamic_pointer_cast<
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>>(it->second.publisher);
    if (pub) {
        pub->publish(msg);
        RCLCPP_INFO(this->get_logger(),
            "TOF 点云已发布: key=%s, %ux%u, %zu points",
            key.c_str(), msg.width, msg.height,
            static_cast<size_t>(msg.width) * msg.height);
    } else {
        RCLCPP_ERROR(this->get_logger(),
            "TOF 点云: dynamic_pointer_cast 失败, key=%s", key.c_str());
    }
}

/**
 * @brief 从 TOF 原始数据构建深度图 (CV_16UC1)
 * 深度平面始终在偏移 0 处 (小端序); 最小数据量由是否含强度平面决定
 * @return 数据充足返回 true, 否则 false (depth_img 内容无效)
 */
bool SensorCaptureNode::buildTofDepthImage(
    const std::vector<uint8_t> & data,
    const TofCamConfig & cfg, cv::Mat & depth_img)
{
    size_t plane_size = cfg.width * cfg.height * 2;
    size_t required = cfg.has_intensity ? plane_size * 2 : plane_size;
    if (data.size() < required) return false;

    // 深度数据在第一个平面 (与原始字节序一致, 直接拷贝即为小端布局)
    depth_img.create(static_cast<int>(cfg.height), static_cast<int>(cfg.width), CV_16UC1);
    std::memcpy(depth_img.data, data.data(), plane_size);
    return true;
}

/**
 * @brief 从 TOF 原始数据构建强度图 (CV_16UC1)
 * 数据布局: [深度平面] [强度平面]，从强度平面提取 (小端序)
 * @return 数据充足返回 true, 否则 false (intensity_img 内容无效)
 */
bool SensorCaptureNode::buildTofIntensityImage(
    const std::vector<uint8_t> & data,
    size_t width, size_t height, cv::Mat & intensity_img)
{
    size_t plane_size = width * height * 2;
    if (data.size() < plane_size * 2) return false;

    // 强度数据在第二个平面 (与原始字节序一致, 直接拷贝即为小端布局)
    intensity_img.create(static_cast<int>(height), static_cast<int>(width), CV_16UC1);
    std::memcpy(intensity_img.data, data.data() + plane_size, plane_size);
    return true;
}

/**
 * @brief 发布 TOF 深度图 (mono16 Image, 尺寸由相机配置决定)
 * 深度平面在偏移 0 处 (含强度平面时位于其前)
 */
void SensorCaptureNode::publishTOFDepthImage(
    const std::string & key,
    const std::vector<uint8_t> & data,
    const TofCamConfig & cfg, const rclcpp::Time & stamp)
{
    auto it = tof_depth_publishers_.find(key);
    if (it == tof_depth_publishers_.end()) return;

    cv::Mat depth_img;
    if (!buildTofDepthImage(data, cfg, depth_img)) return;

    auto msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), "mono16", depth_img).toImageMsg();
    msg->header.stamp = stamp;
    msg->header.frame_id = key;
    it->second->publish(*msg);
}

// 注: publishTOFIntensityImage 已移除 — TOF 强度图不再发布
// (tof_0/tof_1 强度图仅保存磁盘, tof_2/tof_3 无强度数据)

/**
 * @brief 解析 CPU LiDAR 数据构建 PointCloud2
 *
 * 数据格式 (与 verify_data.py 一致):
 *   [PointCount: int32 (4B)] + [x,y,z × float32 ...]
 *   小端序
 * 输出坐标单位: 米 (原始值 cm × lidar_scale_, 默认 0.01)
 * 坐标变换: Y 轴取反 (UE 坐标系 → ROS 坐标系)
 * header (stamp/frame_id) 由调用方设置, 便于发布与保存复用。
 * @return 数据充足返回 true, 否则 false (msg 内容无效)
 */
bool SensorCaptureNode::buildLidarPointCloud(
    const std::vector<uint8_t> & data, sensor_msgs::msg::PointCloud2 & msg)
{
    // 至少需要 4 字节 count 头 + 12 字节一个点
    if (data.size() < 4) {
        return false;
    }

    // 读取 PointCount (int32 LE)
    int32_t count = 0;
    std::memcpy(&count, data.data(), 4);

    constexpr size_t POINT_STEP = 12;  // x,y,z (float32 × 3)
    size_t available = (data.size() - 4) / POINT_STEP;
    size_t n_points = std::min(static_cast<size_t>(count), available);

    sensor_msgs::msg::PointField fx, fy, fz;
    fx.name = "x"; fx.offset = 0; fx.datatype = sensor_msgs::msg::PointField::FLOAT32; fx.count = 1;
    fy.name = "y"; fy.offset = 4; fy.datatype = sensor_msgs::msg::PointField::FLOAT32; fy.count = 1;
    fz.name = "z"; fz.offset = 8; fz.datatype = sensor_msgs::msg::PointField::FLOAT32; fz.count = 1;
    msg.fields = {fx, fy, fz};

    msg.point_step = POINT_STEP;
    msg.row_step = POINT_STEP * n_points;
    msg.height = 1;
    msg.width = n_points;
    msg.is_bigendian = false;
    msg.is_dense = true;

    // 跳过 4 字节 count 头，逐点缩放原始坐标 (mm → 米, 由 lidar_scale_ 控制)
    // Y 轴 (每点第 2 个分量, i % 3 == 1) 取反, 对齐 ROS 坐标系
    msg.data.resize(n_points * POINT_STEP);
    const float scale = static_cast<float>(lidar_scale_);
    for (size_t i = 0; i < n_points * 3; ++i) {
        float v;
        std::memcpy(&v, data.data() + 4 + i * sizeof(float), sizeof(float));
        v *= scale;
        if (i % 3 == 1) {
            v = -v;   // Y 轴取反
        }
        std::memcpy(msg.data.data() + i * sizeof(float), &v, sizeof(float));
    }
    return true;
}

/**
 * @brief 解析 CPU LiDAR 数据并发布为 PointCloud2
 */
void SensorCaptureNode::publishPointCloud(
    const std::string & key,
    const std::vector<uint8_t> & data, const rclcpp::Time & stamp)
{
    auto it = publishers_.find(key);
    if (it == publishers_.end()) {
        RCLCPP_WARN(this->get_logger(),
            "点云发布: 未找到 key=\"%s\" 的发布者 (数据 %zu bytes) — 检查 output_topics 配置",
            key.c_str(), data.size());
        return;
    }

    sensor_msgs::msg::PointCloud2 msg;
    const auto cloud_build_started = std::chrono::steady_clock::now();
    if (!buildLidarPointCloud(data, msg)) {
        RCLCPP_WARN(this->get_logger(), "LiDAR 数据不足 4 字节 (%zu)", data.size());
        return;
    }
    msg.header.stamp = stamp;
    msg.header.frame_id = "sensors";   // 点云统一坐标系

    auto pub = std::dynamic_pointer_cast<
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>>(it->second.publisher);
    if (pub) {
        const auto cloud_write_started = std::chrono::steady_clock::now();
        pub->publish(msg);
        RCLCPP_INFO(this->get_logger(),
            "LiDAR 点云已发布: key=%s, %u points build=%.6fs dds_write=%.6fs", key.c_str(), msg.width,
            std::chrono::duration<double>(cloud_write_started - cloud_build_started).count(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - cloud_write_started).count());
    } else {
        RCLCPP_ERROR(this->get_logger(),
            "LiDAR 点云: dynamic_pointer_cast 失败, key=%s", key.c_str());
    }
}

/** @brief 原始字节发布为 std_msgs/ByteMultiArray (无 header，无法设置 stamp) */
void SensorCaptureNode::publishByteArray(
    const std::string & key,
    const std::vector<uint8_t> & data, const rclcpp::Time & /*stamp*/)
{
    auto it = publishers_.find(key);
    if (it == publishers_.end()) return;

    auto msg = std_msgs::msg::ByteMultiArray();
    std_msgs::msg::MultiArrayDimension dim;
    dim.label = "raw_data";
    dim.size = data.size();
    dim.stride = 1;
    msg.layout.dim.push_back(dim);
    msg.data = data;

    auto pub = std::dynamic_pointer_cast<
        rclcpp::Publisher<std_msgs::msg::ByteMultiArray>>(it->second.publisher);
    if (pub) pub->publish(msg);
}

void SensorCaptureNode::publishCameraInfo(
    const std::string & key, const rclcpp::Time & stamp)
{
    auto it = publishers_.find(key);
    if (it == publishers_.end()) return;

    auto msg = sensor_msgs::msg::CameraInfo();
    msg.header.stamp = stamp;
    msg.header.frame_id = key;

    auto pub = std::dynamic_pointer_cast<
        rclcpp::Publisher<sensor_msgs::msg::CameraInfo>>(it->second.publisher);
    if (pub) pub->publish(msg);
}

/**
 * @brief 解析事件相机数据并发布为 sjcam_msgs/EventArray
 *
 * 协议数据格式 (与 verify_data.py 一致):
 *   [EventCount: int32 (4B)] + [t(float64) + x(int32) + y(int32) + p(int32)] × N
 *   每事件 20 字节，小端序
 *
 * 转换为 sjcam_msgs/Event:
 *   uint16 x, uint16 y, builtin_interfaces/Time ts, bool polarity
 */
void SensorCaptureNode::publishEventArray(
    const std::string & key,
    const std::vector<uint8_t> & data, const rclcpp::Time & stamp)
{
    auto it = publishers_.find(key);
    if (it == publishers_.end()) {
        RCLCPP_WARN(this->get_logger(),
            "事件发布: 未找到 key=\"%s\" 的发布者 (数据 %zu bytes) — 检查 output_topics 配置",
            key.c_str(), data.size());
        return;
    }

    auto msg = sjcam_msgs::msg::EventArray();
    msg.header.stamp = stamp;
    msg.header.frame_id = key;

    // 至少需要 4 字节 count 头
    if (data.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Event 数据不足 4 字节 (%zu)", data.size());
        return;
    }

    // 读取 EventCount (int32 LE)
    int32_t count = 0;
    std::memcpy(&count, data.data(), 4);

    constexpr size_t EVENT_SIZE = 20;  // t(f64) + x(i32) + y(i32) + p(i32)
    size_t available = (data.size() - 4) / EVENT_SIZE;
    size_t n_events = std::min(static_cast<size_t>(count), available);
    msg.events.resize(n_events);

    for (size_t i = 0; i < n_events; ++i) {
        const uint8_t * ptr = data.data() + 4 + i * EVENT_SIZE;

        // t: float64 (秒) → builtin_interfaces/Time
        double t_sec = 0.0;
        std::memcpy(&t_sec, ptr, 8);
        msg.events[i].ts.sec  = static_cast<int32_t>(std::floor(t_sec));
        msg.events[i].ts.nanosec = static_cast<uint32_t>((t_sec - std::floor(t_sec)) * 1e9);

        // x: int32 → uint16
        int32_t x_raw = 0;
        std::memcpy(&x_raw, ptr + 8, 4);
        msg.events[i].x = static_cast<uint16_t>(x_raw);

        // y: int32 → uint16
        int32_t y_raw = 0;
        std::memcpy(&y_raw, ptr + 12, 4);
        msg.events[i].y = static_cast<uint16_t>(y_raw);

        // p: int32 → bool polarity (非零 = true)
        int32_t p_raw = 0;
        std::memcpy(&p_raw, ptr + 16, 4);
        msg.events[i].polarity = (p_raw != 0);
    }

    auto pub = std::dynamic_pointer_cast<
        rclcpp::Publisher<sjcam_msgs::msg::EventArray>>(it->second.publisher);
    if (pub) pub->publish(msg);

    // 发布事件累积图像
    publishEventImage(key, msg, stamp);
}

/**
 * @brief 将事件累积为图像并发布 (bgr8)
 *
 * 与 verify_data.py 中的事件累积图像逻辑一致:
 *   ON 事件 (polarity=true)  → 红色 (BGR: 0,0,255)
 *   OFF 事件 (polarity=false) → 蓝色 (BGR: 255,0,0)
 *   背景 → 黑色 (0,0,0)
 *
 * 图像尺寸由事件坐标动态确定
 */
void SensorCaptureNode::publishEventImage(
    const std::string & key,
    const sjcam_msgs::msg::EventArray & msg,
    const rclcpp::Time & stamp)
{
    auto it = event_image_publishers_.find(key);
    if (it == event_image_publishers_.end()) return;
    if (msg.events.empty()) return;

    // 动态确定图像尺寸 (从事件坐标取最大值)
    uint16_t max_x = 0, max_y = 0;
    for (const auto & e : msg.events) {
        if (e.x > max_x) max_x = e.x;
        if (e.y > max_y) max_y = e.y;
    }
    int width  = max_x + 1;
    int height = max_y + 1;

    // 创建黑色背景 BGR 图像
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);

    // 累积事件到图像
    for (const auto & e : msg.events) {
        cv::Vec3b & pixel = img.at<cv::Vec3b>(e.y, e.x);
        if (e.polarity) {
            pixel[2] = 255;  // ON  → 红色 (BGR 的 R 通道)
        } else {
            pixel[0] = 255;  // OFF → 蓝色 (BGR 的 B 通道)
        }
    }

    auto img_msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), "bgr8", img).toImageMsg();
    img_msg->header.stamp = stamp;
    img_msg->header.frame_id = key;
    it->second->publish(*img_msg);
}

} // namespace sensor_capture_ros2

// ============================================================
//  main
// ============================================================

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<sensor_capture_ros2::SensorCaptureNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
