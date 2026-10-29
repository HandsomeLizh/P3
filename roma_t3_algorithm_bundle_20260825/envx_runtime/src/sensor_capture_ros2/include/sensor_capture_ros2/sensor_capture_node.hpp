/**
 * @file sensor_capture_node.hpp
 * @brief ROS2 传感器数据采集节点 (C++ 版本)
 *
 * 功能概述:
 *   通过 TCP 连接 Unreal Engine 服务器，发送采集指令，接收多传感器数据，
 *   按传感器类型发布到不同 ROS2 Topic，同时保存到本地磁盘。
 *
 * 通信协议:
 *   - 包头格式 (16字节): MAGIC(4) + CMD(4) + PayloadSize(4) + Checksum(4)
 *   - 采集指令载荷 (10字节): HEAD(4) + SensorType(1) + CameraID(1) + Checksum(1)
 *   - 曝光设置载荷 (11字节): HEAD(4) + SensorType(1) + CameraID(1) + ExposureMs(4) + Checksum(1)
 *   - 灯光亮度载荷 (9字节): HEAD(4) + Intensity(4) + Checksum(1)
 *   - 数据返回格式: 包头 + FSensorFileHeader + 文件内容
 *   - 字节序可通过参数 big_endian 配置 (默认小端)
 *
 * 支持的传感器:
 *   - RGB 彩色相机 (0x01): 发布为 sensor_msgs/Image (PNG 解码)
 *   - ToF 深度相机 (0x02): 发布为 sensor_msgs/PointCloud2
 *   - LiDAR 激光雷达 (0x03): 发布为 std_msgs/ByteMultiArray
 *   - CPU LiDAR (0x04): 发布为 sensor_msgs/PointCloud2
 *   - Event 事件相机 (0x05): 发布为 sensor_capture_ros2/EventArray (x,y,t,p 事件流)
 *
 * Topic 配置:
 *   通过 output_topics 参数配置，格式为 "key:=topic_name"
 *   例: ["rgb_0:=/Car/T5/Cam_Left/image_raw/color", ...]
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <cstdint>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include "sjcam_msgs/msg/event_array.hpp"

#include <opencv2/core.hpp>
#include "sensor_capture_ros2/batch_timing.hpp"

namespace sensor_capture_ros2
{

// ============================================================
//  协议常量
// ============================================================

/** 包头魔数 "LIDA" (0x4C494441)，标识有效数据包 */
static constexpr uint32_t MAGIC              = 0x4C494441;

/** 命令类型: 开始采集 */
static constexpr uint32_t CMD_START_CAPTURE  = 0x00000001;

/** 命令类型: 数据传输 */
static constexpr uint32_t CMD_DATA           = 0x00000004;

/** 命令类型: 设置相机曝光时间 */
static constexpr uint32_t CMD_SET_EXPOSURE   = 0x00000005;

/** 命令类型: 设置灯光亮度 */
static constexpr uint32_t CMD_SET_LIGHT_INTENSITY = 0x00000006;

/** 传感器数据头标识，UE 返回数据时以此开头 */
static constexpr uint32_t HEAD_SENSOR_DATA   = 0xEB9AEB9A;

/** 反馈头标识，UE 返回状态反馈时以此开头 */
static constexpr uint32_t HEAD_FEEDBACK      = 0xEB93EB93;

/** 采集请求头标识，发送采集指令时载荷以此开头 */
static constexpr uint32_t HEAD_CAPTURE_REQUEST = 0xEB92EB92;

/** 曝光设置头标识，发送曝光设置指令时载荷以此开头 */
static constexpr uint32_t HEAD_SET_EXPOSURE  = 0xEB90EB90;

/** 灯光亮度设置头标识，发送灯光亮度指令时载荷以此开头 */
static constexpr uint32_t HEAD_SET_LIGHT_INTENSITY = 0xEB91EB91;

// ============================================================
//  传感器类型常量
// ============================================================

/** RGB 彩色相机 */
static constexpr uint8_t SENSOR_RGB        = 0x01;

/** ToF 深度相机 */
static constexpr uint8_t SENSOR_TOF        = 0x02;

/** CPU 端激光雷达 */
static constexpr uint8_t SENSOR_CPU_LIDAR  = 0x04;

/** Event 事件相机 */
static constexpr uint8_t SENSOR_EVENT      = 0x05;

// ============================================================
//  文件魔数签名 (用于识别文件类型)
// ============================================================

/** PNG 文件头 8 字节签名 */
static const std::vector<uint8_t> PNG_MAGIC = {
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
};

/** NPY (NumPy) 文件头 6 字节签名 */
static const std::vector<uint8_t> NPY_MAGIC = {
    0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59
};

// ============================================================
//  传感器文件头结构
//  UE 返回数据时，每个文件前都带有此头信息
// ============================================================
struct SensorFileHeader
{
    uint32_t head;           ///< 文件头标识 (HEAD_SENSOR_DATA)
    uint32_t batch_id;       ///< 当前批次 ID
    uint32_t total_count;    ///< 本批次总文件数
    uint32_t sequence;       ///< 当前文件序号
    uint32_t file_size;      ///< 文件数据大小 (字节)
    uint32_t timestamp;      ///< 时间戳
    uint8_t  sensor_type;    ///< 传感器类型 (SENSOR_RGB/TOF/...)
    uint8_t  camera_id;      ///< 相机编号
    uint8_t  flags;          ///< 标志位
    uint8_t  reserved;       ///< 保留字段
    char     file_name_raw[128]; ///< 原始文件名 (UTF-16LE, 64 TCHAR)
    std::string file_name;   ///< 解析后的文件名 (UTF-8)
};

// ============================================================
//  传感器配置信息
// ============================================================
struct SensorConfig
{
    uint8_t type;            ///< 传感器类型
    int camera_id;           ///< 相机编号
    std::string name;        ///< 传感器名称 (用于 topic key)
};

// ============================================================
//  TOF 相机配置 (按 camera_id 区分)
// ============================================================

/**
 * TOF 相机参数配置:
 *   - tof_0/tof_1: 640×480, FOV 67°×51°, 数据布局 [深度平面][强度平面]
 *   - tof_2/tof_3: 2448×2048, FOV 55.7°×47.7°, 仅深度平面 (无强度数据)
 */
struct TofCamConfig
{
    size_t width;          ///< 图像宽度 (像素)
    size_t height;         ///< 图像高度 (像素)
    double fov_x_deg;      ///< 水平视场角 (度)
    double fov_y_deg;      ///< 垂直视场角 (度)
    bool has_intensity;    ///< 数据中是否包含强度平面 (深度平面始终在偏移 0 处)
};

// ============================================================
//  传感器采集节点类
// ============================================================
class SensorCaptureNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数，初始化参数、发布者、订阅者，启动采集线程
     * @param options ROS2 节点选项
     */
    explicit SensorCaptureNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

    /** @brief 析构函数，停止采集线程，关闭连接 */
    ~SensorCaptureNode() override;

private:
    // ========================================================
    //  可配置参数
    // ========================================================
    std::string tcp_host_;           ///< UE 服务器 IP 地址 (默认: 169.254.146.1)
    int tcp_port_;                   ///< UE 服务器端口 (默认: 6665)
    std::string save_dir_;           ///< 数据保存目录 (默认: /home/yanfa/Env_X/SaveImages)
    int batch_count_;                ///< 采集批次总数 (默认: 600)
    double timeout_;                 ///< 接收超时秒数 (默认: 120.0)
    double batch_interval_;          ///< 批次间隔秒数 (默认: 2.0)
    double capture_freq_;            ///< 采集指令发送频率 Hz (默认: 0.5，即每 2s 一条)
    std::string timestamp_mode_;     ///< legacy_header_us / meta_relative
    bool big_endian_;                ///< 字节序: true=大端, false=小端 (默认: false)
    bool publish_thumbnail_;         ///< 是否同时发布 320x240 轻量版图像 (默认: false)
    int thumb_width_;                ///< 缩略图宽度 (默认: 320)
    int thumb_height_;               ///< 缩略图高度 (默认: 240)
    double tof_fov_x_deg_;           ///< TOF 0/1 (tof_0/tof_1) 水平视场角 (默认: 67°)
    double tof_fov_y_deg_;           ///< TOF 0/1 (tof_0/tof_1) 垂直视场角 (默认: 51°)
    double tof2_fov_x_deg_;          ///< TOF 2/3 (tof_2/tof_3) 水平视场角 (默认: 55.7°)
    double tof2_fov_y_deg_;          ///< TOF 2/3 (tof_2/tof_3) 垂直视场角 (默认: 47.7°)
    double tof_depth_scale_;         ///< TOF 深度原始值→米的换算系数 (默认: 0.01, 即原始单位为 cm)
    double tof_max_range_;           ///< TOF 有效最大量程 (米, 默认: 9.5), 超过此距离的点视为无效并剔除
    double lidar_scale_;             ///< LiDAR 坐标原始值→米的换算系数 (默认: 0.01, 即原始单位为 cm)
    bool normalize_save_;            ///< 保存深度图/强度图前是否归一化到 0~65535 (默认: false)
    int exposure_camera_id_;         ///< (已废弃，保留参数兼容) 曝光设置目标相机编号
    double exposure_offset_ms_;      ///< 曝光偏差 (毫秒): RGB 0/1 曝光 = RGB 2/3 曝光 + 该值 (默认: 600.0)
    std::vector<std::string> topic_mapping_;  ///< Topic 映射列表 "key:=topic"

    /**
     * @brief 队列数据项 (接收线程 → 处理线程)
     */
    struct BatchItem
    {
        SensorFileHeader header;
        std::vector<uint8_t> file_data;
        std::string batch_folder;
        int batch_index;
        bool is_meta;           ///< true = meta.json
        std::vector<uint8_t> meta_raw;  ///< meta.json 原始数据
    };

    // ========================================================
    //  TCP 连接状态
    // ========================================================
    int sock_fd_;                    ///< TCP socket 文件描述符
    std::atomic<bool> running_;      ///< 节点运行标志
    std::thread capture_thread_;     ///< 采集线程
    rclcpp::Time start_time_;        ///< 软件启动基准时间，用于计算发布时时间戳
    rclcpp::Time batch_request_stamp_;
    rclcpp::Time batch_publish_stamp_;
    BatchClock batch_clock_;
    std::optional<uint32_t> last_completed_batch_id_;
    bool save_data_;                 ///< 是否保存数据到磁盘 (默认: true)

    // ---- 接收/处理线程流水线 ----
    std::thread receive_thread_;     ///< 数据接收线程
    std::thread process_thread_;     ///< 数据处理线程
    std::queue<BatchItem> data_queue_;  ///< 接收→处理 数据队列
    std::mutex queue_mutex_;         ///< 队列互斥锁
    std::condition_variable queue_cv_;  ///< 队列条件变量
    std::atomic<bool> batch_done_;   ///< 当前批次接收完成标志
    std::atomic<bool> batch_error_;  ///< 当前批次接收错误标志
    std::vector<BatchItem> batch_items_;  ///< 批次缓存: 接收阶段保存，完成后统一发布

    // ---- 传感器数据包计数器 ----
    std::map<uint8_t, int> batch_recv_counts_;   ///< 当前批次各类型接收包数
    std::map<uint8_t, int> total_recv_counts_;   ///< 累计各类型接收包数

    // ========================================================
    //  ROS2 发布者
    // ========================================================

    /** 发布者信息结构
     * 存储发布者和对应的消息类型
     */
    struct PublisherInfo
    {
        rclcpp::PublisherBase::SharedPtr publisher;  ///< 发布者指针
        std::string msg_type;  ///< 消息类型: "image", "pointcloud", "byte_array", "camera_info"
    };

    /** 动态创建的发布者映射表 (key -> PublisherInfo) */
    std::map<std::string, PublisherInfo> publishers_;

    /** 缩略图发布者映射表 (key -> Image publisher, topic 追加 /thumb) */
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> thumb_publishers_;

    /** TOF 深度图发布者映射表 (key: tof_0_depth -> topic) */
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> tof_depth_publishers_;

    /** TOF 强度图发布者映射表 (key: tof_0_intensity -> topic) */
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> tof_intensity_publishers_;

    /** 事件相机累积图像发布者 (key -> Image publisher, topic 追加 /image) */
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> event_image_publishers_;

    /** 元数据发布者 (批次信息 JSON) */
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr meta_pub_;

    /** 批次状态发布者 (采集进度) */
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr batch_status_pub_;

    // ========================================================
    //  ROS2 订阅者
    // ========================================================

    /** 外部触发采集订阅 (发送 Empty 消息启动采集) */
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr sub_start_capture_;

    /** 曝光时间设置订阅 (Float32, 单位毫秒, 值为 RGB 2/3 曝光; 加固定增量后转发给 RGB 0/1) */
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_set_exposure_;

    /** 灯光亮度设置订阅 (Float32, 0.0 ~ 1.0 或更大) */
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_set_light_intensity_;

    // ========================================================
    //  初始化与设置方法
    // ========================================================

    /** @brief 根据 topic_mapping_ 创建所有发布者 */
    void setupPublishers();

    /**
     * @brief 根据 topic 名称推断消息类型
     * @param topic topic 名称
     * @return 消息类型字符串 ("image", "pointcloud", "byte_array", "camera_info")
     */
    std::string inferMsgType(const std::string & topic);

    /**
     * @brief 外部触发采集回调
     * @param msg Empty 消息
     */
    void onStartCaptureTrigger(const std_msgs::msg::Empty::SharedPtr msg);

    /**
     * @brief 曝光时间设置回调 (调用 applyExposure 下发)
     * @param msg Float32 消息, data 为 RGB 2/3 的曝光时间 (毫秒)
     */
    void onSetExposure(const std_msgs::msg::Float32::SharedPtr msg);

    /**
     * @brief 下发曝光设置到所有 RGB 相机
     * RGB 2/3 (低曝光相机) 直接使用 exposure_23_ms;
     * RGB 0/1 (彩色相机) 曝光 = exposure_23_ms + exposure_offset_ms_ 偏差，计算完成后一并下发 (共 4 条指令)
     * @param exposure_23_ms RGB 2/3 的曝光时间 (毫秒)
     * @param source 来源描述 (用于日志, 如 "订阅消息")
     */
    void applyExposure(float exposure_23_ms, const std::string & source);

    /**
     * @brief 灯光亮度设置回调 (构建灯光亮度指令并通过 TCP 发送)
     * @param msg Float32 消息, data 为亮度值 (0.0 ~ 1.0 或更大)
     */
    void onSetLightIntensity(const std_msgs::msg::Float32::SharedPtr msg);

    // ========================================================
    //  采集控制方法
    // ========================================================

    /** @brief 启动采集 (在独立线程中运行) */
    void startCapture();

    /** @brief 采集主循环: 发送指令 -> 接收数据 -> 保存/发布 */
    void captureLoop();

    // ========================================================
    //  TCP 连接方法
    // ========================================================

    /**
     * @brief 连接到 UE 服务器
     * @return 连接成功返回 true
     */
    bool connectToServer();

    /** @brief 断开 TCP 连接 */
    void disconnectFromServer();

    // ========================================================
    //  数据接收与解析方法
    // ========================================================

    /**
     * @brief 精确接收指定字节数
     * @param buffer 接收缓冲区
     * @param size 需要接收的字节数
     * @return 成功返回 true
     */
    bool recvExact(std::vector<uint8_t> & buffer, size_t size);

    /**
     * @brief 接收一个完整协议包 (包头 + 载荷)
     * @param command 输出: 命令类型
     * @param payload 输出: 载荷数据
     * @return 成功返回 true
     */
    bool recvPacket(uint32_t & command, std::vector<uint8_t> & payload);

    /**
     * @brief 解析传感器文件头
     * @param data 原始数据
     * @param header 输出: 解析后的文件头
     * @return 解析成功返回 true
     */
    bool parseFileHeader(const std::vector<uint8_t> & data, SensorFileHeader & header);

    /**
     * @brief 从数据包中提取文件数据 (跳过文件头)
     * @param payload 完整载荷数据
     * @param header 已解析的文件头
     * @return 提取的文件数据
     */
    std::vector<uint8_t> extractFileData(
        const std::vector<uint8_t> & payload, const SensorFileHeader & header);

    /**
     * @brief 查找有效载荷起始位置 (处理 NPY/PNG 头偏移)
     * @param data 原始数据
     * @param file_name 文件名
     * @return 有效载荷起始偏移
     */
    size_t findPayloadStart(const std::vector<uint8_t> & data,
                            const std::string & file_name);

    // ========================================================
    //  指令构建方法
    // ========================================================

    /**
     * @brief 构建采集指令包
     * @param sensor_type 传感器类型
     * @param camera_id 相机编号
     * @return 完整的指令包 (包头 + 载荷)
     */
    std::vector<uint8_t> buildStartCapture(uint8_t sensor_type, uint8_t camera_id);

    /**
     * @brief 构建曝光时间设置指令包 (CMD_SET_EXPOSURE)
     * 载荷: HEAD_SET_EXPOSURE(4) + SensorType(1) + CameraID(1) + ExposureMs(4, float) + Checksum(1)
     * Checksum = (Head + SensorType + CameraID + *(uint32*)&ExposureMs) & 0xFF
     * @param sensor_type 传感器类型 (SENSOR_RGB)
     * @param camera_id 相机编号 (0~3 指定相机, 0xFF 所有相机)
     * @param exposure_ms 曝光时间 (毫秒)
     * @return 完整的指令包 (包头 + 载荷)
     */
    std::vector<uint8_t> buildSetExposure(uint8_t sensor_type, uint8_t camera_id,
                                          float exposure_ms);

    /**
     * @brief 构建灯光亮度设置指令包 (CMD_SET_LIGHT_INTENSITY)
     * 载荷: HEAD_SET_LIGHT_INTENSITY(4) + Intensity(4, float) + Checksum(1)
     * Checksum = (Head + *(uint32*)&Intensity) & 0xFF
     * @param intensity 亮度值 (0.0 ~ 1.0 或更大)
     * @return 完整的指令包 (包头 + 载荷)
     */
    std::vector<uint8_t> buildSetLightIntensity(float intensity);

    /**
     * @brief 通过 TCP 发送指令包 (线程安全, 未连接时自动尝试连接)
     * @param packet 完整指令包 (包头 + 载荷)
     * @param desc 指令描述 (用于日志输出)
     * @return 发送成功返回 true
     */
    bool sendCommand(const std::vector<uint8_t> & packet, const std::string & desc);

    /** @brief TCP 发送互斥锁 (防止采集线程与订阅回调并发写入导致数据交错) */
    std::mutex send_mutex_;

    /**
     * @brief 按配置的字节序读取 32 位无符号整数
     * @param p 指向 4 字节数据的指针
     * @return 解析后的 uint32 值
     * @note 根据 big_endian_ 参数选择大端或小端序
     */
    uint32_t readU32(const uint8_t * p) const;

    /**
     * @brief 按配置的字节序写入 32 位无符号整数
     * @param buf 输出缓冲区
     * @param v 要写入的值
     * @note 根据 big_endian_ 参数选择大端或小端序
     */
    void pushU32(std::vector<uint8_t> & buf, uint32_t v) const;

    // ========================================================
    //  数据接收与发布方法
    // ========================================================

    /** @brief 接收一个批次的所有数据 (启动接收线程和处理线程) */
    bool receiveBatch(int batch_index);

    /** @brief 数据接收线程: TCP 读取 → 解析 → 入队 */
    void receiveThread(int batch_index, const std::string & batch_folder);

    /** @brief 数据处理线程: 出队 → 保存 → 发布 */
    void processThread(const std::string & batch_folder);

    /**
     * @brief 根据传感器类型发布数据到对应 Topic
     * @param header 文件头信息
     * @param file_data 文件数据
     * @param file_name 文件名
     */
    void publishSensorData(const SensorFileHeader & header,
                           const std::vector<uint8_t> & file_data,
                           const std::string & file_name);

    /**
     * @brief 发布图像数据 (PNG 解码为 sensor_msgs/Image)
     * @param key 发布者 key
     * @param data PNG 编码数据
     * @param timestamp 时间戳
     */
    void publishImage(const std::string & key,
                      const std::vector<uint8_t> & data, const rclcpp::Time & stamp);

    /**
     * @brief 发布缩略图 (将图像缩放到 thumb_width_ x thumb_height_ 后发布)
     * @param key 发布者 key
     * @param decoded 已解码的原始图像 (cv::Mat)
     * @param stamp 时间戳
     */
    void publishThumbnail(const std::string & key,
                          const cv::Mat & decoded, const rclcpp::Time & stamp);

    /**
     * @brief 发布点云数据 (sensor_msgs/PointCloud2)
     * @param key 发布者 key
     * @param data 点云数据
     * @param timestamp 时间戳
     */
    void publishPointCloud(const std::string & key,
                           const std::vector<uint8_t> & data, const rclcpp::Time & stamp);

    /**
     * @brief 获取 TOF 相机配置 (按 camera_id: 0/1 为 640×480 双平面, 2/3 为 2448×2048 仅深度)
     * @param camera_id 相机编号
     * @return 对应的 TOF 相机配置
     */
    TofCamConfig getTofConfig(uint8_t camera_id) const;

    /**
     * @brief 解析 TOF 深度相机数据并发布为 PointCloud2
     * 数据格式: tof_0/1 = [深度平面][强度平面]; tof_2/3 = 仅深度平面
     * 点云字段: x, y, z, intensity，均为 float32 (无强度数据时 intensity 填 0)
     * @param key 发布者 key
     * @param data TOF 原始数据
     * @param cfg TOF 相机配置 (分辨率/视场角/是否含强度平面)
     * @param stamp 时间戳
     */
    void publishTOFPointCloud(const std::string & key,
                              const std::vector<uint8_t> & data,
                              const TofCamConfig & cfg, const rclcpp::Time & stamp);

    /**
     * @brief 发布 TOF 深度图 (mono16 Image)
     * @param key 发布者 key (tof_N_depth)
     * @param data TOF 原始数据
     * @param cfg TOF 相机配置 (分辨率/数据布局)
     * @param stamp 时间戳
     */
    void publishTOFDepthImage(const std::string & key,
                              const std::vector<uint8_t> & data,
                              const TofCamConfig & cfg, const rclcpp::Time & stamp);

    // 注: TOF 强度图不再发布 (tof_0/1 仅保留磁盘保存, tof_2/3 无强度数据)

    /**
     * @brief 发布字节数组数据 (std_msgs/ByteMultiArray)
     * @param key 发布者 key
     * @param data 原始字节数据
     * @param timestamp 时间戳
     */
    void publishByteArray(const std::string & key,
                          const std::vector<uint8_t> & data, const rclcpp::Time & stamp);

    /**
     * @brief 发布相机标定信息 (sensor_msgs/CameraInfo)
     * @param key 发布者 key
     * @param timestamp 时间戳
     */
    void publishCameraInfo(const std::string & key, const rclcpp::Time & stamp);

    /**
     * @brief 发布事件相机数据 (sjcam_msgs/EventArray)
     * 解析原始字节数据为 x,y,ts,polarity 事件流
     * @param key 发布者 key
     * @param data 原始事件数据 (每个事件: x(uint16) + y(uint16) + t(uint64微秒) + p(int8) = 13字节)
     * @param stamp 时间戳
     */
    void publishEventArray(const std::string & key,
                           const std::vector<uint8_t> & data, const rclcpp::Time & stamp);

    /**
     * @brief 将事件累积为图像并发布 (bgr8, ON=红色 OFF=蓝色)
     * @param key 发布者 key
     * @param msg 已解析的 EventArray 消息
     * @param stamp 时间戳
     */
    void publishEventImage(const std::string & key,
                           const sjcam_msgs::msg::EventArray & msg, const rclcpp::Time & stamp);

    // ========================================================
    //  辅助方法
    // ========================================================

    /**
     * @brief 将 PointCloud2 保存为 PCD (binary) 文件
     * @param path 保存路径 (.pcd)
     * @param cloud 点云消息 (使用 fields/width/height/data)
     * @return 保存成功返回 true
     */
    bool savePcdFile(const std::string & path, const sensor_msgs::msg::PointCloud2 & cloud);

    /**
     * @brief 解析 TOF 数据构建 PointCloud2 (基于相机配置的分辨率/视场角计算 3D 坐标)
     * @param data TOF 原始数据 (布局由 cfg.has_intensity 决定)
     * @param cfg TOF 相机配置 (分辨率/视场角/是否含强度平面)
     * @param msg 输出: 点云消息 (不含 header, 由调用方设置; 坐标单位: 米, 由 tof_depth_scale_ 换算)
     * @return 数据充足返回 true
     */
    bool buildTofPointCloud(const std::vector<uint8_t> & data,
                            const TofCamConfig & cfg, sensor_msgs::msg::PointCloud2 & msg);

    /**
     * @brief 从 TOF 原始数据构建深度图 (CV_16UC1)
     * @param data TOF 原始数据 (深度平面在偏移 0 处)
     * @param cfg TOF 相机配置 (决定尺寸与最小数据量校验)
     * @param depth_img 输出: 深度图 (CV_16UC1)
     * @return 数据充足返回 true
     */
    bool buildTofDepthImage(const std::vector<uint8_t> & data,
                            const TofCamConfig & cfg, cv::Mat & depth_img);

    /**
     * @brief 从 TOF 原始数据构建强度图 (CV_16UC1)
     * @param data TOF 原始数据 [深度平面][强度平面]
     * @param width 图像宽度 (640)
     * @param height 图像高度 (480)
     * @param intensity_img 输出: 强度图 (CV_16UC1)
     * @return 数据充足返回 true
     */
    bool buildTofIntensityImage(const std::vector<uint8_t> & data,
                                size_t width, size_t height, cv::Mat & intensity_img);

    /**
     * @brief 解析 CPU LiDAR 数据构建 PointCloud2
     * @param data LiDAR 原始数据 [PointCount: int32] + [x,y,z × float32 ...]
     * @param msg 输出: 点云消息 (不含 header, 由调用方设置; 坐标单位: 米, 由 lidar_scale_ 换算)
     * @return 数据充足返回 true
     */
    bool buildLidarPointCloud(const std::vector<uint8_t> & data, sensor_msgs::msg::PointCloud2 & msg);

    /**
     * @brief 生成发布者 key (格式: sensor_name + "_" + camera_id)
     * @param sensor_type 传感器类型
     * @param camera_id 相机编号
     * @return 发布者 key 字符串
     */
    std::string makePublisherKey(uint8_t sensor_type, uint8_t camera_id);

    /**
     * @brief 获取传感器名称
     * @param sensor_type 传感器类型
     * @return 传感器名称 ("rgb", "tof", "lidar", "cpulidar", "event")
     */
    std::string getSensorName(uint8_t sensor_type);
};

} // namespace sensor_capture_ros2
