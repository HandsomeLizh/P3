cmake_minimum_required(VERSION 3.8)
project(sensor_capture_ros2)

# C++17 标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sjcam_msgs REQUIRED)

# 可执行文件
add_executable(sensor_capture_node
  src/sensor_capture_node.cpp
  src/capture_publication.cpp
)

target_include_directories(sensor_capture_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(sensor_capture_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  sjcam_msgs
)

target_link_libraries(sensor_capture_node
  ${OpenCV_LIBS}
)

# 运行时库搜索路径 (RPATH):
# 节点二进制带 setcap 能力位 (CAP_SYS_NICE) 后以安全执行模式 (AT_SECURE) 启动,
# 动态链接器会忽略 LD_LIBRARY_PATH, 导致找不到 librclcpp.so。
# 因此必须将依赖库路径固化到二进制内: $ORIGIN + ROS 安装目录 + 工作空间消息包。
if(DEFINED ENV{ROS_DISTRO})
  set(ROS_LIB_DIR "/opt/ros/$ENV{ROS_DISTRO}/lib")
else()
  set(ROS_LIB_DIR "/opt/ros/humble/lib")
endif()
set_target_properties(sensor_capture_node PROPERTIES
  INSTALL_RPATH "$ORIGIN;${ROS_LIB_DIR};${CMAKE_INSTALL_PREFIX}/../sjcam_msgs/lib"
)
# 使用传统 DT_RPATH 而非 DT_RUNPATH: RUNPATH 不作用于间接依赖的库搜索,
# 会导致 librcl.so 的依赖 (如 librcl_yaml_param_parser.so) 无法找到
target_link_options(sensor_capture_node PRIVATE "-Wl,--disable-new-dtags")

# 安装
install(TARGETS sensor_capture_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
