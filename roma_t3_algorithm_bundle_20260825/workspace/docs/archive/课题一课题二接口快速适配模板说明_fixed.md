# 课题一/课题二实时 ROS 2 接口快速适配说明（fixed）

> 更新日期：2026-08-16  
> 适用工程：`t3_gridmap_v2_ws`  
> 适用场景：明天直接接课题一、课题二的实时 ROS 2 系统

## 1. Orin 部署与课题一/二接入操作步骤

以 `/opt/t3_gridmap_v2_ws` 为最终部署位置。下面的“原大包 +
三文件更新包”和“直接复制完整工作空间”是两种部署方式，
只选一种，不要重复覆盖。

### 1.1 方式 A：先解压原大包，再覆盖三文件更新包

1. 在 Orin 上先将原来的 RoMa 轻量版大包解压到 `/opt`：

```bash
sudo tar -xzf \
  t3_gridmap_v2_ws_orin_offline_roma_only_fixed_20260816.tar.gz \
  -C /opt
```

2. 将三文件交付包解压到临时目录：

```bash
mkdir -p /tmp/t3_adapter_update

tar -xzf \
  t3_gridmap_v2_ws_t1_t2_live_adapter_fixed_20260816_delivery.tar.gz \
  -C /tmp/t3_adapter_update
```

3. 进入临时目录并确认小更新包可以正常读取：

```bash
cd /tmp/t3_adapter_update
tar -tzf t3_gridmap_v2_ws_t1_t2_live_adapter_fixed_20260816.tar.gz >/dev/null
```

命令无报错才继续。如果文件不存在或压缩包不可读，不要覆盖已部署的工作空间，
应重新获取交付包。

4. 检查成功后，把小更新覆盖到原大工作空间：

```bash
sudo tar -xzf \
  t3_gridmap_v2_ws_t1_t2_live_adapter_fixed_20260816.tar.gz \
  -C /opt
```

覆盖完成后，适配器应位于：

```text
/opt/t3_gridmap_v2_ws/src/t3_upstream_adapters
```

### 1.2 方式 B：直接复制当前完整工作空间

如果直接将当前完整的 `modified/t3_gridmap_v2_ws` 文件夹复制为
`/opt/t3_gridmap_v2_ws`，适配器已经包含在 `src/t3_upstream_adapters/`
中，不需要再执行 1.1 节的小更新包覆盖操作。

### 1.3 Orin 第一次部署

不论使用方式 A 还是方式 B，第一次在 Orin 部署时执行：

```bash
cd /opt/t3_gridmap_v2_ws
bash offline_deploy/scripts/install_native_orin.sh
```

### 1.4 拿到课题一、课题二正式接口包后

将两个完整的 ROS 2 接口包及它们的自定义接口依赖复制到
`src/vendor_interfaces/` 下。以接口包名分别为 `t1_interfaces`
和 `t2_interfaces` 为例：

```bash
cp -a /拿到的/t1_interfaces \
  /opt/t3_gridmap_v2_ws/src/vendor_interfaces/

cp -a /拿到的/t2_interfaces \
  /opt/t3_gridmap_v2_ws/src/vendor_interfaces/

cd /opt/t3_gridmap_v2_ws
bash scripts/build_t1_t2_adapter_template.sh
```

然后启动课题一、课题二的实时节点，执行
`bash scripts/collect_t1_t2_interface_info.sh` 保存话题、消息类型、
QoS 和消息定义。如果是自定义消息，再只修改
`src/t3_upstream_adapters/config/interface_mapping.yaml`。

### 1.5 根据课题一、课题二的输入形式选择 profile

| profile | 课题一输入 | 课题二输入 | 明天需要改什么 |
|---|---|---|---|
| `standard` | 标准独立话题 | 标准 `mono8 Image` | 只在启动命令传真实话题名 |
| `t1` | 自定义聚合消息 | 标准 `mono8 Image` | 只填 YAML 的 `t1_fusion_adapter` |
| `t2` | 标准独立话题 | 自定义语义消息 | 只填 YAML 的 `t2_scene_adapter` |
| `t1_t2` | 自定义聚合消息 | 自定义语义消息 | YAML 两段都填 |

这里的 `standard` 不是某一个叫 Standard 的消息，也不要求话题名必须与
本工程默认值一样。它表示上游已经把每种数据作为独立话题，并直接使用
`sensor_msgs` / `nav_msgs` 的标准类型，课题三只需在启动命令中传入真实话题名。

启动时只选一个 profile：

```bash
# 两边都是标准消息
bash scripts/run_t1_t2_adapter_template.sh --profile standard ...

# 只有课题一是自定义消息
bash scripts/run_t1_t2_adapter_template.sh --profile t1 ...

# 只有课题二是自定义消息
bash scripts/run_t1_t2_adapter_template.sh --profile t2 ...

# 两边都是自定义消息
bash scripts/run_t1_t2_adapter_template.sh --profile t1_t2 ...
```

上面的 `...` 表示需要根据正式接口补充的话题、标定、输出目录等
launch 参数，不要把字面值 `...` 输入到实际命令中。

修改 YAML 不需要重新编译；启动脚本会直接读取源码目录中的该文件。
只有对方 `.msg` 包或适配器 Python 代码改变时才需要重新编译。

这个 profile 只是启动层的小 `case/if`，不进 mapper。适配器内部不为每一版
对方消息写一堆包名分支，而是用“消息类型 + 字段路径”配置解决层级差异。
课题二另有两个已实现的小分支：嵌套 `Image` 或 `uint8[] + width + height + Header`；
通过 YAML 中 `label_image_field` 是否为空自动选择。

## 2. 为什么这样适配

```text
课题一正式自定义消息
        |
        v
t1_fusion_adapter  ---->  Image / Imu / PointCloud2 稳定话题
                                                     |
课题二 SceneSemanticModel                          v
        |                                    roma_semantic_mapper
        v
t2_scene_adapter   ---->  mono8 语义 Image 稳定话题
```

适配器只做三件事：

- 按配置字段路径从自定义消息中取出标准 ROS 2 消息；
- 检查类型、尺寸、编码、载荷长度、`Header.stamp`、`frame_id` 和 QoS；
- 将课题二类别 ID 映射到课题三的固定类别 ID。

它不会重打时间戳、猜测外参、修改 RoMa、修改 VO/LIO/回环、修改地图输出，
也不会在类型或时间不对时静默放行。因此对方接口再变，通常只改边界 YAML，
不会把变化传入定位建图主回调。

### 2.1 三种方案如何选

| 方案 | 何时使用 | 当前状态 | 代价 |
|---|---|---|---|
| A：标准话题直接接入 | 课题一/二已经发布标准 `Image/Imu/PointCloud2/Odometry` | 现在可运行 | 无额外转发，最简单 |
| B：可配置薄适配层 | 对方只发自定义聚合消息 | 现在可编译/运行 | 多一次 DDS 转发，不改 mapper |
| C：直接修改 mapper 输入前端 | 正式接口已锁定，且 Orin 数据证明 B 的转发是瓶颈 | 等正式 `.msg` 后实施 | 延迟低，但耦合、测试和回归成本最高 |

选择顺序固定为 `A -> B -> C`：能标准直连就不转发；确实是聚合消息就先
用 B 完成契约验证；只有实测证明 B 不满足实时性才做 C。三种不同时开启，
否则同一传感器会出现重复 publisher 和不可判定的数据来源。

方案 C 不能在没有正式消息定义时提前写死。正式接口到手后，按下列边界实施：

1. 将对方锁定接口包加入 `t3_semantic_mapping/package.xml` 的显式运行依赖。
2. 新建独立 `custom_input_frontend.py`，只做字段解包、时间同步和契约检查；
   不把字段解析散落进 `roma_semantic_mapper_node.py`。
3. 在 `launch_options.py` 增加默认为 `standard` 的输入前端开关；旧标准订阅路径
   的默认参数、QoS 和同步器必须保持不变。
4. 自定义前端取出标准消息后，复用现有 IMU/LiDAR/Odometry 缓冲回调和
   `_on_synchronized_input(left, right, semantic)`，不复制 VO/LIO/地图逻辑。
5. 加入默认 `standard` 路径回归、自定义路径字段/错帧拒绝测试和同输入结果对账，
   再跑完整质量门禁和 Orin 长时实时压测。

因此本合集中，A 和 B 是已实现路径；C 包含上述修改契约和操作顺序。
明天取得报告后才能把 C 绑定到真实包名、类型和字段，不用占位字段冒充完成。

## 3. 事先必须拿到的东西

课题一和课题二至少需提供：

- 完整、锁定版本的 ROS 2 接口包，不是单独一个 `.msg` 文件；
- 它们依赖的其他自定义接口包；
- 实际话题名、频率、QoS、`frame_id` 和时间戳语义；
- 课题二原始语义 ID 表；
- 相机、IMU、LiDAR、车体的正式标定和坐标变换方向。

若目标机要离线部署，这些自定义包和递归依赖也必须一起拷贝；不能等到
Orin 上再联网补包。

## 4. 第一步：放入正式接口包

在 Orin 上，先按《课题三 Orin 原生离线部署步骤（fixed）》完成基础环境的
`install_native_orin.sh` 和运行时检查，再编译本适配层。适配包不会替代基础
ROS 2、课题三原有 5 个包或 RoMa 的安装。适配器现已是
`src/` 下默认不启动的第 6 个自有 ROS 2 包。

目录示例：

```text
t3_gridmap_v2_ws/
└── src/
    ├── t3_upstream_adapters/
    └── vendor_interfaces/
        ├── t1_interfaces/
        │   ├── package.xml
        │   ├── CMakeLists.txt
        │   └── msg/FusionData.msg
        └── t2_interfaces/
            ├── package.xml
            ├── CMakeLists.txt
            └── msg/SceneSemanticModel.msg
```

包名和消息名可以不同，不需要为模板改名。适配器在运行时按
`package/msg/Type` 动态加载它们。

然后执行：

```bash
cd /path/to/t3_gridmap_v2_ws
bash scripts/build_t1_t2_adapter_template.sh
```

脚本会自动选择当前 x86 ROS 环境或已安装的 Orin 原生离线环境。
普通开发机直接调用原 `scripts/build.sh`，从唯一 `src/` 编译 6 个自有包
和已放入的 vendor 包；已安装好基础运行时的 Orin 则只增量编译：

- `t3_upstream_adapters`；
- `vendor_interfaces/` 下找到的所有正式接口包。

两条路径都不会编译 RoMa 或 FAST-LIO2；适配器本身默认不启动，
所以编译它不会改变原有 VO 运行。

## 5. 第二步：一键收集实时接口

课题一、课题二节点启动后，确保两边一致：

```bash
export ROS_DOMAIN_ID=<项目确认的数值>
export ROS_LOCALHOST_ONLY=0
```

项目环境约定是 Cyclone DDS。如果上游实际不是 Cyclone DDS，先由集成方确认统一
RMW，不要为“看到话题”临时混用未验收配置。

不带话题参数时，脚本自动检查当前 ROS 图中除 `/rosout` 和
`/parameter_events` 之外的所有话题：

```bash
bash scripts/collect_t1_t2_interface_info.sh
```

如果只想检查指定话题：

```bash
bash scripts/collect_t1_t2_interface_info.sh \
  "$PWD/t1_t2_interface_report" \
  /actual/task1/topic \
  /actual/task2/topic
```

报告目录包含：

- `topic_list_with_types.txt`：话题和类型总表；
- `topics/*.txt`：每个话题的发布者、订阅者和详细 QoS；
- `interfaces/*.txt`：`ros2 interface show` 的完整字段定义；
- `environment.txt`：ROS 版本、RMW、主机和检查话题。

该命令不保存图像或点云载荷，不会因打印大数组拖慢实时系统。

性能边界也要说清楚：上游本来就是分开的标准话题时，关闭适配器，没有额外
转发。上游只有自定义聚合消息时，Python 适配器会多一次 DDS 解析和发布，
但课题一图像/点云字节不做采样或压缩。明天必须用第 9 节的频率和回执检查
Orin 实时能力。若转发成为明确瓶颈，后续优先让上游直接增发标准话题，或将
同一边界适配器改为 C++；不把自定义接口解析塞进 mapper 核心。

## 6. 先判断是否真的需要适配器

### 6.1 已经是分开的标准话题

如果课题一已发布 `sensor_msgs/msg/Image`、`sensor_msgs/msg/Imu`、
`sensor_msgs/msg/PointCloud2`，课题二已发布 `mono8 sensor_msgs/msg/Image`，
不做再封装，直接关闭对应适配器并传入真实话题名。

`standard` 的准确判定如下：

| 输入 | 话题的 ROS 2 类型 | 什么时候需要 |
|---|---|---|
| 左目图像 | `sensor_msgs/msg/Image` | 默认 RoMa VO 必须，发布端建议直接给 `bgr8` |
| 右目图像 | `sensor_msgs/msg/Image` | 默认 RoMa VO 必须，尺寸与左图一致 |
| 语义标签图 | `sensor_msgs/msg/Image` | 必须，`mono8`，尺寸与左图一致 |
| IMU | `sensor_msgs/msg/Imu` | `enable_imu:=true` 时需要，默认为 `true` |
| LiDAR | `sensor_msgs/msg/PointCloud2` | `enable_lidar_mapping:=true` 时才需要 |
| LIO 里程计 | `nav_msgs/msg/Odometry` | `localization_mode:=lio` 或 `vo_lio` 时才需要 |

“独立话题”是关键。即使一个自定义 `FusionData` 内部嵌套的叶子恰好是
`Image` / `Imu` / `PointCloud2`，只要上游只发布这一个聚合话题，它仍然不是
`standard`，应选 `t1` 让适配器拆包。

先用下列命令判断：

```bash
ros2 topic list -t
ros2 topic type /actual/left
ros2 topic type /actual/right
ros2 topic type /actual/semantic
ros2 topic info -v /actual/left
```

例如当现场已经有：

```text
/camera/left/image_raw       [sensor_msgs/msg/Image]
/camera/right/image_raw      [sensor_msgs/msg/Image]
/perception/semantic_labels  [sensor_msgs/msg/Image]
/imu/data                    [sensor_msgs/msg/Imu]
```

虽然话题名与工程默认值不同，它们仍属于 `standard`。

两个都不需要时：

```bash
bash scripts/run_t1_t2_adapter_template.sh \
  --profile standard \
  left_topic:=/actual/left \
  right_topic:=/actual/right \
  semantic_topic:=/actual/semantic \
  imu_topic:=/actual/imu \
  localization_mode:=vo \
  fusion_backend:=none \
  enable_lidar_mapping:=false \
  data_dir:=/path/to/calibration \
  output_dir:=/path/to/output
```

默认 RoMa VO 不用 LiDAR 和 `/Odometry`。如果现场没有 IMU，还要在启动命令中
明确传入 `enable_imu:=false`。只有真正启用 LiDAR 建图时才需要传
`lidar_topic:=...`；只有选择 `lio` / `vo_lio` 时才需要传
`lio_odometry_topic:=...`。

只有一边是自定义消息时，选 `--profile t1` 或 `--profile t2`。
被关闭节点的 `CHANGE_ME` 占位符不会被加载。如果不传 `--profile`，
为保持此专用脚本原行为，默认是 `t1_t2`；原项目的其他启动脚本仍不启动适配器。

### 6.2 是自定义聚合消息

使用第 7 节修改 YAML。这是模板主要解决的情况。

### 6.3 不在通用模板能力范围内

遇到下列任一情况，不要只靠 YAML 强行对接：

- 图像叶子字段是 `CompressedImage`、YUV 裸字节或厂商自定义图像；
- 点云不是 `PointCloud2`，或点时间定义不清楚；
- 课题二只给检测框、多边形、特征向量或帧间对应，没有稠密像素标签；
- 语义图是 RGB 颜色图，不是单通道类别 ID；
- 消息没有原始采集 `Header.stamp`；
- 外参、单位或坐标变换方向不确定。

这些情况需要针对正式协议增加一个明确转换函数，并使用实际数据测试。
不能猜一个字段就宣称已经适配。

## 7. 第三步：只修改一个 YAML

编辑：

```text
src/t3_upstream_adapters/config/interface_mapping.yaml
```

### 7.1 课题一配置

先查聚合话题的真实类型和消息定义：

```bash
ros2 topic type /Car/T1/fusion/multimodal
ros2 interface show t1_interfaces/msg/FusionData
```

如果 `FusionData` 里还引用了自定义 `StereoPair`，继续查：

```bash
ros2 interface show t1_interfaces/msg/StereoPair
```

假设正式消息定义是：

```text
# FusionData.msg
t1_interfaces/StereoPair stereo
sensor_msgs/Imu[] imu_samples
sensor_msgs/PointCloud2 dense_cloud
nav_msgs/Odometry lio_odom

# StereoPair.msg
sensor_msgs/Image left_image
sensor_msgs/Image right_image
```

那么“从 `FusionData` 根节点走到最终叶子”的路径就是：

```text
FusionData.stereo.left_image   -> left_image_field: "stereo.left_image"
FusionData.stereo.right_image  -> right_image_field: "stereo.right_image"
FusionData.imu_samples         -> imu_field: "imu_samples"
FusionData.dense_cloud         -> lidar_field: "dense_cloud"
FusionData.lio_odom            -> odometry_field: "lio_odom"
```

字段名大小写必须与 `.msg` 完全一致。不要把消息类型名、话题名或
`FusionData.` 本身写进字段路径。

针对上面这个消息，如果运行默认 RoMa VO、使用 IMU，但不启用 LiDAR
建图，应写成：

```yaml
t1_fusion_adapter:
  ros__parameters:
    input_message_type: "t1_interfaces/msg/FusionData"
    input_topic: "/Car/T1/fusion/multimodal"

    left_image_field: "stereo.left_image"
    right_image_field: "stereo.right_image"
    imu_field: "imu_samples"
    lidar_field: ""
    odometry_field: ""

    input_reliability: "reliable"
    required_image_encoding: "bgr8"
    maximum_stereo_skew_sec: 0.0
```

如果后面开启 LiDAR 建图，并且就是使用聚合消息内的这份点云，再改为：

```yaml
lidar_field: "dense_cloud"
```

如果 FAST-LIO2 已经在独立话题发布 `/Odometry`，仍然保持：

```yaml
odometry_field: ""
```

只有不存在独立 `/Odometry`，且明确要使用 `FusionData.lio_odom` 时才写：

```yaml
odometry_field: "lio_odom"
```

字段路径规则：

- 根字段：`left_image`；
- 嵌套字段：`stereo.left_image`；
- 固定数组元素：`cameras[0].image`；
- `imu_field` 可以是单个 `sensor_msgs/msg/Imu`，也可以是 `Imu[]`；
- 不使用 IMU、LiDAR 或聚合消息内的 Odometry 时，对应字段改为 `""`。

例如 `.msg` 是 `CameraFrame[] cameras`，而双方协议锁定了第 0 个恒为左目、
第 1 个恒为右目，可以写 `cameras[0].image` 和 `cameras[1].image`。
如果数组顺序会变，这个通用模板不能按 `camera_id` 搜索，需要增加专用转换代码。

`left_image_field` 和 `right_image_field` 的最终叶子必须是
`sensor_msgs/msg/Image`；`lidar_field` 的最终叶子必须是
`sensor_msgs/msg/PointCloud2`。
若课题一把 LIO 里程计也放在聚合消息内，`odometry_field` 叶子必须是
`nav_msgs/msg/Odometry`，适配后发布到 `/Odometry`。如果 FAST-LIO2 已经单独发布
`/Odometry`，必须保持 `odometry_field: ""`，避免同一话题出现两个不同来源。

这些 YAML 字段只决定适配器“拆出并转发什么”，不会自动切换算法模式。
例如配置 `lidar_field` 不等于开启 LiDAR 建图，仍需在 launch 中传
`enable_lidar_mapping:=true`；配置 `odometry_field` 也不等于切换到 LIO，仍需要
`localization_mode:=lio` 或 `vo_lio`。

`required_image_encoding` 应写实际编码。当上游是 `rgb8` 时可改为
`"rgb8"`，mapper 会通过 `cv_bridge` 显式转为 BGR8。不建议用空字符串绕过编码检查。

### 7.2 课题二配置

课题二有两种已实现方式，只能二选一。

方式 A：自定义消息内嵌了 `mono8`/`8UC1 Image`：

```yaml
t2_scene_adapter:
  ros__parameters:
    input_message_type: "t2_interfaces/msg/SceneSemanticModel"
    input_topic: "/T2/semantic/scene"
    label_image_field: "semantic.label_image"
    raw_label_data_field: ""
    raw_label_height_field: ""
    raw_label_width_field: ""
    raw_label_header_field: "header"
```

方式 B：自定义消息内是紧凑的 `uint8[]` 类别图：

```yaml
t2_scene_adapter:
  ros__parameters:
    label_image_field: ""
    raw_label_data_field: "semantic.labels"
    raw_label_height_field: "semantic.height"
    raw_label_width_field: "semantic.width"
    raw_label_header_field: "header"
```

语义图必须与左图同尺寸，并保留对应左图的原采集时间戳。建议明天确认
分辨率后填写：

```yaml
expected_width: 1920
expected_height: 1080
```

课题三的类别固定为：

| 课题三 ID | 含义 |
|---:|---|
| 0 | unknown |
| 1 | traversable_ground |
| 2 | shadow |
| 3 | bright_rock |
| 4 | rough_obstacle |

`class_lut` 中“索引=课题二 ID，数值=课题三 ID”。例如课题二的
`0=unknown, 1=rock, 2=ground, 3=shadow` 时：

```yaml
class_lut: [0, 4, 1, 2]
```

未列出的课题二 ID 会映射为 0，不会被误当作可通行地面。

### 7.3 QoS 怎么填

查看报告 `topics/*.txt` 的 publisher QoS：

- publisher 是 `RELIABLE`：`input_reliability: "reliable"`；
- publisher 是 `BEST_EFFORT`：`input_reliability: "best_effort"`。

Reliable 订阅者与 Best Effort 发布者不兼容。不要通过随意把所有话题改为
Best Effort 来解决无数据，大图像会失去完整性保障。

## 8. 第四步：启动

先用纯 VO 联调接口：

```bash
bash scripts/run_t1_t2_adapter_template.sh \
  data_dir:=/path/to/calibration \
  output_dir:=/path/to/output \
  localization_mode:=vo \
  fusion_backend:=none \
  enable_imu:=true \
  enable_lidar_mapping:=false \
  image_sync_mode:=exact
```

这个集成 launch 直接复用原 mapper 的全部参数定义，所以原有模式和开关没有被删减。
接口稳定后再按需启用：

- LiDAR 建图：`enable_lidar_mapping:=true lidar_mapping_mode:=direct`；
- LIO 定位：`localization_mode:=lio`，另外提供已确认的 `/Odometry`；
- VO/LIO 融合：`localization_mode:=vo_lio fusion_backend:=gtsam_gp`；
- 回环：使用原有 `enable_global_loop_closure` 及其配套参数。

这些模式仍受原工程的外参、时间、候选验证和后端保护约束；接入话题不等于
已经满足精度验收。

如果只想启动两个适配器，mapper 仍按原来脚本单独启动：

```bash
T3_ADAPTER_LAUNCH_FILE=adapters_only.launch.py \
  bash scripts/run_t1_t2_adapter_template.sh
```

## 9. 第五步：启动后必须检查

首先检查适配后类型：

```bash
ros2 topic type /Car/T1/fusion/stereo/left/image_raw
ros2 topic type /Car/T1/fusion/stereo/right/image_raw
ros2 topic type /Car/T2/semantic/label_image
ros2 topic type /Car/T1/fusion/imu/data
ros2 topic type /Car/T1/fusion/lidar/points
```

再检查实际频率：

```bash
ros2 topic hz /Car/T1/fusion/stereo/left/image_raw
ros2 topic hz /Car/T2/semantic/label_image
```

只打印 Header，不把整幅图像打到终端：

```bash
ros2 topic echo /Car/T1/fusion/stereo/left/image_raw --once --field header
ros2 topic echo /Car/T1/fusion/stereo/right/image_raw --once --field header
ros2 topic echo /Car/T2/semantic/label_image --once --field header
```

默认 `image_sync_mode:=exact` 时，左、右、语义三者的 `Header.stamp` 必须一致。
语义算法晚发布没关系，但必须保留它所对应原图的采集时间，不能填语义结果的
发布墙钟时间。

检查 mapper 是否完成同步输入回调：

```bash
ros2 topic hz /Car/T3/debug/input_processed
```

最后检查正式输出：

```bash
ros2 topic info --verbose /Car/T3/semantic/current_pose
ros2 topic info --verbose /Car/T3/mapping/semantic_grid
ros2 topic info --verbose /Car/T3/semantic/incremental_map
```

同时看适配器日志。它会明确打印接收、发布和拒绝计数；任何字段、类型、
时间或载荷错误都会拒绝整个聚合消息，不会只发布一半造成帧混用。

## 10. 常见问题

| 现象 | 先检查 | 正确处理 |
|---|---|---|
| 完全看不到对方话题 | `ROS_DOMAIN_ID`、网卡、多播、RMW | 先统一 ROS 网络，不改算法 |
| `Cannot load package/msg/Type` | 接口包是否完整编译/加载 | 补齐递归依赖后重跑 build 脚本 |
| 有 publisher 却没有数据 | QoS Reliability | 按实际 publisher 修改 YAML |
| `has no field` | YAML 字段路径 | 对照 `interfaces/*.txt` 逐层填写 |
| 适配器拒绝图像 | encoding、width/height、step、data 长度 | 修正上游协议或显式转换 |
| 三图有数据却不进 mapper | 三路 `Header.stamp` | 优先让课题二保留源图 stamp |
| 语义图尺寸不同 | 课题二输出分辨率 | 按正式像素对应规则处理，不盲目 resize |
| LiDAR 有数据但不入图 | frame、外参、时间模式、时差 | 按点云时间契约设置，不放宽门限掩盖错误 |

## 11. 明天最好留下的联调证据

联调完成后保留：

1. `collect_t1_t2_interface_info.sh` 产生的整个报告目录；
2. 最终 `interface_mapping.yaml`；
3. 启动日志，包含 adapter 拒绝计数和 mapper 同步计数；
4. 连续 30～60 s 的输入频率、`input_processed` 频率和正式输出频率；
5. 如果后续做精度验收，再保留 rosbag 和独立真值，不用本次“接口已通”代替精度结论。

## 12. 不要改的地方

明天为适配输入，默认不修改：

- `src/t3_semantic_mapping/` 中的 VO/LIO/融合/回环/地图核心；
- RoMa 源码和权重；
- 课题三对外地图和位姿消息定义；
- 适配器的稳定输出话题名，除非课题组已书面确认新协议；
- 任何 `Header.stamp` 和 `frame_id` 来“让它先跑”。

现有大型离线部署包无需重新下载。本次会单独生成小型增量包，
解压时与原包使用相同的 `t3_gridmap_v2_ws/` 顶层目录。它新增第 6 个包和
vendor 入口，并同步更新构建/离线校验脚本；不覆盖 RoMa 或 VO/LIO/地图算法文件。
