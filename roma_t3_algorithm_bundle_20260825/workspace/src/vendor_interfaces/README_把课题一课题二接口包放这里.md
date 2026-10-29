# 课题一/课题二正式 ROS 2 接口包放置位置

把对方提供的完整 ROS 2 消息包目录放到本目录，保留包本身的
`package.xml`、`CMakeLists.txt`、`msg/`以及它的自定义依赖包。示例：

```text
vendor_interfaces/
├── t1_interfaces/
│   ├── package.xml
│   ├── CMakeLists.txt
│   └── msg/FusionData.msg
├── t2_interfaces/
│   ├── package.xml
│   ├── CMakeLists.txt
│   └── msg/SceneSemanticModel.msg
└── README_把课题一课题二接口包放这里.md
```

不要只复制一个 `.msg` 文件，也不要在课题三内自造同名消息。先确认
对方的锁定版本和递归依赖，再执行：

```bash
bash scripts/build_t1_t2_adapter_template.sh
```

完整修改步骤见工作空间根目录的
`课题一课题二接口快速适配模板说明_fixed.md`。
