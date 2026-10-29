from glob import glob
from setuptools import find_packages, setup

package_name = "t3_semantic_mapping"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="guderian",
    maintainer_email="3127775604@qq.com",
    description="RoMa semantic 2.5D GridMap node.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "roma_semantic_mapper = "
            "t3_semantic_mapping.roma_semantic_mapper_node:main",
            "visual_lidar_cloud_mapper = "
            "t3_semantic_mapping.visual_lidar_cloud_node:main",
        ],
    },
)
