from glob import glob

from setuptools import find_packages, setup


package_name = "t3_upstream_adapters"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="guderian",
    maintainer_email="3127775604@qq.com",
    description="Configurable Task 1/Task 2 boundary adapters.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "t1_fusion_adapter = "
            "t3_upstream_adapters.t1_fusion_adapter_node:main",
            "t2_scene_adapter = "
            "t3_upstream_adapters.t2_scene_adapter_node:main",
        ],
    },
)
