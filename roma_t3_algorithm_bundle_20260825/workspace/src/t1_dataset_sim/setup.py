from glob import glob
from setuptools import find_packages, setup

package_name = "t1_dataset_sim"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="guderian",
    maintainer_email="3127775604@qq.com",
    description="Dataset-backed Task 1 sensor publisher.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dataset_sensor_publisher = "
            "t1_dataset_sim.dataset_sensor_publisher:main",
        ],
    },
)
