#!/usr/bin/env python3
# ============================================================
# lunar_car_gui 包安装脚本
# 可执行入口: lunar_car_gui (lunar_car_gui_node.py 的 main)
# ============================================================
from setuptools import setup

package_name = 'lunar_car_gui'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yanfa',
    maintainer_email='yanfa@todo.todo',
    description='月球车控制 GUI: 遥测显示与速度/模式控制',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lunar_car_gui = lunar_car_gui.lunar_car_gui_node:main',
        ],
    },
)
