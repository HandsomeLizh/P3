from setuptools import find_packages, setup

package_name = "t2_basic_semantics"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="guderian",
    maintainer_email="3127775604@qq.com",
    description="Simple semantic segmentation for standalone Task 3 testing.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "basic_semantic_node = "
            "t2_basic_semantics.basic_semantic_node:main",
            "generate_semantic_dataset = "
            "t2_basic_semantics.generate_semantic_dataset:main",
        ],
    },
)
