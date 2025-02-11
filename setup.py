import os
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# 扩展模块设置
setup(
    name="aerodrome",
    version="0.1.0",
    description="A lightweight aircraft reinforcement learning environment",
    author="Chack",
    author_email="ch4acko3@outlook.com",
    license="MIT",
    packages=find_packages(where="python"),  # 查找 Python 包
    package_dir={"": "python"},  # 设置包根目录
    include_package_data=True,
    install_requires=[],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
