import importlib
import os

from aerodrome.core import *

# C++ envs
envs_dir = os.path.join(os.path.dirname(__file__), "simulator")

for root, dirs, files in os.walk(envs_dir):
    for file in files:
        if file.endswith(".pyd"):
            relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(__file__))[:-4].replace('\\', '.').split('.')[:-1]
            relative_path = '.'.join(relative_path)
            module_name = f"aerodrome.{relative_path}"
            importlib.import_module(module_name)


# Python envs
envs_dir = os.path.join(os.path.dirname(__file__), "envs")

for file in os.listdir(envs_dir):
    if file.endswith(".py"):
        module_name = f"aerodrome.envs.{file[:-3]}"
        importlib.import_module(module_name)