import importlib
import os

from aerodrome.core import *

# C++ envs
envs_dir = os.path.join(os.path.dirname(__file__), "simulator")

any_failed = True
failed_cnt = 0

while any_failed and failed_cnt < 10:
    any_failed = False
    for root, dirs, files in os.walk(envs_dir):
        for file in files:
            if file.endswith(".pyd"):
                relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(__file__))[:-4].replace('\\', '.').split('.')[:-1]
                relative_path = '.'.join(relative_path)
                
                module_name = f"aerodrome.{relative_path}"

                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    any_failed = True
                    failed_cnt += 1

            if file.endswith(".so"):
                relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(__file__))[:-4].replace('\\', '.').split('.')[0].replace('/', '.')
                
                module_name = f"aerodrome.{relative_path}"

                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    any_failed = True
                    failed_cnt += 1

if any_failed:
    raise Exception("Failed to import some modules")

# Python envs
envs_dir = os.path.join(os.path.dirname(__file__), "envs")

for file in os.listdir(envs_dir):
    if file.endswith(".py"):
        module_name = f"aerodrome.envs.{file[:-3]}"
        importlib.import_module(module_name)