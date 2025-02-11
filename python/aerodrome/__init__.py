import importlib
import os

from aerodrome.core import *

envs_dir = os.path.join(os.path.dirname(__file__), "envs")

for file in os.listdir(envs_dir):
    if file.endswith(".py"):
        module_name = f"aerodrome.envs.{file[:-3]}"
        importlib.import_module(module_name)