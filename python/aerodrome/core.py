"""Core API for environments"""

from abc import ABC
from aerodrome.registration import registry

class Env(ABC):
    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
    
def make(id: str, arg):
    """Create an environment by ID"""
    entry_point: str = registry.get(id)
    if entry_point is None:
        raise ValueError(f"Environment '{id}' not found")
    
    module_name, class_name = entry_point.split(":")
    module = __import__(module_name, fromlist=[class_name])
    env_class = getattr(module, class_name)
    return env_class(arg)