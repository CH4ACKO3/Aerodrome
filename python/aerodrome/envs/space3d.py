from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.Core.envs import Space3D
from aerodrome.simulator.Core.objects import Object3D

class Space3D_control(Env):
    def __init__(self):
        self.env = Space3D()

    def add_object(self, object: Object3D):
        self.env.add_object(object)

    def reset(self):
        output_dict = self.env.reset()
        return output_dict

    def step(self, action: dict):
        output_dict = self.env.step(action)
        return output_dict
    
register("space3d-v0", "aerodrome.envs.space3d:Space3D")