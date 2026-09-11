"""Optional mutable convenience API for notebooks; not used under JIT."""
class SimulationSession:
    def __init__(self, world, state, parameters):
        self.world = world
        self.state = state
        self.parameters = parameters

    def step(self, inputs):
        self.state, trace = self.world.step(self.state, inputs, self.parameters)
        return trace
