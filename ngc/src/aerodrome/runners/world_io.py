"""Host sampling boundary around an ordinary World.step, with replayable inputs."""
import jax


class WorldIO:
    """One owner loop. Device threads publish channels independently.

    selectors maps every entity ID to snapshots->entity inputs. Outputs is a
    tuple of (OutputChannel, (following_state, records)->values) bindings.
    No physical device handle or side effect enters the compiled World.
    """
    def __init__(self,world,channels,selectors, *, max_age_s=.5,outputs=(),advance=None):
        if set(selectors)!=set(world.entity_ids):
            raise ValueError("I/O selectors must cover every world entity")
        self.world,self.channels,self.selectors = world,dict(channels),dict(selectors)
        self.max_age_s,self.outputs = max_age_s,tuple(outputs)
        self.advance = jax.jit(world.step) if advance is None else advance
        self._sequences = {k:0 for k in channels}

    def sample_inputs(self):
        samples = {name:channel.read(previous_sequence=self._sequences[name],max_age_s=self.max_age_s)
                   for name,channel in self.channels.items()}
        for name,sample in samples.items():
            self._sequences[name] = int(sample["io_sequence"])
        return self.world.pack({name:select(samples) for name,select in self.selectors.items()})

    def step(self,state,parameters):
        """Returns following, record, actual_inputs (record these for replay).

        Inputs hold through all ticks of this step. For faster input sampling,
        choose a shorter step or use a hybrid host source module.
        """
        inputs = self.sample_inputs()
        following,record = self.advance(state,inputs,parameters)
        for channel,select in self.outputs:
            channel.publish(select(following,record),tick=following.tick,
                            time_s=following.tick*self.world.spec.schedule.physics_dt_s)
        return following,record,inputs
