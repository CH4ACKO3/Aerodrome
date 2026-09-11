"""Algorithm-independent Gymnasium wrapper of a three-axis rigid-body task."""
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from aerodrome.adapters.gymnasium import WorldEnv,ResetSpec
from aerodrome.configuration import build_experiment
from aerodrome.models.rigid_body import BodyLoads
from aerodrome.runners.episodes import Task,Outcome


def make_env(*,max_episode_steps=200,render_mode=None,jit=True):
    built = build_experiment({"runtime":{"dtype":"float32"},"entities":[{
        "id":"body","model":{"kind":"rigid_body","version":"1","parameters":{
            "velocity_body_m_s":[0.,0.,0.],"omega_body_rad_s":[0.,0.,0.]}}}]},base_dir=".")
    initial = built.entities[0].initial
    target = jnp.asarray([2.,0.,0.],dtype=jnp.float32)

    def observe(state,parameters):
        return state.entities[0].velocity_body_m_s

    def evaluate(observation,action,following,parameters):
        error = following-parameters
        return Outcome(-jnp.sum(error**2)*built.world.step_dt_s-.01*jnp.sum(action**2),
                       jnp.linalg.norm(error)<.05)

    def inputs(action,state,parameters):
        return (BodyLoads(action*1000.,jnp.zeros(3,dtype=action.dtype)),)

    def reset_factory(rng,options):
        if set(options)-{"velocity_body_m_s"}:
            raise ValueError("unknown reset option")
        velocity = np.asarray(options.get("velocity_body_m_s",rng.uniform(-.1,.1,3)),dtype=np.float32)
        if velocity.shape!=(3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_body_m_s must be a finite length-three vector")
        return ResetSpec({"body":initial._replace(velocity_body_m_s=jnp.asarray(velocity))},built.parameters,target)

    return WorldEnv(built.world,initial_conditions={"body":initial},parameters=built.parameters,
                    task=Task(observe,evaluate,max_episode_steps),task_parameters=target,
                    action_space=spaces.Box(-1.,1.,shape=(3,),dtype=np.float32),
                    observation_space=spaces.Box(-1000.,1000.,shape=(3,),dtype=np.float32),
                    action_to_inputs=inputs,reset_factory=reset_factory,jit=jit,render_mode=render_mode)


if __name__=="__main__":
    from gymnasium.utils.env_checker import check_env
    with make_env(render_mode="ansi") as env:
        check_env(env,skip_render_check=True)
        observation,info = env.reset(seed=42)
        while True:
            action = np.clip(np.asarray([2.,0.,0.],dtype=np.float32)-observation,-1.,1.)
            observation,reward,terminated,truncated,info = env.step(action)
            if terminated or truncated:
                break
        print(env.render())
        print(dict(info,terminated=terminated,truncated=truncated))
