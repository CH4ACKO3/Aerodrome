import jax
import jax.numpy as jnp
import numpy as np
import pytest
gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from gymnasium.error import ResetNeeded
from gym_world import make_env
from aerodrome.models.rigid_body import BodyLoads
from aerodrome.adapters.gymnasium import WorldEnv
from aerodrome.runners.episodes import Task,Outcome


def test_gym_checker_and_lifecycle():
    env = make_env(max_episode_steps=2,render_mode="ansi")
    with pytest.raises(ResetNeeded): env.step(np.zeros(3,np.float32))
    check_env(env,skip_render_check=True)
    env.reset(seed=8)
    assert env.step(np.zeros(3,np.float32))[2:4]==(False,False)
    assert env.step(np.zeros(3,np.float32))[2:4]==(False,True)
    assert "tick=4" in env.render()
    with pytest.raises(ResetNeeded): env.step(np.zeros(3,np.float32))
    env.close(); env.close()
    with pytest.raises(RuntimeError): env.reset()


def test_seed_options_and_no_observation_alias():
    env = make_env()
    first,_ = env.reset(seed=10)
    again,_ = env.reset(seed=10)
    np.testing.assert_array_equal(first,again)
    fresh,_ = env.reset()
    assert not np.array_equal(first,fresh)
    obs,_ = env.reset(options={"velocity_body_m_s":[1.,2.,3.]})
    obs[:] = 99
    np.testing.assert_array_equal(env.state.entities[0].velocity_body_m_s,[1.,2.,3.])


@pytest.mark.parametrize("jit",[True,False])
def test_matches_world_and_native_transition(jit):
    env = make_env(jit=jit)
    obs,_ = env.reset(seed=123)
    action = np.asarray([.1,-.2,.3],np.float32)
    previous = env.state
    expected,_ = env.world.step(previous,(BodyLoads(jnp.asarray(action)*1000.,jnp.zeros(3,jnp.float32)),),env.parameters)
    native = env.transition(previous,jnp.asarray(obs),jnp.asarray(action),env.parameters,env.task_parameters)
    actual,reward,terminated,truncated,info = env.step(action)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(env.state),strict=True):
        np.testing.assert_allclose(a,b,atol=1e-6)
    np.testing.assert_allclose(actual,native[1],atol=1e-6)
    assert reward==pytest.approx(float(native[2]))
    assert info["tick"]==env.world.spec.ticks_per_step and not truncated


def test_terminal_at_time_limit():
    env = make_env(max_episode_steps=1)
    env.reset(options={"velocity_body_m_s":[2.,0.,0.]})
    assert env.step(np.zeros(3,np.float32))[2:4]==(True,False)


def test_standard_vector_wrapper():
    env = gym.vector.SyncVectorEnv([lambda:make_env(max_episode_steps=2) for _ in range(2)])
    try:
        obs,info = env.reset(seed=[1,2])
        assert obs.shape==(2,3)
        obs,reward,terminated,truncated,info = env.step(np.zeros((2,3),np.float32))
        assert reward.shape==(2,) and not truncated.any()
        assert env.step(np.zeros((2,3),np.float32))[3].all()
    finally:
        env.close()


def test_dict_observation_discrete_action_and_failure():
    base = make_env()
    base.reset(seed=0)
    initial = dict(zip(base.world.entity_ids, base.state.entities, strict=True))
    task = Task(lambda s,p:{"velocity":s.entities[0].velocity_body_m_s,"mode":jnp.asarray(0)},
                lambda o,a,n,p:Outcome(jnp.asarray(-1.,jnp.float32),jnp.asarray(False)),2)
    env = WorldEnv(base.world,initial_conditions=initial,parameters=base.parameters,
                   task=task,action_space=gym.spaces.Discrete(2),
                   observation_space=gym.spaces.Dict({"velocity":base.observation_space,"mode":gym.spaces.Discrete(1)}),
                   action_to_inputs=lambda a,s,p:(BodyLoads(jnp.full(3,a,dtype=jnp.float32),jnp.zeros(3,jnp.float32)),))
    check_env(env,skip_render_check=True)
    env.reset(seed=2)
    assert env.step(1)[0]["mode"]==0
    env.transition = lambda *args:(_ for _ in ()).throw(FloatingPointError("failed physics"))
    with pytest.raises(FloatingPointError): env.step(0)
    with pytest.raises(ResetNeeded): env.step(0)


def test_native_transition_vmap_and_gradient():
    env = make_env()
    obs,_ = env.reset(seed=0)
    batch_state = jax.tree.map(lambda x:jnp.stack([x,x]),env.state)
    actions = jnp.zeros((2,3),jnp.float32)
    _,following,rewards,_ = jax.jit(jax.vmap(env.transition,in_axes=(0,0,0,None,None)))(
        batch_state,jnp.stack([obs,obs]),actions,env.parameters,env.task_parameters)
    assert following.shape==(2,3) and rewards.shape==(2,)
    gradient = jax.grad(lambda a:env.transition(env.state,jnp.asarray(obs),a,env.parameters,env.task_parameters)[2])(actions[0])
    assert np.all(np.isfinite(gradient)) and np.linalg.norm(gradient)>0
