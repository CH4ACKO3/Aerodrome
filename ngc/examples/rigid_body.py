"""Two freely rotating bodies, two attitude implementations, native batch rollout."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.models.rigid_body import RigidBody6DoF, BodyLoads, mass_properties
from aerodrome.composition.rigid_body import RigidBodyAssembly
from aerodrome.composition import EntitySpec, WorldSpec, build_world
from aerodrome.core.clock import Schedule
from aerodrome.runners.batch import BatchedWorld


def main():
    jax.config.update("jax_enable_x64", True)
    folder = Path("artifacts/rigid_body")
    folder.mkdir(parents=True, exist_ok=True)
    final, arrays, report = {}, {}, {}
    for mode in ("quaternion", "euler321"):
        model = RigidBody6DoF(mode)
        world = build_world(WorldSpec((EntitySpec("body", RigidBodyAssembly(model)),),
                                     Schedule(physics_dt_s=.01), ticks_per_step=2))
        initial = [model.initialize(euler_rad=[.2,-.3,.4], velocity_body_m_s=[4,2,-1],
                                   omega_body_rad_s=[.12,rate,-.1]) for rate in (.2,-.15)]
        stacked = jax.tree.map(lambda *x:jnp.stack(x), *initial)
        p = mass_properties(4., [[2,.1,-.2],[.1,2.8,.15],[-.2,.15,3.4]], [0,0,0])
        parameters = world.parameters({"body":p})
        batch = BatchedWorld(world,initial_axes=0,input_axes=None)
        state = batch.reset(42,[10,20],{"body":stacked})
        loads = (BodyLoads(jnp.zeros(3),jnp.zeros(3)),)
        def record(state,trace):
            return state.entities[0]  # End of each decision interval, [T,B,...].
        result,trace = jax.jit(lambda s,p:batch.rollout_constant(s,loads,p,steps=500,record=record))(state,parameters)
        final[mode] = jax.vmap(model.rotation_matrix)(result.world.entities[0])
        rotations = jax.vmap(jax.vmap(model.rotation_matrix))(trace)
        H = jnp.einsum("tbij,tbj->tbi",rotations,trace.omega_body_rad_s@p.inertia_body_kg_m2.T)
        R0 = jax.vmap(model.rotation_matrix)(stacked)
        H0 = jnp.einsum("bij,bj->bi",R0,stacked.omega_body_rad_s@p.inertia_body_kg_m2.T)
        error = float(jnp.max(jnp.abs(H-H0)))
        assert error < 1e-9
        report[mode] = {"max_angular_momentum_error_kg_m2_s":error}
        for field,value in trace._asdict().items():
            arrays[f"{mode}_{field}"] = np.asarray(value)
    difference = float(jnp.max(jnp.abs(final["quaternion"]-final["euler321"])))
    assert difference < 1e-9
    report.update(duration_s=10., worlds=2, backend=jax.default_backend(),
                  max_attitude_matrix_difference=difference)
    np.savez_compressed(folder/"trajectory.npz",time_s=np.arange(1,501)*.02,**arrays)
    (folder/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
