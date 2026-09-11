import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from aerodrome.models.f16_longitudinal import load_tables,interpolate,coefficients,rhs,Airframe
from aerodrome.core.integrators import rk4


@pytest.fixture(scope="module")
def experiment():
    pytest.importorskip("control")
    from f16_level_flight import build_experiment
    return build_experiment()


def test_interpolation_nodes_offgrid_and_outside():
    tables = load_tables()
    # Raw HDF5 scalar offsets a=4,beta=9,elevator=2, MATLAB column-major indexing.
    np.testing.assert_allclose([tables[k][4,2] for k in ("Cx","Cz","Cm")],[-.0489,-.025,-.0598],atol=1e-15)
    axes = (tables["alpha1"],tables["dh1"])
    a,e = np.meshgrid(np.asarray(axes[0]),np.asarray(axes[1]),indexing="ij")
    points = np.stack((a.ravel(),e.ravel()),axis=1)
    for name in ("Cx","Cz","Cm"):
        fn = jax.jit(jax.vmap(lambda point:interpolate(axes,tables[name],point)))
        np.testing.assert_allclose(fn(points),np.asarray(tables[name]).ravel(),atol=1e-14)
        queries = np.array([[3.2,-1.3],[9.,7.],[-19.,-24.],[89.,24.]])
        reference = RegularGridInterpolator(tuple(map(np.asarray,axes)),np.asarray(tables[name]))
        np.testing.assert_allclose(fn(queries),reference(queries),atol=1e-14)
        assert np.all(np.isnan(fn(np.array([[-21.,0.],[0.,26.]]))))


def test_data_integrity_refuses_modified_asset(tmp_path):
    import aerodrome.models.f16_longitudinal as model
    source = Path(model.__file__).parent/"data"/"f16"
    (tmp_path/"manifest.json").write_text((source/"manifest.json").read_text())
    (tmp_path/"longitudinal.npz").write_bytes(b"invalid asset")
    with pytest.raises(ValueError,match="integrity"):
        load_tables(tmp_path)


def test_force_equations_match_body_axis_projection(experiment):
    tables = experiment[2].resources
    p = Airframe()
    x = jnp.array([157.,.08,.02,.1,3000.])
    u = jnp.array([-.03,9000.])
    from aerodrome.models.f16_longitudinal import density
    cx,cz,cm = np.asarray(coefficients(x[0],x[1],x[2],u[0],tables))
    force = .5*float(density(x[4]))*float(x[0])**2*p.area
    ub,wb = float(x[0]*jnp.cos(x[1])),float(x[0]*jnp.sin(x[1]))
    du = -float(x[2])*wb-p.gravity*np.sin(float(x[3]))+(force*cx+float(u[1]))/p.mass
    dw = float(x[2])*ub+p.gravity*np.cos(float(x[3]))+force*cz/p.mass
    expected = [(ub*du+wb*dw)/float(x[0]),(ub*dw-wb*du)/float(x[0])**2,
                force*p.chord*cm/p.iyy,float(x[2]),ub*np.sin(float(x[3]))-wb*np.cos(float(x[3]))]
    np.testing.assert_allclose(rhs(x,u,tables),expected,atol=1e-12)


def test_trim_linearization_and_riccati_residual(experiment):
    _,_,parameters,x,u,K,info = experiment
    tables = parameters.resources
    assert np.max(np.abs(rhs(x,u,tables))) < 1e-9
    for arg,steps in ((0,[1e-3,1e-6,1e-6,1e-6,1e-2]),(1,[1e-6,.1])):
        target = x if arg == 0 else u
        columns = []
        for i,delta in enumerate(steps):
            shift = jnp.eye(len(target))[i]*delta
            plus = rhs(x+shift,u,tables) if arg == 0 else rhs(x,u+shift,tables)
            minus = rhs(x-shift,u,tables) if arg == 0 else rhs(x,u-shift,tables)
            columns.append(np.asarray((plus-minus)/(2*delta)))
        np.testing.assert_allclose(np.stack(columns,axis=1),info["A" if arg == 0 else "B"],rtol=1e-5,atol=1e-8)
    F,G,P = (np.asarray(info[k]) for k in ("normalized_F","normalized_G","normalized_riccati"))
    Q,R = np.diag([2.,1.,.2,.5,2.]),np.diag([25.,2.])
    residual = F.T@P@F-P-F.T@P@G@np.linalg.solve(R+G.T@P@G,G.T@P@F)+Q
    assert np.linalg.norm(residual)/np.linalg.norm(P) < 1e-10
    assert max(abs(info["closed_loop_poles"])) < 1


def test_nonlinear_closed_loop_recovery_and_step_halving(experiment):
    batch,state,params,trim,_,_,_ = experiment
    final,_ = jax.jit(lambda s:batch.rollout_constant(s,(trim,),params,steps=3000,record=None))(state)
    error = np.asarray(final.world.entities[0].aircraft)-np.asarray(trim)
    assert np.all(np.isfinite(error))
    assert np.max(np.abs(error)/np.array([.1,1e-3,1e-3,1e-3,.2])) < 1
    from aerodrome.composition import EntitySpec,WorldSpec,build_world
    from aerodrome.composition.f16_longitudinal import F16LongitudinalAssembly
    from aerodrome.core.clock import Schedule
    from aerodrome.runners.batch import BatchedWorld
    fine_world = build_world(WorldSpec((EntitySpec("f16",F16LongitudinalAssembly()),),Schedule(physics_dt_s=.005,control_every=4),ticks_per_step=4))
    fine = BatchedWorld(fine_world,initial_axes=0,input_axes=None)
    coarse,_ = jax.jit(lambda s:batch.rollout_constant(s,(trim,),params,steps=500,record=None))(state)
    refined,_ = jax.jit(lambda s:fine.rollout_constant(s,(trim,),params,steps=500,record=None))(state)
    np.testing.assert_allclose(coarse.world.entities[0].aircraft,refined.world.entities[0].aircraft,rtol=1e-7,atol=2e-5)
