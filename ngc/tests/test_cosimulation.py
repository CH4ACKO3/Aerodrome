from dataclasses import replace
import numpy as np
import pytest
from aerodrome.adapters.external import Port, ExternalSample, Capabilities
from aerodrome.adapters.functional import FunctionalComponent
from aerodrome.runners.cosimulation import CoSimulationRunner, Connection


def component(initial):
    def reset(parameters):
        return np.asarray(initial), {"out": np.asarray(initial)}
    def transition(state, inputs, parameters, dt):
        state = state + inputs["in"] * dt
        return state, {"out": state}
    return FunctionalComponent((Port("in", "1", (), "scalar"),),
                               (Port("out", "1", (), "scalar"),), reset, transition)


def coupled(reverse=False, dt=0.1):
    components = {"a": component(1.), "b": component(2.)}
    if reverse:
        components = dict(reversed(tuple(components.items())))
    return CoSimulationRunner(components, (Connection(("a", "out"), ("b", "in")),
                                          Connection(("b", "out"), ("a", "in"))), communication_dt_s=dt)


def test_jacobi_order_independence_and_reset():
    for reverse in (False, True):
        with coupled(reverse) as runner:
            initial = runner.initialize({"a": {}, "b": {}})
            first = runner.step()
            assert first["a"].values["out"] == 1.2
            assert first["b"].values["out"] == 2.1
            assert first["a"].time_s == 0.1
            assert initial["a"].values["out"] == 1.  # old boundary did not mutate
            runner.reset()
            assert runner.step()["a"].values["out"] == 1.2
        assert all(not c._ready for c in runner.components.values())


def test_zoh_coupling_converges_with_communication_step():
    def error(dt):
        with coupled(dt=dt) as runner:
            runner.initialize({"a": {}, "b": {}})
            for _ in range(round(1 / dt)):
                samples = runner.step()
            # a'=b, b'=a -> a(t)=cosh(t)+2*sinh(t)
            return abs(samples["a"].values["out"] - (np.cosh(1) + 2*np.sinh(1)))
    assert error(0.1) / error(0.05) > 1.8


def test_wrong_time_poisoned_until_reset_and_always_close():
    with coupled() as runner:
        runner.initialize({"a": {}, "b": {}})
        c = runner.components["b"]
        advance = c.advance_to
        def late(t, inputs):
            sample = advance(t, inputs)
            return ExternalSample(t + 0.01, sample.values)
        c.advance_to = late
        with pytest.raises(ValueError, match="stopped at"):
            runner.step()
        with pytest.raises(RuntimeError, match="requires reset"):
            runner.step()
        c.advance_to = advance
        runner.reset()
        assert runner.step()["a"].values["out"] == 1.2
    assert not c._ready


def test_port_validation_and_explicit_external_inputs():
    a, b = component(1.), component(2.)
    b.inputs = (replace(b.inputs[0], unit="rad"),)
    with pytest.raises(ValueError, match="incompatible"):
        CoSimulationRunner({"a": a, "b": b}, (Connection(("a", "out"), ("b", "in")),), communication_dt_s=0.1)
    with pytest.raises(ValueError, match="every input"):
        CoSimulationRunner({"a": a}, (), communication_dt_s=0.1)
    with CoSimulationRunner({"a": a}, (), communication_dt_s=0.1, external_inputs=(("a", "in"),)) as runner:
        runner.initialize({"a": {}})
        with pytest.raises(ValueError, match="shape/dtype"):
            runner.step({("a", "in"): np.asarray([1.])})
        assert runner.step({("a", "in"): np.asarray(3.)})["a"].values["out"] == 1.3
    b.capabilities = Capabilities(True, False, True, True)
    with pytest.raises(ValueError, match="feedthrough"):
        CoSimulationRunner({"b": b}, (), communication_dt_s=0.1)


def test_jax_engine_can_be_replaced_by_external_protocol_fixture():
    from hybrid_propulsion import run, jax_engine, ExternalEngineFixture
    np.testing.assert_allclose(run(jax_engine, steps=12), run(ExternalEngineFixture, steps=12),
                               rtol=1e-12, atol=1e-12)


def test_initialization_failure_closes_all_partitions():
    runner = coupled()
    def fail(time, parameters):
        raise RuntimeError("external solver failed")
    runner.components["b"].initialize = fail
    with pytest.raises(RuntimeError, match="external solver"):
        runner.initialize({"a": {}, "b": {}})
    assert all(not c._ready for c in runner.components.values())
