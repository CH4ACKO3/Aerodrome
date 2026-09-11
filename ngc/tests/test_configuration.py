from dataclasses import asdict,dataclass
from hashlib import sha256
from pathlib import Path
import json
import sys
import jax
import numpy as np
import pytest
from aerodrome.configuration import load_config,validate_config,build_experiment,run_experiment,builtin_registry
from aerodrome.configuration.api import set_override
from aerodrome.configuration.cli import main

CONF = Path(__file__).parents[1]/"src/aerodrome/configuration/conf/experiment.yaml"


def test_compose_and_overlay():
    c = load_config(CONF,["runtime.steps=3","entities.0.model.parameters.mass_kg=1200"],
                    overlays=["environment/earth.yaml","renderer/headless.yaml"])
    assert c["environment"]["parameters"]["gravity_ned_m_s2"][2]>9
    assert c["runtime"]["steps"]==3
    assert c["entities"][0]["model"]["parameters"]["mass_kg"]==1200


@pytest.mark.parametrize("override",["runtime.steps=true","runtime.steps=0","runtime.dtype=half",
    "seed=-1","clock.physics_dt_s=0","runtime.jit='false'","entities.0.model.parameters.typo=3",
    "entities.-1.id=x","runtime.steps=.nan","entities.0.model.version='missing'"])
def test_invalid_override(override):
    with pytest.raises(ValueError):
        load_config(CONF,[override])


def test_include_errors(tmp_path):
    a = tmp_path/"a.yaml"
    a.write_text("includes: [a.yaml]\n")
    with pytest.raises(ValueError,match="cyclic"):
        load_config(a)
    a.write_text("seed: 1\nseed: 2\n")
    with pytest.raises(ValueError,match="duplicate"):
        load_config(a)
    a.write_text("runtime: {typo: 1}\n")
    with pytest.raises(ValueError):
        load_config(a)


def test_assets_and_factory(tmp_path):
    data = b"teaching model data"
    (tmp_path/"data.bin").write_bytes(data)
    c = load_config(CONF)
    c["assets"] = {"table":dict(path="data.bin",sha256=sha256(data).hexdigest(),source="test",license="CC0")}
    registry = builtin_registry()
    @dataclass
    class Options:
        scale: float = 1.
    def gravity(options,context):
        assert context["assets"]["table"]==data
        return context["vector"]([0,0,options.scale],3,"gravity")
    registry.register("environment","custom","1",Options,gravity)
    c["environment"] = dict(kind="custom",version="1",parameters={"scale":2.})
    built = build_experiment(c,base_dir=tmp_path,registry=registry)
    assert built.asset_manifest["table"]["path"]==str(tmp_path/"data.bin")
    (tmp_path/"data.bin").write_bytes(b"changed")
    with pytest.raises(ValueError):
        build_experiment(c,base_dir=tmp_path,registry=registry)


def test_run_matches_world_and_snapshot(tmp_path):
    c = load_config(CONF,["runtime.steps=3","runtime.chunk_steps=2","renderer.parameters.every_steps=1"],overlays=["renderer/headless.yaml"])
    built = build_experiment(c,base_dir=CONF.parent)
    state = built.state
    for _ in range(3):
        state,_ = built.world.step(state,built.inputs,built.parameters)
    folder = run_experiment(c,base_dir=CONF.parent,output_base=tmp_path)
    status = json.loads((folder/"status.json").read_text())
    assert status["status"]=="complete" and status["tick"]==6 and status["rendered_frames"]==3
    with np.load(folder/"final.npz") as saved:
        for i,leaf in enumerate(jax.tree.leaves(state)):
            np.testing.assert_allclose(saved[f"leaf_{i}"],leaf,atol=1e-10)
    assert len(list(folder.glob("trace_*.npz")))==3
    for name in ("requested.json","resolved.yaml","resolved.json","provenance.json","assets.json","device.json"):
        assert (folder/name).is_file()
    set_override(c,"runtime.trace=false")
    other = run_experiment(c,base_dir=CONF.parent,output_base=tmp_path)
    assert other!=folder and not list(other.glob("trace_*.npz"))


def test_failed_run_status(tmp_path):
    c = load_config(CONF,["entities.0.model.parameters.mass_kg=-1"])
    with pytest.raises(ValueError):
        run_experiment(c,base_dir=CONF.parent,output_base=tmp_path)
    status = next(tmp_path.rglob("status.json"))
    assert json.loads(status.read_text())["status"]=="failed"


def test_float32_multi_entity():
    c = load_config(CONF,["runtime.dtype=float32"])
    c["entities"].append(dict(id="second",model=dict(kind="rigid_body",version="1",parameters={})))
    built = build_experiment(c,base_dir=CONF.parent)
    assert len(built.state.entities)==2
    assert all(x.dtype==np.float32 for x in jax.tree.leaves(built.parameters) if x.dtype.kind=="f")


def test_cli_sweep(tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path)
    main(["--config",str(CONF),"--sweep","seed=[1,2]","runtime.steps=1"])
    assert capsys.readouterr().out.count("Run completed:")==2
    assert len(list(tmp_path.rglob("status.json")))==2
    main(["--config",str(CONF),"--validate"])
    assert "schema_version: 1" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--sweep","seed="+json.dumps(list(range(257)))])


def test_free_threaded_imports():
    if sysconfig_free_threaded():
        assert not sys._is_gil_enabled()


def sysconfig_free_threaded():
    import sysconfig
    return sysconfig.get_config_var("Py_GIL_DISABLED")
