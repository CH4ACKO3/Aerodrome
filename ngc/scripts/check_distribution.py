"""Run with python -I in a wheel-installed environment, not an editable checkout."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def inspect_distribution():
    import aerodrome
    import jax
    from aerodrome.configuration import load_config,build_experiment
    from aerodrome.configuration import api
    from aerodrome.models.f16_longitudinal import load_tables
    package_path = Path(aerodrome.__file__).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"expected installed wheel, imported source at {package_path}")
    jax.config.update("jax_enable_x64",True)
    config = load_config(Path(api.__file__).parent/"conf/experiment.yaml")
    built = build_experiment(config,base_dir=Path(api.__file__).parent/"conf")
    state,_ = jax.block_until_ready(jax.jit(built.world.step)(built.state,built.inputs,built.parameters))
    assert int(state.tick)==config["clock"]["ticks_per_step"]
    tables = load_tables()
    if not tables:
        raise RuntimeError("F16 data absent from wheel")
    with tempfile.TemporaryDirectory(prefix="aerodrome-wheel-check-") as cwd:
        for module in ("aerodrome.configuration.cli","aerodrome.configuration.hydra_cli"):
            result = subprocess.run([sys.executable,"-I","-m",module,"--help"],cwd=cwd,
                                    text=True,capture_output=True,timeout=60)
            if result.returncode:
                raise RuntimeError(result.stdout+result.stderr)
    return {"status":"passed","python":sys.version,"package_path":str(package_path),
            "package_version":importlib.metadata.version("aerodrome-ngc-prototype"),
            "tick":int(state.tick),"f16_table_count":len(tables),
            "gil_enabled":sys._is_gil_enabled(),"devices":[str(d) for d in jax.devices()]}


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect_distribution(),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report,encoding="utf-8")
    print(report)
