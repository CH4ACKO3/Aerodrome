"""Check the actual GIL state before AND after loading native dependencies."""
import importlib
from importlib.metadata import version
import json
import platform
import sys
import sysconfig


def inspect_runtime():
    if sys.version_info[:2] != (3, 14) or sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
        raise RuntimeError("This project targets CPython 3.14t; use the pinned free-threaded interpreter")
    stages = {"startup": sys._is_gil_enabled()}
    versions = {}
    for name in ("numpy", "scipy", "jaxlib", "jax"):
        importlib.import_module(name)
        versions[name] = version(name)
        stages[f"after_{name}"] = sys._is_gil_enabled()
    import jax
    # Exercise lazy native loading as well as top-level imports.
    jax.block_until_ready(jax.jit(lambda x: x + 1)(jax.numpy.asarray(1.)))
    stages["after_jit"] = sys._is_gil_enabled()
    from aerodrome.configuration import validate_config
    import yaml
    validate_config(yaml.safe_load("seed: 1"))
    versions["pydantic"] = version("pydantic")
    versions["pyyaml"] = version("pyyaml")
    stages["after_configuration_validation"] = sys._is_gil_enabled()
    if importlib.util.find_spec("gymnasium") is not None:
        importlib.import_module("aerodrome.adapters.gymnasium")
        versions["gymnasium"] = version("gymnasium")
        stages["after_gymnasium"] = sys._is_gil_enabled()
    if importlib.util.find_spec("hydra") is not None:
        from aerodrome.configuration.hydra_cli import compatible_parser
        compatible_parser().format_help()
        versions["hydra-core"] = version("hydra-core")
        versions["omegaconf"] = version("omegaconf")
        stages["after_hydra_parser"] = sys._is_gil_enabled()
    if importlib.util.find_spec("control") is not None:
        import control as ct
        versions["control"] = version("control")
        versions["matplotlib"] = version("matplotlib")
        stages["after_control"] = sys._is_gil_enabled()
        model = ct.sample_system(ct.tf2ss(ct.tf([1.],[1.,1.]),method="scipy"),.01)
        ct.step_response(model)
        stages["after_control_response"] = sys._is_gil_enabled()
    if any(stages.values()):
        raise RuntimeError(f"GIL was enabled at runtime: {stages}; do not force-disable incompatible extensions")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "py_gil_disabled_build": sysconfig.get_config_var("Py_GIL_DISABLED"),
            "gil_enabled_by_stage": stages, "packages": versions,
            "devices": [str(device) for device in jax.devices()]}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect_runtime(), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report,encoding="utf-8")
    print(report)
