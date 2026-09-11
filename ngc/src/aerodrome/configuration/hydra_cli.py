"""Optional Hydra frontend; Python 3.14 compatibility is scoped to this parser."""
import argparse
from dataclasses import asdict
from pathlib import Path
from types import FunctionType,SimpleNamespace
import json
import sys
from uuid import uuid4


class _CompletionParser(argparse.ArgumentParser):
    def add_argument(self,*args,**kwargs):
        # Hydra 1.3.6 passes a __repr__-only lazy object for this one option.
        # Materialize it before 3.14 checks help; leave normal validation intact.
        if "--shell-completion" in args and not isinstance(kwargs.get("help"),str):
            kwargs["help"] = str(kwargs["help"])
        return super().add_argument(*args,**kwargs)


def compatible_parser():
    import hydra
    from hydra._internal.utils import get_args_parser
    if hydra.__version__!="1.3.6":
        raise RuntimeError("Hydra frontend is tested with hydra-core==1.3.6; review compatibility before upgrading")
    if sys.version_info<(3,14):
        return get_args_parser()
    # Clone only the parser factory's global namespace. No assignment to Hydra
    # globals or argparse classes, including during concurrent parser creation.
    facade = SimpleNamespace(**{**vars(argparse),"ArgumentParser":_CompletionParser})
    factory = FunctionType(get_args_parser.__code__,
                           {**get_args_parser.__globals__,"argparse":facade},
                           get_args_parser.__name__,get_args_parser.__defaults__,get_args_parser.__closure__)
    return factory()


def register_configs():
    """Expose the existing YAML presets as Hydra groups, with one source of truth."""
    from hydra.core.config_store import ConfigStore
    from .api import load_config,_load
    folder = Path(__file__).parent/"conf"
    store = ConfigStore.instance()
    root = load_config(folder/"experiment.yaml")
    root["defaults"] = ["_self_",{"scenario":"rigid"},{"environment":"vacuum"},{"renderer":"none"}]
    root["asset_base_dir"] = "."
    run_root = "artifacts/hydra/${now:%Y%m%dT%H%M%S}_"+uuid4().hex[:12]
    root["hydra"] = {"job":{"chdir":False},
                     "run":{"dir":run_root},
                     "sweep":{"dir":run_root,"subdir":"${hydra.job.num}"}}
    store.store(name="aerodrome_experiment",node=root,provider="aerodrome")
    for group in ("scenario","environment","renderer"):
        for path in sorted((folder/group).glob("*.yaml")):
            store.store(group=group,name=path.stem,node=_load(path),package="_global_",provider="aerodrome")


def execute(config):
    import jax
    import yaml
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from .api import validate_config,build_experiment,run_experiment
    data = OmegaConf.to_container(config,resolve=True,throw_on_missing=True)
    hydra_config = HydraConfig.get()
    original_cwd = Path(hydra_config.runtime.cwd)
    base = original_cwd/Path(data.pop("asset_base_dir",str(original_cwd)))
    value = validate_config(data)
    jax.config.update("jax_enable_x64",value.runtime.dtype=="float64")
    if value.action=="validate":
        built = build_experiment(data,base_dir=base)
        print(yaml.safe_dump(asdict(built.config),sort_keys=False))
        return None
    folder = run_experiment(data,base_dir=base,output_base=original_cwd,
                            overrides=list(hydra_config.overrides.task))
    context = dict(hydra_version=hydra_config.runtime.version,
                   output_dir=str(hydra_config.runtime.output_dir),
                   choices=OmegaConf.to_container(hydra_config.runtime.choices,resolve=True),
                   overrides=OmegaConf.to_container(hydra_config.overrides,resolve=True))
    (folder/"hydra-context.json").write_text(json.dumps(context,indent=2)+"\n",encoding="utf-8")
    (Path(hydra_config.runtime.output_dir)/"aerodrome-run.json").write_text(
        json.dumps({"run_dir":str(folder)},indent=2)+"\n",encoding="utf-8")
    print(f"Run completed: {folder}")
    return str(folder)


def main(argv=None):
    try:
        from hydra._internal.utils import _run_hydra
        from hydra import version
    except ModuleNotFoundError as error:
        if error.name=="hydra":
            raise SystemExit("Install Hydra support: uv sync --extra hydra") from error
        raise
    version.setbase("1.3")
    register_configs()
    parser = compatible_parser()
    args = parser.parse_args(argv)
    if args.experimental_rerun:
        parser.error("pickle rerun is not supported; use saved resolved configuration")
    # Pinned private dispatch preserves Hydra's native launcher/sweeper/help flow
    # while allowing an instance-local compatible parser instead of monkeypatches.
    _run_hydra(args=args,args_parser=parser,task_function=execute,
               config_path=None,config_name="aerodrome_experiment")


if __name__=="__main__":
    main()
