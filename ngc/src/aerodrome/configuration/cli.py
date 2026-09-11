"""Typed YAML CLI with explicit, bounded Cartesian parameter sweeps."""
import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import json
import math
import yaml
from .api import load_config,validate_config,build_experiment,run_experiment,ConfigLoader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'conf'/'experiment.yaml')
    parser.add_argument('--overlay',action='append',default=[],help='YAML overlay relative to the main config')
    parser.add_argument('--validate',action='store_true',help='Validate and build without advancing simulation')
    parser.add_argument('--sweep',action='append',default=[],help='Cartesian sweep: path=[value1,value2] (maximum 256 runs)')
    parser.add_argument('overrides',nargs='*',help='Existing dotted path=value; list indices are supported')
    args = parser.parse_args(argv)
    axes = []
    for expression in args.sweep:
        key,sep,raw = expression.partition('=')
        values = yaml.load(raw,Loader=ConfigLoader)
        if not sep or not isinstance(values,list) or not values:
            parser.error('--sweep requires path=[value1,value2]')
        axes.append([f'{key}={json.dumps(v,allow_nan=False)}' for v in values])
    if math.prod(map(len,axes))>256:
        parser.error('sweep exceeds 256 runs')
    plans = []
    for choice in product(*axes):
        overrides = args.overrides+list(choice)
        plans.append((load_config(args.config,overrides,overlays=args.overlay),overrides))
    import jax
    for config,overrides in plans:
        value = validate_config(config)
        jax.config.update('jax_enable_x64',value.runtime.dtype=='float64')
        if args.validate or value.action=='validate':
            built = build_experiment(config,base_dir=args.config.resolve().parent)
            print(yaml.safe_dump(asdict(built.config),sort_keys=False))
        else:
            folder = run_experiment(config,base_dir=args.config.resolve().parent,
                                    overrides=[f'overlay={p}' for p in args.overlay]+overrides)
            print(f'Run completed: {folder}')


if __name__=='__main__':
    main()
