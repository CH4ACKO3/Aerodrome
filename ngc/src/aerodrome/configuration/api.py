"""YAML composition -> typed validation -> explicit build -> bounded run."""
from dataclasses import asdict,dataclass
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
from uuid import uuid4
import json
import math
import sys
import importlib.metadata
import numpy as np
import yaml
from .schema import Experiment
from .registry import typed,builtin_registry

class ConfigLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently discarding user settings."""


def _mapping(loader,node):
    result = {}
    for key_node,value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key,str) or key in result:
            raise ValueError(f"duplicate or non-string YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_mapping)


def _merge(base,overlay):
    for key,value in overlay.items():
        changed_component = isinstance(value,dict) and any(
            field in value and value[field]!=base.get(key,{}).get(field)
            for field in ("kind","version")
        ) if isinstance(base.get(key),dict) else False
        if isinstance(value,dict) and isinstance(base.get(key),dict) and not changed_component:
            _merge(base[key],value)
        else:
            base[key] = value
    return base


def _load(path,stack=()):
    path = Path(path).resolve()
    if path in stack or len(stack)>=16:
        raise ValueError("cyclic or excessively deep configuration includes")
    value = yaml.load(path.read_text(encoding="utf-8"),Loader=ConfigLoader)
    if not isinstance(value,dict):
        raise ValueError("configuration must be a YAML mapping")
    includes = value.pop("includes",[])
    if not isinstance(includes,list) or any(not isinstance(p,str) for p in includes):
        raise ValueError("includes must be a list of paths")
    result = {}
    for include in includes:
        _merge(result,_load(path.parent/include,stack+(path,)))
    return _merge(result,value)


def set_override(config,expression):
    path,sep,raw = expression.partition("=")
    if not sep:
        raise ValueError("override must use path=value")
    parts = path.split(".")
    target = config
    try:
        for part in parts[:-1]:
            if isinstance(target,list) and int(part)<0:
                raise KeyError(part)
            target = target[int(part)] if isinstance(target,list) else target[part]
        key = int(parts[-1]) if isinstance(target,list) else parts[-1]
        if isinstance(key,int) and key<0:
            raise KeyError(key)
        target[key]  # only existing fields may be overridden
        target[key] = yaml.load(raw,Loader=ConfigLoader)
    except (KeyError,IndexError,TypeError,ValueError) as error:
        raise ValueError(f"invalid override path: {path}") from error


def load_config(path,overrides=(), *, overlays=(),registry=None):
    """Compose includes, overlays and typed overrides without changing cwd."""
    path = Path(path).resolve()
    if path.suffix not in (".yaml",".yml") or not path.is_file():
        raise ValueError("configuration must be an existing YAML file")
    config = _load(path)
    for overlay in overlays:
        _merge(config,_load(path.parent/overlay))
    config = asdict(validate_config(config,registry))
    for expression in overrides:
        set_override(config,expression)
    return asdict(validate_config(config,registry))


def validate_config(config,registry=None):
    """Resolve defaults and validate every registered component BEFORE build."""
    registry = registry or builtin_registry()
    value = typed(Experiment,config)
    if value.schema_version!=1:
        raise ValueError("unsupported experiment schema_version")
    if value.action not in ("run","validate"):
        raise ValueError("action must be run or validate")
    if not value.name or not 0<=value.seed<=2**32-1:
        raise ValueError("name required; seed must fit uint32")
    runtime = value.runtime
    if runtime.steps<1 or runtime.chunk_steps<1 or runtime.dtype not in ("float32","float64"):
        raise ValueError("positive steps/chunk_steps and float32/float64 required")
    if runtime.device not in ("cpu","gpu","auto") or not runtime.output_dir:
        raise ValueError("invalid device/output_dir")
    from aerodrome.core.clock import Schedule
    clock = asdict(value.clock)
    ticks = clock.pop("ticks_per_step")
    if ticks<1:
        raise ValueError("ticks_per_step must be positive")
    Schedule(**clock)
    ids = [e.id for e in value.entities]
    if not ids or any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError("unique nonempty entity IDs required")
    for category,spec in [("environment",value.environment),("renderer",value.renderer)]+[("entity",e.model) for e in value.entities]:
        spec.parameters = asdict(registry.validate(category,spec))
    # Reject NaN/infinity also inside plugin configurations before instantiation.
    json.dumps(asdict(value),allow_nan=False)
    return value


@dataclass
class BuiltExperiment:
    config: object
    world: object
    state: object
    parameters: object
    inputs: object
    entities: tuple
    renderer: object
    asset_manifest: dict


@dataclass(frozen=True)
class ScopedAssembly:
    """Keep per-entity resources independent without modifying existing models."""
    entity_id: str
    inner: object
    @property
    def backend(self): return self.inner.backend
    @property
    def systems(self): return self.inner.systems
    @property
    def initial_signals(self): return self.inner.initial_signals
    def initialize(self,initial,key): return self.inner.initialize(initial,key)
    def make_tick(self,schedule):
        fn = self.inner.make_tick(schedule)
        return lambda s,u,p,c:fn(s,u,p,c._replace(resources=c.resources[self.entity_id]))


def build_experiment(config, *, base_dir,registry=None):
    """Trusted factory code creates objects; YAML only selects registered IDs.

    base_dir resolves assets, not output directories. Library callers must set
    JAX precision once at application startup; this function never changes it.
    """
    import jax
    import jax.numpy as jnp
    from aerodrome.catalog import Asset
    from aerodrome.core.clock import Schedule
    from aerodrome.composition import EntitySpec,WorldSpec,build_world
    registry = registry or builtin_registry()
    c = validate_config(config,registry)
    if c.runtime.dtype=="float64" and not jax.config.x64_enabled:
        raise ValueError("enable JAX x64 at application startup for float64 configuration")
    dtype = np.dtype(c.runtime.dtype)
    def vector(value,size,name):
        a = np.asarray(value,dtype)
        if a.shape!=(size,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return jnp.asarray(a)
    clock = asdict(c.clock)
    ticks = clock.pop("ticks_per_step")
    schedule = Schedule(**clock)
    assets,manifest = {},{}
    for name,spec in c.assets.items():
        path = (Path(base_dir)/spec.path).resolve()
        asset = Asset(name,spec.version,path,spec.sha256,spec.source,spec.license)
        assets[name] = asset.read_verified()
        spec.path = str(path)
        manifest[name] = asdict(spec)
    context = dict(config=c,schedule=schedule,assets=assets,asset_manifest=manifest,vector=vector)
    context["gravity"] = registry.create("environment",c.environment,context)
    entities = tuple(registry.create("entity",e.model,context) for e in c.entities)
    # Force explicit requested precision even if global x64 is enabled.
    def cast(tree):
        return jax.tree.map(lambda x:jnp.asarray(x,dtype=dtype) if np.asarray(x).dtype.kind=="f" else jnp.asarray(x),tree)
    for e in entities:
        e.initial,e.parameters,e.inputs,e.resources = map(cast,(e.initial,e.parameters,e.inputs,e.resources))
    world = build_world(WorldSpec(tuple(EntitySpec(spec.id,ScopedAssembly(spec.id,e.assembly))
                         for spec,e in zip(c.entities,entities,strict=True)),schedule,ticks))
    state = world.reset(c.seed,{s.id:e.initial for s,e in zip(c.entities,entities,strict=True)})
    parameters = world.parameters({s.id:e.parameters for s,e in zip(c.entities,entities,strict=True)},
                                  resources={s.id:e.resources for s,e in zip(c.entities,entities,strict=True)})
    inputs = world.pack({s.id:e.inputs for s,e in zip(c.entities,entities,strict=True)})
    world.validate(state,inputs,parameters)
    renderer = registry.create("renderer",c.renderer,context)
    if renderer is not None and any(e.pose is None for e in entities):
        raise ValueError("renderer requires a pose projection for every entity")
    return BuiltExperiment(c,world,state,parameters,inputs,entities,renderer,manifest)


def _save_tree(path,tree):
    import jax
    leaves,structure = jax.tree.flatten_with_path(jax.device_get(tree))
    np.savez_compressed(path,**{f"leaf_{i}":np.asarray(x) for i,(_,x) in enumerate(leaves)})
    path.with_suffix(".paths.json").write_text(json.dumps([jax.tree_util.keystr(p) for p,x in leaves],indent=2)+"\n")


def run_experiment(config, *, base_dir,output_base=None,registry=None,overrides=()):
    """One run, unique output directory, chunked recording, fail-visible status."""
    import jax
    from aerodrome.rendering import RendererSession,Scene,RenderEntity,RenderSample,snapshot
    registry = registry or builtin_registry()
    c = validate_config(config,registry)
    output = Path(c.runtime.output_dir)
    if not output.is_absolute():
        output = Path(output_base or Path.cwd())/output
    folder = output.resolve()/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (folder/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write("requested.json",asdict(c))
    source = Path(__file__).parents[1]
    digest = sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    provenance = dict(overrides=list(overrides),asset_base_dir=str(Path(base_dir).resolve()),
                      source_sha256=digest.hexdigest(),python=sys.version,
                      gil_enabled=sys._is_gil_enabled(),packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()})
    write("provenance.json",provenance)
    write("status.json",dict(status="running"))
    session = None
    try:
        devices = jax.devices() if c.runtime.device=="auto" else jax.devices(c.runtime.device)
        if not devices:
            raise RuntimeError("requested JAX device unavailable")
        with jax.default_device(devices[0]):
            built = build_experiment(config,base_dir=base_dir,registry=registry)
            write("resolved.json",asdict(built.config))
            (folder/"resolved.yaml").write_text(yaml.safe_dump(asdict(built.config),sort_keys=False),encoding="utf-8")
            write("assets.json",built.asset_manifest)
            write("device.json",dict(platform=devices[0].platform,device=str(devices[0])))
            state = built.state
            if built.renderer is not None:
                backend,render_config,render_every = built.renderer
                scene = Scene(tuple(RenderEntity(e.id) for e in c.entities))
                session = RendererSession(backend,scene,render_config)
            chunk_cache = {}
            def chunk(size):
                if size not in chunk_cache:
                    def run(s,u,p):
                        def step(s,_):
                            following,record = built.world.step(s,u,p)
                            return following,record if c.runtime.trace else ()
                        return jax.lax.scan(step,s,None,length=size)
                    chunk_cache[size] = jax.jit(run) if c.runtime.jit else run
                return chunk_cache[size]
            completed = 0
            while completed<c.runtime.steps:
                size = min(c.runtime.chunk_steps,c.runtime.steps-completed)
                # Rendering each configured step needs that step boundary available.
                if session is not None:
                    size = min(size,render_every-completed%render_every)
                state,trace = jax.block_until_ready(chunk(size)(state,built.inputs,built.parameters))
                if any(not np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves((state,trace))):
                    raise FloatingPointError("nonfinite simulation state or record")
                if c.runtime.trace:
                    _save_tree(folder/f"trace_{completed:08d}.npz",trace)
                completed += size
                if session is not None and completed%render_every==0:
                    sample = RenderSample(state.tick,state.tick*c.clock.physics_dt_s,
                                          tuple(e.pose(s) for e,s in zip(built.entities,state.entities,strict=True)))
                    session.submit(snapshot(scene,sample))
                    while not session.ready:
                        session.poll()
                        if not session.ready:
                            import time
                            time.sleep(.001)
            _save_tree(folder/"final.npz",state)
            rendered_frames = session.completed if session else 0
            if session is not None:
                session.close()
                session = None
            write("status.json",dict(status="complete",steps=completed,tick=int(state.tick),
                                      simulated_seconds=float(state.tick)*c.clock.physics_dt_s,
                                      rendered_frames=rendered_frames))
    except BaseException as error:
        write("status.json",dict(status="failed",error_type=type(error).__name__,message=str(error)))
        raise
    finally:
        if session is not None:
            session.close()
    return folder
