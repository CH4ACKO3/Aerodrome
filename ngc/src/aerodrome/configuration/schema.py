"""Host-only structured configuration. Numerical kernels receive arrays only."""
from dataclasses import field
from functools import partial
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import Any, Dict, List

dataclass = partial(pydantic_dataclass, config=ConfigDict(extra="forbid", allow_inf_nan=False))


@dataclass
class Component:
    kind: str = "rigid_body"
    version: str = "1"
    parameters: Dict[str,Any] = field(default_factory=dict)


@dataclass
class Entity:
    id: str = "aircraft"
    model: Component = field(default_factory=Component)


@dataclass
class Clock:
    physics_dt_s: float = .01
    ticks_per_step: int = 2
    sensor_every: int = 2
    guidance_every: int = 10
    control_every: int = 2


@dataclass
class Runtime:
    steps: int = 500
    chunk_steps: int = 100
    dtype: str = "float64"
    device: str = "cpu"
    jit: bool = True
    trace: bool = True
    output_dir: str = "artifacts/config_runs"


@dataclass
class AssetConfig:
    path: str
    sha256: str
    source: str
    license: str
    version: str = "1"


@dataclass
class Experiment:
    schema_version: int = 1
    action: str = "run"
    name: str = "rigid_body_demo"
    seed: int = 0
    clock: Clock = field(default_factory=Clock)
    runtime: Runtime = field(default_factory=Runtime)
    entities: List[Entity] = field(default_factory=lambda:[Entity()])
    environment: Component = field(default_factory=lambda:Component("constant_gravity","1",{}))
    renderer: Component = field(default_factory=lambda:Component("none","1",{}))
    assets: Dict[str,AssetConfig] = field(default_factory=dict)


@dataclass
class RigidBodyOptions:
    attitude: str = "quaternion"
    mass_kg: float = 1000.
    inertia_body_kg_m2: List[List[float]] = field(default_factory=lambda:[[1000.,0.,0.],[0.,1000.,0.],[0.,0.,1000.]])
    position_ned_m: List[float] = field(default_factory=lambda:[0.,0.,-1000.])
    velocity_body_m_s: List[float] = field(default_factory=lambda:[100.,0.,0.])
    euler_rad: List[float] = field(default_factory=lambda:[0.,0.,0.])
    omega_body_rad_s: List[float] = field(default_factory=lambda:[0.,0.,.05])
    force_body_N: List[float] = field(default_factory=lambda:[0.,0.,0.])
    moment_body_Nm: List[float] = field(default_factory=lambda:[0.,0.,0.])


@dataclass
class F16Options:
    speed_m_s: float = 150.
    height_m: float = 3000.
    perturbation: List[float] = field(default_factory=lambda:[0.,0.,0.,0.,0.])


@dataclass
class GravityOptions:
    gravity_ned_m_s2: List[float] = field(default_factory=lambda:[0.,0.,0.])


@dataclass
class EmptyOptions:
    pass


@dataclass
class HeadlessOptions:
    width: int = 1280
    height: int = 720
    every_steps: int = 1
