"""Typed versioned factory registration, deliberately separate from config imports."""
from dataclasses import dataclass
from typing import Any
import json
from functools import lru_cache
from pydantic import TypeAdapter, ConfigDict
from pydantic.dataclasses import dataclass as checked_dataclass, is_pydantic_dataclass
from aerodrome.catalog import Registry


@lru_cache
def _adapter(schema):
    if not is_pydantic_dataclass(schema):
        schema = checked_dataclass(schema, config=ConfigDict(extra="forbid", allow_inf_nan=False))
    return TypeAdapter(schema)


def typed(schema,values):
    return _adapter(schema).validate_json(json.dumps(values,allow_nan=False),strict=True)


class ComponentRegistry:
    def __init__(self):
        self._registry,self._schemas = Registry(),{}

    def register(self,category,kind,version,schema,factory):
        self._registry.register(f"{category}/{kind}",version,factory)
        self._schemas[category,kind,version] = schema

    def validate(self,category,spec):
        key = category,spec.kind,spec.version
        if key not in self._schemas:
            raise ValueError(f"unregistered component: {key}")
        return typed(self._schemas[key],spec.parameters)

    def create(self,category,spec,context):
        return self._registry.create(f"{category}/{spec.kind}",spec.version,
                                     options=self.validate(category,spec),context=context)


@dataclass
class EntityBuild:
    assembly: Any
    initial: Any
    parameters: Any
    inputs: Any
    resources: Any = ()
    pose: Any = None  # optional pure entity_state -> Pose for renderer


def builtin_registry():
    from .schema import RigidBodyOptions,F16Options,GravityOptions,EmptyOptions,HeadlessOptions
    registry = ComponentRegistry()
    def gravity(options,context):
        return context["vector"](options.gravity_ned_m_s2,3,"gravity_ned_m_s2")
    def rigid(options,context):
        from aerodrome.models.rigid_body import RigidBody6DoF,mass_properties,BodyLoads
        from aerodrome.composition.rigid_body import RigidBodyAssembly
        from aerodrome.rendering import rigid_body_pose
        body = RigidBody6DoF(options.attitude)
        parameters = mass_properties(options.mass_kg,options.inertia_body_kg_m2,context["gravity"])
        state = body.initialize(position_ned_m=options.position_ned_m,velocity_body_m_s=options.velocity_body_m_s,
                                euler_rad=options.euler_rad,omega_body_rad_s=options.omega_body_rad_s)
        inputs = BodyLoads(context["vector"](options.force_body_N,3,"force_body_N"),
                           context["vector"](options.moment_body_Nm,3,"moment_body_Nm"))
        return EntityBuild(RigidBodyAssembly(body),state,parameters,inputs,
                           pose=lambda s:rigid_body_pose(s,attitude=options.attitude))
    def f16(options,context):
        import numpy as np
        from aerodrome.models.f16_longitudinal import load_tables,Airframe
        from aerodrome.models import f16_longitudinal
        from pathlib import Path
        from aerodrome.pipelines.f16_trim import design
        from aerodrome.composition.f16_longitudinal import F16LongitudinalAssembly,FlightParameters,FlightState
        if context["config"].runtime.dtype!="float64":
            raise ValueError("F16 trim/design requires float64")
        g = np.asarray(context["gravity"])
        if not np.allclose(g[:2],0) or not 0<g[2]<20:
            raise ValueError("F16 requires positive NED-down gravity, horizontal gravity zero")
        airframe = Airframe(gravity=float(g[2]))
        tables = load_tables()
        context["asset_manifest"]["builtin/f16_longitudinal"] = json.loads(
            (Path(f16_longitudinal.__file__).parent/"data/f16/manifest.json").read_text())
        x,u,K,info = design(tables,speed=options.speed_m_s,height=options.height_m,
                           dt=context["schedule"].control_dt_s,airframe=airframe)
        perturbation = context["vector"](options.perturbation,5,"perturbation")
        return EntityBuild(F16LongitudinalAssembly(),FlightState(x+perturbation,u),
                           FlightParameters(x,u,K,airframe),x,tables)
    registry.register("environment","constant_gravity","1",GravityOptions,gravity)
    registry.register("entity","rigid_body","1",RigidBodyOptions,rigid)
    registry.register("entity","f16_longitudinal","1",F16Options,f16)
    registry.register("renderer","none","1",EmptyOptions,lambda options,context:None)
    def headless(options,context):
        from aerodrome.rendering import HeadlessBackend,RenderConfig
        if options.every_steps<1:
            raise ValueError("renderer.every_steps must be positive")
        return HeadlessBackend(),RenderConfig(mode="offline",width=options.width,height=options.height),options.every_steps
    registry.register("renderer","headless","1",HeadlessOptions,headless)
    return registry
