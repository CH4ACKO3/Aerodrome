"""Versioned render contract; explicit device/host boundary, no engine dependency."""
from dataclasses import dataclass, asdict
from typing import NamedTuple, Any
import json
import math
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.core.rotations import euler321_to_quaternion

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RenderEntity:
    id: str
    asset_uri: str | None = None
    scale: float = 1.
    # Mesh coordinates -> body FRD, Hamilton [w,x,y,z].
    body_from_asset_quaternion: tuple = (1.,0.,0.,0.)

    def __post_init__(self):
        if not isinstance(self.id,str) or not self.id:
            raise ValueError("render entity needs a nonempty string ID")
        if not math.isfinite(self.scale) or self.scale<=0:
            raise ValueError("asset scale must be finite positive")
        if self.asset_uri is not None and not isinstance(self.asset_uri,str):
            raise ValueError("asset URI must be a string or None")
        q = _unit_quaternion(self.body_from_asset_quaternion)
        object.__setattr__(self,"body_from_asset_quaternion",q)


def _unit_quaternion(value):
    q = np.asarray(value,dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q)<1e-12:
        raise ValueError("render quaternion must be a finite nonzero 4-vector")
    return tuple(map(float,q/np.linalg.norm(q)))


@dataclass(frozen=True)
class Scene:
    entities: tuple[RenderEntity,...]
    origin_lla: tuple | None = None  # radians/radians/ellipsoid metres

    def __post_init__(self):
        object.__setattr__(self,"entities",tuple(self.entities))
        ids = [e.id for e in self.entities]
        if not ids or len(set(ids))!=len(ids):
            raise ValueError("scene requires unique entities")
        if self.origin_lla is not None:
            origin = tuple(map(float,self.origin_lla))
            if len(origin)!=3 or not all(map(math.isfinite,origin)) or abs(origin[0])>math.pi/2:
                raise ValueError("invalid scene origin LLA")
            object.__setattr__(self,"origin_lla",origin)

    def to_dict(self):
        return dict(schema_version=SCHEMA_VERSION,kind="scene",position_frame="local_ned",
                    position_unit="m",quaternion_order="wxyz",rotation="body_frd_to_ned",
                    origin_lla=self.origin_lla,entities=[asdict(e) for e in self.entities])


class Pose(NamedTuple):
    position_ned_m: Any
    quaternion_nb: Any


class RenderSample(NamedTuple):
    tick: Any
    time_s: Any
    poses: tuple


def rigid_body_pose(state, *, attitude="quaternion"):
    """JAX projection; strips physics/controller state from rendering data."""
    if attitude not in ("quaternion","euler321"):
        raise ValueError("unknown attitude representation")
    q = state.attitude if attitude=="quaternion" else euler321_to_quaternion(state.attitude)
    return Pose(state.position_ned_m,q)


def make_projection(physics_dt_s, selectors):
    """selectors: static callables WorldState -> Pose, in Scene entity order.

    Can run inside scan/record projection; pure JAX, no copies or side effects.
    Select one world or vmap this projection over a batched WorldState.
    """
    if not math.isfinite(physics_dt_s) or physics_dt_s<=0:
        raise ValueError("physics dt must be positive")
    selectors = tuple(selectors)
    if not selectors:
        raise ValueError("projection requires selectors")
    def project(state):
        return RenderSample(state.tick,state.tick*physics_dt_s,tuple(fn(state) for fn in selectors))
    return project


@dataclass(frozen=True)
class Frame:
    world_id: int
    episode_id: int
    time_s: float
    tick: int | None  # None means interpolated display frame, not a physics tick.
    source_ticks: tuple[int,int]
    poses: tuple[Pose,...]

    def __post_init__(self):
        if any(type(v) is not int or v<0 for v in (self.world_id,self.episode_id)):
            raise ValueError("world/episode IDs must be nonnegative integers")
        if not math.isfinite(self.time_s) or self.time_s<0:
            raise ValueError("frame time must be finite and nonnegative")
        ticks = tuple(self.source_ticks)
        if len(ticks)!=2 or any(type(v) is not int or v<0 for v in ticks) or ticks[1]<ticks[0]:
            raise ValueError("invalid source tick bracket")
        if self.tick is not None and (type(self.tick) is not int or ticks!=(self.tick,self.tick)):
            raise ValueError("snapshot tick must match its source bracket")
        poses = []
        for position,q in self.poses:
            position = tuple(map(float,position))
            if len(position)!=3 or not all(map(math.isfinite,position)):
                raise ValueError("render position must be a finite 3-vector")
            poses.append(Pose(position,_unit_quaternion(q)))
        if not poses:
            raise ValueError("frame must contain poses")
        object.__setattr__(self,"source_ticks",ticks)
        object.__setattr__(self,"poses",tuple(poses))

    def to_dict(self):
        return dict(schema_version=SCHEMA_VERSION,kind="frame",world_id=self.world_id,
                    episode_id=self.episode_id,time_s=self.time_s,tick=self.tick,
                    source_ticks=self.source_ticks,poses=[dict(position_ned_m=p.position_ned_m,
                    quaternion_nb=p.quaternion_nb) for p in self.poses])

    def to_json(self):
        return json.dumps(self.to_dict(),allow_nan=False,separators=(",",":"))


def snapshot(scene, sample, *, world_id=0, episode_id=0):
    """Host boundary: device_get synchronizes only projected data, not full world.

    This may block for device completion. Keep it outside physics JIT and off
    a renderer UI thread when transfers are large. Frame owns immutable tuples.
    """
    tick,time,poses = jax.device_get(sample)
    if np.shape(tick)!=() or np.asarray(tick).dtype.kind not in "iu" or np.shape(time)!=():
        raise ValueError("snapshot needs one world and one time sample")
    if len(poses)!=len(scene.entities):
        raise ValueError("pose count differs from scene")
    return Frame(world_id,episode_id,float(time),int(tick),(int(tick),int(tick)),tuple(poses))


def validate_sequence(previous, frame):
    """A stream is one world; episodes increase and time resets only on reset."""
    if previous is None:
        return
    if frame.world_id!=previous.world_id or len(frame.poses)!=len(previous.poses):
        raise ValueError("stream world/pose topology changed")
    if frame.episode_id<previous.episode_id:
        raise ValueError("stale episode")
    if frame.episode_id==previous.episode_id and (
            frame.time_s<=previous.time_s or frame.source_ticks[0]<previous.source_ticks[0]
            or (frame.tick is not None and previous.tick is not None and frame.tick<=previous.tick)):
        raise ValueError("frame time/ticks must advance within an episode")
