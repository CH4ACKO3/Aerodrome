"""Host-only replaceable renderer contract. No graphics engine in simulation code."""
from dataclasses import dataclass, asdict
from typing import Protocol, Literal
from threading import get_ident
from time import perf_counter
import math
from .schema import Scene, Frame
from .live import RateMeter


@dataclass(frozen=True)
class Capabilities:
    modes: tuple = ("live","offline")
    articulations: bool = False
    camera: bool = False


@dataclass(frozen=True)
class RenderConfig:
    mode: Literal["live","offline"] = "live"
    width: int = 1280
    height: int = 720
    request_timeout_s: float = 30.

    def __post_init__(self):
        if self.mode not in ("live","offline") or any(type(v) is not int or v<=0 for v in (self.width,self.height)):
            raise ValueError("invalid rendering mode or viewport")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s<=0:
            raise ValueError("request timeout must be finite positive")


@dataclass(frozen=True)
class JointValue:
    entity_id: str
    node: str
    angle_rad: float
    axis_local: tuple = (0.,1.,0.)  # node rest-local axes; postmultiply rest rotation

    def __post_init__(self):
        if not self.entity_id or not self.node or not math.isfinite(self.angle_rad):
            raise ValueError("joint requires entity, node and finite angle")
        axis = tuple(map(float,self.axis_local))
        norm = math.sqrt(sum(x*x for x in axis))
        if len(axis)!=3 or not all(map(math.isfinite,axis)) or norm<1e-12:
            raise ValueError("joint axis must be finite nonzero 3-vector")
        object.__setattr__(self,"axis_local",tuple(x/norm for x in axis))


@dataclass(frozen=True)
class Camera:
    position_ned_m: tuple
    target_ned_m: tuple
    up_ned: tuple = (0.,0.,-1.)
    vertical_fov_rad: float = math.pi/3

    def __post_init__(self):
        for name in ("position_ned_m","target_ned_m","up_ned"):
            value = tuple(map(float,getattr(self,name)))
            if len(value)!=3 or not all(map(math.isfinite,value)):
                raise ValueError("camera vectors must be finite 3-vectors")
            object.__setattr__(self,name,value)
        d = tuple(b-a for a,b in zip(self.position_ned_m,self.target_ned_m))
        u = self.up_ned
        cross = (d[1]*u[2]-d[2]*u[1],d[2]*u[0]-d[0]*u[2],d[0]*u[1]-d[1]*u[0])
        if sum(x*x for x in cross)<1e-20 or not 0<self.vertical_fov_rad<math.pi:
            raise ValueError("degenerate camera view or FOV")


@dataclass(frozen=True)
class RenderRequest:
    sequence: int
    frame: Frame
    camera: Camera | None = None
    joints: tuple[JointValue,...] = ()

    def to_dict(self):
        return dict(protocol_version=1,kind="render",sequence=self.sequence,
                    frame=self.frame.to_dict(),camera=asdict(self.camera) if self.camera else None,
                    joints=[asdict(x) for x in self.joints])


@dataclass(frozen=True)
class RenderReceipt:
    sequence: int
    status: Literal["presented","completed","dropped","failed"]
    detail: str = ""
    artifact_uri: str | None = None

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence<1 or self.status not in ("presented","completed","dropped","failed"):
            raise ValueError("invalid render receipt")


class RendererBackend(Protocol):
    """Called on ONE owner thread. submit/poll must be nonblocking.

    open returns only after scene setup; every accepted request eventually
    gets exactly one terminal receipt. close cancels/drains owned work and is
    idempotent. Implement network/process/UI bridging inside the adapter.
    """
    def open(self,scene: Scene,config: RenderConfig) -> Capabilities: ...
    def submit(self,request: RenderRequest) -> None: ...
    def poll(self) -> tuple[RenderReceipt,...]: ...
    def close(self) -> None: ...


class RendererSession:
    """Lifecycle, capability checks and one in-flight request (bounded memory).

    Live consumers should take LatestFrameStream only when ready. Offline
    consumers wait for the receipt before advancing. Seek/replay can submit
    older simulation times: transport sequence, not tick, identifies receipts.
    """
    def __init__(self,backend,scene,config=RenderConfig(), *, meter=None,clock=perf_counter):
        self.backend,self.scene,self.config = backend,scene,config
        self._clock,self._deadline = clock,None
        self.meter = meter if meter is not None else RateMeter()
        self._owner,self._state,self._pending,self._sequence = get_ident(),"opening",None,0
        self.completed = self.presented = self.dropped = 0
        try:
            self.capabilities = backend.open(scene,config)
            if config.mode not in self.capabilities.modes:
                raise ValueError("backend does not support requested mode")
            self._state = "ready"
        except BaseException:
            self._state = "failed"
            backend.close()
            raise

    def _check(self):
        if get_ident()!=self._owner:
            raise RuntimeError("renderer session must run on its owner thread")
        if self._state not in ("ready","busy"):
            raise RuntimeError(f"renderer session is {self._state}")

    @property
    def ready(self):
        return self._state=="ready"

    def submit(self,frame, *, camera=None,joints=()):
        self._check()
        if not self.ready:
            return None  # Not accepted; caller retains the frame.
        joints = tuple(joints)
        ids = {e.id for e in self.scene.entities}
        if len(frame.poses)!=len(ids):
            raise ValueError("frame topology differs from scene")
        if camera is not None and not self.capabilities.camera:
            raise ValueError("backend does not support cameras")
        if joints and not self.capabilities.articulations:
            raise ValueError("backend does not support articulations")
        keys = [(j.entity_id,j.node) for j in joints]
        if len(set(keys))!=len(keys) or any(j.entity_id not in ids for j in joints):
            raise ValueError("duplicate/unknown joint entity")
        self._sequence += 1
        request = RenderRequest(self._sequence,frame,camera,joints)
        self._pending,self._state = request.sequence,"busy"
        self._deadline = self._clock()+self.config.request_timeout_s
        try:
            self.backend.submit(request)
        except BaseException:
            self._state = "failed"
            raise
        return request.sequence

    def poll(self):
        self._check()
        try:
            if self._pending is not None and self._clock()>self._deadline:
                raise TimeoutError("renderer request timed out; close/recreate session")
            receipts = tuple(self.backend.poll())
            for receipt in receipts:
                if receipt.sequence!=self._pending:
                    raise RuntimeError("unexpected/duplicate renderer receipt")
                self._pending = None
                if receipt.status=="failed":
                    raise RuntimeError(f"renderer failed: {receipt.detail}")
                if self.config.mode=="offline" and receipt.status=="dropped":
                    raise RuntimeError("offline renderer dropped a requested frame")
                self.completed += receipt.status in ("completed","presented")
                self.presented += receipt.status=="presented"
                self.dropped += receipt.status=="dropped"
                if receipt.status=="presented":
                    self.meter.frame_presented()
                self._state = "ready"
            return receipts
        except BaseException:
            self._state = "failed"
            raise

    def close(self):
        if get_ident()!=self._owner:
            raise RuntimeError("close on renderer owner thread")
        if self._state=="closed":
            return
        try:
            self.backend.close()
        finally:
            self._state,self._pending = "closed",None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


class HeadlessBackend:
    """Contract/test backend, no pixels. Keeps only the latest request."""
    def __init__(self):
        self.last_request = None
        self._receipt = None
        self._open = False

    def open(self,scene,config):
        if self._open:
            raise RuntimeError("backend already open")
        self._open = True
        return Capabilities(camera=True,articulations=True)

    def submit(self,request):
        if not self._open or self._receipt is not None:
            raise RuntimeError("backend unavailable")
        self.last_request = request
        self._receipt = RenderReceipt(request.sequence,"completed")

    def poll(self):
        receipt,self._receipt = self._receipt,None
        return () if receipt is None else (receipt,)

    def close(self):
        self._open = False
        self._receipt = self.last_request = None
