"""Streaming JSONL archive and simulation-time playback, renderer-neutral."""
import json
import math
from bisect import bisect_left
from pathlib import Path
import numpy as np
from .schema import Scene, RenderEntity, Frame, Pose, validate_sequence, SCHEMA_VERSION


class TrajectoryWriter:
    """Lossless exported snapshots; exclusive creation, synchronous host I/O.

    'Lossless' concerns supplied snapshots only, not unexported physics ticks.
    Use a context manager; the reader rejects partial/truncated final lines.
    """
    def __init__(self,path,scene):
        self.scene,self._previous = scene,None
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self._file = path.open("x",encoding="utf-8")
        self._file.write(json.dumps(scene.to_dict(),allow_nan=False)+"\n")

    def append(self,frame):
        if len(frame.poses)!=len(self.scene.entities):
            raise ValueError("pose count differs from scene")
        validate_sequence(self._previous,frame)
        self._file.write(frame.to_json()+"\n")
        self._previous = frame

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


def read_trajectory(path):
    """Return (Scene, tuple[Frame]); loads the archive into host memory."""
    frames = []
    with Path(path).open(encoding="utf-8") as source:
        header = json.loads(next(source))
        if (header.get("schema_version")!=SCHEMA_VERSION or header.get("kind")!="scene"
                or header.get("position_frame")!="local_ned" or header.get("position_unit")!="m"
                or header.get("quaternion_order")!="wxyz" or header.get("rotation")!="body_frd_to_ned"):
            raise ValueError("unsupported rendering schema or conventions")
        scene = Scene(tuple(RenderEntity(**x) for x in header["entities"]),header.get("origin_lla"))
        for line in source:
            data = json.loads(line)
            if data.pop("schema_version",None)!=SCHEMA_VERSION or data.pop("kind",None)!="frame":
                raise ValueError("unsupported frame schema")
            data["poses"] = tuple(Pose(**p) for p in data["poses"])
            frame = Frame(**data)
            if len(frame.poses)!=len(scene.entities):
                raise ValueError("pose count differs from scene")
            validate_sequence(frames[-1] if frames else None,frame)
            frames.append(frame)
    return scene,tuple(frames)


def interpolate(left,right,time_s):
    """Linear positions, shortest-arc SLERP, no extrapolation or episode crossing."""
    if (left.world_id,left.episode_id)!=(right.world_id,right.episode_id):
        raise ValueError("cannot interpolate across worlds or episode reset")
    if len(left.poses)!=len(right.poses) or right.time_s<=left.time_s or not left.time_s<=time_s<=right.time_s:
        raise ValueError("invalid interpolation bracket")
    if time_s==left.time_s:
        return left
    if time_s==right.time_s:
        return right
    t = (time_s-left.time_s)/(right.time_s-left.time_s)
    poses = []
    for a,b in zip(left.poses,right.poses,strict=True):
        p = (1-t)*np.asarray(a.position_ned_m)+t*np.asarray(b.position_ned_m)
        qa,qb = np.asarray(a.quaternion_nb),np.asarray(b.quaternion_nb)
        dot = float(np.dot(qa,qb))
        if dot<0:
            qb,dot = -qb,-dot
        if dot>.9995:
            q = (1-t)*qa+t*qb
        else:
            angle = math.acos(np.clip(dot,-1,1))
            q = (math.sin((1-t)*angle)*qa+math.sin(t*angle)*qb)/math.sin(angle)
        poses.append(Pose(p,q))
    return Frame(left.world_id,left.episode_id,float(time_s),None,
                 (left.source_ticks[0],right.source_ticks[1]),tuple(poses))


def resample(frames, fps, *, playback_speed=1.):
    """Yield constant display-time FPS for one episode, independent of tick rate.

    Simulation spacing = playback_speed/fps. End is included only when it
    lies on that grid. Input frames must already be sorted, nonempty snapshots.
    """
    frames = tuple(frames)
    if not frames or not math.isfinite(fps) or fps<=0 or not math.isfinite(playback_speed) or playback_speed<=0:
        raise ValueError("nonempty frames, positive FPS and playback speed required")
    for a,b in zip(frames,frames[1:]):
        validate_sequence(a,b)
        if a.episode_id!=b.episode_id:
            raise ValueError("select a single episode for playback")
    times = [f.time_s for f in frames]
    spacing = playback_speed/fps
    count = math.floor((times[-1]-times[0])/spacing+1e-10)+1
    for i in range(count):
        t = min(times[0]+i*spacing,times[-1])
        j = bisect_left(times,t)
        if j<len(times) and times[j]==t:
            yield frames[j]
        else:
            yield interpolate(frames[j-1],frames[j],t)
