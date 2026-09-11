"""Host renderer basis conversion, including left-handed engines and units."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class EngineCoordinates:
    world_from_ned: tuple
    body_from_frd: tuple
    units_per_m: float = 1.

    def __post_init__(self):
        matrices = []
        for name in ("world_from_ned","body_from_frd"):
            R = np.asarray(getattr(self,name),dtype=float)
            if R.shape!=(3,3) or not np.all(np.isfinite(R)) or not np.allclose(R@R.T,np.eye(3),atol=1e-12,rtol=0):
                raise ValueError("engine basis must be orthogonal 3x3")
            matrices.append(R)
            object.__setattr__(self,name,tuple(tuple(map(float,row)) for row in R))
        if np.linalg.det(matrices[0])*np.linalg.det(matrices[1])<0:
            raise ValueError("world and body bases must use matching handedness")
        if not math.isfinite(self.units_per_m) or self.units_per_m<=0:
            raise ValueError("engine unit scale must be positive")

    def convert_pose(self,pose):
        """Return engine position and proper body-to-world rotation matrix.

        q represents canonical body FRD->NED. Asset-local corrections remain
        in Scene and must also be applied by the engine. Never reflect only a
        quaternion component to attempt a handedness conversion.
        """
        position = np.asarray(pose.position_ned_m,dtype=float)
        q = np.asarray(pose.quaternion_nb,dtype=float)
        if position.shape!=(3,) or q.shape!=(4,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(q)) or np.linalg.norm(q)<1e-12:
            raise ValueError("invalid pose")
        w,x,y,z = q/np.linalg.norm(q)
        v = np.array([x,y,z])
        skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        R = (w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*skew
        A,B = np.asarray(self.world_from_ned),np.asarray(self.body_from_frd)
        return self.units_per_m*(A@position),A@R@B.T


# Explicit adapter conventions, not claims about arbitrary imported asset axes.
# Three-style world: East +X, Up +Y, North -Z; model forward -Z/right +X.
THREE_Y_UP = EngineCoordinates(((0,1,0),(0,0,-1),(-1,0,0)),((0,1,0),(0,0,-1),(-1,0,0)))
# UE-style local world: North +X, East +Y, Up +Z; forward +X/right +Y, cm.
UNREAL_Z_UP_CM = EngineCoordinates(((1,0,0),(0,1,0),(0,0,-1)),((1,0,0),(0,1,0),(0,0,-1)),100.)
