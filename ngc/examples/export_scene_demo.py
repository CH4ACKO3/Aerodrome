"""Export a real rigid-body World turn as Scene/Frame v1 for browser playback."""
import json
from pathlib import Path
import math
import jax
import numpy as np
from aerodrome.configuration import build_experiment
from aerodrome.rendering import Scene, RenderEntity, Frame, Pose

def main():
    folder=Path(__file__).resolve().parents[2]/"website/public"
    asset=json.loads((folder/"models/manifest.json").read_text())
    built=build_experiment({"runtime":{"dtype":"float32"},"entities":[{"id":"aircraft","model":{"kind":"rigid_body","version":"1","parameters":{
        "velocity_body_m_s":[100.,0.,0.],"omega_body_rad_s":[0.,0.,.06],
        "force_body_N":[0.,6000.,0.],"position_ned_m":[0.,0.,-1000.]}}}]},base_dir=".")
    def step(state,_):
        new,_=built.world.step(state,built.inputs,built.parameters)
        body=new.entities[0]
        return new,(body.position_ned_m,body.attitude)
    _,(positions,attitudes)=jax.jit(lambda s:jax.lax.scan(step,s,None,length=500))(built.state)
    positions=np.concatenate([np.array([[0.,0.,-1000.]]),np.asarray(positions)])
    attitudes=np.concatenate([np.array([[1.,0.,0.,0.]]),np.asarray(attitudes)])
    scene=Scene((RenderEntity("aircraft",asset_uri=asset["asset_uri"],body_from_asset_quaternion=asset["body_from_asset_quaternion"]),),origin_lla=(math.radians(22.5),math.radians(114.),0.))
    frames=[Frame(0,0,i*.02,i*2,(i*2,i*2),(Pose(p,q),)).to_dict() for i,(p,q) in enumerate(zip(positions,attitudes))]
    target=folder/"examples/rigid-turn.json"
    target.write_text(json.dumps(dict(scene=scene.to_dict(),frames=frames,source="precomputed World/RK4; constant side force and yaw rate; no gravity/aerodynamics"),separators=(",",":"))+"\n",encoding="utf-8")
    print(target)

if __name__=="__main__":main()
