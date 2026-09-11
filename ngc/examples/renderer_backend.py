"""Exercise the replacement backend without a browser or game engine."""
import json
from pathlib import Path
from aerodrome.rendering import (Scene,RenderEntity,Frame,Pose,resample,RendererSession,
    HeadlessBackend,RenderConfig,JointValue,Camera,THREE_Y_UP,UNREAL_Z_UP_CM)


def main():
    scene = Scene((RenderEntity("aircraft",asset_uri="asset://aircraft/f16"),))
    frames = tuple(Frame(0,0,float(i),i*100,(i*100,i*100),
                         (Pose((100.*i,0.,-1000.),(1.,0,0,0)),)) for i in range(2))
    backend = HeadlessBackend()
    with RendererSession(backend,scene,RenderConfig(mode="offline")) as session:
        for frame in resample(frames,60):
            session.submit(frame,camera=Camera((0.,-100.,-1050.),frame.poses[0].position_ned_m),
                           joints=(JointValue("aircraft","elevator",.05),))
            session.poll()  # immediate only for this headless backend
        request = backend.last_request.to_dict()
        summary = dict(completed_frames=session.completed,presented_frames=session.presented,
                       final_sequence=request["sequence"],renderer="headless contract backend; no image output")
    assert summary["completed_frames"]==61 and summary["presented_frames"]==0
    folder = Path("artifacts/renderer_backend")
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"scene.json").write_text(json.dumps(scene.to_dict(),indent=2)+"\n")
    (folder/"render_request.json").write_text(json.dumps(request,indent=2)+"\n")
    (folder/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
