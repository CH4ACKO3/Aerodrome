import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createLayer, disposeTree } from './layers';
import { assetRotation, ned, validateScene, validateFrame, type SceneBackend, type SceneData, type Frame, type LayerOptions, type RenderRequest } from './protocol';

export class ThreeBackend implements SceneBackend {
  private renderer:T.WebGLRenderer;
  private scene=new T.Scene();
  private camera=new T.PerspectiveCamera(50,1,0.2,600000);
  private controls:OrbitControls;
  private models:T.Group[]=[];
  private layer:Awaited<ReturnType<typeof createLayer>>|undefined;
  private observer:ResizeObserver;
  private lastTarget:T.Vector3|undefined;
  private disposed=false;
  private rest=new Map<T.Object3D,T.Quaternion>();
  private lastEpisode='';
  private sequence=0;
  private frame:Frame|undefined;
  private contextLost=(event:Event)=>{event.preventDefault();this.onError('WebGL 上下文丢失；请重新加载场景。');};
  private constructor(private host:HTMLElement,private data:SceneData,private onError:(m:string)=>void) {
    validateScene(data);
    this.renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));this.renderer.outputColorSpace=T.SRGBColorSpace;
    this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=.8;
    this.renderer.domElement.setAttribute('aria-label','三维飞行场景；拖动旋转视角，滚轮缩放');
    this.renderer.domElement.addEventListener('webglcontextlost',this.contextLost);
    host.append(this.renderer.domElement);
    this.camera.position.set(65,1035,70);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    this.controls.minDistance=5;this.controls.maxDistance=150000;
    this.scene.add(new T.HemisphereLight(0xcbe3ff,0x4a5141,2));
    const sun=new T.DirectionalLight(0xffffff,2.5);sun.position.set(500,1200,400);this.scene.add(sun);
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(host);this.resize();
  }
  static async create(host:HTMLElement,data:SceneData,options:LayerOptions,assets:Record<string,string>,onError:(m:string)=>void) {
    const backend=new ThreeBackend(host,data,onError);
    try {
      backend.layer=await createLayer(options,backend.renderer,backend.camera,data.origin_lla,onError);backend.scene.add(backend.layer.group);
      const loader=new GLTFLoader();
      for(const entity of data.entities) {
        const root=new T.Group();backend.models.push(root);backend.scene.add(root);root.scale.setScalar(entity.scale);
        const url=entity.asset_uri?assets[entity.asset_uri]:undefined;
        if(!url) throw Error(`未配置实体 ${entity.id} 的 glTF/GLB 资产映射`);
        const gltf=await loader.loadAsync(url);root.add(gltf.scene);
        gltf.scene.traverse(node=>backend.rest.set(node,node.quaternion.clone()));
      }
      return backend;
    } catch(error) {backend.dispose();throw error;}
  }
  update(frame:Frame) {
    validateFrame(frame,this.data);this.frame=frame;
    const episode=`${frame.world_id}:${frame.episode_id}`;
    if(episode!==this.lastEpisode){this.rest.forEach((q,node)=>node.quaternion.copy(q));this.lastEpisode=episode;this.lastTarget=undefined;}
    frame.poses.forEach((pose,i)=>{this.models[i].position.copy(ned(pose.position_ned_m));this.models[i].quaternion.copy(assetRotation(pose,this.data.entities[i]));});
    const target=this.models[0].position;
    if(this.lastTarget)this.camera.position.add(target.clone().sub(this.lastTarget));
    else this.camera.position.copy(target).add(new T.Vector3(45,22,52));
    this.controls.target.copy(target);this.lastTarget=target.clone();
  }
  /** RenderRequest v1 adapter. Receipt means a WebGL draw was submitted, not GPU timing. */
  submit(request:RenderRequest) {
    try {
      if(this.disposed||request.protocol_version!==1||request.kind!=='render'||!Number.isSafeInteger(request.sequence)||request.sequence<=this.sequence)throw Error('无效或重复的渲染请求');
      this.sequence=request.sequence;this.update(request.frame);
      if(request.camera) {
        const c=request.camera,position=ned(c.position_ned_m),target=ned(c.target_ned_m),up=ned(c.up_ned);
        if(![...position,...target,...up,c.vertical_fov_rad].every(Number.isFinite)||c.vertical_fov_rad<=0||c.vertical_fov_rad>=Math.PI||target.clone().sub(position).cross(up).length()<1e-10)throw Error('相机无效');
        this.camera.position.copy(position);this.camera.up.copy(up);this.camera.fov=T.MathUtils.radToDeg(c.vertical_fov_rad);this.camera.updateProjectionMatrix();this.controls.target.copy(target);
      } else {this.camera.up.set(0,1,0);this.camera.fov=50;this.camera.updateProjectionMatrix();}
      for(const joint of request.joints??[]) {
        const index=this.data.entities.findIndex(e=>e.id===joint.entity_id),node=this.models[index]?.getObjectByName(joint.node),axis=new T.Vector3(...joint.axis_local);
        if(!node||!this.rest.has(node)||!Number.isFinite(joint.angle_rad)||![...axis].every(Number.isFinite)||axis.length()<1e-12)throw Error(`关节无效：${joint.node}`);
        node.quaternion.copy(this.rest.get(node)!).multiply(new T.Quaternion().setFromAxisAngle(axis.normalize(),joint.angle_rad));
      }
      this.draw(0);
      return {protocol_version:1,kind:'receipt',sequence:request.sequence,status:'presented'};
    } catch(error) {return {protocol_version:1,kind:'receipt',sequence:request.sequence,status:'failed',detail:String(error)};}
  }
  resize(){if(this.disposed)return;const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setSize(w,h,false);this.camera.aspect=w/Math.max(1,h);this.camera.updateProjectionMatrix();}
  draw(dt:number){if(this.disposed||!this.frame)return;this.controls.update();this.layer?.update(dt);this.renderer.render(this.scene,this.camera);}
  dispose(){if(this.disposed)return;this.disposed=true;this.observer.disconnect();this.controls.dispose();this.layer?.dispose();this.models.forEach(disposeTree);this.renderer.domElement.removeEventListener('webglcontextlost',this.contextLost);this.renderer.dispose();this.renderer.forceContextLoss();this.renderer.domElement.remove();}
}
