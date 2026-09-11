import { Matrix4, Quaternion, Vector3 } from 'three';

export type V3 = [number, number, number];
export type Q4 = [number, number, number, number];
export interface Entity { id: string; asset_uri: string | null; scale: number; body_from_asset_quaternion: Q4 }
export interface SceneData { schema_version: 1; kind: 'scene'; position_frame: 'local_ned'; position_unit: 'm'; quaternion_order: 'wxyz'; rotation: 'body_frd_to_ned'; origin_lla: V3 | null; entities: Entity[] }
export interface Pose { position_ned_m: V3; quaternion_nb: Q4 }
export interface Frame { schema_version: 1; kind: 'frame'; world_id: number; episode_id: number; time_s: number; tick: number | null; source_ticks: [number,number]; poses: Pose[] }
export interface RenderRequest { protocol_version: 1; kind: 'render'; sequence: number; frame: Frame; camera?: {position_ned_m:V3;target_ned_m:V3;up_ned:V3;vertical_fov_rad:number}|null; joints?: {entity_id:string;node:string;angle_rad:number;axis_local:V3}[] }
export interface Recording { scene: SceneData; frames: Frame[]; source: string }
export interface LayerOptions { preset: 'city'|'mountains'|'ocean'|'altitude'; tilesUrl?: string; tilesFrame?: 'ecef'|'local-y-up'; attribution?: string; mapStyle?: string; terrainUrl?: string }
export interface SceneBackend { update(frame:Frame):void; resize():void; draw(dt:number):void; dispose():void }

function vector(v:unknown,n:number): asserts v is number[] {
  if (!Array.isArray(v)||v.length!==n||!v.every(x=>typeof x==='number'&&Number.isFinite(x))) throw Error('坐标必须为有限数值向量');
}
function quaternion(v:unknown) { vector(v,4); if(Math.hypot(...v)<1e-12) throw Error('四元数不能为零'); }
export function validateScene(s:SceneData) {
  if(!s||s.schema_version!==1||s.kind!=='scene'||s.position_frame!=='local_ned'||s.position_unit!=='m'||s.quaternion_order!=='wxyz'||s.rotation!=='body_frd_to_ned') throw Error('不支持的 Scene 协议或坐标约定');
  if(!Array.isArray(s.entities)||!s.entities.length||s.entities.length>256) throw Error('Scene 需要 1–256 个实体');
  const ids=new Set(); for(const e of s.entities) {
    if(typeof e.id!=='string'||!e.id||ids.has(e.id)||!Number.isFinite(e.scale)||e.scale<=0) throw Error('实体 ID 或比例无效');
    ids.add(e.id); quaternion(e.body_from_asset_quaternion);
  }
  if(s.origin_lla!==null) { vector(s.origin_lla,3); if(Math.abs(s.origin_lla[0])>Math.PI/2) throw Error('纬度超出范围'); }
}
export function validateFrame(f:Frame,s:SceneData) {
  if(!f||f.schema_version!==1||f.kind!=='frame'||!Number.isFinite(f.time_s)||f.time_s<0||!Array.isArray(f.poses)||f.poses.length!==s.entities.length) throw Error('Frame 时间或实体数量无效');
  if(![f.world_id,f.episode_id].every(x=>Number.isSafeInteger(x)&&x>=0)) throw Error('World/episode ID 无效');
  vector(f.source_ticks,2);
  if(!f.source_ticks.every(x=>Number.isSafeInteger(x)&&x>=0)||f.source_ticks[1]<f.source_ticks[0]||(f.tick!==null&&(!Number.isSafeInteger(f.tick)||f.tick<0||f.source_ticks.some(x=>x!==f.tick)))) throw Error('tick 区间无效');
  for(const p of f.poses) { vector(p.position_ned_m,3); quaternion(p.quaternion_nb); }
}
export function validateRecording(value:Recording) {
  validateScene(value.scene);
  if(!Array.isArray(value.frames)||!value.frames.length||value.frames.length>100000) throw Error('轨迹需要 1–100000 帧');
  value.frames.forEach((f,i)=>{validateFrame(f,value.scene); const prev=value.frames[i-1]; if(prev&&(f.world_id!==prev.world_id||f.episode_id!==prev.episode_id||f.time_s<=prev.time_s)) throw Error('回放文件必须是一个 world/episode，时间严格递增');});
  return {...value,source:typeof value.source==='string'?value.source:'导入轨迹（未声明来源）'};
}
export const BASIS = new Matrix4().set(0,1,0,0, 0,0,-1,0, -1,0,0,0, 0,0,0,1);
export function q4(q:Q4) { return new Quaternion(q[1],q[2],q[3],q[0]).normalize(); }
export function ned(v:V3) { return new Vector3(v[1],-v[2],-v[0]); }
export function assetRotation(p:Pose,e:Entity) {
  // Geometry is asset-local: A @ R_nb @ R_body_asset. No ad-hoc quaternion swapping.
  return new Quaternion().setFromRotationMatrix(BASIS).multiply(q4(p.quaternion_nb)).multiply(q4(e.body_from_asset_quaternion));
}
export function cesiumAssetMatrix(p:Pose,e:Entity,origin:V3) {
  // Cesium Model applies Y-up -> Z-up and Z-forward -> X-forward internally.
  // Cancel that correction so the effective transform remains ECEF_NED R_nb R_body_asset.
  const correction=new Matrix4().makeRotationX(Math.PI/2).multiply(new Matrix4().makeRotationY(Math.PI/2));
  return ecefFromNed(origin).setPosition(0,0,0)
    .multiply(new Matrix4().makeRotationFromQuaternion(q4(p.quaternion_nb)))
    .multiply(new Matrix4().makeRotationFromQuaternion(q4(e.body_from_asset_quaternion)))
    .multiply(correction.invert());
}
export function interpolate(a:Frame,b:Frame,t:number):Frame {
  if(a.world_id!==b.world_id||a.episode_id!==b.episode_id||a.poses.length!==b.poses.length) throw Error('不能跨 world/episode 插值');
  const u=b.time_s===a.time_s?0:Math.max(0,Math.min(1,(t-a.time_s)/(b.time_s-a.time_s)));
  return {...a,time_s:a.time_s+(b.time_s-a.time_s)*u,tick:null,source_ticks:[a.source_ticks[0],b.source_ticks[1]],poses:a.poses.map((p,i)=>{
    const q=q4(p.quaternion_nb).slerp(q4(b.poses[i].quaternion_nb),u);
    return {position_ned_m:p.position_ned_m.map((x,j)=>x+(b.poses[i].position_ned_m[j]-x)*u) as V3,quaternion_nb:[q.w,q.x,q.y,q.z]};
  })};
}
export function ecefOrigin(lla:V3) {
  const [lat,lon,h]=lla, n=6378137/Math.sqrt(1-0.0066943799901413165*Math.sin(lat)**2);
  return new Vector3((n+h)*Math.cos(lat)*Math.cos(lon),(n+h)*Math.cos(lat)*Math.sin(lon),(n*(1-0.0066943799901413165)+h)*Math.sin(lat));
}
export function ecefFromNed(lla:V3) {
  const [lat,lon]=lla, s=Math.sin(lat),c=Math.cos(lat),sl=Math.sin(lon),cl=Math.cos(lon),o=ecefOrigin(lla);
  return new Matrix4().set(-s*cl,-sl,-c*cl,o.x, -s*sl,cl,-c*sl,o.y, c,0,-s,o.z, 0,0,0,1);
}
export function geographicPosition(position:V3,origin:V3):V3 {
  const p=new Vector3(...position).applyMatrix4(ecefFromNed(origin)), lon=Math.atan2(p.y,p.x),r=Math.hypot(p.x,p.y),e2=0.0066943799901413165;
  let lat=Math.atan2(p.z,r*(1-e2));
  for(let i=0;i<8;i++) { const n=6378137/Math.sqrt(1-e2*Math.sin(lat)**2); lat=Math.atan2(p.z+e2*n*Math.sin(lat),r); }
  const n=6378137/Math.sqrt(1-e2*Math.sin(lat)**2),h=Math.abs(Math.cos(lat))>1e-8?r/Math.cos(lat)-n:Math.abs(p.z)-n*(1-e2);
  return [lat,lon,h];
}
