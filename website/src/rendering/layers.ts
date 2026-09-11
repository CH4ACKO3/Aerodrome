import * as T from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Water } from 'three/addons/objects/Water.js';
import { BASIS, ecefFromNed, type LayerOptions, type V3 } from './protocol';

export function disposeTree(root:T.Object3D) {
  const textures=new Set<T.Texture>(), materials=new Set<T.Material>(), geometries=new Set<T.BufferGeometry>();
  root.traverse(object=>{const mesh=object as T.Mesh; if(mesh.geometry)geometries.add(mesh.geometry); if(mesh.material)for(const m of Array.isArray(mesh.material)?mesh.material:[mesh.material])materials.add(m);});
  for(const m of materials) { for(const v of Object.values(m))if(v instanceof T.Texture)textures.add(v); m.dispose(); }
  geometries.forEach(g=>g.dispose()); textures.forEach(t=>t.dispose()); root.removeFromParent();
}
export async function createLayer(options:LayerOptions,renderer:T.WebGLRenderer,camera:T.PerspectiveCamera,origin:V3|null,onError:(message:string)=>void) {
  const group=new T.Group(); const sky=new Sky(); sky.scale.setScalar(500000); group.add(sky);
  sky.material.uniforms.sunPosition.value.set(0.4,0.65,-0.5); sky.material.uniforms.turbidity.value=options.preset==='altitude'?1:4;
  sky.material.uniforms.rayleigh.value=options.preset==='altitude'?0.35:2;
  let water:Water|undefined, tiles:import('3d-tiles-renderer/three').TilesRenderer|undefined;
  const resources:{dispose():void}[]=[];
  if(options.preset==='ocean') {
    // Deterministic normal texture generated locally; no unlicensed image downloads.
    const size=64,bytes=new Uint8Array(size*size*4);
    for(let y=0;y<size;y++)for(let x=0;x<size;x++){const i=(y*size+x)*4; bytes[i]=128+30*Math.sin(x*.5+y*.3); bytes[i+1]=128+30*Math.cos(y*.6);bytes[i+2]=250;bytes[i+3]=255;}
    const normal=new T.DataTexture(bytes,size,size); normal.wrapS=normal.wrapT=T.RepeatWrapping;normal.needsUpdate=true;resources.push(normal);
    water=new Water(new T.PlaneGeometry(100000,100000),{textureWidth:256,textureHeight:256,waterNormals:normal,sunDirection:new T.Vector3(.4,.65,-.5).normalize(),sunColor:0xffffff,waterColor:0x07546a,distortionScale:2.5});
    water.rotation.x=-Math.PI/2; group.add(water);
  } else {
    const geometry=new T.PlaneGeometry(120000,120000,options.preset==='mountains'?128:1,options.preset==='mountains'?128:1); geometry.rotateX(-Math.PI/2);
    if(options.preset==='mountains') { const p=geometry.attributes.position;
      for(let i=0;i<p.count;i++){const x=p.getX(i),z=p.getZ(i); p.setY(i,Math.max(0,700*Math.sin(x/2800)*Math.cos(z/3100)+400*Math.sin((x+z)/1400)-100));}geometry.computeVertexNormals();
    }
    group.add(new T.Mesh(geometry,new T.MeshStandardMaterial({color:options.preset==='mountains'?0x6d8463:options.preset==='altitude'?0x355971:0x758473,roughness:1})));
    if(options.preset==='city') {
      const buildings=new T.InstancedMesh(new T.BoxGeometry(1,1,1),new T.MeshStandardMaterial({color:0xa4adb3,roughness:.8}),225);
      let k=0;for(let x=-7;x<=7;x++)for(let z=-7;z<=7;z++){const h=30+((x*31+z*17+900)%8)*22; buildings.setMatrixAt(k++,new T.Matrix4().compose(new T.Vector3(x*180+90,h/2,z*180),new T.Quaternion(),new T.Vector3(65,h,85)));}group.add(buildings);
    }
  }
  if(options.tilesUrl) {
    if(options.tilesFrame==='ecef'&&!origin)throw Error('ECEF 瓦片需要 Scene.origin_lla');
    const {TilesRenderer}=await import('3d-tiles-renderer/three');
    tiles=new TilesRenderer(options.tilesUrl); tiles.setCamera(camera); tiles.setResolutionFromRenderer(camera,renderer);
    tiles.errorTarget=12; tiles.lruCache.maxSize=160; tiles.lruCache.minSize=100;
    if(options.tilesFrame==='ecef') {tiles.group.matrixAutoUpdate=false;tiles.group.matrix.copy(BASIS).multiply(ecefFromNed(origin!).invert());}
    tiles.addEventListener('load-error',()=>onError('瓦片加载失败；请检查 URL、CORS 和数据授权。基础场景仍可使用。'));
    group.add(tiles.group);
  }
  return {group,update(dt:number){if(water)water.material.uniforms.time.value+=dt;if(tiles){tiles.setResolutionFromRenderer(camera,renderer);tiles.update();}},dispose(){tiles?.dispose();disposeTree(group);resources.forEach(r=>r.dispose());}};
}
