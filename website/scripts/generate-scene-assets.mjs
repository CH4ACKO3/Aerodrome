// Original educational geometry, metres, asset forward -Z/right +X/up +Y.
import * as T from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
globalThis.FileReader=class {
  readAsArrayBuffer(blob){blob.arrayBuffer().then(value=>{this.result=value;this.onloadend?.();});}
  readAsDataURL(blob){blob.arrayBuffer().then(value=>{this.result=`data:application/octet-stream;base64,${Buffer.from(value).toString('base64')}`;this.onloadend?.();});}
};
const dir=path.resolve(import.meta.dirname,'../public/models');await mkdir(dir,{recursive:true});
const gray=new T.MeshStandardMaterial({color:0x879ba8,metalness:.35,roughness:.6});
const dark=new T.MeshStandardMaterial({color:0x18323e,metalness:.3,roughness:.25});
const aircraft=new T.Group();aircraft.name='training_fighter';
function part(name,geometry,material=gray,position=[0,0,0]){const mesh=new T.Mesh(geometry,material);mesh.name=name;mesh.position.set(...position);aircraft.add(mesh);return mesh;}
const body=part('fuselage',new T.CylinderGeometry(.65,.9,11,12));body.rotation.x=Math.PI/2;
const nose=part('nose',new T.ConeGeometry(.65,3.4,12),gray,[0,0,-7.2]);nose.rotation.x=-Math.PI/2;
part('canopy',new T.SphereGeometry(1,12,8),dark,[0,.65,-2.7]).scale.set(.52,.5,1.7);
function triangle(points){const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(points,3));g.computeVertexNormals();return g;}
const wingMaterial=gray.clone();wingMaterial.side=T.DoubleSide;
part('wings',triangle([-.6,0,-2,-5.2,0,2.5,-.6,0,2.8, .6,0,2.8,5.2,0,2.5,.6,0,-2]),wingMaterial);
const elevator=part('elevator',new T.BoxGeometry(5.2,.12,1.35),gray,[0,.1,4.5]);
part('rudder',triangle([0,0,0,0,2.6,1.5,0,0,2.5]),wingMaterial,[0,.5,2.8]);
part('aileron_left',new T.BoxGeometry(1.5,.1,.45),gray,[-3.5,0,2.5]);
part('aileron_right',new T.BoxGeometry(1.5,.1,.45),gray,[3.5,0,2.5]);
part('exhaust',new T.CylinderGeometry(.65,.7,.8,12),dark,[0,0,5.5]).rotation.x=Math.PI/2;
const exporter=new GLTFExporter();
await writeFile(path.join(dir,'training-fighter.glb'),Buffer.from(await exporter.parseAsync(aircraft,{binary:true})));
const city=new T.Group();
for(let x=-3;x<=3;x++)for(let z=-3;z<=3;z++){
  const height=30+((x*13+z*7+200)%7)*20;
  const block=new T.Mesh(new T.BoxGeometry(60,height,70),gray);block.position.set(x*160+200,height/2,z*160);city.add(block);
}
await writeFile(path.join(dir,'training-city.glb'),Buffer.from(await exporter.parseAsync(city,{binary:true})));
await writeFile(path.join(dir,'tileset.json'),JSON.stringify({asset:{version:'1.1'},geometricError:0,root:{boundingVolume:{box:[200,0,100,600,0,0,0,600,0,0,0,100]},geometricError:0,refine:'ADD',transform:new T.Matrix4().makeRotationX(-Math.PI/2).elements,content:{uri:'training-city.glb'}}},null,2));
const A=new T.Matrix4().set(0,1,0,0,0,0,-1,0,-1,0,0,0,0,0,0,1);
const q=new T.Quaternion().setFromRotationMatrix(A.invert());
await writeFile(path.join(dir,'manifest.json'),JSON.stringify({asset_uri:'asset://aircraft/training-fighter',url:'training-fighter.glb',units:'m',forward:'-Z',up:'+Y',body_from_asset_quaternion:[q.w,q.x,q.y,q.z],nodes:['elevator','rudder','aileron_left','aileron_right'],source:'Original procedural teaching geometry; not a dimensionally accurate F-16',license:'CC0-1.0'},null,2));
console.log('Generated fighter GLB, city GLB, 3D Tiles fixture and asset manifest.');
