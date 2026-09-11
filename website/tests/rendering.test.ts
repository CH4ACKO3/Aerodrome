import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import { Vector3, Matrix4 } from 'three';
import { ned,assetRotation,cesiumAssetMatrix,ecefFromNed,geographicPosition,interpolate,validateRecording } from '../src/rendering/protocol.ts';
const recording=validateRecording(JSON.parse(readFileSync(new URL('../public/examples/rigid-turn.json',import.meta.url),'utf8')));
test('Python Scene/Frame export maps NED and asset axes correctly',()=>{
  assert.deepEqual(ned([1,2,-3]).toArray(),[2,3,-1]);
  const q=assetRotation(recording.frames[0].poses[0],recording.scene.entities[0]);
  assert.ok(new Vector3(0,0,-1).applyQuaternion(q).distanceTo(new Vector3(0,0,-1))<1e-12);
});
test('WGS84 tangent frame and ECEF position agree at equator',()=>{
  assert.deepEqual(new Vector3(10,20,-30).applyMatrix4(ecefFromNed([0,0,0])).toArray(),[6378167,20,10]);
  const [lat,lon,height]=geographicPosition([0,0,-1000],[0,0,0]);
  assert.equal(lat,0);assert.equal(lon,0);assert.ok(Math.abs(height-1000)<1e-7);
});
test('Cesium built-in glTF correction preserves northward nose and upward asset axis',()=>{
  const correction=new Matrix4().makeRotationX(Math.PI/2).multiply(new Matrix4().makeRotationY(Math.PI/2));
  const rotation=cesiumAssetMatrix(recording.frames[0].poses[0],recording.scene.entities[0],[0,0,0]).multiply(correction);
  assert.ok(new Vector3(0,0,-1).applyMatrix4(rotation).distanceTo(new Vector3(0,0,1))<1e-12);
  assert.ok(new Vector3(0,1,0).applyMatrix4(rotation).distanceTo(new Vector3(1,0,0))<1e-12);
});
test('interpolation retains source ticks and unit quaternion',()=>{
  const f=interpolate(recording.frames[0],recording.frames[1],.01);
  assert.equal(f.tick,null);assert.deepEqual(f.source_ticks,[0,2]);assert.ok(Math.abs(Math.hypot(...f.poses[0].quaternion_nb)-1)<1e-12);
  const flip=structuredClone(recording.frames[1]);flip.poses[0].quaternion_nb=flip.poses[0].quaternion_nb.map(v=>-v) as [number,number,number,number];
  const g=interpolate(recording.frames[0],flip,.01);assert.ok(Math.abs(g.poses[0].quaternion_nb[0])>.99);
});
test('reject unsupported units, invalid poses and cross-episode interpolation',()=>{
  const data=structuredClone(recording);(data.scene as any).position_unit='ft';assert.throws(()=>validateRecording(data));
  const invalid=structuredClone(recording);invalid.frames[0].poses[0].quaternion_nb=[0,0,0,0];assert.throws(()=>validateRecording(invalid));
  assert.throws(()=>interpolate(recording.frames[0],{...recording.frames[1],episode_id:1},.01));
});
