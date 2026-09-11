import { geographicPosition, q4, validateScene, validateFrame, type SceneBackend,type SceneData,type Frame,type LayerOptions } from './protocol';
import { Vector3 } from 'three';

export async function createMap(host:HTMLElement,data:SceneData,options:LayerOptions,onError:(m:string)=>void):Promise<SceneBackend> {
  validateScene(data);if(!data.origin_lla)throw Error('地图需要 origin_lla');
  const M=await import('maplibre-gl');await import('maplibre-gl/dist/maplibre-gl.css');
  const origin=data.origin_lla;
  const map=new M.Map({container:host,center:[origin[1]*180/Math.PI,origin[0]*180/Math.PI],zoom:13,pitch:45,style:options.mapStyle||{version:8,sources:{},layers:[{id:'background',type:'background',paint:{'background-color':'#e0e9e8'}}]},attributionControl:{compact:true}});
  map.addControl(new M.NavigationControl());map.on('error',()=>onError('地图资源加载失败；请检查样式/DEM 的 URL 与 CORS。'));
  let disposed=false,latest:Frame|undefined,lastTime=-1;
  let track:number[][]=[];
  const markers=data.entities.map(e=>{
    const element=document.createElement('div');element.style.cssText='width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:24px solid #123e68';element.title=e.id;
    return new M.Marker({element,rotationAlignment:'map'}).setLngLat([origin[1]*180/Math.PI,origin[0]*180/Math.PI]).addTo(map);
  });
  map.on('load',()=>{if(disposed)return;
    if(options.terrainUrl){map.addSource('terrain',{type:'raster-dem',url:options.terrainUrl,tileSize:256});map.setTerrain({source:'terrain',exaggeration:1});}
    map.addSource('track',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
    map.addLayer({id:'track',type:'line',source:'track',paint:{'line-width':3,'line-color':'#163e63'}});
    if(latest)update(latest);
  });
  function update(frame:Frame) {validateFrame(frame,data);latest=frame;
    frame.poses.forEach((p,i)=>{const [lat,lon]=geographicPosition(p.position_ned_m,origin),forward=new Vector3(1,0,0).applyQuaternion(q4(p.quaternion_nb));markers[i].setLngLat([lon*180/Math.PI,lat*180/Math.PI]).setRotation(Math.atan2(forward.y,forward.x)*180/Math.PI);
      if(i===0){if(frame.time_s<lastTime)track=[];if(frame.time_s!==lastTime){track.push([lon*180/Math.PI,lat*180/Math.PI]);if(track.length>4096)track.shift();}lastTime=frame.time_s;}
    });
    const source=map.getSource('track') as import('maplibre-gl').GeoJSONSource|undefined;
    if(source&&track.length>1)source.setData({type:'Feature',properties:{},geometry:{type:'LineString',coordinates:track}});
  }
  return {update,draw(){},resize(){map.resize();},dispose(){disposed=true;markers.forEach(m=>m.remove());map.remove();}};
}
