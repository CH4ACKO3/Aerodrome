import type { SceneBackend, SceneData, LayerOptions } from './protocol';
import { ecefFromNed, cesiumAssetMatrix, validateScene, validateFrame } from './protocol';
import { Vector3 } from 'three';

export async function createCesium(host:HTMLElement,data:SceneData,options:LayerOptions,assets:Record<string,string>,onError:(m:string)=>void):Promise<SceneBackend> {
  validateScene(data);if(!data.origin_lla)throw Error('Cesium 场景需要 origin_lla');
  const base=import.meta.env.BASE_URL.replace(/\/$/,'');
  (window as Window & {CESIUM_BASE_URL?:string}).CESIUM_BASE_URL=base+'/vendor/cesium/';
  const C=await import('cesium');await import('cesium/Build/Cesium/Widgets/widgets.css');
  const viewer=new C.Viewer(host,{baseLayer:false,baseLayerPicker:false,geocoder:false,animation:false,timeline:false,homeButton:false,sceneModePicker:false,navigationHelpButton:false,fullscreenButton:false,infoBox:false,selectionIndicator:false,terrainProvider:new C.EllipsoidTerrainProvider(),requestRenderMode:true});
  try {
    viewer.scene.globe.baseColor=C.Color.fromCssColorString('#38667a');
    viewer.scene.renderError.addEventListener(()=>onError('Cesium 渲染失败；请重新加载场景或检查显卡支持。'));
    if(options.terrainUrl)viewer.terrainProvider=await C.CesiumTerrainProvider.fromUrl(options.terrainUrl);
    if(options.tilesUrl) {
      if(options.tilesFrame!=='ecef')throw Error('Cesium 需要 ECEF 定位的 3D Tiles');
      viewer.scene.primitives.add(await C.Cesium3DTileset.fromUrl(options.tilesUrl,{maximumScreenSpaceError:16,cacheBytes:128*1024*1024}));
    }
    const entities=data.entities.map(e=>{const uri=e.asset_uri?assets[e.asset_uri]:undefined;if(!uri)throw Error(`未配置资产 ${e.id}`);return viewer.entities.add({id:e.id,model:{uri,scale:e.scale,minimumPixelSize:50,maximumScale:100}});});
    const transform=ecefFromNed(data.origin_lla);
    return {
      update(frame) {validateFrame(frame,data);frame.poses.forEach((p,i)=>{
        const point=new Vector3(...p.position_ned_m).applyMatrix4(transform);
        const rotation=cesiumAssetMatrix(p,data.entities[i],data.origin_lla!);
        const m=C.Matrix4.fromArray(rotation.elements),q=C.Quaternion.fromRotationMatrix(C.Matrix4.getMatrix3(m,new C.Matrix3()));
        entities[i].position=new C.ConstantPositionProperty(new C.Cartesian3(point.x,point.y,point.z));entities[i].orientation=new C.ConstantProperty(q);
      });if(!viewer.trackedEntity)viewer.trackedEntity=entities[0];viewer.scene.requestRender();},
      draw(){},resize(){viewer.resize();},dispose(){viewer.destroy();}
    };
  } catch(error) {viewer.destroy();throw error;}
}
