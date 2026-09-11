import { interpolate, validateRecording, type Recording,type SceneBackend,type LayerOptions } from './protocol';

export async function mount(host:HTMLElement) {
  const field=<T extends HTMLElement>(name:string)=>host.querySelector<T>(`[data-${name}]`)!;
  const status=(value:string)=>{field('status').textContent=value;};
  const base=import.meta.env.BASE_URL.replace(/\/$/,'');
  let backend:SceneBackend|undefined,recording:Recording,playing=false,loading=false,disposed=false,generation=0,time=0,frameCount=0,windowStart=performance.now(),last=performance.now(),raf=0,sequence=0;
  const slider=field<HTMLInputElement>('time'),joint=field<HTMLInputElement>('joint');
  const play=field<HTMLButtonElement>('play'),reset=field<HTMLButtonElement>('reset'),unload=field<HTMLButtonElement>('unload'),load=field<HTMLButtonElement>('load');
  const input=(name:string)=>(field<HTMLInputElement>(name).value.trim());
  const url=(name:string)=>{const value=input(name);if(!value)return undefined;const parsed=new URL(value,location.href);if(!['http:','https:'].includes(parsed.protocol))throw Error('资源只支持 HTTP(S) URL');return parsed.href;};
  const result=await fetch(`${base}/examples/rigid-turn.json`);if(!result.ok)throw Error('参考轨迹加载失败');recording=validateRecording(await result.json());
  function stop(){playing=false;play.textContent='播放轨迹';}
  function controls(){play.disabled=reset.disabled=slider.disabled=unload.disabled=!backend||loading;joint.disabled=!backend||input('backend')!=='three'||loading;}
  function at(t:number) {let a=0,b=recording.frames.length-1;while(a+1<b){const mid=(a+b)>>1;if(recording.frames[mid].time_s<=t)a=mid;else b=mid;}return interpolate(recording.frames[a],recording.frames[b],t);}
  function update() {
    if(!backend)return;const frame=at(time);backend.update(frame);
    const capable=backend as SceneBackend & {submit?:(r:unknown)=>{status:string;detail?:string}};
    if(capable.submit&&Number(joint.value)!==0){const receipt=capable.submit({protocol_version:1,kind:'render',sequence:++sequence,frame,joints:[{entity_id:recording.scene.entities[0].id,node:'elevator',angle_rad:Number(joint.value)*Math.PI/180,axis_local:[1,0,0]}]});if(receipt.status==='failed')status(receipt.detail??'舵面预览失败');}
    slider.value=String(time);
  }
  async function open() {
    const token=++generation;loading=true;load.disabled=true;stop();controls();backend?.dispose();backend=undefined;field('canvas').replaceChildren();
    status('正在加载图形后端与模型…');sequence=0;
    try {
      const kind=input('backend'),options:LayerOptions={preset:input('preset') as LayerOptions['preset'],tilesUrl:url('tiles'),tilesFrame:input('tile-frame') as LayerOptions['tilesFrame'],mapStyle:url('style'),terrainUrl:url('terrain'),attribution:input('attribution')};
      if((options.tilesUrl||options.mapStyle||options.terrainUrl)&&!options.attribution)throw Error('请填写外部场景数据的来源与版权说明');
      if(kind==='three'&&(options.mapStyle||options.terrainUrl))throw Error('Three.js 的真实地形使用 3D Tiles；DEM/地图样式请选择对应后端');
      if(kind==='cesium'&&options.mapStyle)throw Error('地图样式 URL 仅用于 MapLibre');
      if(kind==='maplibre'&&options.tilesUrl)throw Error('3D Tiles 请使用 Three.js 或 CesiumJS');
      const customModel=url('model');
      const assets:Record<string,string>=customModel?Object.fromEntries(recording.scene.entities.map(e=>[e.asset_uri??'',customModel])):{'asset://aircraft/training-fighter':`${base}/models/training-fighter.glb`};
      let next:SceneBackend;
      if(kind==='three') {
        // Offline fixture exercises the real 3D Tiles loader without external accounts.
        if(options.preset==='city'&&!options.tilesUrl){options.tilesUrl=base+'/models/tileset.json';options.tilesFrame='local-y-up';}
        const {ThreeBackend}=await import('./three-backend');next=await ThreeBackend.create(field('canvas'),recording.scene,options,assets,status);
      } else if(kind==='cesium') {const {createCesium}=await import('./cesium-backend');next=await createCesium(field('canvas'),recording.scene,options,assets,status);}
      else {const {createMap}=await import('./maplibre-backend');next=await createMap(field('canvas'),recording.scene,options,status);}
      if(disposed||token!==generation){next.dispose();return;}backend=next;
      time=recording.frames[0].time_s;slider.min=String(time);slider.max=String(recording.frames.at(-1)!.time_s);joint.value='0';update();
      field('source').textContent=`轨迹来源：${recording.source}。场景来源：${options.attribution||'本项目示意资产；无真实城市/地形数据'}。`;
      status(kind==='three'?'场景已就绪 · 可以播放轨迹或检查姿态':kind==='cesium'?'Cesium 已就绪 · 默认椭球，无在线影像或地形服务':'地图已就绪 · 默认空白底图；可配置样式与 DEM 数据');
    }catch(error){status(`加载失败：${String(error)}。修改配置后点击「加载场景」重试。`);}
    finally{loading=false;load.disabled=false;controls();}
  }
  const onLoad=()=>void open();load.addEventListener('click',onLoad);
  play.addEventListener('click',()=>{playing=!playing;play.textContent=playing?'暂停轨迹':'播放轨迹';});
  reset.addEventListener('click',()=>{stop();time=recording.frames[0].time_s;update();});
  slider.addEventListener('input',()=>{stop();time=Number(slider.value);update();});
  joint.addEventListener('input',()=>{if(Number(joint.value)===0) {
    const b=backend as SceneBackend & {submit?:(r:unknown)=>unknown};b?.submit?.({protocol_version:1,kind:'render',sequence:++sequence,frame:at(time),joints:[{entity_id:recording.scene.entities[0].id,node:'elevator',angle_rad:0,axis_local:[1,0,0]}]});
  }else update();});
  unload.addEventListener('click',()=>{stop();++generation;backend?.dispose();backend=undefined;controls();status('图形资源已释放。手册与轨迹文件仍保留。');});
  field<HTMLInputElement>('file').addEventListener('change',async event=>{
    const file=(event.target as HTMLInputElement).files?.[0];if(!file)return;
    try{if(file.size>20*1024*1024)throw Error('轨迹文件最多 20 MiB');const value=JSON.parse(await file.text());recording=validateRecording(value.render??value);await open();}catch(error){status(`导入失败：${String(error)}`);}
  });
  function loop(now:number){if(disposed)return;const dt=Math.min((now-last)/1000,.1);last=now;
    if(backend&&!document.hidden){if(playing){time=Math.min(recording.frames.at(-1)!.time_s,time+dt);update();if(time>=recording.frames.at(-1)!.time_s)stop();}backend.draw(dt);frameCount++;
      if(now-windowStart>=1000){const fps=frameCount*1000/(now-windowStart);field('metrics').textContent=`回放 ${time.toFixed(2)} s · ${input('backend')==='three'?`WebGL 绘制 ${fps.toFixed(0)} FPS`:'地图引擎自主刷新，未测量 FPS'} · 仿真 TPS：离线回放不适用`;frameCount=0;windowStart=now;}}
    else {windowStart=now;frameCount=0;}raf=requestAnimationFrame(loop);
  }
  const onResult=(event:Event)=>{const value=(event as CustomEvent).detail;if(!value?.render)return;try{recording=validateRecording(value.render);void open();}catch(error){status(String(error));}};
  window.addEventListener('aerodrome:result',onResult);
  window.addEventListener('pagehide',()=>{disposed=true;++generation;cancelAnimationFrame(raf);backend?.dispose();window.removeEventListener('aerodrome:result',onResult);},{once:true});
  field<HTMLInputElement>('file').disabled=false;
  raf=requestAnimationFrame(loop);await open();
}
