"""Bounded CPU video decoding and a shared-frame local viewing page, no simulation."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from rlflow.store import canonical
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_decision_packet import persist_exact
from tools.r6_video_trajectory import bounded_json
from tools import r6_video_presentation as presentation

CAP=2*1024**2


def require(ok,reason):
    if not ok:raise RecoveryError('VIDEO_VALIDATION_'+reason)


def inputs(pin):
    q=bounded_json(pin,256*1024)
    require(set(q)=={'render_manifest','extraction','renderer','source_snapshot','inventory','closure','postrun','spec'},'INPUT_FIELDS')
    for name in ('render_manifest','extraction','renderer','inventory','closure','postrun','spec'):checked(q[name])
    inv=bounded_json(q['inventory'],64*1024**2)
    require(inv['output_inventory_complete']is True and inv['closure']==q['closure'] and inv['spec']==q['spec'],'INVENTORY')
    manifest=bounded_json(q['render_manifest'],1024**2)
    sources=[file_pin(Path(q['source_snapshot'])/p) for p in (
        'isaaclab-training/instinctlab/task/scene_h1.py','isaaclab-training/instinctlab/assets/taslab_uav/taslab_uav.usd')]
    require(manifest['scene_sources']==sources,'SCENE_SOURCES')
    pins=[q['render_manifest']]+[v[k] for v in manifest['videos'] for k in ('video','telemetry')]
    for pin in pins:
        rows=[r for r in inv['files'] if r['path']==pin['path']]
        require(len(rows)==1 and rows[0]['sha256']==pin['sha256'] and rows[0]['bytes']==checked(pin).stat().st_size,'ORIGINAL_INVENTORY')
    return q


def viewing_rows(extraction,verified):
    clips=presentation.load(extraction);rows=[]
    for item,video in zip(clips,verified['videos']):
        c=item['decoded']['clip'];assignment=c['assignment']
        rows.append(dict(index=video['index'],assignment_index=c['assignment_index'],model_role=assignment['model_role'],
            evaluation_seed=assignment['evaluation_seed'],checkpoint=c['checkpoint'],block_id=c['block_id'],lane=c['lane'],
            domain=c['domain'],motion_pass=c['motion_pass'],selection_reasons=c['selection_reasons'],
            source_spec=c['source_spec'],raw=c['raw'],video=video['video'],telemetry=video['telemetry'],frames=video['frames']))
    groups=[];seen=set()
    for row in rows:
        key=(row['block_id'],row['lane'],row['evaluation_seed'])
        if key in seen:continue
        seen.add(key);members=[r['index'] for r in rows if (r['block_id'],r['lane'],r['evaluation_seed'])==key]
        require(len({rows[i]['frames'] for i in members})==1,'PAIRED_CLOCK')
        groups.append(dict(block_id=key[0],lane=key[1],evaluation_seed=key[2],videos=members,
            comparison='SAME_CONDITION_AND_RANDOM_REALIZATION' if len(members)>1 else 'SINGLE_REGISTERED_TRAJECTORY'))
    return rows,groups


def page(rows,groups,directory):
    body=['<!doctype html><meta charset="utf-8"><title>R6 recorded replay</title>',
        '<style>body{font:16px sans-serif;margin:24px;background:#eee}section{margin-bottom:32px} .clips{display:flex;flex-wrap:wrap;gap:12px}video{width:480px;max-width:90vw}figure{margin:0}button{margin:8px}input{width:300px}</style>',
        '<h1>R6 原始评估轨迹回放</h1><p>无新物理积分。共同帧号显示相同状态时刻；控制命令来自前一控制时刻。逐帧查看失败和重置，不用录像代替全部 lane 的定量结果。尚需独立视觉审计。</p>']
    for group in groups:
        count=rows[group['videos'][0]]['frames']
        body.append('<section data-count="'+str(count)+'"><h2>'+html.escape(f"{group['block_id']} / lane {group['lane']} / seed {group['evaluation_seed']}")+'</h2><button class="prev">上一帧</button><button class="next">下一帧</button><button class="play">播放/暂停</button><input type="range" min="0" max="'+str(count-1)+'" value="0"><output></output><div class="clips">')
        for index in group['videos']:
            row=rows[index];src=quote(os.path.relpath(row['video']['path'],directory),safe='/')
            label=f"{row['model_role']} / assignment {row['assignment_index']} / motion_pass={row['motion_pass']} / {row['checkpoint']['sha256']}"
            body.append('<figure><figcaption>'+html.escape(label)+'</figcaption><video preload="auto" muted src="'+html.escape(src,quote=True)+'"></video></figure>')
        body.append('</div></section>')
    body.append('''<script>
for(const s of document.querySelectorAll('section')){
 const vs=[...s.querySelectorAll('video')], slider=s.querySelector('input'), out=s.querySelector('output');
 let frame=0,playing=false,busy=false;
 async function show(n){if(busy)return;busy=true;frame=Math.max(0,Math.min(+s.dataset.count-1,n));slider.value=frame;
  try{await Promise.all(vs.map(v=>new Promise((resolve,reject)=>{
   const t=(frame+0.5)/10;
   if(v.readyState>=2&&Math.abs(v.currentTime-t)<.001){resolve();return;}
   const timeout=setTimeout(()=>{cleanup();reject(Error('视频定位超时'));},10000);
   function cleanup(){clearTimeout(timeout);v.removeEventListener('seeked',done);v.removeEventListener('error',fail);}
   function done(){cleanup();resolve();}function fail(){cleanup();reject(Error('视频读取失败'));}
   v.addEventListener('seeked',done);v.addEventListener('error',fail);v.currentTime=t;
  })));
  out.textContent=`帧 ${frame}；控制 t=${(frame/10).toFixed(1)}s；状态 t=${((frame+1)/10).toFixed(1)}s`;
  }catch(e){playing=false;out.textContent=e.message;}finally{busy=false;}
 }
 slider.oninput=()=>{playing=false;show(+slider.value);};
 s.querySelector('.prev').onclick=()=>{playing=false;show(frame-1);};
 s.querySelector('.next').onclick=()=>{playing=false;show(frame+1);};
 s.querySelector('.play').onclick=()=>{playing=!playing;};
 setInterval(()=>{if(playing&&!busy){if(frame+1>=+s.dataset.count)playing=false;else show(frame+1);}},100);
 Promise.all(vs.map(v=>new Promise(resolve=>{if(v.readyState>=1)resolve();else v.addEventListener('loadedmetadata',resolve,{once:true});}))).then(()=>show(0));
}
</script>''')
    return '\n'.join(body).encode()


def prepare(input_pin,directory):
    q=inputs(input_pin);directory=Path(directory).absolute()
    require(not directory.is_symlink() and directory.resolve()==directory,'OUTPUT_ALIAS')
    for pin in q.values():
        if isinstance(pin,dict):require(not checked(pin).is_relative_to(directory),'OUTPUT_OVERLAP')
    if directory.exists():require(set(p.name for p in directory.iterdir())<={'result.json','VIDEO_MANIFEST.json','watch.html','README.txt'},'UNCLAIMED_OUTPUT')
    verified=presentation.verify(q['render_manifest'],q['extraction'],renderer_pin=q['renderer'])
    rows,groups=viewing_rows(q['extraction'],verified)
    selection=bounded_json(bounded_json(q['extraction'],256*1024)['selection'],256*1024)
    def save(name,data):
        if not isinstance(data,bytes):data=canonical(data)+b'\n'
        size=sum(p.stat().st_size for p in directory.iterdir() if p.name!=name) if directory.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP');persist_exact(directory/name,data);return file_pin(directory/name)
    viewer=save('watch.html',page(rows,groups,directory))
    instructions=save('README.txt',('Open watch.html locally with original MP4 paths preserved. Shared frame controls align the same recorded state times; slow decoding pauses progression instead of skipping frames. Separate seeds are not paired realizations. All clips retain failures/resets. Visual pose-to-pixel audit and final scientific review remain required.\n').encode())
    manifest=save('VIDEO_MANIFEST.json',dict(schema_version='r6-video-delivery/1',status='CPU_VALIDATED_REQUIRES_VISUAL_AUDIT',
        inputs=input_pin,validation=verified,videos=rows,groups=groups,unavailable_categories=selection.get('unavailable_categories',[]),
        viewer=viewer,instructions=instructions,physics_reintegrated=False,scientific_acceptance=False))
    require(inputs(input_pin)==q,'INPUT_DRIFT')
    result=dict(status='VIDEOS_CPU_VALIDATED_REQUIRES_VISUAL_AUDIT',inputs=input_pin,manifest=manifest,viewer=viewer,
        instructions=instructions,validator=file_pin(__file__),presentation=file_pin(presentation.__file__),
        video_count=len(rows),total_frames=sum(v['frames'] for v in rows),scientific_acceptance=False,dispatch_authorized=False)
    save('result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('inputs',type=Path);parser.add_argument('--sha256',required=True);parser.add_argument('output',type=Path)
    a=parser.parse_args();print(json.dumps(prepare(dict(path=str(a.inputs.resolve()),sha256=a.sha256),a.output)))


if __name__=='__main__':main()
