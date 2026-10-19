"""Build CPU-only final evidence audit and plots from immutable originals."""
from pathlib import Path
import json, hashlib, sys, shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r5-w-closure-auto'
def read(p): return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve()
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),sha256=h)
def save(name,v):
    p=BASE/name
    assert not p.exists(),p
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    invs=[]; allrows=[]; summaries=[]
    budget=read(BASE/'goal_budget.json')
    for entry in budget['reservations']['live_entries']:
        p=BASE/entry['inventory']; inv=read(p); checked=0
        for row in inv['files']:
            for key in ('original','archived'):
                if row[key]:
                    f=Path(row[key]); assert f.stat().st_size==row['bytes'] and pin(f)['sha256']==row['sha256'],f
                    checked+=1
            allrows.append(dict(experiment=entry['experiment_id'],**row))
        directory=p.parent.parent.parent
        mode=entry['mode']; name='R1-'+mode
        summary=read(directory/(name+'-execution-summary.json'))
        assert summary==entry and summary['settled'] and summary['postrun_queries_used']==2
        assert len(list(directory.glob(name+'-postrun-*.json')))==2
        assert summary['owned_cleanup']=='NORMAL_ACK_2_STABLE_EMPTY_NO_SURVIVORS'
        invs.append(dict(experiment=entry['experiment_id'],inventory=pin(p),entries=len(inv['files']),checked=checked,
            execution_summary=pin(directory/(name+'-execution-summary.json')),final_job=pin(directory/(name+'-final-job.json')),
            postrun=[pin(directory/(name+f'-postrun-{i}.json')) for i in range(2)]))
        summaries.append(summary)
    save('FINAL_ORIGINALS_INDEX.json',dict(inventories=invs,files=allrows,all_original_and_archive_hashes_verified=True))
    entries=budget['reservations']['live_entries']; snaps=budget['reservations']['source_snapshots']
    conservative={k:budget['used'][k]+sum(e['reserved'][k] for e in entries) for k in entries[0]['reserved']}
    output_reserved=sum(e['output_cap_bytes'] for e in entries)+sum(s['reserved_bytes'] for s in snaps)
    assert output_reserved==66*2**30 and len(entries)==7 and len(snaps)==5
    assert all(conservative[k]<=budget['caps'][k] for k in conservative)
    assert shutil.disk_usage(ROOT).free>=30*2**30
    save('FINAL_RESOURCE_AUDIT.json',dict(budget=pin(BASE/'goal_budget.json'),caps=budget['caps'],used=budget['used'],observed=budget['observed'],
        conservative_used_plus_workload_reservations=conservative,live_entries=summaries,source_snapshots=snaps,
        output_live_plus_snapshots_reserved_bytes=output_reserved,metadata_allowance_bytes=2*2**30,total_with_metadata_reserve_bytes=68*2**30,
        stage_cap_bytes=72*2**30,master_cap_bytes=128*2**30,disk_free_bytes=shutil.disk_usage(ROOT).free,disk_floor_bytes=30*2**30,
        note='Observed bytes are not added again to their full output caps. Four INCOMPLETE workloads keep full logical reserves. Three FINAL0 prelaunch failures retain start/init and full8GiB each. All five snapshots retained; no refund.'))
    independent=read(BASE/'INDEPENDENT_RESOURCE_V2_CPU_REPLAY.json')
    old=ROOT/'docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01'
    records=[]
    for directory,update,label,rawlabel in [
        (old,512,'R1 u512 margin0','formal_R1_u512'),
        (BASE/'experiments/R5-W-E1-CONTROL-01',512,'E1 u512 margin.15','w_R1_u512'),
        (BASE/'experiments/R5-W-L1-EVAL-01',640,'L1 u640 margin.15','w_R1_u640'),
        (BASE/'experiments/R5-W-L1-EVAL-01',768,'L1 u768 margin.15','w_R1_u768'),
        (BASE/'experiments/R5-W-L1-INDEPENDENT-RESOURCE-V2',640,'Independent u640 seed20261020','w_independent_R1_u640')]:
        ip=directory/'live/R1-research-formal-evaluation/INVENTORY.json'; inv=read(ip)
        out=Path(inv['roots']['attempt'])/'results/research-formal-evaluation'
        rp=out/(rawlabel+'.json'); result=read(rp)
        if 'original_metric_result' in result: result=result['original_metric_result']
        raw=out/'evaluation_raw'/rawlabel/'W.jsonl'
        records.append(dict(label=label,update=update,inventory=pin(ip),result=pin(rp),raw=pin(raw),cases=result['cases'],episodes=result['episodes']))
    fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True)
    for rec in records:
        rows=[json.loads(x) for x in Path(rec['raw']['path']).read_text().splitlines()][1:]
        rec['walls']={}
        for axis in range(2):
            eps=[e for e in rec['episodes'] if e['case']=='W' and axis*16<=e['env_index']<(axis+1)*16]
            rec['walls']['xy'[axis]]=dict(min_geometry_m=min(e['metrics']['min_distance'] for e in eps),max_stop_s=max(e['metrics']['stop_time'] for e in eps))
            values=[np.array([e['geometry_distance_m'] for e in eps]).mean(axis=0),
                np.array([[r['root_state_w'][l][7+axis] for l in range(axis*16,(axis+1)*16)] for r in rows]).mean(axis=1)]
            values += [np.array([[r[k][l] for l in range(axis*16,(axis+1)*16)] for r in rows]).reshape(len(rows),16).mean(axis=1) for k in ('q','ics_beta')]
            for col,v in enumerate(values): axes[axis,col].plot((np.arange(len(v))+(col<2))*.1,v,label=rec['label'])
        del rec['episodes']
    for axis in range(2):
        for col,title in enumerate(['Geometry (m), observation','World normal velocity (m/s), observation','q, control issue','ICS beta, control issue']):
            axes[axis,col].set_title('xy'[axis]+' wall: '+title);axes[axis,col].set_xlabel('seconds');axes[axis,col].grid(alpha=.2);axes[axis,col].set_xlim(0,6)
        axes[axis,0].axhline(.8,color='red',linestyle='--');axes[axis,0].axhline(1.1,color='gray',linestyle=':');axes[axis,1].axhline(.1,color='gray',linestyle=':')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(BASE/'FINAL_W_TIMING.png',dpi=150)
    save('FINAL_METHOD_COMPARISON.json',dict(runs=records,plot=pin(BASE/'FINAL_W_TIMING.png'),
        interpretation='R1->E1 isolates fixed-policy control margin .15m and establishes engineering benefit. E1->L1 adds mixed-nearwall training; all already pass in E1, so no isolated RL safety improvement established. L1 provides finite q-positive training and retained five-task capability. Independent seed keeps original fixed suite initial states.',
        timing='Group mean visualizations only; all original per-lane scores retained. Geometry/root observation at(i+1)*.1; q/beta control issue at i*.1.'))
    training=read(BASE/'reviews/l1-recovery1-actual-independent.json')
    manifest=dict(origin=training['accepted_checkpoints'][0],selected_model=training['accepted_checkpoints'][1],secondary_model=training['accepted_checkpoints'][2],
        candidate_source_identity=training['candidate_source_identity'],candidate_method_hashes=training['candidate_method_hashes'],
        independent_snapshot=pin(BASE/'experiments/R5-W-L1-INDEPENDENT-RESOURCE-V2/readonly-snapshot.json'),
        independent_protocol=pin(BASE/'experiments/R5-W-L1-INDEPENDENT-RESOURCE-V2/INDEPENDENT_PROTOCOL.json'),
        training=dict(updates='u513-768',transitions=131072,logical_updates=256,actor_adam=2048,critic_adam=2048,checkpoint_writes=2,
            center=98304,near_x=16384,near_y=16384,danger_q=2004,danger_q_x=1342,danger_q_y=662,ics_x=1946,ics_y=993,
            center_nonzero_commands=49152,x_nonzero_commands=8192,y_nonzero_commands=8192,
            note='One scientific training candidate; prelaunch failure and RECOVERY1 are not two candidates. Full trainer/Adam/RNG restored; fresh physical reset.'),
        evaluations=[dict(label=r['label'],update=r['update'],cases=r['cases'],result=r['result']) for r in records],
        exact_reproducible_inputs=[dict(experiment=e['experiment_id'],packet=pin(BASE/'experiments'/e['experiment_id']/'launch_inputs.proposed.json')) for e in entries])
    for c in training['accepted_checkpoints']: assert pin(c['path'])['sha256']==c['sha256']
    save('TRAINING_AND_EVALUATION_MANIFEST.json',manifest)
    print(json.dumps(dict(inventories=len(invs),files=len(allrows),reserved_GiB=output_reserved/2**30,runs=[dict(label=r['label'],walls=r['walls']) for r in records])))
if __name__=='__main__': main()
