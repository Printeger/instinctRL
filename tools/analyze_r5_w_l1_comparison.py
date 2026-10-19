"""CPU raw timing comparison; separates fixed-controller and training effects."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.r5_w_commander import BASE,read,pin,save


def main():
    fig,axes=plt.subplots(2,4,figsize=(17,7),sharex=True)
    summaries=[]
    for experiment,update,label in [('R5-W-E1-CONTROL-01',512,'E1 fixed u512 + .15m'),('R5-W-L1-EVAL-01',640,'L1 u640 + .15m'),('R5-W-L1-EVAL-01',768,'L1 u768 + .15m')]:
        inventory=BASE/'experiments'/experiment/'live/R1-research-formal-evaluation/INVENTORY.json'
        out=Path(read(inventory)['roots']['attempt'])/'results/research-formal-evaluation'
        raw=out/f'evaluation_raw/w_R1_u{update}/W.jsonl';data=[json.loads(x) for x in raw.read_text().splitlines()][1:]
        result=read(out/f'w_R1_u{update}.json');episodes=[e for e in result['episodes'] if e['case']=='W']
        groups={}
        for wall,(begin,end,axis) in enumerate([(0,16,0),(16,32,1)]):
            lanes=episodes[begin:end];groups['xy'[wall]]=dict(minimum_geometry_m=min(e['metrics']['min_distance'] for e in lanes),maximum_stop_time_s=max(e['metrics']['stop_time'] for e in lanes))
            geometry=np.array([e['geometry_distance_m'] for e in lanes]).mean(axis=0)
            velocity=np.array([[r['root_state_w'][l][7+axis] for l in range(begin,end)] for r in data]).mean(axis=1)
            for col,values in enumerate([geometry,velocity]):axes[wall,col].plot(np.arange(1,len(values)+1)*.1,values,label=label)
            for col,key in [(2,'q'),(3,'ics_beta')]:
                values=np.array([[r[key][l] for l in range(begin,end)] for r in data]).reshape(len(data),end-begin).mean(axis=1)
                axes[wall,col].plot(np.arange(len(values))*.1,values,label=label)
        summaries.append(dict(label=label,inventory=pin(inventory),raw=pin(raw),original_result=pin(out/f'w_R1_u{update}.json'),groups=groups))
    for wall in range(2):
        for col,title in enumerate(['Geometry (m), observation time','World normal velocity (m/s), observation time','q, control issue time','ICS beta, control issue time']):
            axes[wall,col].set_title('xy'[wall]+' wall: '+title);axes[wall,col].set_xlim(0,6);axes[wall,col].grid(alpha=.2);axes[wall,col].set_xlabel('seconds')
        axes[wall,0].axhline(.8,color='r',linestyle='--');axes[wall,0].axhline(1.1,color='gray',linestyle=':');axes[wall,1].axhline(.1,color='gray',linestyle=':')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(BASE/'L1_W_TIMING.png',dpi=150)
    save(BASE/'L1_METHOD_COMPARISON.json',dict(status='INTERMEDIATE_NOT_GOAL_COMPLETE',runs=summaries,interpretation='E1 fixed policy already passes. L1 preserves original tasks and provides actual q-positive training exposure, but no isolated added safety benefit established. Original R1 no-margin negative baseline remains in W_CAUSAL_DIAGNOSIS. Independent seed verification pending.',timing='Geometry and velocity observed after control; q/beta indexed by control issue. Group means are visualization only; pass requires every original lane.'),new=True)
    print(json.dumps(summaries))


if __name__=='__main__':main()
