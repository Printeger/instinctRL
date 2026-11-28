"""Re-render the new shared prefix before the optimizer's assumed branch."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import torch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths(),stop_paths()
    path=Path(__file__).resolve()
    registry=path.with_name('OBSERVATION-INPUTS.json')
    refs=load(registry)
    plan=load(checked(refs['solver_inputs']))
    arm=load(checked(refs['strict_arm']))
    assert arm['inputs']==refs['solver_inputs'] and arm['script']==refs['solver_script']
    mode=plan['modes'][arm['mode']]
    cutoff=mode['split_step']
    result=arm['result']['winner']
    split=runpy.run_path(str(checked(plan['split_script'])))
    data=split['load_inputs'](load(checked(plan['split_inputs'])))
    start=data['config']['start']
    alternatives=[out['issues'] for out in result['best']['group_outputs']]
    prefix=np.asarray(alternatives[1])
    assert np.array_equal(prefix[:cutoff-start],alternatives[0][:cutoff-start])
    data['positive']=prefix
    source=runpy.run_path(str(checked(refs['measurement_script'])))
    with torch.inference_mode():
        observations=source['measurements'](dict(reopening_inputs=refs['reopening_inputs'],
                                                 observed_through=cutoff),data)
    audit=source['compare'](observations)
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(registry),script=pin(path),
                          cutoff=cutoff,menu_id=result['menu_id'],audit=audit,observations=observations)),
                          at_cutoff=audit[-1])),flush=True)


if __name__=='__main__':
    main()
