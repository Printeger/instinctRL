"""Transfer the existing certified command reversal into the recorded-attitude CPU replay."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent))
import replay
from replay_opposed_planes import reprove_rotation


def run(preset,lane,lag,filename):
    registration = root/'INPUTS.json'
    record = load(registration)
    old_inputs = checked(record['inputs'])
    assert checked(record['replay']) == root.parent/'replay.py'
    checked(record['rotation'])
    arm = checked(record['reversal_harness'])
    inputs = dict(load(old_inputs),replay_harness=record['reversal_harness'])

    def prepare(path):
        assert Path(path) == arm
        harness = runpy.run_path(path)
        trial = harness['StopTrial']
        init = trial.__init__

        def initialize(self,option,mode,hold_horizon,crawl_speed):
            assert mode == 'crawl'
            init(self,option,'bridge30',hold_horizon,crawl_speed)

        trial.__init__ = initialize
        return harness

    with (patch.object(replay,'runpy',SimpleNamespace(run_path=prepare)),
          patch.object(replay,'reprove_plane_alignment',reprove_rotation),
          patch.dict(os.environ,PRODUCER_TAPER=preset)):
        result = replay.run(inputs,lane,lag,'native','recorded',True)
    assert not stop_paths(), stop_paths()
    return save(filename,dict(inputs=pin(old_inputs),replay=record['replay'],handoff=pin(registration),
                              wrapper=pin(__file__),plane_reproof=True,taper=preset,**result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taper',required=True,choices=('diagnostic','late'))
    parser.add_argument('--lane',required=True,type=int,choices=(19,22))
    parser.add_argument('--lag',required=True,type=float,choices=(.25,.35,.45))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    print(run(args.taper,args.lane,args.lag,args.output))
