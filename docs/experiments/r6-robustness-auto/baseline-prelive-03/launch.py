"""One bounded handoff for this continuation thread, then existing R6 controller.

Read-only platform observation. Does not pause a Goal or issue Store permission.
The existing controller retains all Store, resource, STOP and identity guards.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from tools.r6_platform_rpc import Reader, settings
from tools.r6_postrun import stop_paths
from rlflow.jobs import owns

ROOT=Path('/home/mint/rl_dev/instinctRL-dev')
BASE=ROOT/'docs/experiments/r6-robustness-auto'
THREAD='01a16275-c109-7302-84bf-4b916ea448e6'
OBJECTIVE='pasted text file: /home/mint/.codex/attachments/7d738239-2883-4e11-9aab-785b93ffd910/pasted-text-1.txt. Read this file before continuing.'
TRACE='/home/mint/.codex/sessions/2026/10/22/rollout-2026-10-22T15-14-06-01a16275-c109-7302-84bf-4b916ea448e6.jsonl'
OWNER='c62b1dec82a1236964ce4d839ee30c077d81d83242d800d667ef1a3d04da7aac'


def observe(journal):
    if stop_paths(journal):raise RecoveryError('CONTINUATION_STOP')
    cfg=settings(journal.read())
    with Reader(cfg) as client:
        client.send(dict(id=1,method='initialize',params=dict(clientInfo=dict(name='r6_continuation_handoff',version='1'),capabilities=dict(experimentalApi=True))))
        client.receive(1);client.send(dict(method='initialized',params={}))
        rows=[]
        for n,method,params in [(2,'thread/goal/get',dict(threadId=THREAD)),
                (3,'thread/read',dict(threadId=THREAD,includeTurns=False)),
                (4,'thread/turns/list',dict(threadId=THREAD,limit=1,sortDirection='desc',itemsView='notLoaded')),
                (5,'thread/goal/get',dict(threadId=THREAD))]:
            client.send(dict(id=n,method=method,params=params));rows.append(client.receive(n))
        goals=[rows[i]['result']['goal'] for i in (0,3)]
        thread=rows[1]['result']['thread'];turns=rows[2]['result']['data']
        if not (all(g['threadId']==THREAD and g['objective']==OBJECTIVE for g in goals)
                and thread['id']==THREAD and thread['cwd']==str(ROOT) and thread['path']==TRACE
                and len(turns)==1 and turns[0]['rootTurnId']==turns[0]['id']):
            raise RecoveryError('CONTINUATION_PLATFORM_IDENTITY')
        turn=turns[0]
        ready=(all(g['status']=='paused' for g in goals) and goals[0]==goals[1]
            and thread['status']['type']=='idle' and turn['status']=='completed'
            and turn['error'] is None and isinstance(turn.get('completedAt'),(int,float))
            and time.time()-turn['completedAt']>=2 and owns(client.peer['process']))
        return dict(ready=ready,observed_at=time.time(),peer=client.peer,responses=rows,
            launch_script=file_pin(__file__),scope='CURRENT_CONTINUATION_START_HANDOFF_ONLY',
            whole_wait_platform_calls='UNKNOWN_UNTIL_POSTRUN_AUDIT')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    j=Journal(BASE/'batch')
    if args.check:
        out=observe(j);print(json.dumps(out,sort_keys=True));return
    deadline=time.monotonic()+600
    while True:
        with j.writer():
            j.guard()
            owner=j.read()['events'][OWNER]
            if owns(owner.get('controller_process')):raise RecoveryError('CONTINUATION_CONTROLLER_ALREADY_LIVE')
            if not owner.get('baseline_input_repair'):raise RecoveryError('CONTINUATION_REVIEWED_INPUT_ADOPTION_REQUIRED')
            record=observe(j)
            if record['ready']:
                target=BASE/'baseline-prelive-03/current-thread-handoff.json'
                from tools.r6_decision_packet import persist_exact
                from rlflow.store import canonical
                persist_exact(target,canonical(record)+b'\n')
                break
        if time.monotonic()>=deadline:raise RecoveryError('CONTINUATION_HANDOFF_TIMEOUT_NO_LAUNCH')
        time.sleep(5)
    request=file_pin(BASE/'controller-candidate-01/request.json')
    checked(request)
    os.execv(sys.executable,[sys.executable,'-B','-m','tools.r6_controller',str(BASE/'batch'),request['path'],
        '--request-sha256',request['sha256'],'--store',str(ROOT/'coordination/research')])

if __name__=='__main__':main()
