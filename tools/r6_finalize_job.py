"""Inventory one already closed job and complete its two exact postrun queries.

Optional CPU analysis and one registered proposal-only decision follow.
No submission, automatic retry, budget refund or science acceptance.
"""
import argparse
import json
from pathlib import Path
import shutil

from tools.r6_job_inventory import record_inventory
from tools.r6_postrun import register_postruns, complete_postruns
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin


def finalize(journal, key, store, executable, *, free_disk_bytes):
    journal.require_writer()
    checked(executable)
    inventory=record_inventory(journal,key,store)
    state=journal.read();job=journal.canonical_job(key,state=state)
    existing={}
    for event_key,event in state['events'].items():
        if event['kind']=='postrun' and event['identity']['job_id']==job['identity']['job_id']:
            ordinal=event['request'].get('ordinal')
            expected=dict(job_event_key=key,closure=job['closure'],inventory=job['inventory'],
                          executable=executable,ordinal=ordinal)
            if type(ordinal) is not int or ordinal not in (0,1) or ordinal in existing or event['request']!=expected:
                raise RecoveryError('FINALIZE_POSTRUN_BINDING')
            existing[ordinal]=event_key
    # Recover existing events directly, including after STOP. Registration of a
    # missing ordinal still traverses the normal STOP and budget gates.
    keys=([existing[0],existing[1]] if len(existing)==2 else
          register_postruns(journal,key,executable,free_disk_bytes=free_disk_bytes))
    postrun=complete_postruns(journal,key,keys)
    return dict(inventory=inventory,postrun=postrun,scientific_acceptance=False)


def analyze(journal, key, store, *, free_disk_bytes, python=None, program=None, timeout_seconds=3600):
    """Register once or reconnect the same bounded CPU analysis event.

Overrides exist for isolated CPU fixtures; the existing analysis validator
requires the frozen interpreter/analyzer for every real-mode job.
"""
    from tools import r6_cpu_analysis as cpu
    journal.require_writer()
    state=journal.read();job=journal.canonical_job(key,state=state)
    from tools.r6_analysis_repair import active
    selected=active(journal,key)
    existing=[] if selected is None else [selected]
    expected=dict(python=file_pin(cpu.PYTHON) if python is None else python,
        program=file_pin(cpu.program_for(json.loads(checked(job['request']['spec']).read_text())['h1']['mode'])) if program is None else program,
        timeout_seconds=timeout_seconds)
    if existing:
        if len(existing)!=1:raise RecoveryError('FINALIZE_ANALYSIS_ALIAS')
        event_key=existing[0];entry,_=cpu.validate(journal,event_key)
        if (entry['request']['job_event_key']!=key
                or any(entry['request'][name]!=value for name,value in expected.items())):
            raise RecoveryError('FINALIZE_ANALYSIS_BINDING')
    else:
        event_key=cpu.register_analysis(journal,key,store,free_disk_bytes=free_disk_bytes,**expected)
    return cpu.handoff(journal,event_key,store)


def decide(journal, key, store, executable, *, free_disk_bytes, sandbox='read-only', poll_seconds=15):
    """Register/recover the exact post-analysis model event, never a next job."""
    from tools import r6_decision as decision
    from tools.r6_analysis_binding import bind_analysis
    from rlflow.store import sha
    journal.require_writer()
    state=journal.read();job=journal.canonical_job(key,state=state)
    path=checked(executable)
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':
        if not path.is_relative_to(journal.directory.parent.resolve()):
            raise RecoveryError('FINALIZE_DECISION_FIXTURE_EXECUTABLE')
    elif executable!={'path':str(decision.CLI),'sha256':decision.CLI_SHA}:
        raise RecoveryError('FINALIZE_DECISION_PINNED_CLI')
    existing=[event_key for event_key,event in state['events'].items()
        if event['kind']=='model' and event['identity']['job_id']==job['identity']['job_id']
        and event['identity']['role']=='post-analysis-decision']
    if len(existing)>1:raise RecoveryError('FINALIZE_DECISION_ALIAS')
    if not existing or not state['events'][existing[0]].get('decision_input'):
        if existing and (state['events'][existing[0]].get('dispatch_claim') or state['events'][existing[0]].get('receipt')):
            raise RecoveryError('FINALIZE_DECISION_INPUT_AMBIGUOUS')
        event_key=decision.register(journal,key,store,executable=executable,
            free_disk_bytes=free_disk_bytes,sandbox=sandbox)
    else:
        event_key=existing[0];entry=state['events'][event_key];packet_input=entry['decision_input']
        expected=dict(prompt=packet_input['prompt'],schema=file_pin(decision.SCHEMA),
            executable=executable,cwd=str(decision.ROOT),sandbox=sandbox,
            timeout_seconds=3600,output_cap_bytes=decision.MODEL_CAP,
            stop_paths=decision.model_stop_paths(journal))
        identity=dict(goal_id=state['goal_id'],event_id='DECISION:'+key,
            job_id=job['identity']['job_id'],input_sha256=sha(expected),role='post-analysis-decision')
        if (entry['request']!=expected or entry['identity']!=identity or sha(identity)!=event_key
                or packet_input['job_event_key']!=key):
            raise RecoveryError('FINALIZE_DECISION_BINDING')
        packet=json.loads(checked(packet_input['packet']).read_text())
        binding=json.loads(checked(job['analysis']).read_text())
        if (packet['analysis']!=job['analysis']
                or binding!=bind_analysis(job,key,binding['report'],store)):
            raise RecoveryError('FINALIZE_DECISION_ANALYSIS_DRIFT')
    return decision.handoff(journal,event_key,store,poll_seconds=poll_seconds)


def finish_and_decide(journal, key, store, query_executable, decision_executable, *,
                      free_disk_bytes, sandbox='read-only', poll_seconds=15):
    """Resume a registered decision before revisiting earlier completed phases."""
    journal.require_writer()
    state=journal.read();job=journal.canonical_job(key,state=state)
    has_decision=any(event['kind']=='model' and event['identity']['job_id']==job['identity']['job_id']
        and event['identity']['role']=='post-analysis-decision' for event in state['events'].values())
    if not has_decision:
        finalize(journal,key,store,query_executable,free_disk_bytes=free_disk_bytes)
        analyze(journal,key,store,free_disk_bytes=free_disk_bytes)
    proposal=decide(journal,key,store,decision_executable,free_disk_bytes=free_disk_bytes,
                    sandbox=sandbox,poll_seconds=poll_seconds)
    job=journal.canonical_job(key)
    return dict(inventory=json.loads(checked(job['inventory']).read_text()),
        postrun=json.loads(checked(job['postrun']).read_text()),
        analysis=json.loads(checked(job['analysis']).read_text()),decision=proposal,
        scientific_acceptance=False)



def finish_final(journal,key,store,query_executable,*,free_disk_bytes):
    """Close final evidence without a per-model scientific decision or tuning."""
    journal.require_writer();job=journal.canonical_job(key);spec=json.loads(checked(job['request']['spec']).read_text())
    if spec.get('h1',{}).get('mode')!='research-r6-final-evaluation' or job['purpose']!='confirmation_delivery':
        raise RecoveryError('FINALIZE_FINAL_STAGE_REQUIRED')
    if any(e['kind']=='model' and e['identity'].get('job_id')==job['identity']['job_id']
           and e['identity'].get('role')=='post-analysis-decision' for e in journal.read()['events'].values()):
        raise RecoveryError('FINALIZE_FINAL_UNEXPECTED_MODEL_DECISION')
    finalize(journal,key,store,query_executable,free_disk_bytes=free_disk_bytes)
    analyze(journal,key,store,free_disk_bytes=free_disk_bytes)
    job=journal.canonical_job(key)
    return dict(inventory=json.loads(checked(job['inventory']).read_text()),
        postrun=json.loads(checked(job['postrun']).read_text()),analysis=json.loads(checked(job['analysis']).read_text()),
        scientific_acceptance=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--analyze',action='store_true',help='Run/recover the bounded CPU replay after postrun')
    parser.add_argument('--decision-executable',type=Path,help='After analysis, run/recover one exact proposal-only model decision')
    parser.add_argument('--decision-sandbox',choices=['read-only','danger-full-access'],default='read-only')
    args=parser.parse_args(argv)
    if args.decision_executable is not None and not args.analyze:
        parser.error('--decision-executable requires --analyze')
    from rlflow.store import Store
    journal=Journal(args.directory.resolve())
    with journal.writer():
        root=args.store.resolve(strict=True)
        if not (root/'runtime/research.db').is_file():raise RecoveryError('FINALIZE_STORE_DATABASE_REQUIRED')
        if args.decision_executable is not None:
            result=finish_and_decide(journal,args.event_key,Store(root),file_pin(args.executable),file_pin(args.decision_executable),
                free_disk_bytes=shutil.disk_usage(journal.directory).free,sandbox=args.decision_sandbox)
        else:
            result=finalize(journal,args.event_key,Store(root),file_pin(args.executable),
                            free_disk_bytes=shutil.disk_usage(journal.directory).free)
            if args.analyze:
                result['analysis']=analyze(journal,args.event_key,Store(root),
                    free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
