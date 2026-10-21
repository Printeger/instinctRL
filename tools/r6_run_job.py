"""Run one exact authorized R6 evaluation through its local decision handoff.

Composes existing phases under one writer. A decision remains a proposal: this
entry point neither issues permission nor launches an unreviewed next job.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from rlflow import experiments
from rlflow.store import sha
from tools.r6_recovery import Journal, RecoveryError, checked
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools.r6_resource_wait import wait_for_observation
from tools.r6_register_job import register_and_close
from tools.r6_finalize_job import finish_and_decide,finish_final


def run(journal, spec_pin, store, query_executable, decision_executable, *,
        free_disk_bytes, queue_review=None, sampler_review=None,
        purpose='development', sandbox='read-only'):
    journal.require_writer()
    spec=json.loads(checked(spec_pin).read_text())
    final=spec.get('h1',{}).get('mode')=='research-r6-final-evaluation'
    replay=spec.get('h1',{}).get('mode')=='research-r6-video-replay'
    if (final or replay) and purpose!='confirmation_delivery':raise RecoveryError('RUN_JOB_FINAL_PURPOSE')
    jid='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    entries=[e for e in journal.read()['events'].values()
             if e['kind']=='job' and e['identity']['job_id']==jid]
    if len(entries)>1:raise RecoveryError('RUN_JOB_ALIAS')
    claim=entries[0].get('submission_claim') if entries else None
    if claim is not None:
        # Even an ambiguous claim must reach submit's reconciliation checks,
        # never a new resource observation or another submission attempt.
        queue_key=claim['queue_key']
    else:
        journal.guard()
        if stop_paths(journal):raise RecoveryError('RUN_JOB_STOP')
        if journal.read()['budget']['capacity_pass'] is not True:
            raise RecoveryError('RUN_JOB_CAPACITY_NOT_ACCEPTED')
        if entries and entries[0]['request']['wait_deadline_unix']<=time.time():
            raise RecoveryError('RUN_JOB_EXPIRED_RESERVATION')
        checked(query_executable)
        decision_path=checked(decision_executable)
        if journal.read()['goal_id']=='R6-CPU-TEST-FIXTURE':
            if not decision_path.is_relative_to(journal.directory.parent.resolve()):
                raise RecoveryError('RUN_JOB_DECISION_FIXTURE_EXECUTABLE')
        else:
            from tools.r6_decision import CLI, CLI_SHA
            if decision_executable!={'path':str(CLI),'sha256':CLI_SHA}:
                raise RecoveryError('RUN_JOB_DECISION_PINNED_CLI')
        no_model_workers(journal)
        # Reject missing/revoked authority before spending queue queries.
        # Existing submit still revalidates authority immediately before effects.
        experiments.validate(spec,store.root)
        experiments.authorized(store.replay(),spec,store.root)
        observation=wait_for_observation(journal,spec_pin,query_executable,
            free_disk_bytes=free_disk_bytes,queue_review=queue_review,
            sampler_review=sampler_review)
        if observation['status']!='OBSERVATION_PASS':
            return dict(status='RESOURCE_OBSERVATION_REJECTED',observation=observation,
                        scientific_acceptance=False)
        queue_key=observation['event_key']
    def refresh_observation():
        observation=wait_for_observation(journal,spec_pin,query_executable,
            free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free),
            queue_review=queue_review,sampler_review=sampler_review)
        if observation['status']!='OBSERVATION_PASS':
            raise RecoveryError('RUN_JOB_REFRESH_OBSERVATION_REJECTED')
        return observation['event_key']
    closed=register_and_close(journal,spec_pin,store,queue_key=queue_key,
        free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free),
        purpose=purpose,refresh_observation=refresh_observation)
    if replay:
        from tools.r6_finalize_job import finalize
        result=finalize(journal,closed['event_key'],store,query_executable,
            free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free))
        return dict(status='RENDER_JOB_CLOSED_REQUIRES_VIDEO_VALIDATION',event_key=closed['event_key'],
                    closure=closed['closure'],result=result,scientific_acceptance=False)
    if final:
        result=finish_final(journal,closed['event_key'],store,query_executable,
            free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free))
        return dict(status='FINAL_JOB_ANALYSIS_RETURNED',event_key=closed['event_key'],
                    closure=closed['closure'],result=result,scientific_acceptance=False)
    result=finish_and_decide(journal,closed['event_key'],store,
        query_executable,decision_executable,
        free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free),
        sandbox=sandbox)
    return dict(status='DECISION_PROPOSAL_RETURNED',event_key=closed['event_key'],
                closure=closed['closure'],result=result,scientific_acceptance=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--store',type=Path,required=True)
    args=parser.parse_args(argv)
    plan=json.loads(checked(dict(path=str(args.plan.resolve()),sha256=args.plan_sha256)).read_text())
    fields={'spec_pin','query_executable','decision_executable','queue_review',
            'sampler_review','purpose','sandbox'}
    if set(plan)!=fields:raise RecoveryError('RUN_JOB_PLAN_FIELDS')
    if plan['sandbox'] not in {'read-only','danger-full-access'}:
        raise RecoveryError('RUN_JOB_PLAN_SANDBOX')
    from rlflow.store import Store
    journal=Journal(args.directory.resolve())
    with journal.writer():
        root=args.store.resolve(strict=True)
        if not (root/'runtime/research.db').is_file():
            raise RecoveryError('RUN_JOB_STORE_DATABASE_REQUIRED')
        result=run(journal,store=Store(root),
            free_disk_bytes=shutil.disk_usage(journal.directory).free,**plan)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
