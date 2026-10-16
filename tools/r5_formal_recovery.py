"""One user-authorized recovery of a rejected checkpoint review, never retraining.

Reuse the existing Batch's exact job waits, Commander and owned model worker.
The only bootstrap action is one CPU independent review. Its explicit actual
checkpoint and recovery-code acceptance gates the two existing evaluations.
"""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import signal
import subprocess
import time
import traceback
from tools.r5_formal_batch import (Batch, ROOT, CASE, RESULTS, STATIC, BASE,
    read, pin, sha, checked, write, manifest_pins, model_worker)

RECOVERY = 'recovery-20261016'
PLAN = STATIC/RECOVERY/'plan.json'


def validate_recovery_origin(state):
    if (state['status'] != 'STOPPED' or state['phase'] != 'MODEL:checkpoint-review'
            or 'recovery' in state or set(state['model_calls']) != {'checkpoint-review'}
            or state['model_calls']['checkpoint-review'].get('accepted')
            or state['model_call_limit'] != 2):
        raise RuntimeError('NOT_THE_SINGLE_REJECTED_REVIEW')
    expected={g+'-research-formal-training' for g in ['R0','R1']}
    if set(state['entries']) != expected or not all(e.get('done') for e in state['entries'].values()):
        raise RuntimeError('ORIGINAL_TRAINING_NOT_CLOSED_OR_EVALUATION_ALREADY_EXISTS')


class RecoveryBatch(Batch):
    def __init__(self, root=ROOT):
        super().__init__(root)
        self.original_plan=self.plan
        self.plan=read(self.root/PLAN)

    def checkpoint_slot(self): return 'checkpoint-recovery'
    def checkpoint_review_path(self): return self.root/RESULTS/RECOVERY/'joint-actual-training.json'
    def training_analysis_path(self, group): return self.root/RESULTS/RECOVERY/f'{group}-training-analysis.json'
    def worker_command(self, slot):
        return [self.plan['control_python'],str(self.root/'tools/r5_formal_recovery.py'),'_model-worker',slot]

    def verify_pins(self):
        required={'batch','recovery','analyzer','binding_analyzer','commander','launch_gate','manifest',
            'ready','permission','final_review','checkpoint-recovery_prompt','final-analysis_prompt',
            'response_schema','codex','tmux','authority','original_plan','previous_state','previous_failure',
            'previous_review','previous_call_receipt','previous_call_answer','R0_analysis','R1_analysis',
            'evidence_argument','tests'}
        if set(self.plan['pins']) != required: raise RuntimeError('RECOVERY_PLAN_PIN_SET')
        for p in self.plan['pins'].values(): checked(p)
        for p in manifest_pins(read(checked(self.plan['pins']['manifest']))): checked(p)
        if (self.plan['experiment_id'] != CASE or self.plan['model_call_limit'] != 3
                or self.plan['additional_training_transitions'] != 0
                or self.plan['additional_optimizer_updates'] != 0
                or self.plan['additional_training_checkpoint_writes'] != 0
                or self.plan['additional_review_calls'] != 1
                or self.plan['automatic_retries'] != 0
                or self.plan['live_entries_maximum'] != 4
                or self.plan['evaluation_transitions'] != 329600):
            raise RuntimeError('RECOVERY_SCOPE_DRIFT')
        previous=read(checked(self.plan['pins']['previous_state']))
        validate_recovery_origin(previous)
        if previous['plan'] != self.plan['pins']['original_plan']:
            raise RuntimeError('ORIGINAL_PLAN_DRIFT')
        failure=read(checked(self.plan['pins']['previous_failure']))
        if failure['phase'] != 'MODEL:checkpoint-review' or not failure['error'].startswith('MODEL_REJECTED:'):
            raise RuntimeError('NOT_A_CHECKPOINT_REJECTION')
        if self.state and 'recovery' in self.state:
            if (self.state['recovery']['plan'] != pin(self.root/PLAN)
                    or self.state['plan'] != pin(self.root/PLAN)
                    or self.state['model_calls']['checkpoint-review'] != previous['model_calls']['checkpoint-review']):
                raise RuntimeError('RECOVERY_STATE_OR_OLD_CALL_DRIFT')
            for name, entry in previous['entries'].items():
                if self.state['entries'].get(name) != entry: raise RuntimeError('TRAINING_ENTRY_CHANGED')

    def arm(self):
        from rlflow.jobs import owns
        from tools.r5_formal_commander import idle
        self.run.mkdir(parents=True,exist_ok=True)
        with (self.run/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.guard();validate_recovery_origin(self.state)
            if self.state != read(checked(self.plan['pins']['previous_state'])):
                raise RuntimeError('PREVIOUS_STATE_CHANGED')
            if owns(self.state.get('controller_identity')): raise RuntimeError('PREVIOUS_CONTROLLER_ALIVE')
            request=read(self.run/'calls/checkpoint-review/request.json')
            for key in ['worker_identity','codex_identity']:
                if owns(request.get(key)): raise RuntimeError('PREVIOUS_MODEL_ALIVE')
            idle(self.store())
            for g in ['R0','R1']:
                name=g+'-research-formal-evaluation'
                if (self.case/'inputs'/f'{name}.authorized.json').exists() or (self.case/f'{name}-approval.json').exists():
                    raise RuntimeError('EVALUATION_ALREADY_ISSUED')
                spec=read(self.case/'inputs'/f'{name}-spec.proposed.json')
                if Path(spec['h1']['attempt']).exists(): raise RuntimeError('PARTIAL_EVALUATION_ATTEMPT')
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root)!=subprocess.check_output(['git','rev-parse','origin/main'],cwd=self.root):
                raise RuntimeError('HEAD_NOT_PUSHED')
            if subprocess.check_output(['git','show','HEAD:'+str(PLAN)],cwd=self.root)!= (self.root/PLAN).read_bytes():
                raise RuntimeError('RECOVERY_PLAN_NOT_COMMITTED')
            old_failure=self.state['failure']
            self.save(status='ARMED',phase='RECOVERY_ARMED',plan=pin(self.root/PLAN),model_call_limit=3,
                tmux_session=self.plan['tmux_session'],
                recovery={'plan':pin(self.root/PLAN),'previous_state':self.plan['pins']['previous_state'],
                    'previous_failure':old_failure,'explicit_user_recovery':True,'extra_review_limit':1})
            self.log('USER_AUTHORIZED_RECOVERY_ARMED',previous_failure=old_failure)

    def require_review(self):
        row=self.state['model_calls'].get(self.checkpoint_slot(),{})
        if not row.get('accepted'): raise RuntimeError('RECOVERY_REVIEW_NOT_ACCEPTED')
        self.verify_accepted_model(row)
        review=read(self.checkpoint_review_path())
        decision=review.get('recovery_decision',{})
        identities=review.get('identities',{})
        if (review.get('verdict')!='PASS' or review.get('must_fix')!=[]
                or identities.get('recovery_plan_sha256')!=sha(self.root/PLAN)
                or decision.get('status')!='ACCEPT_ORIGINAL_RUNTIME_EVIDENCE'
                or decision.get('no_contract_waiver') is not True
                or decision.get('original_missing_file_preserved') is not True
                or decision.get('prelaunch_safe') is not True):
            raise RuntimeError('RECOVERY_EVIDENCE_OR_CODE_NOT_INDEPENDENTLY_ACCEPTED')
        expected=[]
        for g in ['R0','R1']:
            for ck in read(self.training_analysis_path(g))['accepted_checkpoint_candidates']:
                checked(ck);expected.append(ck)
        key=lambda r:(r['group'],r['update'])
        if sorted(review.get('accepted_evaluation_checkpoints',[]),key=key)!=sorted(expected,key=key):
            raise RuntimeError('ACTUAL_CHECKPOINT_ACCEPTANCE_MISMATCH')

    def run_entry(self, group, mode):
        if mode!='research-formal-evaluation':raise RuntimeError('RECOVERY_NEVER_RUNS_TRAINING')
        self.require_review()
        return super().run_entry(group,mode)

    def model_protected_files(self):
        values=super().model_protected_files()
        previous=read(checked(self.plan['pins']['previous_state']))
        # Every original protected object remains byte-identical during review;
        # ledger may advance only later in the unchanged Commander evaluation.
        if not self.state['model_calls'].get(self.checkpoint_slot(),{}).get('accepted'):
            for p in previous['model_calls']['checkpoint-review']['protected']:checked(p)
        extra=[self.root/RESULTS/'reviews/joint-actual-training.json']
        extra += [self.root/RESULTS/f'{g}-training-analysis.json' for g in ['R0','R1']]
        extra += [checked(p) for p in self.plan['pins'].values()]
        extra += [p for p in (self.root/RESULTS).rglob('*') if p.is_file()]
        if self.checkpoint_review_path().exists():extra.append(self.checkpoint_review_path())
        values += [pin(p) for p in extra]
        return list({p['path']:p for p in values}.values())

    def execute(self):
        from rlflow.jobs import identity
        with (self.run/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if self.state['status']=='COMPLETE':return
            if self.state['status']=='STOPPED':raise RuntimeError('RECOVERY_STOPPED_NO_AUTOMATIC_RETRY')
            if 'recovery' not in self.state:raise RuntimeError('EXPLICIT_RECOVERY_ARM_REQUIRED')
            try:
                self.guard();self.save(status='RUNNING',controller_identity=identity(__import__('os').getpid()))
                for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:setattr(self,'interrupted',True))
                self.call_model(self.checkpoint_slot())
                self.require_review()
                for g in ['R0','R1']:self.run_entry(g,'research-formal-evaluation')
                self.verify_evaluations()
                self.call_model('final-analysis')
                self.save(status='COMPLETE',phase='COMPLETE',finished_at=time.time())
                self.log('COMPLETE',analysis=self.state['final_analysis'])
            except BaseException as error:
                path=self.run/RECOVERY/'FAILURE.json'
                write(path,{'status':'STOPPED','phase':self.state['phase'],'error_type':type(error).__name__,
                    'error':str(error),'traceback':traceback.format_exc(),'time':time.time(),
                    'no_automatic_retry':True,'original_failure':self.state['recovery']['previous_failure']})
                self.save(status='STOPPED',failure=pin(path));self.log('RECOVERY_STOPPED',error=str(error));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['arm','run','_model-worker']);p.add_argument('slot',nargs='?',choices=['checkpoint-recovery','final-analysis']);a=p.parse_args()
    b=RecoveryBatch()
    if a.action=='_model-worker':
        if not a.slot:p.error('slot required')
        raise SystemExit(model_worker(a.slot,instance=b))
    if a.action=='arm':b.arm()
    else:b.execute()

if __name__=='__main__':main()
