"""One fixed R5 first-segment batch. Python waits; exactly two bounded model slots.

All experiment mutations remain in the unchanged exact Commander. Ambiguous
side effects fail closed; no retry, budget expansion or new experiment identity.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASE = 'R5-FORMAL-MEAN10-FIRST-SEGMENT-01'
STATIC = Path('coordination/batches/R5_FORMAL_FIRST_SEGMENT_20261015')
RUNTIME = Path('coordination/research/batches') / CASE
BASE = Path('docs/experiments/r5-formal-readiness-auto')
RESULTS = Path('docs/experiments/r5-formal-first-segment-20261015')
TERMINAL = {'FAILED', 'STOPPED', 'SUCCEEDED'}


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def pin(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def write(path, value):
    from rlflow.store import atomic
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
def checked(pin_value):
    p = Path(pin_value['path'])
    if p.is_symlink() or sha(p) != pin_value['sha256']: raise RuntimeError('PIN_DRIFT: '+str(p))
    return p


def can_submit(entry, job, *, charged, admission):
    if job is not None: return False
    if entry.get('submit_intent') or charged or admission:
        raise RuntimeError('AMBIGUOUS_SUBMIT: never repeat a possibly consumed approval')
    return True


def allowed_model_changes(slot, paths):
    case = BASE/'experiments'/CASE
    prefixes = [str(RESULTS)+'/']
    exact = {str(BASE/'goal_budget.json'), 'coordination/PROJECT_STATE.json'}
    # Existing case files are immutable during either model call. Only the
    # already declared evaluation authorization products may be added by #1.
    if slot == 'checkpoint-review':
        for g in ['R0','R1']:
            n = g+'-research-formal-evaluation'
            exact |= {str(case/'inputs'/(n+'.authorized.json')), str(case/'inputs'/(n+'-AUTH.bound.json')),
                      str(case/(n+'-approval.json'))}
    if slot == 'final-analysis': exact |= {'coordination/ACTIVE_WORKFLOW.json','coordination/CURRENT_TASK.md','docs/PROJECT_STATUS.md'}
    return all(p in exact or any(p.startswith(prefix) for prefix in prefixes) for p in paths)


def manifest_pins(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str): yield value
        for child in value.values(): yield from manifest_pins(child)
    elif isinstance(value, list):
        for child in value: yield from manifest_pins(child)


def stop_owned(identity_value, *, wait_seconds=15):
    """Terminate only this recorded process group; unknown ownership fails shut."""
    from rlflow.jobs import owns
    if not identity_value: raise RuntimeError('UNKNOWN_MODEL_CHILD_OWNERSHIP')
    pid = identity_value['pid']
    if owns(identity_value):
        if os.getpgid(pid) != pid: raise RuntimeError('MODEL_CHILD_GROUP_MISMATCH')
        os.killpg(pid, signal.SIGTERM)
        until = time.time()+wait_seconds
        while owns(identity_value) and time.time()<until: time.sleep(.1)
        if owns(identity_value): os.killpg(pid, signal.SIGKILL)
        until = time.time()+5
        while owns(identity_value) and time.time()<until: time.sleep(.1)
        if owns(identity_value): raise RuntimeError('OWNED_MODEL_CHILD_SURVIVED')
    return {'identity':identity_value, 'no_owned_survivor':not owns(identity_value)}


def postrun_complete(count, files):
    if count == 0 and not any(files): return False
    if count == 2 and all(files): return True
    raise RuntimeError('PARTIAL_POSTRUN: do not repeat GPU queries')


def science_complete(spec, receipt, output):
    raw = read(output/'status.json'); c = raw['counters']; training = spec['h1']['mode'].endswith('training')
    if raw['status'] not in ['PASS', 'PASS_PENDING_CLOSE'] or raw['primary_error'] or raw['cleanup_errors']:
        raise RuntimeError('INCOMPLETE_SCIENCE_OR_RESOURCE_FAILURE: '+str(raw.get('primary_error')))
    if receipt['status'] != 'SUCCEEDED':
        close = read(output/'cleanup_timeout.json')
        if not (receipt['status'] == 'FAILED' and receipt['exit_code'] == -9
                and close['status'] == 'HARD_TIMEOUT' and close['cleanup'] == 'app.close'
                and close['timeout_seconds'] == 9.5):
            raise RuntimeError('UNKNOWN_NATIVE_FAILURE')
    expected = {'training_transitions':131072 if training else 0, 'nontraining_transitions':0 if training else 164800,
                'completed_updates':256 if training else 0, 'real_optimizer_updates':256 if training else 0,
                'actor_optimizer_steps':2048 if training else 0, 'critic_optimizer_steps':2048 if training else 0,
                'real_checkpoint_writes':5 if training else 0}
    if any(c[k] != v for k, v in expected.items()): raise RuntimeError('INCOMPLETE_FORMAL_COUNTERS')
    if training:
        gate = read(output/'pilot_diagnostic/first_update_gate.json')
        if not (gate['status'] == 'PASS' and gate['sample_count'] == 512 and gate['starting_update'] == 256
                and gate['first_absolute_update'] == 257 and gate['minibatches'] == 8
                and gate['float32_atol'] == 1e-6 and gate['float32_rtol'] == 1e-5
                and gate['policy_rng_consumed'] is False): raise RuntimeError('FIRST_ROLLOUT_GATE_NOT_ACCEPTED')
        if (output/'pilot_diagnostic/first_update_failure.json').exists(): raise RuntimeError('FIRST_UPDATE_FAILURE')
        for u in [272,288,320,384,512]:
            ck = output/'checkpoints'/f'u{u}.pt'; meta = read(ck.with_suffix('.json'))
            if meta['path'] != str(ck) or meta['sha256'] != sha(ck): raise RuntimeError('CHECKPOINT_WRITE_DRIFT')
    return {'raw_status':raw['status'], 'native_status':receipt['status'], 'native_exit_code':receipt['exit_code'],
            'known_close_preserved':receipt['status'] != 'SUCCEEDED', 'additional_counters':expected}


class Batch:
    def __init__(self, root=ROOT):
        self.root = Path(root); self.run = self.root/RUNTIME; self.static = self.root/STATIC
        self.plan = read(self.static/'plan.json'); self.path = self.run/'state.json'
        self.state = read(self.path) if self.path.exists() else None
        self.control = self.root/'coordination/research'; self.case = self.root/BASE/'experiments'/CASE
        self.interrupted = False

    def log(self, event, **data):
        self.run.mkdir(parents=True, exist_ok=True)
        row = {'time':time.time(), 'event':event, **data}
        with (self.run/'controller.log').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())

    def save(self, **changes):
        self.state.update(changes, updated_at=time.time()); write(self.path, self.state)

    def verify_pins(self):
        required = {'batch','analyzer','commander','launch_gate','manifest','ready','permission','final_review',
                    'checkpoint-review_prompt','final-analysis_prompt','response_schema','codex','tmux','authority','prelaunch_review'}
        if set(self.plan['pins']) != required: raise RuntimeError('PLAN_PIN_SET')
        for value in self.plan['pins'].values(): checked(value)
        for value in manifest_pins(read(checked(self.plan['pins']['manifest']))): checked(value)
        review = read(checked(self.plan['pins']['prelaunch_review']))
        if review['verdict'] != 'PASS' or review['must_fix'] or review['identities']['batch_sha256'] != sha(ROOT/'tools/r5_formal_batch.py'):
            raise RuntimeError('PRELAUNCH_REVIEW_NOT_BOUND')
        if self.plan['experiment_id'] != CASE or self.plan['model_call_limit'] != 2: raise RuntimeError('PLAN_IDENTITY')

    def store(self):
        from rlflow.store import Store
        return Store(self.control)

    def stopped(self):
        from rlflow import policy
        return policy.stopped(self.control) + ([str(self.run/'STOP')] if (self.run/'STOP').exists() else [])

    def guard(self):
        self.verify_pins()
        if self.interrupted or self.stopped() or self.store().replay()['paused']:
            raise RuntimeError('STOP_OR_PAUSE: '+str(self.stopped()))

    def command(self, label, argv):
        """An intent is durable before any side effect; never retry ambiguities."""
        ops = self.state['operations']; prior = ops.get(label)
        if prior:
            if prior.get('exit_code') == 0: return
            raise RuntimeError('AMBIGUOUS_OR_FAILED_OPERATION: '+label)
        ops[label] = {'argv':argv, 'started_at':time.time(), 'intent':True}; self.save()
        self.log('COMMAND_START', label=label, argv=argv)
        with (self.run/(label+'.stdout')).open('xb') as stdout, (self.run/(label+'.stderr')).open('xb') as stderr:
            result = subprocess.run(argv, cwd=self.root, env=self.environment(), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=240)
        ops[label].update(exit_code=result.returncode, finished_at=time.time()); self.save()
        self.log('COMMAND_EXIT', label=label, exit_code=result.returncode)
        if result.returncode: raise RuntimeError('COMMAND_FAILED: '+label)

    def environment(self):
        env = dict(os.environ); env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='.:tools/rlflow:isaaclab-training', CUDA_VISIBLE_DEVICES='', GIT_TERMINAL_PROMPT='0')
        env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes')
        return env

    def commander(self, action, group, mode):
        argv = [self.plan['control_python'], 'tools/r5_formal_commander.py', action, CASE, group, mode]
        if action == 'submit': argv.append('--start-formal')
        self.command(group+'-'+mode+'-'+action, argv)

    def initialize(self):
        from tools.r5_formal_commander import idle
        from tools.r5_formal_launch_gate import require_formal_start
        self.verify_pins(); state = idle(self.store())
        if self.path.exists(): raise RuntimeError('STATE_ALREADY_EXISTS: use run to recover, never reinitialize')
        if self.run.exists() and any(self.run.iterdir()): raise RuntimeError('RUN_DIRECTORY_NOT_EMPTY')
        if self.stopped(): raise RuntimeError('STOP_PRESENT')
        for g in ['R0','R1']:
            sp = self.case/f'inputs/{g}-research-formal-training.authorized.json'
            require_formal_start(self.root, sp, self.plan['training_approvals'][g], explicit_start=True, accepted_evidence=state['evidence'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root) != subprocess.check_output(['git','rev-parse','origin/main'],cwd=self.root):
            raise RuntimeError('HEAD_NOT_PUSHED')
        self.state = {'schema_version':'r5-formal-local-batch/1', 'experiment_id':CASE, 'status':'ARMED',
            'phase':'ARMED', 'created_at':time.time(), 'plan':pin(self.static/'plan.json'),
            'tmux_session':self.plan['tmux_session'], 'controller_identity':None, 'entries':{}, 'operations':{},
            'model_calls':{}, 'model_call_limit':2, 'automatic_retries':0, 'budget_expansions':0,
            'final_analysis':str(self.root/RESULTS/'FINAL_ANALYSIS.md')}
        self.save(); self.log('ARMED')

    def run_entry(self, group, mode):
        from rlflow import experiments
        from rlflow.store import sha as object_sha
        from tools.r5_formal_commander import previous_quiescence
        name = group+'-'+mode; specpath = self.case/'inputs'/f'{name}.authorized.json'; spec = read(specpath)
        jid = 'J-'+object_sha({'id':spec['id'],'spec':spec})[:16]
        entry = self.state['entries'].setdefault(name, {'job_id':jid, 'spec':pin(specpath)})
        if entry['job_id'] != jid or entry['spec'] != pin(specpath): raise RuntimeError('ENTRY_DRIFT')
        if entry.get('done'): return
        self.save(phase='ENTRY:'+name); self.verify_pins()
        store = self.store(); existing = store.replay()['jobs'].get(jid)
        ledger = read(self.root/BASE/'goal_budget.json')
        charged = any(e['spec_id'] == spec['id'] for e in ledger['reservations']['live_entries'])
        if can_submit(entry, existing, charged=charged, admission=(self.case/(name+'-admission.json')).exists()):
            self.guard()
            entry['submit_intent'] = True; self.save()
            self.commander('submit', group, mode)
        job = store.replay()['jobs'].get(jid)
        if not job or job['spec'] != spec: raise RuntimeError('EXACT_JOB_MISSING_OR_DRIFTED')
        entry.setdefault('wait_deadline', time.time()+spec['resources']['wall_seconds']+180); self.save()
        stop_seen = entry.get('stop_seen', False)
        while True:
            stop_seen = stop_seen or bool(self.stopped()) or self.interrupted or store.replay()['paused']
            if stop_seen and not entry.get('stop_seen'):
                entry['stop_seen'] = True; self.save()
            jd = self.control/'jobs'/jid
            if (jd/'receipt.json').exists():
                job = experiments.reconcile(store, jid)
                if job['status'] in TERMINAL and (jd/'lifeguard-ack.json').exists():
                    previous_quiescence(self.control, store.replay()); break
            elif not __import__('rlflow.jobs', fromlist=['owns']).owns(job.get('identity')):
                job = experiments.reconcile(store, jid)
            if time.time() > entry['wait_deadline']: raise RuntimeError('OWNERSHIP_RECEIPT_TIMEOUT: '+jid)
            self.log('LOCAL_WAIT', job_id=jid, status=job['status'], STOP=stop_seen)
            time.sleep(self.plan['poll_seconds'])
        e = next(e for e in read(self.root/BASE/'goal_budget.json')['reservations']['live_entries'] if e['job_id'] == jid)
        if not e['settled']: self.commander('settle', group, mode)
        e = next(e for e in read(self.root/BASE/'goal_budget.json')['reservations']['live_entries'] if e['job_id'] == jid)
        paths = [self.case/f'{name}-postrun-{i}.json' for i in range(2)]
        if not postrun_complete(e['postrun_queries_used'], [p.exists() for p in paths]): self.commander('postrun', group, mode)
        for p in paths:
            audit = read(p)
            if audit['job_id'] != jid or audit['exit_code'] != 0: raise RuntimeError('POSTRUN_IDENTITY_OR_FAILURE')
        output = Path(spec['h1']['attempt'])/'results'/mode
        entry['science_boundary'] = science_complete(spec, read(jd/'receipt.json'), output)
        entry['done'] = True; self.save(); self.log('ENTRY_CLOSED', entry=name, job_id=jid)
        if stop_seen: raise RuntimeError('STOP_AFTER_OWNED_CLEANUP: '+jid)

    def verify_evaluations(self):
        """Local independent raw SHA/original-function replay precedes model #2."""
        for g in ['R0','R1']:
            self.guard()
            output = self.root/RESULTS/f'{g}-evaluation-analysis.json'
            self.command(g+'-evaluation-evidence-check', [self.plan['cpu_python'], '-B',
                'tools/analyze_r5_formal_segment.py', 'evaluation', g, '--output', str(output)])
            result = read(output)
            if result['status'] != 'PASS_CPU_ORIGINAL_EVALUATION_REPLAY' or not result['coverage_complete']:
                raise RuntimeError('RAW_EVALUATION_VERIFICATION_FAILED')
            self.state.setdefault('evaluations_checked', {})[g] = pin(output); self.save()

    def call_model(self, slot):
        from rlflow.jobs import identity, owns
        self.guard(); self.save(phase='MODEL:'+slot)
        calls = self.state['model_calls']; result_path = self.run/'calls'/slot/'receipt.json'
        row = calls.get(slot)
        if row and row.get('accepted'):
            self.verify_accepted_model(row)
            return
        if row is None:
            if len(calls) >= 2: raise RuntimeError('MODEL_CALL_LIMIT')
            calls[slot] = {'intent':True, 'prompt':self.plan['pins'][slot+'_prompt'], 'started_at':time.time(),
                           'start_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip(),
                           'protected':self.model_protected_files()}; self.save()
            argv = [self.plan['control_python'], str(self.root/'tools/r5_formal_batch.py'), '_model-worker', slot]
            with (self.run/(slot+'-worker.log')).open('ab') as log:
                proc = subprocess.Popen(argv, cwd=self.root, env=self.environment(), stdout=log, stderr=log, start_new_session=True)
            calls[slot]['worker_identity'] = identity(proc.pid); self.save(); row = calls[slot]
        # Worker self-records its exact boot/starttime identity before invoking CLI.
        # This closes controller death between Popen and its identity save.
        request_path = self.run/'calls'/slot/'request.json'
        if not row.get('worker_identity') and not result_path.exists():
            until = time.time()+10
            while not request_path.exists() and not result_path.exists() and time.time()<until: time.sleep(.1)
            if request_path.exists():
                request = read(request_path)
                if request['slot'] != slot or request['prompt'] != row['prompt']: raise RuntimeError('MODEL_REQUEST_IDENTITY_MISMATCH')
                row['worker_identity'] = request['worker_identity']; self.save()
        while not result_path.exists():
            if not row.get('worker_identity') or not owns(row['worker_identity']):
                if request_path.exists() and read(request_path).get('codex_identity'):
                    cleanup = stop_owned(read(request_path)['codex_identity'])
                    write(self.run/'calls'/slot/'orphan-cleanup.json', cleanup)
                raise RuntimeError('AMBIGUOUS_MODEL_CALL: '+slot)
            try: self.guard()
            except BaseException:
                write(self.run/'calls'/slot/'cancel.json', {'reason':'CONTROLLER_STOP_OR_PAUSE', 'time':time.time()})
                deadline = time.time()+30
                while not result_path.exists() and owns(row['worker_identity']) and time.time()<deadline: time.sleep(.2)
                raise
            time.sleep(2)
        result = read(result_path); row['receipt'] = pin(result_path); row['exit_code'] = result['exit_code']; self.save()
        if result['exit_code'] != 0 or result['error'] is not None: raise RuntimeError('MODEL_EXIT_FAILURE: '+slot)
        for protected in row['protected']: checked(protected)
        answer = read(checked(result['last_message']))
        for artifact in answer['artifacts']: checked(artifact)
        if answer['commit'] != subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip():
            raise RuntimeError('MODEL_COMMIT_IDENTITY_MISMATCH')
        if answer['status'] != 'PASS': raise RuntimeError('MODEL_REJECTED: '+answer['summary'])
        self.guard()
        subprocess.run(['git','merge-base','--is-ancestor',row['start_commit'],'HEAD'],cwd=self.root,check=True)
        changes = subprocess.check_output(['git','diff','--name-only',row['start_commit'],'HEAD'],cwd=self.root,text=True).splitlines()
        protected_names = {str(Path(p['path']).relative_to(self.root)) for p in row['protected'] if Path(p['path']).is_relative_to(self.root)}
        if not allowed_model_changes(slot, [p for p in changes if p not in protected_names]): raise RuntimeError('MODEL_EDITED_OUTSIDE_AUTHORIZED_SCOPE')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=self.root):
            raise RuntimeError('MODEL_LEFT_UNCOMMITTED_TRACKED_CHANGES')
        untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=self.root,text=True).splitlines()
        if any(not p.startswith('docs/goal/') for p in untracked): raise RuntimeError('MODEL_LEFT_UNCOMMITTED_OUTPUTS')
        self.log('MODEL_ACCEPTED', slot=slot, summary=answer['summary'])
        if slot == 'checkpoint-review':
            review = read(self.root/RESULTS/'reviews/joint-actual-training.json')
            if review['verdict'] != 'PASS' or review['must_fix']: raise RuntimeError('CHECKPOINT_REVIEW_REJECTED')
            for g in ['R0','R1']:
                from tools.r5_formal_launch_gate import require_formal_start
                from rlflow import experiments
                sp = self.case/f'inputs/{g}-research-formal-evaluation.authorized.json'; spec = read(sp)
                approval = experiments.authorized(self.store().replay(), spec, self.control)
                require_formal_start(self.root, sp, approval['approval_id'], explicit_start=True, accepted_evidence=self.store().replay()['evidence'])
        else:
            review = read(self.root/RESULTS/'reviews/final-scientific-review.json')
            if review['verdict'] != 'PASS' or review['must_fix']: raise RuntimeError('FINAL_EVIDENCE_REJECTED')
            if not (self.root/RESULTS/'FINAL_ANALYSIS.md').is_file(): raise RuntimeError('FINAL_ANALYSIS_MISSING')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root) != subprocess.check_output(['git','rev-parse','origin/main'],cwd=self.root): raise RuntimeError('MODEL_OUTPUT_NOT_PUSHED')
        mandatory = self.model_mandatory_files(slot)
        artifacts = {str(checked(p)):p for p in answer['artifacts']}
        for p in mandatory:
            if str(p.resolve()) not in artifacts: raise RuntimeError('REQUIRED_MODEL_ARTIFACT_NOT_LISTED: '+str(p))
            content = subprocess.check_output(['git','show',answer['commit']+':'+str(p.relative_to(self.root))],cwd=self.root)
            if content != p.read_bytes(): raise RuntimeError('REQUIRED_MODEL_ARTIFACT_NOT_COMMITTED: '+str(p))
        # All previously accepted outputs remain immutable through later calls.
        for other in calls.values():
            if other.get('accepted'): self.verify_accepted_model(other)
        row.update(accepted=True, accepted_commit=answer['commit'], accepted_artifacts=answer['artifacts'], last_message=result['last_message'])
        self.save()

    def verify_accepted_model(self, row):
        checked(row['receipt']); checked(row['last_message'])
        for p in row['accepted_artifacts']: checked(p)
        subprocess.run(['git','merge-base','--is-ancestor',row['accepted_commit'],'HEAD'],cwd=self.root,check=True)

    def model_mandatory_files(self, slot):
        if slot == 'checkpoint-review':
            files = [self.root/RESULTS/'reviews/joint-actual-training.json']
            for g in ['R0','R1']:
                files += [self.root/RESULTS/f'{g}-training-analysis.json',
                          self.case/'inputs'/f'{g}-research-formal-evaluation.authorized.json',
                          self.case/'inputs'/f'{g}-research-formal-evaluation-AUTH.bound.json',
                          self.case/f'{g}-research-formal-evaluation-approval.json']
            return files
        return [self.root/RESULTS/p for p in ['reviews/final-scientific-review.json','EXPERIMENT_RESULTS.json',
                'FINAL_ANALYSIS.md','R0-evaluation-analysis.json','R1-evaluation-analysis.json']]

    def model_protected_files(self):
        """Protect accounting byte-for-byte and every existing raw/copy/input.

        Commander has finished its mutations before handing the directory to
        the model. Issuing evaluation approvals does not modify this ledger.
        """
        files = {self.root/BASE/'goal_budget.json'}
        files.update(p for p in self.case.rglob('*') if p.is_file())
        for entry in self.state['entries'].values():
            if not entry.get('done'): raise RuntimeError('MODEL_BEFORE_ENTRY_CLOSED')
            name = Path(entry['spec']['path']).name.removesuffix('.authorized.json')
            inv = read(self.case/'live'/name/'INVENTORY.json')
            for item in inv['files']:
                for key in ['original','archived']:
                    if item[key]: files.add(Path(item[key]))
        return [pin(p) for p in sorted(files)]

    def execute(self):
        from rlflow.jobs import identity
        self.run.mkdir(parents=True, exist_ok=True)
        with (self.run/'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.state['status'] == 'COMPLETE': return
            if self.state['status'] == 'STOPPED': raise RuntimeError('BATCH_STOPPED: preserve failure; no automatic retry')
            try:
                if self.state['plan'] != pin(self.static/'plan.json'): raise RuntimeError('PLAN_DRIFT')
                self.save(status='RUNNING', controller_identity=identity(os.getpid()))
                for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, lambda *_: setattr(self, 'interrupted', True))
                for g in ['R0','R1']: self.run_entry(g, 'research-formal-training')
                self.call_model('checkpoint-review')
                for g in ['R0','R1']: self.run_entry(g, 'research-formal-evaluation')
                self.verify_evaluations()
                self.call_model('final-analysis')
                self.save(status='COMPLETE', phase='COMPLETE', finished_at=time.time())
                self.log('COMPLETE', analysis=self.state['final_analysis'])
            except BaseException as error:
                detail = {'status':'STOPPED', 'phase':self.state['phase'], 'error_type':type(error).__name__,
                          'error':str(error), 'traceback':traceback.format_exc(), 'time':time.time(),
                          'no_automatic_retry':True, 'resume':'Do not reinitialize/resubmit. Inspect this exact state/Store/receipts; external STOP is never cleared here.'}
                write(self.run/'FAILURE.json', detail); self.save(status='STOPPED', failure=pin(self.run/'FAILURE.json'))
                self.log('STOPPED', error=str(error)); raise


def model_worker(slot):
    """A separate owned waiter saves the CLI exit even if the controller dies."""
    b = Batch(); b.guard()
    required = 'research-formal-training' if slot == 'checkpoint-review' else 'research-formal-evaluation'
    if b.state['model_calls'].get(slot, {}).get('prompt') != b.plan['pins'][slot+'_prompt']:
        raise RuntimeError('MODEL_INTENT_NOT_BOUND')
    if not all(b.state['entries'].get(g+'-'+required, {}).get('done') for g in ['R0','R1']):
        raise RuntimeError('MODEL_STAGE_PREREQUISITES_NOT_MET')
    if slot == 'final-analysis':
        for g in ['R0','R1']: checked(b.state['evaluations_checked'][g])
    d = b.run/'calls'/slot; d.mkdir(parents=True, exist_ok=True)
    request = d/'request.json'
    fd = os.open(d/'ONCE', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, b'one model invocation only\n'); os.fsync(fd); os.close(fd)
    prompt = checked(b.plan['pins'][slot+'_prompt']); schema = checked(b.plan['pins']['response_schema'])
    argv = [str(checked(b.plan['pins']['codex'])), '-a','never','-s','danger-full-access',
            'exec','--json','--color','never','-C',str(ROOT), '--output-schema',str(schema),
            '--output-last-message',str(d/'last-message.json'), '-']
    from rlflow.jobs import identity
    write(request, {'slot':slot, 'argv':argv, 'prompt':pin(prompt), 'schema':pin(schema), 'started_at':time.time(), 'worker_pid':os.getpid(), 'worker_identity':identity(os.getpid())})
    started = time.time(); code = -1; error = None; proc = None; ident = None; cleanup = None
    try:
        with prompt.open('rb') as stdin, (d/'events.jsonl').open('xb') as stdout, (d/'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=stdin, stdout=stdout, stderr=stderr, env=b.environment(), start_new_session=True)
            from rlflow.jobs import identity, owns
            ident = identity(proc.pid)
            write(request, {**read(request), 'codex_identity':ident})
            while proc.poll() is None:
                if (b.stopped() or b.store().replay()['paused'] or (d/'cancel.json').exists()
                        or time.time()-started > b.plan['model_timeout_seconds']
                        or sum(p.stat().st_size for p in d.iterdir() if p.is_file()) > b.plan['model_output_cap_bytes']):
                    error = 'STOP_OR_MODEL_TIMEOUT'
                    if owns(ident): os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if owns(ident): os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    break
                time.sleep(2)
            code = proc.returncode
    except BaseException as exc: error = repr(exc)
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    # Includes failure to persist request after Popen. Retain
                    # the actual child identity and reap before a receipt.
                    if ident is None: ident = identity(proc.pid)
                    cleanup = stop_owned(ident)
                    proc.wait(timeout=20)
                code = proc.returncode
            except BaseException as exc:
                error = str(error)+'; OWNED_CHILD_CLEANUP_UNKNOWN: '+repr(exc)
    last = d/'last-message.json'
    summary = None
    if last.exists():
        try: summary = read(last).get('summary')
        except (ValueError, AttributeError): pass
    write(d/'receipt.json', {'slot':slot, 'exit_code':code, 'error':error, 'started_at':started,
         'finished_at':time.time(), 'prompt':pin(prompt), 'argv':argv,
         'child_identity':ident, 'child_cleanup':cleanup,
         'last_message':pin(last) if last.exists() else None,
         'events_path':str(d/'events.jsonl'), 'stderr_path':str(d/'stderr.log'), 'summary':summary})
    return 0 if code == 0 and error is None else 1


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['init','run','_model-worker']); p.add_argument('slot', nargs='?', choices=['checkpoint-review','final-analysis']); a = p.parse_args()
    if a.action == '_model-worker':
        if not a.slot: p.error('worker slot required')
        raise SystemExit(model_worker(a.slot))
    b = Batch()
    if a.action == 'init': b.initialize()
    else:
        if b.state is None: p.error('init required, never infer a new batch')
        b.execute()

if __name__ == '__main__': main()
