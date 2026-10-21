"""Budgeted source preparation after a scientific decision; never live issuance.

The worker writes source/config/card inputs. The controller alone freezes the
committed source through the existing charged snapshot helper. Parameters,
actual volume bindings, compatibility and exact prelive review follow later.
"""
import json
import math
import re
from pathlib import Path

from rlflow.store import canonical, sha
from tools.r6_recovery import RecoveryError, checked, file_pin
from tools.r6_decision import record as parent_decision, model_stop_paths, CLI, CLI_SHA, MODEL_CAP
from tools.r6_decision_packet import persist_exact, pin_bytes
from tools.r6_batch import model_handoff, verified_decision
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools.r6_budget import totals
from tools import prepare_r6_snapshot as snapshot

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r6-robustness-auto'
SCHEMA=BASE/'prompts/source-preparation.schema.json'
ROLE='source-preparation'
ARTIFACT_CAP=8*1024**2


def require(value, reason):
    if not value:raise RecoveryError('SOURCE_PREPARATION_'+reason)


def repository(journal):
    if journal.read()['goal_id']=='R6-CPU-TEST-FIXTURE':
        return journal.directory.parent/'source-fixture'
    return ROOT


def control_identity():
    # Source preparation cannot replace the controller currently supervising it,
    # including uncommitted edits which would not appear in the reported commit.
    files=list((ROOT/'tools').rglob('*.py'))+[
        ROOT/'training_workloads/unified_h1/config.py',SCHEMA]
    return sha({str(p.relative_to(ROOT)):file_pin(p)['sha256'] for p in files})


def register(journal,parent_key,store,*,executable,free_disk_bytes,repair_pin=None,engineering_pin=None):
    journal.require_writer();journal.guard()
    require(not stop_paths(journal),'STOP');no_model_workers(journal)
    parent=parent_decision(journal,parent_key,store)
    require(parent['decision']['decision']!='REQUEST_FINAL_REVIEW','FINAL_REVIEW_ROUTE')
    state=journal.read();require(state.get('current_job') is None,'ACTIVE_JOB')
    if engineering_pin is not None:
        from tools.r6_repair_input import validate
        validate(journal,engineering_pin,parent_key,store)
    exe=checked(executable)
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':
        require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXECUTABLE')
    else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
    from tools import r6_source_repair as repair
    name='SOURCE_PREPARATION:'+parent_key
    if repair_pin is not None:name+=':REPAIR:'+repair_pin['sha256']
    previous=[(k,e) for k,e in state['events'].items()
              if e['identity']['role']==ROLE and e['identity']['event_id']==name]
    require(len(previous)<=1,'PARENT_ALIAS')
    if previous:
        key,entry=previous[0]
        require('source_input' in entry,'INPUT_MISSING_NO_RETRY')
        require(repair.active(journal,parent_key)==key,'NOT_ACTIVE')
        require(entry['source_input'].get('repair')==repair_pin,'REPAIR_PIN_DRIFT')
        require(entry['source_input'].get('engineering_repair')==engineering_pin,'ENGINEERING_INPUT_DRIFT')
        require(entry['request']['executable']==executable,'EXECUTABLE_DRIFT')
        require(entry['source_input']['parent']==state['events'][parent_key]['decision_record'],'PARENT_DRIFT')
        # Intent includes recoverable prompt bytes before the first dispatch.
        if not entry.get('dispatch_claim') and not entry.get('receipt'):
            if repair_pin is not None:repair.validate_start(journal,repair_pin,parent_key)
            persist_exact(Path(entry['request']['prompt']['path']),entry['source_input']['prompt'].encode())
        return key
    if repair_pin is not None:
        predecessor=repair.validate_start(journal,repair_pin,parent_key)
        require(repair.active(journal,parent_key)==predecessor,'REPAIR_NOT_ACTIVE_TIP')
    if state['goal_id']!='R6-CPU-TEST-FIXTURE':
        from tools.r6_review_access import check
        check(executable,state['events'][parent_key]['decision_record'])
    repo=repository(journal)
    base_commit=snapshot.git(repo,'rev-parse','HEAD').decode().strip()
    packet=dict(parent=state['events'][parent_key]['decision_record'],parent_key=parent_key,
        base_commit=base_commit,repository=str(repo),
        output_directory=str(repo/'docs/experiments/r6-robustness-auto/preparations'/parent_key),
        remaining_model_sessions=state['budget']['caps']['automatic_model_sessions']-
            totals(state['budget'])['automatic_model_sessions']-1,
        requirements=file_pin(ROOT/'docs/goals/InstinctRL_R6_Robustness_Autonomous_Goal_Requirements.md'),
        master=file_pin(BASE/'MASTER_AUTHORITY_V2.json'),
        budget_before_preparation=dict(caps=state['budget']['caps'],charged=totals(state['budget']),
            protected=state['budget']['protected_confirmation_delivery_repair']),
        implementation=file_pin(__file__),snapshot_tool=file_pin(snapshot.__file__),
        control_identity=control_identity(),
        store_state_sha256=sha(store.replay()))
    if repair_pin is not None:
        packet['repair']=repair_pin
        packet['output_directory']=str(Path(packet['output_directory'])/'repairs'/repair_pin['sha256'])
    if engineering_pin is not None:packet['engineering_repair']=engineering_pin
    prompt=('Prepare the concrete R6 source/config and experiment card for the pinned scientific '
        'decision, then exit. You are the sole source writer while the controller holds the Journal '
        'lock. Do not mutate Journal/Store, issue authority, create snapshots or volumes, query GPUs, '
        'launch jobs, install dependencies, spawn agents, or wait for training. Read the pinned '
        'decision, original evidence and requirements; current pinned master amendments and ledger '
        'caps govern budget, without implicit expansion or refunds. Preserve actor inputs, frozen scoring/splits, '
        'u640 identity, native failures, and method attribution. Work only on main; commit all new '
        'source/config/card inputs with an additive commit. Do not push or change control tools. '
        'Allowed changes: isaaclab-training/, training_workloads/ (except unified_h1/config.py), '
        'tests/, and NEW files under the exact output_directory. Never modify old artifacts. '
        'Keep total changed-file bytes within 8 MiB. The JSON design card must contain exactly '
        'schema_version=r6-source-design-card/1, parent_decision_sha256, experiment_id, hypothesis, '
        'comparison, source_scope (nonempty strings), and planned_resources (a nonempty map of '
        'ledger resource names to nonnegative finite amounts). For training use schema_version=r6-source-design-card/2 '
        'and additionally training_design pin to a committed r6-training-design/1 JSON with exactly '
        'schema_version, recipe_template, plan_template, origin_config. recipe_template has all '
        'r6-training-recipe/1 fields except source_identity; plan_template has all r6-training-plan/1 '
        'fields except recipe and recipe_review. Controller binds frozen identity and real review pins. '
        'Commit method and curriculum JSON inside output_directory and pin them in recipe_template; '
        'plan experiment_id must equal card experiment_id, hypothesis/comparison must match recipe. '
        'Training config uses unified-h1-r6-training-config.v1. Never invent review receipts. '
        'It is a proposal, not acceptance. Config must satisfy '
        'tools.prepare_r6_snapshot.preflight with a fresh target and exact source hashes. '
        'Return SOURCE_READY with the exact 40-character commit and config/card pins only if '
        'ready for the existing CPU snapshot helper. Otherwise return PREPARATION_INCOMPLETE '
        'with null source_commit/config/card and concrete must_fix; do not fabricate readiness. '
        'Include parent pin in evidence_pins and every git diff path since base_commit in '
        'changed_files. Controller freezes source only after receipt, cleanup and input verification. '
        'For candidate evaluation use r6-source-design-card/3 with candidate_design pin to committed '
        'r6-candidate-design/1: selection_template (all selection fields except independent_review), '
        'evaluation_template (experiment_id, wall_seconds, protocol_review, r6), origin_config, '
        'result_payload_bytes. Use dedicated unified-h1-r6-candidate-config.v1. Commit comparison '
        'input with the same hypothesis; controller owns independent selection and compatibility '
        'reviews. Never fabricate reviewer pins. Future actual volume bindings and prelive reviews remain mandatory.\n'
        +('Read the repair document and reconcile all predecessor writes and commits. '
          'Preserve prior attempt outputs; only this fresh output_directory may receive new artifacts. '
          'This is repair of the same scientific candidate, not permission for another candidate. '
          'If repair includes rejected_review, read every reviewer finding, include exact '
          'review_findings_addressed in card.engineering_repair, and change implementation, '
          'tests or config to resolve it; new artifact names alone are not a repair. '
          'Use an experiment ID different from the rejected candidate.\n' if repair_pin is not None else '')
        +('This is an engineering repair. Read engineering_repair and every original finding. '
          'Do not change scientific method, actor inputs, scoring or splits to hide a failure. '
          'Add engineering_repair to the design card, exactly {input: the pinned engineering_repair, '
          'disposition: NEW_REVIEWED_LIVE_ATTEMPT, findings_addressed: all original must_fix strings}, '
          'plus review_findings_addressed when the repair document pins a rejected_review. '
          'Use a new experiment ID. SOURCE_READY only proposes a fresh reviewed attempt; it never '
          'authorizes dispatch. If repair only needs offline reanalysis, needs controller changes, '
          'has an external blocker, or cannot address every finding, return PREPARATION_INCOMPLETE '
          'with explicit evidence and reason. Do not force a live attempt or alter control tools.\n' if engineering_pin is not None else '')
        +json.dumps(packet,sort_keys=True)+'\n')
    require(len(prompt.encode())<=65536,'CONTEXT_CAP')
    target=journal.directory/'source-preparation'/parent_key/'prompt.txt'
    if repair_pin is not None:target=target.parent/'repairs'/repair_pin['sha256']/'prompt.txt'
    request=dict(prompt=pin_bytes(target,prompt.encode()),schema=file_pin(SCHEMA),
        executable=executable,cwd=str(ROOT),sandbox='workspace-write',timeout_seconds=3600,
        output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id=name,
        job_id=state['events'][parent_key]['identity']['job_id'],input_sha256=sha(request),role=ROLE)
    resources=dict(automatic_model_sessions=1,output_actual_plus_unreleased_reservations_bytes=MODEL_CAP+ARTIFACT_CAP+65536)
    if parent['decision']['decision']=='PROPOSE_CANDIDATE' and repair_pin is None:resources['major_scientific_candidates']=1
    key,_=journal.intent(event,kind='model',request=request,resources=resources,
        purpose='repair' if repair_pin is not None or parent['decision']['decision']=='NEEDS_REPAIR' else 'development',
        free_disk_bytes=free_disk_bytes)
    state=journal.read();state['events'][key]['source_input']=dict(packet,prompt=prompt,
        budget_sha256=sha(state['budget']))
    journal.save(state);persist_exact(target,prompt.encode())
    return key


def committed(pin,commit,repo):
    path=checked(pin)
    require(path.is_relative_to(repo),'ARTIFACT_PATH')
    require(snapshot.git(repo,'show',commit+':'+str(path.relative_to(repo)))==path.read_bytes(),
            'UNCOMMITTED_ARTIFACT')
    return path


def record(journal,key,store):
    journal.require_writer();journal.guard();require(not stop_paths(journal),'STOP')
    no_model_workers(journal);state=journal.read();entry=state['events'][key];packet=entry['source_input']
    require(entry['kind']=='model' and entry['identity']['role']==ROLE
        and sha(entry['identity'])==key and entry['identity']['input_sha256']==sha(entry['request']), 'IDENTITY')
    from tools.r6_source_repair import active
    require(active(journal,packet['parent_key'])==key,'NOT_ACTIVE')
    require(entry['request']['schema']==file_pin(SCHEMA),'SCHEMA_DRIFT')
    checked(packet['implementation']);checked(packet['snapshot_tool'])
    checked(packet['master']);checked(packet['requirements'])
    require(control_identity()==packet['control_identity'],'CONTROL_DRIFT')
    parent_decision(journal,packet['parent_key'],store)
    require(state['events'][packet['parent_key']]['decision_record']==packet['parent'],'PARENT_DRIFT')
    result=verified_decision(journal,key)
    require(result['parent_decision_sha256']==packet['parent']['sha256']
        and packet['parent'] in result['evidence_pins'],'PARENT_BINDING')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==packet['remaining_model_sessions'],'BUDGET_RESULT')
    require(sha(store.replay())==packet['store_state_sha256'],'STORE_MUTATED')
    if not entry.get('source_result'):
        require(sha(state['budget'])==packet['budget_sha256'],'BUDGET_MUTATED')
    require(result['decision']=='SOURCE_READY' and not result['must_fix'],'INCOMPLETE')
    commit=result['source_commit'];require(isinstance(commit,str) and re.fullmatch('[a-f0-9]{40}',commit),'COMMIT')
    repo=repository(journal);require(str(repo)==packet['repository'],'REPOSITORY_DRIFT')
    require(snapshot.git(repo,'branch','--show-current').decode().strip()=='main','BRANCH')
    require(snapshot.git(repo,'rev-parse','HEAD').decode().strip()==commit,'HEAD_DRIFT')
    snapshot.git(repo,'merge-base','--is-ancestor',packet['base_commit'],commit)
    changes=snapshot.git(repo,'diff','--name-only','-z',packet['base_commit'],commit).decode().split('\0')[:-1]
    require(sorted(result['changed_files'])==sorted(changes),'CHANGED_FILES')
    total=0;output=Path(packet['output_directory']).relative_to(repo).as_posix()+'/'
    for name in changes:
        require(name.startswith(('isaaclab-training/','training_workloads/','tests/',output))
            and name!='training_workloads/unified_h1/config.py','CHANGE_SCOPE')
        data=snapshot.git(repo,'show',commit+':'+name);total+=len(data)
        require((repo/name).is_file() and (repo/name).read_bytes()==data,'SOURCE_WORKTREE_DRIFT')
        if name.startswith(output):
            require(not snapshot.git(repo,'ls-tree',packet['base_commit'],'--',name),'OLD_ARTIFACT')
    require(total<=ARTIFACT_CAP,'ARTIFACT_CAP')
    config=committed(result['config'],commit,repo);card=committed(result['card'],commit,repo)
    require(config.is_relative_to(repo/output) and card.is_relative_to(repo/output),'OUTPUT_DIRECTORY')
    require(result['config'] in result['evidence_pins'] and result['card'] in result['evidence_pins'],'ARTIFACT_EVIDENCE')
    design=json.loads(card.read_text())
    training=isinstance(design,dict) and design.get('schema_version')=='r6-source-design-card/2'
    candidate=isinstance(design,dict) and design.get('schema_version')=='r6-source-design-card/3'
    fields={'schema_version','parent_decision_sha256','experiment_id','hypothesis','comparison','source_scope','planned_resources'}
    if packet.get('engineering_repair'):fields.add('engineering_repair')
    require(isinstance(design,dict) and set(design)==(fields|{'training_design'} if training else fields|{'candidate_design'} if candidate else fields)
        and design['schema_version'] in ('r6-source-design-card/1','r6-source-design-card/2','r6-source-design-card/3')
        and design['parent_decision_sha256']==packet['parent']['sha256'],'DESIGN_CARD')
    if packet.get('engineering_repair'):
        from tools.r6_repair_input import validate
        repair=validate(journal,packet['engineering_repair'],packet['parent_key'],store)
        findings=repair['description']['decision']['decision']['must_fix']
        expected=dict(input=packet['engineering_repair'],disposition='NEW_REVIEWED_LIVE_ATTEMPT',findings_addressed=findings)
        if packet.get('repair'):
            document=json.loads(checked(packet['repair']).read_text())
            if document.get('rejected_review'):
                rejected=json.loads(checked(document['rejected_review']).read_text())
                expected['review_findings_addressed']=rejected['must_fix']
                prior=json.loads(checked(rejected['source_result']).read_text())
                prior_card=json.loads(checked(prior['card']).read_text())
                require(design['experiment_id']!=prior_card['experiment_id'],'REWORK_EXPERIMENT_ID')
                diff=snapshot.git(repo,'diff','--name-only','-z',prior['source_commit'],commit).decode().split('\0')[:-1]
                require(result['config']['sha256']!=prior['config']['sha256'] or
                    any(n.startswith(('isaaclab-training/','training_workloads/','tests/')) for n in diff),
                    'REWORK_NO_IMPLEMENTATION_CHANGE')
        require(design['engineering_repair']==expected,'ENGINEERING_FINDINGS')
        old_spec=json.loads(checked(repair['description']['job_spec']).read_text())
        require(design['experiment_id']!=old_spec['id'],'ENGINEERING_NEW_EXPERIMENT')
    if training:
        from training_workloads.unified_h1.robustness_training_plan import FIELDS,RECIPE_FIELDS
        path=committed(design['training_design'],commit,repo)
        require(path.is_relative_to(repo/output),'TRAINING_DESIGN_PATH')
        draft=json.loads(path.read_text())
        require(set(draft)=={'schema_version','recipe_template','plan_template','origin_config'}
            and draft['schema_version']=='r6-training-design/1','TRAINING_DESIGN')
        recipe=draft['recipe_template'];plan=draft['plan_template']
        require(isinstance(recipe,dict) and set(recipe)==RECIPE_FIELDS-{'source_identity'}
            and isinstance(plan,dict) and set(plan)==FIELDS-{'recipe','recipe_review'},'TRAINING_TEMPLATES')
        require(plan['experiment_id']==design['experiment_id'] and all(recipe[k]==design[k]
            for k in ('hypothesis','comparison')),'TRAINING_SCIENTIFIC_BINDING')
        require(json.loads(config.read_text()).get('schema_version')=='unified-h1-r6-training-config.v1','TRAINING_CONFIG')
        checked(draft['origin_config'])
        for name in ('method','curriculum'):
            require(committed(recipe[name],commit,repo).is_relative_to(repo/output),'TRAINING_INPUT_PATH')
    if candidate:
        from tools.r6_candidate_preparation import validate_design
        path=committed(design['candidate_design'],commit,repo)
        require(path.is_relative_to(repo/output),'CANDIDATE_DESIGN_PATH')
        draft=json.loads(path.read_text());validate_design(draft)
        require(draft['evaluation_template']['experiment_id']==design['experiment_id'],'CANDIDATE_EXPERIMENT')
        require(json.loads(config.read_text()).get('schema_version')=='unified-h1-r6-candidate-config.v1','CANDIDATE_CONFIG')
        selected=draft['selection_template']
        comparison=committed(selected['comparison'],commit,repo)
        require(json.loads(comparison.read_text())['hypothesis']==design['hypothesis'],'CANDIDATE_HYPOTHESIS')
        checked(selected['method'])
    require(all(isinstance(design[k],str) and design[k].strip()
        for k in ('experiment_id','hypothesis','comparison','source_scope')),'DESIGN_CARD_FIELDS')
    planned=design['planned_resources']
    require(isinstance(planned,dict) and bool(planned) and all(k in state['budget']['caps']
        and type(v) in (int,float) and math.isfinite(v) and 0<=v<=state['budget']['caps'][k]
        for k,v in planned.items()),'DESIGN_RESOURCES')
    snapshot.preflight(config,commit)
    artifact=dict(status='SOURCE_INPUTS_READY_NOT_LIVE_AUTHORITY',parent=packet['parent'],
        source_commit=commit,config=result['config'],card=result['card'],review_required=True,
        worker_receipt=entry['receipt'],scientific_acceptance=False)
    if packet.get('engineering_repair'):artifact['engineering_repair']=packet['engineering_repair']
    if training:artifact['training_design']=design['training_design']
    if candidate:artifact['candidate_design']=design['candidate_design']
    path=Path(entry['request']['prompt']['path']).parent/'result.json'
    persist_exact(path,canonical(artifact)+b'\n')
    state=journal.read();state['events'][key]['source_result']=file_pin(path);journal.save(state)
    return artifact


def prepare_source(journal,parent_key,store,*,executable,free_disk_bytes,engineering_pin=None):
    """Reconnect an existing worker before any registration idle check."""
    journal.require_writer();state=journal.read()
    from tools.r6_source_repair import active,validate_start
    selected=active(journal,parent_key)
    entries=[] if selected is None else [(selected,state['events'][selected])]
    if entries:
        key,entry=entries[0];require(entry['request']['executable']==executable,'EXECUTABLE_DRIFT')
        require('source_input' in entry,'INPUT_MISSING_NO_RETRY')
        require(entry['source_input'].get('engineering_repair')==engineering_pin,'ENGINEERING_INPUT_DRIFT')
        if not entry.get('dispatch_claim') and not entry.get('receipt'):
            register(journal,parent_key,store,executable=executable,free_disk_bytes=free_disk_bytes,
                repair_pin=entry['source_input'].get('repair'),engineering_pin=engineering_pin)
    else:key=register(journal,parent_key,store,executable=executable,free_disk_bytes=free_disk_bytes,engineering_pin=engineering_pin)
    entry=journal.read()['events'][key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        packet=entry['source_input']
        if engineering_pin is not None:
            from tools.r6_repair_input import validate
            validate(journal,engineering_pin,parent_key,store)
        if packet.get('repair'):validate_start(journal,packet['repair'],parent_key)
        require(sha(journal.read()['budget'])==packet['budget_sha256'],'BUDGET_MUTATED')
        require(sha(store.replay())==packet['store_state_sha256'],'STORE_MUTATED')
        require(control_identity()==packet['control_identity'],'CONTROL_DRIFT')
    model_handoff(journal,key)
    result=record(journal,key,store)
    require(not stop_paths(journal),'STOP');journal.require_dispatch_budget()
    options={'purpose':'repair'} if engineering_pin is not None else {}
    frozen=snapshot.prepare(journal,checked(result['config']),result['source_commit'],**options)
    return dict(preparation=result,snapshot=frozen,scientific_acceptance=False)


def after_job(journal,job_key,store,*,executable,free_disk_bytes):
    """Resolve the finished job's unique decision, then prepare its new source."""
    journal.require_writer();state=journal.read();job=journal.canonical_job(job_key,state=state)
    from tools.r6_decision_repair import active
    parent=active(journal,job_key)
    require(parent is not None,'EXACT_JOB_DECISION')
    # Do not re-enter a new-action STOP gate before reconnecting an already
    # claimed worker. Existing handoff owns its cancellation and receipt.
    require(state['events'][parent].get('decision_input',{}).get('job_event_key')==job_key,'PARENT_JOB')
    return prepare_source(journal,parent,store,executable=executable,
                          free_disk_bytes=free_disk_bytes)


def main():
    import argparse
    import shutil
    from rlflow.store import Store
    from tools.r6_recovery import Journal
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('job_event_key')
    parser.add_argument('--store',required=True,type=Path)
    parser.add_argument('--executable',required=True,type=Path)
    args=parser.parse_args();root=args.store.resolve(strict=True)
    require((root/'runtime/research.db').is_file(),'STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=after_job(journal,args.job_event_key,Store(root),executable=file_pin(args.executable),
                         free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
