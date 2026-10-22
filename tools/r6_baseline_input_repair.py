"""Narrow reviewed resource-plan replacement for an unissued first baseline."""
import copy
import json
from rlflow.jobs import owns
from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile,r6_initial_baseline as initial
from tools.r6_baseline_failure import require,failed

def effective_initial(journal,controller_key,original):
    """Read an additive, reviewed first-plan repair without rewriting old inputs."""
    owner=journal.read()['events'][controller_key]
    transition=owner.get('baseline_input_repair')
    if transition is None:return original
    record=json.loads(checked(transition).read_text())
    require(record['controller_key']==controller_key and record['before']==original,
        'INPUT_REPAIR_BINDING')
    after=json.loads(checked(record['initial_request']).read_text())
    require(set(after)==set(original) and after['master_pin']==original['master_pin']
        and after['review_executable']==original['review_executable'],'INPUT_REPAIR_AUTHORITY')
    plan=json.loads(checked(after['plan_pin']).read_text())
    oldplan=json.loads(checked(original['plan_pin']).read_text())
    require(set(plan)==initial.FIELDS and all(plan[k]==oldplan[k] for k in
        initial.FIELDS-{'queue_review','sampler_review'}),'INPUT_REPAIR_SCOPE')
    spec=json.loads(checked(plan['spec_pin']).read_text())
    jid='J-'+sha(dict(id=spec['id'],spec=spec))[:16]
    initial.verify_review(journal,after['plan_pin'],record['review_key'],record['review'],plan,jid)
    return after


def adopt_reviewed_inputs(journal,controller_key,initial_pin,context_pin,review_key,store):
    """Resolve the observed stale resource acceptance, only before any issuance.

    The new plan may change resource review pins only. Both failed originals and
    the failure cursor are retained; an exact independent prelive PASS is needed.
    No Store mutation, dispatch, budget refund or new controller identity occurs.
    """
    profile.guard(journal)
    from tools.r6_job_handoff import no_model_workers
    no_model_workers(journal)
    state=journal.read();owner=state['events'][controller_key]
    require(not owns(owner.get('controller_process')),'CONTROLLER_STILL_LIVE')
    require(owner['cursor']['phase']=='BASELINE_FAILURE_HANDOFF'
        and owner.get('baseline_input_repair') is None,'INPUT_REPAIR_PHASE')
    require(not any(e['kind']=='job' or e['identity']['role']=='initial-baseline-authorization'
        for e in state['events'].values()),'INPUT_REPAIR_AFTER_ISSUANCE')
    input_key=owner['cursor']['baseline_failure_input']
    packet=state['events'][input_key]['request']
    require(packet['controller_key']==controller_key and packet['target']['boundary']=='initial_review'
        and packet['store_sha256']==sha(store.replay()),'INPUT_REPAIR_FAILURE')
    require(failed(journal,packet['target']['failed_key']),'INPUT_REPAIR_PREDECESSOR')
    original=packet['initial'];after=json.loads(checked(initial_pin).read_text())
    require(set(after)==set(original) and after['master_pin']==original['master_pin']
        and after['review_executable']==original['review_executable'],'INPUT_REPAIR_AUTHORITY')
    plan,spec,jid=initial.first_plan(journal,after['plan_pin'],store)
    oldplan=json.loads(checked(original['plan_pin']).read_text())
    require(after['plan_pin']!=original['plan_pin'] and all(plan[k]==oldplan[k]
        for k in initial.FIELDS-{'queue_review','sampler_review'}),'INPUT_REPAIR_SCOPE')
    receipt=json.loads(checked(state['events'][review_key]['receipt']).read_text())
    review=receipt['last_message']
    initial.verify_review(journal,after['plan_pin'],review_key,review,plan,jid)
    oldcontext=json.loads(checked(packet['cursor']['context']).read_text())
    context=json.loads(checked(context_pin).read_text())
    expected=copy.deepcopy(oldcontext)
    expected['runner'].update(queue_review=plan['queue_review'],sampler_review=plan['sampler_review'])
    require(context==expected,'INPUT_REPAIR_CONTEXT_SCOPE')
    require(dict(packet['cursor'],phase='BASELINE_FAILURE_HANDOFF',baseline_failure_input=input_key)
        ==owner['cursor'],'INPUT_REPAIR_CURSOR')
    record=dict(controller_key=controller_key,before=original,initial_request=initial_pin,
        context=context_pin,review_key=review_key,review=review,failure_input=input_key,
        prior_cursor=owner['cursor'],prior_status=owner['status'],store_sha256=sha(store.replay()))
    path=journal.directory/'baseline-failure'/input_key/'reviewed-input-repair.json'
    persist_exact(path,canonical(record)+b'\n')
    owner['baseline_input_repair']=file_pin(path)
    owner['cursor']=dict(copy.deepcopy(packet['cursor']),context=context_pin)
    owner['status']='BASELINE_REVIEWED_INPUTS_RESTORED'
    journal.save(state)
    return effective_initial(journal,controller_key,original)


