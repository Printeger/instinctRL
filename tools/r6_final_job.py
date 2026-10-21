"""Controller-side final-batch provenance gate, before any new job side effect.

The frozen workload separately validates checkpoint/model/method and capacity.
This gate binds that request to the controller's single published/released batch.
"""
import json
from tools.r6_recovery import checked,RecoveryError
from tools import r6_holdout_publication as publication


def require(value,reason):
    if not value:raise RecoveryError('FINAL_JOB_'+reason)


def validate(journal,parameters,purpose):
    require(purpose=='confirmation_delivery','PROTECTED_PURPOSE')
    require(isinstance(parameters,dict) and 'final_selection' in parameters
        and not {'evaluation_selection','r6_training_plan'}.intersection(parameters),'EXCLUSIVE_STAGE')
    selection=parameters['final_selection']
    require(isinstance(selection,dict) and set(selection)=={'schema_version','batch','assignment_index'}
        and selection['schema_version']=='r6-final-selection/1','SELECTION_FIELDS')
    batch=publication.validate_batch(journal,selection['batch'])
    index=selection['assignment_index']
    require(type(index)is int and 0<=index<len(batch['assignments']),'ASSIGNMENT_INDEX')
    assignment=batch['assignments'][index]
    protocol=json.loads(checked(batch['protocol']).read_text())
    freeze=json.loads(checked(batch['freeze']).read_text())
    require(parameters.get('baseline_bindings')==freeze['evidence']['baseline_bindings'],'BASELINE_BINDING')
    r=parameters.get('r6',{})
    require(all(r.get(k)==v for k,v in dict(protocol=batch['protocol'],seed=assignment['evaluation_seed'],
        split='final',block_ids=[b['id'] for b in protocol['blocks']]).items()),'EXACT_BATCH_REQUEST')
    require(parameters.get('evaluation_seed')==20260909,'ORIGINAL_SEED')
    return assignment
