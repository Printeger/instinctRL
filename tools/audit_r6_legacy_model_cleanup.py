"""Read-only historical model-event reconciliation proposal, never admission.

No signals, process starts, receipt rewrites or automatic acceptance. Original
command transcripts remain pinned for independent review of ownership scope.
"""
import json
import os
from pathlib import Path
from rlflow.jobs import owns
from tools.r6_recovery import checked,file_pin

LEGACY_EVENTS=frozenset({
 '655b94d7271bec2c5e1b6132bebcff0bbb620170f49122caf77595acfd35fe01',
 'badf0aab50e96ec4ec4a8c2aedc325d7f8e6dc8f09bacefcf79566456584ec75'})


def audit_event(key,entry):
    if key not in LEGACY_EVENTS or entry['kind']!='model':raise ValueError('LEGACY_EVENT_NOT_LISTED')
    receipt=json.loads(checked(entry['receipt']).read_text())
    if receipt.get('event_key')!=key or 'cleanup' in receipt:raise ValueError('LEGACY_RECEIPT_IDENTITY_OR_SCHEMA')
    for field in ('events','stderr','last_message'):checked(receipt[field])
    rows=[json.loads(line) for line in checked(receipt['events']).read_text().splitlines()]
    started={};completed={};mcp_started={};mcp_completed={};other=[];turns=[]
    for row in rows:
        item=row.get('item',{});kind=row.get('type')
        if kind=='turn.completed':turns.append(row)
        if item.get('type')=='command_execution':
            target=started if kind=='item.started' else completed if kind=='item.completed' else None
            if target is not None:
                if item['id'] in target:raise ValueError('DUPLICATE_COMMAND_EVENT')
                target[item['id']]=item
        elif item.get('type')=='mcp_tool_call':
            target=mcp_started if kind=='item.started' else mcp_completed if kind=='item.completed' else None
            if target is not None:
                if item['id'] in target:raise ValueError('DUPLICATE_MCP_EVENT')
                target[item['id']]=item
        elif item.get('type') not in (None,'reasoning','agent_message'):

            other.append(dict(event_type=kind,item_type=item.get('type'),item_id=item.get('id')))
    commands=[]
    for identity,item in started.items():
        finish=completed.get(identity)
        commands.append(dict(item_id=identity,command=item.get('command'),
            completed=finish is not None,exit_code=None if finish is None else finish.get('exit_code'),
            exact_command_matches=finish is not None and finish.get('command')==item.get('command')))
    mcp=[]
    for identity,item in mcp_started.items():
        finish=mcp_completed.get(identity)
        mcp.append(dict(item_id=identity,server=item.get('server'),tool=item.get('tool'),arguments=item.get('arguments'),
            completed=finish is not None and finish.get('status') in ('completed','failed'),
            exact_call_matches=finish is not None and all(finish.get(k)==item.get(k) for k in ('server','tool','arguments')),
            error=None if finish is None else finish.get('error')))
    commands_terminal=bool(turns) and all(c['completed'] and type(c['exit_code'])is int and c['exact_command_matches'] for c in commands) and set(completed)==set(started)
    return dict(event_key=key,original_receipt=entry['receipt'],original_events=receipt['events'],
        original_stderr=receipt['stderr'],original_last_message=receipt['last_message'],
        receipt_exit_code=receipt.get('exit_code'),receipt_error=receipt.get('error'),
        worker_identity=receipt.get('process'),cli_identity=receipt.get('child_identity'),
        worker_currently_alive=owns(receipt.get('process')),cli_currently_alive=owns(receipt.get('child_identity')),
        turn_completions=len(turns),commands=commands,unmatched_completed_ids=sorted(set(completed)-set(started)),
        other_item_types=other,mcp_calls=mcp,unmatched_completed_mcp_ids=sorted(set(mcp_completed)-set(mcp_started)),
        all_reported_commands_terminal=commands_terminal,
        all_reported_tool_items_terminal=commands_terminal and not other and set(mcp_started)==set(mcp_completed) and all(c['completed'] and c['exact_call_matches'] for c in mcp),
        historical_descendant_cleanup_proven=False,independent_reconciliation_required=True)


def build(directory):
    state_path=directory/'state.json';state=json.loads(state_path.read_text())
    rows=[audit_event(key,state['events'][key]) for key in sorted(LEGACY_EVENTS)]
    return dict(schema_version='r6-legacy-model-reconciliation-proposal/1',
        status='EVIDENCE_ONLY_PENDING_INDEPENDENT_RECONCILIATION',state=file_pin(state_path),
        auditor=file_pin(Path(__file__)),events=rows,process_observation_pid=os.getpid(),
        observed_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        modern_cleanup_schema_waived=False,no_model_workers_gate_changed=False,
        runtime_dispatch_authorized=False,claims={'original_receipts_unmodified':True,
        'process_checks_are_current_observations_not_historical_cleanup_proof':True},
        remaining=['Independent inspection of complete terminal tool commands for spawning/ownership ambiguity',
        'Explicit narrow reconciliation acceptance and validator before any legacy receipt can satisfy no_model_workers',
        'All future worker/CLI absence must still be checked at actual handoff; this snapshot cannot authorize later training'])


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(build(root/'docs/experiments/r6-robustness-auto/batch'),indent=2))
