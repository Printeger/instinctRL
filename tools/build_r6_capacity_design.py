"""Compose measured components into an explicit, non-authorizing capacity draft."""
import json
import math
from pathlib import Path
from training_workloads.unified_h1.robustness_analysis import pin

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r6-robustness-auto'
MIB=1024**2;GIB=1024**3


def run():
    names=['TEST_PROTOCOL.json','ARCHIVE_TEMPORAL_CPU_RNG_CAPACITY.json','FORCE_ARCHIVE_CAPACITY_CPU_V3.json',
        'STEP_ARCHIVE_CAPACITY_CPU_V2.json','TRAINING_CHUNK_STORE_CAPACITY_CPU.json','CAPACITY_ALLOCATION.json','goal_budget.json']
    p,r,f,s,t,allocation,budget=[json.loads((BASE/name).read_text()) for name in names]
    ranges={b['block_id']:b for b in r['blocks']};forces={b['block_id']:b for b in f['blocks']}
    blocks=[]
    rounding=16*MIB
    for block in p['blocks']:
        name=block['id'];range_sample=ranges[name]['estimated_measurement_bytes']
        audit_sample=forces[name]['estimated_force_bytes']+block['steps']*s['sampled_pair_bytes']
        range_quota=math.ceil(range_sample*1.15/rounding)*rounding
        audit_quota=math.ceil(audit_sample*1.15/rounding)*rounding+16*MIB
        if max(range_quota,audit_quota)>2*GIB:raise ValueError('per-file contract ceiling')
        blocks.append(dict(block_id=name,range_sample_bytes=range_sample,force_and_step_sample_bytes=audit_sample,
            range_quota_bytes=range_quota,audit_quota_bytes=audit_quota,motion_metadata_quota_bytes=16*MIB))
    native=3*GIB;original=64*MIB
    full=native+original+sum(b['range_quota_bytes']+b['audit_quota_bytes']+b['motion_metadata_quota_bytes'] for b in blocks)
    final_components=dict(four_full_evaluations=4*full,three_training_jobs=6*GIB,
        snapshots=GIB,videos=2*GIB,controller_and_reviews=GIB,failed_cleanup_and_delivery_margin=GIB)
    final_total=sum(final_components.values());final_protection=math.ceil(final_total/GIB)*GIB
    dev_limit=budget['caps']['output_actual_plus_unreleased_reservations_bytes']-final_protection
    subset=native+original+max(b['range_quota_bytes']+b['audit_quota_bytes']+b['motion_metadata_quota_bytes'] for b in blocks)
    # Retain every component and its15% sample margin; avoid wasting almost a
    # GiB merely rounding each small development evaluation to a whole GiB.
    subset_quota=math.ceil(subset/(256*MIB))*(256*MIB)
    development=dict(initial_full_baseline=full,two_affected_block_plus_original_evaluations=2*subset_quota,
        two_256_update_training_jobs=2*GIB,snapshots=GIB,controller_and_reviews=GIB,failed_prelaunch_and_repair_margin=2*GIB)
    dev_total=sum(development.values())
    fixed=sum(row['compressed_bytes'] for row in t['files'] if row['relative'].startswith('pilot_diagnostic/'))
    variable=t['projected_total_bytes']-fixed
    training_1024=fixed+4*variable
    range_seconds=sum((b['steps']+1)*max(next(c for c in r['cases'] if c['sigma_m']==b['noise_std_m'] and c['delay_cycles']==b['delay_cycles'])['encoding_seconds']) for b in p['blocks'])
    force_seconds=sum(b['steps']*20*max(next(c for c in f['cases'] if c['acceleration_m_s2']==b.get('pulse_acceleration_m_s2',0.))['encode_write_fsync_seconds']) for b in p['blocks'])
    step_seconds=sum(b['steps'] for b in p['blocks'])*2*max(row['seconds'] for row in s['samples'])
    transitions=32960+sum(b['steps']*32 for b in p['blocks'])
    base_seconds=allocation['historical_evaluation']['elapsed_seconds']*transitions/32960
    per_model_seconds=base_seconds+range_seconds+force_seconds+step_seconds
    final_train_seconds=allocation['historical_training']['elapsed_seconds']*12
    final_seconds=4*per_model_seconds+final_train_seconds+7*180+1200
    return dict(schema_version='r6-composed-capacity-draft/1',status='PROPOSAL_NOT_ACCEPTED',capacity_pass=False,
        inputs=[pin(BASE/name) for name in names],tool=pin(Path(__file__).resolve()),blocks=blocks,
        unchanged_native_allowance_bytes=native,original_suite_metadata_allowance_bytes=original,
        full_evaluation_quota_bytes=full,full_evaluation_transitions=transitions,final_nontraining_transitions=4*transitions,
        sampling_headroom='15% above CPU samples rounded up16MiB; extra16MiB/block reset/header allowance and16MiB motion/metadata. Proposed quotas, not worst-case proofs.',
        final_components_bytes=final_components,final_total_bytes=final_total,
        proposed_final_storage_protection_bytes=final_protection,existing_final_storage_protection_bytes=80*GIB,
        existing_protection_deficit_bytes=max(0,final_total-80*GIB),
        development_limit_after_proposal_bytes=dev_limit,development_example_components_bytes=development,
        development_example_total_bytes=dev_total,development_example_fits=dev_total<=dev_limit,
        scope='Final all16blocks plus original suite for three confirmation models and paired baseline unchanged. Development example is two256-update candidates, each original suite plus one preregistered affected block. Not method registration or a promise all three maximum candidates fit simultaneously; longer segments/control arms require exact admission against the same pool.',
        training_compression=dict(historical_fixed_bytes=fixed,historical_variable_per256_bytes=variable,
            extrapolated_per1024_bytes=training_1024,proposed_per_confirmation_job_bytes=2*GIB,
            implementation_status='OPT_IN_CPU_ADAPTER_TESTED_NOT_LIVE_INTEGRATED; exact historical bytes recovered including per-record flush/fsync; checkpoints may remain uncompressed'),
        final_time_seconds=dict(historical_scaled_physics_per_model=base_seconds,range_encode_per_model=range_seconds,
            force_encode_fsync_per_model=force_seconds,step_encode_fsync_per_model=step_seconds,
            three_training_1024=final_train_seconds,seven_cleanup_allowances=7*180,video_allowance=1200,
            total_sample_extrapolation=final_seconds,existing_protection=24000,proposed_protection=27000,
            development_remaining_if_accepted=43200-27000),
        unresolved=['Independent acceptance of output/time allocation changes; current ledger unchanged',
            'Opt-in lossless training adapters CPU-tested; live lifecycle integration, consumers and independent review still required',
            'Reset/header/motion and new training-noise field quotas require actual serialization checks',
            'Native logs across runtime paths require bounded guards; job control-directory sizes alone insufficient',
            'Snapshot/video/runtime metadata quotas need exact implementation and input identities',
            'GPU synchronization/CPU noise and scientific feasibility not measured by these CPU samples',
            'Full failed job may exceed repair pool; explicit future storage-release settlement needs independent review, never implicit refunds'],
        superseded_fixture_note='STEP_ARCHIVE_CAPACITY_CPU.json initial sample omitted per-lane policy_step/command fields and is excluded; V2 includes them. Initial record retained.',
        ledger_changed=False,final_coverage_reduced=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
