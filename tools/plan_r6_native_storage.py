"""Quantify a native-output hard-limit requirement; never authorize or apply it."""
import hashlib
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto'
GIB=1024**3


def plan():
    path=BASE/'CAPACITY_COMPOSED_DRAFT_V5.json'
    d=json.loads(path.read_text())
    old=d['unchanged_native_allowance_bytes'];proposed=3*GIB//2
    delta=old-proposed
    # Four evaluations and three training jobs remain in final confirmation;
    # baseline, one affected-block evaluation and one training job in development.
    final=dict(d['final_components_bytes'])
    final['four_full_evaluations']-=4*delta
    final['three_training_jobs']-=3*delta
    development=dict(d['development_example_components_bytes'])
    for key in ('initial_full_baseline','one_affected_block_plus_original_evaluation',
                'one_256_update_training_job'):
        development[key]-=delta
    protection=math.ceil(sum(final.values())/GIB)*GIB
    remaining=128*GIB-protection
    return dict(status='CONDITIONAL_NATIVE_STORAGE_REQUIREMENT_NOT_IMPLEMENTED',capacity_pass=False,
        input=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        original_native_bytes_per_job=old,proposed_native_bytes_per_job=proposed,
        final_components_bytes=final,development_components_bytes=development,
        hypothetical_final_protection_bytes=protection,
        hypothetical_development_limit_bytes=remaining,
        hypothetical_development_total_bytes=sum(development.values()),
        hypothetical_development_headroom_bytes=remaining-sum(development.values()),
        required_enforcement=[
            'Hard aggregate limit of 1610612736 bytes for every job native output, including caches and crash artifacts; a per-file limit or periodic scan is insufficient',
            'All writable native paths identified and contained; reads of frozen runtime/assets/extensions remain available without mutating them',
            'Original console/crash evidence retained on exhaustion; fail closed and owned-process cleanup, without deleting old evidence or refunding reservations',
            'Persistent per-job evidence identity and accounting across controller restart; no reuse of a previous job quota as an implicit refund',
            'CPU multi-file and restart/exhaustion tests followed by exact independent review before permission issuance',
            'Verify actual Kit path settings and output containment during authorized initialization; a configured mount alone is not a proof'],
        current_evidence=dict(namespace_probe='NATIVE_FILESYSTEM_LIMITATION_CPU.json',
            native_paths='NATIVE_WRITER_SOURCE_AUDIT.json',
            console_capture='CPU_ACTUAL_NATIVE_LOGGER_VERIFICATION.json'),
        constraints=['No filesystem quota or mount created by this planning tool',
            'No lower native limit in production entry or ledger until enforcement and review pass',
            'No compression probe assumed; original measurement quotas and complete final coverage retained',
            'Existing snapshot/repair/video/controller reservations retained; additional candidates require remaining-budget admission'],
        ledger_changed=False,runtime_changed=False,coverage_reduced=False)


if __name__=='__main__':print(json.dumps(plan(),indent=2))
