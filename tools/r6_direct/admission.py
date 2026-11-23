"""Direct-run scheduling permits the user's active remote desktop connection."""
from training_workloads.unified_h1.resource_admission import POLICY as LEGACY_POLICY
from training_workloads.unified_h1.robustness_prelaunch import run_gate as shared_gate
from training_workloads.scale_003.resources import GPU_IDENTITY
from tools.r6_direct.resource_policy import CYCLE_PREFIX

POLICY=dict(LEGACY_POLICY,schema_version='r6-direct-remote-desktop-admission/1',
    memory_used_max_mib=GPU_IDENTITY['memory_total_mib'],
    allowed_apps={'gnome-remote-desktop-daemon':GPU_IDENTITY['memory_total_mib'],
                  'rustdesk':GPU_IDENTITY['memory_total_mib']})
CYCLE_POLICY=dict(POLICY,schema_version='r6-direct-remote-desktop-admission/2',
    allowed_app_process_counts={'gnome-remote-desktop-daemon':1,'rustdesk':2})
# Available memory, measured load, STOP and fresh sampling still apply. The
# old 1 GiB used-memory ceiling and single-app rule are not scientific gates.
def run_gate(*args,**kwargs):
    policy=CYCLE_POLICY if kwargs['attempt_id'].startswith(CYCLE_PREFIX) else POLICY
    return shared_gate(*args,**kwargs,window_policy=policy)
