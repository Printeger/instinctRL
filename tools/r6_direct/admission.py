"""Direct-run scheduling permits the user's active remote desktop connection."""
from training_workloads.unified_h1.resource_admission import POLICY as LEGACY_POLICY
from training_workloads.unified_h1.robustness_prelaunch import run_gate as shared_gate
from training_workloads.scale_003.resources import GPU_IDENTITY

POLICY=dict(LEGACY_POLICY,schema_version='r6-direct-remote-desktop-admission/1',
    memory_used_max_mib=GPU_IDENTITY['memory_total_mib'],
    allowed_apps={'gnome-remote-desktop-daemon':GPU_IDENTITY['memory_total_mib'],
                  'rustdesk':GPU_IDENTITY['memory_total_mib']})
# Available memory, measured load, STOP and fresh sampling still apply. The
# old 1 GiB used-memory ceiling and single-app rule are not scientific gates.
def run_gate(*args,**kwargs):
    return shared_gate(*args,**kwargs,window_policy=POLICY)
