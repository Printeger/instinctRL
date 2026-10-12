"""Execute the actual backend method against CPU protocol mocks.

Metric JSON uses the real writer/parser; all environment effects are synthetic.
No optimizer, policy, GPU, or simulator is constructed.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any,Mapping
import torch
from training_workloads.scale_003.persistence import atomic_json_replace
from training_workloads.unified_h1.budget import ExecutionCounters,required_mode_budget
from training_workloads.unified_h1.control_diagnostic import COMMAND_PASSTHROUGH_ID,ACTOR_NOT_EVALUATED,fixed_b0_action,select_actor_input
from training_workloads.unified_h1.evaluation import CASES,case_commands,replay_raw_case
from training_workloads.unified_h1.logging import LearningLog

def test_fixed_new_backend_never_loads_policy_and_emits_original_sensor_metric_raw(tmp_path):
    torch.set_num_threads(1)
    source=Path('training_workloads/unified_h1/backend.py')
    method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='run_control_diagnostic')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Backend',bases=[],keywords=[],decorator_list=[],body=[method])],type_ignores=[]))
    writes=[]
    class Writer:
        def __init__(self,*args,**kwargs):pass
        def append(self,case,step,payload):
            assert payload['schema_version']=='unified-h1-control-diagnostic.step.v3'
            for key in ('actor_alpha','actor_beta','actor_mean'):assert payload['pre_step'][key]==ACTOR_NOT_EVALUATED
            assert torch.equal(payload['pre_step']['issued_action'],fixed_b0_action(32))
            writes.append((case,step))
        def finalize(self):return {'transitions':20160,'fixture_only':True}
    ns={'__name__':'training_workloads.unified_h1._mock_backend','__package__':'training_workloads.unified_h1',
        'Path':Path,'Mapping':Mapping,'Any':Any,'torch':torch,'DiagnosticChunkWriter':Writer,
        'LearningLog':LearningLog,'DIAGNOSTIC_CASES':('Z','T','S'),'CASES':CASES,'case_commands':case_commands,
        'TIMING_SEMANTICS':'pre_step is captured before environment.step; post_step and reset/cache validity are returned or observed from that same step',
        'select_actor_input':select_actor_input,'fixed_b0_action':fixed_b0_action,'atomic_json_replace':atomic_json_replace}
    exec(compile(module,str(source),'exec'),ns);backend=ns['Backend']()
    root=torch.zeros(32,13);root[:,2]=2.;root[:,3]=1.
    starts=root[:,:3,None].transpose(1,2).clone()
    policy={'lidar_grid':torch.zeros(1).expand(32,12,59,360),'state_vec':torch.zeros(32,52)}
    env=NS(device='cpu',_robot=NS(data=NS(root_state_w=root)),scene=NS(env_origins=torch.zeros(32,3)),
        _evaluation_ray_snapshot=NS(ray_starts_w=starts),_lee=None,_wrench=None)
    env.set_h1_evaluation_case=lambda case:setattr(env,'case',case)
    env.set_h1_initial_command=lambda command,step:setattr(env,'command',command)
    env._refresh_task_snapshot=lambda:None
    env.prepare_h1_step=lambda **kw:setattr(env,'command',kw['command_b'])
    def step(action):
        masks={key:torch.zeros(32,1,dtype=torch.bool) for key in ('collision','below_height','above_height')}
        pre={'task':{'actual_velocity_b':env.command,'termination':{'masks':masks},'exact_clearance_m':torch.full((32,1),2.)},
             'control':{'ics_metrics':{'beta':torch.ones(32,1)}}}
        extras={'pre_reset':pre,'evaluation_ray_source':{'ray_starts_w':starts},'reward_diagnostics':{'q':torch.ones(32,1,dtype=torch.bool)}}
        return {'policy':policy},torch.zeros(32),torch.zeros(32,dtype=torch.bool),torch.zeros(32,dtype=torch.bool),extras
    env.step=step;backend.environment=env;backend.output=tmp_path;backend.attempt=tmp_path/'attempt';backend.trainer=None
    backend.counters=ExecutionCounters();backend.resource_check=lambda *args:None
    params={'condition':'B0','experiment_id':COMMAND_PASSTHROUGH_ID,'checkpoint':None,
        'checkpoint_sha256':None,'checkpoint_source_identity':None,'cases':['Z','T','S'],'transitions':20160}
    backend.authorization={'budgets':required_mode_budget('control-diagnostic',params)}
    def reset(*,seed):
        backend.counters.reserve_reset(automatic=False,maximum=3)
        return {'policy':policy},{}
    backend._preserving_reset=reset
    def forbidden(*args,**kwargs):raise AssertionError('fixed passthrough must not load a policy')
    backend._load_evaluation_checkpoint=forbidden
    backend.run_control_diagnostic(params)
    assert len(writes)==630 and backend.counters.real_optimizer_updates==0
    for case in ('Z','T','S'):
        traces=replay_raw_case(str(tmp_path/f'evaluation_raw/command_passthrough/{case}.jsonl'),case=case)
        assert len(traces)==32 and traces[0].initial_sensor_position_w_m.tolist()==[0.,0.,2.]
        assert traces[0].seed==20260909
