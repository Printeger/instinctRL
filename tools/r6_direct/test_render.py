"""Recorded trajectory CPU contract and admitted-app orchestration, no GPU."""
import json
from types import SimpleNamespace
import pytest
from tools.test_r6_video_trajectory import trajectory
from tools.prepare_r6_video_trajectories import prepare
from tools.r6_evidence import file_pin
from tools.r6_direct.render import run, startup_settings, validate_cache_tokens
from tools.r6_direct.run import run_resources
from training_workloads.unified_h1.direct_contract import render_runtime_contract, validate_runtime_contract
from training_workloads.unified_h1.budget import ExecutionCounters


def test_local_render_contract_and_process_resource_accounting(trajectory,tmp_path,monkeypatch):
    _,_,write=trajectory;clip=write()
    selection=dict(schema_version='r6-video-selection/1',status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',
        clips=[clip],total_frames=200,fps=10,physics_reintegrated=False,dispatch_authorized=False)
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection))
    prepare(file_pin(path),tmp_path/'extracted')
    contract=render_runtime_contract(dict(experiment_id='R6-DIRECT-RENDER-CPU',
        extraction=file_pin(tmp_path/'extracted/result.json'),wall_seconds=60,output_cap_bytes=20250099712))
    assert validate_runtime_contract(contract)==contract
    resources=run_resources(contract)
    assert resources['live_starts']==resources['simulator_initializations']==1
    assert resources['training_transitions']==resources['nontraining_transitions']==0
    from tools import render_r6_isaac_video
    calls=[]
    def fake_render(*args,**kwargs):
        kwargs['check']();calls.append(kwargs)
        return dict(physics_steps=0,policy_inference_steps=0,optimizer_updates=0,checkpoint_writes=0)
    monkeypatch.setattr(render_r6_isaac_video,'render',fake_render)
    output=tmp_path/'output';output.mkdir();checks=[]
    args=(dict(snapshot=str(tmp_path)),contract,dict(execution={'isaacsim_asset_root':'fixture'}),
          output,object(),ExecutionCounters(),SimpleNamespace(check=lambda **kw:checks.append(kw)))
    run(*args)
    assert len(calls)==1 and checks
    assert json.loads((output/'status.json').read_text())['counters']['real_optimizer_updates']==0
    ticks=iter([0.,61.])
    with pytest.raises(TimeoutError):run(*args,clock=lambda:next(ticks))


def test_render_cache_must_resolve_inside_this_attempt(tmp_path):
    settings = startup_settings(tmp_path)
    assert settings['/exts/omni.kit.viewport.window/startup/disableWindowOnLoad'] is True
    assert settings['/exts/omni.replicator.core/Orchestrator/enabled'] is True
    resolved = {'${cache}': str(tmp_path/'render_cache'),
                '${omni_global_cache}': str(tmp_path/'render_global_cache')}
    validate_cache_tokens(settings, SimpleNamespace(resolve=resolved.__getitem__))
    for name in resolved:
        wrong = dict(resolved, **{name: '/frozen/runtime/cache'})
        with pytest.raises(RuntimeError, match='DIRECT_RENDER_CACHE_TOKEN'):
            validate_cache_tokens(settings, SimpleNamespace(resolve=wrong.__getitem__))
