from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from training_workloads.unified_h1.config import CONFIG_PATH, METHOD_SHA256, load_config, sha256_file
from training_workloads.unified_h1.budget import ExecutionCounters, required_mode_budget
from training_workloads.unified_h1.driver import UnifiedH1Driver, UnifiedH1TrainingLoop
from training_workloads.unified_h1.evaluation import EpisodeTrace, case_commands, case_initial_positions, continuation_decision, evaluate_checkpoint, evaluate_episode, evaluate_stop_episode, replay_raw_case, sensor_origin_surface_distance, wilson_interval
from training_workloads.unified_h1.logging import LearningLog, grouped_reward_summary
from training_workloads.unified_h1.r3_sequence import run_r3_controlled_failure, run_r3_main
from training_workloads.unified_h1.schedule import ScheduleClock, command_at


def test_commander_stop_and_finite_geometry_counterexamples_are_closed():
    n = 331; positions = torch.zeros(n, 3); positions[:, 2] = 2
    positions[:101, 0] = torch.arange(101) * .02; positions[101:, 0] = 2; positions[111:, 2] += torch.arange(1, n - 110) * .05
    climb = evaluate_stop_episode(speed_norm_m_s=[.2]*100 + [0.]*11 + [.5]*(n-111), positions_w_m=positions,
                                  zero_effective_index=100, episode_ids=[1]*n)
    still = evaluate_stop_episode(speed_norm_m_s=[0.]*n, positions_w_m=torch.tensor([0.,0.,2.]).repeat(n,1),
                                  zero_effective_index=100, episode_ids=[1]*n)
    assert not climb["success"] and climb["reason"] == "POST_STOP_Z_GATE"
    assert not still["success"] and still["reason"] == "INCOMING_TRACKING_FAILED"
    points = torch.tensor([[4.6,0,2],[504.6,0,2],[6,20,2]])
    distance, surface = sensor_origin_surface_distance(points, torch.tensor([[0,0,0],[500,0,0],[0,0,0]]))
    assert torch.allclose(distance, torch.tensor([1.4,1.4,2.]), atol=1e-5) and surface == ("x_wall","x_wall","floor")


def _trace(case: str, lane: int, *, fail: str | None = None) -> EpisodeTrace:
    commands = case_commands(case)[:, lane].clone(); steps = len(commands); velocity = commands.clone()
    origin = torch.tensor([500. * (lane % 2), 0., 0.]); position = origin.repeat(steps,1); position[:,2] = 2.
    if case == "Z": velocity.zero_()
    elif case == "S":
        velocity[100:].zero_(); position[:,0] += torch.cat((torch.arange(100)*.02, torch.full((steps-100,),1.98)))
    elif case == "X":
        position[:,0] += torch.cumsum(velocity[:,0],0)*.1; position[:,1] += torch.cumsum(velocity[:,1],0)*.1
    elif case == "W":
        velocity.zero_(); velocity[:11] = commands[:11] * .5
        if lane < 16: position[:,0] += 4.6 + torch.cumsum(velocity[:,0],0)*.1
        else: position[:,1] += 4.6 + torch.cumsum(velocity[:,1],0)*.1
    collision = torch.zeros(steps,dtype=torch.bool); below=collision.clone(); above=collision.clone(); missing=collision.clone()
    if fail == "collision": collision[3]=True
    if fail == "missing": missing[4]=True
    initial_root = case_initial_positions(case)[lane] + origin
    initial_sensor=initial_root+torch.tensor([0.,0.,.05])
    if case == "W": position[0] = initial_sensor
    elif case == "Z": position[0] = initial_sensor
    return EpisodeTrace(case,lane,commands,velocity,position,origin,tuple([1]*steps),collision,below,above,missing,
        initial_sensor_position_w_m=initial_sensor, initial_root_position_w_m=initial_root,
        initial_velocity_b=torch.zeros(3), initial_root_quaternion_wxyz=torch.tensor([1.,0.,0.,0.]),
        sample_policy_steps=tuple(range(steps)))


def test_complete_raw_episode_parser_denominators_penalties_and_safety():
    traces = [_trace(case,lane) for case in ("Z","T","S","X","W") for lane in range(32)]
    result = evaluate_checkpoint(traces, update=64)
    assert result["all_gates_pass"] and all(row["total"] == 32 for row in result["cases"].values())
    broken = list(traces); broken[0] = _trace("Z",0,fail="collision"); broken[33] = _trace("T",1,fail="missing")
    failed = evaluate_checkpoint(broken, update=128)
    assert not failed["all_gates_pass"] and failed["collisions"] == 1
    assert failed["failures"]["Z"] == 1 and failed["failures"]["T"] == 1
    assert failed["error_components"]["z_speed"] >= 5/32


def _summary(update: int, value: float, *, collisions=0, heights=0):
    cases = {name:{"successes":32,"total":32,"pass":True,"wilson_95":[.89,1.],
        "subgroups":({f"d{i}":{"successes":4,"total":4,"pass":True} for i in range(8)} if name != "W" else
                     {axis:{"successes":16,"total":16,"pass":True} for axis in ("x_wall","y_wall")})} for name in ("Z","T","S","X","W")}
    interval=wilson_interval(32,32)
    for outcome in cases.values(): outcome["wilson_95"]=interval
    metrics={"Z":{"speed":value*.08,"drift":value*.3,"height":value*.2},
        "T":{"rmse":value*.15},"S":{"stop_time":value*3,"drift":value*.3},
        "X":{"rmse":value*.15},"W":{"min_distance":.8*(1-value)}}
    episodes=[]
    for name in ("Z","T","S","X","W"):
        for lane in range(32):
            episodes.append({"case":name,"env_index":lane,"success":True,"reason":"PASS","metrics":metrics[name],
                "engineering":{"collision":lane < collisions and name == "Z",
                    "height_failure":lane < heights and name == "T","missing":False,"reset_interrupted":False,
                    "evidence_valid":True}})
    return {"schema_version":"metric-h1-v1.checkpoint.v1","update":update,"episodes":episodes,"cases":cases,
            "error_components":{key:value for key in ("z_speed","z_drift","z_height","t_rmse","s_stop_time","s_drift","x_rmse","w_min_distance")},
            "loss":value,"failures":{key:0 for key in ("Z","T","S","X","W")},
            "collisions":collisions,"height_failures":heights,"invalid_evidence_episodes":0,
            "engineering_gates_pass":collisions==heights==0,"all_gates_pass":collisions==heights==0}


def test_r4_r5_continuation_requires_all_nodes_safety_forgetting_and_complete_cases():
    assert continuation_decision(baseline=_summary(0,2),current=_summary(256,1))["reason"] == "R4_REQUIRES_U192_U256_AND_U64_EVIDENCE"
    assert not continuation_decision(baseline=_summary(0,2),current=_summary(256,1),previous=_summary(192,1.1,collisions=1),u64=_summary(64,1))["extend"]
    assert continuation_decision(baseline=_summary(0,.8),current=_summary(256,.4),previous=_summary(192,.5),u64=_summary(64,.5))["extend"]
    assert not continuation_decision(baseline=_summary(256,2),current=_summary(320,1),previous=None,stage="R5")["extend"]


class _SequenceBackend:
    def __init__(self, fail_preflight=False, close_error=False): self.events=[]; self.fail_preflight=fail_preflight; self.close_error=close_error
    def resource_check(self,*args):
        if self.fail_preflight: raise ValueError("unexpected_preflight_failure")
    def initial_preflight(self): pass
    def evaluate_b0(self,case): pass
    def cold_start(self): pass
    def save_checkpoint(self,label): pass
    def smoke_update(self,profile): pass
    def reload_final(self): pass
    def mixed_reload_rollout(self,steps): pass
    def initial_raw_rays(self): self.events.append("raw-rays")
    def preserve(self,status,outcome=None): self.events.append((status, None if outcome is None or outcome.primary_error is None else str(outcome.primary_error), 0 if outcome is None else len(outcome.cleanup_failures)))
    def close_bounded(self):
        self.events.append(("close",None,0))
        if self.close_error: raise TimeoutError("bounded close deadline")


def test_new_sequence_preserves_primary_cleanup_double_failure_and_exact_injection():
    clean = _SequenceBackend(); assert run_r3_main(clean) == 0 and clean.events[-1][0] == "PASS"
    cleanup = _SequenceBackend(close_error=True); assert run_r3_main(cleanup) == 1
    assert cleanup.events[-1][0] == "FAIL" and cleanup.events[-1][2] == 1
    double = _SequenceBackend(fail_preflight=True,close_error=True); assert run_r3_main(double) == 1
    assert double.events[-1][1] == "unexpected_preflight_failure" and double.events[-1][2] == 1
    wrong = _SequenceBackend(fail_preflight=True); assert run_r3_controlled_failure(wrong) == 1
    assert all(event[0] != "EXPECTED_CONTROLLED_FAILURE" for event in wrong.events)
    exact = _SequenceBackend(); assert run_r3_controlled_failure(exact) == 1
    assert [event[0] for event in exact.events][-1] == "EXPECTED_CONTROLLED_FAILURE"


class _TrainEnv:
    def __init__(self, observation, fail_step=None): self.observation=observation; self.prepared=None; self.index=0; self.fail_step=fail_step
    def prepare_h1_step(self,**kwargs): self.prepared=kwargs
    def step(self,action):
        self.index += 1
        if self.fail_step == self.index: raise RuntimeError("injected env.step failure")
        state=self.observation["policy"]["state_vec"].clone(); state[:,45:48]=self.prepared["next_command_b"]
        next_obs={"policy":{"lidar_grid":self.observation["policy"]["lidar_grid"].clone(),"state_vec":state},"critic":torch.zeros(32,21)}
        zero=torch.zeros(32,1); root=torch.zeros(32,13); root[:,2]=2
        extras={"reward_components":{"reward_velocity":zero},"reward_diagnostics":{"q":zero.to(torch.bool)},
            "pre_reset":{"root_state_w":root,"task":{"actual_velocity_b":torch.zeros(32,3),
                "exact_clearance_m":torch.ones(32,1),"termination":{"masks":{},"reason_code":torch.zeros(32,1,dtype=torch.long)}}}}
        self.observation=next_obs
        return next_obs,torch.zeros(32),torch.zeros(32,dtype=torch.bool),torch.zeros(32,dtype=torch.bool),extras


def _training(tmp_path, fail_step=None):
    from instinctlab.training import HistoryActor,NetworkConfig,PPOConfig,PPOTrainer
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    network=NetworkConfig(lidar_channels=(2,2,2),state_hidden=4,actor_hidden=8,critic_hidden=(8,8))
    trainer=PPOTrainer(HistoryActor(network),PrivilegedCriticH1(network),PPOConfig(minibatch_size=64,epoch_count=1))
    state=torch.zeros(32,52); observation={"policy":{"lidar_grid":torch.zeros(32,12,59,360),"state_vec":state},"critic":torch.zeros(32,21)}
    counters=ExecutionCounters(); log=LearningLog(tmp_path); env=_TrainEnv(observation,fail_step)
    driver=UnifiedH1Driver(env,observation,clock=ScheduleClock(1023),counters=counters,journal=log)
    return UnifiedH1TrainingLoop(driver,trainer,log),trainer,counters


def test_actual_driver_rollout_ppo_logging_and_durable_attempt(tmp_path):
    loop,trainer,counters=_training(tmp_path); result=loop.run_update()
    assert result["update_counter"]==1 and counters.attempted_updates==counters.completed_updates==1
    assert counters.actor_optimizer_steps==counters.critic_optimizer_steps==8
    assert len((tmp_path/"compact_rollouts.jsonl").read_text().splitlines())==16
    records=[json.loads(row) for row in (tmp_path/"accounting.jsonl").read_text().splitlines()]
    assert records[0]["event"]=="UPDATE_ATTEMPT" and sum(row["event"]=="ENV_STEP_ATTEMPT" for row in records)==16
    assert len((tmp_path/"minibatches.jsonl").read_text().splitlines())==8
    assert (tmp_path/"command_blocks.jsonl").is_file() and (tmp_path/"measurement_windows.jsonl").is_file()
    loop.journal.finalize_right_censored()
    assert len((tmp_path/"episodes.jsonl").read_text().splitlines())==32


def test_env_and_partial_critic_optimizer_failures_keep_actual_counters(tmp_path):
    loop,trainer,counters=_training(tmp_path/"env",fail_step=3)
    with pytest.raises(RuntimeError,match="env.step"): loop.run_update()
    assert counters.attempted_policy_steps==3 and counters.completed_policy_steps==2 and counters.completed_updates==0
    assert len((tmp_path/"env/partial_rollout_steps.jsonl").read_text().splitlines())==2
    assert len((tmp_path/"env/partial_episodes.jsonl").read_text().splitlines())==32
    boundary=json.loads((tmp_path/"env/partial_rollout_boundaries.jsonl").read_text())
    assert boundary["durable_step_rows"]==2 and boundary["unproven_attempts"]==1
    loop,trainer,counters=_training(tmp_path/"critic")
    original=trainer.critic_optimizer.step
    def fail(*args,**kwargs): raise RuntimeError("injected critic optimizer failure")
    trainer.critic_optimizer.step=fail
    with pytest.raises(RuntimeError,match="critic optimizer"): loop.run_update()
    failure=json.loads((tmp_path/"critic/failures.jsonl").read_text())
    assert failure["actor_optimizer_steps"]==1 and failure["critic_optimizer_steps"]==0
    trainer.critic_optimizer.step=original
    loop,trainer,counters=_training(tmp_path/"actor")
    actor_original=trainer.actor_optimizer.step
    trainer.actor_optimizer.step=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError("injected actor optimizer failure"))
    with pytest.raises(RuntimeError,match="actor optimizer"): loop.run_update()
    failure=json.loads((tmp_path/"actor/failures.jsonl").read_text())
    assert failure["actor_optimizer_steps"]==failure["critic_optimizer_steps"]==0
    trainer.actor_optimizer.step=actor_original


def test_nonfinite_log_values_are_explicitly_unavailable_and_keep_counts():
    result=grouped_reward_summary({"reward":torch.tensor([-1.,float("nan")])},{"zero":torch.tensor([True,True])})
    assert result["zero"]["reward"]=={"status":"UNAVAILABLE","reason":"NONFINITE_SAMPLES","count":2,"finite_count":1,"nonfinite_count":1}


def test_step_budget_and_resource_decline_stop_before_environment_action():
    observation={"policy":{"state_vec":torch.zeros(32,52)}}
    class Env:
        calls=[]
        def prepare_h1_step(self,**kwargs): self.calls.append("prepare")
        def step(self,action): self.calls.append("step")
    env=Env(); counters=ExecutionCounters()
    driver=UnifiedH1Driver(env,observation,counters=counters,transition_limit=0)
    with pytest.raises(RuntimeError,match="training-transition"): driver.step(torch.zeros(32,4))
    assert env.calls==[] and counters.attempted_policy_steps==0
    driver=UnifiedH1Driver(env,observation,counters=counters,before_step=lambda ordinal: (_ for _ in ()).throw(RuntimeError("resource declined")))
    with pytest.raises(RuntimeError,match="resource declined"): driver.step(torch.zeros(32,4))
    assert env.calls==[] and counters.attempted_policy_steps==0


def test_raw_evaluation_trace_is_read_only_replayable_and_incomplete_input_rejected(tmp_path):
    case="T"; source=[_trace(case,lane) for lane in range(32)]; log=LearningLog(tmp_path)
    roots=torch.stack([row.initial_root_position_w_m for row in source])
    origins=torch.stack([row.env_origin_w_m for row in source])
    log.append("T.jsonl",{"kind":"T0","case":case,"seed":20260909,
        "initial_sensor_position_w_m":torch.stack([row.initial_sensor_position_w_m for row in source]),
        "initial_root_state_w":torch.cat((roots,torch.tensor([[1.,0,0,0]]).repeat(32,1),torch.zeros(32,6)),dim=1),
        "env_origins_w_m":origins})
    for step in range(100):
        log.append("T.jsonl",{"kind":"STEP","case":case,"step":step,
            "command_b":torch.stack([row.command_b[step] for row in source]),
            "actual_velocity_b":torch.stack([row.velocity_b[step] for row in source]),
            "sensor_positions_w_m":torch.stack([row.sensor_positions_w_m[step] for row in source]),
            "episode_ids":torch.ones(32,dtype=torch.int64),"collision":torch.zeros(32,dtype=torch.bool),
            "below_height":torch.zeros(32,dtype=torch.bool),"above_height":torch.zeros(32,dtype=torch.bool),
            "ray_clearance_m":torch.ones(32),"q":torch.zeros(32),"ics_beta":torch.ones(32)})
    replayed=replay_raw_case(str(tmp_path/"T.jsonl"),case=case)
    assert len(replayed)==32 and torch.equal(replayed[7].command_b,source[7].command_b)
    incomplete=tmp_path/"incomplete.jsonl"
    incomplete.write_text("\n".join((tmp_path/"T.jsonl").read_text().splitlines()[:-1])+"\n")
    with pytest.raises(ValueError,match="incomplete"): replay_raw_case(str(incomplete),case=case)


def test_authorization_exact_budget_order_and_checkpoint_bytes(tmp_path):
    from training_workloads.unified_h1.authorization import load_r3_authorization
    config=load_config(); identity=config["resolved_identity"]; attempt=(tmp_path/"attempt").resolve(); attempt.mkdir()
    def authorization(mode,order,parameters):
        return {"schema_version":"unified-h1-r3-authorization.v2","status":"AUTHORIZED","order_id":order,
            "method_sha256":METHOD_SHA256,"config_sha256":sha256_file(CONFIG_PATH),
            "product_commit":identity["product_commit"],"product_tree":identity["product_tree"],
            "workload_commit":identity["workload_commit"],"workload_tree":identity["workload_tree"],
            "attempt":str(attempt),"mode":mode,"budgets":required_mode_budget(mode,parameters),"parameters":parameters}
    path=(tmp_path/"auth.json").resolve(); path.write_text(json.dumps(authorization("main","UNIFIED-H1-R3",{})))
    assert load_r3_authorization(path,attempt,"main")["budgets"]["training_transitions"]==2048
    wrong=authorization("main","UNIFIED-H1-R4",{}); path.write_text(json.dumps(wrong))
    with pytest.raises(PermissionError,match="order"): load_r3_authorization(path,attempt,"main")
    wrong=authorization("main","UNIFIED-H1-R3",{}); wrong["budgets"]["training_transitions"]-=1
    path.write_text(json.dumps(wrong))
    with pytest.raises(PermissionError,match="budget"): load_r3_authorization(path,attempt,"main")
    checkpoints=[]
    for update in (448,512):
        checkpoint=(tmp_path/f"u{update}.pt").resolve(); checkpoint.write_bytes(f"checkpoint-{update}".encode())
        checkpoints.append({"path":str(checkpoint),"sha256":sha256_file(checkpoint),"update":update})
    parameters={"stage":"R5-512","checkpoints":checkpoints}
    path.write_text(json.dumps(authorization("evaluation","UNIFIED-H1-R5",parameters)))
    checkpoints[0]["path"] = checkpoints[0]["path"]
    Path(checkpoints[0]["path"]).write_bytes(b"replaced")
    with pytest.raises(PermissionError,match="changed"): load_r3_authorization(path,attempt,"evaluation")


def test_config_rejects_method_version_source_set_versions_and_candidate_identity(tmp_path):
    original=json.loads(CONFIG_PATH.read_text())
    mutations=[]
    value=deepcopy(original); value["method"]["version"]="unapproved-method"; mutations.append(value)
    value=deepcopy(original); value["product"]["source_sha256"]={}; mutations.append(value)
    value=deepcopy(original); value["versions"]["reward"]="reward-h1-v1"; mutations.append(value)
    value=deepcopy(original); value["product"]["candidate_commit"]="0"*40; mutations.append(value)
    for index,value in enumerate(mutations):
        path=(tmp_path/f"bad_{index}.json").resolve(); path.write_text(json.dumps(value))
        with pytest.raises(ValueError): load_config(path)


def test_live_adapter_dispatches_both_r3_modes_through_real_sequence(monkeypatch,tmp_path):
    import training_workloads.unified_h1.backend as backend_module
    from training_workloads.unified_h1.budget import required_mode_budget
    from training_workloads.unified_h1.live_adapter import run_main_sequence
    class Resource:
        h1_initialization_physics_ticks=1
        def close(self, *, wait_for_replicator=None, skip_cleanup=None): pass
    def factory(**kwargs):
        backend=_SequenceBackend(); backend.environment=Resource(); kwargs["resources"].update(environment=backend.environment,app=Resource())
        return backend
    monkeypatch.setattr(backend_module,"create_live_backend",factory)
    for mode, expected in (("main",0),("controlled-failure",1)):
        attempt=tmp_path/mode; (attempt/"results").mkdir(parents=True)
        auth={"mode":mode,"parameters":{},"budgets":required_mode_budget(mode,{}),"authorization_sha256":"a"*64}
        assert run_main_sequence(authorization=auth,attempt=attempt)==expected


def test_a0002_metric_counterexamples_fail_closed_and_w_proactive_stop_passes():
    t = _trace("T", 0)
    rejected = evaluate_episode(replace(t, command_b=torch.zeros_like(t.command_b)))
    assert not rejected["success"] and rejected["reason"] == "CASE_COMMAND_MISMATCH"
    w = _trace("W", 0)
    velocity = torch.zeros_like(w.velocity_b); velocity[:11,0]=.1; velocity[11:31,0]=.081
    position = w.sensor_positions_w_m.clone(); position[:,0]=4.6+torch.cumsum(velocity[:,0],0)*.1
    proactive = evaluate_episode(replace(w, velocity_b=velocity, sensor_positions_w_m=position,
        initial_sensor_position_w_m=w.initial_sensor_position_w_m))
    assert proactive["success"] and proactive["metrics"]["proactive_stop_outside_trigger"]
    incomplete = _summary(8, 1); incomplete["cases"] = {}
    assert not continuation_decision(baseline=_summary(7,2), current=incomplete,
        previous=_summary(9,1.1), u64=_summary(10,1))["extend"]


def test_diagnostic_smoke_profiles_drive_real_schedule_roles(tmp_path):
    from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
    backend=object.__new__(UnifiedH1LiveBackend); backend.output=tmp_path
    state=torch.zeros(32,52)
    class Env:
        device="cpu"
        def __init__(self): self._command_b=torch.zeros(32,3); self._policy={"state_vec":state.clone()}
        def set_h1_initial_command(self,command,step):
            self._command_b.copy_(command); self._policy["state_vec"][:,45:48].copy_(command)
    backend.environment=Env()
    backend.driver=type("Driver",(),{"observation":{"policy":{"state_vec":state,"lidar_grid":torch.zeros(32,1)},
        "critic":torch.zeros(32,21)}})()
    counts=[]
    class Loop:
        def run_update(self):
            actor=backend.driver.observation["policy"]["state_vec"][:,45:48]
            critic=backend.driver.observation["critic"][:,-3:]
            control=backend.environment._policy["state_vec"][:,45:48]
            assert torch.equal(actor,critic) and torch.equal(actor,control) and torch.equal(actor,backend.environment._command_b)
            counts.append(int((torch.linalg.vector_norm(actor,dim=-1)>0).sum()))
            for _ in range(16): backend.driver.clock.complete_step()
    backend.loop=Loop(); backend.counters=ExecutionCounters(); backend.trainer=object()
    for profile in ("zero","mixed","role-swap-a","role-swap-b"): backend.smoke_update(profile)
    assert counts == [0, 16, 16, 16]
    rows=[json.loads((tmp_path/f"smoke_profiles/{name}.json").read_text()) for name in ("zero","mixed","role-swap-a","role-swap-b")]
    assert [row["diagnostic_clock_start"] for row in rows] == [0,1024,1152,1280]
    assert all(row["production_curriculum"] is False for row in rows)


def test_live_construction_and_independent_bounded_cleanup_are_one_boundary(tmp_path):
    import threading
    from training_workloads.unified_h1.budget import required_mode_budget
    from training_workloads.unified_h1.live_adapter import run_main_sequence
    auth={"mode":"main","parameters":{},"budgets":required_mode_budget("main",{}),"authorization_sha256":"b"*64}
    calls=[]
    class App:
        def close(self, *, wait_for_replicator, skip_cleanup):
            assert (wait_for_replicator, skip_cleanup) == (False, True)
            calls.append("app")
    attempt=tmp_path/"constructor"; (attempt/"results").mkdir(parents=True)
    def constructor_failure(**kwargs):
        kwargs["resources"]["app"]=App(); raise RuntimeError("environment constructor failed")
    assert run_main_sequence(authorization=auth,attempt=attempt,backend_factory=constructor_failure)==1
    assert calls==["app"]
    assert json.loads((attempt/"results/main/status.json").read_text())["primary_error"]["message"]=="environment constructor failed"

    calls.clear(); attempt=tmp_path/"cleanup"; (attempt/"results").mkdir(parents=True)
    class Environment:
        h1_initialization_physics_ticks=1
        def close(self): calls.append("environment"); raise OSError("env close failed")
    def cleanup_failure(**kwargs):
        backend=_SequenceBackend(); backend.environment=Environment()
        kwargs["resources"].update(environment=backend.environment,app=App())
        return backend
    assert run_main_sequence(authorization=auth,attempt=attempt,backend_factory=cleanup_failure)==1
    assert calls==["environment","app"]

    calls.clear(); attempt=tmp_path/"hung"; (attempt/"results").mkdir(parents=True)
    blocker=threading.Event()
    class Hung(Environment):
        def close(self): calls.append("environment"); blocker.wait(10)
    def hung(**kwargs):
        backend=_SequenceBackend(); backend.environment=Hung()
        kwargs["resources"].update(environment=backend.environment,app=App())
        return backend
    assert run_main_sequence(authorization=auth,attempt=attempt,backend_factory=hung,cleanup_timeout_seconds=.01)==1
    assert calls==["environment","app"]


def test_initialization_tick_budget_and_repeated_start_fail_closed(tmp_path):
    from types import SimpleNamespace
    from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
    from training_workloads.unified_h1.live_adapter import run_main_sequence
    auth={"mode":"main","parameters":{},"budgets":required_mode_budget("main",{}),"authorization_sha256":"c"*64}
    factory_calls=[]
    class Resource:
        h1_initialization_physics_ticks=11
        h1_initialization_tick_source="cpu-boundary-counter"
        def close(self, *, wait_for_replicator=None, skip_cleanup=None): pass
    def factory(**kwargs):
        factory_calls.append(1); environment=Resource(); app=Resource()
        kwargs["resources"].update(environment=environment,app=app)
        return UnifiedH1LiveBackend(authorization=kwargs["authorization"],attempt=kwargs["attempt"],
            app=app,environment=environment,resource_guard=SimpleNamespace(check=lambda **unused:None),
            counters=kwargs["counters"])
    attempt=(tmp_path/"ticks"); (attempt/"results").mkdir(parents=True)
    assert run_main_sequence(authorization=auth,attempt=attempt,backend_factory=factory)==1
    status=json.loads((attempt/"results/main/status.json").read_text())
    assert status["counters"]["initialization_physics_ticks"]==0
    assert status["primary_error"]["message"]=="authorized initialization-tick budget exhausted"
    assert status["initialization_measurement"] == {
        "status":"OBSERVED", "counter_source":"cpu-boundary-counter",
        "observed_physics_ticks":11,
        "measurement_boundary":"after_environment_construction_before_explicit_reset_or_policy_step",
        "reserved_simulator_initializations":1,
    }
    receipt=json.loads((attempt/"results/main/initialization_accounting.json").read_text())
    assert receipt["observed_physics_ticks"]==11 and receipt["maximum_physics_ticks"]==10
    with pytest.raises(ValueError,match="already exists"):
        run_main_sequence(authorization=auth,attempt=attempt,backend_factory=factory)
    assert len(factory_calls)==1
