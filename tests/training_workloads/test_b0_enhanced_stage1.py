from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from training_workloads.unified_h1.b0_enhanced import *
from training_workloads.unified_h1.b0_handoff import _candidate_diff, validate_handoff
from training_workloads.unified_h1.b0_causal_analyzer import derive_mechanics, discriminate_signatures
from training_workloads.unified_h1.b0_observer import EnhancedB0ExecutionAccounting, FormalEnhancedB0EnvironmentAdapter, ReadOnlyObserver, attach_supported_post_physics, PROTECTED_STATE
from training_workloads.unified_h1.budget import ExecutionCounters, required_mode_budget


def row():
    body={"name":"base","mass_kg":2.0,"local_com_m":[0.5,0,0],"local_com_quaternion_wxyz":[1,0,0,0],"inertia_at_com_body_kg_m2":[[1,0,0],[0,2,0],[0,0,3]],"world_position_m":[1,2,3],"world_quaternion_wxyz":[1,0,0,0],"world_linear_velocity_m_s":[0,0,0],"world_angular_velocity_rad_s":[0,0,0]}
    rotor={"name":"r0","command":0.5,"target_throttle":0.5,"actual_throttle":0.5,"thrust_n":2.0,"reaction_moment_nm":0.1,"application_point_world_m":[1.5,3,3],"direction_world":[0,0,1],"reaction_body":"base","reaction_direction_world":[0,0,1]}
    available=lambda f,t:{"availability":"AVAILABLE","force_n":f,"torque_at_com_nm":t}
    return OrderedDict((
      ("schema_version",SCHEMA),("experiment_id",EXPERIMENT_ID),("case","Z"),("policy_step",0),("physics_step",0),("lane",0),("episode_id",0),
      ("reset",{"mask":False,"reason":"","initial_boundary":False,"last_pre_reset_state":None,"post_reset_states":[]}),
      ("root_state",{"pose":[1,2,3,1,0,0,0],"linear_velocity":[0,0,0],"angular_velocity":[0,0,0]}),
      ("system",{"total_mass_kg":2.0,"combined_com_world_m":[1.5,2,3],"linear_momentum_kg_m_s":[0,0,0],"angular_momentum_at_com_kg_m2_s":[0,0,0]}),
      ("bodies",[body]),("controller",{"target_velocity_b_m_s":[0,0,0],"velocity_error_b_m_s":[0,0,0],"desired_rotation_w":[[1,0,0],[0,1,0],[0,0,1]],"collective_n":2}),
      ("rotors",[rotor]),
      ("wrenches",OrderedDict((("gravity",available([0,0,-19.62],[0,0,0])),("rotor",available([0,0,2],[2,0,0.1])),("drag_aero",{"availability":UNAVAILABLE}),("contact",{"availability":UNAVAILABLE}),("joint_constraint",{"availability":UNAVAILABLE}),("other",{"availability":UNAVAILABLE})))),
      ("momentum",{"dt_s":0.01,"inferred_wrench":{"force_n":[0,0,0],"torque_at_com_nm":[0,0,0]}}),
      ("response",{"position_world_m":[1,2,3],"velocity_world_m_s":[0,0,0],"acceleration_world_m_s2":[0,0,0]})))

def initial_row(lane=0):
    r=row(); r["lane"]=lane
    snapshot={"controller":copy.deepcopy(r["controller"]),"rotors":copy.deepcopy(r["rotors"])}
    r["reset"]={"mask":True,"reason":"explicit_case_reset","initial_boundary":True,"last_pre_reset_state":None,"post_reset_states":[snapshot]}
    return r

class MechanicsTests(unittest.TestCase):
    def test_positive_com_rotor_and_exact_row(self):
        r=row(); self.assertIs(validate_row(r),r)
        self.assertEqual(validate_row(json.loads(json.dumps(r, sort_keys=False))), r)
        self.assertEqual(reconstruct_com(r["bodies"])["combined_com_world_m"],(1.5,2.0,3.0))
        self.assertEqual(rotor_wrench(r["rotors"],[1.5,2,3])["torque_at_com_nm"],(2.0,0.0,0.1))
    def test_exact_403200_coordinate_coverage(self):
        self.assertEqual(PHYSICS_ROWS,403200)
        seen={TraceValidator.coordinate(i) for i in range(PHYSICS_ROWS)}
        self.assertEqual(len(seen),PHYSICS_ROWS)
        with self.assertRaises(EvidenceError): TraceValidator.coordinate(PHYSICS_ROWS)
    def test_adjacent_momentum(self):
        a={"linear_momentum_kg_m_s":[0,0,0],"angular_momentum_at_com_kg_m2_s":[0,0,0]}; b={"linear_momentum_kg_m_s":[1,2,3],"angular_momentum_at_com_kg_m2_s":[.1,.2,.3]}
        self.assertEqual(infer_momentum_wrench(a,b,.5)["force_n"],(2.,4.,6.))
    def test_analyzer_independently_derives_primitives(self):
        r=row(); d=derive_mechanics(None,r)
        self.assertEqual(d["total_mass_kg"],2.0); self.assertEqual(d["combined_com_world_m"],(1.5,2.0,3.0))
        self.assertEqual(d["rotor_arms_world_m"],[(0.0,1.0,0.0)])
        self.assertEqual(d["angular_momentum_at_com_kg_m2_s"],(0.0,0.0,0.0))
        bad=copy.deepcopy(r); bad["system"]["angular_momentum_at_com_kg_m2_s"]=[1,0,0]
        with self.assertRaises(EvidenceError): validate_row(bad)
        prior=copy.deepcopy(r); d=derive_mechanics(prior,r)
        self.assertEqual(d["momentum_wrench"]["force_n"],(0.0,0.0,0.0))
        self.assertIn("residual_force_n",d["known_residual_wrench"])
    def test_causal_signatures_are_distinct_and_unavailable_fails_closed(self):
        classes={name:[0.0,0.0,0.0] for name in WRENCH_CLASSES}
        fixed=discriminate_signatures(residual_forces=[(2.,0.,0.)]*3,residual_torques=[(0.,0.,0.)]*3,rotor_torques=[(0.,0.,0.)]*3,class_magnitudes=classes,unavailable=set(),reset_rows=0)
        self.assertEqual(fixed["fixed_force"],1.0); self.assertEqual(fixed["gravity_mass"],0.0)
        gravity=discriminate_signatures(residual_forces=[(0.,0.,2.)]*3,residual_torques=[(0.,0.,0.)]*3,rotor_torques=[(0.,0.,0.)]*3,class_magnitudes=classes,unavailable=set(),reset_rows=0)
        self.assertEqual(gravity["gravity_mass"],1.0); self.assertEqual(gravity["rotor_mixer_frame"],0.0)
        missing=discriminate_signatures(residual_forces=[(0.,0.,0.)]*3,residual_torques=[(0.,0.,0.)]*3,rotor_torques=[(0.,0.,0.)]*3,class_magnitudes=classes,unavailable={"drag_aero","contact"},reset_rows=0)
        self.assertIsNone(missing["drag_aero"]); self.assertIsNone(missing["contact_joint_constraint"])

class FrozenNegativeTests(unittest.TestCase):
    def reject(self,change):
        r=copy.deepcopy(row()); change(r)
        with self.assertRaises(EvidenceError): validate_row(r)
    def test_1_missing_required(self): self.reject(lambda r:r.pop("bodies"))
    def test_2_out_of_range_lane(self): self.reject(lambda r:r.__setitem__("lane",32))
    def test_3_nonfinite(self): self.reject(lambda r:r["bodies"][0].__setitem__("mass_kg",float("nan")))
    def test_4_bad_com(self): self.reject(lambda r:r["system"].__setitem__("combined_com_world_m",[0,0,0]))
    def test_5_bad_rotor_wrench(self): self.reject(lambda r:r["wrenches"]["rotor"].__setitem__("force_n",[0,0,0]))
    def test_6_zero_fill_without_availability(self): self.reject(lambda r:r["wrenches"].__setitem__("drag_aero",{"force_n":[0,0,0],"torque_at_com_nm":[0,0,0]}))
    def test_7_incomplete_reset_boundary(self):
        def change(r): r["reset"]={"mask":True,"reason":"below_height","initial_boundary":False,"last_pre_reset_state":{"controller":{},"rotors":[]},"post_reset_states":[]}
        self.reject(change)
    def test_initial_reset_accepts_only_absent_pre_cache_and_real_post_cache(self):
        r=row(); snapshot={"controller":copy.deepcopy(r["controller"]),"rotors":copy.deepcopy(r["rotors"])}
        r["reset"]={"mask":True,"reason":"explicit_case_reset","initial_boundary":True,"last_pre_reset_state":None,"post_reset_states":[snapshot]}
        validate_row(r)
        stale=copy.deepcopy(r);stale["reset"]["last_pre_reset_state"]=copy.deepcopy(snapshot)
        with self.assertRaisesRegex(EvidenceError,"truthfully omit"):validate_row(stale)
        wrong=copy.deepcopy(r);wrong["case"]="T"
        with self.assertRaisesRegex(EvidenceError,"truthfully omit"):validate_row(wrong)
    def test_initial_and_later_reset_mapping_and_reasons_are_exact(self):
        snapshot={"controller":copy.deepcopy(row()["controller"]),"rotors":copy.deepcopy(row()["rotors"])}
        for mutate in (
            lambda r:r.__setitem__("policy_step",17),
            lambda r:r.__setitem__("physics_step",9),
            lambda r:r["reset"].__setitem__("reason","arbitrary_wrong_reason"),
        ):
            r=row();r["reset"]={"mask":True,"reason":"explicit_case_reset","initial_boundary":True,"last_pre_reset_state":None,"post_reset_states":[copy.deepcopy(snapshot)]};mutate(r)
            with self.assertRaises(EvidenceError):validate_row(r)
        later=row();later["reset"]={"mask":True,"reason":"","initial_boundary":False,"last_pre_reset_state":copy.deepcopy(snapshot),"post_reset_states":[copy.deepcopy(snapshot)]}
        with self.assertRaisesRegex(EvidenceError,"exact reason"):validate_row(later)

    def test_reset_requires_real_post_reset_snapshot_without_zero_substitution(self):
        r=row(); snapshot={"controller":copy.deepcopy(r["controller"]),"rotors":copy.deepcopy(r["rotors"])}
        r["policy_step"]=1; r["episode_id"]=1
        r["reset"]={"mask":True,"reason":"automatic_environment_reset","initial_boundary":False,"last_pre_reset_state":copy.deepcopy(snapshot),"post_reset_states":[copy.deepcopy(snapshot)]}
        self.assertIs(validate_row(r),r)
        missing=copy.deepcopy(r); missing["reset"]["post_reset_states"]=[]
        with self.assertRaisesRegex(EvidenceError,"reset boundary lacks a real post-reset"):
            validate_row(missing)
        zero=copy.deepcopy(r); zero["reset"]["post_reset_states"][0]["rotors"][0]["direction_world"]=[0,0,0]
        with self.assertRaisesRegex(EvidenceError,"unit length"):
            validate_row(zero)
    def test_missing_initial_boundary_and_duplicate_reorder_fail_at_append(self):
        with self.assertRaisesRegex(EvidenceError,"truthful initial reset"):
            TraceValidator().append(row())
        v=TraceValidator(); v.append(initial_row())
        for transform in (lambda r:r,lambda r:r.__setitem__("lane",2),lambda r:r.__setitem__("physics_step",1)):
            r=copy.deepcopy(row()); transform(r)
            with self.assertRaises(EvidenceError): v.append(r)

    def test_mid_z_and_wrong_later_explicit_boundaries_are_rejected(self):
        snapshot={"controller":copy.deepcopy(row()["controller"]),"rotors":copy.deepcopy(row()["rotors"])}
        for case,policy,physics,episode in (("Z",17,9,3),("T",1,0,0),("S",0,1,0),("T",0,0,2)):
            r=row();r["case"]=case;r["policy_step"]=policy;r["physics_step"]=physics;r["episode_id"]=episode
            r["reset"]={"mask":True,"reason":"explicit_case_reset","initial_boundary":False,"last_pre_reset_state":copy.deepcopy(snapshot),"post_reset_states":[copy.deepcopy(snapshot)]}
            with self.assertRaisesRegex(EvidenceError,"later explicit reset"):
                validate_row(r)

class IntegrationSourceTests(unittest.TestCase):
    def test_formal_backend_dispatch_and_post_physics_boundary_are_wired(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[2] / "training_workloads" / "unified_h1"
        backend=(root / "backend.py").read_text(); adapter=(root / "live_adapter.py").read_text()
        self.assertIn("def run_enhanced_b0_diagnostic",backend)
        self.assertIn("attach_supported_post_physics(formal_environment, observer)",backend)
        self.assertIn("formal_environment.run_frozen_enhanced_b0_zts(seed=20260909",backend)
        self.assertIn("json.dumps(row, sort_keys=False",backend)
        self.assertIn("backend.run_enhanced_b0_diagnostic(parameters)",adapter)

class ObserverTests(unittest.TestCase):
    def test_supported_hook_and_no_write(self):
        state=OrderedDict((x,{"digest":x}) for x in PROTECTED_STATE); written=[]
        observer=ReadOnlyObserver(written.append,lambda:state)
        class Env:
            def register_read_only_post_physics_observer(self,callback,**kw):
                self.callback=callback
                return {"boundary":"after_physics_before_reset","ordering_unchanged":True,"read_only":True,"exactly_once":True,"lanes":32}
        env=Env(); attach_supported_post_physics(env,observer); env.callback([row() for _ in range(32)])
        self.assertEqual(observer.calls,1); self.assertEqual(observer.rows,32); self.assertEqual(len(written),32)
    def test_mutation_fails(self):
        state=OrderedDict((x,{"digest":x}) for x in PROTECTED_STATE)
        def writer(r): state["control"]["digest"]="changed"
        with self.assertRaises(RuntimeError): ReadOnlyObserver(writer,lambda:state)([row() for _ in range(32)])

    def test_production_adapter_exact_batches_and_order(self):
        state=OrderedDict((x,{"digest":x}) for x in PROTECTED_STATE); events=[]; holder={}; rows=0
        def register(callback,**binding):
            holder["callback"]=callback
            return {"boundary":"after_physics_before_reset","ordering_unchanged":True,"read_only":True,"exactly_once":True,"lanes":32}
        def run(**parameters):
            for _ in range(12600):
                events.extend(("control","physics")); holder["callback"]([None]*32); events.extend(("reset","observation"))
            return {"physics_callback_batches":12600,"physics_rows":403200,"control_physics_reset_observation_order":"UNCHANGED"}
        class Env:
            def enhanced_b0_read_only_protocol(self): return {"identity":"post-physics-before-reset/v1","register":register,"state_probe":lambda:state,"accounting":lambda value:holder.setdefault("accounting",value),"run":run}
        adapter=FormalEnhancedB0EnvironmentAdapter(Env())
        adapter.configure_execution_accounting(EnhancedB0ExecutionAccounting(ExecutionCounters(), {"explicit_resets":3,"automatic_resets":20160,"nontraining_transitions":20160}, lambda *args:None))
        def observe(batch):
            nonlocal rows
            self.assertEqual(len(batch),32); events.append("observer"); rows+=len(batch)
        attach_supported_post_physics(adapter,observe)
        adapter.run_frozen_enhanced_b0_zts(seed=20260909,cases=("Z","T","S"),target_velocity_b_m_s=(0.0,0.0,0.0),retries=0)
        self.assertEqual(rows,403200); self.assertEqual(len(events),63000)
        self.assertEqual(events[:5],["control","physics","observer","reset","observation"])

class ResetAndBudgetRegressionTests(unittest.TestCase):
    def test_all_false_reset_trace_fails_closed(self):
        validator=TraceValidator(); validator.count=PHYSICS_ROWS
        with self.assertRaisesRegex(EvidenceError,"lacks observed reset"):
            validator.finalize()
    def test_reset_row_advances_exact_lane_episode(self):
        first=initial_row(); snapshot={"controller":copy.deepcopy(first["controller"]),"rotors":copy.deepcopy(first["rotors"])}
        validator=TraceValidator(); validator.append(first)
        second=initial_row(1)
        validator.append(second)
        self.assertEqual(validator.episodes[("Z",0)],1)
    def test_exact_accounting_and_every_ceiling_fails_before_action(self):
        budgets={"explicit_resets":3,"automatic_resets":20160,"nontraining_transitions":20160}
        counters=ExecutionCounters(); stages=[]; accounting=EnhancedB0ExecutionAccounting(counters,budgets,lambda stage,ordinal:stages.append((stage,ordinal)))
        for case in ("Z","T","S"): accounting.before_reset(case); accounting.after_reset(case)
        for step in range(630): accounting.before_step("Z",step); accounting.after_step("Z",step,0)
        self.assertEqual((counters.explicit_resets,counters.nontraining_transitions,counters.training_transitions,counters.real_optimizer_updates,counters.real_checkpoint_writes),(3,20160,0,0,0))
        with self.assertRaises(RuntimeError): accounting.before_reset("Z")
        attempted=counters.attempted_policy_steps
        with self.assertRaises(RuntimeError): accounting.before_step("Z",630)
        self.assertEqual(counters.attempted_policy_steps,attempted)
        counters.automatic_resets=20160; counters.nontraining_transitions=0
        with self.assertRaises(RuntimeError): accounting.before_step("Z",0)

class WallMarginContractTests(unittest.TestCase):
    def test_enhanced_workload_900_and_real_rlflow_reservation_910(self):
        from unittest import mock
        from rlflow import h1

        root=Path(__file__).resolve().parents[2]
        config_path=root/"training_workloads/unified_h1/candidate_config.json"
        config=json.loads(config_path.read_text())
        budget=required_mode_budget("enhanced-b0-diagnostic", {"handoff":"HANDOFF.json"})
        self.assertEqual(budget["wall_seconds"],900)
        with tempfile.TemporaryDirectory() as temporary:
            attempt=Path(temporary)/"attempt"; (attempt/"results").mkdir(parents=True)
            executable=config["execution"]["python_path"]
            authorization=str(attempt/"commands/AUTHORIZATION.json")
            argv=[executable,"-B","-m","training_workloads.unified_h1.formal_unified_h1","--authorization",authorization,"--attempt",str(attempt),"--mode","enhanced-b0-diagnostic"]
            product_entry=Path(config["product"]["path"])/"isaaclab-training/instinctlab/envs/unified_h1_direct_env.py"
            spec={"h1":{"workload_root":str(root),"attempt":str(attempt),"mode":"enhanced-b0-diagnostic","config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),"gpu_devices":["0"],"gpu_accounting":"allocated_device_wall_seconds"},"cwd":str(root),"argv":argv,"pins":{"workload":{"path":str(root/"training_workloads/unified_h1/formal_unified_h1.py")},"dispatcher":{"path":str(root/"training_workloads/unified_h1/local_dispatch.py")},"executable":{"path":executable},"authorization":{"path":authorization},"product":{"path":str(product_entry)},"runtime":{"path":executable}},"resources":{"starts":1,"wall_seconds":910,"gpu_seconds":910,"transitions":20160},"seed":20260909,"method":config["method"]["version"],"evaluation":config["versions"]["metric"],"runtime":config["execution"]["frozen_runtime_root"],"observations":{"status":{},"counters":{},"process_identity":{},"process_start":{}}}
            observed={"config":config,"authorization":{"budgets":budget}}
            with mock.patch.object(h1,"digest",return_value=spec["h1"]["config_sha256"]), mock.patch.object(h1,"inspect_authority",return_value=observed):
                self.assertIs(h1.validate(spec,Path(temporary)),observed)
                for mutate in (
                    lambda value:value["argv"].remove("-B"),
                    lambda value:value.__setitem__("seed",20260908),
                    lambda value:value.__setitem__("resources",{**value["resources"],"wall_seconds":920,"gpu_seconds":920}),
                    lambda value:value["argv"].__setitem__(-1,"main"),
                ):
                    invalid=copy.deepcopy(spec); mutate(invalid)
                    with self.assertRaises(h1.Conflict): h1.validate(invalid,Path(temporary))

                evaluation_modes={"evaluation","control-diagnostic","enhanced-b0-diagnostic"}
                all_modes=("main","controlled-failure","r4-training","evaluation","r5-resume","control-diagnostic","pilot-training","pilot-evaluation","enhanced-b0-diagnostic")
                for mode in all_modes:
                    variant=copy.deepcopy(spec); variant["h1"]["mode"]=mode; variant["argv"][-1]=mode
                    variant["seed"]=20260909 if mode in evaluation_modes else 20260908
                    self.assertIs(h1.validate(variant,Path(temporary)),observed)

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.root=Path(self.temporary.name)
        self.repo=Path(__file__).resolve().parents[2]
        self.commit=subprocess.run(["git","rev-parse","HEAD"],cwd=self.repo,check=True,capture_output=True,text=True).stdout.strip()
        self.tree=subprocess.run(["git","rev-parse","HEAD^{tree}"],cwd=self.repo,check=True,capture_output=True,text=True).stdout.strip()
        self.checkpoint=self.root/"checkpoint.pt"; self.checkpoint.write_bytes(b"checkpoint")
        self.runtime=self.root/"python"; self.runtime.write_bytes(b"runtime")
        self.assets=self.root/"assets.json"; self.assets.write_bytes(b"assets")
    def tearDown(self): self.temporary.cleanup()
    def digest(self,path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def handoff(self):
        diff=_candidate_diff(self.repo,self.commit)
        source={"product_repository":str(self.repo),"product_commit":self.commit,"product_tree":self.tree,"workload_repository":str(self.repo),"workload_base_commit":self.commit,"workload_base_tree":self.tree,"runtime_python":str(self.runtime),"runtime_python_sha256":self.digest(self.runtime),"asset_manifest":str(self.assets),"asset_manifest_sha256":self.digest(self.assets)}
        auth=str(self.root/"attempt"/"commands"/"AUTHORIZATION.json"); attempt=str(self.root/"attempt")
        return OrderedDict((
          ("schema_version","instinctrl-enhanced-b0-handoff/2"),("status","VALIDATE_ONLY_NOT_AUTHORIZED"),("experiment_id",EXPERIMENT_ID),
          ("argv",["/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11","-B","-m","training_workloads.unified_h1.formal_unified_h1","--authorization",auth,"--attempt",attempt,"--mode","enhanced-b0-diagnostic"]),
          ("authorization",auth),("attempt",attempt),("checkpoint",None),("source_identity",source),
          ("candidate",{"repository":str(self.repo),"base_commit":self.commit,"commit":self.commit,"diff_sha256":hashlib.sha256(diff).hexdigest()}),
          ("frozen",{"seed":20260909,"cases":["Z","T","S"],"case_policy_steps":{"Z":200,"T":100,"S":330},"lanes":32,"decimation":20,"target_velocity_b_m_s":[0.,0.,0.],"initial_state":"frozen-R4-B0","curriculum":"frozen-Z/T/S","control_frequency_hz":200,"policy_frequency_hz":10,"safety_standards":"unchanged-R4-B0"}),
          ("budgets",{"live_process_starts":1,"simulator_initializations":1,"nontraining_transitions":20160,"physics_rows":403200,"training_transitions":0,"optimizer_updates":0,"checkpoint_writes":0,"p0_plr":0,"policy_evaluations":0,"automatic_retries":0,"wall_seconds":910,"gpu_allocation_seconds":910,"diagnostic_bytes":17179869184}),
          ("output_manifest",OrderedDict((("raw_physics_jsonl","enhanced_b0_physics.jsonl"),("raw_manifest","enhanced_b0_manifest.json"),("causal_analysis","enhanced_b0_causal_analysis.json"),("measurement_failure","enhanced_b0_measurement_failure.json"),("budget_ledger","budget_ledger.json"),("status","status.json"),("cleanup_evidence","cleanup_evidence.json")))),
          ("cleanup",{"environment_close":True,"app_close":True,"no_retry_on_failure":True})))
    def test_validate_only_exact(self): self.assertFalse(validate_handoff(self.handoff())["live_authorized"])
    def test_no_retry_and_identity_fail_closed(self):
        edits=(lambda h:h["budgets"].__setitem__("automatic_retries",1),lambda h:h["candidate"].__setitem__("diff_sha256","0"*64),lambda h:h["argv"].append("--retry"),lambda h:h["argv"].remove("-B"),lambda h:h["argv"].__setitem__(-1,"main"))
        for edit in edits:
            h=self.handoff(); edit(h)
            with self.assertRaises(PermissionError): validate_handoff(h)
    def test_provided_predecessor_candidate_hash_is_rejected_after_rework(self):
        h=self.handoff(); h["candidate"]["diff_sha256"]="f223be389f2b0a835cf03d8892a171b9ea3376cf2accbb28a9b0da3011c7d5d6"
        with self.assertRaises(PermissionError): validate_handoff(h)
    def test_zero_target_has_no_policy_checkpoint_and_uses_actual_frequencies(self):
        h=self.handoff();h["checkpoint"]={"path":str(self.checkpoint)}
        with self.assertRaisesRegex(PermissionError,"must not bind/load"):validate_handoff(h)
        h=self.handoff();h["frozen"]["control_frequency_hz"]=100
        with self.assertRaisesRegex(PermissionError,"parameters changed"):validate_handoff(h)

    def test_runtime_asset_and_candidate_bytes_are_recomputed(self):
        for target in ("runtime","assets","candidate"):
            h=self.handoff()
            if target=="runtime": self.runtime.write_bytes(b"changed")
            elif target=="assets": self.assets.write_bytes(b"changed")
            else: h["candidate"]["diff_sha256"]="0"*64
            with self.assertRaises(PermissionError): validate_handoff(h)
            if target != "candidate":
                if target=="runtime": self.runtime.write_bytes(b"runtime")
                else: self.assets.write_bytes(b"assets")

if __name__ == "__main__": unittest.main()
