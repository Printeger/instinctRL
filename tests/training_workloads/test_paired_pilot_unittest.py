from __future__ import annotations
import ast
import json
from pathlib import Path
import unittest
from types import ModuleType, SimpleNamespace as NS
from contextlib import nullcontext
from unittest.mock import patch
import importlib
import random
import sys
import tempfile

from training_workloads.unified_h1.budget import required_mode_budget
from training_workloads.unified_h1.pilot_contract import CHECKPOINTS,beta_kl_scalar,frozen_copy,pilot_actor_learning_rate,probability_distribution_gate,probability_gate,validate_shapes

ROOT=Path(__file__).resolve().parents[2]
class PairedPilotTests(unittest.TestCase):
    def test_distribution_only_mismatch_is_zero_state_change(self):
        class Tensor:
            def __init__(self, values, shape): self.values=list(values); self.shape=tuple(shape)
            @property
            def ndim(self): return len(self.shape)
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return Tensor(self.values,self.shape)
            def reshape(self,*shape): return Tensor(self.values,shape)
            def tolist(self): return list(self.values)
        class Truth:
            def all(self): return self
            def item(self): return True
        fake_torch=ModuleType("torch")
        fake_torch.Tensor=Tensor; fake_torch.float32="float32"; fake_torch.uint8="uint8"
        fake_torch.optim=NS(Optimizer=type("Optimizer",(),{}))
        fake_torch.cuda=NS(is_available=lambda:False)
        fake_torch.isfinite=lambda value:Truth()
        fake_torch.no_grad=lambda:nullcontext()
        fake_torch.get_rng_state=lambda:Tensor([20260908],(1,))
        fake_torch.equal=lambda left,right:left.shape==right.shape and left.values==right.values

        class Model:
            def __init__(self, state): self.state=state
            def state_dict(self): return frozen_copy(self.state)
        class Actor(Model):
            def __init__(self): super().__init__({"weight":[2.0]})
            def __call__(self, policy):
                return NS(alpha=Tensor([2.0]*(512*4),(512,4)),beta=Tensor([3.0]*(512*4),(512,4)))
            def log_probability_from_output(self, output, actions): return Tensor([-0.75]*512,(512,1))
        class Optimizer:
            def __init__(self): self.state={"state":{"step":7,"momentum":[0.125]}}; self.step_calls=0
            def state_dict(self): return frozen_copy(self.state)
            def step(self): self.step_calls+=1
        class PPOTrainer:
            update_counter=0
            def __init__(self):
                self.actor=Actor(); self.critic=Model({"weight":[1.0]})
                self.actor_optimizer=Optimizer(); self.critic_optimizer=Optimizer(); self.update_calls=0
            def update(self, batch):
                self.update_calls+=1; self.actor_optimizer.step(); self.critic_optimizer.step()

        policy={"history":Tensor([0.0]*512,(512,1))}
        common=dict(policy=policy,privileged=Tensor([0.0]*512,(512,1)),actions=Tensor([0.5]*(512*4),(512,4)),
                    old_log_probabilities=Tensor([-0.75]*512,(512,1)),advantages=Tensor([1.0]*512,(512,1)),returns=Tensor([1.0]*512,(512,1)))
        with patch.dict(sys.modules,{"torch":fake_torch}):
            diagnostic_module=importlib.reload(importlib.import_module("training_workloads.unified_h1.pilot_diagnostic"))
            for field,index in (("old_alpha",731),("old_beta",997)):
                trainer=PPOTrainer(); alpha=[2.0]*(512*4); beta=[3.0]*(512*4)
                if field=="old_alpha": alpha[index]=2.25
                else: beta[index]=3.25
                batch=NS(**common,old_alpha=Tensor(alpha,(512,4)),old_beta=Tensor(beta,(512,4)))
                before={"actor":trainer.actor.state_dict(),"critic":trainer.critic.state_dict(),
                        "actor_optimizer":trainer.actor_optimizer.state_dict(),"critic_optimizer":trainer.critic_optimizer.state_dict(),
                        "python_rng":random.getstate(),"numpy_rng":diagnostic_module.np.random.get_state()}
                with tempfile.TemporaryDirectory() as directory:
                    gate=diagnostic_module.FirstUpdateDiagnostic(Path(directory),enabled=True)
                    with self.assertRaisesRegex(RuntimeError,"pre-update probability/distribution gate failed"):
                        gate.before_update(trainer,batch)
                self.assertEqual(trainer.actor.state_dict(),before["actor"])
                self.assertEqual(trainer.critic.state_dict(),before["critic"])
                self.assertEqual(trainer.actor_optimizer.state_dict(),before["actor_optimizer"])
                self.assertEqual(trainer.critic_optimizer.state_dict(),before["critic_optimizer"])
                self.assertEqual(random.getstate(),before["python_rng"])
                after_numpy=diagnostic_module.np.random.get_state()
                self.assertEqual(after_numpy[0],before["numpy_rng"][0]); self.assertTrue(diagnostic_module.np.array_equal(after_numpy[1],before["numpy_rng"][1])); self.assertEqual(after_numpy[2:],before["numpy_rng"][2:])
                self.assertEqual((trainer.update_calls,trainer.actor_optimizer.step_calls,trainer.critic_optimizer.step_calls),(0,0,0))
        diagnostic=(ROOT/"training_workloads/unified_h1/pilot_diagnostic.py").read_text(); driver=(ROOT/"training_workloads/unified_h1/driver.py").read_text()
        self.assertIn("captured.append",diagnostic); self.assertIn("bind_rollout_distribution",driver)
        self.assertNotIn('object.__setattr__(batch,"old_alpha",_clone(_finite',diagnostic)

    def test_disabled_diagnostic_learning_equivalence(self):
        fake_torch=ModuleType("torch")
        fake_torch.Tensor=type("Tensor",(),{}); fake_torch.float32="float32"; fake_torch.optim=NS(Optimizer=type("Optimizer",(),{})); fake_torch.cuda=NS(is_available=lambda:False)
        with patch.dict(sys.modules,{"torch":fake_torch}):
            driver_module=importlib.reload(importlib.import_module("training_workloads.unified_h1.driver"))
            diagnostic_module=importlib.reload(importlib.import_module("training_workloads.unified_h1.pilot_diagnostic"))
            budget_module=importlib.import_module("training_workloads.unified_h1.budget")
            def execute(diagnostic):
                rng=random.Random(917); model={"weight":2.0}; optimizer_state={"step":0}
                class Optimizer:
                    def state_dict(self): return {"state":{}}
                class Trainer:
                    update_counter=0; actor=NS(distribution=lambda policy:None); actor_optimizer=Optimizer(); critic_optimizer=Optimizer()
                    def update(self,batch):
                        model["weight"]-=.01*(model["weight"]-rng.random()); optimizer_state["step"]+=1; self.update_counter+=1
                        return {"input":list(batch.value)}
                trainer=Trainer(); batch=NS(value=[4.0]); rollout_module=ModuleType("instinctlab.training.rollout_h1"); rollout_module.collect_h1_rollout=lambda *args:NS(segment=NS(batch=batch))
                config_module=ModuleType("instinctlab.training.rollout"); config_module.RolloutConfig=lambda *args:args
                noop=lambda *args,**kwargs:None
                journal=NS(durable_step_rows=0,record_update_attempt=noop,record_rollout=lambda *args,**kwargs:{},record_update=noop,record_partial_episodes=noop,record_partial_rollout_boundary=noop,record_partial_failure=noop)
                loop_driver=NS(counters=budget_module.ExecutionCounters(),observation={},take_evidence=lambda:(),evidence_buffer=[],journal=None)
                with patch.dict(sys.modules,{"instinctlab.training.rollout_h1":rollout_module,"instinctlab.training.rollout":config_module}): result=driver_module.UnifiedH1TrainingLoop(loop_driver,trainer,journal,diagnostic=diagnostic,update_limit=1).run_update()
                return model,optimizer_state,rng.getstate(),result
            baseline=execute(None)
            with tempfile.TemporaryDirectory() as directory: disabled=execute(diagnostic_module.FirstUpdateDiagnostic(Path(directory),enabled=False))
            self.assertEqual(baseline,disabled)

    def test_evaluation_loader_binds_each_group_learning_rate(self):
        self.assertEqual(pilot_actor_learning_rate("P0"),3e-4); self.assertEqual(pilot_actor_learning_rate("PLR"),1e-4)
        with self.assertRaises(ValueError): pilot_actor_learning_rate("OTHER")
        backend=(ROOT/"training_workloads/unified_h1/backend.py").read_text()
        self.assertIn("actor_learning_rate=pilot_actor_learning_rate(group)",backend)
        self.assertIn("PPOConfig(minibatch_size=64, epoch_count=1, actor_learning_rate=actor_learning_rate",backend)

    def test_no_update_probability_consistency_and_mismatch(self):
        values=[0.125]*512; state={"actor":[1,2],"critic":[3],"optimizers":{"actor":{},"critic":{}}}; before=frozen_copy(state)
        probability_gate(values,list(values)); self.assertEqual(state,before)
        changed=list(values); changed[91]=0.25
        with self.assertRaisesRegex(RuntimeError,"index 91"): probability_gate(values,changed)
        self.assertEqual(state,before)
    def test_action_index_mismatch_and_broadcasting(self):
        valid=dict(policy_rows={"history":512,"state":512},privileged_rows=512,action_shape=(512,4),old_logp_shape=(512,1),advantage_shape=(512,1),return_shape=(512,1))
        validate_shapes(**valid)
        for field,value in (("action_shape",(512,1)),("old_logp_shape",(512,)),("privileged_rows",511)):
            bad=dict(valid); bad[field]=value
            with self.assertRaises(ValueError): validate_shapes(**bad)
        with self.assertRaises(ValueError): probability_gate([0.0]*511,[0.0]*511)
    def test_history_snapshot_does_not_mutate(self):
        history={"state":[[1.0],[2.0]],"nested":{"old_beta":[3.0]}}; frozen=frozen_copy(history); history["state"][0][0]=99.0; history["nested"]["old_beta"][0]=8.0
        self.assertEqual(frozen,{"state":[[1.0],[2.0]],"nested":{"old_beta":[3.0]}})
    def test_identical_distribution_analytic_kl(self):
        self.assertAlmostEqual(beta_kl_scalar(2.0,3.0,2.0,3.0),0.0,places=12)
    def test_actual_update_path_and_required_probe_fields_are_wired(self):
        driver=(ROOT/"training_workloads/unified_h1/driver.py").read_text(); diagnostic=(ROOT/"training_workloads/unified_h1/pilot_diagnostic.py").read_text()
        tree=ast.parse(driver); self.assertIn("diagnostic.before_update",driver); self.assertIn("diagnostic.after_update",driver); self.assertIn("diagnostic.retain_rollout",driver)
        for field in ("analytic_beta_kl","legacy_ppo_approximate_kl","action_mean","action_std","xyz_correction","clipped_gradients","unclipped_gradients","clipped_gradient_norm","unclipped_gradient_norm","parameter_deltas","parameter_delta_after","advantage","return","value"):
            self.assertIn(field,diagnostic)
        self.assertIn("register_step_pre_hook",diagnostic); self.assertIn("register_step_post_hook",diagnostic)
    def test_diagnostic_off_is_explicit_noop(self):
        source=(ROOT/"training_workloads/unified_h1/pilot_diagnostic.py").read_text()
        self.assertGreaterEqual(source.count("if not self.enabled"),4)
    def test_configs_differ_only_actor_lr_and_exact_caps(self):
        p0=json.loads((ROOT/"training_workloads/unified_h1/pilot_p0.json").read_text()); plr=json.loads((ROOT/"training_workloads/unified_h1/pilot_plr.json").read_text())
        self.assertEqual({key for key in p0 if p0[key]!=plr[key]},{"actor_learning_rate"}); self.assertEqual(tuple(p0["checkpoints"]),CHECKPOINTS); self.assertEqual(p0["development_evaluation"],{"case":"Z","updates":[16,64]})
        train=required_mode_budget("pilot-training",{"group":"P0"}); evaluation=required_mode_budget("pilot-evaluation",{"group":"PLR"}); self.assertEqual((train["training_transitions"],train["optimizer_updates"],train["checkpoint_writes"]),(32768,64,7)); self.assertEqual(evaluation["nontraining_transitions"],12800)
        proposal=json.loads((ROOT/"docs/history/consolidation-20261012/r5/r5_delivery/R5_PAIRED_PILOT_TRAINING_PROPOSAL.json").read_text()); self.assertEqual(proposal["symmetric_hard_caps"]["automatic_retries"],0); self.assertEqual(proposal["symmetric_hard_caps"]["actor_minibatch_steps"],1024)
        evaluation_proposal=json.loads((ROOT/"docs/history/consolidation-20261012/r5/r5_delivery/R5_PAIRED_PILOT_EVALUATION_PROPOSAL.json").read_text()); self.assertEqual(evaluation_proposal["symmetric_hard_caps"]["disk_bytes"],proposal["symmetric_hard_caps"]["disk_bytes"])
        review=(ROOT/"docs/history/consolidation-20261012/r5/r5_delivery/R5_PILOT_FINAL_REVIEW.md").read_text(); self.assertIn("evaluation_disk_bytes_settled: NOT_STARTED",review)
if __name__=="__main__": unittest.main()
