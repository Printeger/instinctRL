"""CPU-only contract tests for the production enhanced-B0 observation hook."""
from __future__ import annotations
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"isaaclab-training/instinctlab/envs/unified_h1_direct_env.py"
class ProductHookSourceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.text=SOURCE.read_text(); cls.tree=ast.parse(cls.text)
  cls.methods={node.name:node for node in ast.walk(cls.tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
 def test_exact_supported_protocol_is_product_owned(self):
  for name in ("enhanced_b0_read_only_protocol","register_read_only_post_physics_observer","configure_enhanced_b0_execution_accounting","read_only_diagnostic_state_fingerprints","run_frozen_enhanced_b0_zts","_enhanced_b0_physics_rows"): self.assertIn(name,self.methods)
  protocol=ast.get_source_segment(self.text,self.methods["enhanced_b0_read_only_protocol"])
  for token in ('"post-physics-before-reset/v1"','"register"','"state_probe"','"accounting"','"run"'): self.assertIn(token,protocol)
 def test_event_log_has_exact_order_and_coverage(self):
  events=[]; coordinates=[]
  for case,steps in (("Z",200),("T",100),("S",330)):
   for policy in range(steps):
    for physics in range(20):
     events.extend(("control","physics","observer","reset_pending","observation_pending")); coordinates.extend((case,policy,physics,lane) for lane in range(32))
  self.assertEqual(len(events)//5,12600); self.assertEqual(len(coordinates),403200); self.assertEqual(len(set(coordinates)),403200)
  self.assertEqual(events[:5],["control","physics","observer","reset_pending","observation_pending"])
 def test_observer_is_after_scene_refresh_and_before_reset_path(self):
  register=ast.get_source_segment(self.text,self.methods["register_read_only_post_physics_observer"])
  self.assertLess(register.index("original_update(dt)"),register.index("callback(self._enhanced_b0_physics_rows())"))
  for token in ("_reset_idx(","write_","set_external_force"): self.assertNotIn(token,register)
 def test_protected_state_and_fail_closed_unavailable_wrenches(self):
  probe=ast.get_source_segment(self.text,self.methods["read_only_diagnostic_state_fingerprints"])
  for name in ("control","dynamics","rng","reset","actor_history","simulator"): self.assertIn(f'"{name}"',probe)
  rows=ast.get_source_segment(self.text,self.methods["_enhanced_b0_physics_rows"])
  self.assertIn('{"availability": "UNAVAILABLE"}',rows)
  for name in ("drag_aero","contact","joint_constraint","other"): self.assertIn(f'"{name}"',rows)
  self.assertNotIn('"availability":"AVAILABLE","force_n":[0',rows)
 def test_successor_identity_and_math_binding_are_exact(self):
  self.assertIn("from isaaclab.utils.math import quat_apply",self.text)
  self.assertIn('"UNIFIED-H1-R5-READINESS-B0-PLANT-R1-20261012"',self.text)
  self.assertNotIn("UNIFIED-H1-R5-STAGE1-B0-ENHANCED-20260927",self.text)
 def test_reset_episode_mapping_is_state_derived_and_three_resets_only(self):
  rows=ast.get_source_segment(self.text,self.methods["_enhanced_b0_physics_rows"]); run=ast.get_source_segment(self.text,self.methods["run_frozen_enhanced_b0_zts"]); reset=ast.get_source_segment(self.text,self.methods["_reset_idx"] )
  self.assertNotIn('(\"episode_id\",0)',rows); self.assertIn("_enhanced_b0_episode_ids",rows); self.assertIn("_enhanced_b0_reset_boundaries",reset)
  self.assertIn('self.reset(seed=20260909 if case == "Z" else None)',run); self.assertNotIn('self._reset_idx(',run)
  for token in ('accounting["before_reset"]','accounting["before_step"]','accounting["after_step"]'): self.assertIn(token,run)
  self.assertIn('_enhanced_b0_accounting["after_reset"]',reset)
 def test_reset_cache_lifecycle_is_fail_closed_on_actual_wrapper_methods(self):
  reset=ast.get_source_segment(self.text,self.methods["_reset_idx"])
  rows=ast.get_source_segment(self.text,self.methods["_enhanced_b0_physics_rows"])
  apply_action=ast.get_source_segment(self.text,self.methods["_apply_action"])
  run=ast.get_source_segment(self.text,self.methods["run_frozen_enhanced_b0_zts"])
  self.assertIn("last=self._enhanced_b0_last_snapshots.get(key)",reset)
  self.assertIn("last=self._enhanced_b0_snapshot(lane)",reset)
  self.assertNotIn("first_post_reset_state",reset)
  self.assertIn("_enhanced_b0_cache_unavailable.add(key)",reset)
  self.assertIn('"cache_generation":self._enhanced_b0_cache_generation',reset)
  self.assertIn("super()._apply_action()",apply_action)
  self.assertIn("_enhanced_b0_cache_generation += 1",apply_action)
  self.assertLess(apply_action.index("super()._apply_action()"),apply_action.index("_enhanced_b0_cache_unavailable.clear()"))
  self.assertIn("post-reset cache remains explicitly unavailable",rows)
  self.assertIn("post-reset cache was not rebuilt",rows)
  self.assertIn('"post_reset_states":[current_snapshot]',rows)
  finally_nodes=[node for node in ast.walk(self.methods["_reset_idx"]) if isinstance(node,ast.Try) and node.finalbody]
  self.assertTrue(any('_enhanced_b0_accounting["after_reset"]' in ast.get_source_segment(self.text,node) for node in finally_nodes))
  self.assertNotIn('accounting["after_reset"]',run)
 def test_all_lane_reset_window_is_explicit_and_generation_gated(self):
  reset=ast.get_source_segment(self.text,self.methods["_reset_idx"])
  self.assertIn("lanes=[int(lane) for lane in indices.detach().cpu().tolist()]",reset)
  self.assertIn("for lane in lanes",reset)
  self.assertIn("for lane,(last,initial) in before.items()",reset)
  rows=ast.get_source_segment(self.text,self.methods["_enhanced_b0_physics_rows"])
  self.assertLess(rows.index("cache remains explicitly unavailable"),rows.index("_enhanced_b0_reset_boundaries.pop"))
 def test_driver_is_exact_zero_target_no_retry(self):
  run=ast.get_source_segment(self.text,self.methods["run_frozen_enhanced_b0_zts"])
  for token in ("20260909",'(0.0, 0.0, 0.0)',"retries","12600","403200"): self.assertIn(token,run)
  for token in ("optimizer","checkpoint","train("): self.assertNotIn(token,run)

class ProductResetLifecycleExecutionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  tree=ast.parse(SOURCE.read_text())
  source_class=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="UnifiedH1DirectEnv")
  selected=[node for node in source_class.body if isinstance(node,ast.FunctionDef) and node.name in {"configure_enhanced_b0_execution_accounting","_reset_idx","_apply_action"}]
  base=ast.parse("class WrapperBase:\n def _apply_action(self):\n  self._lee=object(); self._wrench=object()\n").body[0]
  wrapper=ast.ClassDef(name="ExecutableWrapper",bases=[ast.Name(id="WrapperBase",ctx=ast.Load())],keywords=[],decorator_list=[],body=selected)
  module=ast.fix_missing_locations(ast.Module(body=[base,wrapper],type_ignores=[]))
  namespace={"Mapping":__import__("collections.abc").abc.Mapping,"Sequence":object,"torch":type("Torch",(),{"Tensor":object})}
  exec(compile(module,str(SOURCE),"exec"),namespace)
  cls.wrapper_type=namespace["ExecutableWrapper"]

 def wrapper(self, *, fail_reset=False):
  value=self.wrapper_type()
  class Indices:
   def detach(self): return self
   def cpu(self): return self
   def tolist(self): return list(range(32))
  value._validate_reset_indices=lambda env_ids:Indices()
  value._enhanced_b0_active=True; value._h1_evaluation_case="Z"; value._enhanced_b0_explicit_reset=True
  value._enhanced_b0_last_snapshots={("Z",lane):{"controller":{"marker":lane},"rotors":[{"marker":lane}]} for lane in range(32)}
  value._enhanced_b0_snapshot=lambda lane:(_ for _ in ()).throw(AssertionError("stale cache fallback used"))
  value._enhanced_b0_previous_system={("Z",lane):{"old":True} for lane in range(32)}
  value._enhanced_b0_previous_velocity={("Z",lane):(1,2,3) for lane in range(32)}
  value._enhanced_b0_episode_ids={("Z",lane):7 for lane in range(32)}
  value._enhanced_b0_reset_boundaries={}; value._enhanced_b0_cache_unavailable=set(); value._enhanced_b0_cache_generation=11
  value._lee=object(); value._wrench=object(); value.reset_calls=0; value.settled=[]
  def reset_impl(indices):
   value.reset_calls+=1; value._lee=None; value._wrench=None
   if fail_reset: raise RuntimeError("injected reset failure")
  value._reset_idx_impl=reset_impl
  value._enhanced_b0_accounting={"after_reset":lambda case:value.settled.append(("explicit",case,1)),"after_automatic_reset":lambda case,step,lanes:value.settled.append(("automatic",case,lanes))}
  return value

 def test_accounting_binding_requires_the_automatic_boundary_callback(self):
  value=self.wrapper(); value._enhanced_b0_accounting=None
  exact={name:(lambda *args:None) for name in ("before_reset","after_reset","before_step","after_automatic_reset","after_step")}
  value.configure_enhanced_b0_execution_accounting(exact)
  self.assertEqual(tuple(value._enhanced_b0_accounting),tuple(exact))
  value=self.wrapper(); value._enhanced_b0_accounting=None
  incomplete={name:callback for name,callback in exact.items() if name!="after_automatic_reset"}
  with self.assertRaisesRegex(RuntimeError,"binding changed"): value.configure_enhanced_b0_execution_accounting(incomplete)

 def test_initial_reset_truthfully_has_no_pre_cache_and_never_snapshots_cleared_state(self):
  value=self.wrapper();value._enhanced_b0_last_snapshots={};value._enhanced_b0_cache_generation=0
  value._enhanced_b0_snapshot=lambda lane:(_ for _ in ()).throw(AssertionError("initial reset must not snapshot cache"))
  value._reset_idx(None)
  self.assertEqual(value.settled,[("explicit","Z",1)])
  for boundary in value._enhanced_b0_reset_boundaries.values():
   self.assertTrue(boundary["initial_boundary"]);self.assertIsNone(boundary["last_pre_reset_state"])
 def test_initial_reset_exception_settles_once_without_invented_cache(self):
  value=self.wrapper(fail_reset=True);value._enhanced_b0_last_snapshots={};value._enhanced_b0_cache_generation=0
  with self.assertRaisesRegex(RuntimeError,"injected reset failure"):value._reset_idx(None)
  self.assertEqual(value.settled,[("explicit","Z",1)])
  self.assertTrue(all(row["initial_boundary"] and row["last_pre_reset_state"] is None for row in value._enhanced_b0_reset_boundaries.values()))
 def test_full_lane_reset_preserves_last_valid_state_and_unavailable_window(self):
  value=self.wrapper(); value._reset_idx(None)
  self.assertEqual(len(value._enhanced_b0_reset_boundaries),32)
  self.assertEqual(value._enhanced_b0_cache_unavailable,{("Z",lane) for lane in range(32)})
  self.assertIsNone(value._lee); self.assertIsNone(value._wrench)
  self.assertEqual(value.settled,[("explicit","Z",1)])
  for lane in range(32):
   boundary=value._enhanced_b0_reset_boundaries[("Z",lane)]
   self.assertEqual(boundary["episode_id"],7)
   self.assertEqual(boundary["cache_generation"],11)
   self.assertEqual(boundary["last_pre_reset_state"]["controller"]["marker"],lane)

 def test_real_action_rebuild_is_required_before_window_closes(self):
  value=self.wrapper(); value._reset_idx(None)
  self.assertTrue(value._enhanced_b0_cache_unavailable)
  value._apply_action()
  self.assertIsNotNone(value._lee); self.assertIsNotNone(value._wrench)
  self.assertEqual(value._enhanced_b0_cache_generation,12)
  self.assertFalse(value._enhanced_b0_cache_unavailable)

 def test_reset_exception_is_settled_once_and_mapping_is_not_lost_or_duplicated(self):
  value=self.wrapper(fail_reset=True)
  with self.assertRaisesRegex(RuntimeError,"injected reset failure"): value._reset_idx(None)
  self.assertEqual(value.settled,[("explicit","Z",1)])
  self.assertEqual(len(value._enhanced_b0_reset_boundaries),32)
  with self.assertRaisesRegex(RuntimeError,"already awaiting"): value._reset_idx(None)
  self.assertEqual(value.reset_calls,1); self.assertEqual(value.settled,[("explicit","Z",1)])

 def test_automatic_partial_reset_exception_settles_at_boundary(self):
  value=self.wrapper(fail_reset=True); value._enhanced_b0_explicit_reset=False
  value._enhanced_b0_coordinate=("Z",19,20)
  class Partial:
   def detach(self): return self
   def cpu(self): return self
   def tolist(self): return [2,5]
  value._validate_reset_indices=lambda env_ids:Partial()
  with self.assertRaisesRegex(RuntimeError,"injected reset failure"): value._reset_idx(None)
  self.assertEqual(value.settled,[("automatic","Z",2)])
  self.assertEqual(set(value._enhanced_b0_reset_boundaries),{("Z",2),("Z",5)})

 def test_automatic_full_reset_success_settles_once(self):
  value=self.wrapper(); value._enhanced_b0_explicit_reset=False
  value._enhanced_b0_coordinate=("Z",21,20); value._reset_idx(None)
  self.assertEqual(value.settled,[("automatic","Z",32)])

 def test_automatic_full_reset_failure_settles_once(self):
  value=self.wrapper(fail_reset=True); value._enhanced_b0_explicit_reset=False
  value._enhanced_b0_coordinate=("T",4,20); value._h1_evaluation_case="T"
  value._enhanced_b0_last_snapshots={("T",lane):{"controller":{"marker":lane},"rotors":[{"marker":lane}]} for lane in range(32)}
  with self.assertRaisesRegex(RuntimeError,"injected reset failure"): value._reset_idx(None)
  self.assertEqual(value.settled,[("automatic","T",32)])
  self.assertEqual({key[0] for key in value._enhanced_b0_reset_boundaries},{"T"})

 def test_successful_automatic_reset_then_step_diagnostic_failure_is_not_double_counted(self):
  value=self.wrapper(); value._enhanced_b0_explicit_reset=False
  value._enhanced_b0_coordinate=("S",8,20); value._h1_evaluation_case="S"
  value._enhanced_b0_last_snapshots={("S",lane):{"controller":{"marker":lane},"rotors":[{"marker":lane}]} for lane in range(32)}
  value._reset_idx(None)
  try:
   raise RuntimeError("injected post-reset diagnostic failure")
  except RuntimeError as error:
   self.assertIn("post-reset diagnostic",str(error))
  self.assertEqual(value.settled,[("automatic","S",32)])
  self.assertEqual(len(value._enhanced_b0_reset_boundaries),32)

 def test_empty_reset_does_not_settle_or_create_cross_episode_rows(self):
  value=self.wrapper(); value._enhanced_b0_explicit_reset=False
  value._enhanced_b0_coordinate=("Z",3,20)
  class Empty:
   def detach(self): return self
   def cpu(self): return self
   def tolist(self): return []
  value._validate_reset_indices=lambda env_ids:Empty()
  value._reset_idx(None)
  self.assertEqual(value.settled,[])
  self.assertEqual(value._enhanced_b0_reset_boundaries,{})

if __name__=="__main__": unittest.main()
