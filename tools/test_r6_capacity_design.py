"""Check capacity arithmetic against the actual research entry contract."""
import ast
import json
from pathlib import Path
import unittest

from tools.build_r6_capacity_design import run, BASE, ROOT, GIB


class CapacityDesignTests(unittest.TestCase):
    def test_training_total_includes_actual_entry_native_subtraction(self):
        tree = ast.parse((ROOT/'training_workloads/unified_h1/live_adapter.py').read_text())
        deductions = [n.value.right for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == 'PILOT_WORKLOAD_WRITE_CAP_BYTES' for t in n.targets)
            and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Sub)]
        self.assertEqual(len(deductions), 1)
        # The actual entry uses a literal arithmetic expression; prohibit names
        # and calls so this remains a CPU source check, not execution of runtime.
        self.assertTrue(all(isinstance(n, (ast.Constant, ast.BinOp, ast.Mult, ast.Pow))
                            for n in ast.walk(deductions[0])))
        deduction = eval(compile(ast.Expression(deductions[0]), '<native-deduction>', 'eval'), {'__builtins__': {}})
        report = run(); training = report['training_compression']
        self.assertEqual(training['native_per_training_job_bytes'], deduction)
        for kind in ('confirmation', 'development'):
            self.assertEqual(training[f'proposed_per_{kind}_job_bytes'] - deduction,
                             training[f'proposed_per_{kind}_payload_bytes'])
        self.assertEqual(report['final_components_bytes']['three_training_jobs'],
                         3*training['proposed_per_confirmation_job_bytes'])

    def test_old_snapshot_is_charged_separately_from_new_input(self):
        report = run(); development = report['development_example_components_bytes']
        self.assertEqual(development['historical_snapshot_reservation'], 2*GIB)
        self.assertEqual(development['new_baseline_snapshot'], 2*GIB)

    def test_current_plan_reports_deficit_without_reducing_final_coverage(self):
        report = run(); previous = json.loads((BASE/'CAPACITY_COMPOSED_DRAFT_V4.json').read_text())
        self.assertEqual(report['full_evaluation_transitions'], 193600)
        self.assertEqual(report['final_nontraining_transitions'], 774400)
        self.assertEqual(report['blocks'], previous['blocks'])
        self.assertFalse(report['development_example_fits'])
        self.assertFalse(report['baseline_only_fits'])
        for prefix in ('development_example', 'baseline_only'):
            self.assertEqual(report[prefix+'_deficit_bytes'],
                report[prefix+'_total_bytes']-report['development_limit_after_proposal_bytes'])
        self.assertFalse(report['capacity_pass'])
        self.assertFalse(report['ledger_changed'])

    def test_read_only_rebuild_preserves_operating_budget_and_old_evidence(self):
        paths = [BASE/'goal_budget.json', BASE/'batch/state.json', BASE/'CAPACITY_COMPOSED_DRAFT_V4.json']
        before = [p.read_bytes() for p in paths]
        report = run()
        self.assertEqual(before, [p.read_bytes() for p in paths])
        self.assertEqual(sum(report['final_components_bytes'].values()), report['final_total_bytes'])
        self.assertEqual(sum(report['development_example_components_bytes'].values()),
                         report['development_example_total_bytes'])


if __name__ == '__main__':
    unittest.main()
