from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_preparation.scale_002.budget import (  # noqa: E402
    BudgetError,
    derive_budget,
    load_and_validate_proposal,
    validate_proposal,
)


class Scale002Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal, self.budgets = load_and_validate_proposal()

    def test_recommended_exact_budget(self) -> None:
        budget = derive_budget(additional_transitions=872_000)
        self.assertEqual(budget.full_updates, 1703)
        self.assertEqual(budget.tail_samples, 64)
        self.assertEqual(budget.tail_horizon, 2)
        self.assertEqual(budget.new_updates, 1704)
        self.assertEqual(budget.target_update_counter, 1954)
        self.assertEqual(budget.new_optimizer_steps_per_actor_and_critic, 13_625)
        self.assertEqual(budget.target_optimizer_steps_per_actor_and_critic, 15_625)
        self.assertEqual(budget.policy_steps_per_environment, 27_250)
        self.assertEqual(budget.physics_steps_per_environment, 545_000)
        self.assertEqual(budget.aggregate_policy_driven_physics_transitions, 17_440_000)
        self.assertEqual(budget.regular_checkpoint_count, 68)
        self.assertEqual(budget.new_checkpoint_count, 70)
        self.assertEqual(budget.checkpoint_counters[-3:], (1950, 1953, 1954))

    def test_alternative_one_million_new_budget(self) -> None:
        budget = derive_budget(additional_transitions=1_000_000)
        self.assertEqual(budget.cumulative_transitions, 1_128_000)
        self.assertEqual(budget.full_updates, 1953)
        self.assertEqual(budget.tail_samples, 64)
        self.assertEqual(budget.new_updates, 1954)
        self.assertEqual(budget.target_update_counter, 2204)
        self.assertEqual(budget.new_optimizer_steps_per_actor_and_critic, 15_625)
        self.assertEqual(budget.target_optimizer_steps_per_actor_and_critic, 17_625)
        self.assertEqual(budget.policy_steps_per_environment, 31_250)
        self.assertEqual(budget.physics_steps_per_environment, 625_000)

    def test_proposal_and_curriculum_cover_every_sample(self) -> None:
        recommended = self.budgets["recommended"]
        self.assertEqual(recommended["additional_transitions"], 872_000)
        self.assertEqual(recommended["cumulative_transitions"], 1_000_000)
        phase_2 = self.proposal["command_curriculum"]["phase_2"]
        self.assertEqual(phase_2["complete_cycles"] * 8 * 32 + 50, 19_250)
        self.assertEqual(sum(item["policy_steps"] for item in phase_2["final_partial_cycle"]), 50)

    def test_rejects_non_integral_tail_or_minibatch(self) -> None:
        with self.assertRaises(BudgetError):
            derive_budget(additional_transitions=872_001)
        with self.assertRaises(BudgetError):
            derive_budget(additional_transitions=872_032)

    def test_rejects_authorization_target_and_identity_mutations(self) -> None:
        mutations = []
        authorized = deepcopy(self.proposal)
        authorized["authorization"]["training_authorized"] = True
        mutations.append(authorized)
        rounded = deepcopy(self.proposal)
        rounded["target"]["additional_transitions"] = 872_448
        mutations.append(rounded)
        missing_pin = deepcopy(self.proposal)
        del missing_pin["pins"]
        mutations.append(missing_pin)
        fake_entry = deepcopy(self.proposal)
        fake_entry["future_execution"]["entry_status"] = "READY"
        mutations.append(fake_entry)
        restored_environment = deepcopy(self.proposal)
        restored_environment["initialization"]["environment_state_restored"] = True
        mutations.append(restored_environment)
        wrong_resource_fact = deepcopy(self.proposal)
        wrong_resource_fact["resource_estimate"]["source_minimum_free_cuda_memory_bytes"] -= 1
        mutations.append(wrong_resource_fact)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(BudgetError):
                    validate_proposal(mutation)

    def test_strict_json_rejects_duplicate_and_nan(self) -> None:
        for content in ('{"schema_version": 1, "schema_version": 2}', '{"value": NaN}'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "proposal.json"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(BudgetError):
                    load_and_validate_proposal(path)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Scale002Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
