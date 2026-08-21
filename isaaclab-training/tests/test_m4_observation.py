"""Semantic, negative, source, and provenance tests for the M4 actor boundary."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys
import textwrap
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instinctlab.observations import (  # noqa: E402
    APPROVED_INPUT_NAMES,
    OUTPUT_SLICE_PROVENANCE,
    RAY_COUNT,
    ActorObservationAuditError,
    ActorObservationHistory,
    audit_actor_observation,
)
from instinctlab.observations import actor as actor_module  # noqa: E402
from instinctlab.observations import audit as audit_module  # noqa: E402


def frame(
    batch: int = 2,
    value: float = 1.0,
    sensor_time: float = 1.0,
    policy_time: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    ranges = torch.full((batch, RAY_COUNT), value, device=device, dtype=dtype)
    validity = torch.ones((batch, RAY_COUNT), device=device, dtype=torch.bool)
    angular = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=dtype).expand(batch, -1).clone()
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype).expand(batch, -1).clone()
    command = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype).expand(batch, -1).clone()
    previous = torch.tensor([-1.0, -2.0, -3.0], device=device, dtype=dtype).expand(batch, -1).clone()
    return {
        "ranges_m": ranges,
        "validity": validity,
        "sensor_frame_time_s": torch.full((batch, 1), sensor_time, device=device, dtype=dtype),
        "policy_time_s": torch.full((batch, 1), policy_time, device=device, dtype=dtype),
        "body_angular_velocity_rad_s": angular,
        "gravity_direction_body": gravity,
        "commanded_body_velocity_m_s": command,
        "previous_issued_body_velocity_m_s": previous,
    }


class ObservationContractTests(unittest.TestCase):
    def test_exact_schema_mapping_interleave_fields_and_initial_zero_fill(self) -> None:
        builder = ActorObservationHistory(tau_s=2.0)
        inputs = frame(sensor_time=4.0, policy_time=4.5)
        sentinels = {
            (0, 0): 0.1,
            (0, 359): 1.25,
            (1, 1): 40.0,
            (58, 0): 2.5,
            (58, 359): 39.9,
            (17, 123): 7.75,
        }
        inputs["validity"].zero_()
        inputs["ranges_m"].fill_(40.0)
        for (elevation, azimuth), value in sentinels.items():
            flat_index = elevation * 360 + azimuth
            inputs["ranges_m"][:, flat_index] = value
            inputs["validity"][:, flat_index] = True
        output = builder.update(**inputs)
        self.assertEqual(set(output), {"lidar_grid", "state_vec"})
        self.assertEqual(tuple(output["lidar_grid"].shape), (2, 12, 59, 360))
        self.assertEqual(tuple(output["state_vec"].shape), (2, 52))
        self.assertEqual(torch.count_nonzero(output["lidar_grid"][:, :9]).item(), 0)
        self.assertEqual(torch.count_nonzero(output["state_vec"][:, :39]).item(), 0)
        for (elevation, azimuth), value in sentinels.items():
            flat_index = elevation * 360 + azimuth
            self.assertEqual(
                output["lidar_grid"][0, 9, elevation, azimuth].item(),
                inputs["ranges_m"][0, flat_index].item(),
            )
        expected_reliability = torch.exp(torch.tensor(-0.25))
        self.assertTrue(torch.equal(output["lidar_grid"][:, 10], inputs["validity"].float().reshape(2, 59, 360)))
        self.assertTrue(torch.allclose(output["lidar_grid"][:, 11][inputs["validity"].reshape(2, 59, 360)], expected_reliability))
        self.assertEqual(torch.count_nonzero(output["lidar_grid"][:, 11][~inputs["validity"].reshape(2, 59, 360)]).item(), 0)
        expected_state = torch.cat(
            [
                inputs["body_angular_velocity_rad_s"],
                inputs["gravity_direction_body"],
                inputs["commanded_body_velocity_m_s"],
                inputs["previous_issued_body_velocity_m_s"],
                torch.full((2, 1), 0.5),
            ],
            dim=-1,
        )
        self.assertTrue(torch.equal(output["state_vec"][:, 39:52], expected_state))
        audit_actor_observation(output)

    def test_invalid_range_pair_rejection_is_exact_and_atomic(self) -> None:
        builder = ActorObservationHistory()
        reference = ActorObservationHistory()
        first = frame(value=1.0, sensor_time=1.0, policy_time=1.0)
        builder.update(**first)
        reference.update(**frame(value=1.0, sensor_time=1.0, policy_time=1.0))
        for bad_fill in (1.0, 39.999996):
            bad = frame(value=2.0, sensor_time=2.0, policy_time=2.0)
            bad["validity"][0, 123] = False
            bad["ranges_m"][0, 123] = bad_fill
            with self.subTest(bad_fill=bad_fill):
                with self.assertRaisesRegex(ValueError, "exact 40.0"):
                    builder.update(**bad)
        second = frame(value=2.0, sensor_time=2.0, policy_time=2.0)
        actual = builder.update(**second)
        expected = reference.update(**frame(value=2.0, sensor_time=2.0, policy_time=2.0))
        self.assertTrue(torch.equal(actual["lidar_grid"], expected["lidar_grid"]))
        self.assertTrue(torch.equal(actual["state_vec"], expected["state_vec"]))

    def test_history_oldest_to_newest_and_rollover(self) -> None:
        builder = ActorObservationHistory()
        output = None
        for step in range(1, 6):
            output = builder.update(**frame(value=float(step), sensor_time=float(step), policy_time=float(step)))
        assert output is not None
        self.assertEqual(output["lidar_grid"][0, 0::3, 0, 0].tolist(), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(output["state_vec"][0, 6::13].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_repeated_frame_age_decay_and_timestamp_rules(self) -> None:
        builder = ActorObservationHistory(tau_s=1.0)
        first = builder.update(**frame(sensor_time=1.0, policy_time=1.0))
        repeated = builder.update(**frame(sensor_time=1.0, policy_time=2.0))
        self.assertEqual(first["lidar_grid"][0, 11, 0, 0].item(), 1.0)
        self.assertAlmostEqual(repeated["lidar_grid"][0, 11, 0, 0].item(), torch.exp(torch.tensor(-1.0)).item())
        self.assertEqual(repeated["state_vec"][0, 51].item(), 1.0)
        with self.assertRaisesRegex(ValueError, "sensor_frame_time_s regressed"):
            builder.update(**frame(sensor_time=0.9, policy_time=2.1))
        with self.assertRaisesRegex(ValueError, "policy_time_s regressed"):
            builder.update(**frame(sensor_time=1.0, policy_time=1.9))
        with self.assertRaisesRegex(ValueError, "must not precede"):
            ActorObservationHistory().update(**frame(sensor_time=2.0, policy_time=1.0))

    def test_selective_and_full_reset_clear_history_and_time_provenance(self) -> None:
        builder = ActorObservationHistory()
        builder.update(**frame(sensor_time=5.0, policy_time=5.0))
        builder.update(**frame(value=2.0, sensor_time=6.0, policy_time=6.0))
        builder.reset(torch.tensor([1], dtype=torch.long))
        mixed = frame(value=3.0, sensor_time=7.0, policy_time=7.0)
        mixed["sensor_frame_time_s"][1] = 1.0
        mixed["policy_time_s"][1] = 1.0
        output = builder.update(**mixed)
        self.assertGreater(torch.count_nonzero(output["lidar_grid"][0, :9]).item(), 0)
        self.assertEqual(torch.count_nonzero(output["lidar_grid"][1, :9]).item(), 0)
        builder.reset()
        restarted = builder.update(**frame(sensor_time=0.0, policy_time=0.0))
        self.assertEqual(torch.count_nonzero(restarted["lidar_grid"][:, :9]).item(), 0)

    def test_reset_rejects_malformed_indices(self) -> None:
        builder = ActorObservationHistory()
        with self.assertRaises(RuntimeError):
            builder.reset(torch.tensor([0]))
        builder.update(**frame())
        for bad in (torch.tensor([[0]]), torch.tensor([0.0]), torch.tensor([0, 0]), torch.tensor([2])):
            with self.assertRaises((TypeError, ValueError, IndexError)):
                builder.reset(bad)

    def test_tau_and_gravity_fail_closed(self) -> None:
        for bad_tau in (0.0, -1.0, float("nan"), float("inf"), True, "0.5"):
            with self.assertRaises((TypeError, ValueError)):
                ActorObservationHistory(tau_s=bad_tau)  # type: ignore[arg-type]
        inputs = frame()
        inputs["gravity_direction_body"][:, 2] = -0.9
        with self.assertRaisesRegex(ValueError, "unit length"):
            ActorObservationHistory().update(**inputs)

    def test_malformed_shapes_types_batches_dtypes_and_nonfinite_fail_closed(self) -> None:
        cases: list[tuple[str, object]] = [
            ("ranges_m", torch.ones(2, RAY_COUNT - 1)),
            ("validity", torch.ones(2, RAY_COUNT)),
            ("sensor_frame_time_s", torch.ones(2)),
            ("body_angular_velocity_rad_s", torch.ones(3, 3)),
            ("commanded_body_velocity_m_s", torch.ones(2, 3, dtype=torch.float64)),
            ("ranges_m", torch.full((2, RAY_COUNT), float("nan"))),
            ("ranges_m", torch.full((2, RAY_COUNT), 40.1)),
            ("ranges_m", "not-a-tensor"),
        ]
        for name, bad_value in cases:
            inputs = frame()
            inputs[name] = bad_value  # type: ignore[assignment]
            with self.subTest(name=name, bad=type(bad_value).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    ActorObservationHistory().update(**inputs)

    def test_missing_input_raises_instead_of_supplying_a_default(self) -> None:
        inputs = frame()
        del inputs["previous_issued_body_velocity_m_s"]
        with self.assertRaises(TypeError):
            ActorObservationHistory().update(**inputs)

    def test_device_mismatch_fails_closed_when_cuda_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        inputs = frame(device="cuda")
        inputs["policy_time_s"] = inputs["policy_time_s"].cpu()
        with self.assertRaisesRegex(ValueError, "device"):
            ActorObservationHistory().update(**inputs)

    def test_inputs_and_returned_outputs_do_not_alias_internal_history(self) -> None:
        builder = ActorObservationHistory()
        inputs = frame()
        output = builder.update(**inputs)
        inputs["ranges_m"].fill_(9.0)
        output["lidar_grid"].fill_(8.0)
        next_output = builder.update(**frame(value=2.0, sensor_time=2.0, policy_time=2.0))
        self.assertEqual(next_output["lidar_grid"][0, 6, 0, 0].item(), 1.0)
        self.assertNotEqual(next_output["lidar_grid"][0, 6, 0, 0].item(), 8.0)


class ActorAuditTests(unittest.TestCase):
    def valid_output(self) -> dict[str, torch.Tensor]:
        return ActorObservationHistory().update(**frame())

    def test_positive_audit_and_approved_command_name_distinction(self) -> None:
        audit_actor_observation(self.valid_output())
        for name in ("commanded_body_velocity_m_s", "previous_issued_body_velocity_m_s"):
            self.assertIsNone(audit_module._forbidden_reason(name))
        for name in ("actual_velocity", "estimated_velocity", "reconstructed_velocity"):
            self.assertIsNotNone(audit_module._forbidden_reason(name))

    def test_nested_forbidden_and_every_leakage_class_are_rejected(self) -> None:
        forbidden = (
            "pose", "position", "odometry", "actual_velocity", "estimated_velocity",
            "reconstructed_velocity", "root_state", "local_map", "slam_state",
            "obstacle_truth", "privileged_simulator_state", "future_frame",
            "critic_features", "reward_terms", "logging_payload", "controller_internal",
        )
        for key in forbidden:
            payload = self.valid_output()
            payload["nested"] = {"deeper": {key: torch.zeros(1)}}  # type: ignore[assignment]
            with self.subTest(key=key):
                with self.assertRaises(ActorObservationAuditError):
                    audit_actor_observation(payload)

    def test_missing_extra_nested_and_malformed_values_are_rejected(self) -> None:
        fixtures: list[object] = [
            {"lidar_grid": self.valid_output()["lidar_grid"]},
            {**self.valid_output(), "extra": torch.zeros(1)},
            {"lidar_grid": {}, "state_vec": self.valid_output()["state_vec"]},
            [self.valid_output()],
        ]
        bad_shape = self.valid_output()
        bad_shape["state_vec"] = torch.zeros(2, 51)
        fixtures.append(bad_shape)
        bad_finite = self.valid_output()
        bad_finite["lidar_grid"][0, 0, 0, 0] = float("nan")
        fixtures.append(bad_finite)
        bad_validity = self.valid_output()
        bad_validity["lidar_grid"][0, 10, 0, 0] = 0.5
        fixtures.append(bad_validity)
        bad_reliability = self.valid_output()
        bad_reliability["lidar_grid"][0, 11, 0, 0] = 1.1
        fixtures.append(bad_reliability)
        for fixture in fixtures:
            with self.assertRaises(ActorObservationAuditError):
                audit_actor_observation(fixture)  # type: ignore[arg-type]

    def test_audit_rejects_wrong_dtype_device_and_range_domain(self) -> None:
        wrong_dtype = self.valid_output()
        wrong_dtype["state_vec"] = wrong_dtype["state_vec"].double()
        with self.assertRaises(ActorObservationAuditError):
            audit_actor_observation(wrong_dtype)
        wrong_range = self.valid_output()
        wrong_range["lidar_grid"][0, 9, 0, 0] = 41.0
        with self.assertRaises(ActorObservationAuditError):
            audit_actor_observation(wrong_range)
        if torch.cuda.is_available():
            wrong_device = self.valid_output()
            wrong_device["state_vec"] = wrong_device["state_vec"].cuda()
            with self.assertRaises(ActorObservationAuditError):
                audit_actor_observation(wrong_device)

    def test_audit_rejects_incoherent_pairs_partial_frames_and_gravity(self) -> None:
        invalid_nonfill = self.valid_output()
        invalid_nonfill["lidar_grid"][0, 10, 0, 0] = 0.0
        invalid_nonfill["lidar_grid"][0, 11, 0, 0] = 0.0
        with self.assertRaisesRegex(ActorObservationAuditError, "exact 40.0"):
            audit_actor_observation(invalid_nonfill)

        valid_zero = self.valid_output()
        valid_zero["lidar_grid"][0, 9, 0, 0] = 0.0
        with self.assertRaisesRegex(ActorObservationAuditError, "closed"):
            audit_actor_observation(valid_zero)

        partial_zero = self.valid_output()
        partial_zero["lidar_grid"][0, 0, 0, 0] = 1.0
        with self.assertRaises(ActorObservationAuditError):
            audit_actor_observation(partial_zero)

        nonunit_gravity = self.valid_output()
        nonunit_gravity["state_vec"][0, 42:45] = torch.tensor([0.0, 0.0, -0.9])
        with self.assertRaisesRegex(ActorObservationAuditError, "unit length"):
            audit_actor_observation(nonunit_gravity)

    def test_audit_accepts_zero_history_and_mixed_valid_invalid_40m(self) -> None:
        inputs = frame()
        inputs["ranges_m"][:, 0] = 40.0
        inputs["validity"][0, 0] = True
        inputs["validity"][1, 0] = False
        output = ActorObservationHistory().update(**inputs)
        audit_actor_observation(output)
        self.assertEqual(output["lidar_grid"][0, 10, 0, 0].item(), 1.0)
        self.assertEqual(output["lidar_grid"][1, 10, 0, 0].item(), 0.0)


class SourceAndProvenanceTests(unittest.TestCase):
    def test_update_signature_is_exact_and_has_no_generic_container(self) -> None:
        signature = inspect.signature(ActorObservationHistory.update)
        names = tuple(name for name in signature.parameters if name != "self")
        self.assertEqual(names, APPROVED_INPUT_NAMES)
        self.assertTrue(all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()))

    def test_builder_source_has_no_forbidden_producer_access(self) -> None:
        tree = ast.parse(inspect.getsource(actor_module))
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        forbidden_identifiers = {
            "drone_state", "root_state", "pose", "position", "odometry", "actual_velocity",
            "estimated_velocity", "reconstructed_velocity", "map", "slam", "obstacle_truth",
            "simulator_state", "future_frames", "critic", "reward", "logging", "controller",
        }
        self.assertTrue(forbidden_identifiers.isdisjoint(referenced))

    def test_contract_machine_readable_provenance_matches_code(self) -> None:
        contract = ROOT.parent / "docs" / "blackwell_devlog" / "phases" / "M4" / "OBSERVATION_CONTRACT.md"
        text = contract.read_text(encoding="utf-8")
        payload = text.split("<!-- PROVENANCE_JSON_BEGIN -->", 1)[1].split("<!-- PROVENANCE_JSON_END -->", 1)[0]
        documented = json.loads(payload.strip().removeprefix("```json").removesuffix("```").strip())
        self.assertEqual(documented, list(OUTPUT_SLICE_PROVENANCE))

    def test_only_two_actor_output_literals_exist_in_return_builder(self) -> None:
        source = textwrap.dedent(inspect.getsource(ActorObservationHistory._actor_output_copy))
        literals = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertEqual(literals, {"lidar_grid", "state_vec"})

    def test_m4_constants_and_flattening_match_accepted_m3_source_ast(self) -> None:
        m3_path = ROOT / "instinctlab" / "sensors" / "mid360.py"
        m3_tree = ast.parse(m3_path.read_text(encoding="utf-8"))
        assignments = {
            node.targets[0].id: node.value
            for node in m3_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        numeric = {
            name: ast.literal_eval(assignments[name])
            for name in ("AZIMUTH_SAMPLES", "ELEVATION_SAMPLES", "MIN_RANGE_M", "MAX_RANGE_M")
        }
        ray_expression = assignments["RAY_COUNT"]
        self.assertIsInstance(ray_expression, ast.BinOp)
        self.assertIsInstance(ray_expression.op, ast.Mult)
        numeric["RAY_COUNT"] = numeric[ray_expression.left.id] * numeric[ray_expression.right.id]  # type: ignore[attr-defined]
        self.assertEqual(numeric["AZIMUTH_SAMPLES"], actor_module.AZIMUTH_SAMPLES)
        self.assertEqual(numeric["ELEVATION_SAMPLES"], actor_module.ELEVATION_SAMPLES)
        self.assertEqual(numeric["RAY_COUNT"], actor_module.RAY_COUNT)
        self.assertEqual(numeric["MIN_RANGE_M"], actor_module.M3_MIN_RANGE_M)
        self.assertEqual(numeric["MAX_RANGE_M"], actor_module.M3_MAX_RANGE_M)

        pattern = next(
            node for node in m3_tree.body if isinstance(node, ast.FunctionDef) and node.name == "mid360_pattern"
        )
        pattern_source = ast.unparse(pattern)
        self.assertIn(
            "elevation_grid, azimuth_grid = torch.meshgrid(elevation_deg, azimuth_deg, indexing='ij')",
            pattern_source,
        )
        self.assertIn("azimuth_grid.reshape(-1)", pattern_source)
        self.assertIn("elevation_grid.reshape(-1)", pattern_source)
        for elevation, azimuth in ((0, 0), (0, 359), (1, 0), (58, 359)):
            flat_index = elevation * numeric["AZIMUTH_SAMPLES"] + azimuth
            self.assertEqual(divmod(flat_index, actor_module.AZIMUTH_SAMPLES), (elevation, azimuth))


if __name__ == "__main__":
    unittest.main(verbosity=2)
