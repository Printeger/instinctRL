#!/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11
"""Deterministic, CPU-only E0 evidence checks for the frozen R4 run."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

SCHEMA = "instinctrl-r4-review-e0/1"
ROOT = Path("/home/mint/rl_dev")
HERE = Path(__file__).resolve().parent
LOG = ROOT / "instinctRL_coordination/attempts/development/UNIFIED-H1-R4/UNIFIED-H1-R4-D0001/results/r4-training/learning_log"
TRAIN = LOG.parent
EVAL = ROOT / "instinctRL_coordination/attempts/development/UNIFIED-H1-R4/UNIFIED-H1-R4-E0002/results/evaluation"
PRODUCT = ROOT / "worktrees/instinctRL-unified-h1-r3-d0011-product"
PYROOT = PRODUCT / "isaaclab-training"


def lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def max_abs(left, right) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def clip_norm(vector, limit=2.0):
    norm = math.sqrt(sum(float(x) ** 2 for x in vector))
    scale = min(1.0, limit / max(norm, float.fromhex("0x1p-126")))
    return [float(x) * scale for x in vector]


def minibatches() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(lines(LOG / "minibatches.jsonl"))
    updates = list(lines(LOG / "updates.jsonl"))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["update"]].append(row)
    local_ok = all([r["minibatch_ordinal"] for r in rs] == list(range(8)) for rs in grouped.values())
    coverage_ok = all(len(u["coverage"]) == 1 and len(u["coverage"][0]) == 512
                      and set(u["coverage"][0]) == {1} for u in updates)
    optimizer_ok = all(u["actor_optimizer_steps"] == 8 * u["completed_updates"]
                       and u["critic_optimizer_steps"] == 8 * u["completed_updates"] for u in updates)
    early = [r for r in rows if r["update"] <= 64]
    top = sorted(early, key=lambda r: (-r["approximate_kl"], r["update"], r["minibatch_ordinal"]))[:10]
    linked = []
    compact_cache: dict[tuple[int, int], dict[str, Any]] = {}
    wanted = set()
    for rank, row in enumerate(top, 1):
        seed = 1701 + row["update"] - 1
        generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
        permutation = torch.randperm(512, generator=generator)
        ordinal = row["minibatch_ordinal"]
        indices = permutation[ordinal * 64:(ordinal + 1) * 64].tolist()
        for index in indices:
            wanted.add((row["update"], index // 32))
        linked.append({"rank": rank, "update": row["update"], "minibatch_ordinal": ordinal,
                       "approximate_kl": row["approximate_kl"], "ratio_mean": row["ratio_mean"],
                       "clip_fraction": row["clip_fraction"], "sample_indices": indices})
    for row in lines(LOG / "compact_rollouts.jsonl"):
        key = (row["update"], row["step"] % 16)
        if key in wanted:
            compact_cache[key] = row
    for item in linked:
        samples = []
        for index in item.pop("sample_indices"):
            step, lane = divmod(index, 32)
            row = compact_cache[(item["update"], step)]
            samples.append({"flat_index": index, "step_in_update": step, "lane": lane,
                            "action_sample": row["action_sample"][lane],
                            "action_mean": row["action_mean"][lane],
                            "command_b": row["command_b"][lane]})
        item["samples"] = samples
        item["probability_replay"] = "NOT_PROVEN"
        item["reason"] = "old/new log probabilities and Beta concentrations were not retained"
    result = {
        "status": "PASS" if local_ok and coverage_ok and optimizer_ok and len(rows) == 2048 else "FAIL",
        "records": len(rows), "updates": len(updates), "per_update_records": dict(Counter(r["update"] for r in rows)),
        "minibatch_ordinal_scope": "per_update", "per_update_ordinals_0_through_7": local_ok,
        "global_ordinal_0_through_2047": False,
        "global_ordinal_interpretation": "DIAGNOSTIC_ASSUMPTION_ERROR",
        "coverage_all_512_exactly_once": coverage_ok, "optimizer_cumulative_8_per_update": optimizer_ok,
    }
    return result, linked


def mapping() -> dict[str, Any]:
    errors = defaultdict(list); categories = Counter(); reset_distances = Counter()
    previous_velocity: list[list[float]] | None = None
    same_tracking, lag_tracking = [], []
    rows = samples = reset_count = 0
    since_reset = [0] * 32
    for row in lines(LOG / "compact_rollouts.jsonl"):
        rows += 1
        action = row["action_sample"]; command = row["command_b"]
        control = row["pre_reset"]["control"]
        actual = row["pre_reset"]["task"]["actual_velocity_b"]
        reset = row["pre_reset"]["reset_mask"]
        beta = control["ics_metrics"]["beta"]
        emergency = control["ics_metrics"]["emergency"]
        for lane in range(32):
            samples += 1
            alpha = [action[lane][0]]
            corr = [(2.0 * x - 1.0) * 0.5 for x in action[lane][1:]]
            gov = clip_norm([alpha[0] * command[lane][j] + corr[j] for j in range(3)])
            final = clip_norm([float(beta[lane][0]) * x for x in gov])
            errors["alpha"].append(max_abs(alpha, control["alpha"][lane]))
            errors["v_corr_b"].append(max_abs(corr, control["v_corr_b"][lane]))
            errors["v_gov_b"].append(max_abs(gov, control["v_gov_b"][lane]))
            errors["v_final_b"].append(max_abs(final, control["v_final_b"][lane]))
            same_tracking.append(max_abs(actual[lane], control["v_final_b"][lane]))
            if previous_velocity is not None:
                lag_tracking.append(max_abs(actual[lane], previous_velocity[lane]))
            label = "emergency" if bool(emergency[lane][0]) else ("intervened" if float(beta[lane][0]) < 1.0 else "normal")
            categories[label] += 1
            reset_distances[min(since_reset[lane], 16)] += 1
            if reset[lane]:
                since_reset[lane] = 0; reset_count += 1
            else:
                since_reset[lane] += 1
        previous_velocity = control["v_final_b"]
    return {"status": "PASS" if max(max(v) for v in errors.values()) < 2e-6 else "FAIL",
            "rows": rows, "lane_samples": samples, "reset_events": reset_count,
            "maximum_absolute_error": {k: max(v) for k, v in errors.items()},
            "ics_categories": dict(categories), "samples_by_steps_since_reset_capped_16": dict(reset_distances),
            "actual_velocity_vs_same_row_v_final_max_abs": {"mean": sum(same_tracking)/len(same_tracking), "max": max(same_tracking)},
            "actual_velocity_vs_previous_row_v_final_max_abs": {"mean": sum(lag_tracking)/len(lag_tracking), "max": max(lag_tracking)},
            "timing": "action_sample is computed before environment.step; logged pre_reset control/task are returned by that same step",
            "causal_limit": "The trace does not retain Lee/wrench/rotor signals; full downstream control causality is NOT_PROVEN."}


def schema_evidence() -> dict[str, Any]:
    files = [LOG / x for x in ("compact_rollouts.jsonl", "partial_rollout_steps.jsonl", "measurement_windows.jsonl", "minibatches.jsonl")]
    eval_files = sorted((EVAL / "evaluation_raw").glob("**/*.jsonl"))
    needles = [b'"lidar_grid"', b'"state_vec"', b'"old_log_probability"', b'"alpha"', b'"beta"']
    scans = {}
    for path in files + eval_files:
        data = path.read_bytes()
        scans[str(path)] = {n.decode(): data.count(n) for n in needles}
    first_measurement = next(lines(LOG / "measurement_windows.jsonl"))
    source = first_measurement["rows"][0]["source"]
    checkpoint = torch.load(TRAIN / "checkpoints/u0256.pt", map_location="cpu", weights_only=True)
    checkpoint_keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    complete = any(v['"lidar_grid"'] and v['"state_vec"'] for v in scans.values())
    return {"status": "FAIL" if complete else "NOT_PROVEN", "historical_actor_input_recoverable": complete,
            "byte_key_occurrences": scans, "measurement_window_kinds": sorted(source),
            "measurement_selected_lanes": first_measurement["rows"][0]["lanes"],
            "checkpoint_top_level_keys": checkpoint_keys,
            "reason": "No artifact jointly retains exact policy lidar_grid, state_vec/history-processing state, and probability state. Ray windows are partial sensor-source evidence only.",
            "synthetic_replay_used": False}


def fixtures() -> dict[str, Any]:
    sys.path.insert(0, str(PYROOT))
    from instinctlab.training.ppo import PPOConfig, PPOTrainer, compute_gae
    from instinctlab.training.networks import HistoryActor, NetworkConfig
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    from instinctlab.training.rollout import RolloutConfig
    from instinctlab.training.rollout_h1 import collect_h1_transition
    def t(vals): return torch.tensor(vals, dtype=torch.float32).reshape(-1, 1, 1)
    def b(vals): return torch.tensor(vals, dtype=torch.bool).reshape(-1, 1, 1)
    rewards=t([1,2,3]); values=t([.5,.7,1.1]); nxt=t([.7,1.1,2.0]); false=b([0,0,0])
    ordinary=compute_gae(rewards,values,nxt,false,false,false,timeout_bootstrap_values=t([0,0,0]),gamma=.9,gae_lambda=.8)
    expected_last=3+.9*2-1.1
    timeout=compute_gae(t([1]),t([.25]),t([99]),b([0]),b([1]),b([1]),timeout_bootstrap_values=t([4]),gamma=.9,gae_lambda=.8)
    both=compute_gae(t([1]),t([.25]),t([99]),b([1]),b([1]),b([1]),timeout_bootstrap_values=t([4]),gamma=.9,gae_lambda=.8)
    boundary=compute_gae(t([1,10]),t([0,0]),t([0,0]),b([0,0]),b([0,0]),b([1,0]),timeout_bootstrap_values=t([0,0]),gamma=1,gae_lambda=1)
    checks = {
        "nonzero_value_and_rollout_tail": abs(float(ordinary.advantages[-1]) - expected_last) < 1e-6,
        "pure_timeout_terminal_bootstrap": abs(float(timeout.advantages[0]) - (1+.9*4-.25)) < 1e-6,
        "terminated_precedes_truncated": abs(float(both.advantages[0]) - (1-.25)) < 1e-6,
        "reset_boundary_stops_recursion": abs(float(boundary.advantages[0]) - 1) < 1e-6,
    }
    torch.manual_seed(7)
    network=NetworkConfig(); actor=HistoryActor(network); critic=PrivilegedCriticH1(network)
    critic.network=torch.nn.Linear(21,1,bias=False)
    with torch.no_grad(): critic.network.weight.zero_(); critic.network.weight[0,0]=1
    trainer=PPOTrainer(actor,critic,PPOConfig(minibatch_size=1))
    policy={"lidar_grid":torch.zeros(1,12,59,360),"state_vec":torch.zeros(1,52)}
    initial={"policy":policy,"critic":torch.full((1,21),2.0)}
    returned={"policy":policy,"critic":torch.full((1,21),11.0)}
    terminal=torch.full((1,21),7.0)
    class Fake:
        def step(self, action):
            return returned, torch.ones(1), torch.tensor([False]), torch.tensor([True]), {"pre_reset":{"critic_h1":terminal}}
    transition,_=collect_h1_transition(Fake(),trainer,initial,RolloutConfig(horizon=1,batch_size=1))
    checks["terminal_observation_selected_for_timeout"] = abs(float(transition.timeout_value)-7) < 1e-6
    checks["returned_reset_observation_kept_as_next"] = abs(float(transition.next_value)-11) < 1e-6
    return {"status":"PASS" if all(checks.values()) else "FAIL", "checks":checks,
            "values":{"initial_value":float(transition.value),"timeout_terminal_value":float(transition.timeout_value),"returned_reset_next_value":float(transition.next_value)},
            "simulator_imported":False}


def main() -> None:
    mb, kl = minibatches()
    inputs = {str(p): sha(p) for p in [LOG/"updates.jsonl",LOG/"minibatches.jsonl",LOG/"compact_rollouts.jsonl",
             LOG/"partial_rollout_steps.jsonl",LOG/"measurement_windows.jsonl",TRAIN/"checkpoints/u0256.pt"]}
    result={"schema_version":SCHEMA,"inputs":inputs,"e0":{
        "minibatch_update_optimizer":mb,"early_kl_top10":kl,"action_control_mapping":mapping(),
        "actor_input_recoverability":schema_evidence(),"cpu_gae_terminal_fixtures":fixtures()}}
    checked = (mb, result["e0"]["action_control_mapping"],
               result["e0"]["actor_input_recoverability"], result["e0"]["cpu_gae_terminal_fixtures"])
    result["status"] = "PASS_WITH_NOT_PROVEN" if all(item["status"] in {"PASS","NOT_PROVEN"} for item in checked) else "FAIL"
    output=HERE/"r4_review_e0.json"
    output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(output),"sha256":sha(output),"status":result["status"]},sort_keys=True))


if __name__ == "__main__": main()
