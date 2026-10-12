from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

diagnostic = importlib.import_module("training_workloads.unified_h1.control_diagnostic")


def parameters(checkpoint: Path, condition: str = "D7-R"):
    return {"condition": condition, "experiment_id": f"UNIFIED-H1-R4-{condition}",
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_source_identity": {"product_commit":"a"*40,"product_tree":"b"*40,
                "workload_commit":"c"*40,"workload_tree":"d"*40,"config_sha256":"e"*64},
            "cases":["Z","T","S"], "transitions":20160}


def test_exact_conditions_and_budget_binding(tmp_path):
    checkpoint=tmp_path/"u0256.pt"; checkpoint.write_bytes(b"frozen")
    for condition in diagnostic.CONDITIONS:
        assert diagnostic.validate_parameters(parameters(checkpoint, condition))["condition"] == condition
    wrong=parameters(checkpoint); wrong["transitions"]=20161
    with pytest.raises(PermissionError): diagnostic.validate_parameters(wrong)


def test_b0_is_fixed_normalized_action_through_h1_contract():
    action=diagnostic.fixed_b0_action(32)
    assert action.shape == (32,4)
    assert torch.equal(action[:,0],torch.ones(32))
    assert torch.equal(action[:,1:],torch.full((32,3),.5))


def test_actor_input_is_exact_lossless_selected_lanes():
    lidar=torch.arange(32*12*59*360,dtype=torch.float32).reshape(32,12,59,360)
    state=torch.arange(32*52,dtype=torch.float32).reshape(32,52)
    result=diagnostic.select_actor_input({"lidar_grid":lidar,"state_vec":state})
    assert result["lanes"].tolist() == list(range(8))
    assert torch.equal(result["lidar_grid"],lidar[:8].cpu())
    assert torch.equal(result["state_vec"],state[:8].cpu())


def test_chunk_writer_rejects_gaps_and_hashes_lossless_payload(tmp_path):
    writer=diagnostic.DiagnosticChunkWriter(tmp_path/"trace",condition="B0",experiment_id="UNIFIED-H1-R4-B0")
    payload={"issued_action":diagnostic.fixed_b0_action(1),"flag":torch.tensor([True])}
    for case in diagnostic.CASES:
        for step in range(diagnostic.CASE_STEPS[case]): writer.append(case,step,payload)
    manifest=writer.finalize()
    assert manifest["policy_steps"] == 630 and manifest["transitions"] == 20160
    first=tmp_path/"trace"/manifest["chunks"][0]["path"]
    assert hashlib.sha256(first.read_bytes()).hexdigest() == manifest["chunks"][0]["sha256"]
    restored=torch.load(first,map_location="cpu",weights_only=True)
    assert torch.equal(restored["issued_action"],payload["issued_action"])
