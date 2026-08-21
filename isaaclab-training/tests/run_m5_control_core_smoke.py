"""Persist complete deterministic CUDA M5 outputs and internal snapshots."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from instinctlab.control.pipeline import ControlCorePipeline


def actor_fixture(device: torch.device):
    lidar = torch.empty(2, 12, 59, 360, device=device)
    lidar[:, 0::3] = 40.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(2, 52, device=device)
    state[:, 5::13] = 1.0
    return lidar, state


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M5 smoke")
    device = torch.device("cuda:0")
    torch.manual_seed(5001)
    torch.cuda.manual_seed_all(5001)
    lidar, state = actor_fixture(device)
    rays_local_index = 7 * 360 + 180
    row, column = divmod(rays_local_index, 360)
    reset = torch.zeros(2, dtype=torch.bool, device=device)
    pipeline = ControlCorePipeline()
    outputs = {}

    action = torch.tensor([[1.0, 0.5, 0.5, 0.5], [0.5, 0.6, 0.4, 0.5]], device=device)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.70710677, 0.0, 0.0, 0.70710677]], device=device)
    outputs["capture_empty"] = asdict(pipeline(lidar, state, action, 0.1, reset, reset, quaternion))
    outputs["capture_empty_internal"] = pipeline.internal_snapshot()

    state[:, -7:-4] = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device)
    lidar[0, 0::3, row, column] = 1.0
    lidar[1, 0::3, row, column] = 0.2
    outputs["active_emergency"] = asdict(pipeline(lidar, state, action, 0.1, reset, reset, quaternion))
    outputs["active_emergency_internal"] = pipeline.internal_snapshot()

    explicit = torch.tensor([True, False], device=device)
    quaternion = torch.tensor([[0.9238795, 0.38268343, 0.0, 0.0], [0.9659258, 0.0, 0.25881904, 0.0]], device=device)
    outputs["reset_rotations"] = asdict(pipeline(lidar, state, action, 0.1, reset, explicit, quaternion))
    outputs["reset_rotations_internal"] = pipeline.internal_snapshot()
    torch.save(cpu_tree(outputs), args.output)
    print("M5 CUDA smoke PASS: environments=2 scenarios=3 complete_tensors=true")


if __name__ == "__main__":
    main()
