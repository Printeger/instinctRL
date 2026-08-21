"""Pure-CUDA deterministic M4 observation/history smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instinctlab.observations import (  # noqa: E402
    RAY_COUNT,
    ActorObservationHistory,
    audit_actor_observation,
)


def tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def make_inputs(step: int, device: torch.device) -> dict[str, torch.Tensor]:
    batch = 2
    dtype = torch.float32
    ray_ids = torch.arange(RAY_COUNT, device=device)
    ranges = torch.linspace(0.1, 39.0, RAY_COUNT, device=device, dtype=dtype).repeat(batch, 1)
    ranges.add_(0.01 * step)
    env_ids = torch.arange(batch, device=device).reshape(batch, 1)
    validity = ((ray_ids.reshape(1, -1) + step + env_ids) % 7) != 0
    ranges = torch.where(validity, ranges, torch.full_like(ranges, 40.0))
    # Mixed authoritative mask fixture: identical 40 m serialization is valid
    # in env 0 and invalid in env 1 because M3 uses a closed valid interval.
    ranges[:, 0] = 40.0
    validity[0, 0] = True
    validity[1, 0] = False
    sensor = torch.tensor([[float(step)], [float(step)]], device=device, dtype=dtype)
    if step == 5:
        sensor[1, 0] = 0.5
    policy = sensor + torch.tensor([[0.0], [0.25]], device=device, dtype=dtype)
    angular = torch.tensor(
        [[0.1 * step, -0.2, 0.3], [-0.4, 0.05 * step, 0.6]], device=device, dtype=dtype
    )
    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    command = torch.tensor([[1.0, 0.0, -0.5], [-1.0, 0.25, 0.75]], device=device, dtype=dtype)
    previous = command * (0.1 * step)
    return {
        "ranges_m": ranges,
        "validity": validity,
        "sensor_frame_time_s": sensor,
        "policy_time_s": policy,
        "body_angular_velocity_rad_s": angular,
        "gravity_direction_body": gravity,
        "commanded_body_velocity_m_s": command,
        "previous_issued_body_velocity_m_s": previous,
    }


def run(output_path: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("M4 CUDA smoke requires an available CUDA device")
    device = torch.device("cuda:0")
    builder = ActorObservationHistory(tau_s=0.75)
    complete_outputs: list[dict[str, torch.Tensor]] = []
    step_records: list[dict[str, object]] = []
    for step in range(1, 7):
        if step == 5:
            builder.reset(torch.tensor([1], device=device, dtype=torch.long))
        observation = builder.update(**make_inputs(step, device))
        audit_actor_observation(observation)
        cpu_observation = {key: value.detach().cpu() for key, value in observation.items()}
        complete_outputs.append(cpu_observation)
        step_records.append(
            {
                "step": step,
                "lidar_grid_sha256": tensor_hash(observation["lidar_grid"]),
                "state_vec_sha256": tensor_hash(observation["state_vec"]),
                "finite": bool(
                    torch.isfinite(observation["lidar_grid"]).all().item()
                    and torch.isfinite(observation["state_vec"]).all().item()
                ),
                "lidar_nonzero": int(torch.count_nonzero(observation["lidar_grid"]).item()),
                "state_nonzero": int(torch.count_nonzero(observation["state_vec"]).item()),
            }
        )
    bundle_path = output_path.with_suffix(".pt")
    torch.save(complete_outputs, bundle_path)
    final = complete_outputs[-1]
    result: dict[str, object] = {
        "schema": "m4-observation-smoke-v1",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "environment_count": 2,
        "history_length": 4,
        "ray_count": RAY_COUNT,
        "steps": step_records,
        "final_shapes": {key: list(value.shape) for key, value in final.items()},
        "all_finite": all(bool(record["finite"]) for record in step_records),
        "rollover_exercised": True,
        "selective_reset_exercised": True,
    }
    output_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
