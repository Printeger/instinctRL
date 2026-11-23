"""CPU-only probe of fixed-observation command authority in an ideal square tube."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
from instinctlab.task.rewards_h1 import compute_command_risk


def body_to_world(yaw_degrees, reference):
    yaw = math.radians(yaw_degrees)
    c, s = math.cos(yaw), math.sin(yaw)
    return reference.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tube_observation(rays, body_rotation, half_width, endcap=None, offset_y=0.0):
    """Return one ideal scan. Plane definitions are fixture inputs, not policy inputs."""
    normals = rays.new_tensor([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0]])
    distances = [half_width - offset_y, half_width + offset_y, half_width, half_width]
    if endcap is not None:
        distances.append(endcap)
    normals_body = normals[:len(distances)] @ body_rotation
    denominators = rays @ normals_body.T
    paths = torch.where(denominators > 1e-8, rays.new_tensor(distances) / denominators, torch.inf)
    ranges, _ = paths.min(dim=-1)
    valid = (ranges >= 0.1) & (ranges <= 40.0)
    return torch.where(valid, ranges, 40.0), valid


def lidar_history(ranges, valid, batch_size):
    """Use four identical scans, as for constant axial motion in a uniform tube."""
    frame = torch.stack((ranges, valid, valid)).reshape(3, 59, 360)
    return frame.repeat(batch_size, 4, 1, 1)


def blocking_cone(rays_world, zero_speed):
    """Test if positive-x zero-limit rays can cancel their transverse components."""
    selected = rays_world[zero_speed & (rays_world[:, 0] > 1e-6)]
    if len(selected) < 3:
        return {"forward_zero_speed_rays": len(selected), "positive_axial_blocked": False}
    angles = torch.atan2(selected[:, 2], selected[:, 1]).sort().values
    largest_gap = torch.diff(torch.cat((angles, angles[:1] + 2 * math.pi))).max().item()
    return {
        "forward_zero_speed_rays": len(selected),
        "maximum_transverse_gap_deg": math.degrees(largest_gap),
        "positive_axial_blocked": largest_gap < math.pi - 1e-6,
    }


def probe_case(rays, config, inputs, half_width, yaw, endcap):
    rotation = body_to_world(yaw, rays)
    ranges, valid = tube_observation(rays, rotation, half_width, endcap)
    forward = inputs["axial_command_m_s"]
    sideways = inputs["lateral_command_m_s"]
    commands = rays.new_tensor([[forward, 0, 0], [forward, sideways, 0], [0, sideways, 0]])
    history = lidar_history(ranges, valid, len(commands))
    result = attenuate_command(history, rays, commands @ rotation, inputs["history_dt_s"], config)
    reward_risk = compute_command_risk(history, commands @ rotation, rays_body=rays)
    velocity_world = result.v_final_b @ rotation.T
    zero_speed = valid & (result.cache["v_safe_m_s"][0] == 0)
    cone = blocking_cone(rays @ rotation.T, zero_speed)
    if cone["positive_axial_blocked"] and velocity_world[:, 0].max().item() > 1e-6:
        raise RuntimeError("observed output contradicts the fixed-observation certificate")
    return {
        "half_width_m": half_width,
        "yaw_degrees": yaw,
        "endcap_m": endcap,
        "geometric_clearance_m": min(half_width, endcap) if endcap is not None else half_width,
        "minimum_return_m": ranges.min().item(),
        "ics_effective_minimum_m": result.metrics["minimum_clearance_m"][0, 0].item(),
        "command_order": ["axial", "oblique", "toward_side_wall"],
        "issued_world_m_s": velocity_world.tolist(),
        "beta": result.metrics["beta"].flatten().tolist(),
        "reward_marks_dangerous": reward_risk.dangerous.flatten().tolist(),
        "certificate": cone,
    }


@torch.inference_mode()
def evaluate(inputs):
    rays = mounted_mid360_rays_body(device="cpu", dtype=torch.float32)
    config = ICSConfig(clearance_margin_m=inputs["ics_clearance_margin_m"])
    cases = [probe_case(rays, config, inputs, width, yaw, None)
             for width in inputs["half_widths_m"] for yaw in inputs["yaw_degrees"]]
    cases.extend(probe_case(rays, config, inputs, inputs["half_widths_m"][-1], yaw, endcap)
                 for endcap in inputs["endcap_distances_m"] for yaw in inputs["yaw_degrees"])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = args.inputs.read_bytes()
    inputs = json.loads(raw)
    print(json.dumps({
        "source_commit": args.source_commit,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "torch": torch.__version__,
        "device": "cpu",
        "inputs": inputs,
        "cases": evaluate(inputs),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
