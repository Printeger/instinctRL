"""Shadow-test direction-selective command projection through the unchanged ICS."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
from instinct_tube_probe import body_to_world, tube_observation


def project_command(command, rays, limits, steps, tolerance):
    """Project away the worst violation repeatedly; the original ICS checks the result."""
    if not len(limits):
        return command, 0
    value = command.clone()
    for step in range(steps):
        violations = rays @ value - limits
        index = violations.argmax()
        if violations[index].item() <= tolerance:
            return value, step
        value -= violations[index] * rays[index]
    return value, steps


def endcap_at(fixture, time):
    if "endcap_m" in fixture:
        return fixture["endcap_m"] + fixture.get("endcap_velocity_m_s", 0.0) * time
    return None


def scene_scan(rays, rotation, fixture, time):
    ranges, valid = tube_observation(rays, rotation, fixture["half_width_m"],
                                     endcap_at(fixture, time), fixture.get("offset_y_m", 0.0))
    rays_world = rays @ rotation.T
    x_ray, y_ray = rays_world[:, 0], rays_world[:, 1]
    if "doorway" in fixture:
        door = fixture["doorway"]
        depth = door["distance_m"] / x_ray
        hits = (x_ray > 1e-8) & ((depth * y_ray).abs() >= door["half_opening_m"])
        ranges = torch.minimum(ranges, torch.where(hits, depth, torch.inf))
    for x, y, radius in fixture.get("posts", []):
        a = x_ray.square() + y_ray.square()
        b = x * x_ray + y * y_ray
        determinant = b.square() - a * (x * x + y * y - radius * radius)
        depth = (b - determinant.clamp_min(0).sqrt()) / a
        hits = (determinant >= 0) & (a > 1e-8) & (depth >= 0.1)
        ranges = torch.minimum(ranges, torch.where(hits, depth, torch.inf))
    # A hit at the exact fill distance can be valid in the original scan.
    valid |= ranges < 40.0
    if "drop_forward_cone_cosine" in fixture:
        valid &= x_ray < fixture["drop_forward_cone_cosine"]
    return torch.where(valid, ranges, 40.0), valid


def linear_clearance(fixture, velocity, dt):
    """Score translation at fixed orientation. This is fixture truth, not flight dynamics."""
    x, y, z = (velocity * dt).tolist()
    width, offset, endcap = fixture["half_width_m"], fixture.get("offset_y_m", 0.0), endcap_at(fixture, dt)
    distances = [width - abs(offset + y), width - abs(z)]
    if endcap is not None:
        distances.append(endcap - x)
    if "doorway" in fixture:
        door = fixture["doorway"]
        distances.append(math.hypot(door["distance_m"] - x, max(door["half_opening_m"] - abs(y), 0.0)))
    distances.extend(math.hypot(cx - x, cy - y) - radius for cx, cy, radius in fixture.get("posts", []))
    return min(distances)


def comparison(rays, config, inputs, fixture, yaw):
    rotation = body_to_world(yaw, rays)
    dt = inputs["history_dt_s"]
    frames = [scene_scan(rays, rotation, fixture, index * dt) for index in (-3, -2, -1, 0)]
    _, valid = frames[-1]
    history = torch.cat([torch.stack((ranges, mask, mask)).reshape(3, 59, 360)
                         for ranges, mask in frames])[None]
    command_world = rays.new_tensor(inputs["command_world_m_s"])
    command_body = command_world @ rotation
    baseline = attenuate_command(history, rays, command_body[None], dt, config)
    baseline_world = baseline.v_final_b[0] @ rotation.T
    effective = baseline.cache["effective_clearance_m"][0]
    available = valid.float().mean().item()
    result = {
        "fixture": fixture["name"],
        "yaw_degrees": yaw,
        "geometric_clearance_m": linear_clearance(fixture, command_world, 0.0),
        "valid_fraction": available,
        "baseline_world_m_s": baseline_world.tolist(),
        "baseline_command_error_m_s": torch.linalg.vector_norm(baseline_world - command_world).item(),
        "baseline_linear_clearance_m": linear_clearance(fixture, baseline_world, dt),
    }
    if available < inputs["min_reliable_fraction"]:
        return {**result, "status": "abstained_low_measurement_coverage"}

    considered = valid & (effective <= config.d_safe_m + config.active_horizon_margin_m)
    active_rays = rays[considered]
    limits = baseline.cache["v_safe_m_s"][0][considered]
    proposed, iterations = project_command(command_body, active_rays, limits,
                                           inputs["projection_limit"], inputs["projection_tolerance_m_s"])
    candidate = attenuate_command(history, rays, proposed[None], dt, config)
    issued = candidate.v_final_b[0]
    max_violation = max(0.0, (active_rays @ issued - limits).max().item()) if len(limits) else 0.0
    if max_violation > inputs["projection_tolerance_m_s"]:
        raise RuntimeError("final shadow output violates the same fixed-observation ICS inequalities")
    candidate_world = issued @ rotation.T
    axial_gain = candidate_world[0].item() - result["baseline_world_m_s"][0]
    return {
        **result,
        "status": "compared",
        "projection_iterations": iterations,
        "candidate_world_m_s": candidate_world.tolist(),
        "candidate_command_error_m_s": torch.linalg.vector_norm(candidate_world - command_world).item(),
        "candidate_linear_clearance_m": linear_clearance(fixture, candidate_world, dt),
        "candidate_beta": candidate.metrics["beta"].item(),
        "axial_gain_m_s": axial_gain,
        "axial_gain_at_least_threshold": axial_gain >= inputs["minimum_axial_gain_m_s"],
        "maximum_same_guard_violation_m_s": max_violation,
    }


@torch.inference_mode()
def evaluate(inputs):
    config = ICSConfig(clearance_margin_m=inputs["ics_clearance_margin_m"])
    # This projection uses frozen speed-independent halfspaces. The final guard is always rerun.
    if config.latency_s != 0.0 or config.range_rate_filter:
        raise ValueError("shadow check requires the R6 speed-independent ray constraints")
    rays = mounted_mid360_rays_body(device="cpu", dtype=torch.float32)
    return [comparison(rays, config, inputs, fixture, yaw)
            for fixture in inputs["fixtures"] for yaw in inputs["yaw_degrees"]]


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
