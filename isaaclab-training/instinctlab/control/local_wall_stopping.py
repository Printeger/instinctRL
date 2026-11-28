"""Check a finite braking path beside observed stationary surfaces."""

from dataclasses import replace
from functools import partial
import math

import numpy as np
import torch

from .command_adapter import rotate_body_to_world
from .escape_option import DURATION_STEPS
from .escape_passage import _return_seed, _supported_closing_returns, moving_intercepts, rotate_scan_rays
from .governor import clip_vector_norm
from .ics import AZIMUTH_SAMPLES, ELEVATION_SAMPLES, ICSResult, attenuate_command, mounted_mid360_rays_body
from .tube_stopping import (
    TubeStoppingDecision, TubeStoppingPipeline, _align_scans, _remaining_ratio, _surface_normals, _tube_planes,
)


_ANGLES = (0., -.15, .15, -.30, .30, -.50, .50, -.70, .70)
_NORMAL_BINS = np.linspace(-np.pi, np.pi, 181)
_DISTANCE_BINS = np.linspace(.25, 1.65, 71)
_HOLD_S = .60
_MAX_UNPAIRED_INWARD_PROJECTION = .50


def _local_planes(scans, history, rays, mounted, valid, up):
    """Associate first returns with vertical planes seen on three consecutive scans."""
    rotations, mounts, _ = _align_scans(torch.eye(3, dtype=history.dtype, device=history.device), history)
    axis = np.cross(up, np.array([0., 1., 0.]))
    axis /= np.linalg.norm(axis)
    across = np.cross(up, axis)
    prepared = []
    for scan, rotation, mount in zip(scans, rotations, mounts, strict=True):
        points, normals = _surface_normals(scan, rays)
        points = points @ rotation + mount
        normals = normals @ rotation
        nearby = (np.linalg.norm(points, axis=1) < 1.65) & (abs(normals @ up) < .07)
        prepared.append((points[nearby], normals[nearby]))

    points, normals = prepared[-1]
    azimuth = np.arctan2(normals @ across, normals @ axis)
    unused = np.ones(len(points), dtype=bool)
    planes = []
    for _ in range(24):
        counts, _ = np.histogram(azimuth[unused], bins=_NORMAL_BINS)
        peak = int(counts.argmax())
        if counts[peak] < 20:
            break
        angle = .5*(_NORMAL_BINS[peak]+_NORMAL_BINS[peak+1])
        angular_error = (azimuth-angle+np.pi)%(2.*np.pi)-np.pi
        cluster = unused & (abs(angular_error) <= .045)
        unused &= ~cluster
        normal = normals[cluster].mean(axis=0)
        normal -= (normal @ up)*up
        normal /= np.linalg.norm(normal)
        tangent = np.cross(up, normal)
        counts, _ = np.histogram(points[cluster] @ normal, bins=_DISTANCE_BINS)
        for cell in np.flatnonzero(counts >= 18):
            center = .5*(_DISTANCE_BINS[cell]+_DISTANCE_BINS[cell+1])
            distances = []
            for samples, directions in prepared:
                offsets = samples @ normal
                similar = (directions @ normal >= math.cos(.045)) & (abs(offsets-center) <= .055)
                if similar.sum() < 24:
                    break
                distance = float(np.median(offsets[similar]))
                plane = samples[similar & (abs(offsets-distance) <= .013)]
                if len(plane) < 24 or np.ptp(plane @ tangent) < .14 or np.ptp(plane @ up) < .20:
                    break
                if np.sqrt(np.mean((plane @ normal-distance)**2)) > .006:
                    break
                distances.append(distance)
            if len(distances) != 3 or abs(np.diff(distances)).max() > .045:
                continue
            associated = valid.all(axis=0) & (abs(mounted @ normal-np.asarray(distances)[:, None]) <= .018).all(axis=0)
            planes.append((normal, min(distances), associated))
    return planes


def _braking_capsule(points, valid, prior, heading, travel, tail):
    axial = prior @ heading
    lateral = np.linalg.norm(prior-axial[:, None]*heading, axis=1)
    lower = -.35*max(0., -axial.min())-.10*max(0., -axial[-1])
    upper = travel+.35*max(0., axial.max())+.10*max(0., axial[-1])
    required = .41+.015+.008+tail+.35*lateral.max()+.10*lateral[-1]
    along = points @ heading
    transverse = np.linalg.norm(points-along[..., None]*heading, axis=2)
    gap = float(np.hypot(along-along.clip(lower, upper), transverse).min(initial=np.inf, where=valid))
    return lower, upper, float(required), along, gap


def _oblique_facets(scans, history, rays, mounted, valid, up):
    """Refit persistent longitudinal facets in the current IMU-aligned frame."""
    candidates = _tube_planes(scans, history.detach().cpu().numpy(), rays)
    if not candidates:
        return []
    rotations, mounts, _ = _align_scans(torch.eye(3, dtype=history.dtype, device=history.device), history)
    prepared = []
    for scan, rotation, mount in zip(scans, rotations, mounts, strict=True):
        points, normals = _surface_normals(scan, rays)
        points = points @ rotation + mount
        nearby = np.linalg.norm(points, axis=1) < 1.65
        prepared.append((points[nearby], (normals @ rotation)[nearby]))

    planes = []
    for normal, estimates in candidates:
        if abs(normal @ up) <= .07:
            continue
        tangent = np.cross(up, normal)
        if np.linalg.norm(tangent) < 1.e-6:
            tangent = np.cross(np.array([0., 1., 0.]), normal)
        tangent /= np.linalg.norm(tangent)
        across = np.cross(normal, tangent)
        distances = []
        for (points, directions), estimate, mount in zip(prepared, estimates, mounts, strict=True):
            offsets = points @ normal
            centered = estimate + mount[0] @ normal
            similar = (directions @ normal >= math.cos(.045)) & (abs(offsets-centered) <= .055)
            if similar.sum() < 24:
                break
            distance = float(np.median(offsets[similar]))
            patch = points[similar & (abs(offsets-distance) <= .013)]
            if len(patch) < 24 or np.ptp(patch @ tangent) < .14 or np.ptp(patch @ across) < .20:
                break
            if np.sqrt(np.mean((patch @ normal-distance)**2)) > .006:
                break
            distances.append(distance)
        if len(distances) != 3 or abs(np.diff(distances)).max() > .045:
            continue
        associated = valid.all(axis=0) & (abs(mounted @ normal-np.asarray(distances)[:, None]) <= .018).all(axis=0)
        planes.append((normal, min(distances), associated))
    return planes


def _forward_coverage(valid, aligned, along, heading, up, upper, required):
    visible = valid & (aligned @ heading >= .99) & (along > upper+required+.025)
    bearing = aligned @ np.cross(up, heading)
    shape = (3, ELEVATION_SAMPLES, AZIMUTH_SAMPLES)
    return all(((visible & (sign*bearing > 0.)).reshape(shape).any(axis=1).sum(axis=1) >= 3).all()
               for sign in (-1., 1.))


def certify_stop_at_front_wall(lidar, state, rays, *, previous_speed_bound, require_closing, approach=None):
    """Attribute supported closing returns to the same observed wall before stopping."""
    history = state.reshape(4, 13)
    request = history[-1, 6:9] if approach is None else approach
    speed = float(request.norm())
    age = history[:, 12]
    if (not .03 < speed <= .30 or bool((age > .04).any() or (age.diff().abs() > .03).any()
            or history[-3:, :3].norm(dim=1).max() > .38)):
        return None
    rays_fresh = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    elapsed = age.new_tensor([-.3, -.2, -.1, 0.])-age+age[-1]
    supported = _supported_closing_returns(lidar[:, 0::3], rays_fresh, elapsed[None], maximum_range_m=2.65)[0]
    closing = (supported > .005).detach().cpu().numpy()
    if require_closing and not closing.any():
        return None
    moving, *_ = moving_intercepts(lidar, state, rays, torch.ones(1, dtype=torch.bool, device=state.device),
                                  maximum_range_m=2.65)
    if len(moving):
        return None

    up = (-history[-1, 3:6]).detach().cpu().numpy()
    if abs(up[1]) > .80:
        return None
    scans = lidar.reshape(4, 3, -1)[-3:].detach().cpu().numpy()
    aligned, mounts, prior = _align_scans(rays, history)
    valid = (scans[:, 1] > .5) & (scans[:, 2] >= .65)
    points = scans[:, 0, :, None]*aligned+mounts
    heading = request.detach().cpu().numpy()/speed
    tail = .35*max(.30, previous_speed_bound)*math.exp(-.30/.35)
    lower, upper, required, along, gap = _braking_capsule(points, valid, prior, heading, 0., tail)
    if gap < required or not _forward_coverage(valid, aligned, along, heading, up, upper, required):
        return None
    planes = _local_planes(scans, history, rays.detach().cpu().numpy(), points, valid, up)
    for normal, distance, associated in planes:
        forward = float(heading @ normal)
        if (forward >= .70 and distance-max(lower*forward, upper*forward) >= required
                and int(associated.sum()) >= 24 and not (closing & ~associated).any()):
            return gap, required, int(closing.sum())
    return None


def _opposite_wall_pair(first, first_distance, second, second_distance):
    return first @ second <= -math.cos(.04) and 1.65 <= first_distance+second_distance <= 2.35


def _certify_associated_request(trial, planes, config, *, facets=(), require_supported_approach=False):
    result, proposed, heading, lower, upper, gap, required, projection, request_speed = trial
    if not planes and not facets:
        return None
    explained = np.zeros(result.cache['active_mask'].shape[1], dtype=bool)
    eligible = []
    for index, (normal, distance, associated) in enumerate((*planes, *facets)):
        advance = heading @ normal
        if distance-max(lower*advance, upper*advance) >= required:
            eligible.append((normal, distance, associated, advance, index >= len(planes)))
    for normal, distance, associated, advance, is_facet in eligible:
        # A finite capsule can be renewed while approaching the same lone wall.
        # Keep its radial constraint unless there is signed support on both sides
        # or the surface belongs to a separately verified enclosing tube.
        if require_supported_approach and not is_facet and advance > _MAX_UNPAIRED_INWARD_PROJECTION:
            if not any(_opposite_wall_pair(normal, distance, other, other_distance)
                       and np.count_nonzero(support) >= 24
                       for other, other_distance, support, _, _ in eligible):
                continue
        explained |= associated
    closing, ratios = _remaining_ratio(result.cache, config)
    active = result.cache['active_mask']
    matched = torch.as_tensor(explained, device=proposed.device)[None]
    cleared = int((active & matched & (ratios < .90)).sum())
    remaining = active & ~matched
    count = remaining.sum(dim=1, keepdim=True)
    beta = ratios.masked_fill(~remaining, 1.).amin(dim=1, keepdim=True)
    beta = torch.where(count > 0, beta, torch.full_like(beta, config.empty_active_set_beta))
    if not cleared or float(beta[0, 0]) < .90 or float(beta[0, 0])*projection < .60*request_speed**2:
        return None
    margin = (result.cache['v_safe_m_s']-closing).masked_fill(~remaining, torch.inf).amin(dim=1, keepdim=True)
    metrics = dict(result.metrics, beta=beta, active_beam_count=count.to(proposed.dtype),
                   worst_margin_m_s=torch.where(count > 0, margin, 0.))
    certified = ICSResult(clip_vector_norm(proposed*beta, config.velocity_limit_m_s), metrics,
                          dict(result.cache, active_mask=remaining))
    return TubeStoppingDecision(certified, proposed, _HOLD_S, gap, required, cleared, bool(facets))


def certify_local_wall_request(lidar, state, rays, governed, config, baseline, *, previous_speed_bound,
                               allow_oblique_facets=False, require_supported_approach=False):
    """Re-evaluate a bounded heading without removing unidentified restrictions."""
    history = state.reshape(4, 13)
    request = history[-1, 6:9]
    speed = float(request.norm())
    governed_speed = float(governed[0].norm())
    if (not .03 < speed <= .30 or not .005 < governed_speed <= .30
            or bool(baseline.metrics['emergency'][0, 0]) or float(baseline.v_final_b[0] @ request) >= .60*speed**2):
        return None
    age = history[:, 12]
    if bool((age > .04).any() or (age.diff().abs() > .03).any() or history[-3:, :3].norm(dim=1).max() > .38):
        return None
    if bool(_return_seed(lidar, state, maximum_range_m=2.65)[0][0]):
        return None

    array = lambda value: value.detach().cpu().numpy()
    up = array(-history[-1, 3:6])
    if abs(up[1]) > .80:
        return None
    scans = array(lidar.reshape(4, 3, -1)[-3:])
    aligned, mounts, prior = _align_scans(rays, history)
    valid = (scans[:, 1] > .5) & (scans[:, 2] >= .65)
    points = scans[:, 0, :, None]*aligned+mounts
    velocity = array(governed[0])
    vertical = (velocity @ up)*up
    forward = velocity-vertical
    side = np.cross(up, forward)
    intent = array(request)
    tail = .35*max(.30, previous_speed_bound)*math.exp(-.30/.35)
    planes = None
    blocked_trials = []
    for angle in _ANGLES:
        target = math.cos(angle)*forward+math.sin(angle)*side+vertical
        projection = float(target @ intent)
        magnitude = float(np.linalg.norm(target))
        if projection < .60*speed**2 or magnitude > .30:
            continue
        heading = target/magnitude
        lower, upper, required, along, gap = _braking_capsule(points, valid, prior, heading, _HOLD_S*magnitude, tail)
        if gap < required or not _forward_coverage(valid, aligned, along, heading, up, upper, required):
            continue

        proposed = governed.new_tensor(target)[None]
        result = attenuate_command(lidar, rays, proposed, .10, config, state_vec=state)
        if bool(result.metrics['emergency'][0, 0]):
            continue
        if float(result.v_final_b[0] @ request) >= .60*speed**2:
            return TubeStoppingDecision(result, proposed, _HOLD_S, gap, float(required), 0)
        if planes is None:
            planes = _local_planes(scans, history, array(rays), points, valid, up)
        trial = (result, proposed, heading, lower, upper, gap, float(required), projection, speed)
        decision = _certify_associated_request(trial, planes, config,
                                                require_supported_approach=require_supported_approach)
        if decision is not None:
            return decision
        if allow_oblique_facets:
            blocked_trials.append(trial)
    if blocked_trials:
        facets = _oblique_facets(scans, history, array(rays), points, valid, up)
        if facets:
            for trial in blocked_trials:
                decision = _certify_associated_request(trial, planes, config, facets=facets,
                                                        require_supported_approach=require_supported_approach)
                if decision is not None:
                    return decision
    return None


class LocalWallStoppingPipeline(TubeStoppingPipeline):
    certify = staticmethod(certify_local_wall_request)


class FrontWallBrakingPipeline(LocalWallStoppingPipeline):
    """Use a newly verified stop when a static front wall triggers an opposing option."""

    def __init__(self, baseline):
        super().__init__(baseline)
        self.allow_front_wall_brake = False
        self.front_wall_brake = None
        self._front_wall_ticks = 0

    def __call__(self, lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                 explicit_reset, controller_quaternion_wxyz):
        last_local_motion = self.tube_stopping is not None
        result = super().__call__(lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                                  explicit_reset, controller_quaternion_wxyz)
        self.front_wall_brake = None
        if (not self.allow_front_wall_brake or bool(episode_reset[0] or explicit_reset[0])
                or not math.isclose(history_dt_s, .10, abs_tol=1.e-6)):
            self._front_wall_ticks = 0
            return result
        continuing = self._front_wall_ticks > 0
        self._front_wall_ticks = max(0, self._front_wall_ticks-1)
        request = state_vec[0, 45:48]
        if (not (last_local_motion or continuing) or bool(result.ics_metrics['emergency'][0, 0])
                or float(result.v_final_b[0] @ request) >= -.08*float(request.norm())):
            return result
        rays = self.lidar_rays_body
        if rays is None:
            rays = mounted_mid360_rays_body(device=lidar_grid.device, dtype=lidar_grid.dtype)
        proof = certify_stop_at_front_wall(lidar_grid, state_vec, rays,
            previous_speed_bound=self._speed_at_oldest_scan, require_closing=not continuing)
        if proof is None:
            return result
        neutral = torch.zeros_like(result.v_final_b)
        checked = attenuate_command(lidar_grid, rays, neutral, history_dt_s, self.config.ics, state_vec=state_vec)
        if bool(checked.metrics['emergency'][0, 0]):
            return result
        if not continuing:
            self._front_wall_ticks = DURATION_STEPS-1
        self.front_wall_brake = (*proof, result.v_final_b)
        self._last_ics_cache = checked.cache
        world = rotate_body_to_world(checked.v_final_b, controller_quaternion_wxyz)
        return replace(result, v_final_b=checked.v_final_b, v_final_w=world, ics_metrics=checked.metrics)


class FacetedWallBrakingPipeline(FrontWallBrakingPipeline):
    certify = staticmethod(partial(certify_local_wall_request, allow_oblique_facets=True))


def _signed_edge_clearance(lidar, history, rays, target, oldest_speed, cache):
    scans = lidar.reshape(4, 3, -1)[-3:].detach().cpu().numpy()
    up = -history[-1, 3:6].detach().cpu().numpy()
    if abs(up[1]) > .80:
        return None
    aligned, mounts, prior = _align_scans(rays, history)
    valid = (scans[:, 1] > .5) & (scans[:, 2] >= .65)
    points = scans[:, 0, :, None]*aligned+mounts
    tail = .35*max(.30, oldest_speed)*math.exp(-.30/.35)
    velocity = target.detach().cpu().numpy()
    speed = np.linalg.norm(velocity)
    _, _, required, _, gap = _braking_capsule(points, valid, prior, velocity/speed, _HOLD_S*speed, tail)
    if gap < required:
        return None

    planes = _local_planes(scans, history, rays.detach().cpu().numpy(), points, valid, up)
    walls = []
    explained = np.zeros(len(rays), dtype=bool)
    for normal, distance, associated in planes:
        inward = prior @ normal
        rest = distance-.015-.008-tail-.35*max(0., inward.max())-.10*max(0., inward[-1])
        rest -= _HOLD_S*max(0., velocity @ normal)
        if rest >= .41 and int(associated.sum()) >= 24:
            walls.append((normal, distance, float(rest)))
            explained |= associated
    paired = [min(first_rest, second_rest) for i,(first, first_distance, first_rest) in enumerate(walls)
              for second, second_distance, second_rest in walls[i+1:]
              if _opposite_wall_pair(first, first_distance, second, second_distance)]
    if not paired:
        return None
    hard = (cache['active_mask'][0] & (cache['v_safe_m_s'][0] <= 1.e-5)).detach().cpu().numpy()
    if (hard & ~explained).any():
        return None
    return gap, required, max(paired)


class WallEdgeReproofPipeline(FrontWallBrakingPipeline):
    """Recheck a recently observed edge command against current walls and full ICS."""

    def __init__(self, baseline):
        super().__init__(baseline)
        self.fresh_edge = False
        self.other_detour = False
        self.edge_reproof = None
        self._edge = None

    def __call__(self, lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                 explicit_reset, controller_quaternion_wxyz):
        result = super().__call__(lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                                  explicit_reset, controller_quaternion_wxyz)
        anchor = self._edge
        self._edge = None
        self.edge_reproof = None
        if (not self.allow_tube_stopping or self.tube_stopping is not None or self.front_wall_brake is not None
                or self.other_detour or bool(episode_reset[0] or explicit_reset[0])
                or not math.isclose(history_dt_s, .10, abs_tol=1.e-6)):
            return result
        history = state_vec.reshape(4, 13)
        request = history[-1, 6:9]
        speed = float(request.norm())
        if not .10 <= speed <= .30 or bool(result.ics_metrics['emergency'][0, 0]):
            return result
        beta = float(result.ics_metrics['beta'][0, 0])
        if self.fresh_edge:
            target = result.v_final_b[0]
            if (0. < float(target.norm()) <= .30 and beta >= .99
                    and float(target @ request) >= .60*speed**2):
                self._edge = target.detach().clone(), request.detach().clone(), 0
            return result
        if anchor is None or anchor[2] >= 3 or beta > .25:
            return result
        ages = history[:, 12]
        if bool((ages > .04).any() or (ages.diff().abs() > .03).any()
                or history[-3:, :3].norm(dim=1).max() > .38):
            return result
        rotation = -.05*(history[-2, :3]+history[-1, :3])
        target, prior_request = rotate_scan_rays(torch.stack(anchor[:2]), rotation)
        if float((request-prior_request).norm()) > .012:
            return result

        rays = self.lidar_rays_body
        if rays is None:
            rays = mounted_mid360_rays_body(device=lidar_grid.device, dtype=lidar_grid.dtype)
        if bool(_return_seed(lidar_grid, state_vec, maximum_range_m=2.65)[0][0]):
            return result
        checked = attenuate_command(lidar_grid, rays, target[None], history_dt_s, self.config.ics, state_vec=state_vec)
        if (bool(checked.metrics['emergency'][0, 0]) or float(checked.metrics['beta'][0, 0]) < .99
                or float(checked.v_final_b[0] @ request) < .60*speed**2):
            return result
        proof = _signed_edge_clearance(lidar_grid, history, rays, target, self._speed_at_oldest_scan,
                                       self._last_ics_cache)
        if proof is None:
            return result
        elapsed = anchor[2]+1
        self._edge = target, prior_request, elapsed
        self.edge_reproof = elapsed, *proof
        self._last_ics_cache = checked.cache
        world = rotate_body_to_world(checked.v_final_b, controller_quaternion_wxyz)
        return replace(result, v_final_b=checked.v_final_b, v_final_w=world, ics_metrics=checked.metrics)


class FacetedWallEdgeReproofPipeline(WallEdgeReproofPipeline):
    certify = staticmethod(FacetedWallBrakingPipeline.certify)


class SupportedWallEdgeReproofPipeline(FacetedWallEdgeReproofPipeline):
    certify = staticmethod(partial(certify_local_wall_request, allow_oblique_facets=True,
                                   require_supported_approach=True))
