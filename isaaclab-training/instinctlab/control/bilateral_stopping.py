"""Identify observed wall constraints that leave room for a finite motion and brake."""

from dataclasses import replace

import torch
from torch.nn import functional as F

from .command_adapter import rotate_body_to_world
from .escape_passage import moving_intercepts
from .governor import clip_vector_norm
from .ics import AZIMUTH_SAMPLES, ICSResult, _fresh_history, mounted_mid360_rays_body
from .pipeline import ControlCorePipeline


_LONGITUDINAL_BANDS = ((-1.15, -.70), (-.70, -.25), (-.25, .25), (.25, .70), (.70, 1.15))


def _plane_normal(points):
    centered = points-points.mean(dim=0)
    variance, basis = torch.linalg.eigh(centered.T@centered/(len(points)-1))
    return basis[:, 0], variance[0].clamp_min(0.).sqrt()


def _wall_distances(points, valid, tangent, normal, up, azimuth, elevation):
    along = points@tangent
    across = points@normal
    at_height = valid & ((points@up).abs() <= .32)
    both = []
    for sign in (1., -1.):
        signed = sign*across
        nearby = at_height & (along.abs() < 1.22) & (signed >= .35) & (signed <= 1.65)
        if int(nearby.sum()) < 20:
            return None
        bins = ((signed[nearby]-.35)/.015).floor().long().clamp(0, 86)
        histogram = torch.zeros(87, device=points.device, dtype=torch.long)
        histogram.scatter_add_(0, bins, torch.ones_like(bins))
        center = .35+.015*(histogram.argmax()+.5)
        local = signed[nearby & ((signed-center).abs() <= .035)]
        if len(local) < 20:
            return None
        distance = local.median()
        wall = at_height & ((signed-distance).abs() <= .018)
        fit = points[wall & (along.abs() < 1.15)]
        if len(fit) < 20:
            return None
        fit_normal, fit_rms = _plane_normal(fit)
        if bool(fit_rms > .018 or torch.cross(fit_normal, normal, dim=0).norm() > .020):
            return None
        for lower, upper in _LONGITUDINAL_BANDS:
            band = wall & (along >= lower) & (along < upper)
            if len(azimuth[band].unique()) < 5 or len(elevation[band].unique()) < 2:
                return None
        both.append(distance)
    return torch.stack(both)


def clear_bilateral_constraints(lidar, state, rays, governed, active, ratios, interval):
    """Return only blockers associated with both observed walls on a clear braking path."""
    cleared = torch.zeros_like(active)
    history = state.reshape(-1, 4, 13)
    fresh = _fresh_history(state, governed, interval)[:, 0]
    request = history[:, -1, 6:9]
    request_speed = request.norm(dim=1)
    target_speed = governed.norm(dim=1)
    blocked = active & (ratios < 1.e-5)
    follows_request = (request*governed).sum(dim=1) >= (.60/.90)*request_speed.square()
    eligible = fresh & follows_request & (request_speed > .03) & (request_speed <= .30) & (target_speed <= .30)
    candidates = (eligible & blocked.any(dim=1)).nonzero().flatten()
    if not len(candidates):
        return cleared
    closing, *_ = moving_intercepts(lidar[candidates], state[candidates], rays, eligible[candidates])
    moving = torch.zeros_like(candidates, dtype=torch.bool)
    moving[closing] = True
    beam = torch.arange(len(rays), device=rays.device)
    azimuth, elevation = beam % AZIMUTH_SAMPLES, beam // AZIMUTH_SAMPLES
    for lane in candidates[~moving]:
        scan = lidar[lane].reshape(4, 3, -1)[-3:]
        ranges = scan[:, 0]
        valid = (scan[:, 1] > .5) & (scan[:, 2] >= .65)
        points = ranges[..., None]*rays
        up = -history[lane, -1, 3:6]
        anchor = points[-1, ranges[-1].masked_fill(~blocked[lane], torch.inf).argmin()]
        patch = points[-1, valid[-1] & ((points[-1]-anchor).norm(dim=1) <= .47)]
        if len(patch) < 20:
            continue
        normal, rms = _plane_normal(patch)
        tilt = normal@up
        if bool(rms > .018 or tilt.abs() > .075):
            continue
        normal = F.normalize(normal-tilt*up, dim=0)
        normal = normal*torch.sign(anchor@normal)
        tangent = torch.cross(up, normal, dim=0)
        distances = [_wall_distances(frame, reliable, tangent, normal, up, azimuth, elevation)
                     for frame, reliable in zip(points, valid, strict=True)]
        if any(distance is None for distance in distances):
            continue
        distances = torch.stack(distances)
        width = distances.sum(dim=1)
        if bool(width[-1] < 1.65 or width[-1] > 2.35 or width.diff().abs().amax() > .025
                or distances.diff(dim=0).abs().amax() > .055):
            continue
        walls = torch.stack((normal, -normal))
        recent = history[lane, :, 9:12]@walls.T
        coasting = .35*recent.amax(dim=0).clamp_min(0.)+interval*recent[-1].clamp_min(0.)
        toward = (walls@governed[lane]).clamp_min(0.)
        mount_uncertainty = .05*normal[2].abs()
        stopping_floor = distances.amin(dim=0)-.015-mount_uncertainty-coasting-.60*toward
        if bool((stopping_floor < .45).any()):
            continue
        across = rays@normal
        first_plane = torch.where(across > 0., distances[:, :1], -distances[:, 1:])
        fit = valid.all(dim=0) & ((ranges*across-first_plane).abs() <= .018).all(dim=0) & (across.abs() >= .30)
        if bool((fit & blocked[lane]).any() and ratios[lane].masked_fill(~active[lane] | fit, 1.).amin() >= .90):
            cleared[lane] = fit & active[lane]
    return cleared


def apply_bilateral_stopping(lidar, state, rays, governed, interval, config, baseline):
    """Recompute the final bound after removing only certified wall constraints."""
    cache = baseline.cache
    active = cache['active_mask']
    closing = cache['approach_speed_m_s']
    if config.range_rate_filter:
        closing = torch.maximum(closing, (-cache['range_rate_m_s']).clamp_min(0.))
    safe = cache['v_safe_m_s']
    ratios = (safe/closing.clamp_min(torch.finfo(governed.dtype).tiny)).clamp(0., 1.)
    emergency = baseline.metrics['emergency'] != 0.
    cleared = clear_bilateral_constraints(lidar, state, rays, governed, active & ~emergency, ratios, interval)
    changed = cleared.any(dim=1, keepdim=True)
    if not bool(changed.any()):
        return baseline
    remaining = active & ~cleared
    count = remaining.sum(dim=1, keepdim=True)
    beta = ratios.masked_fill(~remaining, 1.).amin(dim=1, keepdim=True)
    beta = torch.where(count > 0, beta, torch.full_like(beta, config.empty_active_set_beta))
    worst = (safe-closing).masked_fill(~remaining, torch.inf).amin(dim=1, keepdim=True)
    worst = torch.where(count > 0, worst, torch.zeros_like(worst))
    metrics = dict(baseline.metrics)
    metrics['beta'] = torch.where(changed, beta, metrics['beta'])
    metrics['active_beam_count'] = torch.where(changed, count.to(governed.dtype), metrics['active_beam_count'])
    metrics['worst_margin_m_s'] = torch.where(changed, worst, metrics['worst_margin_m_s'])
    issued = clip_vector_norm(governed*metrics['beta'], config.velocity_limit_m_s)
    velocity = torch.where(changed, issued, baseline.v_final_b)
    return ICSResult(velocity, metrics, dict(cache, active_mask=remaining))


class BilateralStoppingPipeline(ControlCorePipeline):
    """Opt-in execution after the unchanged control pipeline and emergency decision."""

    def __init__(self, baseline):
        super().__init__(baseline.config)
        self.lidar_rays_body = baseline.lidar_rays_body
        self._anchor = baseline._anchor
        self._last_ics_cache = baseline._last_ics_cache
        self.allow_bilateral_stopping = False
        self.bilateral_stopping_used = False

    def __call__(self, lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                 explicit_reset, controller_quaternion_wxyz):
        result = super().__call__(lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                                  explicit_reset, controller_quaternion_wxyz)
        self.bilateral_stopping_used = False
        if not self.allow_bilateral_stopping:
            return result
        rays = self.lidar_rays_body
        if rays is None:
            rays = mounted_mid360_rays_body(device=lidar_grid.device, dtype=lidar_grid.dtype)
        baseline = ICSResult(result.v_final_b, result.ics_metrics, self._last_ics_cache)
        certified = apply_bilateral_stopping(lidar_grid, state_vec, rays, result.v_gov_b,
                                            history_dt_s, self.config.ics, baseline)
        if certified is baseline:
            return result
        self.bilateral_stopping_used = True
        self._last_ics_cache = certified.cache
        world = rotate_body_to_world(certified.v_final_b, controller_quaternion_wxyz)
        return replace(result, v_final_b=certified.v_final_b, v_final_w=world, ics_metrics=certified.metrics)
