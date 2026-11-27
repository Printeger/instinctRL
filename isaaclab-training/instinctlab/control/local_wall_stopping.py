"""Check a finite braking path beside observed stationary surfaces."""

import math

import numpy as np
import torch

from .escape_passage import _return_seed
from .governor import clip_vector_norm
from .ics import AZIMUTH_SAMPLES, ELEVATION_SAMPLES, ICSResult, attenuate_command
from .tube_stopping import (
    TubeStoppingDecision, TubeStoppingPipeline, _align_scans, _remaining_ratio, _surface_normals,
)


_ANGLES = (0., -.15, .15, -.30, .30, -.50, .50, -.70, .70)
_NORMAL_BINS = np.linspace(-np.pi, np.pi, 181)
_DISTANCE_BINS = np.linspace(.25, 1.65, 71)
_HOLD_S = .60


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


def certify_local_wall_request(lidar, state, rays, governed, config, baseline, *, previous_speed_bound):
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
    for angle in _ANGLES:
        target = math.cos(angle)*forward+math.sin(angle)*side+vertical
        projection = float(target @ intent)
        magnitude = float(np.linalg.norm(target))
        if projection < .60*speed**2 or magnitude > .30:
            continue
        heading = target/magnitude
        axial = prior @ heading
        lateral = np.linalg.norm(prior-axial[:, None]*heading, axis=1)
        lower = -.35*max(0., -axial.min())-.10*max(0., -axial[-1])
        upper = _HOLD_S*magnitude+.35*max(0., axial.max())+.10*max(0., axial[-1])
        required = .41+.015+.008+tail+.35*lateral.max()+.10*lateral[-1]
        along = points @ heading
        transverse = np.linalg.norm(points-along[..., None]*heading, axis=2)
        gap = float(np.hypot(along-along.clip(lower, upper), transverse).min(initial=np.inf, where=valid))
        if gap < required:
            continue
        visible = valid & (aligned @ heading >= .99) & (along > upper+required+.025)
        bearing = aligned @ np.cross(up, heading)
        if any(((visible & (sign*bearing > 0.)).reshape(3, ELEVATION_SAMPLES, AZIMUTH_SAMPLES)
                .any(axis=1).sum(axis=1) < 3).any()
               for sign in (-1., 1.)):
            continue

        proposed = governed.new_tensor(target)[None]
        result = attenuate_command(lidar, rays, proposed, .10, config, state_vec=state)
        if bool(result.metrics['emergency'][0, 0]):
            continue
        if float(result.v_final_b[0] @ request) >= .60*speed**2:
            return TubeStoppingDecision(result, proposed, _HOLD_S, gap, float(required), 0)
        if planes is None:
            planes = _local_planes(scans, history, array(rays), points, valid, up)
        if not planes:
            continue
        explained = np.zeros(len(rays), dtype=bool)
        for normal, distance, associated in planes:
            advance = heading @ normal
            if distance-max(lower*advance, upper*advance) >= required:
                explained |= associated
        closing, ratios = _remaining_ratio(result.cache, config)
        active = result.cache['active_mask']
        matched = torch.as_tensor(explained, device=rays.device)[None]
        cleared = int((active & matched & (ratios < .90)).sum())
        remaining = active & ~matched
        count = remaining.sum(dim=1, keepdim=True)
        beta = ratios.masked_fill(~remaining, 1.).amin(dim=1, keepdim=True)
        beta = torch.where(count > 0, beta, torch.full_like(beta, config.empty_active_set_beta))
        if not cleared or float(beta[0, 0]) < .90 or float(beta[0, 0])*projection < .60*speed**2:
            continue
        margin = (result.cache['v_safe_m_s']-closing).masked_fill(~remaining, torch.inf).amin(dim=1, keepdim=True)
        metrics = dict(result.metrics, beta=beta, active_beam_count=count.to(proposed.dtype),
                       worst_margin_m_s=torch.where(count > 0, margin, 0.))
        certified = ICSResult(clip_vector_norm(proposed*beta, config.velocity_limit_m_s), metrics,
                              dict(result.cache, active_mask=remaining))
        return TubeStoppingDecision(certified, proposed, _HOLD_S, gap, float(required), cleared)
    return None


class LocalWallStoppingPipeline(TubeStoppingPipeline):
    certify = staticmethod(certify_local_wall_request)
