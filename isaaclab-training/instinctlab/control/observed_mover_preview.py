"""Short deflection with a fresh range certificate for an approaching return."""

from dataclasses import dataclass

import numpy as np
import torch

from .escape_passage import observed_path_clearance, rotate_scan_rays
from .measured_curve_deflection import (
    _LAGS_S, _SURFACE_FLOOR_M, _fit_return, _limited_entry, fresh_motion_history,
)


_ACCELERATION_M_S2 = 1.05
_MIN_PROJECTED_GAP_M = .12
_MAX_PRIOR_PROJECTED_GAP_M = .15
_MIN_PROJECTED_GAIN_M = .08
_MAX_PACKETS = 10
_INITIAL_HOLD_S = .60
_CERTIFICATE_TIMES_S = np.arange(201)*.005
_RANK_TIMES_S = np.arange(321)*.01


@dataclass
class _Deflection:
    action: torch.Tensor
    count: int


def _response(time, lag):
    return time + lag*np.expm1(-time/lag)


def _own_path(time, lag, queue, coast, queued, targets, hold):
    active = np.maximum(0., time-queue)
    drift = (time-_response(time, lag))[:, None]*np.r_[coast, 0.]
    drift += (_response(time, lag)-_response(active, lag))[:, None]*queued
    travel = _response(active, lag)-_response(np.maximum(0., active-hold), lag)
    return drift+travel[None, :, None]*targets[:, None]


def _clearance(fit, coast, queued, targets, hold, horizon):
    time = _CERTIFICATE_TIMES_S[:round(horizon/.005)+1]
    radius = fit.radius+.025+.05*time+.5*_ACCELERATION_M_S2*time**2
    low = np.full(len(targets), np.inf)
    for lag, drift, velocity in zip(_LAGS_S, coast, fit.velocity, strict=True):
        mover = fit.center+time[:, None]*velocity
        for queue in (0., .05, .10):
            own = _own_path(time, lag, queue, drift, queued, targets, hold)
            low = np.minimum(low, (np.linalg.norm(own-mover, axis=2)-radius).min(axis=1))
    return low


def _rank_continuations(fit, coast, queued, targets):
    """Rank certified choices by projected separation after the short certificate."""
    time = _RANK_TIMES_S
    low = np.full(len(targets), np.inf)
    for lag, drift, velocity, accel in zip(_LAGS_S, coast, fit.velocity, fit.mean_acceleration(), strict=True):
        own = _own_path(time, lag, .10, drift, queued, targets, _INITIAL_HOLD_S)
        for duration in (0., .65, 1., 1.30):
            active = np.minimum(time, duration)
            mover = fit.center+time[:, None]*velocity+(time*active-.5*active**2)[:, None]*accel
            low = np.minimum(low, (np.linalg.norm(own-mover, axis=2)-fit.radius).min(axis=1))
    return low


class ObservedMoverPreview:
    def __init__(self, lanes, reference, *, gravity_reproof=False):
        self.passes = [None]*lanes
        self.rearm = reference.new_zeros(lanes)
        self.gravity_reproof = gravity_reproof
        self.record_width = 6 if gravity_reproof else 5

    def reset(self, lanes):
        for lane in lanes.tolist():
            self.passes[lane] = None
        self.rearm[lanes] = 0.

    def apply(self, lidar, state, selected, config, *, rays, blocked, observer, issued, action_for_velocity):
        self.rearm.sub_(1).clamp_min_(0)
        history = state.reshape(-1, 4, 13)
        speed = history[:, -1, 6:9].norm(dim=1)
        fresh = fresh_motion_history(history) & ~blocked & (observer.age >= 4.) & (speed > 1.e-6)
        active = fresh.new_tensor([item is not None for item in self.passes])
        usable = (lidar[:, 10] > .5) & (lidar[:, 11] > .55)
        near = lidar[:, 9].masked_fill(~usable, torch.inf).flatten(1).amin(dim=1)
        entry = (speed >= .06) & (speed <= .30) & (near >= .85) & (near <= 1.4) & (self.rearm == 0.)
        update = state.new_zeros((len(state), self.record_width))
        for lane in (active & ~fresh).nonzero().flatten().tolist():
            self.passes[lane] = None
        for lane in (fresh & (entry | active)).nonzero().flatten().tolist():
            scan, packet = lidar[lane:lane+1], state[lane:lane+1]
            offsets = torch.stack([o[lane] for o in observer.offsets]).cpu().numpy()
            coast = torch.stack([v[lane] for v in observer.coast]).cpu().numpy()
            item = self.passes[lane]
            starting = item is None
            fit = _fit_return(scan, packet, rays, offsets, coast, entry=starting,
                              require_approach=starting, coherent=False)
            aligned = (self.gravity_reproof and not starting and fit is not None
                       and np.linalg.norm(fit.acceleration, axis=2).max() > _ACCELERATION_M_S2)
            if aligned:
                fit = _fit_return(scan, packet, rays, offsets, coast, entry=False, coherent=False, align_gravity=True)
            if fit is None or np.linalg.norm(fit.acceleration, axis=2).max() > _ACCELERATION_M_S2:
                self.passes[lane] = None
                continue
            queued = packet[0, 48:51].cpu().numpy()
            prior = issued(scan, packet, selected[lane:lane+1], config)
            latest = float(_clearance(fit, coast, queued, prior.v_final_b.cpu().numpy(), _INITIAL_HOLD_S, 1.)[0])
            before, projected = 0., 0.
            if starting:
                if latest >= .36 or bool(prior.metrics['emergency'][0, 0]):
                    continue
                before = float(_rank_continuations(fit, coast, queued, prior.v_final_b.cpu().numpy())[0])
                if before >= _MAX_PRIOR_PROJECTED_GAP_M:
                    continue
                choice = self._enter(scan, packet, rays, fit, coast, queued, before, config, issued, action_for_velocity)
                if choice is None:
                    continue
                action, target, margin, projected = choice
                item = _Deflection(target, 0)
                self.rearm[lane] = _MAX_PACKETS
            else:
                choice = self._renew(item, scan, packet, rays, fit, coast, queued, latest, prior, config, issued)
                if choice is None:
                    self.passes[lane] = None
                    continue
                action, margin = choice
                item.action = action
            selected[lane] = action
            item.count += 1
            update[lane, :5] = packet.new_tensor([item.count, margin, latest, before, projected])
            if self.gravity_reproof:
                update[lane, 5] = float(aligned)
            self.passes[lane] = item if item.count < _MAX_PACKETS else None
        return update

    @staticmethod
    def _enter(scan, state, rays, fit, coast, queued, before, config, issued, action_for_velocity):
        request = state[0, 45:48]
        up = -state[0, 42:45]
        up = up/up.norm()
        x = state.new_tensor([1., 0., 0.])
        x = x-(x@up)*up
        x = x/x.norm()
        y = torch.cross(up, x, dim=0)
        angles = torch.arange(36, device=state.device, dtype=state.dtype)*np.pi/18.
        directions = angles.cos()[:, None]*x+angles.sin()[:, None]*y
        velocities = torch.cat((.2999*directions, .4499*directions))
        actions = torch.cat((state.new_zeros((len(velocities), 1)), .5+velocities), dim=1)
        result = issued(scan.expand(len(actions), -1, -1, -1), state.expand(len(actions), -1), actions, config)
        available = (result.metrics['beta'][:, 0] >= .90) & (result.metrics['emergency'][:, 0] == 0.)
        available &= result.v_final_b@request >= .60*(request@request)
        indices = available.nonzero().flatten()
        if not len(indices):
            return None
        velocity = result.v_final_b[indices]
        static = observed_path_clearance(scan, rays, state.new_tensor(coast[1:2]),
                                         velocity[None], hold_s=_INITIAL_HOLD_S)[0].cpu().numpy()-.055
        targets = velocity.cpu().numpy()
        floor = np.minimum(static, _clearance(fit, coast, queued, targets, _INITIAL_HOLD_S, 1.))
        projections = _rank_continuations(fit, coast, queued, targets)
        minimum_projection = max(_MIN_PROJECTED_GAP_M, before+_MIN_PROJECTED_GAIN_M)
        candidates = np.flatnonzero((floor >= _SURFACE_FLOOR_M) & (projections >= minimum_projection))
        for best in candidates[np.argsort(-projections[candidates])]:
            target = actions[indices[best]]
            startup, margin = _limited_entry(scan, state, target, float(floor[best]), config, issued, action_for_velocity)
            if margin < _SURFACE_FLOOR_M:
                continue
            check = issued(scan, state, startup[None], config)
            if float(check.metrics['beta'][0, 0]) >= .90 and not bool(check.metrics['emergency'][0, 0]):
                return startup, target, margin, float(projections[best])
        return None

    @staticmethod
    def _renew(item, scan, state, rays, fit, coast, queued, latest, prior, config, issued):
        requested = state[0, 45:48]
        if (item.count >= 6 and latest >= .36 and float(prior.metrics['beta'][0, 0]) >= .90
                and float(prior.v_final_b[0]@requested) >= .60*float(requested@requested)):
            return None
        history = state.reshape(4, 13)
        rotation = -.05*(history[-2, :3]+history[-1, :3])
        action = item.action.clone()
        action[1:] = .5+rotate_scan_rays((action[1:]-.5)[None], rotation)[0]
        check = issued(scan, state, action[None], config)
        if float(check.metrics['beta'][0, 0]) < .90 or bool(check.metrics['emergency'][0, 0]):
            return None
        hold = max(.20, _INITIAL_HOLD_S-.10*item.count)
        horizon = 1.-.10*item.count if item.count < 6 else .70
        static = observed_path_clearance(scan, rays, state.new_tensor(coast[1:2]),
                                         check.v_final_b[None], hold_s=hold)[0, 0]
        dynamic = _clearance(fit, coast, queued, check.v_final_b.cpu().numpy(), hold, horizon)[0]
        margin = min(float(static)-.055, float(dynamic))
        return (action, margin) if margin >= _SURFACE_FLOOR_M else None
