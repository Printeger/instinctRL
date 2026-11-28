"""Bounded deflection and updated bearing for an observed moving return."""

from dataclasses import dataclass

import numpy as np
import torch

from .escape_passage import (_return_seed, advance_command_response, fitted_sphere_returns,
                             observed_path_clearance, rotate_scan_rays, track_nearby_return)


_LAGS_S = (.25, .35, .45)
_QUEUES_S = (0., .025, .05, .075, .10)
_TIMES_S = np.arange(601)*.005
_PASS_S = 2.4
_SURFACE_FLOOR_M = .39
_ENTRY_DELTA_M_S = .30


@dataclass
class _Fit:
    center: np.ndarray
    radius: float
    velocity: np.ndarray
    acceleration: np.ndarray

    def mean_acceleration(self):
        return np.pad(self.acceleration.mean(axis=1), ((0, 0), (0, 1)))


def _fit_return(lidar, state, rays, offsets, coast, *, entry):
    history = state.reshape(4, 13)
    ages = history[:, 12]
    valid = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    distance, seed = lidar[:, 9].masked_fill(~valid[:, -1], torch.inf).flatten(1).min(dim=1)
    if entry and (not .85 <= float(distance[0]) <= 1.4 or float(valid.all(dim=1).float().mean()) < .50):
        return None
    ready = distance.isfinite() & (distance <= 2.65)
    if not bool(ready[0]):
        return None
    _, _, times = _return_seed(lidar, state, maximum_range_m=2.65)
    tracked, _, times, shapes = track_nearby_return(
        lidar, state, rays, curved=True, fit_full_center=True, maximum_previous_command=.46,
        seeded=(ready, seed, times))
    centers, radii, supported = (torch.stack(values, dim=1)[0] for values in zip(*shapes, strict=True))
    if not bool(tracked[0] & supported.all()) or float(radii.max()-radii.min()) > (.025 if entry else .04):
        return None
    if entry:
        near, other, visible = fitted_sphere_returns(lidar, rays, centers[-1:], radii[-1:])
        if not bool(visible[0] & (other[0] >= near[0]+.06)):
            return None

    gaps = times[0].diff().cpu().numpy()
    if ((gaps < .065) | (gaps > .135)).any():
        return None
    age = (ages[-1]-ages).cpu().numpy()
    points = centers[:, :2].cpu().numpy()[None]+offsets+age[None, :, None]*coast[:, None]
    velocities = np.diff(points, axis=1)/gaps[None, :, None]
    acceleration = np.diff(velocities, axis=1)/(.5*(gaps[:-1]+gaps[1:]))[None, :, None]
    if entry:
        lengths = np.linalg.norm(acceleration, axis=2)
        if lengths.min() < .35 or lengths.max() > .85:
            return None
        alignment = (acceleration[:, 0]*acceleration[:, 1]).sum(axis=1)/lengths.prod(axis=1)
        if alignment.min() < .999 or np.linalg.norm(np.diff(acceleration, axis=1), axis=2).max() > .09:
            return None
    mean = acceleration.mean(axis=1)
    velocity = np.pad(velocities[:, -1]+.5*gaps[-1]*mean, ((0, 0), (0, 1)))
    if np.linalg.norm(velocity, axis=1).max() > 1.42:
        return None
    return _Fit(centers[-1].cpu().numpy(), float(radii.max()), velocity, acceleration)


def _curved_clearance(fit, targets, coast, queued, hold):
    """Rank paths across the estimated one-direction bend and response bracket."""
    lower = np.full(len(targets), np.inf)
    centers = fit.center+_TIMES_S[None, :, None]*fit.velocity[:, None]
    for i, (lag, acceleration) in enumerate(zip(_LAGS_S, fit.mean_acceleration(), strict=True)):
        length = np.linalg.norm(acceleration)
        if length < .20:
            return np.full(len(targets), -np.inf)
        axis = acceleration/length
        maximum = length+.08
        along = fit.velocity[i]@axis
        across_squared = np.linalg.norm(fit.velocity[i]-along*axis)**2
        maximum_time = max(0., (np.sqrt(max(0., 1.5**2-across_squared))-along)/maximum)
        active = np.minimum(_TIMES_S, maximum_time)
        extent = maximum*(_TIMES_S*active-.5*active**2)
        spread = .05/(length-.08)
        drift = np.r_[coast[i], 0.]

        def response(t):
            return t+lag*np.expm1(-t/lag)

        for queue in _QUEUES_S:
            time = np.maximum(0., _TIMES_S-queue)
            travel = response(time)-response(np.maximum(0., time-hold))
            base = (_TIMES_S-response(_TIMES_S))[:, None]*drift
            base += (response(_TIMES_S)-response(time))[:, None]*queued
            relative = base[None]+travel[None, :, None]*targets[:, None]-centers[i]
            parallel = relative@axis
            transverse = relative-parallel[..., None]*axis
            transverse[..., 2] = np.maximum(0., np.abs(transverse[..., 2])-.05)
            across = np.linalg.norm(transverse, axis=2)
            nearest = np.clip(parallel+spread*across/np.sqrt(1.-spread**2), 0., extent)
            margin = np.hypot(parallel-nearest, across)-spread*nearest-fit.radius-.015-.010*_TIMES_S
            lower = np.minimum(lower, margin.min(axis=1))
    return lower


def _choices(lidar, state, rays, fit, coast, config, issued, *, entry):
    up = -state[0, 42:45]
    up = up/up.norm()
    x = state.new_tensor([1., 0., 0.])
    x = x-(x@up)*up
    x = x/x.norm()
    y = torch.cross(up, x, dim=0)
    angles = torch.arange(72, dtype=state.dtype, device=state.device)*np.pi/36.
    speed = .45 if entry else .2999
    headings = speed*(angles.cos()[:, None]*x+angles.sin()[:, None]*y)
    actions = torch.cat((state.new_zeros((len(headings), 1)), .5+headings), dim=1)
    checked = issued(lidar.expand(len(headings), -1, -1, -1), state.expand(len(headings), -1), actions, config)
    latest = state[0, 45:48]
    usable = (checked.metrics['beta'][:, 0] >= .90) & (checked.metrics['emergency'][:, 0] == 0.)
    usable &= (checked.v_final_b@latest >= (.66 if entry else .60)*(latest@latest))
    valid = usable.nonzero().flatten()
    if not len(valid):
        return None
    velocity = checked.v_final_b[valid]
    targets = velocity.cpu().numpy()
    queued = state[0, 48:51].cpu().numpy()
    best = None
    for hold in (_PASS_S,) if entry else (.8, 1.2, 1.8, _PASS_S):
        wall = observed_path_clearance(lidar, rays, state.new_tensor(coast[1:2]), velocity[None],
                                       hold_s=hold)[0].cpu().numpy()-.055
        score = np.minimum(wall, _curved_clearance(fit, targets, coast, queued, hold))-.02
        score[wall < _SURFACE_FLOOR_M] = -np.inf
        choice = int(score.argmax())
        if np.isfinite(score[choice]) and (best is None or score[choice] > best[2]):
            best = actions[valid[choice]], round(hold*10), float(score[choice])
    if entry and (best is None or best[2] < _SURFACE_FLOOR_M):
        return None
    return best


def _limited_entry(scan, state, action, score, config, issued, action_for_velocity):
    full = issued(scan, state, action[None], config).v_final_b
    previous = state[:, 48:51]
    change = full-previous
    size = float(change.norm())
    if size <= _ENTRY_DELTA_M_S:
        return action, score

    step = previous+(_ENTRY_DELTA_M_S/size)*change
    trial = action_for_velocity(step, previous, config.governor)
    limited = issued(scan, state, trial, config).v_final_b
    if float((limited-previous).norm()) > _ENTRY_DELTA_M_S+1.e-5:
        return None, 0.
    # A one-packet change can shift the subsequent response by at most its
    # velocity difference times that packet's duration in the response model.
    return trial[0], score-.10*float((full-limited).norm())


@dataclass
class _Pass:
    action: torch.Tensor
    side: torch.Tensor
    remaining: int
    elapsed: int = 0
    recognized: bool = False
    turned: bool = False


class MeasuredCurveDeflection:
    def __init__(self, lanes, reference, *, limit_entry=False):
        self.passes = [None]*lanes
        self.offsets = [reference.new_zeros((lanes, 4, 2)) for _ in _LAGS_S]
        self.coast = [reference.new_zeros((lanes, 2)) for _ in _LAGS_S]
        self.age = reference.new_zeros(lanes)
        self.rearm = reference.new_zeros(lanes)
        self.limit_entry = limit_entry

    def reset(self, lanes):
        for lane in lanes.tolist():
            self.passes[lane] = None
        for offsets, coast in zip(self.offsets, self.coast, strict=True):
            offsets[lanes] = 0.
            coast[lanes] = 0.
        self.age[lanes] = 0.
        self.rearm[lanes] = 0.

    def apply(self, lidar, state, selected, config, *, rays, blocked, issued, action_for_velocity):
        for index, lag in enumerate(_LAGS_S):
            self.offsets[index], self.coast[index] = advance_command_response(
                self.offsets[index], self.coast[index], state, lag=lag)
        self.age.add_(1)
        self.rearm.sub_(1).clamp_min_(0)
        history = state.reshape(-1, 4, 13)
        age = history[:, :, 12]
        fresh = (((age >= 0.) & (age <= .04)).all(dim=1) & (age.diff(dim=1).abs() <= .03).all(dim=1)
                 & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .40)
                 & ((-history[:, -1, 3:6]).norm(dim=1) > .90) & ~blocked & (self.age >= 4.))
        speed = history[:, -1, 6:9].norm(dim=1)
        current = fresh & (speed > .03)
        active = state.new_tensor([item is not None for item in self.passes], dtype=torch.bool)
        trusted = (lidar[:, 10] > .5) & (lidar[:, 11] > .55)
        nearest = lidar[:, 9].masked_fill(~trusted, torch.inf).flatten(1).amin(dim=1)
        can_enter = (speed >= .08) & (speed <= .30) & (nearest >= .85) & (nearest <= 1.4) & (self.rearm == 0.)
        candidates = current & (active | can_enter)
        update = state.new_zeros((len(state), 3))

        for lane in (active & ~candidates).nonzero().flatten().tolist():
            self.passes[lane] = None
        for lane in candidates.nonzero().flatten().tolist():
            scan, packet = lidar[lane:lane+1], state[lane:lane+1]
            offsets = torch.stack([o[lane] for o in self.offsets]).cpu().numpy()
            coast = torch.stack([v[lane] for v in self.coast]).cpu().numpy()
            item = self.passes[lane]
            score = 0.
            startup = None
            if item is None:
                fit = _fit_return(scan, packet, rays, offsets, coast, entry=True)
                if fit is None:
                    continue
                original = issued(scan, packet, selected[lane:lane+1], config)
                if bool(original.metrics['emergency'][0, 0]):
                    continue
                floor = _curved_clearance(fit, original.v_final_b.cpu().numpy(), coast,
                                          packet[0, 48:51].cpu().numpy(), .60)
                if floor[0] >= .36:
                    continue
                choice = _choices(scan, packet, rays, fit, coast, config, issued, entry=True)
                if choice is None:
                    continue
                action, packets, score = choice
                if self.limit_entry:
                    startup, score = _limited_entry(scan, packet, action, score, config, issued, action_for_velocity)
                    if score < _SURFACE_FLOOR_M:
                        continue
                side = torch.cross(-packet[0, 42:45], packet.new_tensor(fit.mean_acceleration()[1]), dim=0)
                item = _Pass(action.clone(), side/side.norm(), packets)
                self.passes[lane] = item
            elif not self._renew(item, scan, packet, rays, offsets, coast, config, issued):
                self.passes[lane] = None
                continue
            proposed = item.action if startup is None else startup
            check = issued(scan, packet, proposed[None], config)
            if bool(check.metrics['emergency'][0, 0]) or float(check.metrics['beta'][0, 0]) < .90:
                self.passes[lane] = None
                continue
            latest = packet[0, 45:48]
            if float(check.v_final_b[0]@latest) < 0.:
                original = issued(scan, packet, selected[lane:lane+1], config)
                if (float(original.metrics['beta'][0, 0]) >= .90
                        and float(original.v_final_b[0]@latest) >= .60*float(latest@latest)):
                    self.passes[lane] = None
                    continue
            selected[lane] = proposed
            if score:
                self.rearm[lane] = item.remaining
            item.elapsed += 1
            item.remaining -= 1
            update[lane] = packet.new_tensor([score, item.elapsed, item.turned])
            if not item.remaining:
                self.passes[lane] = None
        return update

    @staticmethod
    def _renew(item, scan, state, rays, offsets, coast, config, issued):
        history = state.reshape(4, 13)
        delta = -.05*(history[-2, :3]+history[-1, :3])
        item.action[1:] = .5+rotate_scan_rays((item.action[1:]-.5)[None], delta)[0]
        side = rotate_scan_rays(item.side[None], delta)[0]
        up = -history[-1, 3:6]
        side = side-(side@up)*up
        item.side = side/side.norm()
        if item.elapsed < 5 or item.turned:
            return True
        fit = _fit_return(scan, state, rays, offsets, coast, entry=False)
        if fit is not None:
            transverse = fit.acceleration@item.side[:2].cpu().numpy()
            turned = any((sign*transverse[:, 0]).min() > .17 and (sign*transverse[:, 1]).min() > .20
                         for sign in (-1., 1.))
            if turned:
                choice = _choices(scan, state, rays, fit, coast, config, issued, entry=False)
                if choice is None:
                    return False
                item.action, packets, _ = choice
                item.action = item.action.clone()
                item.remaining = min(item.remaining, packets)
                item.turned = True
                item.recognized = True
            elif np.abs(transverse[:, 1]).max() <= .05:
                item.recognized = True
        return item.recognized or item.elapsed < 7
