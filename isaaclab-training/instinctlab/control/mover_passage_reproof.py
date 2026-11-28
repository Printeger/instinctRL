"""Renew an observed mover escape when the selected protection is stalled."""

from dataclasses import dataclass

import numpy as np
import torch

from .escape_passage import observed_path_clearance, rotate_scan_rays
from .measured_curve_deflection import _fit_return, _return_shapes, fresh_motion_history
from .observed_mover_preview import _clearance, _rank_continuations
from .registered_mover_entry import reprove_plane_alignment


_MAX_PACKETS = 28
_STEP_CHANGE = .30
_FLOOR = .39
_CURRENT_COMMAND = object()


@dataclass
class _Pass:
    direction: torch.Tensor
    count: int


class MoverPassageReproof:
    def __init__(self, reference):
        self.passes = [None]*len(reference)
        self.recent_neutral = reference.new_zeros(len(reference))
        self.start_witnesses = reference.new_zeros(len(reference))
        self.rearm = reference.new_zeros(len(reference))

    def reset(self, lanes):
        for lane in lanes.tolist():
            self.passes[lane] = None
        self.recent_neutral[lanes] = 0.
        self.start_witnesses[lanes] = 0.
        self.rearm[lanes] = 0.

    def apply(self, lidar, state, selected, config, *, rays, observer, issued, active, ranked, braking,
              intercepting, blocked):
        frames = state.reshape(-1, 4, 13)
        requests = frames[:, :, 6:9]
        speeds = requests.norm(dim=2)
        self.recent_neutral.sub_(1).clamp_min_(0)
        self.rearm.sub_(1).clamp_min_(0)
        self.recent_neutral[(speeds[:, -1] >= .06) & (speeds[:, 1:3].amax(dim=1) <= 1.e-6)] = 7.
        prior_witnesses = self.start_witnesses.clone()
        self.start_witnesses.zero_()
        pending = blocked.new_tensor([item is not None for item in self.passes])
        stable = (requests[:, -1]-requests[:, -2]).norm(dim=1) <= .04
        stopped = frames[:, :, 9:12].norm(dim=2).amax(dim=1) <= .02
        parked_brake = braking & stopped & (pending | (self.recent_neutral > 0.))
        eligible = (active & (ranked | parked_brake) & stable & ~intercepting
                    & (speeds[:, -1] >= .06) & (speeds[:, -1] <= .30))
        usable = (lidar[:, 10] > .5) & (lidar[:, 11] > .55)
        near = lidar[:, 9].masked_fill(~usable, torch.inf).flatten(1).amin(dim=1)
        eligible &= (near >= .85) & (near <= 1.4) & (self.rearm == 0.)
        fresh = fresh_motion_history(frames)
        consider = (eligible | pending) & ~blocked & (~braking | parked_brake) & (observer.age >= 4.)
        update = state.new_zeros((len(state), 7))
        stop_backup = pending & ~blocked
        for lane in (pending & ~consider).nonzero().flatten().tolist():
            self.passes[lane] = None
        for lane in consider.nonzero().flatten().tolist():
            item = self.passes[lane]
            scan, packet = lidar[lane:lane+1], state[lane:lane+1]
            self.passes[lane] = None
            if float(speeds[lane, -1]) < .03:
                continue
            alignment = None
            native_fresh = bool(fresh[lane])
            if not native_fresh:
                if item is None:
                    continue
                without_gyro = frames[lane:lane+1].clone()
                without_gyro[:, :, :3] = 0.
                if not bool(fresh_motion_history(without_gyro)[0]):
                    continue
                alignment = reprove_plane_alignment(scan, packet, rays)
                if alignment is None:
                    continue
            coast = torch.stack([value[lane] for value in observer.coast]).cpu().numpy()
            offsets = torch.stack([value[lane] for value in observer.offsets]).cpu().numpy()
            if item is None:
                shapes = _return_shapes(scan, packet, rays, entry=True, align_gravity=True)
                if shapes is None or not bool((shapes[0].norm(dim=1).diff()[-2:] <= -.006).all()):
                    continue
            fit = _fit_return(scan, packet, rays, offsets, coast, entry=False, coherent=False, align_gravity=True)
            if fit is None or np.linalg.norm(fit.acceleration, axis=2).max() > 1.05:
                continue
            current = issued(scan, packet, selected[lane:lane+1], config)
            if bool(current.metrics['emergency'][0, 0]) or (bool(intercepting[lane])
                    and float(current.v_final_b[0].norm()) > .01):
                continue
            choice = self._choose(scan, packet, rays, fit, coast, current.v_final_b, item,
                                  native_fresh, alignment, config, issued)
            if choice is _CURRENT_COMMAND:
                stop_backup[lane] = False
                continue
            if choice is None:
                continue
            if item is None and bool(ranked[lane]):
                self.start_witnesses[lane] = prior_witnesses[lane]+1.
                if float(self.start_witnesses[lane]) < 3.:
                    continue
            action, direction, floor, backup, before, ranking, original_ranking = choice
            count = 1 if item is None else item.count+1
            selected[lane] = action
            stop_backup[lane] = False
            if item is None:
                self.rearm[lane] = _MAX_PACKETS+10
            self.passes[lane] = _Pass(direction, count)
            update[lane] = packet.new_tensor((count, floor, backup, before, original_ranking, ranking,
                                               alignment[0] if alignment is not None else 0.))
        neutral = state.new_tensor([[0., .5, .5, .5]])
        for lane in stop_backup.nonzero().flatten().tolist():
            scan, packet = lidar[lane:lane+1], state[lane:lane+1]
            current = issued(scan, packet, selected[lane:lane+1], config)
            if bool(current.metrics['emergency'][0, 0]) or (bool(intercepting[lane])
                    and float(current.v_final_b[0].norm()) > .01):
                continue
            stop = issued(scan, packet, neutral, config)
            if not bool(stop.metrics['emergency'][0, 0]):
                selected[lane] = neutral[0]
                update[lane, 0] = -1.
        return update

    @staticmethod
    def _choose(scan, state, rays, fit, coast, current, item, native_fresh, alignment, config, issued):
        queued = state[0, 48:51].cpu().numpy()
        request = state[0, 45:48]
        old = current.cpu().numpy()
        horizon, hold = (.90, .50) if item is None else (.70, .40)
        before = float(_clearance(fit, coast, queued, old, hold, horizon)[0])
        original_rank = float(_rank_continuations(fit, coast, queued, old)[0])
        projection = float(current[0]@request)
        if item is None:
            if before >= .36 or projection > .05*float(request@request):
                return None
        elif (native_fresh and before >= _FLOOR and original_rank >= .15
              and projection >= .60*float(request@request)
              and float((current-state[:, 48:51]).norm()) <= _STEP_CHANGE+.00001):
            shape = _return_shapes(scan, state, rays, entry=False, align_gravity=True)
            if (shape is not None and float(shape[0][-1].norm()-shape[1]) >= .95
                    and bool((shape[0].norm(dim=1).diff()[-2:] > .003).all())
                    and float(_clearance(fit, coast, queued, old, .60, 1.)[0]) >= _FLOOR):
                return _CURRENT_COMMAND
        if item is not None and item.count >= _MAX_PACKETS:
            return None
        angle = torch.arange(72, device=state.device, dtype=state.dtype)*np.pi/36.
        headings = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), dim=1)
        goals = torch.cat((.2999*headings, .4499*headings))
        actions = torch.cat((state.new_zeros((len(goals), 1)), .5+goals), dim=1)
        result = issued(scan.expand(len(actions), -1, -1, -1), state.expand(len(actions), -1), actions, config)
        velocity = result.v_final_b
        ready = ((result.metrics['beta'][:, 0] >= .90) & (result.metrics['emergency'][:, 0] == 0.)
                 & ((velocity-state[:, 48:51]).norm(dim=1) <= _STEP_CHANGE+.00001))
        if item is not None:
            history = state.reshape(4, 13)
            turn = -.05*(history[-2, :3]+history[-1, :3])
            direction = rotate_scan_rays(item.direction[None], turn)[0]
            ready &= velocity@direction > 0.
        available = ready.nonzero().flatten()
        if not len(available):
            return None
        velocity = velocity[available]
        static = observed_path_clearance(scan, rays, state.new_tensor(coast[1:2]), velocity[None], hold_s=hold)[0]
        targets = velocity.cpu().numpy()
        dynamic = _clearance(fit, coast, queued, targets, hold, horizon)
        backup = _clearance(fit, coast, queued, targets, .10, horizon-.10)
        floor = np.minimum(np.minimum(static.cpu().numpy()-.055, dynamic), backup)
        if alignment is not None:
            angle, _, interval = alignment
            floor -= np.linalg.norm(fit.center)*angle*(1.+2.*horizon/interval)
        projected = _rank_continuations(fit, coast, queued, targets)
        valid = floor >= _FLOOR
        if item is None:
            valid &= projected >= max(.15, original_rank+.08)
            along = (velocity@request >= .60*(request@request)).cpu().numpy()
            if (valid & along).any():
                valid &= along
        if not valid.any():
            return None
        best = int(np.where(valid, projected, -np.inf).argmax())
        return (actions[available[best]], velocity[best], float(floor[best]), float(backup[best]),
                before, float(projected[best]), original_rank)
