"""Recheck a short neutral brake when four associated scans show a turning mover."""

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from .escape_passage import _return_seed, observed_path_clearance, track_nearby_return


_MAX_PROVED_PACKETS = 6
_FLOOR_M = .39
_HORIZON_S = 1.15
_QUEUE_S = .10
_RISK_TIMES_S = np.linspace(0., 6., 121)


def _rotate_xy(xy, yaw):
    x, y = np.moveaxis(xy, -1, 0)
    return np.stack((x*math.cos(yaw)-y*math.sin(yaw), x*math.sin(yaw)+y*math.cos(yaw)), axis=-1)


@dataclass
class _Track:
    points: np.ndarray | None = None
    radii: np.ndarray | None = None
    axis: np.ndarray | None = None
    request: np.ndarray | None = None
    velocities: list = field(default_factory=list)
    matched: int = 0
    packets: int = 0
    spent: bool = False

    def end_brake(self):
        self.axis = None
        self.packets = 0

    def lose_association(self):
        self.points = self.radii = None
        self.velocities.clear()
        self.matched = 0
        self.end_brake()


class ObservedTurnBrake:
    """Match the same measured shape and renew a finite brake on fresh evidence."""

    def __init__(self, lanes):
        self.tracks = [_Track() for _ in range(lanes)]

    def reset(self, lanes):
        for lane in lanes.tolist():
            self.tracks[lane] = _Track()

    def apply(self, lidar, state, selected, config, *, rays, offsets, coast, protected, issued):
        history = state.reshape(-1, 4, 13)
        ages = history[:, :, 12]
        speed = history[:, -1, 6:8].norm(dim=1)
        fresh = (((ages >= 0.) & (ages <= .04)).all(dim=1)
                 & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .50)
                 & (history[:, :, 3:6].norm(dim=2) > .90).all(dim=1)
                 & (speed >= .12) & (speed <= .30))
        measurements = self._measure(lidar, state, rays, offsets, coast, ages, fresh)
        frame = history.detach().cpu().numpy()
        blocked = protected.detach().cpu().tolist()
        floors = state.new_zeros(len(state))
        for lane, track in enumerate(self.tracks):
            estimate = measurements.get(lane)
            request = frame[lane, -1, 6:9]
            yaw = -.05*float(frame[lane, 2, 2]+frame[lane, 3, 2])
            if track.request is not None:
                track.request = _rotate_xy(track.request, yaw)
                if np.linalg.norm(request[:2]-track.request) > .04:
                    track.request = None
                    track.spent = False
                    track.end_brake()
            if track.axis is not None:
                track.axis = _rotate_xy(track.axis, yaw)
            if estimate is None:
                track.lose_association()
                continue
            points, radii, times, distance, previous_shift, stopped_model = estimate
            if track.points is not None:
                shared = _rotate_xy(track.points[1:], yaw)+previous_shift
                association = (np.linalg.norm(points[:3]-shared, axis=1).max() <= .035
                               and np.abs(radii[:3]-track.radii[1:]).max() <= .04)
            else:
                association = False
            if association:
                track.velocities = [_rotate_xy(v, yaw) for v in track.velocities[-14:]]
                track.matched += 1
            else:
                track.lose_association()
                track.matched = 1
            track.points, track.radii = points, radii
            speeds = np.diff(points, axis=0)/np.diff(times)[:, None]
            previous_velocities = tuple(track.velocities)
            track.velocities.append(speeds[-1])
            speed = np.linalg.norm(request[:2])
            if (blocked[lane] or not .12 <= speed <= .30 or (track.spent and track.axis is None)
                    or abs(float(request @ frame[lane, -1, 3:6])) > 1.e-5):
                track.end_brake()
                continue
            if track.matched < 3:
                continue
            scans, local, action = lidar[lane:lane+1], state[lane:lane+1], selected[lane:lane+1]
            nominal = issued(scans, local, action, config)
            original = nominal.v_final_b[0, :2].detach().cpu().numpy()
            if bool(nominal.metrics['emergency'][0, 0]) or np.linalg.norm(original) < .03:
                track.end_brake()
                continue
            axis = track.axis
            if axis is None:
                if (not 1.9 <= distance <= 2.65
                        or np.linalg.norm(frame[lane, :3, 6:9]-request, axis=1).max() > .04):
                    continue
                axis = self._onset_axis(points[-1], speeds, times, previous_velocities,
                                        original, stopped_model, radii.max())
                if axis is None:
                    continue
            gap, speed_along = float(points[-1] @ axis), float(speeds[-1] @ axis)
            if not (float(request[:2] @ axis) >= .12 and gap >= .08 and -.85 <= speed_along <= -.20):
                track.end_brake()
                continue
            pending = frame[lane, -1, 9:11]
            swept = _axis_floor(gap, speed_along, float(stopped_model @ axis), float(pending @ axis), radii.max())
            if swept < _FLOOR_M:
                track.end_brake()
                continue
            neutral = state.new_tensor([[0., .5, .5, .5]])
            checked = issued(scans, local, neutral, config)
            if (float(checked.metrics['beta'][0, 0]) < .90 or bool(checked.metrics['emergency'][0, 0])
                    or float(checked.v_final_b.norm()) > .001):
                track.end_brake()
                continue
            static = observed_path_clearance(scans, rays, coast[lane:lane+1], checked.v_final_b[:, None], hold_s=.6)
            static = float(static[0, 0])-_QUEUE_S*float(np.linalg.norm(frame[lane, -1, 9:12]))-.004
            if not math.isfinite(static) or static < _FLOOR_M:
                track.end_brake()
                continue
            selected[lane] = neutral[0]
            floors[lane] = min(swept, static)
            track.axis = axis
            if track.request is None:
                track.request = request[:2].copy()
            track.packets += 1
            track.spent = True
            if track.packets >= _MAX_PROVED_PACKETS:
                track.end_brake()
        return floors

    @staticmethod
    def _measure(lidar, state, rays, offsets, coast, ages, fresh):
        lanes = fresh.nonzero().flatten()
        if not len(lanes):
            return {}
        scans, frames = (lidar, state) if len(lanes) == len(state) else (lidar[lanes], state[lanes])
        closing, seed, times = _return_seed(scans, frames, maximum_range_m=2.65)
        tracked = closing.nonzero().flatten()
        if not len(tracked):
            return {}
        support, _, elapsed, shapes = track_nearby_return(
            scans[tracked], frames[tracked], rays, curved=True, maximum_previous_command=.46,
            seeded=(closing[tracked], seed[tracked], times[tracked]))
        centers, radii, fits = (torch.stack(values, dim=1) for values in zip(*shapes, strict=True))
        valid = support & fits.all(dim=1) & ((radii.amax(dim=1)-radii.amin(dim=1)) <= .04)
        if not bool(valid.any()):
            return {}
        selected = lanes[tracked[valid]]
        delta = ages[selected, -1:]-ages[selected]
        position = centers[valid]+offsets[selected]+delta[:, :, None]*coast[selected, None]
        distance = lidar[selected, 9].flatten(1).gather(1, seed[tracked[valid], None])[:, 0]
        values = (position, radii[valid], elapsed[valid], distance, offsets[selected, 2], coast[selected])
        return dict(zip(selected.tolist(), zip(*(x.detach().cpu().numpy() for x in values), strict=True), strict=True))

    @staticmethod
    def _onset_axis(position, speeds, elapsed, prior, nominal, coast, radius):
        acceleration = np.diff(speeds, axis=0)/np.diff(elapsed)[-2:, None]
        mean = acceleration.mean(axis=0)
        size = np.linalg.norm(mean)
        if not (.30 <= size <= .95 and np.linalg.norm(acceleration[1]-acceleration[0]) <= .16
                and position @ mean < 0.):
            return None
        bend = mean/size
        normal = np.array([-bend[1], bend[0]])
        axis = np.sign(position @ normal)*normal
        cap = max((-v @ bend for v in prior), default=0.)
        if not (.25 <= cap <= .85 and nominal @ axis >= .12):
            return None
        # This longer forecast identifies a reason to brake. It grants no
        # authority beyond the independently rechecked short separating bound.
        t = _RISK_TIMES_S
        hold = np.minimum(t, .60)
        travel = hold-.35*(-np.expm1(-hold/.35))*np.exp(-(t-hold)/.35)
        original = travel[:, None]*nominal+(-.35*np.expm1(-t/.35))[:, None]*coast
        along = float(speeds[-1] @ bend)+.05*size
        for gain in (.75, 1., 1.25):
            for allowance in (-.12, 0., .12):
                final_speed = max(.20, cap+allowance)
                rate = size*gain
                duration = np.clip((final_speed-along)/rate, 0., 3.)
                bending = np.minimum(t, duration)
                distance = along*bending+.5*rate*bending**2+final_speed*(t-bending)
                for shift in (-.04, 0., .04):
                    curve = position+t[:, None]*(speeds[-1] @ normal+shift)*normal+distance[:, None]*bend
                    if np.linalg.norm(curve-original, axis=1).min()-radius < .36:
                        return axis
        return None


def _axis_floor(gap, incoming, coast, queued, radius):
    # Nonnegative coast and queued responses grow monotonically. Charging each
    # at its worst response lag also covers a queue shorter than one packet.
    t = _HORIZON_S
    carry = -.45*math.expm1(-t/.45)*max(0., coast)
    queue_gain = _QUEUE_S+.25*math.expm1(-_QUEUE_S/.25)*math.exp(-(t-_QUEUE_S)/.25)
    approach = (incoming-.12)*t-.075*t*t
    return gap+approach-carry-queue_gain*max(0., queued)-radius
