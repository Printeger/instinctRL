"""Certify a neutral retry on an observed mover's line after a missed release."""

import numpy as np
import torch

from .escape_passage import _return_seed, fitted_sphere_returns, track_nearby_return


RESPONSE_LAGS_S = (.25, .264, .320, .45)
POLICY_S = .10
ALONG_TRACK_SPEED_ALLOWANCE_M_S = .12
TRACK_ANGLE_ALLOWANCE_RAD = .04
SWEEP_SAMPLES_S = np.arange(601, dtype=np.float64)*.005

# Integral of |(x²−4x+2)e⁻ˣ| for the second lag derivative of the response filter.
_ZEROES = 2.+np.array([-1., 1.])*np.sqrt(2.)
_PRIMITIVE = -(np.square(_ZEROES)-2.*_ZEROES)*np.exp(-_ZEROES)
FILTER_CURVATURE = 2.*(_PRIMITIVE[0]-_PRIMITIVE[1])


def _rotate_xy(values, angle):
    x, y = values.unbind(dim=-1)
    return torch.stack((x*angle.cos()-y*angle.sin(), x*angle.sin()+y*angle.cos()), dim=-1)


class NeutralRetryObserver:
    """Causal bracket of the delayed issued-command response at four LiDAR times."""

    def __init__(self, lanes, device):
        self.lags = torch.tensor(RESPONSE_LAGS_S, dtype=torch.float64, device=device).reshape(1, -1, 1)
        self.decay = (-POLICY_S/self.lags).exp()
        self.coast = self.lags.new_zeros((lanes, len(RESPONSE_LAGS_S), 2))
        self.offsets = self.lags.new_zeros((lanes, len(RESPONSE_LAGS_S), 4, 2))
        self.peak = self.lags.new_zeros(lanes)

    def reset(self, lanes):
        self.coast[lanes] = 0.
        self.offsets[lanes] = 0.
        self.peak[lanes] = 0.

    def advance(self, state):
        frames = state.reshape(-1, 4, 13).to(dtype=self.coast.dtype)
        yaw = -.05*(frames[:, 2, 2]+frames[:, 3, 2])
        previous_yaw = -.05*(frames[:, 1, 2]+frames[:, 2, 2])
        delivered = _rotate_xy(frames[:, 2, 9:11], yaw+previous_yaw)
        coast = _rotate_xy(self.coast, yaw[:, None])
        target = delivered[:, None]
        travel = POLICY_S*target+self.lags*(1.-self.decay)*(coast-target)
        earlier = _rotate_xy(self.offsets[:, :, 1:], yaw[:, None, None])-travel[:, :, None]
        self.offsets = torch.cat((earlier, torch.zeros_like(self.offsets[:, :, :1])), dim=2)
        self.coast = self.decay*coast+(1.-self.decay)*target
        self.peak = torch.maximum(self.peak, delivered.norm(dim=1))

    def clearance(self, lane, lidar, state, rays):
        """Return the conservative swept/settled surface lower bound, or reject."""
        frames = state.reshape(4, 13)
        ages = frames[:, 12]
        closing, _, elapsed = _return_seed(lidar, state, maximum_range_m=2.65)
        gaps = elapsed.diff(dim=1)
        if (bool(closing[0]) or bool(((ages < 0.) | (ages > .04)).any())
                or bool(((gaps < .07) | (gaps > .13)).any())
                or float(frames[:, :3].norm(dim=1).max()) > .18
                or float(frames[:, 9:12].norm(dim=1).max()) > .31):
            return None
        valid = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
        coverage = valid.all(dim=1).float().mean()
        depth = lidar[:, -3].flatten(1).masked_fill(~valid[:, -1].flatten(1), torch.inf)
        distance, seed = depth.min(dim=1)
        ready = (coverage >= .55) & (distance >= .85) & (distance <= 2.65)
        if not bool(ready[0]):
            return None
        tracked, _, elapsed, shapes = track_nearby_return(
            lidar, state, rays, curved=True, maximum_previous_command=.46,
            seeded=(ready, seed, elapsed), fit_full_center=True)
        centers, radii, supported = (torch.stack(values, dim=1) for values in zip(*shapes, strict=True))
        if not bool(tracked[0] & supported.all() & (radii.max()-radii.min() <= .04)):
            return None
        nearest, other, visible = fitted_sphere_returns(lidar, rays, centers[:, -1], radii[:, -1])
        if not bool(visible[0] & (nearest[0] >= .85) & (other[0] >= .85) & other[0].isfinite()):
            return None
        # Only this lane's small associated fit crosses to float64 CPU algebra.
        def cpu(value):
            return value.detach().to(dtype=torch.float64, device='cpu').numpy()

        offsets = cpu(self.offsets[lane])
        coast = cpu(self.coast[lane])
        return _certify(cpu(centers[0]), cpu(elapsed[0]), cpu(frames), float(radii.max()), float(other[0]),
                        offsets, coast, float(self.peak[lane]))


def _fit_weights(times):
    centered = times-times.mean()
    speed = centered/(centered @ centered)
    return 1./len(times)-times.mean()*speed, speed


def _certify(centers, times, history, radius, other_surface, offsets, coast, peak):
    """Conservatively enclose every response lag in the three adjacent cells."""
    if not all(np.isfinite(value).all() for value in (centers, times, history, radius, other_surface, offsets, coast, peak)):
        return None
    length = np.linalg.norm
    age = history[-1, 12]-history[:, 12]
    points = centers[None, :, :2]+offsets+age[None, :, None]*coast[:, None]
    center_full, motion_full = _fit_weights(times)
    center_recent, motion_recent = (np.r_[0., w] for w in _fit_weights(times[1:]))
    positions = np.einsum('nfd,pf->npd', points, np.stack((center_full, center_recent, np.eye(4)[-1])))
    velocity_weights = np.vstack((motion_full, motion_recent, np.diff(np.eye(4), axis=0)/np.diff(times)[:, None]))
    velocities = np.einsum('nfd,pf->npd', points, velocity_weights)
    velocity_norms = length(velocities, axis=2)
    if velocity_norms.min() < .20 or velocity_norms.max()+ALONG_TRACK_SPEED_ALLOWANCE_M_S > 1.5:
        return None
    endpoint_normals = np.stack((-velocities[:, :, 1], velocities[:, :, 0]), axis=2)
    signed = np.einsum('nd,npd->np', points[:, -1], endpoint_normals)
    if not ((signed > 0.).all() or (signed < 0.).all()):
        return None
    pending = history[-1, 9:11]
    pending_speed = length(pending)
    gravity_discrepancy = length(np.diff(history[:, 3:6], axis=0), axis=1).sum()
    orientation_error = gravity_discrepancy+.01*length(history[:, :3], axis=1).max()
    sensor_reserve = .008+orientation_error*length(centers, axis=1).max()+radius
    position_weights = (center_full, center_recent, np.eye(4)[-1])
    motion_residual = (np.eye(4)-center_full-times[:, None]*motion_full,
                       np.eye(4)[1:]-center_recent-times[1:, None]*motion_recent)
    sweep = SWEEP_SAMPLES_S
    minimum_floor = float('inf')
    maximum_coast = 0.
    for cell, (low, high) in enumerate(zip(RESPONSE_LAGS_S[:-1], RESPONSE_LAGS_S[1:], strict=True)):
        interval = slice(cell, cell+2)
        endpoint_lags = np.array([low, high])
        lag_error = (high-low)**2/8.
        first_derivative = 2.*peak/(np.e*low)
        second_derivative = FILTER_CURVATURE*peak/low**2
        coast_error = second_derivative*lag_error

        def interpolation_error(weights):
            # Combine shared past displacements before charging the secant error.
            return coast_error*(POLICY_S*np.abs(np.cumsum(weights)[0:3]).sum()+abs(weights @ age))

        position_error = np.array([interpolation_error(w) for w in position_weights])
        velocity_error = np.array([interpolation_error(w) for w in velocity_weights])
        axis = velocities[interval].sum(axis=0)
        axis_norm = length(axis, axis=1)
        if (axis_norm <= 0.).any():
            return None
        axis /= axis_norm[:, None]
        normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
        along = (velocities[interval]*axis).sum(axis=2)
        lower_speed = along.min(axis=0)-velocity_error
        upper_speed = (length(velocities[interval], axis=2).max(axis=0)+velocity_error).max()
        mover_speed = upper_speed+ALONG_TRACK_SPEED_ALLOWANCE_M_S
        if lower_speed.min() < .20 or mover_speed > 1.5:
            return None
        angle = np.arccos(np.clip(along/length(velocities[interval], axis=2), -1., 1.)).max(axis=0)
        angle += np.arcsin(np.minimum(velocity_error/lower_speed, 1.))
        relative_angle = np.arccos(np.clip(axis @ axis[0], -1., 1.))
        if (angle[0]+angle[1:]+relative_angle[1:]).max() > np.arccos(.99):
            return None
        for residual in motion_residual:
            endpoint_error = np.einsum('nfd,pf->npd', points[interval], residual)
            sampled = np.sqrt(np.square(length(endpoint_error, axis=2)).mean(axis=1)).max()
            interpolation = np.sqrt(np.square([interpolation_error(w) for w in residual]).mean())
            if sampled+interpolation > .02:
                return None

        cone = angle+TRACK_ANGLE_ALLOWANCE_RAD
        sine, cosine = np.sin(cone), np.cos(cone)
        cell_coast = coast[interval]
        coast_limit = length(cell_coast, axis=1).max()+coast_error
        maximum_coast = max(maximum_coast, coast_limit)
        response_error = (4.*peak/(np.e**2*low)+2.*first_derivative+high*second_derivative)*lag_error
        error = position_error[:, None, None]+response_error
        displacement = -endpoint_lags[:, None]*np.expm1(-sweep/endpoint_lags[:, None])
        path = positions[interval, :, None]-displacement[:, None, :, None]*cell_coast[:, None, None]
        forward = np.einsum('nptd,md->npmt', path, axis)
        cross = np.einsum('nptd,md->npmt', path, normal)
        side = np.sign(cross[0, :, :, 0])
        fwd_max = np.abs(forward).max(axis=0)+error
        fwd_min = np.where(np.prod(forward, axis=0)>0., np.abs(forward).min(axis=0), 0.)
        fwd_min = np.maximum(fwd_min-error, 0.)
        cross_max = np.abs(cross).max(axis=0)+error
        cross_min = (side[None, :, :, None]*cross).min(axis=0)-error
        lateral = np.where(cross_min >= 0., cross_min*cosine[:, None], cross_min)-fwd_max*sine[:, None]
        if lateral[:, :, 0].min() <= 0.:
            return None
        # The previous request can persist for any fraction of one policy step.
        pending_forward = axis @ pending
        pending_cross = side*(normal @ pending)[None]
        toward = np.maximum(np.maximum(pending_cross, pending_cross*cosine)+np.abs(pending_forward)*sine, 0.)
        lateral -= POLICY_S*toward[:, :, None]
        forward_queue = POLICY_S*(np.abs(pending_forward)+np.abs(pending_cross)*sine)
        along = np.maximum(fwd_min*cosine[:, None]-cross_max*sine[:, None]-forward_queue[:, :, None]-mover_speed*sweep, 0.)
        fit_reserve = length(positions[interval, 0]-positions[interval, 1], axis=1).max()
        fit_reserve += interpolation_error(center_full-center_recent)
        reserve = sensor_reserve+fit_reserve+POLICY_S*pending_speed*sine+POLICY_S*coast_limit
        floor = (np.hypot(np.maximum(lateral, 0.), along)-reserve[:, None]).min()-.0025*(mover_speed+coast_limit)
        tail = (lateral[:, :, -1]-reserve).min()
        coast_forward = np.abs(cell_coast @ axis.T).max(axis=0)
        coast_cross = (side[None]*(cell_coast @ normal.T)[:, None]).max(axis=0)
        outward = -(np.maximum(coast_cross, coast_cross*cosine)+coast_forward*sine+coast_error)
        if not np.isfinite((floor, tail, outward.min())).all() or outward.min() < .12:
            return None
        minimum_floor = min(minimum_floor, floor, tail)
    if other_surface-.45*maximum_coast-POLICY_S*pending_speed-.055 < .85 or minimum_floor < .65:
        return None
    return float(minimum_floor)
