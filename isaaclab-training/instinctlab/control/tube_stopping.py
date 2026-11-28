"""Certify a finite command and brake through an observed stationary tube."""

from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from .command_adapter import rotate_body_to_world
from .escape_passage import _return_seed, rotate_scan_rays
from .governor import clip_vector_norm
from .ics import AZIMUTH_SAMPLES, ELEVATION_SAMPLES, ICSResult, attenuate_command, mounted_mid360_rays_body
from .pipeline import ControlCorePipeline


_GRID = (ELEVATION_SAMPLES, AZIMUTH_SAMPLES)
_LAG = .35
_INTERVAL = .10
_STAMP_LIMIT = .04
_MAX_REQUEST = .30
_ANGLE_BINS = np.linspace(-math.pi, math.pi, 181)
_MOUNT = (0., 0., .05)


@dataclass(frozen=True)
class TubeStoppingDecision:
    result: ICSResult
    target: torch.Tensor
    hold_s: float
    observed_gap_m: float
    required_gap_m: float
    cleared_limiting_beams: int
    used_oblique_facets: bool = False


def _surface_normals(scan, rays):
    ranges, seen, reliability = scan.reshape(3, *_GRID)
    valid = (seen > .5) & (reliability >= .65)
    points = rays.reshape(*_GRID, 3)*ranges[..., None]
    vertical = .5*(points[2:]-points[:-2])
    azimuth = .5*(np.roll(points,-1,axis=1)-np.roll(points,1,axis=1))[1:-1]
    normal = np.cross(azimuth, vertical)
    norm = np.linalg.norm(normal, axis=-1)
    supported = valid[1:-1] & valid[:-2] & valid[2:] & np.roll(valid,1,axis=1)[1:-1] & np.roll(valid,-1,axis=1)[1:-1]
    supported &= ((np.linalg.norm(vertical,axis=-1) < .18) & (np.linalg.norm(azimuth,axis=-1) < .18)
                  & (norm > 1.e-5) & (ranges[1:-1] < 3.5))
    samples = points[1:-1][supported]
    normal = normal[supported]/norm[supported,None]
    normal *= np.sign((normal*samples).sum(axis=1,keepdims=True))
    return samples, normal


def _longitudinal_axis(normals):
    if len(normals) < 25:
        return None
    _, basis = np.linalg.eigh(normals.T@normals/len(normals))
    selected = normals[abs(normals@basis[:,0]) <= .05]
    if len(selected)*4 < len(normals)*3:
        return None
    variance, basis = np.linalg.eigh(selected.T@selected/len(selected))
    return basis[:,0], variance


def _angular_patches(normals, axis, angles, plane_tolerance, angular_tolerance, maximum):
    available = abs(normals@axis) <= plane_tolerance
    for _ in range(maximum):
        histogram, _ = np.histogram(angles[available], bins=_ANGLE_BINS)
        index = int(histogram.argmax())
        if histogram[index] < 25:
            break
        mid = .5*(_ANGLE_BINS[index]+_ANGLE_BINS[index+1])
        patch = available & (abs((angles-mid+np.pi)%(2.*np.pi)-np.pi) <= angular_tolerance)
        available &= ~patch
        normal = normals[patch].mean(axis=0)
        normal -= (normal@axis)*axis
        yield normal/np.linalg.norm(normal)


def _tube_planes(scans, state, rays):
    frames = [_surface_normals(frame,rays) for frame in scans]
    axes = [_longitudinal_axis(normals) for _, normals in frames]
    if any(axis is None for axis in axes):
        return ()
    axis, variance = axes[-1]
    up = -state[-1,3:6]
    if (abs(up@axis) > .04 or variance[0] > .0004 or variance[1] < .12
            or any(np.linalg.norm(np.cross(axis, previous)) > .03 for previous,_ in axes)):
        return ()
    across = np.cross(up, axis)
    across /= np.linalg.norm(across)
    vertical = np.cross(axis, across)
    normals = frames[-1][1]
    angles = np.arctan2(normals@vertical, normals@across)
    axial = [points@axis for points,_ in frames]
    enclosing = []
    for normal in _angular_patches(normals,axis,angles,.025,.04,32):
        distances, errors = [], []
        for (points,directions), longitudinal in zip(frames,axial,strict=True):
            matches = (directions@normal >= math.cos(.035)) & (abs(longitudinal) < 1.1)
            offset = points[matches]@normal
            if len(offset) < 25:
                break
            median = np.median(offset)
            close = abs(offset-median) <= .018
            occupied = longitudinal[matches][close]
            bands = ((-.85,-.25),(-.25,.25),(.25,.85))
            if any(((occupied >= lo) & (occupied < hi)).sum() < 4 for lo,hi in bands):
                break
            distances.append(median)
            errors.append(np.sqrt(np.mean((offset[close]-median)**2)))
        if len(distances) == 3 and min(distances) > .25:
            enclosing.append((normal,np.asarray(distances),max(errors)))
    if len(enclosing) < 4 or any(error > .010 or abs(np.diff(d)).max() > .045 for _,d,error in enclosing):
        return ()

    enclosing_normals = np.stack([normal for normal,_,_ in enclosing])
    planes = [(normal,d) for normal,d,_ in enclosing]
    for normal in _angular_patches(normals,axis,angles,.035,.05,20):
        if (enclosing_normals@normal >= math.cos(.035)).any():
            continue
        transverse = np.cross(axis,normal)
        distances = []
        for (points,directions), longitudinal in zip(frames,axial,strict=True):
            nearby = (directions@normal >= math.cos(.040)) & (abs(longitudinal) < 1.35)
            samples = points[nearby]
            if len(samples) < 25:
                break
            distance = np.median(samples@normal)
            patch = samples[abs(samples@normal-distance) <= .013]
            if len(patch) < 25 or np.ptp(patch@axis) < .24 or np.ptp(patch@transverse) < .12:
                break
            variance, basis = np.linalg.eigh(np.cov(patch.T))
            if np.sqrt(max(0.,variance[0])) > .008 or np.linalg.norm(np.cross(basis[:,0],normal)) > .030:
                break
            distances.append(distance)
        if len(distances) == 3:
            planes.append((normal,np.asarray(distances)))
    return planes


def _align_scans(rays, history):
    aligned = [rays]
    mount = [rays.new_tensor(_MOUNT)[None]]
    issued = history[:,9:12]
    for frame in range(3):
        rotation = -.05*(history[frame,:3]+history[frame+1,:3])
        issued = torch.cat((rotate_scan_rays(issued[:frame+1],rotation),issued[frame+1:]),dim=0)
    for i in (2,1):
        gap = .1-(history[i+1,12]-history[i,12])
        delta = -.5*gap*(history[i,:3]+history[i+1,:3])
        aligned.append(rotate_scan_rays(aligned[-1],delta))
        mount.append(rotate_scan_rays(mount[-1],delta))
    return tuple(tensor.detach().cpu().numpy() for tensor in
                 (torch.stack(aligned[::-1]), torch.stack(mount[::-1]), issued))


def _bounded_wall_mask(scans, rays, aligned, mount_offsets, issued, gravity, velocity, planes, hold, tail):
    speed = np.linalg.norm(velocity)
    heading = velocity/speed
    side = np.cross(-gravity,heading)
    axial = issued@heading
    lateral = np.linalg.norm(issued-axial[:,None]*heading,axis=1)
    lower = -_LAG*max(0.,-axial.min())-_INTERVAL*max(0.,-axial[-1])
    upper = hold*speed+_LAG*max(0.,axial.max())+_INTERVAL*max(0.,axial[-1])
    core = .41+.015+tail+_LAG*lateral.max()+_INTERVAL*lateral[-1]+.008
    ranges = scans[:,0]
    valid = (scans[:,1] > .5) & (scans[:,2] >= .65)
    points = ranges[:,:,None]*aligned+mount_offsets
    along = points@heading
    across = np.linalg.norm(points-along[:,:,None]*heading,axis=2)
    gap = np.hypot(along-along.clip(lower,upper),across).min(initial=np.inf,where=valid)
    if gap < core:
        return None
    visible = valid & (aligned@heading >= .99) & (along > upper+core+.025)
    sides = aligned@side
    for direction in (1.,-1.):
        bracket = (visible & (direction*sides > 0.)).reshape(3,*_GRID)
        if (bracket.any(axis=1).sum(axis=1) < 3).any():
            return None

    clear = np.zeros(len(rays),dtype=bool)
    verified = 0
    mount = np.asarray(_MOUNT,dtype=rays.dtype)
    for normal, distances in planes:
        plane = distances.min()+mount@normal-max(lower*(heading@normal),upper*(heading@normal))
        if plane < core or abs(np.diff(distances)).max() > .035:
            continue
        outward = rays@normal
        association = abs(ranges*outward-distances[:,None]) <= .018
        clear |= (outward >= .30) & valid.all(axis=0) & association.all(axis=0)
        verified += 1
    return (clear,float(gap),float(core)) if verified >= 3 else None


def _remaining_ratio(cache, config):
    closing = cache['approach_speed_m_s']
    if config.range_rate_filter:
        closing = torch.maximum(closing,(-cache['range_rate_m_s']).clamp_min(0.))
    return closing, (cache['v_safe_m_s']/closing.clamp_min(torch.finfo(closing.dtype).tiny)).clamp(0.,1.)


def certify_tube_request(lidar, state, rays, governed, config, baseline, *, previous_speed_bound):
    """Keep any unidentified constraint and recheck each finite command on new scans."""
    history = state.reshape(4,13)
    command = history[-1,6:9]
    speed = float(command.norm())
    if (bool(baseline.metrics['emergency'][0,0]) or not .03 < speed <= _MAX_REQUEST
            or float(baseline.v_final_b[0]@command) >= .60*speed**2):
        return None
    ages = history[:,12]
    if (bool((ages > _STAMP_LIMIT).any() or (ages.diff().abs() > .03).any())
            or float(history[-3:,:3].norm(dim=1).max()) > .22 or bool(_return_seed(lidar,state)[0][0])):
        return None

    scans = lidar.reshape(4,3,-1)[-3:].detach().cpu().numpy()
    state_values = history.detach().cpu().numpy()
    ray_values = rays.detach().cpu().numpy()
    planes = _tube_planes(scans,state_values,ray_values)
    if not planes:
        return None
    aligned, mounts, issued = _align_scans(rays,history)
    tail = _LAG*max(_MAX_REQUEST,previous_speed_bound)*math.exp(-.30/_LAG)
    heading = command/speed
    decay = math.exp(-_INTERVAL/_LAG)
    previous = issued@heading.detach().cpu().numpy()
    response = float(previous[0])
    for sent in previous:
        response = decay*response+(1.-decay)*sent
    preferred = min(.2999,1.35*speed,max(0.,(.75*speed-decay*response)/(1.-decay)))
    if preferred < .005:
        return None
    target = preferred*heading
    for proposed in (target[None],governed):
        candidate = attenuate_command(lidar,rays,proposed,_INTERVAL,config,state_vec=state)
        if bool(candidate.metrics['emergency'][0,0]) or not 0. < float(proposed.norm()) <= _MAX_REQUEST:
            continue
        active = candidate.cache['active_mask']
        closing, ratios = _remaining_ratio(candidate.cache,config)
        if not bool((active & (ratios < .90)).any()):
            continue
        velocity = proposed[0].detach().cpu().numpy()
        for hold in (.60,.30):
            proved = _bounded_wall_mask(scans,ray_values,aligned,mounts,issued,state_values[-1,3:6],velocity,planes,hold,tail)
            if proved is None:
                continue
            wall_beams, gap, core = proved
            associated = torch.as_tensor(wall_beams,device=rays.device)[None]
            remaining = active & ~associated
            count = remaining.sum(dim=1,keepdim=True)
            beta = ratios.masked_fill(~remaining,1.).amin(dim=1,keepdim=True)
            beta = torch.where(count > 0,beta,torch.full_like(beta,config.empty_active_set_beta))
            projection = float((proposed[0]@heading)*beta[0,0])
            expected = decay*response+(1.-decay)*projection
            if float(beta[0,0]) < .90 or projection <= 0. or max(projection,expected) < .60*speed:
                continue
            margin = (candidate.cache['v_safe_m_s']-closing).masked_fill(~remaining,torch.inf).amin(dim=1,keepdim=True)
            metrics = dict(candidate.metrics,beta=beta,active_beam_count=count.to(proposed.dtype),
                           worst_margin_m_s=torch.where(count > 0,margin,0.))
            result = ICSResult(clip_vector_norm(proposed*beta,config.velocity_limit_m_s),metrics,
                               dict(candidate.cache,active_mask=remaining))
            cleared = int((active & associated & (ratios < .90)).sum())
            return TubeStoppingDecision(result,proposed,hold,gap,core,cleared)
    return None


class TubeStoppingPipeline(ControlCorePipeline):
    """Apply a separately verified finite command after the original pipeline."""

    certify = staticmethod(certify_tube_request)

    def __init__(self, baseline):
        super().__init__(baseline.config)
        self.lidar_rays_body = baseline.lidar_rays_body
        self._anchor = baseline._anchor
        self._last_ics_cache = baseline._last_ics_cache
        self.allow_tube_stopping = False
        self.tube_stopping = None
        self._oldest_issued_speed = self.config.ics.velocity_limit_m_s
        self._speed_at_oldest_scan = self.config.ics.velocity_limit_m_s

    @property
    def oldest_speed_bound(self):
        return self._speed_at_oldest_scan

    def __call__(self, lidar_grid, state_vec, governor_action, history_dt_s, episode_reset,
                 explicit_reset, controller_quaternion_wxyz):
        result = super().__call__(lidar_grid,state_vec,governor_action,history_dt_s,episode_reset,
                                  explicit_reset,controller_quaternion_wxyz)
        self.tube_stopping = None
        oldest_issued = float(state_vec[0,9:12].norm())
        decay = math.exp(-history_dt_s/_LAG)
        self._speed_at_oldest_scan = (decay*self._speed_at_oldest_scan+(1.-decay)*self._oldest_issued_speed)
        self._oldest_issued_speed = oldest_issued
        if bool(episode_reset[0] or explicit_reset[0]):
            self._speed_at_oldest_scan = self.config.ics.velocity_limit_m_s
            return result
        if not self.allow_tube_stopping or not math.isclose(history_dt_s,_INTERVAL,abs_tol=1.e-6):
            return result
        rays = self.lidar_rays_body
        if rays is None:
            rays = mounted_mid360_rays_body(device=lidar_grid.device,dtype=lidar_grid.dtype)
        checked = self.certify(lidar_grid,state_vec,rays,result.v_gov_b,self.config.ics,
                               ICSResult(result.v_final_b,result.ics_metrics,self._last_ics_cache),
                               previous_speed_bound=self._speed_at_oldest_scan)
        if checked is None:
            return result
        certified = checked.result
        self.tube_stopping = (checked,result.v_final_b,result.ics_metrics['beta'])
        self._last_ics_cache = certified.cache
        world = rotate_body_to_world(certified.v_final_b,controller_quaternion_wxyz)
        return replace(result,v_final_b=certified.v_final_b,v_final_w=world,ics_metrics=certified.metrics)
