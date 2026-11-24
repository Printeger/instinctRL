"""Candidate planar actor and frozen R6 observation adapter."""
import math

import torch
from torch import nn

from .world import ACTIONS, ANGLES, BEAMS, DT, DIRECTIONS, HISTORY, LOOK, action_vectors


def command_heading(obs, threat_reference=False):
    speed = obs.command.norm(dim=-1)
    commanded = torch.atan2(obs.command[:,1], obs.command[:,0])
    if not threat_reference:
        return torch.where(speed > .01, commanded, 0.)
    age = (obs.clock[:, -1:] - obs.acquisition)[..., 0]
    rotated = ANGLES[None,None] + (obs.gyro[...,0] * age)[..., None]
    idx = (rotated*(BEAMS/(2*math.pi))).round().long() % BEAMS
    ranges, valid = obs.ranges.gather(-1,idx), obs.valid.gather(-1,idx)
    gap = (obs.acquisition[:, -1:] - obs.acquisition[:, :-1])[..., 0]
    eligible = (gap > .06) & (gap <= .32)
    best_gap, previous = torch.where(eligible, gap, 0.).max(dim=1)
    lane = torch.arange(len(speed))
    closure = (ranges[lane,previous] - ranges[:,-1]) / best_gap[:,None].clamp_min(.06)
    credible = valid[lane,previous] & valid[:,-1] & (best_gap[:,None] > 0.) & (closure <= 2.5)
    potential = torch.where(credible, (closure-.16).clamp_min(0.) * (-ranges[:,-1]/1.2).exp(), 0.)
    coherent = torch.minimum(potential, torch.maximum(potential.roll(1,-1),potential.roll(-1,-1)))
    risk, bearing = coherent.max(dim=1)
    neutral = torch.where((risk > .035) & (age[:,-1] <= .22), ANGLES[bearing], 0.)
    return torch.where(speed > .01, commanded, neutral)


def features(obs, mode, heading=None):
    speed = obs.command.norm(dim=-1)
    heading = command_heading(obs) if heading is None else heading
    age = (obs.clock[:, -1:] - obs.acquisition)[..., 0]
    # This is relative rotation within a measurement window, with no world heading.
    offset = heading[:, None] + (obs.gyro[..., 0] * age if mode != 'raw' else 0.)
    bins = (ANGLES[None, None] + offset[..., None]) * (BEAMS/(2*math.pi))
    indices = bins.round().long() % BEAMS
    ranges = obs.ranges.gather(-1, indices.expand(-1, HISTORY, -1))
    valid = obs.valid.gather(-1, indices.expand(-1, HISTORY, -1))
    temporal = [((2.5-ranges)/2.5).clamp(0.,1.) * valid, valid.float()]
    if mode != 'raw':
        gap = (obs.acquisition[:, 1:] - obs.acquisition[:, :-1])[..., 0]
        different = (gap > .015) & (gap <= .25)
        closure = (ranges[:, :-1] - ranges[:, 1:]) / gap[..., None].clamp_min(.015)
        usable = different[...,None] & valid[:,1:] & valid[:,:-1]
        temporal.extend(((closure / 1.5).clamp(-1.,1.) * usable,
                         ((ranges[:,1:] < 1.5) & usable).float()))
    prev_heading = torch.stack((heading.cos(), heading.sin()), -1)
    prev_lateral = torch.stack((-heading.sin(), heading.cos()), -1)
    state = torch.cat((age / .3, obs.gyro[...,0] / .5,
                       speed[:, None], (obs.previous * prev_heading).sum(-1,keepdim=True),
                       (obs.previous * prev_lateral).sum(-1,keepdim=True)), dim=1)
    grid = torch.cat([value.reshape(len(speed), -1, BEAMS) for value in temporal], dim=1)
    return grid, state


class Policy(nn.Module):
    def __init__(self, mode, objective, threat_reference=False):
        super().__init__()
        self.mode = mode
        self.objective = objective
        self.threat_reference = threat_reference
        channels = 2*HISTORY if mode == 'raw' else 4*HISTORY-2
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 24, 5, stride=2, padding=2, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(24, 20, 3, stride=3, padding=1, padding_mode='circular'), nn.SiLU(), nn.Flatten())
        outputs = 2 * len(ACTIONS) if objective == 'hybrid' else len(ACTIONS)
        self.head = nn.Sequential(nn.Linear(311, 96), nn.SiLU(), nn.Linear(96, outputs))

    def forward(self, grid, state):
        return self.head(torch.cat((self.encoder(grid), state), dim=-1))

    def act(self, obs):
        heading = command_heading(obs, self.threat_reference)
        score = self(*features(obs, self.mode, heading))
        actions = action_vectors(obs.command, obs.previous, heading=heading)
        if self.objective == 'imitate':
            choice = score.argmax(-1)
        else:
            imitation_cost = (actions-obs.command[:,None]).square().sum(-1)
            inertial_cost = .045*(actions-obs.previous[:,None]).square().sum(-1)
            if self.objective == 'hybrid':
                logits, score = score.chunk(2, dim=-1)
                preferred = logits.argmax(-1)
                risk = score.sigmoid()
                alternate = (imitation_cost + inertial_cost + 3.*risk).argmin(-1)
                lane = torch.arange(len(preferred))
                danger = (risk[lane, preferred] > .20) & (risk[lane, preferred] - risk[lane, alternate] > .08)
                choice = torch.where(danger, alternate, preferred)
            else:
                choice = (imitation_cost + inertial_cost + 3.*score.sigmoid()).argmin(-1)
        return actions[torch.arange(len(choice)), choice]


def clearance_guard(obs, proposal):
    """Current-scan swept strip for all research actors. Dynamic cues stay with the actor."""
    delta = proposal.norm(dim=-1, keepdim=True)
    direction = proposal / delta.clamp_min(.001)
    projection = direction @ DIRECTIONS.T
    depth = obs.ranges[:, -1]
    age = obs.clock[:, -1, 0] - obs.acquisition[:, -1, 0]
    finite = obs.valid[:,-1] & (projection > 0.)
    transverse = depth * (1 - projection.square()).clamp_min(0.).sqrt()
    along = depth * projection - .33 - .65*age[:,None]
    stop_distance = torch.where(finite & (transverse < .32), along, torch.inf).amin(1, keepdim=True)
    limit = (2. * 1.2 * stop_distance.clamp_min(0.)).sqrt()
    return proposal * (limit / delta.clamp_min(.001)).clamp(max=1.)


class FrozenR6:
    """R2 on an azimuth-projected 3D scan, with its exact network and ICS decode."""
    def __init__(self, checkpoint):
        from instinctlab.control.ics import ICSConfig, mounted_mid360_rays_body
        from instinctlab.training.networks import HistoryActor, NetworkConfig

        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.net = HistoryActor(NetworkConfig(**payload['network_config'])).eval()
        self.net.load_state_dict(payload['actor'], strict=True)
        self.rays = mounted_mid360_rays_body()
        self.config = ICSConfig(clearance_margin_m=.15)
        self.planar = self.rays[:,:2].norm(dim=-1).clamp_min(.001)
        self.boundary = 1.5/self.rays[:, 2].abs().clamp_min(.001)
        azimuth = torch.atan2(self.rays[:,1], self.rays[:,0])
        self.indices = (azimuth * (BEAMS/(2*math.pi))).round().long() % BEAMS

    def act(self, obs):
        from instinctlab.control.ics import attenuate_command
        from instinctlab.control.governor import decode_governor

        n = obs.ranges.shape[0]
        issued = []
        for i in range(0,n,16):
            cut = slice(i,i+16)
            age = (obs.clock[cut,-1:] - obs.acquisition[cut]).clamp_min(0.)
            hit = torch.where(obs.valid[cut,:, self.indices], obs.ranges[cut,:, self.indices]/self.planar, torch.inf)
            distance = torch.minimum(hit, self.boundary).clamp_min(.1)
            valid = distance <= 40.
            grid = torch.stack((distance.clamp(max=40.),valid,valid*(-age/.5).exp()),dim=2).reshape(-1,12,59,360)
            state = torch.zeros(len(grid), 4, 13)
            state[:,:,2:3] = obs.gyro[cut]
            state[:,:,5] = -1.
            state[:,:,6:8] = obs.command_history[cut]
            state[:,:,9:11] = obs.previous_history[cut]
            state[:,:,12:13] = age
            model_input = dict(lidar_grid=grid, state_vec=state.flatten(1))
            action = self.net(model_input).mean_action
            command = decode_governor(action, model_input['state_vec']).v_gov_b
            issued.append(attenuate_command(grid, self.rays, command, DT, self.config).v_final_b[:,:2])
        return torch.cat(issued)
