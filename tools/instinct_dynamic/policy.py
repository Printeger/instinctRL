"""Candidate planar actor and frozen R6 observation adapter."""
import math

import torch
from torch import nn

from .world import ACTIONS, ANGLES, BEAMS, DT, DIRECTIONS, HISTORY, LOOK, action_vectors


def features(obs, mode):
    speed = obs.command.norm(dim=-1)
    heading = torch.where(speed > .01, torch.atan2(obs.command[:,1], obs.command[:,0]), 0.)
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
    def __init__(self, mode, objective):
        super().__init__()
        self.mode = mode
        self.objective = objective
        channels = 2*HISTORY if mode == 'raw' else 4*HISTORY-2
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 24, 5, stride=2, padding=2, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(24, 20, 3, stride=3, padding=1, padding_mode='circular'), nn.SiLU(), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(311, 96), nn.SiLU(), nn.Linear(96,len(ACTIONS)))

    def forward(self, grid, state):
        return self.head(torch.cat((self.encoder(grid), state), dim=-1))

    def act(self, obs):
        score = self(*features(obs, self.mode))
        if self.objective == 'imitate':
            choice = score.argmax(-1)
        else:
            actions = action_vectors(obs.command, obs.previous)
            imitation_cost = (actions-obs.command[:,None]).square().sum(-1)
            inertial_cost = .045*(actions-obs.previous[:,None]).square().sum(-1)
            choice = (imitation_cost + inertial_cost + 3.*score.sigmoid()).argmin(-1)
        return action_vectors(obs.command, obs.previous)[torch.arange(len(choice)), choice]


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
