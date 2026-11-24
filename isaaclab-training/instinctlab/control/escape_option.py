"""Finite side-step from delivered M4 measurements, before governor and ICS."""

import torch

from instinctlab.training.hover_reflex import HoverReflex


METHOD_FIELD = 'r6_action_option'
METHOD_VERSION = 'm4-escape-option-v1'
DURATION_STEPS = 25
COOLDOWN_STEPS = 10
SPEED_M_S = .30


class EscapeOption:
    def __init__(self, *, lanes, device, dtype):
        self.detector = HoverReflex(device=device, dtype=dtype)
        self.remaining = torch.zeros(lanes, device=device, dtype=torch.long)
        self.cooldown = torch.zeros_like(self.remaining)
        self.direction = torch.zeros((lanes, 3), device=device, dtype=dtype)
        self.applied = torch.zeros(lanes, device=device, dtype=torch.bool)

    def reset(self, lanes):
        self.remaining[lanes] = 0
        self.cooldown[lanes] = 0
        self.direction[lanes] = 0

    @torch.no_grad()
    def apply(self, policy, action):
        lidar, state = policy['lidar_grid'], policy['state_vec']
        command = state[:, 45:48]
        fresh = state[:, -1] <= .22
        continuing = (self.remaining > 0) & fresh
        available = (self.remaining == 0) & (self.cooldown == 0) & fresh
        start = torch.zeros_like(available)
        if bool(available.any()):
            lanes = available.nonzero().flatten()
            retreat = self.detector.retreat_direction(lidar[lanes], state[lanes], hover_only=False)
            hover = command[lanes].norm(dim=1) <= .001
            towards = (command[lanes] * retreat).sum(dim=1) < -.08
            threats = (hover | towards) & (retreat.norm(dim=1) > .1)
            if bool(threats.any()):
                lanes = lanes[threats]
                direction, authority = self.detector.choose_escape(lidar[lanes], state[lanes], retreat[threats])
                feasible = authority >= .25
                start[lanes[feasible]] = True
                self.direction[lanes[feasible]] = direction[feasible]

        # Update the held lateral direction in the current body frame using
        # gyro yaw only. No world pose or inferred linear velocity is needed.
        yaw = -.1 * state[:, 41]
        x, y, z = self.direction.unbind(dim=1)
        rotated = torch.stack((x*yaw.cos() - y*yaw.sin(), x*yaw.sin() + y*yaw.cos(), z), dim=1)
        self.direction = torch.where(continuing[:, None], rotated, self.direction)
        self.applied = start | continuing
        self.remaining = torch.where(start, DURATION_STEPS, self.remaining)
        self.remaining = torch.where(self.applied, self.remaining - 1, 0)
        self.cooldown = torch.where(self.applied, COOLDOWN_STEPS, (self.cooldown - 1).clamp_min(0))
        escape_action = torch.cat((torch.zeros_like(action[:, :1]), .5 + SPEED_M_S * self.direction), dim=1)
        return torch.where(self.applied[:, None], escape_action, action)
