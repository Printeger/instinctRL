"""Bounded output bias from observed range change during hover."""

import torch
from torch import nn
from torch.nn import functional as F

from instinctlab.control.ics import mounted_mid360_rays_body


HOVER_REFLEX_METHOD = 'm4-hover-reflex-v1'
HOVER_REFLEX_FIELD = 'r6_actor_reflex'
HOVER_REFLEX_PARAMETER = 'hover_reflex.gain'
GAIN_SCALE = 128.


class HoverReflex(nn.Module):
    def __init__(self, *, device, dtype):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        rays = mounted_mid360_rays_body(device=device, dtype=dtype)
        self.register_buffer('_rays', rays.reshape(59, 360, 3)[::3].reshape(-1, 3), persistent=False)

    @torch.no_grad()
    def retreat_direction(self, lidar, state):
        history = state.reshape(-1, 4, 13)
        scan = lidar[:, :, ::3]
        latest = scan[:, 9]
        weight = torch.zeros_like(latest)
        for k in range(3):
            gap = (3 - k) * .1 + history[:, k, 12] - history[:, -1, 12]
            gap = gap[:, None, None]
            old = lidar[:, k * 3:k * 3 + 1]
            reliable = lidar[:, k * 3 + 2:k * 3 + 3] > .55
            negative = -torch.where(reliable, old, torch.inf)
            # A static feature may move across beams during a small body turn.
            # Require closure even relative to the nearest old local return.
            negative = F.pad(negative, (8, 8, 0, 0), mode='circular')
            negative = F.max_pool2d(negative, (1, 17), stride=1)
            neighborhood = -F.max_pool2d(negative, (17, 1), stride=1, padding=(8, 0))[:, 0, ::3]
            difference = neighborhood - latest
            coherent = (gap >= .16) & (gap <= .33) & (difference > .20 * gap) & (difference < 1.5 * gap)
            coherent &= neighborhood.isfinite() & (latest < 1.9) & (scan[:, 11] > .55)
            supported = coherent.clone()
            for offset in (-2, -1, 1, 2):
                supported &= coherent.roll(offset, dims=-1)
            weight = torch.maximum(weight, torch.where(supported, difference, 0.))
        bearing = weight.flatten(1) @ self._rays
        bearing[:, 2] = 0.
        fresh = history[:, -1, 12] <= .22
        stable = history[:, :, :3].norm(dim=2).amax(dim=1) < .22
        hover = state[:, 45:48].norm(dim=1) <= .001
        return -bearing / bearing.norm(dim=1, keepdim=True).clamp_min(1.e-6) * (fresh & stable & hover)[:, None]

    def forward(self, lidar, state):
        direction = self.retreat_direction(lidar, state)
        # Keep the common optimizer and learning rate while allowing one
        # scalar to change the bounded logit bias over this training budget.
        shift = (GAIN_SCALE * self.gain).clamp(0., 8.) * direction
        return torch.cat((torch.zeros_like(shift[:, :1]), shift), dim=1)

    @torch.no_grad()
    def project_gain(self):
        # Preserve a gradient at zero even after an optimizer step points
        # outside the permitted interval.
        self.gain.clamp_(0., 8. / GAIN_SCALE)
