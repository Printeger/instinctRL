"""One-shot persistence intervention for the frozen R3 physical comparison."""

import torch


HOLD_STEPS = 25
POLICY_PERIOD_S = .1


class EscapeHoldProbe:
    def __init__(self, state):
        self.remaining = torch.zeros(len(state), device=state.device, dtype=torch.long)
        self.started = torch.zeros_like(self.remaining, dtype=torch.bool)
        self.bias = torch.zeros_like(state[:, :3])

    def reset(self, terminal):
        self.remaining[terminal] = 0
        self.started[terminal] = False
        self.bias[terminal] = 0

    @torch.no_grad()
    def __call__(self, module, inputs, shift):
        _, state = inputs
        fresh = state[:, -1] <= .22
        hover = state[:, 45:48].norm(dim=1) <= .001
        start = ~self.started & fresh & hover & (shift[:, 1:].norm(dim=1) > .001)
        self.started |= start
        self.remaining = torch.where(start, HOLD_STEPS, self.remaining)
        active = (self.remaining > 0) & fresh & hover

        # Transport the planar direction between body frames with the latest
        # measured yaw rate. The actual ICS still screens the full action.
        angle = -POLICY_PERIOD_S * state[:, 39 + 2]
        x, y, z = self.bias.unbind(dim=1)
        rotated = torch.stack((x * angle.cos() - y * angle.sin(),
                               x * angle.sin() + y * angle.cos(), z), dim=1)
        self.bias = torch.where(start[:, None], shift[:, 1:], rotated)
        self.bias = torch.where(active[:, None], self.bias, 0.)
        self.remaining = torch.where(active, self.remaining - 1, 0)
        return torch.cat((shift[:, :1], torch.where(active[:, None], self.bias, shift[:, 1:])), dim=1)
