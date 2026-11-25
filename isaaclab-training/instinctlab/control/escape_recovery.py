"""Test bounded command recovery in the same governor and ICS action space."""

import torch

from .escape_option import DURATION_STEPS, SPEED_M_S, EscapeOption
from .governor import decode_governor, extract_latest_commands
from .ics import attenuate_command


MIXED_VERSION = 'm4-escape-mixed-v1'
RECOVERY_VERSION = 'm4-escape-recovery-v1'
LATERAL_VERSION = 'm4-escape-lateral-recovery-v1'
RECOVERY_DELAY = 28
RECOVERY_LIMIT = 65


class EscapeRecovery(EscapeOption):
    def __init__(self, *, lanes, device, dtype, recenter, lateral=False):
        super().__init__(lanes=lanes, device=device, dtype=dtype)
        self.recenter = recenter
        self.lateral = lateral
        self.since_escape = torch.full_like(self.remaining, RECOVERY_LIMIT)
        self.recovering = torch.zeros_like(self.applied)

    def reset(self, lanes):
        super().reset(lanes)
        self.since_escape[lanes] = RECOVERY_LIMIT
        self.recovering[lanes] = False

    def issued(self, lidar, state, action, config):
        request = decode_governor(action, state, config.governor).v_gov_b
        return attenuate_command(lidar, self.detector._full_rays, request, .1, config.ics)

    def select_lateral(self, lidar, state, selected, config):
        # Keep the same trigger and hold. Select using the command this path can
        # actually issue at the escape speed, including the shared emergency stop.
        lanes = (self.applied & (self.remaining == DURATION_STEPS - 1)).nonzero().flatten()
        if not len(lanes):
            return
        local_lidar, local_state = lidar[lanes], state[lanes]
        retreat = self.detector.retreat_direction(local_lidar, local_state, hover_only=False)
        side = torch.stack((-retreat[:, 1], retreat[:, 0], retreat[:, 2]), dim=1)
        baseline = self.issued(local_lidar, local_state, selected[lanes], config).v_final_b
        best_score = (baseline * side).sum(1).abs() + .035
        best_direction = self.direction[lanes]
        for axial in (0., .45, .85):
            for sign in (1., -1.):
                direction = (sign * side + axial * retreat) / (1. + axial**2)**.5
                trial = torch.cat((torch.zeros_like(local_state[:, :1]), .5 + SPEED_M_S * direction), dim=1)
                issued = self.issued(local_lidar, local_state, trial, config).v_final_b
                score = (issued * side).sum(1).abs()
                better = score > best_score
                best_direction = torch.where(better[:, None], direction, best_direction)
                best_score = torch.where(better, score, best_score)
        self.direction[lanes] = best_direction
        selected[lanes, 1:] = .5 + SPEED_M_S * best_direction

    @torch.no_grad()
    def apply_recovery(self, policy, action, config):
        selected = super().apply(policy, action)
        lidar, state = policy['lidar_grid'], policy['state_vec']
        if self.lateral:
            self.select_lateral(lidar, state, selected, config)
        command, _ = extract_latest_commands(state)
        speed = command.norm(dim=1)
        moving = speed > .03
        self.since_escape = torch.where(self.applied, 0, (self.since_escape + 1).clamp_max(RECOVERY_LIMIT))
        self.recovering.zero_()
        mixed = (self.applied & moving).nonzero().flatten()
        if len(mixed):
            base = selected[mixed]
            trial = base.clone()
            trial[:, 0] = .25
            previous = self.issued(lidar[mixed], state[mixed], base, config).v_final_b
            issued = self.issued(lidar[mixed], state[mixed], trial, config).v_final_b
            lateral = self.direction[mixed]
            preserves_escape = (issued * lateral).sum(1) >= (previous * lateral).sum(1) - 1.e-5
            gains_forward = ((issued - previous) * command[mixed]).sum(1) > .002 * speed[mixed]
            selected[mixed] = torch.where((preserves_escape & gains_forward)[:, None], trial, base)
        if not self.recenter:
            return selected

        lanes = (moving & (state[:, -1] <= .22) & (self.since_escape >= RECOVERY_DELAY)
                 & (self.since_escape < RECOVERY_LIMIT)).nonzero().flatten()
        if not len(lanes):
            return selected

        local_lidar, local_state = lidar[lanes], state[lanes]
        heading = command[lanes] / speed[lanes, None]
        original = self.issued(local_lidar, local_state, selected[lanes], config)
        progress = (original.v_final_b * heading).sum(1)
        has_room = original.metrics['emergency'].flatten() == 0
        blocked = (progress < .03) & has_room
        if not bool(blocked.any()):
            return selected

        retreat = self.detector.retreat_direction(local_lidar, local_state, hover_only=False)
        eligible = blocked & ~((retreat * heading).sum(1) < -.25)
        if not bool(eligible.any()):
            return selected

        lanes, heading = lanes[eligible], heading[eligible]
        local_lidar, local_state = lidar[lanes], state[lanes]
        side = torch.stack((-heading[:, 1], heading[:, 0], torch.zeros_like(heading[:, 0])), dim=1)
        side /= side.norm(dim=1, keepdim=True).clamp_min(.1)
        best = selected[lanes]
        best_score = progress[eligible].clamp_min(0.) + .015
        for alpha, offset in ((1., 0.), (1., .12), (1., -.12), (.65, .22), (.65, -.22)):
            trial = torch.cat((torch.full_like(best[:, :1], alpha), .5 + offset * side), dim=1)
            issued = self.issued(local_lidar, local_state, trial, config).v_final_b
            score = (issued * heading).sum(1) - .18 * (issued * side).sum(1).abs()
            better = score > best_score
            best = torch.where(better[:, None], trial, best)
            best_score = torch.where(better, score, best_score)
            self.recovering[lanes] |= better
        selected[lanes] = best
        return selected
