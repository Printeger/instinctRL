"""Test bounded command recovery in the same governor and ICS action space."""

import torch

from .escape_option import DURATION_STEPS, SPEED_M_S, EscapeOption
from .governor import decode_governor, extract_latest_commands
from .ics import attenuate_command


MIXED_VERSION = 'm4-escape-mixed-v1'
RECOVERY_VERSION = 'm4-escape-recovery-v1'
LATERAL_VERSION = 'm4-escape-lateral-recovery-v1'
FAST_LATERAL_VERSION = 'm4-escape-lateral-recovery-v2'
MOUNT_25_VERSION = 'm4-escape-lateral-recovery-p25-v1'
MOUNT_28_VERSION = 'm4-escape-lateral-recovery-p28-v1'
REDIRECT_25_VERSION = 'm4-escape-redirect-p25-v1'
FALLBACK_25_VERSION = 'm4-escape-fallback-p25-v1'
FRONT_SWITCH_25_VERSION = 'm4-escape-front-switch-p25-v1'
REAR_SWITCH_25_VERSION = 'm4-escape-rear-switch-p25-v1'
REAR_CLEARANCE_25_VERSION = 'm4-escape-rear-clearance-p25-v1'
MOUNT_PITCH_DEGREES = {MOUNT_25_VERSION: 25, MOUNT_28_VERSION: 28,
                       REDIRECT_25_VERSION: 25, FALLBACK_25_VERSION: 25, FRONT_SWITCH_25_VERSION: 25,
                       REAR_SWITCH_25_VERSION: 25, REAR_CLEARANCE_25_VERSION: 25}
FAST_LATERAL_SPEED_M_S = .45
RECOVERY_DELAY = 28
RECOVERY_LIMIT = 65


class EscapeRecovery(EscapeOption):
    def __init__(self, *, lanes, device, dtype, recenter, lateral=False, fast_lateral=False, redirect=False,
                 fallback=False, front_switch=False, rear_switch=False, rear_clearance=False):
        super().__init__(lanes=lanes, device=device, dtype=dtype)
        self.recenter = recenter
        self.lateral = lateral
        self.fast_lateral = fast_lateral
        self.redirect = redirect
        self.fallback = fallback
        self.front_switch = front_switch
        self.front_switched = torch.zeros_like(self.applied)
        self.rear_switch = rear_switch
        self.rear_clearance = rear_clearance
        self.rear_switched = torch.zeros_like(self.applied)
        self.escape_speed = self.direction.new_full((lanes,), SPEED_M_S)
        self.retreat_basis = torch.zeros_like(self.direction)
        self.redirected = torch.zeros_like(self.applied)
        self.since_escape = torch.full_like(self.remaining, RECOVERY_LIMIT)
        self.recovering = torch.zeros_like(self.applied)

    def reset(self, lanes):
        super().reset(lanes)
        self.since_escape[lanes] = RECOVERY_LIMIT
        self.recovering[lanes] = False
        self.escape_speed[lanes] = SPEED_M_S
        self.retreat_basis[lanes] = 0.
        self.redirected[lanes] = False
        self.front_switched[lanes] = False
        self.rear_switched[lanes] = False

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
        if self.redirect:
            self.retreat_basis[lanes] = retreat
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
        if self.rear_clearance:
            best_direction = self.choose_rear_clearance(local_lidar, local_state, retreat, side, best_direction, config)
        self.direction[lanes] = best_direction
        selected[lanes, 1:] = .5 + SPEED_M_S * best_direction
        if self.fast_lateral:
            self.escape_speed[lanes] = SPEED_M_S
            command, _ = extract_latest_commands(local_state)
            rear_hover = ((command.norm(dim=1) <= .001) & (retreat[:, 0] > .65)
                          & ((side * best_direction).sum(1).abs() > .5))
            if bool(rear_hover.any()):
                trial = torch.cat((torch.zeros_like(local_state[:, :1]),
                                   .5 + FAST_LATERAL_SPEED_M_S * best_direction), dim=1)
                issued = self.issued(local_lidar[rear_hover], local_state[rear_hover], trial[rear_hover], config).v_final_b
                useful = (issued * side[rear_hover]).sum(1).abs() > best_score[rear_hover] + .035
                self.escape_speed[lanes[rear_hover][useful]] = FAST_LATERAL_SPEED_M_S

    def choose_rear_clearance(self, lidar, state, retreat, side, direction, config):
        from .escape_clearance import wider_side

        command, _ = extract_latest_commands(state)
        previous_side = (side * direction).sum(dim=1)
        eligible = (command.norm(dim=1) <= .001) & (retreat[:, 0] > .65) & (previous_side.abs() > .5)
        lanes = eligible.nonzero().flatten()
        if not len(lanes):
            return direction
        sign = wider_side(lidar[lanes], state[lanes], self.detector._full_rays, retreat[lanes], side[lanes])
        changed = sign * previous_side[lanes] < 0.
        lanes, sign = lanes[changed], sign[changed]
        if not len(lanes):
            return direction
        preferred = side[lanes] * sign[:, None]
        trial = torch.cat((torch.zeros_like(state[lanes, :1]), .5 + SPEED_M_S * preferred), dim=1)
        command = self.issued(lidar[lanes], state[lanes], trial, config)
        viable = ((command.v_final_b * preferred).sum(dim=1) > .08) & (command.metrics['emergency'].flatten() == 0.)
        direction[lanes[viable]] = preferred[viable]
        return direction

    def redirect_clipped(self, lidar, state, selected, config):
        self.redirected.zero_()
        continuing = self.applied & (self.remaining < DURATION_STEPS - 1)
        if self.rear_switch:
            self.rear_switched &= continuing
        x, y, z = self.retreat_basis.unbind(dim=1)
        yaw = -.1 * state[:, 41]
        rotated = torch.stack((x*yaw.cos() - y*yaw.sin(), x*yaw.sin() + y*yaw.cos(), z), dim=1)
        self.retreat_basis = torch.where(continuing[:, None], rotated, self.retreat_basis)
        command, _ = extract_latest_commands(state)
        lanes = (self.applied & (command.norm(dim=1) <= .001) & (self.retreat_basis[:, 0] > .65)
                 & (self.escape_speed >= FAST_LATERAL_SPEED_M_S)).nonzero().flatten()
        if not len(lanes):
            return
        original = self.issued(lidar[lanes], state[lanes], selected[lanes], config)
        blocked = (original.metrics['beta'].flatten() == 0.) & (original.metrics['emergency'].flatten() == 0.)
        lanes = lanes[blocked]
        if not len(lanes):
            return
        retreat, direction = self.retreat_basis[lanes], self.direction[lanes]
        side = torch.stack((-retreat[:, 1], retreat[:, 0], retreat[:, 2]), dim=1)
        signed_side = side * (direction * side).sum(dim=1, keepdim=True).sign()
        proposal = direction + .45 * retreat
        proposal /= proposal.norm(dim=1, keepdim=True).clamp_min(.01)
        trial = torch.cat((selected[lanes, :1], .5 + self.escape_speed[lanes, None] * proposal), dim=1)
        issued = self.issued(lidar[lanes], state[lanes], trial, config).v_final_b
        releases_side = (issued * signed_side).sum(dim=1) > .08
        selected[lanes[releases_side]] = trial[releases_side]
        self.redirected[lanes[releases_side]] = True
        if self.fallback:
            unresolved = ~releases_side & (signed_side.norm(dim=1) > .5)
            self.try_steeper_retreat(lidar, state, selected, config, lanes[unresolved], signed_side[unresolved])
            if self.rear_switch:
                switch = unresolved & continuing[lanes] & ~self.redirected[lanes] & ~self.rear_switched[lanes]
                self.try_rear_opposite(lidar,state,selected,config,lanes[switch], -signed_side[switch])

    def try_steeper_retreat(self, lidar, state, selected, config, lanes, side):
        # Search only while the held action and the original redirection both
        # fail. Preserve side commitment and use the least axial legal trial.
        for axial in (.85, 1.4, 2.2):
            if not len(lanes):
                break
            direction = (side + axial * self.retreat_basis[lanes]) / (1. + axial**2)**.5
            trial = torch.cat((selected[lanes, :1], .5 + self.escape_speed[lanes, None] * direction), dim=1)
            result = self.issued(lidar[lanes], state[lanes], trial, config)
            viable = ((result.v_final_b * side).sum(dim=1) > .08)
            viable &= result.metrics['emergency'].flatten() == 0.
            selected[lanes[viable]] = trial[viable]
            self.redirected[lanes[viable]] = True
            lanes, side = lanes[~viable], side[~viable]

    def try_rear_opposite(self, lidar, state, selected, config, lanes, opposite):
        """Commit a single previously untried side when the rear option is blocked."""
        for axial in (1.4, 2.2):
            if not len(lanes):
                break
            direction = (opposite + axial*self.retreat_basis[lanes])/(1.+axial**2)**.5
            trial = torch.cat((selected[lanes, :1], .5+self.escape_speed[lanes, None]*direction), dim=1)
            result = self.issued(lidar[lanes],state[lanes],trial,config)
            viable = ((result.v_final_b*opposite).sum(dim=1)>.08) & (result.metrics['emergency'].flatten()==0.)
            used = lanes[viable]
            selected[used] = trial[viable]
            self.direction[used] = direction[viable]
            self.rear_switched[used] = self.redirected[used] = True
            lanes, opposite = lanes[~viable], opposite[~viable]

    def switch_blocked_front(self, lidar, state, selected, config):
        # Give a blocked frontal hover response one new side. A chosen
        # feasible side stays committed until the current option ends.
        continuing = self.applied & (self.remaining < DURATION_STEPS - 1)
        self.front_switched &= continuing
        command, _ = extract_latest_commands(state)
        lanes = (continuing & ~self.front_switched & (command.norm(dim=1) <= .001)
                 & (self.retreat_basis[:, 0] < -.65)).nonzero().flatten()
        if not len(lanes):
            return
        retreat = self.retreat_basis[lanes]
        side = torch.stack((-retreat[:, 1], retreat[:, 0], retreat[:, 2]), dim=1)
        signed = (side * self.direction[lanes]).sum(dim=1)
        original = self.issued(lidar[lanes], state[lanes], selected[lanes], config)
        blocked = ((original.metrics['beta'].flatten() == 0.)
                   & (original.metrics['emergency'].flatten() == 0.) & (signed.abs() > .5))
        lanes = lanes[blocked]
        if not len(lanes):
            return
        opposite = -signed[blocked, None].sign() * side[blocked]
        trial = torch.cat((selected[lanes, :1], .5 + self.escape_speed[lanes, None] * opposite), dim=1)
        issued = self.issued(lidar[lanes], state[lanes], trial, config)
        viable = ((issued.v_final_b * opposite).sum(dim=1) > .08) & (issued.metrics['emergency'].flatten() == 0.)
        used = lanes[viable]
        self.direction[used] = opposite[viable]
        selected[used] = trial[viable]
        self.front_switched[used] = self.redirected[used] = True

    @torch.no_grad()
    def apply_recovery(self, policy, action, config):
        selected = super().apply(policy, action)
        lidar, state = policy['lidar_grid'], policy['state_vec']
        if self.lateral:
            self.select_lateral(lidar, state, selected, config)
        if self.fast_lateral:
            held = self.applied.nonzero().flatten()
            selected[held, 1:] = .5 + self.escape_speed[held, None] * self.direction[held]
        if self.redirect:
            self.redirect_clipped(lidar, state, selected, config)
        if self.front_switch:
            self.switch_blocked_front(lidar, state, selected, config)
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
