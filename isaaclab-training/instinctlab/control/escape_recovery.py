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
FRONT_CLEARANCE_25_VERSION = 'm4-escape-front-clearance-p25-v1'
BRAKE_FORWARD_25_VERSION = 'm4-escape-brake-forward-p25-v1'
BRAKE_TUBE_25_VERSION = 'm4-escape-brake-tube-p25-v1'
BRAKE_TUBE_CONFIRM_25_VERSION = 'm4-escape-brake-tube-p25-v2'
RECOVERY_RETREAT_25_VERSION = 'm4-escape-brake-tube-p25-recovery-retreat-v1'
EARLY_RETREAT_25_VERSION = 'm4-escape-brake-tube-p25-early-retreat-v1'
SIDE_HOVER_RETREAT_25_VERSION = 'm4-escape-brake-tube-p25-side-hover-v1'
CLEAR_PASS_25_VERSION = 'm4-escape-brake-tube-p25-clear-pass-v1'
MOUNT_PITCH_DEGREES = {MOUNT_25_VERSION: 25, MOUNT_28_VERSION: 28,
                       REDIRECT_25_VERSION: 25, FALLBACK_25_VERSION: 25, FRONT_SWITCH_25_VERSION: 25,
                       REAR_SWITCH_25_VERSION: 25, REAR_CLEARANCE_25_VERSION: 25, FRONT_CLEARANCE_25_VERSION: 25,
                       BRAKE_FORWARD_25_VERSION: 25, BRAKE_TUBE_25_VERSION: 25, BRAKE_TUBE_CONFIRM_25_VERSION: 25,
                       RECOVERY_RETREAT_25_VERSION: 25, EARLY_RETREAT_25_VERSION: 25,
                       SIDE_HOVER_RETREAT_25_VERSION: 25, CLEAR_PASS_25_VERSION: 25}
FAST_LATERAL_SPEED_M_S = .45
RECOVERY_DELAY = 28
RECOVERY_LIMIT = 65


class EscapeRecovery(EscapeOption):
    def __init__(self, *, lanes, device, dtype, recenter, lateral=False, fast_lateral=False, redirect=False,
                 fallback=False, front_switch=False, rear_switch=False, rear_clearance=False, front_clearance=False,
                 brake_forward_escape=False, recovery_retreat=False, early_retreat=False, side_hover_retreat=False,
                 clear_pass=False):
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
        self.front_clearance = front_clearance
        self.brake_forward_escape = brake_forward_escape
        self.braking = torch.zeros_like(self.applied)
        self.recovery_retreat = recovery_retreat
        self.early_retreat = early_retreat
        self.side_hover_retreat = side_hover_retreat
        self.clear_pass = clear_pass
        if clear_pass:
            self.holding = torch.zeros_like(self.applied)
        self.retreat_hold_steps = torch.zeros_like(self.remaining)
        self.retreat_hold_direction = torch.zeros_like(self.direction)
        self.retreating = torch.zeros_like(self.applied)
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
        self.braking[lanes] = False
        self.retreat_hold_steps[lanes] = 0
        self.retreat_hold_direction[lanes] = 0.
        self.retreating[lanes] = False

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
        if self.front_clearance:
            best_direction = self.choose_front_clearance(local_lidar, local_state, retreat, side, best_direction, config)
        if self.side_hover_retreat:
            best_direction = self.choose_side_hover_retreat(local_lidar, local_state, retreat, best_direction, config)
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

    def choose_front_clearance(self, lidar, state, retreat, side, direction, config):
        command, _ = extract_latest_commands(state)
        lateral = (direction * side).sum(dim=1)
        lanes = ((command.norm(dim=1) <= .001) & (retreat[:, 0] < -.65) & (lateral.abs() > .85)).nonzero().flatten()
        if not len(lanes):
            return direction
        baseline = side[lanes] * lateral[lanes, None].sign()
        gain = torch.zeros_like(state[lanes, :1])
        left, right = (self.issued(lidar[lanes], state[lanes], torch.cat((gain, .5 + sign*SPEED_M_S*baseline), dim=1),
                                   config).metrics for sign in (1., -1.))
        unattenuated = (left['beta'] == 1.) & (right['beta'] == 1.) & (left['emergency'] == 0.) & (right['emergency'] == 0.)
        switch = (unattenuated & (left['active_beam_count'] > 0.) & (right['active_beam_count'] == 0.)).flatten()
        direction[lanes[switch]] = -baseline[switch]
        return direction

    def choose_side_hover_retreat(self, lidar, state, retreat, direction, config):
        command, _ = extract_latest_commands(state)
        alignment = (direction*retreat).sum(dim=1)
        lanes = ((command.norm(dim=1) <= .001) & (retreat[:, 0].abs() < .45)
                 & (retreat.norm(dim=1) > .1) & (alignment < .35)).nonzero().flatten()
        if not len(lanes):
            return direction
        away = retreat[lanes]
        action = torch.cat((torch.zeros_like(away[:, :1]), .5 + SPEED_M_S*away), dim=1)
        result = self.issued(lidar[lanes], state[lanes], action, config)
        viable = ((result.v_final_b * away).sum(1) > .08) & (result.metrics['emergency'].flatten() == 0.)
        direction[lanes[viable]] = away[viable]
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

    def hold_recovery_retreat(self, state, selected):
        self.retreating = self.retreat_hold_steps > 0
        if not self.early_retreat:
            self.retreating &= ~self.applied
        self.retreat_hold_steps = torch.where(self.retreating, self.retreat_hold_steps-1, 0)
        lanes = self.retreating.nonzero().flatten()
        if not len(lanes):
            return
        yaw = -.1 * state[lanes, 41]
        x, y, z = self.retreat_hold_direction[lanes].unbind(dim=1)
        direction = torch.stack((x*yaw.cos()-y*yaw.sin(), x*yaw.sin()+y*yaw.cos(), z), dim=1)
        self.retreat_hold_direction[lanes] = direction
        selected[lanes, 0] = 0.
        selected[lanes, 1:] = .5 + SPEED_M_S * direction

    def start_recovery_retreat(self, lidar, state, selected, config, lanes, heading, *, brake_when_blocked=True):
        reverse = -heading
        trial = torch.cat((torch.zeros_like(heading[:, :1]), .5 + SPEED_M_S * reverse), dim=1)
        result = self.issued(lidar[lanes], state[lanes], trial, config)
        usable = ((result.v_final_b * reverse).sum(1) > .08) & (result.metrics['emergency'].flatten() == 0.)
        if brake_when_blocked:
            selected[lanes] = selected.new_tensor([0., .5, .5, .5])
        used = lanes[usable]
        selected[used] = trial[usable]
        self.retreating[used] = True
        self.retreat_hold_steps[used] = 14
        self.retreat_hold_direction[used] = reverse[usable]

    def start_early_retreat(self, lidar, state, selected, config, command, speed):
        lanes = ((speed >= .08) & ~self.applied & ~self.retreating & (state[:, -1] <= .22)).nonzero().flatten()
        if not len(lanes):
            return
        heading = command[lanes]/speed[lanes, None]
        away = self.detector.retreat_direction(lidar[lanes], state[lanes], hover_only=False)
        alignment = (away*heading).sum(dim=1)
        oblique = (alignment > -.45) & (alignment < -.12)
        if bool(oblique.any()):
            self.start_recovery_retreat(lidar, state, selected, config, lanes[oblique], heading[oblique],
                                        brake_when_blocked=False)

    def hold_during_clear_pass(self, lidar, state, selected):
        from .escape_passage import clear_hover_passage

        self.holding.zero_()
        onset = (self.applied & (self.remaining == DURATION_STEPS-1) & (state[:, 45:48].norm(dim=1) <= .001)
                 & (self.retreat_hold_steps == 0))
        lanes = onset.nonzero().flatten()
        if not len(lanes):
            return
        clear = clear_hover_passage(lidar[lanes], state[lanes], self.detector._full_rays)
        held = lanes[clear]
        self.holding[held] = True
        self.applied[held] = False
        self.remaining[held] = 0
        self.cooldown[held] = 0
        selected[held] = selected.new_tensor([0., .5, .5, .5])

    @torch.no_grad()
    def apply_recovery(self, policy, action, config):
        selected = super().apply(policy, action)
        lidar, state = policy['lidar_grid'], policy['state_vec']
        if self.clear_pass:
            self.hold_during_clear_pass(lidar, state, selected)
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
        self.braking &= self.applied
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
            use_trial = (preserves_escape & gains_forward)[:, None]
            selected[mixed] = torch.where(use_trial, trial, base)
            if self.brake_forward_escape:
                # Compare the admissible escape with the latest human command at its trigger.
                chosen = torch.where(use_trial, issued, previous)
                projected = (chosen * command[mixed]).sum(1) / speed[mixed]
                starts = self.remaining[mixed] == DURATION_STEPS - 1
                self.braking[mixed] |= starts & (projected > speed[mixed] + .02)
        if self.brake_forward_escape:
            selected[self.braking] = selected.new_tensor([0., .5, .5, .5])
        if self.recovery_retreat or self.early_retreat:
            self.hold_recovery_retreat(state, selected)
        if self.early_retreat:
            self.start_early_retreat(lidar, state, selected, config, command, speed)
        if not self.recenter:
            return selected

        lanes = (moving & ~self.retreating & (state[:, -1] <= .22) & (self.since_escape >= RECOVERY_DELAY)
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
        alignment = (retreat * heading).sum(1)
        eligible = blocked & (alignment >= -.25)
        if self.recovery_retreat:
            closing = eligible & (retreat.norm(dim=1) > .1) & (alignment < 0.)
            if bool(closing.any()):
                self.start_recovery_retreat(lidar, state, selected, config, lanes[closing], heading[closing])
            eligible &= ~closing
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
