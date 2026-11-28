"""Replay the archived reopening counterexample with its finite zero backup."""
import argparse
import json
from pathlib import Path
import runpy
from unittest.mock import patch

import torch

from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.source import checked


class Backup:
    """Retain a still-valid first-packet-then-zero witness after renewal fails."""

    def __init__(self, trial, cfg):
        self.trial = trial
        self.cfg = cfg
        self.act = trial.apply
        self.tick = 0
        self.lease = None
        trial.option.apply_recovery = self.apply

    def protect(self):
        option = self.trial.option
        option.supersede_mover_edge_choice([0])
        option.measured_mover_active[0] = True

    def observed_stop_floor(self, scan, state, first):
        depth = scan[0, 9]
        valid = (scan[0, 10] > .5) & (scan[0, 11] > .55) & depth.isfinite()
        if not valid.any() or not 0. <= float(state[0, 51]) <= .04:
            return None
        # Triangle inequality over all registered lags and queues. The
        # maximum displacement includes coast, queued issue, and one first issue.
        peak_coast = max(float(coast.norm()) for _, coast in self.trial.observers)
        return (float(depth[valid].min()) - .45 * peak_coast
                - .1 * (float(state[0, 48:51].norm()) + float(first.norm())) - .055)

    def apply(self, policy, action, config):
        self.tick += 1
        trial = self.trial
        option = trial.option
        was_active = trial.phase == 'active'
        selected = self.act(policy, action, config)
        event = trial.event
        if trial.phase == 'waiting':
            return selected
        scan, state = policy['lidar_grid'], policy['state_vec']
        if event.get('decision') in ('onset', 'renewal', 'verified_zero'):
            static = self.observed_stop_floor(scan, state, state.new_tensor(event['body']))
            if static is not None and min(static, event['stop_backup_m']) >= self.cfg['certificate_floor_m']:
                self.lease = dict(tick=self.tick, static_m=static, mover_m=event['stop_backup_m'])
            else:
                self.lease = None
            return selected
        if self.lease is None:
            return selected
        if was_active and event.get('reason') not in self.cfg['eligible_loss']:
            self.lease = None
            return selected

        age = self.tick - self.lease['tick']
        # One queued packet is still active today. At age 8, next issue
        # starts at age 9, exactly on the last witness's expiry.
        if age >= self.cfg['expiry_ticks'] - self.cfg['queue_ticks']:
            self.lease = None
            trial.event = dict(decision='expired_return_to_v60', witness_age_ticks=age)
            return selected
        previous_reason = event.get('reason')
        veto = [name for name in self.cfg['priority_flags'] if bool(getattr(option, name)[0])]
        if bool(option.zero_crossing_commit[0] or option.observed_turn_brake_floor[0]):
            veto.append('committed_or_braking')
        current = option.issued(scan, state, selected, config)
        neutral = option.action_for_velocity(state.new_zeros((1, 3)), state[:, 48:51], config.governor)
        stop = option.issued(scan, state, neutral, config)
        floor = self.observed_stop_floor(scan, state, state.new_zeros(3))
        witness = dict(previous_reason=previous_reason, witness_age_ticks=age,
                       witness_mover_m=self.lease['mover_m'], witness_static_m=self.lease['static_m'],
                       latest_stop_static_m=floor, priority=veto, proposed_beta=float(stop.metrics['beta'][0, 0]),
                       predicted_issue_b=stop.v_final_b[0].tolist(),
                       previous_issue_norm_m_s=float(state[0, 48:51].norm()))
        if (veto or bool(current.metrics['emergency'][0, 0] or stop.metrics['emergency'][0, 0])
                or float(stop.v_final_b[0].norm()) > 2.e-6
                or floor is None or floor < self.cfg['certificate_floor_m']):
            trial.event = dict(decision='backup_cancelled', **witness)
            self.lease = None
            return selected
        selected[0] = neutral[0]
        self.protect()
        trial.event = dict(decision='certified_stop_backup', **witness)
        return selected


def compare(records, reference, cfg):
    first = next((row['step'] for row in records if row['proof'].get('decision') == 'certified_stop_backup'), None)
    if first is None:
        return dict(first_backup=None, identical_prefix=records == reference)
    onset = first - records[0]['step']
    if records[:onset] != reference[:onset]:
        raise ValueError('arm diverged before backup dispatch')
    anchor = reference[onset]
    changed = records[onset]
    for name in ('action', 'position_before', 'velocity_before', 'request_w'):
        if changed[name] != anchor[name]:
            raise ValueError(f'arm diverged in {name} at dispatch')
    last = [r for r in records if r['proof'].get('decision') == 'certified_stop_backup']
    if any(max(map(abs, r['issued_w'])) > 2.e-6 or r['emergency'] for r in last):
        raise ValueError('final pipeline did not emit the certified neutral command')
    valid_through = first - changed['proof']['witness_age_ticks'] + cfg['expiry_ticks']

    def exposure(rows, start, end):
        segment = [r for r in rows if start <= r['step'] < end]
        if not segment:
            raise ValueError(f'no samples in {start}–{end}')
        earliest = next((r['step'] for r in segment if min(r['mover_m'], r['static_m']) < .15), None)
        return dict(minimum_m=min(min(r['mover_m'], r['static_m']) for r in segment),
                    mover_minimum_m=min(r['mover_m'] for r in segment),
                    static_minimum_m=min(r['static_m'] for r in segment),
                    first_breach_packet=earliest, final_speed_m_s=segment[-1]['speed_end_m_s'])

    return dict(first_backup=first, identical_prefix=True, last_zero_issue=last[-1]['step'],
                valid_through_exclusive=valid_through,
                stroke=[r['step'] for r in last],
                during=dict(backup=exposure(records, first, valid_through),
                            prior=exposure(reference, first, valid_through)),
                after=dict(backup=exposure(records, valid_through, records[-1]['step'] + 1),
                           prior=exposure(reference, valid_through, records[-1]['step'] + 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('hard_positive', 'nominal'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    registry = script.with_name('INPUTS.json')
    inputs = load(registry)
    config = inputs['protocol']
    ref_pin = inputs['case_references'][args.case]
    ref = load(checked(ref_pin))
    entry = checked(inputs['replay_script'])
    assert ref['script'] == inputs['replay_script'] and ref['inputs'] == inputs['replay_inputs']
    reference = ref['arms'][1]
    context = runpy.run_path(str(entry))['main'].__globals__
    persist = context['save']
    trial_cls = context['Trial']
    replay = context['replay']

    def backup_trial(option, observers):
        trial = trial_cls(option, observers)
        Backup(trial, config)
        return trial

    def one_arm(*params):
        mode = params[9]
        if mode == 'stock':
            return ref['arms'][0]
        return replay(*params)

    def save_evidence(path, data):
        arm = data['arms'][1]
        delta = compare(arm['records'], reference['records'], config)
        bound = persist(path, dict(
            registry=pin(registry), method=pin(script), previous=ref_pin, replay_inputs=data['inputs'],
            replay_script=data['script'], case=data['case'], comparison=delta, arm=arm,
        ))
        print(json.dumps(dict(comparison=delta, previous_metrics=reference['metrics'],
                              backup_metrics=arm['metrics'])), flush=True)
        return bound

    with patch.dict(context, Trial=backup_trial, replay=one_arm, save=save_evidence):
        context['main']()


if __name__ == '__main__':
    main()
