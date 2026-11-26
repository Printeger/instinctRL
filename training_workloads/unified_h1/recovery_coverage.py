"""Recompute actual training exposure from compact pre-reset tube records."""

from dataclasses import dataclass
import io
import json
import math


RUN_PREFIX = 'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-NEXT-LONG-'
STREAM = 'pipe_coverage.jsonl'
WINDOW = dict(closing_steps=3, closing_gap_m=1.9, closing_rate_mps=.20,
              recovery_steps=10, recovery_gap_m=.60, minimum_command_mps=.03,
              maximum_recovery_closing_mps=.05, quiet_hover_steps=10, zero_command_mps=.001)
_COLUMNS = (
    'episode_step', 'terminal', 'collision', 'active', 'age_before_s', 'age_after_s',
    'gap_before_m', 'gap_after_m', 'winning_beams_before', 'winning_beams_after',
    'escape_active', 'emergency', 'command_w_mps', 'issued_w_mps', 'position_w_m',
)
_FIELDS = {'step', 'tick_before', 'tick_after', 'contact_samples', 'lanes', *_COLUMNS}


def tracks_run(run_id):
    from .dynamic_assay import FRONT_STUDY_PREFIX, U1792_STUDY_PREFIX

    return run_id.startswith((RUN_PREFIX, U1792_STUDY_PREFIX, FRONT_STUDY_PREFIX))


def capture(before, after, evidence, lanes):
    """Read existing diagnostic tensors only, after the current step has completed."""
    step = evidence['_h1_driver']
    raw = evidence['pre_reset']
    control = raw['control']

    def sample(value):
        return value[list(lanes)].detach().cpu().tolist()

    values = (
        raw['episode_length'].reshape(-1),
        (step['terminated'] | step['truncated']).reshape(-1),
        raw['task']['collision'].reshape(-1),
        before['active'], before['applied_age_s'], after['applied_age_s'],
        before['sphere_sensor_clearance_m'], after['sphere_sensor_clearance_m'],
        before['winning_beams'], after['winning_beams'],
        raw['escape_option']['active'], control['ics_metrics']['emergency'].reshape(-1),
        step['command_world'], control['v_final_w'], raw['root_state_w'][:, :3],
    )
    return dict(step=step['global_policy_step'], tick_before=before['tick'], tick_after=after['tick'],
                contact_samples=after['contact_samples'], lanes=list(lanes),
                **{key: sample(value) for key, value in zip(_COLUMNS, values, strict=True)})


@dataclass
class _Episode:
    length: int = 0
    age_s: float = 0.
    closing: int = 0
    eligible: int = 0
    issuing: int = 0
    quiet: int = 0
    saw_closing: bool = False
    saw_recovery: bool = False
    saw_issuing: bool = False
    saw_quiet: bool = False
    recovery_start: tuple | None = None
    recovery_heading: tuple | None = None
    previous_position_w_m: tuple | None = None


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _vector(value):
    return isinstance(value, list) and len(value) == 3 and all(map(_number, value))


def _add_sample(episode, totals, sample):
    (length, terminal, collision, active, age_before, age_after, gap_before, gap_after,
     beams_before, beams_after, escaping, emergency, command, issued, position) = sample
    if (type(length) is not int or length != episode.length + 1
            or any(type(flag) is not bool for flag in (terminal, collision, active, escaping))
            or (collision and not terminal) or type(emergency) not in (int, float) or emergency not in (0, 1)
            or any(type(n) is not int or not 0 <= n <= 21240 for n in (beams_before, beams_after))
            or not all(map(_number, (age_before, age_after, gap_before, gap_after)))
            or not math.isclose(age_before, episode.age_s, abs_tol=.0005)
            or not math.isclose(age_after - age_before, .1, abs_tol=.0005)
            or not all(map(_vector, (command, issued, position)))
            or command[2] != 0. or math.hypot(*command) > .300001):
        raise ValueError('R6_PIPE_COVERAGE_SAMPLE_OR_EPISODE_CLOCK')
    episode.length = length
    episode.age_s = age_after
    if terminal:
        # The first terminal step closes the episode and cannot contribute a live window.
        return

    closing_rate = (gap_before - gap_after) / .1
    closing = (active and beams_before > 0 and beams_after > 0 and gap_before <= WINDOW['closing_gap_m']
               and closing_rate >= WINDOW['closing_rate_mps'])
    episode.closing = episode.closing + 1 if closing else 0
    episode.saw_closing |= episode.closing >= WINDOW['closing_steps']
    totals['visible_closing_steps'] += int(closing)

    speed = math.hypot(*command)
    eligible = (episode.saw_closing and active and not escaping and speed > WINDOW['minimum_command_mps']
                and gap_before >= WINDOW['recovery_gap_m'] and gap_after >= WINDOW['recovery_gap_m']
                and closing_rate <= WINDOW['maximum_recovery_closing_mps'])
    episode.eligible = episode.eligible + 1 if eligible else 0
    totals['eligible_recovery_steps'] += int(eligible)
    issued_forward = sum(a * b for a, b in zip(issued, command, strict=True)) / speed if eligible else 0.
    issuing = eligible and emergency == 0 and issued_forward > WINDOW['minimum_command_mps']
    episode.issuing = episode.issuing + 1 if issuing else 0
    episode.saw_issuing |= episode.issuing >= WINDOW['recovery_steps']
    totals['issued_recovery_steps'] += int(issuing)

    if episode.eligible == 1:
        episode.recovery_start = episode.previous_position_w_m or tuple(position)
        episode.recovery_heading = tuple(component / speed for component in command)
    if not episode.saw_recovery and episode.eligible == WINDOW['recovery_steps']:
        episode.saw_recovery = True
        progress = sum((end - start) * heading for end, start, heading in
                       zip(position, episode.recovery_start, episode.recovery_heading, strict=True))
        totals['first_recovery_progress_sum_m'] += progress

    quiet = not active and not escaping and speed <= WINDOW['zero_command_mps']
    episode.quiet = episode.quiet + 1 if quiet else 0
    episode.saw_quiet |= episode.quiet >= WINDOW['quiet_hover_steps']
    totals['quiet_hover_steps'] += int(quiet)
    episode.previous_position_w_m = tuple(position)


def _close_episode(episode, totals, *, censored, collision):
    totals['initial_episodes'] += 1
    totals['right_censored_episodes'] += int(censored)
    totals['contact_episodes'] += int(collision)
    totals['visible_closing_episodes'] += int(episode.saw_closing)
    totals['eligible_recovery_episodes'] += int(episode.saw_recovery)
    totals['issued_recovery_episodes'] += int(episode.saw_issuing)
    totals['quiet_hover_episodes'] += int(episode.saw_quiet)


def summarize(rows, *, start_step, end_step):
    """Count unique lane episodes with contiguous windows on a complete phase."""
    from .recovery_training_scene import TUBE_LANES

    if type(start_step) is not int or type(end_step) is not int or not 0 <= start_step < end_step:
        raise ValueError('R6_PIPE_COVERAGE_PHASE_CLOCK')
    totals = dict(initial_episodes=0, right_censored_episodes=0, contact_episodes=0,
                  visible_closing_episodes=0, eligible_recovery_episodes=0, issued_recovery_episodes=0,
                  quiet_hover_episodes=0, visible_closing_steps=0, eligible_recovery_steps=0,
                  issued_recovery_steps=0, quiet_hover_steps=0, first_recovery_progress_sum_m=0.)
    by_lane = {lane: totals.copy() for lane in TUBE_LANES}
    episodes = {lane: _Episode() for lane in TUBE_LANES}
    fixtures = {}
    next_step = start_step
    next_tick = None
    for row in rows:
        if (not isinstance(row, dict) or set(row) != _FIELDS or type(row['step']) is not int
                or row['step'] != next_step or row['step'] >= end_step or row['lanes'] != list(TUBE_LANES)
                or type(row['tick_before']) is not int or type(row['tick_after']) is not int
                or row['tick_before'] < 0 or row['tick_after'] != row['tick_before'] + 20
                or (next_tick is not None and row['tick_before'] != next_tick)
                or type(row['contact_samples']) is not int or row['contact_samples'] != 20
                or any(not isinstance(row[k], list) or len(row[k]) != len(TUBE_LANES) for k in _COLUMNS)):
            raise ValueError('R6_PIPE_COVERAGE_STEP_OR_PHYSICS_CLOCK')
        for lane, sample in zip(TUBE_LANES, zip(*(row[k] for k in _COLUMNS), strict=True), strict=True):
            active = sample[3]
            if fixtures.setdefault(lane, active) is not active:
                raise ValueError('R6_PIPE_COVERAGE_MOVER_ASSIGNMENT_CHANGED')
            episode = episodes[lane]
            counts = by_lane[lane]
            _add_sample(episode, counts, sample)
            if sample[1]:
                _close_episode(episode, counts, censored=False, collision=sample[2])
                episodes[lane] = _Episode()
        next_step += 1
        next_tick = row['tick_after']
    if next_step != end_step:
        raise ValueError('R6_PIPE_COVERAGE_INCOMPLETE_PHASE')
    for lane, episode in episodes.items():
        if episode.length:
            _close_episode(episode, by_lane[lane], censored=True, collision=False)
    for key in totals:
        totals[key] = sum(counts[key] for counts in by_lane.values())
    return dict(start_step=start_step, end_step=end_step, policy_steps=end_step-start_step,
                pipe_lane_steps=(end_step-start_step)*len(TUBE_LANES), window=WINDOW,
                active_mover_lanes=[lane for lane, active in fixtures.items() if active],
                totals=totals, by_lane={str(lane): counts for lane, counts in by_lane.items()})


def audit(binding):
    """Require a closed analyzed job and recover every robust phase's original stream."""
    from tools.r6_direct.evidence import validate_binding
    from tools.r6_evidence import checked
    from .checkpoint_compatibility import pinned_json
    from .lossless_evidence import ChunkReader
    from .robustness_curriculum import slices
    from .robustness_training_plan import branch_start_update

    _, ctx, report = validate_binding(binding)
    plan = ctx['contract']['parameters']['r6_training_plan']
    run_id = plan['experiment_id']
    if not tracks_run(run_id) or report['issues']:
        raise ValueError('R6_PIPE_COVERAGE_CLOSED_TRAINING_REQUIRED')
    recipe = pinned_json(plan['recipe'])
    curriculum = pinned_json(recipe['curriculum'])
    phases = slices(curriculum, plan['start_update'], plan['target_update'],
                    branch_start_update=branch_start_update(recipe))
    streams = {item['logical_name']: item for item in report['streams']}
    results = []
    for phase in phases:
        if phase['kind'] != 'robust':
            continue
        start, end = (phase[key] * 16 for key in ('start_update', 'target_update'))
        name = f"phases/{phase['index']}-{start}-{end}/learning_log/{STREAM}"
        record = streams.get(name)
        if record is None or not record['complete'] or record['rows'] != end-start:
            raise ValueError('R6_PIPE_COVERAGE_INCOMPLETE_ORIGINAL')
        path = checked(record['pin'])
        if path != ctx['output']/'compressed_evidence'/f'{name}.r6z':
            raise ValueError('R6_PIPE_COVERAGE_STREAM_PATH')
        with ChunkReader(path, run_id=run_id, logical_name=name, maximum_raw_bytes=record['raw_bytes']) as stream:
            if stream.raw_sha256 != record['raw_sha256'] or stream.size != record['raw_bytes']:
                raise ValueError('R6_PIPE_COVERAGE_ORIGINAL_DRIFT')
            with io.BufferedReader(stream) as reader:
                rows = (json.loads(line) for line in reader)
                result = summarize(rows, start_step=start, end_step=end)
        results.append(dict(result, stream=record['pin'], phase=phase))
    return dict(schema_version='r6-recovery-training-coverage/1', binding=binding, run_id=run_id, phases=results)


def main():
    import argparse
    from pathlib import Path
    from tools.r6_direct.instinct_cycle import save
    from tools.r6_direct.instinct_cycle_finish import guard
    from tools.r6_evidence import file_pin

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binding', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    guard()
    result = audit(file_pin(args.binding))
    guard()
    print(json.dumps(save(args.output, result)))


if __name__ == '__main__':
    main()
