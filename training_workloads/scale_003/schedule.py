"""Exact lossless rollout, command, optimizer, and checkpoint schedule."""

from __future__ import annotations

from dataclasses import dataclass


NUM_ENVS = 32
FULL_HORIZON = 16
FULL_ROLLOUTS = 1703
TAIL_HORIZON = 2
TOTAL_POLICY_STEPS = 27250
PHASE_1_STEPS = 8000
INITIAL_COUNTER = 250
INITIAL_OPTIMIZER_STEP = 2000
COMMAND_HOLD_STEPS = 32
COMMAND_CYCLE = (
    (0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0),
    (0.15, 0.15, 0.0), (-0.15, 0.15, 0.0), (0.15, -0.15, 0.0), (-0.15, -0.15, 0.0),
)


@dataclass(frozen=True)
class Segment:
    ordinal: int
    horizon: int
    samples: int
    policy_step_start: int
    policy_step_end: int
    phase: int
    command_b_m_s: tuple[float, float, float]
    update_counter: int
    optimizer_step_delta: int
    target_optimizer_step: int
    checkpoint_kind: str | None


def command_at(policy_step: int) -> tuple[float, float, float]:
    if isinstance(policy_step, bool) or not isinstance(policy_step, int) or not 0 <= policy_step <= TOTAL_POLICY_STEPS:
        raise ValueError("policy step is outside the global command clock")
    if policy_step < PHASE_1_STEPS:
        return (0.0, 0.0, 0.0)
    phase_step = min(policy_step - PHASE_1_STEPS, TOTAL_POLICY_STEPS - PHASE_1_STEPS - 1)
    return COMMAND_CYCLE[(phase_step // COMMAND_HOLD_STEPS) % len(COMMAND_CYCLE)]


def checkpoint_kind(ordinal: int) -> str | None:
    if ordinal == 1703:
        return "pre_tail"
    if ordinal == 1704:
        return "final"
    if ordinal <= 1700 and ordinal % 25 == 0:
        return "regular"
    return None


def build_segments() -> tuple[Segment, ...]:
    segments: list[Segment] = []
    policy_step = 0
    optimizer_step = INITIAL_OPTIMIZER_STEP
    for ordinal in range(1, FULL_ROLLOUTS + 2):
        horizon = FULL_HORIZON if ordinal <= FULL_ROLLOUTS else TAIL_HORIZON
        samples = horizon * NUM_ENVS
        if samples % 64:
            raise AssertionError("segment is not divisible by the fixed minibatch")
        command = command_at(policy_step)
        end = policy_step + horizon
        if any(command_at(index) != command for index in range(policy_step, end)):
            raise AssertionError("a command boundary falls inside a rollout")
        delta = samples // 64
        optimizer_step += delta
        segments.append(Segment(
            ordinal=ordinal, horizon=horizon, samples=samples,
            policy_step_start=policy_step, policy_step_end=end,
            phase=1 if policy_step < PHASE_1_STEPS else 2,
            command_b_m_s=command, update_counter=INITIAL_COUNTER + ordinal,
            optimizer_step_delta=delta, target_optimizer_step=optimizer_step,
            checkpoint_kind=checkpoint_kind(ordinal),
        ))
        policy_step = end
    if policy_step != TOTAL_POLICY_STEPS or optimizer_step != 15625:
        raise AssertionError("terminal schedule accounting mismatch")
    if sum(segment.samples for segment in segments) != 872000:
        raise AssertionError("transition schedule mismatch")
    return tuple(segments)


def checkpoint_counters() -> tuple[int, ...]:
    return tuple(segment.update_counter for segment in build_segments() if segment.checkpoint_kind is not None)


class CommandClock:
    """Process-global command clock; resets have no mutation method."""

    def __init__(self) -> None:
        self._index = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def command(self) -> tuple[float, float, float]:
        return command_at(self._index)

    def advance_after_action(self) -> None:
        if self._index >= TOTAL_POLICY_STEPS:
            raise RuntimeError("global policy budget exhausted")
        self._index += 1

    def observe_reset(self) -> None:
        """Record the semantic boundary without changing the global index."""
