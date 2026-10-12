# Paired pilot start readiness

Status: BLOCKED_CONTROL_BASELINE; Goal IN_PROGRESS. This is the actual current readiness decision, not an approved training launch.

The next paired P0/PLR must share one independently accepted repaired control baseline. Current live baseline, physical repair/regression and command-passthrough acceptance remain pending. Zero-target enhanced T/S filenames cannot certify command tracking. No persistent horizontal/vertical anomaly is waived.

FirstUpdateDiagnostic is implemented and CPU verified; its gate remains PENDING_NEXT_PILOT and must pass on the first real rollout before either optimizer advances. Current Goal has zero training transitions, optimizer updates and training checkpoint writes.

The intended unchanged single-variable comparison is actorLR3e-4 vs1e-4, criticLR3e-4,32envs/16horizon/64minibatch/1epoch, original1024stepzero-commandwarmup,64updates/32768transitionspergroup, originalsave/evaluationnodes. These are future proposal limits, not current execution authority.

Actual future argv and exact shared repaired baseline will be emitted after the current real control gates. Training authorization has not been issued. Source/input/tool/output/STOP/cleanup identity must then be bound through the existing Commander mechanism.
