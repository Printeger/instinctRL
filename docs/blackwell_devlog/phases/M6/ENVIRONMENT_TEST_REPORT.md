# M6-002 Environment Test Report

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Syntax/static API | PASS | command 4 |
| M3/M4/M5-001/M5-002/M6-001/new suite, pass 1 | PASS — 117 tests + 33 subtests | command 5 |
| Same regression set, pass 2 | PASS — 117 tests + 33 subtests | command 6 |
| M2 static | PASS — 10 tests | command 7 |
| Accepted M5 runtime integration regression | PASS — 256 physics steps | command 8 |
| Environment rollout process 1 | PASS — 2 envs, 32 policy/640 physics steps | command 9 |
| Environment rollout process 2 | PASS — 2 envs, 32 policy/640 physics steps | command 10 |
| Complete-tree comparator | PASS — 1,800 tensors, 16,834,412 elements, max error `0.0` | command 11 |
| Missing/extra/reordered/nonfinite/any-error rejection | PASS — five expected exit-1 cases | commands 13-17 |
| Accepted M2-M6-001 byte/diff integrity | PASS | command 18 |
| Diff hygiene and scope scans | PASS | commands 19-24 |
| Frozen/formal/external/runtime integrity capture | PASS | commands 25-35 |

All command numbers resolve through the raw evidence ledger. `NOT RUN`: actor
or critic networks, PPO/GAE, optimization, checkpointing, learning, evaluation,
randomization, curriculum, M7 work, and scale training, because the Order
forbids them.

## Negative and reset coverage

The persisted rollout rejects action shape, dtype, device, nonfinite, below
range, and above range before root/action/counter/rotor mutation. It rejects an
out-of-range reset index before mutation. Static tests cover missing actor keys,
privileged actor contamination, public producer usage, exact step ordering,
direct M5 metric names, and the reset subsystem ledger. Runtime step 16 resets
only environment 0 and proves byte equality for environment 1 root state,
policy cache, rotor state, and episode counter at the reset boundary.

## Rollout capture

Every one of the 32 returned steps persists actions, both policy tensors,
critic state, all M5 outputs and public metrics, reward total/components and
diagnostics, terminated/truncated/reason/contact/clearance values, root state,
forces, torques, rotor/controller output, reset marker, and episode counter.
Each output tree is 67,831,799 bytes. The processes are independent Isaac Sim
launches and compare with exact zero maximum error.
