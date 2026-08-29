# M7-008 Revision 1 Exact Import-Boundary Rework Plan

## Goal

Close only Commander finding F1 while preserving the accepted fail-closed B0
manifest behavior: restore the exact direct adapter submodule dependency,
permit it in only the named execution module, and bind the corrected bytes in
fresh D0002 evidence. No runtime, environment, controller, simulator, scenario,
or baseline execution is authorized.

## Source locks

| Source | Accepted SHA-256 | Use |
| --- | --- | --- |
| `instinctlab/control/baseline_b0.py` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` | Sole adapter invoked after manifest validation |
| `instinctlab/evaluation/baseline_execution.py` | `db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108` | Corrected fourth exact B0 readiness capability |
| M7-007 base | `fb9b9c4af7b0004c0b0647b80acef2e9e85e361b` | Accepted registry, manifest, and adapter contract |

## Writable files

Only the 11 rework paths enumerated by Order M7-008 revision 1 are writable.
The first exact commit contains import/binding, three tests, plan, and contract.
The second exact commit contains only the four bounded outcome records.

## Implementation and tests

1. Import the three accepted B0 objects directly and only from
   `instinctlab.control.baseline_b0`; forbid aggregate `instinctlab.control`.
2. Give only `evaluation/baseline_execution.py` one exact historical-test
   exception while retaining the prohibition everywhere else.
3. Update the exact fourth source and dependent manifest hashes without any
   adapter, manifest, actor, registry-row, or B1-B8 semantic change.
4. Run the Commander-equivalent probe, focused suites twice, accepted pure
   regressions twice, two-process comparison, and full integrity evidence.

## Stop conditions

Stop if work requires an unlisted path, changed accepted adapter/manifest
semantics, actor/research changes, packages or frozen inputs, runtime or
controller integration, CUDA/simulation, baseline execution, training, M8,
scale work, merge/push/history rewrite, or any product correction after either
of the two authorized commits.
