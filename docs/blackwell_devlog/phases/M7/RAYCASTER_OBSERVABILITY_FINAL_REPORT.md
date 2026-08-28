# RayCaster Exact-Normal Final Report

## Delivered boundary

The evaluation-runtime-only adapter now invokes the frozen official
normal-producing RayCaster/Warp operation on owned copies of the exact current
world rays and the same frozen mesh/maximum distance used by the accepted
sensor. It verifies exact accepted hits, validity, distances, raw miss
conventions, unit normals, no source mutation, and independently owned
zero-placeholder handoff into the unchanged M7-003 exact-J core.

Two independent headless CUDA processes each cover 84,960 ray samples. Their
complete acquisition, provenance, normal, mask, hit, distance, and
observability trees match at maximum numeric error zero. Eleven complete-tree
mutations fail closed. Every valid plane normal is exact unit world ±Z; every
scan/environment result is rank one with the expected two-dimensional-nullspace
unavailability state.

Static and runtime checks show the adapter is not re-exported, actor/deployed
consumer modules contain no adapter import or surface-normal field, policy
inputs remain unchanged around acquisition, and zero actor inference occurs.
Accepted M7-003 implementation bytes remain unchanged and M7-002 runtime
observability remains `UNAVAILABLE`.

## Scope declaration

No sensor/environment/actor/control/task/training/core edit, runtime-record
wiring, proxy, plot, B0-B8 implementation or execution, learning, checkpoint
mutation, M8, 128k/1M, package/runtime/frozen mutation, merge, push, or history
rewrite was performed.

Executor recommendation is `READY_FOR_REVIEW`; this is not approval and does
not authorize runtime integration or the next phase.
