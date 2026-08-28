# M7-002 Runtime Evaluation Final Report

Revision 1 adds exactly one correction commit after the preserved D0001 commits;
the exact final SHA and parent chain are recorded in the D0002 Executor report.
The rework changes exactly the seven paths authorized by Order M7-002 revision 1.

F1 is closed: the step-4 selective timeout retains the full returned post-reset
`policy` mapping (`lidar_grid`, `state_vec`), ordered field hashes, and exact
key/shape/dtype/device/value/hash binding to the step-5 audited policy-only actor
call. Its same-state terminal root is independently bound to the step-4
evaluation row; the reset environment adopts its returned station reference and
the non-reset reference remains byte-identical.

F2 is closed: all 23 accepted evaluation inputs now have uniform structured
source/source-shape/record-shape/dtype/unit/frame/validity/ownership/privilege
metadata. The exact manifest digest is
`08c62ddb3a6ce6ce6df0f673f14b2826e2de85107d59eb79d4eeed2530f64af0`.
Validation is fail closed and each output owns a deep manifest copy.

F3 is closed: 43 focused pure tests cover exact positives plus extra,
reordered, missing, malformed, wrong shape/dtype/device, nonfinite, invalid
anchor/clearance/ICS/termination domains and precedence, policy/source and
source/source alias, output storage ownership, stale/post-reset terminal
substitution, reset-reference corruption, and exact returned-to-next-policy
binding failures.

F4 is closed: the formal validator asserts exact formal/Isaac Lab/OmniDrones/
NavRL heads and status bytes, the sole known Isaac Lab kit file and SHA256
`0a0dee5fbbdcbf66f8124a07eced9a87c2f70e5df6ee892ec297444c1d1b01c2`,
driver `580.173.02`, Python/PyTorch/Isaac Sim/Isaac Lab/TorchRL/TensorDict
versions, and current `pip freeze` byte identity/hash
`145c77d8a47079eef6ce3e396b6db1e44677fe52efb543a63bf7d5cb7fbf78c9`.

Two fresh headless CUDA processes each produced 8 rows × 2 environments = 16
samples and exactly 160 physics steps. Complete ordered trees compare at maximum
numeric error zero, and every comparator mutation fails nonzero. Accepted M7-001
and M3-M6 pure regressions pass twice; M2 static and the accepted M6 environment
smoke pass. The complete comparison covers 33 tensors and 1,020,410 elements.
No metric semantic, actor schema, policy action, or accepted M1-M7
core changed.

No observability algorithm, B0-B8 scenario/ablation, optimizer, PPO/update,
sampling, checkpoint, training, M8, 128k, 1M, merge, push, history rewrite,
package/runtime/system, frozen baseline, or external repository mutation was
performed. Result: `READY_FOR_REVIEW`; no next-phase work was started.
