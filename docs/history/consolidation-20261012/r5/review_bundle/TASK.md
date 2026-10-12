# UNIFIED-H1-R3 A0009 resource-window CPU review

D0005 is a sealed single-use real attempt. It failed before AppLauncher because the three-sample coexistence window was shorter than the fixed ten-second contract. The actual three samples were discarded by the old collector and cannot be reconstructed. Review the prepared A0009 successor only at the CPU layer.

The candidate is already implemented and committed outside this isolated review bundle. PI must interpret this as an engineering verification task. Engineer must inspect the exact baseline/candidate files and evidence, run the task CPU tests, and leave the prepared candidate unchanged unless a concrete defect exists. Evaluator must independently inspect the semantic diff, run at least one actual CPU command, and PASS only if all criteria hold.

Acceptance criteria:

1. The schedule is anchored after the first completed sample, so first-provider latency cannot shorten the first-to-last timestamp span below ten seconds. The fixed thresholds, sample count, GPU identity and compute-app rules remain unchanged.
2. Validation remains fail-closed. A collected window rejected by validation is saved verbatim as `observed_window`; no D0005 values are invented.
3. Exact real successor identity is workload commit `fd47dd321d571c4d6c983048f77bb3913e7e7b01`, tree `f2ccbe684e9d0fb56a861152b91ea4b4870500ee`, config SHA256 `824feeb87fccb33ddb5b56b401b6591688b042606fda7d45f4d5f92c041f110b`, real diff SHA256 `82b56cecc7680f8d57f23353c1caf5c53898ac55f1b76327ab3f8c703a194672`.
4. D0005 accounting stays: live start 1, simulator-initialization slot 1, AppLauncher constructions 0, transitions/PPO/checkpoints 0, retry 0, and no owned processes remain.
5. No method, actor, threshold, runtime, product, ray, PPO, checkpoint, cleanup or formal-training change is accepted. CPU PASS does not authorize D0006 or R4.

Do not access GPU, query GPU state, launch AppLauncher/simulation, execute real environment steps, train, write checkpoints, install packages, modify evidence, or resume/retry D0005.
