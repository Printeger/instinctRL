# R5 zero-command residual-neutrality preregistration (2026-10-14)

This is the user-authorized finite follow-up to the accepted negative P0/PLR
pair. Historical method and evidence bytes remain immutable.

The only effective method variable is the coefficient of `reward_zero_extra`:

`-w * I[norm(command_b) <= 0.001] * sum((v_corr_b / 0.1)^2)`.

C0: w=0.0. C1: w=0.05. Both actor LR=0.0001, critic LR=0.0003.
The existing decoded body-frame correction is passed forward to reward only.
No control override, actor input, network, PPO, controller, curriculum,
termination or metric-h1-v1 threshold changes. C0 preserves every original
reward component and total. The coefficient is fixed before any new live run.
This is a falsifiable objective experiment, not a proven root cause or paper
innovation. The old PLR is historical context, not the contemporaneous control.

Each group starts from the same seed 20260908, 32 environments, horizon 16,
minibatch 64, one epoch; 64 updates, 32768 training transitions; checkpoints
init0000/u1/u2/u8/u16/u32/u64. The actual first 512 sampled actions, inputs,
indices, old Beta concentrations and log probabilities must pass the unchanged
atol=1e-6/rtol=1e-5 gate before either optimizer. Gate failure stops with zero
updates; no retries. Original Z at u16/u64 uses seed 20260909 and all 32 lanes
for all 200 steps, retaining sensor T0, startup, all failures and subgroups.
The preregistered primary comparison is u64; u16 is diagnostic.

New Commander envelope: 4 live starts / 4 initialization reservations,
65536 training / 25600 nontraining transitions, 128 logical PPO updates,
14 training checkpoint writes, 0 automatic retries, 10840 seconds wall and
10840 allocated GPU seconds including cleanup, 64 GiB aggregate output.
Per train entry: 3610 seconds including 10-second cleanup. Per eval: 1810.
Each entry reserves 16 GiB output: 13 GiB conservative before-write workload
quota, 3 GiB control/native/evidence allowance. No quota refunds on overwrite
or partial failure. Four reservations cannot exceed the aggregate disk cap.
Independent review, exact source/config/inputs/tool hashes, actual Commander
approval, STOP/coexistence/cleanup checks precede each unique release.

No 256-update extension, extra weights, dependency installation, frozen runtime
changes or repeated physical diagnostics. Native close FAILED/-9 and collector
INCOMPLETE remain distinct from independently accepted scientific evidence.
Complete trusted negative results are VALID_EXPERIMENT_NEGATIVE; positive
results require credible original-Z improvement and do not authorize long
training. Missing data/authority yield a specific blocker, not completion.
