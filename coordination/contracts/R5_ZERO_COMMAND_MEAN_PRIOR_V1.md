# R5 zero-command policy-mean prior v1

This is a new prospective method after the frozen w.05/anchor4 pure-hover method
failed confirmation seed20261015 at u256 (original Z 0/32). The old freeze and
failure are retained. No seed substitution, posthoc node selection, continuation
beyond256 or threshold change qualifies the old method.

The sole new variable is a finite differentiable actor-loss term on the existing
Beta mean. For minibatch approved actor state command c=state_vec[:,45:48],
m=output.mean_action[:,1:4], Lmean = lambda * mean(1[||c||<=.001] *
sum(((m-.5)/.1)^2)). The indicator uses the existing approved command alone.
The three action coordinates decode exactly as v_corr_b=m-.5 in the current
governor. Lambda=0 for R0 and 10 for R1. Both groups retain residual reward .05,
anchor4, actorLR1e-4, criticLR3e-4, original PPO/entropy/gradient clipping,
network, sensors, actor information, physics and all safety/evaluation gates.

There is no hard-zero actor, output replacement, clipping, runtime override,
privileged supervision, additional sampling, forward, optimizer step or physics
tick. The deployed actor remains the learned original Beta mean. The finite soft
term competes with the existing objective; nonzero commands have exactly zero
auxiliary loss and gradient. The old path without this registered profile never
calls the optional loss hook and preserves original arithmetic and log schema.
The shared product hook receives only actor policy and its already-computed
output, never privileged tensors. Every minibatch persists the actual auxiliary
scalar. A nonfinite/shape-invalid term fails before either optimizer mutation.

Cold-start development seed20260908, pure-hover256, preregistered late nodes
192 and256 (primary256), original Z32/all8subgroups/T0/200steps/.08speed/.30drift/
.20height must pass at both nodes. Fair source/config/budget/init/first512 pair.
This is a finite test of whether low-variance direct policy-mean regularization
reduces the observed persistent deterministic bias. It does not prove the cause
of seed sensitivity or solve general reference recovery. Failure requires a
recorded new research decision, never unbounded lambda/horizon searches.

After complete development acceptance freeze this method/budget/primary before
two previously unused training seeds are selected and launched. Both must pass
the same primary; no2of3, failed-seed replacement or tuning after freeze. New
profile has exact mean_prior_contract pin and mean_prior_weight field; both are
included in effective checkpoint identity. Legacy profiles/contracts are intact.
Future formal continuation must explicitly register the same term and its phase
contract; no inheritance of an old formal permission or checkpoint acceptance.
