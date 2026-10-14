# R5 formal training readiness

Status: **ACTUAL_VALIDATION_COMPLETE, FINAL_REVIEW_AND_CONDITIONAL_PACKAGE_PENDING**. Target is `READY_FOR_FORMAL_R5_TRAINING`; formal training has not been submitted. This is the current report, updated as actual independent evidence arrives. [Goal state](GOAL_STATE.md), [full ledger](goal_budget.json), [all results](EXPERIMENT_RESULTS.json) and [evidence index](EVIDENCE_INDEX.md) preserve historical stages and negative results.

The current candidate adds a finite actor-only soft prior on zero-command Beta mean xyz corrections, coefficient10, to the shared residual reward.05/anchor4/actorLR1e-4/criticLR3e-4 baseline. The decoder remains xyz=mean[:,1:4]−.5; the actor retains its original deployment measurement/history/validity/time/IMU/command inputs. No hard-zero output, action substitution, simulator truth input or threshold change is used. Nonzero samples have zero direct auxiliary loss/gradient; the shared network can still change, so nonzero behavior needs its own implementation check and subsequent training evaluation. The large auxiliary coefficient may dominate the PPO direction and is a disclosed method choice.

Actual development uses read-only source617af4d90c35e333c97692bdee90146804476edf/config2e9a72840a57ad6806d16298829563439e9ce950e8510bccc3010147d6b0e547, trainingseed20260908, evaluationseed20260909, purehover256. Both actual arms have bit-identical initialization/RNG and complete first512 policy/privileged/actions/oldBeta/logp/advantages/returns. Both actual probability gates passed before either optimizer. The sole registered intervention is mean-prior0/10, with the same reward. Actual scalar/formula/coverage and checkpoint evidence were independently accepted by Roles22/23.

| Actual development arm/node | Original Z | Maxspeed m/s | Maxdrift m | Maxheight m |
| --- | --- | --- | --- | --- |
| R0 prior0/u192 | 32/32 | .016317671 | .263728857 | .094328642 |
| R0 prior0/u256 | 0/32 | .017724242 | .236132398 | .229004145 |
| R1 prior10/u192 | 32/32 | .010518798 | .031789482 | .046015739 |
| R1 prior10/u256 | 32/32 | .010450779 | .010402661 | .051122665 |

[Role24](reviews/mean-prior-DEV02-actual-pair-Z.json) independently rehashed every original and archive copy and replayed original scoring for all four nodes. Every node retained T0+steps0..199, all32 lanes/eight direction subgroups, startup and failures; all128 episodes were engineering valid, without missing/reset interruption/collision. R0/u256 fails the original height threshold despite its engineering safety-height flag being false. Candidate passing nodes each have all eight subgroups4/4. This supports a credible improvement in this realized development pair and G2/G3; it is not cross-seed confirmation or a unique root-cause proof.

The previous w.05-only method passed development192/256 but FAILED both frozen confirmationseeds20261015/16 at primary256, each originalZ0/32. Confirm01 drift.787366509m; confirm02 drift.327583998m and height.216544390m. Their immutable freeze, actual data, negative independent acceptance and all native failures remain. No thirdseed replacement or horizon extension requalifies that method. Earlier H128/u96failure and u128pass, original C0/C1 positive-relative-but-Z-negative and accepted engineering controlZTS also remain; no repeat E0/mass/control experiment was used.

The new [immutable method freeze](MEAN_PRIOR_CONFIRMATION_METHOD_FREEZE.json) fixes new unused trainingseeds20261017/18 and common primary256 before either start; both must pass the unchanged original Z. Old failedseeds15/16 are separate fixed newmethod regressions and NEVER count for G4. They cannot tune this frozen method. Only the candidate is confirmed, so no cross-seed relative ablation effect is claimed. Exact cards, source, tool, AUTH proposals and budgets are committed. D016 reallocates only unused future reserves inside the unchanged master and leaves the full formal first-segment reserve and all old charges untouched.

Actual new confirmations use the same frozen method and primaryu256. Seed17 completed its actual originalZ and was accepted by Role28. Seed18 completed training accepted by Role29 and actual originalZ in the distinct evaluation-only resource recovery. [Role31](reviews/mean-prior-G4-regression16-training.json) independently accepted both raw results and the immutable freeze, explicitly grantingG4. The beforeApp resource refusals produced zero science and remain separately charged, never classified as failed Z or used for seed replacement.

| Prospectively frozen training seed / primary | Original Z | Maxspeed m/s | Maxdrift m | Maxheight m |
| --- | --- | --- | --- | --- |
| 20261017/u256 | 32/32 | .010489066 | .019358605 | .047899961 |
| 20261018/u256 | 32/32 | .010393206 | .007172675 | .045454979 |

All eight direction subgroups are4/4 for both confirmations, with full sensorT0+200steps and all32 engineering-valid lanes/no missing or reset. [Actual17 analysis](experiments/R5-MEAN10-CONFIRM-01/R1-evaluation-analysis.json) and [actual18 analysis](experiments/R5-MEAN10-CONFIRM-02-Z-RECOVERY1/R1-evaluation-analysis.json) bind original raw and actual checkpoint SHAs. Oldseed15/16 newmethod regression training was accepted by Roles30/31; both complete actual primaryZ regressions now also pass32/32/all8groups4/4, with originalT0/200/all32 and noreset/missing. Their separate regressions cannot replace either frozen new confirmation.

| Gate | Current evidence |
| --- | --- |
| G1 method | Implemented/reviewed finite prior with deployment boundary preserved; final audit pending |
| G2 primary original Z | Development PASS, Role24, primary256 |
| G3 two predefined late nodes | Development PASS, Role24,192/256 |
| G4 unused seed confirmation | PASS, both immutable17/18 primaryZ32/32, independentRole31 G4_granted:true |
| G5 learning implementation | Both DEV02 real512 gates precede both optimizers; actual training accepted Roles22/23 |
| G6 finite nonzero transition | Actual112×32/3584/0reset PASS_IMPLEMENTATION_ONLY; CPUfullreplay and Role32 actualaudit pass, finalexportpending |
| G7 same-device checkpoint recalculation | ActualCUDA u1afterMB8 mean/std/logp and finalu256strictindependentreload alpha/beta/mean/logp errorsALL0 at original1e-6/1e-5; state/Adam/RNGunchanged |
| G8 formal package | Exact frozen formal_r5_launch_manifest.json includes fullS3/config/card/AUTH/spec/argv/source/control/actualqualification/budget; finalconditionalissuancepending G9 |
| G9 final independent review | Role32 actualfinalreview inprogress; all33researchentries terminal andsettled, noformaljobs |

| Old failed seed, new method / primary | Original Z | Maxspeed m/s | Maxdrift m | Maxheight m |
| --- | --- | --- | --- | --- |
| 20261015/u256 | 32/32 | .010415698 | .011686444 | .046066999 |
| 20261016/u256 | 32/32 | .010371472 | .012983061 | .046873569 |

[Actual finite G6/G7](experiments/R5-MEAN10-INTEGRATION-02/R1-transition-analysis.json) binds the actual112-step zero16/positive32/negative32/zero32 protocol,3584nontraining and zeroreset/train/optimizer/checkpointwrites. Full-policy command/shape/index/xyzdecoder/control/reference checks ran in production and compactrecord checks were independently replayed. The actualu1 afterMB8 own512-input CUDA mean/std/logp and finalu256 independentstrictreload on the same fresh32-input both have maximumabsoluteerrors0 using unchangedatol1e-6/rtol1e-5; model/Adam/RNGstates remain unchanged. The full nonzero task success rate is not asserted. All33entries are terminal with normalownedACK/twostableempty scans/noowners ordescendants, completeinventories and exactlytwopostrun; allnativeFAILED/INCOMPLETE/fullreserves remain. Noadditionaldiagnosticrun isrequired.

Compatible integration sourcefeb1e96/configd263cdef has identical critical product hashes to mean-method development, with explicit strict origin/execution compatibility declarations. The prospective formal segment restores the SAME actual DEV02 R1/u256 into both arms, then fresh-resets simulation/history/reference. It reinstalls current prior0/10 after origin load, both reward.05, retains first real512 gate, and runs additional256 updates (absolute257–512), original mixed commands with explicit center-only resets. Five new checkpoints272/288/320/384/512 each receive full original Z/T/S/X/W evaluation. [Formal plan](FORMAL_R5_TRAINING_PLAN.md) and [guarded later entry](START_FORMAL_R5.md) are prepared; this Goal never submits those formal jobs. Performance on the full nonzero task remains unknown and is not asserted by finite implementation checks.

All four DEV02 native jobs remain FAILED/-9/rawPASS_PENDING_CLOSE/INCOMPLETE/null, with full logical/output reservations retained. Normal owned ACK, two stable empty scans/no survivors, complete inventory and exactly two authorized postrun queries per job permit separate scientific acceptance. They do not create a process PASS or refund. The ledger distinguishes observed research totals from conservative used+reserved totals; no old failed task was resumed, no old quota reset, no unrelated process killed, no runtime dependency changed. Current main and origin identities are read from git; precise run inputs and tags are recorded separately from evidence commits.
