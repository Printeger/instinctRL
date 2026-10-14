# Formal R5 first segment

Status: prospective plan frozen for review; G4, actual G6/G7, exact control-tool package and final independent acceptance remain pending. This Goal does not submit formal training.

Both formal arms start from the same accepted development R1/u256, not separate R0/R1 checkpoints. Original checkpoint trainingseed20260908 and all actor/critic/Adam/RNG/clock state are restored exactly. Reconstruct simulation and explicitly fresh-reset it; no physics/history/anchor/actuator continuation is claimed. This is a prospective warm-start residual-reward intervention. It does not replace the completed cold-start development comparison or establish each ingredient's independent effect.

| Item | Frozen first-segment choice |
| --- | --- |
| Execution source | 90aeeb22c7af5ee77853ad5ee34d2098ce23f904, tree9cb3f7ec0256945ce7be2bcc4c0754509f1dcb0a |
| Config SHA | dfb4fd1f0c143e3ac94f88ded9292900ff327dddb2b68082389f6dcbf10bcb5f |
| Execution checkout | coordination/research/workspaces/R5-formal-integration-01-20261014, exact read-only snapshot |
| Origin source/config | a13c5a459d471799f30644a9a04189d6e1805f02 / 73aeb4ff6db0c72fdd4a0e01efec7785b23fa3e8d8885c546fb91b7261007721 |
| Shared actual origin | R5-H256-DEV-01-R1-RESEARCH-TRAINING-a13c5a459d47/results/research-training/checkpoints/u256.pt |
| Origin checkpoint SHA | ddc4a93a4f3b2e3689663bc69f187b29f7fae70e4c0a4f834ceae30b24fdcac1 |
| R0 / R1 future residual weight | 0 / .05; both restore the same qualified R1 origin |
| All other method settings | HistoryActor Beta four actions; approved sensor/measurement history+validity/time+IMU+commands only; privileged21 critic; actorLR1e-4/criticLR3e-4; anchor4; original governor/ICS/Lee control and mass/allocation fixes |
| Update budget per arm | 256 additional updates, absolute257–512; 131072 new transitions; 2048 additional Adam steps each |
| Command phase | Existing mixed clock global4096–8191, alternating16hover/16nonzero lanes,128-step role blocks, original .2axis/.15diagonal directions |
| Reset phase | Center-only; old near-wall reset placement suppressed explicitly, independent of mixed command clock |
| Fresh-reset seed | actual original trainingseed + saved reset_number +1; same input checkpoint gives both arms the same seed |
| Evaluation seed | original20260909, unchanged case construction/T0/steps/functions/thresholds |
| New checkpoints | Absolute272,288,320,384,512;5 new writes per arm, never overwrite origin |
| Evaluation | All five actual checkpoints, complete originalZ/T/S/X/W×32 lanes;164800 nontraining per arm |

Actor inputs do not gain position, odometry, actual/estimated velocity, map, SLAM, obstacles or simulator private state. Reward/critic/evaluation privileged uses retain their existing boundaries. All critical product source hashes are identical between origin and execution snapshots; the explicit compatibility declaration binds the original and execution five-field identities/configs and rechecks both manifests. New workload code integrates diagnostics and the formal phase. Method hashes explicitly bind contract675fe7b24c858d9aa24c963acd591f77c2a99b45dc181073bcc39663314791c7, actual mixed curriculum-h1-v1, center-only reset distribution, warmup/fresh-reset initialization and origin checkpoint SHA. No serializer enum expansion is used.

The newly reconstructed trainer's first actual512 rollout in EACH arm must pass the original probability/Beta gate before either optimizer. starting_update=256, actual retained inputs257/258; original atol1e-6/rtol1e-5. Gate/nonfinite failure stops that update and job, preserves all evidence and partial accounting, and never permits a few updates to repair a mismatch. Adam entry charges subtract the restored2048 steps while actual checkpoints/logs retain absolute totals. Reset counts accumulate origin plus new resets honestly.

Per training entry:7200s action+10s cleanup,8GiB output,1start/init,10init ticks,1freshreset,131072 maximum auto-reset lanes,256updates and5writes. Per evaluation:3600s action+10s cleanup,8GiB output,1start/init,10ticks,25case resets,164800nontraining/auto-reset maximum,0training/updates/writes. Aggregate:262144training,329600evaluation,4starts/init,512updates,4096Adam each,10writes,21640GPU/wall seconds and32GiB output. Existing formal reservation21680seconds includes40seconds conservative extra buffer; it is not free retry or unused-budget refund.

Run serially through the existing rlflow exact approval/submit/receipt/normal owned-cleanup path. The conditional formal permission must bind the final independently accepted readiness evidence, this exact source/config, experiment card, actual origin, tool release and exact inputs. A later explicit formal-start action is required; this readiness Goal never calls the formal submit action. STOP, owner/guardian/descendant quiescence, disk≥30GiB floor, original coexistence and remaining master quotas are rechecked at actual release. Consumed formal reservations are transferred to charged live entries without resetting historical or confirmation buffers.

Evaluation hashes that do not exist yet are defined as generated inputs, never prefilled. After each bounded training job has terminal evidence and normal owned cleanup, independent actual-training/checkpoint acceptance binds all five actual checkpoint SHAs before its exact evaluation release. Full original evaluation retains T0, all startup, reset interruptions, worst lanes and failures. Any originalZ regression at any node blocks further continuation. T/S/X/W performance is recorded as measured, not assumed from the finite implementation check or warmup. Retain all nodes, never select only the best formal checkpoint after seeing results.

The first segment stops at absolute512. Further low-speed tracking, stopping/reversal, near-wall or generalization stages require measured competence, a new prospective exact phase decision, remaining-budget accounting and independent review/release. No automatic horizon extension or complete R5 convergence is promised. Known native-close FAILED is tolerated only with complete separately accepted scientific evidence, preserved INCOMPLETE/full conservative reservations and verified normal owned cleanup; unknown exits or ownership ambiguity stop further work.

Executable workload modes are research-formal-training and research-formal-evaluation in the existing formal_unified_h1 module. Exact control-tool inputs, conditional release identities and operator commands are supplied by formal_r5_launch_manifest.json and START_FORMAL_R5.md after final package acceptance; this prospective plan itself is not permission.
