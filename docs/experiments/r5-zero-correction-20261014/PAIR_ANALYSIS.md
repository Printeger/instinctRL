# R5 zero-command pair: actual evidence, original-Z comparison blocked

**BLOCKED_BY_SPECIFIC_DEFECT; Goal unfinished.** C0/C1 complete real64-update training and accepted first512 gates. C0 originalZ complete. C1 originalZ has no trajectories: fourthentry failed existingGPUcoexistence beforeApp, andall4start/initreservations are consumed. This is neither VALID_EXPERIMENT_POSITIVE nor VALID_EXPERIMENT_NEGATIVE.

| group/node | original Z successes | max speed RMS m/s | max XY drift m | max height deviation m |
| --- | --- | --- | --- | --- |
| C0/u16 | 0/32 | 0.060848549 | 0.828733861 | 0.817441702 |
| C0/u64 | 0/32 | 0.102369525 | 1.959831357 | 0.471305609 |
| C1/u16 | NOT_EXECUTED | UNKNOWN | UNKNOWN | UNKNOWN |
| C1/u64 | NOT_EXECUTED | UNKNOWN | UNKNOWN | UNKNOWN |

OriginalZ thresholds remain speed<=.08, drift<=.30, height<=.20 withsensorT0/startup/safety/reset/32lane/d0-d7denominators intact. C0 allthreshold failures remainactualfailures. C1 has zeroactualevaluationsteps/transitions; UNKNOWN isnot0/32 andno syntheticmetric orsamplefilled. Primarycomparisonu64 cannotbeconcluded.

Bothgroups useexactread-only d257f9/config4746f5... method ecc75... andsameactorLR1e-4/critic3e-4; source52/tool96pinned. Actualactor/criticinit andfirst512policy/privileged/actions/oldBeta/logp bitexactsame, independentRole4accepted. Actualadv/returns maxdifference2.457344 isallowedrewardeffect. C0w0 equivalence andnewformulaC1all32768samples proved; C1maximumformulaerror1.008133e-7.

u64trainingmeanissuedresidual: C0[+.075160,+.042557,+.021312], C1[-.006825,+.009512,-.002581]m/s. C1 reducesmeanresidualinthistrainingwindow; thisdoesnot substitute a deterministicZtrajectory orproveclosedloopcausality. Body/worldtiming, resetsegments andactorconditionalmeans remainseparate fromsampledactionstatistics. Firstmb8 analyticBetaKL C0=.276194468/C1=.196501259; gradientnorms/all64updateKL/advantage/rewardvalues remainintrains/actualprobe inventories andpair_analysis.blocked.json.

PreviousLR progress remains historical: P0/PLRfirstmb8KL15.012778/.276192, oldPLRu64speed/heightpassbutdrift1.088824mfail. NewC0 iscontemporaneouscontrol; itsu64outcome differsfromoldPLR, sooldPLR cannotreplaceC0 orbeusedasafakepairedcandidateZresult. Onepairedseed andnumericaltrainingvariability limitgeneralization.

Threecompleteddataentries remainnativeFAILED/-9/PASS_PENDING_CLOSE/INCOMPLETE/null withfull65536train+12800eval+128update+14write reservations. FourthC1eval remainsFAILED/1/rawFAIL/formalFINAL0; originalcollectorproveszeroandsettlesits12800unusednontrainingreservation. Itsstart/init/output16GiBreservations remaincharged. AllnormalownedACKtwoempty/no survivors; exact2postrunauditsperentry. Noautoretry, nooldjobsresumed, noextraweight/256update.

C1prelaunch observedthreewindows4%,6%,2%, max6>original5%; memory521MiB, onlyexistingGNOME3950/164MiB. Laterauthorizedpostrunaudit7% retained; cannotcausallyattributeGPUutil solely toGNOME orrelaxthreshold. NoAppconstructed/physicscountunobserved. No unrelatedprocesskilled.

Used4starts/init, observed65536train/12800eval/128updates/14writes, wall747.398786380s/GPU744.903665893s; original+archiveoutput4282373184bytes beforetaskmetadatareconciliation. Outputreservationsall64GiB retained. Capsnotreset. Exactfifthresourceproposal requestsstarts/init4to5/output64to80GiB; othercapsincludingnontraining25600/wallGPU10840 unchanged. Existing12800nontrainingand>=10092s wallremainbutnewstartandoutputreservationcannotfitcurrentcaps. Proposalonly, actualAUTH/approval absent.

Unique recovery: explicitone-time resourceexception, then existingCommander bindsnewacceptedreview/userdecisionSHA, exactfreshC1Zspec/AUTH realu16/u64SHA, commits/tags/pushes, idle/STOP/disk/originalcoexistence gates, singlefreshrelease. Freshattemptneverresumesfailedentry; noadditionaltraining oroptimizer. CompleteC1Zandindependentfinalscientificreview then fullpairresult, notanotherbroadinvestigation.
