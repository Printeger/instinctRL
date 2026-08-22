# M7-001 Final Report

## Revision 1 result

`READY_FOR_REVIEW`. M7-001-D0002 corrects preserved D0001 HEAD
`95cfd849d55425d5d9f7dd5a07903b99a080a544`. It is limited to rejecting
policy/evaluation shared storage and replacing the weak arithmetic isolation
test with accepted `HistoryActor` distribution/action/log-probability proof.
Metric and record semantics below remain unchanged.

The public boundary now rejects direct, differently shaped, strided,
nonzero-offset, and LiDAR/state-originating shared storage before metrics or
record construction. Independent equal-valued storage remains accepted. Under
restored identical RNG, materially changing evaluation values preserves exact
HistoryActor alpha, beta, sampled action, and log probability; a valid policy
perturbation changes the distribution.

Fresh D0002 formal evidence passes 44/44 commands and 7/7 assertions: M7 38,
M3-M6 twice at 204 tests plus 33 subtests, and M2 static 10. Both independent
records retain exact D0001 SHA256 and compare at maximum error zero. The five
record negatives reject, all 315 D0001 evidence files remain byte-identical,
and unaffected product/frozen/package identities pass.

## D0001 result (historical; Commander requested rework)

`READY_FOR_REVIEW`. Starting from accepted base
`033514d8fa7d8d2bb2ece5faa1521d528e32b0a4`, M7-001 adds the ordered pure-
tensor evaluation core, exact M4 audit boundary, exhaustive contract tests, and
bounded evidence. Implementation commit:
`fbc0cea12eaaf2afc4bb6d2a176c629021d2d126`.

## Information contract

The only actor branch is the unchanged M4 pair. All actual velocity, world
position/reference, clearance/collision, ICS diagnostics, termination, and
observability fields enter only the separately named evaluation branch. The M4
audit executes before evaluation validation or computation. Static reverse-
import audit and runtime perturbation tests prove evaluation-only changes do not
change a fixed actor output, while actor input changes discriminate.

## Metric and record result

Ten ordered metrics cover body-frame command tracking, null-command world-frame
station drift and body speed, measurement-anchor error, minimum clearance,
collision, ICS violation, command preservation, intervention frequency, and
source-labelled observability. Conditional metrics explicitly distinguish
zero, one, and mixed valid samples. Empty selections are unavailable/null with
zero denominator and valid count.

The canonical finite JSON identifies schema `m7.evaluation.v1`, platform
`TASLAB_UAV`, sensor `Livox MID360`, actor audit and input hashes, `T,N` and
sample count, config, source classifications, metrics/counts, independent M6
termination counts, and exact provenance. Two independent processes generated
byte-identical complete records; the strict comparator measured maximum error
zero and rejected all five mutation classes.

## Verification

Accepted formal candidate `0002` passed 38/38 commands. The M7 suite passed 34
tests; accepted M3-M6 pure suites passed twice (204 tests and 33 subtests each);
M2 static passed 10. Accepted implementation/test paths, formal source, Isaac
Lab, OmniDrones, NavRL, package listing, driver identity, and clean branch were
checked without mutation. Exact logs and hashes are indexed in
`EVALUATION_EVIDENCE_INDEX.md`.

## Scope declaration

No simulator or runtime smoke was launched. No rollout was collected; no policy
quality was evaluated; no observability algorithm or B0-B8 ablation was
implemented or run. No training, M8, 128k, 1M, scale run, package/system/runtime
change, frozen/external change, merge, push, or history rewrite occurred.

NO NEXT-PHASE WORK WAS STARTED. AWAITING COMMANDER REVIEW.
