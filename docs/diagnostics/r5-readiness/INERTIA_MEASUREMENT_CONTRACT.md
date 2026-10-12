Float32 inertia measurement candidate, before the next live run

Independent Reviewer call 8 identified LIVE-R1/R2. Actual v3 raw has zero
rows and no matrix: its inertia failure branch and physical cause remain unknown.
`inertia-float32-counterexample.json` is constructed CPU evidence, never a live
body measurement. It proves the fixed absolute 1e-10 symmetry check can reject a
positive float32 rotated inertia (maximum skew 1.862645149230957e-9).

New source-precision contract: PhysX public get_inertias float32 readback,
epsilon=2^-23, gamma7=7*epsilon/(1-7*epsilon). A rotated positive principal
inertia R D R^T involves two three-term dot products and readback rounding.
Absolute dot sums are bounded by the trace of positive D; opposite entries can
differ by two rounding bounds. Account for uncertainty in the measured trace:
maximum_skew <= 2*gamma7/(1-gamma7)*sum(abs(diagonal)). No absolute floor,
dataset fitting, matrix averaging, mass replacement or tensor repair occurs.
Raw accepted entries also remain untouched in subsequent double-precision
derived arithmetic. Positive diagonal remains strict; this check does not
claim to establish all physical inertia properties. COM/momentum/wrench
comparison thresholds remain 1e-8. This is a measurement contract correction,
not a change to control acceptance criteria.

At a failed body derivation, product captures separate UNVALIDATED full body
primitives, exact coordinate/lane, failed body index/name/branch, getter names,
dtype/shape and root/feedback state. Nonfinite values have explicit tagged
representations rather than zeros. Backend persists a separate artifact and
hash reference while keeping the original error and zero accepted physics rows.
Observer/reset state is not advanced. Failed data never enter causal analysis.

`measurement-failure-cpu-v1` verifies these boundaries with AST-extracted product
methods, actual CPU tensors and backend JSON writes; it is mock integration,
not Isaac Sim integration. The next new identity is
UNIFIED-H1-R5-READINESS-B0-MEASUREMENT-R1-20261012. The consumed v3 attempt is
preserved; its full 20,160 transition reservation is retained. No live release
is conferred by this document or CPU tests.
