# Shared evaluation tooling

This package implements the prospective Control Plane V2 DRAFT-to-SEALED
evaluation protocol. It has no third-party dependency, performs no network or
package-install action, and remains **UNCERTIFIED** until a Commander record
names an exact tooling commit.

Future evaluation namespaces are Builder-authored with exactly two canonical
files: `ATTEMPT_MANIFEST.json` and `INPUT_HASHES.json`. All other evidence is
created by this package. Commands are explicit argv arrays. The only certified
launch profile is `DIRECT_PINNED_PYTHON_SOURCE_V1`: every primary and cleanup
must use the exact non-symlink Python interpreter currently executing the
tooling plus one absolute, regular, non-symlink, exact-pinned source file.
Native utilities, renamed/copied executables, wrappers/loaders, interpreter
`-c`/module/stdin forms, relative or attempt-local source, and downloaded
executables are rejected regardless of filename, role or executable bit.
The manifest declares exact tooling/product repository, commit, tree and
product-clean requirements. Preflight derives the loaded package origin,
compares every package file with the pinned Git tree, and proves actual product
HEAD/tree/status; caller-provided commit strings alone are never accepted.

Run from the coordination repository without installation:

```text
python -m evaluation_tooling preflight ATTEMPT --tooling-commit COMMIT --product-commit COMMIT
python -m evaluation_tooling run ATTEMPT --tooling-commit COMMIT --product-commit COMMIT
python -m evaluation_tooling run-command ATTEMPT 0 --tooling-commit COMMIT --product-commit COMMIT [--repair-of ID]
python -m evaluation_tooling seal ATTEMPT --tooling-commit COMMIT --product-commit COMMIT
python -m evaluation_tooling inspect ATTEMPT
```

`preflight` and post-runtime checks collect all cheap deterministic findings.
Runtime commands stop on the first semantic/safety failure. `seal` creates
`TERMINAL_MANIFEST.json` atomically as the last attempt write. `inspect` reads
only persisted evidence and never executes product or attempt-local code.
Pinned live inputs, executables, and sources are rechecked immediately before
and after each primary and cleanup launch. A versioned launch-profile proof is
bound into the persisted identity proof. Offline inspection verifies that proof
against sealed manifest and input-pin bytes without consulting the inspector's
Python executable or requiring the former live files to remain available; this
does not authorize a different interpreter for live execution. Each command uses a disclosed persistence transaction;
interrupted streams/metadata/receipts are preserved, recovered as an explicit
failed ledger row, and may be retried only through `--repair-of`. Successful
seals end in an exact zero-child `seal` receipt with `result=PASS`; failed seals
retain DRAFT and an exact `result=FAIL` receipt.

After persisted runner state proves runtime start, executable-looking regular
files are treated as inert output only below the exact
`results/runtime_support/tmp/` and `results/runtime_support/cache/` roots.
They remain unusable as primary or cleanup executables/sources and are bound in
the v2 terminal inventory by path, type, permission mode, size and SHA-256.
Pre-runtime content, symlinks and executable-looking files elsewhere remain
fail-closed. Offline inspection retains compatibility with sealed v1 packets.

Canonical JSON uses sorted keys, compact separators, ASCII escaping, finite
numbers, and exactly one trailing newline. The versioned schema examples live
in `schema_documents/`. The initially approved cross-backend geometry policy
is `FLOAT32_GEOMETRY_ULP2_ABS1E5_V1`: every value must be both within two local
float32 ULPs and within `1e-5` metres.
