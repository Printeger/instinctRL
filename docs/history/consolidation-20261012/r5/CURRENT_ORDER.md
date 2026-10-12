# Order M1A-001 — TASLAB_UAV parity provenance

Status: `ORDERED`

Commander: sole order writer and approver

Executor: sole implementation writer

## 1. Objective

Produce a reproducible, static, evidence-backed provenance and parity package for the legacy TASLAB_UAV and the frozen modern comparison stack. Establish what exists, where it came from, how the legacy runtime consumes it, what is absent or different in modern OmniDrones/Isaac Lab, and what M1B could later migrate. Do not migrate, copy, convert, or execute the platform.

TASLAB_UAV must be described as derived from a third-party AechPro Isaac Sim 5.1 fork plus local feasibility patches. It must not be described as an official modern OmniDrones implementation. The frozen modern OmniDrones tree is expected not to contain TASLAB_UAV; prove and record that absence.

## 2. Immutable execution coordinates

- Order ID: `M1A-001`
- Formal source witness: `/home/mint/rl_dev/instinctRL`
- Formal source branch/commit: `main@6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
- Executor worktree: `/home/mint/rl_dev/worktrees/instinctRL-blackwell`
- Executor branch: `blackwell-migration`
- `BASE_COMMIT`: `a0619aa4e60ecc23a016d339f5c32fa4a98ad967`
- Frozen baseline: `BLACKWELL_BASELINE_V2_R580`
- Frozen modern OmniDrones comparison pin: `5045c71` as recorded by qualification evidence
- Frozen NavRL comparison pin: `3725bcc2e7c1be4ecf1455d922299ae85042603a`

Before any write, prove that the Executor branch is exactly at `BASE_COMMIT`, both source and Executor worktrees are clean, the formal source still has the branch/HEAD above, the two order files are byte-identical, and the Commander/Executor states are `ORDERED`/`IDLE`. If any fact differs, stop without changing either repository and report `BLOCKED`.

The short OmniDrones pin must be resolved only from the completed qualification evidence. If the evidence cannot identify its exact repository and full commit object, record the failure and stop without a product commit. Do not substitute an upstream tip, AechPro tip, release, cached version, or guessed full SHA.

## 3. Product-tree allowlist

Create and modify exactly these four files, all under the M1A documentation directory:

1. `docs/blackwell_devlog/m1a/M1A_TASLAB_PROVENANCE.md`
2. `docs/blackwell_devlog/m1a/TASLAB_SOURCE_MANIFEST.md`
3. `docs/blackwell_devlog/m1a/TASLAB_ORIGINAL_VS_MODERN_DIFF.md`
4. `docs/blackwell_devlog/m1a/TASLAB_MIGRATION_PLAN.md`

No other product-tree file or path may be created, modified, deleted, renamed, reformatted, staged, or committed. Coordination evidence may be written only under `raw_logs/M1A-001/`, `reports/`, and the Executor state/report files in the independent coordination repository; those artifacts are not part of the product commit.

## 4. Required provenance inventory

Inventory every TASLAB-specific source and every directly relevant dependency/consumer. At minimum include:

- legacy `TaslabUAV` class and its import/registry path;
- TASLAB parameter YAML;
- TASLAB USD crate;
- TASLAB Lee-controller gains;
- `MultirotorBase` registry, parameter loading, spawn/initialize and action path;
- `RotorGroup` construction, parameter meaning, ordering and force/moment application;
- `VelController`, its wrapped Lee controller, command frame, required state and motor-action output;
- training/config consumers selecting `TaslabUAV` and controller gains;
- corresponding frozen modern OmniDrones/Isaac Lab classes, registry/config/spawn/actuator APIs and consumers, including explicit evidence that TASLAB_UAV itself is absent;
- historical README/audit statements that claim a different asset, name, size, authority, or parameter status.

For each inventory row record all of the following, using `unknown` plus a reason when evidence is insufficient:

1. repository identity and authority level;
2. exact commit;
3. repository-relative path;
4. Git blob object ID (or `N/A` with reason for a proven absence);
5. SHA-256 of the exact file bytes;
6. byte size and `file` type;
7. introduction commit and relevant update commit(s), with evidence method;
8. copyright/license owner and applicable license source; never infer ownership from a directory name;
9. runtime consumer(s) and the static call/config path by which the file becomes effective;
10. authority classification: runtime-effective, authored asset, configuration intent, historical documentation, third-party source, local patch, or unresolved.

The current formal-source USD expectation is path `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd`, Git blob `b16b1b16f21166d82579a9a5b190cb49191ddef5`, and 9,563,347 bytes. Recompute rather than copy these values. Compute and record SHA-256 and file type. Explain the conflict with the stale approximately-158-KB historical devlog statement using object evidence, without rewriting history.

## 5. Required static comparison

Compare original/legacy and frozen modern behavior at symbol, config and asset-contract level. Cover, at minimum:

- TASLAB class declaration, inheritance, name/case and registry mechanism;
- drone import and config selection;
- YAML schema and every TASLAB value, with units and comments separated from effective values;
- USD crate identity, default/root prim, rigid bodies, joints, transforms, articulation and collision APIs, mass/inertia authorship, rotor prim count/names/order/axes and relationships;
- Lee gains and lookup/loading path;
- `RotorGroup` input tensors, broadcasting/device behavior, parameterization, output/application semantics and rotor ordering;
- `MultirotorBase` config/parameter load, spawn, initialize, state and action semantics;
- `VelController` input/output keys, command coordinate frame, privileged controller state, saturation and Lee-controller coupling;
- modern equivalents and removed/renamed APIs, with TASLAB absence stated explicitly;
- legacy Isaac Sim/Orbit assumptions versus the frozen Isaac Lab/modern OmniDrones APIs.

Use static source/object evidence only. A textual similarity is not behavioral parity. Mark anything requiring USD stage introspection or runtime execution as an M1B validation requirement rather than claiming it has passed.

## 6. Authority conflicts and risk register

The four documents must explicitly track, not silently resolve:

- stale README paths, model-name casing, placeholder-Hummingbird claims and the obsolete USD-size statement;
- parameter comments that conflict with current numerical values or calibration claims;
- whether USD or YAML is runtime-authoritative for mass, inertia, transforms, joints and rotor placement, traced through the consumer;
- prim paths, root/default prim, rotor/body topology, ordering, axes, transforms and controller-to-rotor mapping;
- binary USD crate compatibility with the frozen modern runtime;
- command/world/body frames, quaternion convention, units, tensor/device/dtype expectations and saturation;
- third-party and local-patch license/copyright provenance, including any uncertainty about the binary USD and AechPro-derived code;
- the unresolved identity of any baseline or source object.

No guess, undocumented default, or “likely equivalent” may close a risk. Each risk needs evidence, impact, required resolution owner/milestone and proposed acceptance check.

## 7. M1B candidate plan only

`TASLAB_MIGRATION_PLAN.md` may propose candidate target paths under the reserved `isaaclab-training/` root, dependency order, copy-versus-rewrite decisions, compatibility adapters, static/runtime validations and acceptance recommendations for a future M1B order. It must distinguish mandatory work from alternatives and identify licensing prerequisites.

It must not create `isaaclab-training/`, copy or convert any asset, implement a class/config/controller, patch a frozen baseline, choose an unresolved topology by assumption, or claim runtime parity. M1B remains unauthorized regardless of the recommendation.

## 8. Command and execution envelope

Only the following classes of operations are authorized:

- read-only Git identity/history/object/status/tree/blame/show/diff operations against exact pins;
- cryptographic hashing (`sha256sum` or equivalent) and Git blob hashing;
- `file`, `stat`, byte counts and read-only filesystem listing;
- read-only text display/search and symbol search (for example `rg`, `sed`, `head`, `tail`);
- static textual/structural diff; no import or execution of project Python;
- creation/editing of the four allowlisted Markdown files and coordination raw logs/report;
- staging exactly the four allowlisted files and creating exactly one commit after all checks pass.

Do not install packages, fetch a moving branch as evidence, modify Git configuration, initialize/modify submodules, run project code, import Isaac/OmniDrones modules, open or convert the USD in a simulator, run Isaac Sim/Isaac Lab, execute TASLAB, RotorGroup, MultirotorBase, VelController, Lee control or MID360, run tests that import runtime code, change PPO/reward/governor/anchor/ICS, or start training/evaluation. No network source may replace a frozen qualified object.

Every command, including failed probes and the final commit command, must have a stable sequence ID and retain:

- exact working directory and exact command text;
- start/end timestamp;
- complete stdout and stderr in separate or unambiguously delimited raw records;
- numeric exit code;
- output SHA-256 when practical.

Create `raw_logs/M1A-001/INDEX.md` in the coordination repo mapping sequence ID to command, working directory, raw-log path, exit code and the document/claim it supports. Never place secrets or environment dumps in logs.

## 9. Commit shape

After content validation, create exactly one Executor commit whose parent is `BASE_COMMIT`. The commit must contain exactly the four allowlisted Markdown files and no binary or runtime changes. Recommended subject:

`document TASLAB provenance and migration parity`

Do not amend the governance commit. Do not create fixup/merge commits. Do not push.

## 10. Acceptance checks

The Executor must run and report at least:

1. preflight branch/HEAD/status and order-byte-identity checks;
2. JSON parse checks for both state files, using a non-project parser/tool without importing runtime code;
3. manifest completeness check showing every required field for every row;
4. exact Git blob/SHA-256/size/type checks for each inventoried file;
5. static searches proving the legacy registry/consumer paths and modern TASLAB absence;
6. static diffs for the required class/YAML/controller/dependency surfaces;
7. `git diff --check` and an allowlist comparison before commit;
8. post-commit proof that `HEAD^` equals `BASE_COMMIT`, the base-to-HEAD diff contains exactly four Markdown files, the commit is docs-only and the worktree is clean;
9. proof that the formal source checkout remains clean at `main@6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`;
10. proof that no `isaaclab-training/` path or asset copy was created.

Static-only checks must be labelled as such. Runtime/physics parity, USD loadability and license clearance cannot be reported as passed without their separately authorized evidence.

## 11. Final report

Write the Executor report to `reports/M1A-001.md` and update coordination Executor state only after the product commit exists. The report must include:

- order ID, `BASE_COMMIT`, Executor commit SHA and commit parent;
- `git diff --name-status BASE_COMMIT..HEAD` and `git diff --stat BASE_COMMIT..HEAD`;
- acceptance/test table with command IDs, results and evidence links;
- raw-log index path and its SHA-256;
- change ledger mapping each of the four files to its purpose and evidence;
- provenance/authority conflicts and known limitations;
- explicit declarations that the product commit is docs-only, exactly one commit was created, the source witness is unchanged, no asset/runtime/new root was created, no runtime/controller/MID360/training was run, and nothing was merged or pushed.

Set Executor state to a report-ready state without self-approving. The Commander alone may review and accept/reject the report.

## 12. Stop conditions

Stop immediately and report without improvisation if a pin cannot be resolved exactly, a qualified baseline is missing, branch/base/status changes, order copies differ, a required license or source identity cannot be investigated statically, an unallowlisted product file changes, a runtime command would be needed to support a claim, or the one-commit shape cannot be preserved.

On completion, stop. Do not issue or begin M1B, do not migrate TASLAB, do not run training, and do not merge or push.
