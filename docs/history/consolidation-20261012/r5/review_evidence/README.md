# Review evidence boundary

This directory separates the sealed predecessor receipt from the current host-executor successor. `HOST_CPU_EVIDENCE.json`, its associated v1 files, and the v2 runner/manifest are historical/provenance-only evidence; they are not current-successor acceptance evidence and their bytes remain unchanged.

The current successor exposes exactly one external host command:

```json
["/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python", "-m", "training_workloads.unified_h1.host_cpu_evidence_successor"]
```

Run it once from the repository root on the authorized fork/signal-capable CPU host. The runner requires the exact installed interpreter, pytest 8.3.5, and the `referencing` dependency, disables third-party plugin autoload, and performs separate collection and execution for every module in `host_cpu_test_manifest_successor.json`. It fails closed on interpreter, dependency, pytest, identity, inventory, source, output, collection, or test-result drift and does not install or modify dependencies.

On success it creates immutable regular files `HOST_CPU_EVIDENCE_SUCCESSOR.json`, `HOST_CPU_EVIDENCE_SUCCESSOR.sha256`, and four successor-specific collection/execution output files. The receipt binds the current package/task/base identity, reviewed predecessor diff, exact host and per-module pytest argv, explicit node inventory, per-module collected/pass counts, executor identity, all invoked source bytes, all outputs, failed predecessor-attempt provenance, historical predecessor-only provenance, and the detached-child cleanup audit. Independent review must recompute the receipt hash and call `verify_host_cpu_evidence` against these current successor files.

## Source-closure successor

The current source-closure successor identity is package `INSTINCTRL-R5-STAGE1-SOURCE-CLOSURE-SUCCESSOR-20260929`, task `T-1620eea16becdc19`. The prior host-executor successor remains immutable predecessor provenance.

Its only external host command is:

```json
["/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python", "-m", "training_workloads.unified_h1.host_cpu_evidence_source_closure_successor"]
```

The runner deterministically resolves direct and transitive repository-local Python imports under the exact host `PYTHONPATH`, including package initializers and helper test modules. It rejects omissions, ambiguity, path escape, duplicates, or closure drift before execution. The immutable receipt is `HOST_CPU_EVIDENCE_SOURCE_CLOSURE_SUCCESSOR.json` with companion SHA-256 and source-closure-specific output files; every verified closure member is included in its source hash inventory.
