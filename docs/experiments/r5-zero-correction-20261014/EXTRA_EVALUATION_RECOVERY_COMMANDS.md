# Exact completion route after an explicit user resource decision

Current status: **UNAUTHORIZED / no new live action**. The active Goal's
four starts are exhausted. Automatic continuation, conditional Reviewer PASS,
or the old diagnostic eight-start decision supplies no fifth-entry authority.
The exact pending proposal remains C1_EXTRA_EVALUATION_RESOURCE_PROPOSAL.json
SHA256 ef1f8b31c86fb22acb9ac0a1935fb95e99d0fa01a96eef622b4b5f3d4d7f12e7.

Once an explicit user grant arrives, Commander records its actual provenance in
user_resource_decision_extra_evaluation.json: decision APPROVED, proposal_sha256
above, approved_caps exactly the proposal's requested caps, one fresh entry,
automatic_retries 0. Update only the approved caps in the current ledger; retain
all old jobs, charged starts/init, logical and disk reservations and observations.
The unchanged frozen d257f9 candidate remains the common method baseline.

Use the prepared existing-Commander adapter:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python docs/experiments/r5-zero-correction-20261014/commander_tools/r5_zero_authorize_extra_eval.py
```

It refuses missing/mismatched user authority, accepts the exact Role5 predicate
and decision SHA, issues the prospective AUTH bytes only once, and creates a
new proposal/approval and inputs/C1-extra-evaluation-spec.authorized.json.
Commit all actual authority/spec inputs, tag the input version, normally push
main/tag, and verify HEAD equals origin/main before the separate single submit:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python docs/experiments/r5-zero-correction-20261014/commander_tools/r5_zero_submit_extra_eval.py
```

Poll only that exact fresh job handle. The original GPU<=5% coexistence gate,
STOP, owned cleanup, output quota, and zero automatic retries remain mandatory.
Do not resubmit or reuse the failed C1 job. No training update or checkpoint write
is authorized. Terminal receipt plus normal owned ACK must precede settlement
and the two already bounded postrun audit calls:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python docs/experiments/r5-zero-correction-20261014/commander_tools/r5_zero_settle_extra_eval.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python docs/experiments/r5-zero-correction-20261014/commander_tools/r5_zero_postrun_extra_eval.py
```

Then analyze the new issued spec and write new named evidence. Do not overwrite
the original failed C1 analysis or pair_analysis.blocked.json:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:isaaclab-training /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 tools/analyze_r5_zero_correction_pair.py --mode evaluation --group C1 --evaluation-spec docs/experiments/r5-zero-correction-20261014/inputs/C1-extra-evaluation-spec.authorized.json --output docs/experiments/r5-zero-correction-20261014/C1-extra-evaluation-data-analysis.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:isaaclab-training /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 tools/analyze_r5_zero_correction_pair.py --mode pair --c1-evaluation-analysis docs/experiments/r5-zero-correction-20261014/C1-extra-evaluation-data-analysis.json --output docs/experiments/r5-zero-correction-20261014/pair_analysis.complete.json
```

The CPU binding verifies the actual issued AUTH/source/coefficient/checkpoint
hashes, exact argv, unique real job request and terminal cleaned receipt.
Incomplete C1 evaluation cannot generate a complete pair. Full numerical
comparisons remain PENDING_INDEPENDENT_SCIENTIFIC_ASSESSMENT: no heuristic
percentage score labels a result positive/negative or grants policy admission.

Independent final review must assess original Z coverage and the actual
preregistered u64 comparison, retaining u16 diagnostics, old PLR as historical
context, all failures and native process/collector statuses. After that review,
reconcile the current report/index/machine state with references to the immutable
blocked report and new complete artifact, perform the whole requirement audit,
commit/normal push, and end this Goal only if the requested scientific conclusion
is actually achieved. No 256-update extension is part of this route.

CPU-only validation before user decision: cpu-analysis-binding-final.xml records
9 PASS. cpu-analysis-binding-C0-replay.json and
cpu-analysis-binding-verification.json prove exact existing C0 metrics/coverage
were preserved. The recovery commands above have not been executed.
