# W Goal event controller

The root prepares/independently reviews exact inputs and pushes main before starting one tmux Python controller. It submits only the approved spec named in the plan, observes that exact Store job and owned process, then performs normal ACK verification, settle and exactly two postrun checks. The inherited FAILED/-9 close-timeout fact remains visible.

Python waits every15s; it invokes Codex only after completed job cleanup, once for a scientific decision. The bounded worker keeps argv/prompt/schema hashes, process identity, native event stream, stderr, final response and exit receipt. Maximum output64MiB per decision; ≤24job starts means ≤1.5GiB such output inside the2GiB metadata reservation. No configured model override. Local CLI0.161.0 actual --help validated; no API probe or dependency change.

The decision model is the only workspace writer while the controller awaits it. It may implement, run CPU checks, commission independent read-only review, register bounded stages and issue/push the next exact plan; it must not submit or start another controller. A fresh reviewed plan then returns control to Python. Ordinary negative results continue within the user envelope. Existing and new STOP paths cancel only the exact owned job and preserve evidence; no new model/job follows STOP.

Controller state is `docs/experiments/r5-w-closure-auto/batch/state.json`. Read it with ordinary shell tools. Never reinitialize or resubmit because an observation timed out. Ambiguous operations fail closed and require evidence-based reconciliation. The old batch and budgets remain unchanged.

Launch after final review and push:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow:isaaclab-training CUDA_VISIBLE_DEVICES='' /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python tools/r5_w_batch.py run coordination/batches/R5_W_CLOSURE_20261019/plan.json
```
