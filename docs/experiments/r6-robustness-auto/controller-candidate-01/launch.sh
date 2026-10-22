#!/bin/bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:tools:tools/rlflow
export GIT_TERMINAL_PROMPT=0
exec /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python -u -B -m tools.r6_controller \
  docs/experiments/r6-robustness-auto/batch \
  docs/experiments/r6-robustness-auto/controller-candidate-01/request.json \
  --request-sha256 4a21a73cb775b37ae1699f6b1fefc6d68ad1ec2c0e266f6f2de8e3c5e90cf9e5 \
  --store /home/mint/rl_dev/instinctRL-dev/coordination/research \
  --wait-platform-handoff 600 \
  >> docs/experiments/r6-robustness-auto/controller-candidate-01/controller.log 2>&1
