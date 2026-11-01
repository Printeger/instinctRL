#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-FINAL-SEED-01
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-FINAL-SEED-01
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-FINAL-SEED-01
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01
printf '%s  %s\n' 16f23c39c114031d6f3048399691d241b2df451dba5073b0e3adcb1b29022fea /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/final-evaluation-01/seed-01-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/final-evaluation-01/seed-01-inputs.json
