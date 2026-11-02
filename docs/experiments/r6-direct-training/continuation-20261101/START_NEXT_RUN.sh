#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-FINAL-SEED-02
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-FINAL-SEED-02
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-FINAL-SEED-02
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01
printf '%s  %s\n' eff94edc9c36ddad0e4933412f461cb62fa712df7e228e4c507b468ae6e1cb95 /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/final-evaluation-01/seed-02-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/final-evaluation-01/seed-02-inputs.json
