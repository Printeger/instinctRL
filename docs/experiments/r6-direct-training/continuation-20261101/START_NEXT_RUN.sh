#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-FINAL-BASELINE-01
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-FINAL-BASELINE-01
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-FINAL-BASELINE-01
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01
printf '%s  %s\n' 3dfa5f1b275b2265374d524cc5bfe26c23e89c8ea1f6d1c234a8e5b22c323a2d /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/final-evaluation-01/baseline-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/final-01/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/final-evaluation-01/baseline-inputs.json
