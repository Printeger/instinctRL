#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-CONFIRM-128-SEED-02
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-CONFIRM-128-SEED-02
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-CONFIRM-128-SEED-02
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/confirmation-128-02
printf '%s  %s\n' 706171478579feb03abc5a505cab78a55f0e7511d9ad90da2519bdb79beb88ff /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/confirmation-128/seed-02-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/confirmation-128-02:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/confirmation-128-02/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/confirmation-128/seed-02-inputs.json
