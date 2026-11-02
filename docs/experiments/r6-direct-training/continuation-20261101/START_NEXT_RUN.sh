#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-VIDEO-04
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-VIDEO-04
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-VIDEO-04
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-04
printf '%s  %s\n' 0c9f82da66d94c8e746b45050ca3f9d04608c7891af77c20a596d8194b8d0c5e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-04/inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-04:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-04/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-04/inputs.json
