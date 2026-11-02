#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-VIDEO-03
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-VIDEO-03
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-VIDEO-03
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-03
printf '%s  %s\n' 3dd9228c8a993b5344bb7716c75a667fd9cc4283a99d2f88a9c0b3a9f2b572bd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-03/inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-03:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-03/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-03/inputs.json
