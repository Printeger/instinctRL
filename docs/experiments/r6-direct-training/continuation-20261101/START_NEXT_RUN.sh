#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-VIDEO-07
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-VIDEO-07
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-VIDEO-07
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-07
printf '%s  %s\n' 06ebecba21b8bace6b7506ab75fae88adffb0651dad84bace5aa380ffe371a03 /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-07/inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-07:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-07/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-07/inputs.json
