#!/usr/bin/env bash
set -euo pipefail
test ! -e /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/runs/R6-DIRECT-VIDEO-02
test ! -e /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONFIRMATION-128/results/R6-DIRECT-VIDEO-02
test ! -e /var/lib/instinctrl-r6-native/J-324fe806adf2c0f0/output/R6-DIRECT-VIDEO-02
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-02
printf '%s  %s\n' b9b9879720054c35316f2777c766db7f4f862da33a9e394d4878ef9a5ba0ee64 /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-02/inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-02:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/render-02/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-02/inputs.json
