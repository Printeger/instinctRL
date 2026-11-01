#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
printf '%s  %s\n' '51a51147036bef23944a601b467e141755fb91b7202221fbd8293efbdf37ad7e' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/train-02b/inputs.json' | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/train-02b:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/train-02b/isaaclab-training'
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/train-02b/inputs.json'
