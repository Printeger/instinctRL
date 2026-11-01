#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
printf '%s  %s
' 8331b4c6a75aa221f0f0138d390ae31d851fdbe870809237f628f64e05026dce /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/paired-01/u640-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-01:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-01/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/paired-01/u640-inputs.json
