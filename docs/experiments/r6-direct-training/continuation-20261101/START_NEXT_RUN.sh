#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
printf '%s  %s\n' 'bba2289d1c2fab729854b1a57624eca6caee2e434c880d44e47d219f3b723a10' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-04/plan.json' | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-04:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-04/isaaclab-training'
exec /usr/bin/python3 -B '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-04/tools/r6_direct/cuda_diagnostic.py' --plan '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-04/plan.json'
