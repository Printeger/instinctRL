#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
printf '%s  %s\n' '45cce275b3f1ec322abc165fcf1a913c89d9dd68b5aa757d428b68b967d9f4a8' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-01/plan.json' | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-01:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-01/isaaclab-training'
exec /usr/bin/python3 -B '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-01/tools/r6_direct/cuda_diagnostic.py' --plan '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-01/plan.json'
