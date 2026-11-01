#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
printf '%s  %s\n' '2e681c4a909d38ee0667906c3b73b84934ea27cd7a8bce2bac32371531f4a000' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-03/plan.json' | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-03:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-03/isaaclab-training'
exec /usr/bin/python3 -B '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-03/tools/r6_direct/cuda_diagnostic.py' --plan '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-03/plan.json'
