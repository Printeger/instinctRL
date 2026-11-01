#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev
printf '%s  %s\n' 'faf5ecf3b901b0d6eaec114215e159d4d4fd68549bb26ed600a1fef3e3690bfc' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-02/plan.json' | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-02:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-02/isaaclab-training'
exec /usr/bin/python3 -B '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/probability-cuda-02/tools/r6_direct/cuda_diagnostic.py' --plan '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-02/plan.json'
