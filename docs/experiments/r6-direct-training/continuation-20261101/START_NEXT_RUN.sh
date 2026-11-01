#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-02
printf '%s  %s
' be4aa9b06ff2c8966aa3518e12d7c15b38e6c62e1aea5d72efa1012e865ad3f7 /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/paired-02/u768-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-02:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-02/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/paired-02/u768-inputs.json
