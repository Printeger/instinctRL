#!/usr/bin/env bash
set -euo pipefail
cd /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-02
printf '%s  %s
' d026982f9f651debd52a3c8aa63794dadf56c2c389486b7eddeb26224cf787d4 /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/paired-02/u640-inputs.json | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-02:/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/snapshots/paired-02/isaaclab-training
exec /usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/paired-02/u640-inputs.json
