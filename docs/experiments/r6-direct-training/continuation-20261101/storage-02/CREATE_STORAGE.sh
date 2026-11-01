#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\n' '4a4489bf36fec2b6207a7d0fc96fc4c06013a1c226268f9d1270a899e527e5eb' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/storage-02/provision_r6_native_volume.py' | sha256sum --check --status
printf '%s  %s\n' '221b3db6f38690e01f46868f845ec36b5519030be43474815b3554ff3010c0cd' '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/storage-02/provision_r6_result_volume.py' | sha256sum --check --status
sudo /usr/bin/python3 '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/storage-02/provision_r6_native_volume.py' 'J-60c22d4806aa6798' --apply
sudo /usr/bin/python3 '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/storage-02/provision_r6_result_volume.py' 'J-60c22d4806aa6798' '/home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONTINUATION-02/results' '34359738368' --apply
findmnt --mountpoint '/var/lib/instinctrl-r6-native/J-60c22d4806aa6798/output'
findmnt --mountpoint '/home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-DIRECT-CONTINUATION-02/results'
