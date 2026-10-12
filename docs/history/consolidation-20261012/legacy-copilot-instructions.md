# Copilot instructions: Blackwell migration

Follow `AGENTS.md` and the one active Commander order. The order is an allowlist, not a suggestion.

- Work only on `blackwell-migration` in `/home/mint/rl_dev/worktrees/instinctRL-blackwell`; keep `/home/mint/rl_dev/instinctRL` read-only and unchanged.
- The Commander is the only order writer/approver. The Executor is the only implementation writer and cannot approve its own work.
- Inspect frozen baselines and legacy paths read-only. Keep legacy code in `isaac-training/`, `ros1/`, and `ros2/`; reserve `isaaclab-training/` for later explicitly ordered new-runtime work.
- Follow M1 through M8 sequentially. Never begin a successor milestone before written acceptance of its predecessor.
- Preserve the hard actor contract: no pose, odometry, explicit/reconstructed velocity estimate, map, SLAM, obstacle ground truth, goal-relative privileged state, or simulator-private state in actor input. Privileged state is limited to critic/reward/evaluation/logging/labelled upper bounds; low-level controller stabilization is outside the learned actor.
- Stop if the branch, base SHA, clean status, allowed paths, command allowlist, or required commit count differs from the active order.
- Do not guess missing provenance, units, frames, prim/rotor topology, licenses, or API behavior. Record uncertainty as a risk.
- Do not modify Git configuration, merge, push, patch frozen runtime, or run unapproved simulation/training. In particular, no 128k or 1M training is authorized by bootstrap or M1A.
- Deliver the exact commit count plus base-to-HEAD diff, test/check evidence, raw command-log index, change ledger, and known limitations required by the order.
