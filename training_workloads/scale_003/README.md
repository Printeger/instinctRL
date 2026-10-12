# SCALE-003 workload candidate

This directory implements the source-only candidate authorized by SCALE-003 revision 0. It does not authorize a simulator, CUDA, training, evaluation, or checkpoint-writing process. Both direct entries validate a separate canonical Commander live-execution authorization before importing their live modules.

The proposed future training process resumes the exact approved update-250 checkpoint and collects 872,000 additional transitions in 1,703 horizon-16 rollouts plus one horizon-2 tail. It performs 1,704 PPO updates and writes the reviewed 70-checkpoint schedule. The command clock is process-global and survives episode resets; command selection occurs only while constructing the observation returned for the next action.

The training scene derives all translations from the live 32-row `scene.env_origins`. One explicit non-colliding combined mesh contains a local floor and two inward-facing walls per environment. Collision remains one inherited global ground plane plus 64 wall-only collider prims. Before training, the live path reads back USD and Warp geometry and requires actual raw sensor hits on all three surfaces, exact normal-Jacobian numerical rank three, and a finite positive smallest singular value in every environment. The local fixture has a documented coverage limit after robots drift outside its floor/wall extent.

The proposed future evaluation uses two separate simulator processes, `station` and `tracking`, with the frozen EVAL-002 fixture and product runtime collector. Each process takes 256 policy steps across two environments from the exact final update-1954 checkpoint, uses actor mean actions, preserves terminal samples, and applies no performance threshold.

Every live process must first pass the frozen EVAL-002 coexistence gate before `AppLauncher`, environment construction, or CUDA initialization: three samples over at least ten seconds, the exact GPU/driver identity, utilization at most 5%, used memory at most 1024 MiB, free memory at least 10240 MiB, and only the approved desktop compute process. Training then records a fresh free-memory check before each rollout and update; evaluation does so before every step. A process stops before starting more work when free CUDA memory is below 1 GiB.

Future attempt manifests must be created through `build_attempt_documents`. Training is one immutable attempt that binds the update-250 input. After its PASS semantic result, final checkpoint, and terminal manifest have been sealed, `build_training_handoff_document` creates a checked handoff for a distinct evaluation attempt. That evaluation attempt pins the sealed training terminal manifest, semantic/status files, and final update-1954 checkpoint path, hash, size, and counter. It never edits or predicts the training manifest or its input bytes. Both stages use the pinned regular Python executable and direct absolute source paths. Attempt and checkpoint paths are supplied through pinned environment values so the existing tooling does not mistake data arguments for attempt-local executable sources. Current validation commands are inert:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B \
  training_workloads/scale_003/formal_scale_training.py --validate-only
```
