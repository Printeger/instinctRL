"""Frozen semantic identities and dimensions for EVAL-001."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


COORDINATION_ROOT = Path("/home/mint/rl_dev/instinctRL_coordination")
PRODUCT_ROOT = Path("/home/mint/rl_dev/worktrees/instinctRL-blackwell")
PRODUCT_COMMIT = "e438c4ccef72b088c95db6f01b049f74a1921180"
PRODUCT_TREE = "75173a1c42af0196162de3a02dacca55c5e7c365"
CHECKPOINT_PATH = COORDINATION_ROOT / "attempts/development/SCALE-001/SCALE-001-D0002/formal/checkpoints/update_0250.pt"
CHECKPOINT_SHA256 = "d58450a99e43cb2178f49840ed91787714c42042f78cfeafddd9df5f553307b3"
ACTOR_SHA256 = "2e84ef16a2acb82515fa768db94b37f12961e70598738ddfe3cfe5466fccccad"
INTERPRETER = Path("/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11")
INTERPRETER_SHA256 = "efa5468dc4ff3850e950cb68c0e430b3a1621e2c6bd9e0ef48f784ba90a9667a"

NUM_ENVS = 32
POLICY_STEPS = 256
DECIMATION = 20
PHYSICS_STEPS_PER_ENVIRONMENT = 5120
ENVIRONMENT_TRANSITIONS = 8192
RAYS_PER_ENVIRONMENT_PER_STEP = 21240
RAY_SAMPLES = 173998080
ENVIRONMENT_SEED = 20260901
INFERENCE_SEED = 20260901
ACTION_MODE = "deterministic_mean"

RUN_IDS = ("mean_cert_a", "mean_cert_b")
SOURCE_SHA256 = OrderedDict((
    ("isaaclab-training/instinctlab/evaluation/runtime.py", "7abaa594a63b0a90d9a0bb5b1ec66a9f71d3c91aa464f2890001a2040384cde3"),
    ("isaaclab-training/instinctlab/evaluation/metrics.py", "a37cf88a04d55ac4e32fee93f5504a6ff446f8804694a310013fb704438a3960"),
    ("isaaclab-training/instinctlab/evaluation/audit.py", "03a6acb86c1aed86385b7ff3d265940791656ee8e38de07f242ac3128b35fd0e"),
    ("isaaclab-training/instinctlab/evaluation/raycaster_observability.py", "a98098e8290726f9abda1e74f613ae4b9c91859ee39395eb7f036ed47dca1989"),
    ("isaaclab-training/instinctlab/envs/taslab_direct_env.py", "e4099b8085e2fe1cf69aac6d249e86d14f0d06ad6dfedc400f717eb14af161b7"),
    ("isaaclab-training/instinctlab/training/networks.py", "a0790d5f80c8a9f2f06d05fc80bcdcfe8fd3e1a5c71a482443f1b3f5769dc129"),
    ("isaaclab-training/instinctlab/training/ppo.py", "796f8bcb2309cea09fd229784f6b8c6b154b01be69ec5605b793952db89ec189"),
    ("isaaclab-training/instinctlab/training/checkpoint.py", "d04f77d3e443ed8466db5dc6f0c05d633fd012b799eedf4d6047d4e395e9b37b"),
    ("isaaclab-training/instinctlab/training/rollout.py", "a1d04a84a6099a711cab1d8cb8b6a550addf292dde4682ea82d3a9a0d32ab58d"),
))

TOP_LEVEL_KEYS = (
    "schema_version", "scope", "identity", "campaign", "checkpoint_state_before",
    "checkpoint_state_after", "gradients", "training_claims", "actor_boundary",
    "action_diagnostics", "alpha_diagnostics", "beta_diagnostics", "entropy_diagnostics",
    "log_probability_diagnostics", "reward_diagnostics", "termination_reason_counts",
    "reset_count", "cuda_resources", "raycaster_provenance", "runtime_evaluation",
)
