# EVAL-002 workload

Only fixed-checkpoint inference in the ordered combined scene. No training or performance gate. All privilege is evaluation-only.

## Adaptation provenance

[
  {
    "source": "evaluation_workloads/eval_001/constants.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "300220098c49b9378297b2d5d0d28dbea43a63ab1459ae69be73d3da3b1de3de"
  },
  {
    "source": "evaluation_workloads/eval_001/attempt_binding.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "54dfd61b0444c8b77c2328002ea19e74c8fb54663911d84418fd612a75e5ab96"
  },
  {
    "source": "evaluation_workloads/eval_001/io_utils.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "45ef56ce17ad583c301eceff0e47f0921d7aeb0eaa1eb6a441001da5a46945b5"
  },
  {
    "source": "evaluation_workloads/eval_001/runtime_support.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "84863fa0617f1b5b5f074714f759f56f7d91fbd197241245133e8591735a286c"
  },
  {
    "source": "evaluation_workloads/eval_001/checkpoint_audit.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "1aaa8970975bd195a20705c3191396237423f200e9575ff70d5dee19f026d642"
  },
  {
    "source": "evaluation_workloads/eval_001/gpu_gate.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "41ee81315976b280108c056b2514a6b0b87fa73171f1eddf8fdf7923c2397e77"
  },
  {
    "source": "evaluation_workloads/eval_001/formal_evaluator.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "040a9650a2f00a895064471f428f459c2f3caf8b92f32b2076cbd3c6dd819a6a"
  },
  {
    "source": "evaluation_workloads/eval_001/semantic_validator.py",
    "commit": "16751691bad95a223c20e72eed4abd034ceb9fdc",
    "sha256": "fcbb078906403e42498f0837e92a0734e7075d5f7a33f6c602ef19abfb40b056"
  }
]

Source files remain immutable. Binding, case identities, dimensions and fixture-specific assertions are adapted for EVAL-002; serialization, accepted checkpoint audit and resource thresholds are retained. Product geometry comes from scenario_corner.py and taslab_direct_env.py at e438c4ccef72b088c95db6f01b049f74a1921180, with hashes pinned in the attempt.

## Product geometry provenance

[
  {
    "path": "isaaclab-training/instinctlab/envs/taslab_direct_env.py",
    "commit": "e438c4ccef72b088c95db6f01b049f74a1921180",
    "sha256": "e4099b8085e2fe1cf69aac6d249e86d14f0d06ad6dfedc400f717eb14af161b7"
  },
  {
    "path": "isaaclab-training/instinctlab/evaluation/scenario_corner.py",
    "commit": "e438c4ccef72b088c95db6f01b049f74a1921180",
    "sha256": "0bda06f61c40a02d3cb77aa89e4035648005f4c56f49ce78948a03cf76bf3c16"
  }
]
