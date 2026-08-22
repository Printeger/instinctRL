# M6-002 Environment Evidence Index

## Root

`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M6-002/M6-002-D0001`

| Evidence | SHA256 / fact |
| --- | --- |
| `COMMAND_LEDGER.json` | `066d3e8334d55e22b17aa364786e6dace47e3a6c75ac9a99bb53f8dd2af52234` |
| `OUTPUT_MANIFEST.json` | `6e113f086e36e3456636bd47871a47ef06d72937826117d3110fa251f8301c1e` |
| `VALIDATION_SUMMARY.json` | 35/35 PASS |
| `derived/environment_smoke_1.pt` | `0a53f429e28d0726611ef6616b12535ee8414bd8a559b993e0efd541ca676303` |
| `derived/environment_smoke_2.pt` | `61cad0c0c97d05b9b62be414466b79bc7a4e5e300098017319f39ceb6d565845` |
| `derived/environment_compare.json` | `55f647e899d6c44be488f8428f46a0ef765905a7b257236b2acdd5d3a32168b8` |
| comparator result | PASS; 1,800 tensors; 16,834,412 elements; maximum error `0.0` |

`raw_logs/` contains one stdout, stderr, and exit file per command.
`derived/mutations/` retains all five malformed comparator inputs; expected
nonzero exits and tracebacks are preserved. `generate_comparator_mutations.py`
and `run_formal_validation.py` were persisted before formal execution.

## Runtime and external capture

Driver is `580.173.02`; GPU is NVIDIA GeForce RTX 5070. Python is `3.11.15`,
Isaac Sim is `5.1.0`, TorchRL and TensorDict are `0.10.0`. Frozen Isaac Lab HEAD
is exact `37ddf626871758333d6ed89cf64ad702aef127d0`; its pre-existing tracked
`apps/isaaclab.python.headless.kit` modification remains present and unchanged.
Formal source HEAD is `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3` and clean. OmniDrones and NavRL
remain at `5045c71cc73d224144cc08bc38d65cabeedaec8c` and
`3725bcc2e7c1be4ecf1455d922299ae85042603a`, both clean.
