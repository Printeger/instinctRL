# M7-011 Exact B0 TASLAB Environment Evidence Index

## Accepted candidate

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-011/M7-011-D0003`

Accepted formal candidate: `formal_candidates/0001`.

| Evidence | Path | SHA-256 / fact |
| --- | --- | --- |
| formal summary | `formal_candidates/0001/SUMMARY.json` | `28c831d3bc564c62f0a26440064a5682178c2f12072b8497117f0ec4ab08c6e6`; PASS; 1610 passed |
| command ledger | `formal_candidates/0001/COMMAND_LEDGER.json` | `781601b2537c156d903ff9015b16071bbf9b7c37a42f4f6f5700a205aa88c9b3`; 41 exact command records |
| smoke validation | `formal_candidates/0001/SMOKE_VALIDATION.json` | `c53e5c985e6b9a4fd18037b3f7f0b4f26360ac4621ef6fee2f47c807cfbfd880` |
| environment process 1 | `formal_candidates/0001/smokes/environment_process_1.pt` | `9b42084407e0cdf42b816ebcc111dac26b0be54a8e720501b1f7bd136bc9e6c0` |
| environment process 2 | `formal_candidates/0001/smokes/environment_process_2.pt` | `a39ca694eb37269a3579db53bc463a484e42da116bf2bd2c9db27ddca47ca7b7` |
| exact comparator | `formal_candidates/0001/smokes/environment_exact_comparison.json` | `51de74373d9efcb5e00b4166232a2e62d872213da9ec5ed28f31fd87337d34e7` |
| comparator mutations | `formal_candidates/0001/COMPARATOR_MUTATIONS.json` | `c08788625e617993c13db5661787cff6ebdf6065eb785188cab7347834e4e7cd` |
| M6 CUDA regression | `formal_candidates/0001/smokes/m6_environment.pt` | `d95fcb6302bffc2405f5e4292907167b98a49d60e5a31e0b3d08c2dd45ad5428` |
| M7-009 CUDA regression | `formal_candidates/0001/smokes/m7_controller.pt` | `44fa2ff0775e80f3ce090ca57c1b99278886975719b31a94b673fdd999a8f094` |
| prior-attempt hash manifest | `PRIOR_ATTEMPT_HASHES.json` | `05457de86e8e7282f7163f5343ed438df99e4652f9de2bb87eced739ffb6202a`; 228 files |
| raw stdout/stderr | `formal_candidates/0001/raw_logs/` | indexed by command ledger |

## Bound product sources

- Environment source SHA-256:
  `b1ecdaa8b7d01ff0f9dd49891c2266908cea5a2540e8a91c33b9626464ded584`.
- Repaired smoke runner SHA-256:
  `192d527b3bdb4e9f3e532bf0303f2de55f18ab175ab505333cfe66cace347d08`.
- Unchanged exact comparator SHA-256:
  `1d5ab4ba37770b895968b53506efff2d71117b6c8460b88b13ad569220864158`.
- READY manifest SHA-256:
  `fad781d3f9151c98d67c42be261374b6b5de4c575eb7604372c5794b7e8608c6`.
- EXECUTED manifest SHA-256:
  `82d65b60ffb3510d77ba73739237d072e83530bd50ba0ad576b99a7646888d3b`.

## Runtime, determinism and privacy evidence

- Each independent CUDA process records exactly 8 policy, 160 physics, 160
  lifecycle and 160 wrench calls across 320 samples.
- Exact comparison covers 66 tensors and 45,496 elements with trajectory
  digest `97ba25dd50bfdaf89e6f5da1dfb58f23972f98dee7cfa9e7c146bfb0bfc65e96`.
- Body/world command diagnostics and each of the five accepted Lee outputs
  have mismatch count zero and maximum error zero in both process bundles.
- All seven missing/extra/order/bit/count/receipt/private mutations are
  rejected by the unchanged comparator.
- Frozen integrity binds Python 3.11.15, Torch 2.7.0+cu128, RTX 5070, Isaac
  Lab, OmniDrones `5045c71c...`, and NavRL `3725bcc2...`.

D0001 and D0002 evidence remains immutable. D0002 candidate `0001` is failed
evidence solely because of the redundant root bundle key; D0003 candidate
`0001` is the sole acceptance candidate.
