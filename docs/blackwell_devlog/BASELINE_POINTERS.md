# Blackwell baseline pointers

Status: frozen governance input

Baseline name: `BLACKWELL_BASELINE_V2_R580`

Bootstrap source reviewed: `main@6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`

## Identity and authority

| Layer | Frozen pointer | Role and authority |
| --- | --- | --- |
| Formal source | `Printeger/instinctRL`, `main@6f6dee3da72d0b0202846b330adcaeeec8ee7bc3` | Authoritative legacy source and the only branch point for this migration. At bootstrap it equals `origin/main` with ahead/behind `0/0`. |
| Blackwell platform baseline | `BLACKWELL_BASELINE_V2_R580` | Authoritative qualified environment envelope. Its recorded Isaac Sim/Isaac Lab, NVIDIA R580 driver, CUDA/PyTorch and host facts must be consumed read-only. |
| Qualified modern OmniDrones reference | commit `5045c71` | Authoritative modern API comparison snapshot. The short identifier comes from completed qualification evidence; this repository does not contain the object. Do not invent a full SHA or repository URL—resolve both from that evidence before runtime use. |
| Qualified NavRL reference | `Zhefan-Xu/NavRL@3725bcc2e7c1be4ecf1455d922299ae85042603a` | Authoritative frozen NavRL comparison snapshot. |
| Legacy TASLAB source | this repository at `6f6dee3…`; introduced by `c4cca8b`, asset/parameter update `8c9f80a` | Authoritative source for what this project currently contains, not evidence of upstream official support. |

All pointers are immutable inputs. If a pointer cannot be resolved exactly, stop and record the discrepancy; do not substitute a newer branch tip, tag, package release, cached checkout, or similarly named asset.

## TASLAB_UAV provenance warning

The qualified modern OmniDrones tree does **not** contain TASLAB_UAV. The legacy TASLAB implementation was derived from an AechPro Isaac Sim 5.1 OmniDrones fork and then changed by local feasibility commits (`c4cca8b`, `8c9f80a`). It must be described as **third-party fork-derived plus local patches**, never as an official modern OmniDrones implementation.

The current formal-source TASLAB USD is Git blob `b16b1b16f21166d82579a9a5b190cb49191ddef5` at `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd`; the reviewed worktree file size is 9,563,347 bytes. Historical documentation describing it as approximately 158 KB is stale. M1A must recompute the Git blob, SHA-256, byte size, and file type from the frozen source and retain the raw command evidence.

## Source precedence

For migration decisions, apply this precedence:

1. Exact Git object/file evidence and the qualified baseline record.
2. Executable source and asset metadata at the pinned commit.
3. Tests and static consumers at the pinned commit.
4. Maintainer documentation at the same pin.
5. Historical project docs, READMEs, comments, and unpinned external descriptions.

Conflicts must be recorded explicitly. YAML values do not automatically override USD-authored mass, inertia, joints, transforms, or rotor prim topology; nor does the USD automatically define controller gains or semantic intent. The runtime consumer decides which source is effective, and M1A must trace it without executing the runtime.

## Read-only rule

Do not modify, reformat, vendor, copy, convert, or regenerate any frozen baseline or TASLAB asset during M1A. Baseline refresh is a separate Commander decision and requires a new baseline identity; it cannot occur incidentally inside a milestone.
