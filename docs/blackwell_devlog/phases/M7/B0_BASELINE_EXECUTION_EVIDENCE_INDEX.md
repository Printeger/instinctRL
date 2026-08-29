# M7-008 Revision 1 Import-Boundary Rework Evidence Index

## Fresh namespace

`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-008/M7-008-D0002/`

## Accepted candidate

`formal_candidates/0001/`

- `SUMMARY.json` SHA-256:
  `6c026403fd2f00d42e761d2dea2d8d7fbe905a54f6b6817b77524c8b3e5db7a6`
- `COMMAND_LEDGER.json` SHA-256:
  `325439293b0ef32150129b8f4ea88f96f9d45f320a09295127312e6937ec8e07`
- `REQUIREMENT_MAP.json` SHA-256:
  `50c0ca8167707252143e6f8557bb2140ea5e0d9c1f5cd14ad5141b93cc82a2e2`
- Functional commit:
  `666aafc1e3f2fa6c153d7c9ea2cda43e7f87de81`

The candidate contains 50-command raw stdout/stderr, exact argv/environment/
UTC/exit/hash ledger entries, canonical fixtures, 17 negative manifests, two
independent process records, source/frozen integrity, full rework and original-
base diffs, and the requirement map.

## D0001 immutability

`D0001_SNAPSHOT.json` records SHA-256 for all 351 D0001 evidence files and has
SHA-256 `0c7b3e00c99b760d1cdf4f90dec85ed2af7afa3b95628a4f207a89e21dd40d4f`.
The formal candidate rechecked the snapshot exactly. D0001 programs,
candidates, final audit, and reports were not changed.

## Persisted D0002 programs

The attempt root contains the import probe, D0001 snapshot verifier, manifest/
process generators and comparator, 17-case validator matrix, source/integrity
auditors, formal runner, and final-audit runner. Candidate `SUMMARY.json` binds
every program and artifact hash. No formal program used stdin, heredoc, here-
string, process substitution, interpreter/shell `-c`, or ephemeral code.
