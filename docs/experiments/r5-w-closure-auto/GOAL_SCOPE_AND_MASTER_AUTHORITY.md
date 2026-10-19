# R5 W safety closure — new incremental authorization

The 2026-10-19 user prompt authorizes the full diagnosis → implementation → training → original five-task evaluation → independent review loop, including delegated exact internal approvals. The exact prompt is preserved in USER_GOAL_PROMPT.txt; MASTER_AUTHORITY.json binds it and the complete requirement document by SHA256. This task is not complete when its diagnosis or plan is written.

New hard caps: 1,048,576 training transitions, 659,200 nontraining transitions, 2,048 logical updates, 24 starts and initializations, 40 checkpoint writes, 86,400 GPU seconds and wall seconds each including cleanup, and 128 GiB output plus conservative reservations. Keep ≥30 GiB free. At most two scientific candidate branches, one GPU job, no blind retries. Derived actor/critic step caps are 16,384 each at the unchanged eight minibatches per update.

Root acts as delegated Commander and sole workspace writer. Independent read-only reviewers examine exact methods, code, packets, actual evidence and final conclusions. A new master acceptance and stage allocation are appended to the existing Store; the old Goal's master and budget are never edited or reused as new authority. The new master and E1 stage are now registered in the existing append-only Store (registration receipt: registrations/W-E1-001.json). No new job has been issued.

Current preflight: local and remote main both f4f759f4c38d0fd1e40d03e46392b05c5415cb71; old batch COMPLETE; no external STOP, Store not paused; all 52 historical jobs pass the existing process-identity/quiescence check. Original FAILED/INCOMPLETE/null and reservations remain intact. Untracked docs/goal belongs to the user and is left untouched.

Stops are external STOP, ownership/evidence anomalies, hard risk, inability to establish legal authority or exhausted total resources. Ordinary implementation defects or negative performance lead to bounded repair or a preregistered single-variable next candidate. Final success requires all original gates plus independent confirmation/review; otherwise deliver a rigorously supported final negative result, never a fabricated pass.
