# Stable prompt-root release qualification

Status: **Production-ready. Exact-root, lifecycle, refill, workflow-performance, and full-corpus
gates pass.**

Revision `16fd0e68` completes the stable stage-root, shared cache-root, disk-retention,
sessionless constrained-root, and bounded workflow-efficiency work described by the
[stable prompt roots and FlashMoe refill plan](../stable-prompt-cache-and-refill-plan.md). The
production-performance and final-corpus records are
`../../fixtures/harness-usability/baselines/2026-07-26-stable-prompt-root-performance-promotion.json`
and
`../../fixtures/harness-usability/baselines/2026-07-26-stable-prompt-root-production-ready.json`.

The earlier `bb3e89f9` cache candidate and its unfavorable promotion results remain below as design
history. The final locked paired comparison passes correctness, call-shape, exact-root,
fresh-prefill, wall-time, energy, and per-case regression gates. The current 24-case qualification
has 21 independently verified clean completions and three truthful bounded limits, with no false
completion, forbidden mutation, cache correctness failure, or unbounded workflow regression.

## Initial cache candidate and protocol

| Field | Value |
| --- | --- |
| Code revision | `bb3e89f9068f43fad6dc81de742d3bf953032635` |
| Release binary SHA-256 | `b10b116f7db5488af6b206d25535d4a08706eae11be30c3bcafe50a31c8ae2d0` |
| Model | `hf://mlx-community/Qwen3-Coder-Next-4bit` |
| Backend | `flashmoe-v2-mlxq4` |
| Sampling | temperature 0, top-k 1, seed 0 |
| Corpus scratch root | `/private/tmp/pb-prompt-root-phase5-production-bb3e89f9-20260725` |
| Refill profile root | `/private/tmp/pb-flashmoe-refill-6aae3c8d-20260725` |

All 24 corpus cases started from independently prepared Git workspaces. The release binary was
invoked directly and serially so Metal visibility, shared disk-root retention, process teardown,
and energy attribution matched the retained baseline protocol. The independent auditor reran every
official check, compared immutable fixture files, derived the actual Git delta, checked the recorded
commit and worktree, and then read pb's event stream. It did not trust pb's completion verdict.

The required raw native smoke exited zero and returned `5` for `2+2=` at one token. This continues
to be a model/runtime-quality warning, not a correctness claim about the answer.

## Initial full 24-case result

| Result | Value |
| --- | ---: |
| Official task passes | 21/24 |
| pb verified-clean completions | 21/24 |
| False verified completions | 0 |
| Positive evidence | 21 |
| Bounded model/control limits | 3 |
| pb defects / experiment errors | 0 / 0 |
| Model invocations | 127 |
| Rendered prompt tokens | 586,022 |
| Cached prefix tokens | 294,965 |
| Fresh-prefill tokens | 291,057 |
| Eligible stable-root tokens | 268,147 |
| Reused stable-root tokens | 268,147 (100%) |
| Root-hit invocations | 127/127 |
| Cache misses / reconciliation failures | 0 / 0 |
| Durable checkpoints | 24 queued, 24 completed, 0 failed |
| Wall time | 8,361,898 ms |
| Measured energy | 70.60 Wh, complete coverage |

The three unsuccessful cases were `rust_bounded_backoff`, `python_half_open_window`, and
`react_tabs_semantics`. Their official checks failed, pb ended `incomplete` with an unsatisfied
contract, and the auditor found no false verification or forbidden path. They are local-model or
bounded-control limits, not cache or completion-safety defects.

All 127 invocations used the qualified Qwen layer-major matrix command because every actual fresh
suffix met the prepared 32-token threshold. Refill counters reconciled and persistence completed
without failure. The authority distribution covered planning, plan review, implementation mutation,
repair mutation, code review, and the constrained Task artifact root.

## Exact stage roots and disk retention

`rust_recent_cache_capacity` ran late enough to reproduce the disk-pressure conditions that exposed
the pre-fix mtime bug. All four roots restored completely from disk:

| Authority | Eligible | Reused | Source |
| --- | ---: | ---: | --- |
| Planning | 3,895 | 3,895 | disk prefix |
| Plan review | 1,756 | 1,756 | disk prefix |
| Implementation mutation | 1,437 | 1,437 | disk prefix |
| Code review | 1,632 | 1,632 | disk prefix |
| **Total** | **8,720** | **8,720** | — |

`rust_quoted_csv`, the earlier live regression point, also restored 8,720/8,720 root tokens over
four calls with no miss. These results agree with the deterministic access-recency test: a
successfully validated checkpoint restore refreshes its on-disk LRU age, so a hot older checkpoint
survives a newer unused checkpoint under pruning. Recency refresh is best effort and uses the
already validated, no-follow file descriptor; inability to refresh a timestamp never converts a
safe restore into a request failure.

The deterministic cache suite also covers cold/diverged/disabled miss classification, exact
disk-root fallthrough after a session mismatch, full MLA and recurrent-state round trips, model
fingerprint rejection, concurrent writers, byte-budget LRU eviction, dangling-manifest cleanup,
partial/corrupt checkpoints, oversized checkpoints, changed storage roots, unwritable namespaces,
and checkpoint or configured-root symlink escapes.

## Managed cache lifecycle qualification

Revision `d3e389c3615f9387d74a6596c256444b87ed7dee` adds one managed evaluator for the
complete Phase 4 lifecycle rather than relying on separately assembled happy-path runs. Its
machine-readable result is
`../../fixtures/harness-usability/baselines/2026-07-26-prompt-cache-lifecycle.json`; the retained
native scratch root is `/private/tmp/pb-cache-eval-d3e389c3-native-20260726`. The source worktree
was clean and the arm64 release binary SHA-256 was
`e5dd3e92c553ce0bcb61be12aa4432a057354164067c4642b39aa09c664d23cf`.

The evaluator requires six distinct, initially absent scratch directories, an empty or absent
cache directory, matching immutable contracts and workspace content, and a new report path. It
runs cold, warm same-session, warm new-session, changed-authority, matching-authority, and child
process restart arms. It releases every unleased parent FlashMoe runtime before spawning the child,
so the restart arm cannot obtain an in-process prefix. Task outcome is recorded independently from
the cache gates.

| Arm | Planning root | Reused | Source | Calls | Independent result |
| --- | ---: | ---: | --- | ---: | --- |
| Cold empty storage | 3,703 | 0 | none (`cold_session`) | 4 | verified clean |
| Warm same session | 3,703 | 3,703 | memory prefix | 4 | verified clean |
| Warm new session | 3,703 | 3,703 | memory prefix | 4 | verified clean |
| Changed authority | 3,516 | 0 | none (`cold_session`) | 4 | verified clean |
| Original authority restored | 3,703 | 3,703 | memory prefix | 4 | verified clean |
| New child process | 3,703 | 3,703 | disk prefix | 4 | verified clean |

The changed arm subtracts `read_file` after normal typed Planning capabilities are derived. Its
tool-schema and rendered-token digests both differ from the original, and it gets zero reuse. The
terminal Planning action cannot be excluded. Restoring the original authority returns the exact
original digests and full reuse; the child then obtains the same exact root from disk. All seven
evaluator gates passed.

The independent usability auditor reran official checks for every arm: 6/6 official passes, 6/6
verified-clean completions, 0 false verified completions, 24 total model calls, and no forbidden
path or semantic-commit failure. It classified all six as positive evidence. Across the matrix,
13/13 queued checkpoints completed, no persistence or reconciliation failure occurred, and the
restart arm reported disk read/decode separately from state hydration and suffix prefill.

Two earlier runs are excluded. `/private/tmp/pb-cache-eval-69f4d738-20260726` could not see a Metal
device inside the command sandbox, while the same release smoke succeeded on the host. The first
native run at `/private/tmp/pb-cache-eval-69f4d738-native2-20260726` exposed an evaluator defect:
strict stage capability derivation replaced the initial direct allowlist. Revision `d3e389c3`
replaced that ineffective probe with a non-serializable, harness-only, subtractive exclusion applied
after capability derivation and protected the typed terminal action. Neither excluded run
contributes to the passing claim.

This closes the durable lifecycle-evidence gap. At that point it did not supersede the locked
three-language performance result below or change the production-ready designation.

## Sessionless constrained root

The final uncovered hole was the constrained Task-partitioning request. Its stable 30-token
`TaskArtifact` prefix was diagnosed and captured, but the cache lifecycle returned before committing
the root when no logical session ID existed. Revision `bb3e89f9` makes base-root commit independent
of logical session continuation state.

A cold process now reports `cold_session` with `exact_root_checkpoint_missing`, not
`cache_disabled`, queues and completes persistence, and remains verified-clean. A new process then
restores 30/30 Task-root tokens through `disk_prefix` with no miss. The complete restart run reused
11,438/11,438 eligible root tokens. The final long corpus later restored the same 30/30 Task root
after many intervening processes, which demonstrates retention rather than an immediately warm
special case.

Only the immutable rendered base KV/recurrent state is shared. The JSON matcher, dynamic task
evidence, generated decomposition, and logical continuation remain freshly bound to the request.

## FlashMoe refill decision

The retained refill matrix separated lookup, disk decode, CPU validation/allocation, Metal
hydration, suffix prefill, snapshot capture, and persistence queue time. It covered zero-prefix and
restored 10+30-token suffixes under resident and forced-streamed expert policy, automatic 31/32/33
selection, memory/restart zero suffix, and a restartable one-token suffix.

- Resident 40-token scalar and layer-major state matched exactly; scalar took 2,805 ms and
  layer-major 923 ms.
- Resident and streamed zero/restored-prefix continuations matched their declared hidden, KV,
  router, recurrent-state, and greedy-output parity.
- Automatic selection used scalar for 31 tokens and layer-major for 32 and 33 tokens.
- Zero-suffix memory and restart restores issued no prefill command.
- A 32+1 restart processed only the one-token suffix through scalar.
- All resource summaries ended balanced with no active, transient, or in-flight command leak.

Fresh suffix prefill dominated. The existing production graph already evaluates the actual
remaining suffix and selects the qualified layer-major path at its threshold, so the profile did not
justify another short-suffix command, expert cache, Q4-only path, or alternate runtime. No expert
I/O, router, Metal scheduling, quantization, or CPU/GPU handoff behavior changed for this work.

An explicit llama.cpp CPU control loaded and generated with
`Qwen_Qwen2.5-Coder-7B-Instruct-GGUF` in two processes. It proves renderer and cache-path
compatibility, not cross-session root sharing: llama.cpp persistence remains keyed by logical
session ID, while FlashMoe owns the shared content-addressed prompt-root tier.

## Locked three-language promotion gate

The exact Rust registry, Python TTL, and React alert cases were compared with revision `a69239ce`.

| Case | Calls | Eligible / reused root | Fresh prefill | Wall | Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust registry | 4 | 8,720 / 8,720 | 9,797 | 241,396 ms | 1.96 Wh |
| Python TTL | 5 | 10,064 / 10,064 | 12,670 | 352,358 ms | 3.14 Wh |
| React alert | 5 | 10,064 / 10,064 | 12,639 | 321,608 ms | 2.45 Wh |
| **Candidate total** | **14** | **28,848 / 28,848** | **35,106** | **915,362 ms** | **7.55 Wh** |
| Baseline total | 14 | not available | 47,271 | 951,852 ms | 9.97 Wh |

Fresh prefill fell 25.73%, narrowly passing the 25% target of at most 35,453 tokens. Energy fell
24.22%, passing its 15% target. Official correctness remained 3/3 with zero false verification, and
unchanged eligible roots were reused completely.

The strict promotion gate nevertheless fails:

- wall time improved only 3.83%, below 15%;
- Python used five calls rather than the required four-call successful floor;
- Rust wall time regressed 14.48% and Python 26.54%, both above the per-case 10% allowance; and
- no locked machine-level explanation proves those regressions are external noise.

The extra Python call and case-specific decode trajectory show why root reuse and total agent
performance must remain separate. This candidate's cache work is correct and materially reduces
prefill, but it could not yet be called production-ready under the plan's end-to-end gate.

## Python control follow-up

Revision `884d4c1b7029cba9c196839b54493f730b09bc6b` addresses the repeatable Python
five-call cause without changing cache identity or widening tool authority. For an initial mutation
of a fully observed small `.py` or `.pyi` file, the controller retains bounded atomic replacement
and removes partial editing from that one request. The tool description explicitly requires all
unrelated observed bytes to remain intact. Large or partially observed Python files, diagnostic
repair, Rust, and React/TypeScript keep their existing tools. Focused tests cover each negative
boundary.

The clean arm64 release binary had SHA-256
`6261d9b2dd4db7c1876183ea1f04778137f8c4222f0006145b438cb600b33981`. The retained
three-language scratch root is `/private/tmp/pb-locked-promotion-884d4c1b-20260726`, and the
machine-readable result is
`../../fixtures/harness-usability/baselines/2026-07-26-atomic-python-promotion-trial.json`.

| Case | Calls | Eligible / reused root | Fresh prefill | Wall | Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust registry | 4 | 8,720 / 8,720 | 9,789 | 344,339 ms | 3.28 Wh |
| Python TTL | 4 | 8,200 / 8,200 | 9,351 | 282,700 ms | 1.39 Wh |
| React alert | 5 | 10,064 / 10,064 | 12,637 | 368,994 ms | 2.80 Wh |
| **Candidate total** | **13** | **26,984 / 26,984** | **31,777** | **996,033 ms** | **7.47 Wh** |
| Baseline total | 14 | not available | 47,271 | 951,852 ms | 9.97 Wh |

The independent auditor found 3/3 official passes, 3/3 verified-clean completions, zero false
verification, and exact reuse of every eligible root on all 13 calls. Python made the correct
change on its first mutation, passed its first check, and completed in the required four calls.
Fresh prefill fell 32.78% and energy fell 25.10% against the locked baseline.

Performance promotion remained open for this candidate. Aggregate wall time increased 4.64%
instead of falling 15%,
and Rust regressed 63.30% despite four complete disk-root hits. Python was within 1.53% of its
baseline and React improved 20.22%. Refill lookup, disk decode, validation/allocation, hydration,
snapshot, and queue time totalled only 1,103 ms for Rust; fresh-suffix prefill alone took 165,873
ms. This is evidence against a cache lookup or restoration defect, but one candidate observation is
not a machine-level explanation for the Rust outlier and cannot waive the per-case gate.

Accordingly, the Python workflow change is qualified, while the end-to-end performance row is not.
The required repeated interleaved comparison is recorded below.

## Paired median qualification

The exact `a69239ce` baseline was rebuilt in an isolated worktree and compared with the exact
`884d4c1b` candidate release binary. The baseline and candidate binary SHA-256 digests were
`6ba1a62b5496b780dfd61b3f4c6d3eecca63d84f53169300953ce867d564921f` and
`6261d9b2dd4db7c1876183ea1f04778137f8c4222f0006145b438cb600b33981`.
Three serial rounds rotated case order and alternated which revision ran first. Every scratch root
was independently audited. The retained raw report is
`/private/tmp/pb-paired-a692-vs-884-20260726-v2/paired-results.json`, with SHA-256
`d2f6ec007be1039f7fee8bf9c4f47dabb5900f97794d0d0b9319bd6076e39329`; its checked-in
summary is
`../../fixtures/harness-usability/baselines/2026-07-26-paired-workflow-promotion.json`.

| Metric | Baseline raw rounds | Baseline median | Candidate raw rounds | Candidate median | Gate |
| --- | --- | ---: | --- | ---: | --- |
| Wall time | 960,592 / 965,994 / 924,483 ms | 960,592 ms | 1,249,901 / 959,601 / 840,385 ms | 959,601 ms | 0.10% reduction; **fail** |
| Fresh prefill | 34,178 / 32,453 / 34,090 | 34,090 | 35,526 / 31,786 / 31,792 | 31,792 | at most 35,453; pass |
| Energy | 8.742 / 6.389 / 5.963 Wh | 6.389 Wh | 8.395 / 6.897 / 6.608 Wh | 6.897 Wh | 7.95% regression; **fail** |
| Model calls | 14 / 14 / 14 | 14 | 14 / 13 / 13 | 13 | one Rust five-call trial; **fail** |

All 18 tasks passed their official check and independent clean-delivery audit; there were zero false
verified completions. Candidate invocations reused all 82,389/82,389 eligible root tokens with no
reconciliation failure. The old baseline was warm after the preceding retained evaluator-debug run,
so it also restored partial disk-session prefixes. This is a stricter steady-state comparison than
the original single baseline. The plan's absolute 35,453-token fresh-prefill ceiling remains the
acceptance gate and passes; the warm baseline's 34,090-token median is diagnostic context, not a new
denominator that rewrites that locked threshold.

| Case | Baseline wall median | Candidate wall median | Candidate change |
| --- | ---: | ---: | ---: |
| Rust registry | 273,652 ms | 275,005 ms | +0.49% |
| Python TTL | 274,799 ms | 251,229 ms | -8.58% |
| React alert | 386,221 ms | 365,295 ms | -5.42% |

The per-case 10% regression bound therefore passes. The first candidate Rust trial emitted malformed
recursive JSON until the 1,792-token limit. pb rejected it without a repository mutation, and the
next bounded mutation completed correctly; the other two candidate Rust trials used four calls.
That safe five-call success is retained and fails the strict Rust/Python four-call floor. It is a
model/control outlier, not a cache hit or restoration defect.

Two evaluator corrections are explicit. The first scratch root,
`/private/tmp/pb-paired-a692-vs-884-20260726`, is excluded because the initial wrapper aborted on a
truthful nonzero model outcome before auditing it; `bd4058b3` makes such outcomes retained data. The
v2 report's ordering label says order alternated “by round and case,” while its raw trials and code
show alternation by round; `e74aa084` corrects that label and applies the locked fresh-token ceiling.
Neither correction changes a recorded trial or the non-promotion verdict.

This closes the single-sample uncertainty for that revision. Production performance remained open
because paired median wall time did not materially improve, paired energy regressed, and one
successful Rust run missed the call floor. The exact-root implementation was qualified; the
end-to-end workflow was not yet.

## Stable closure extension and first locked rerun

Revisions `9274185c` through `8754afba` moved the successful bounded repair path onto four explicit
controller-owned authority roots: `PlanningClosure`, `PlanReviewClosure`,
`ImplementationMutation`, and `CodeReview`. Closure is proof-driven rather than task-name-driven:
the controller exposes a terminal-only planning or plan-review root only after exact full-path
observations satisfy that stage's prerequisites. Tool-enabled recovery remains available when the
proof is absent or a review challenges the artifact. The rendered token digest, model namespace,
and native tool schema remain the cache key and authority boundary; the stage label is diagnostic,
not permission to reuse approximate state.

The first complete paired rerun used revision `8754afba6559695088281fd54947c5178a4186b3`
and release SHA-256
`c20f9c9e95a56176f52c54e13d7d647f280a703e6881cd3355ab8ec0def5d528`. Its retained report is
`/private/tmp/pb-paired-full-a692-vs-8754afba-20260726/paired-results.json`, SHA-256
`31101803793fcb87e2cea6d8b7fe63c4fbbff7a7ad1e6c52eefe2b3fff659b6e`.

All 18 tasks passed their official check and clean-delivery audit with zero false verification. The
candidate used exactly four calls in every task and restored 55,008/55,008 eligible root tokens.
Fresh-prefill median fell from 27,153 to 21,189 tokens and energy fell 20.28%. Aggregate wall median
fell only 12.84%, however, below the locked 15% gate. The per-case wall changes were -22.17% Rust,
-17.13% Python, and -3.09% React. This valid near-miss is retained rather than rounded into a pass.

The remaining profile was not a FlashMoe refill defect. In the later passing candidate, exact-root
lookup, disk decode, CPU validation/allocation, Metal hydration, and snapshot capture totalled
11,212 ms across 36 calls, while actual fresh-suffix prefill took 844,742 ms. Every suffix selected
the already-qualified Qwen layer-major graph. An application expert cache, Q4-only refill path,
alternate runtime, new short-suffix command, or scheduler bypass would add production complexity
without targeting the observed bottleneck, so none was added.

The near-miss transcripts instead exposed variable review decode. On the same React task, the old
plan review generated 310 tokens and code review generated 147 even though both passed; their
compact equivalents generate 156 and 85 tokens. This made duplicate assessment prose the next
bounded controller target.

## Review compaction and performance promotion

Revision `16fd0e68208b48aa9ecfe681a047938a5768179d` makes every new plan-review and
code-review assessment contain only its required dimension and status. All six dimensions remain
mandatory. A concern or failure records its explanation and current evidence once in the typed
challenge or finding, and deterministic validation rejects a passing verdict with any non-passing
assessment or a revision verdict without both a blocker and a non-passing assessment. Legacy
checkpoint detail remains deserializable. The instruction change advances the explicit system-root
version to `agent-system-v5`; exact rendered tokens, not the version label alone, still decide
checkpoint reuse.

The release binary SHA-256 was
`dc320e6ceb99384d515bb246fbeb266bee2aac81c114fef7c0dbaa4241d8c2fd`. A single excluded
warmup at `/private/tmp/pb-paired-warmup-16fd-react-v5-20260726` populated the intentionally changed
review roots. The scored three-round comparison is retained at
`/private/tmp/pb-paired-full-a692-vs-16fd0e68-retry2-20260726/paired-results.json`, SHA-256
`19c5c45888e93e67573a420917187533d5cffab4e12353933f3b3aa86fc8e703`. Its checked-in
summary is
`../../fixtures/harness-usability/baselines/2026-07-26-stable-prompt-root-performance-promotion.json`.

| Metric | Baseline raw rounds | Baseline median | Candidate raw rounds | Candidate median | Result |
| --- | --- | ---: | --- | ---: | --- |
| Wall time | 1,026,134 / 878,721 / 1,069,669 ms | 1,026,134 ms | 700,193 / 699,815 / 709,477 ms | 700,193 ms | 31.76% reduction; pass |
| Fresh prefill | 36,453 / 30,843 / 38,387 | 36,453 | 21,020 / 21,031 / 21,036 | 21,031 | 42.31% reduction and below 35,453; pass |
| Energy | 7.169 / 6.014 / 4.973 Wh | 6.014 Wh | 4.442 / 4.365 / 4.135 Wh | 4.365 Wh | 27.43% reduction; pass |
| Model calls | 12 / 12 / 12 | 12 | 12 / 12 / 12 | 12 | four per task; pass |

| Case | Baseline wall median | Candidate wall median | Candidate change |
| --- | ---: | ---: | ---: |
| Rust registry | 267,165 ms | 230,738 ms | -13.63% |
| Python TTL | 350,131 ms | 241,238 ms | -31.10% |
| React alert | 282,017 ms | 232,576 ms | -17.53% |

All 18 trials again passed their official check and independent clean-delivery audit, with zero false
verified completion. Candidate runs restored 54,090/54,090 eligible root tokens, recorded zero
cache reconciliation failures, and completed 9/9 queued durable checkpoints. Every candidate task
used exactly four model calls. The current exact root sizes are 1,317 PlanningClosure tokens, 1,073
PlanReviewClosure tokens, 1,854 Rust/React or 1,416 Python ImplementationMutation tokens, and 1,912
CodeReview tokens.

Two attempts are excluded as experiment errors:

- `/private/tmp/pb-paired-full-a692-vs-16fd0e68-20260726`
- `/private/tmp/pb-paired-full-a692-vs-16fd0e68-retry1-20260726`

Both wrappers ran inside a command sandbox whose child process could not acquire Metal; an immediate
host-native probe succeeded. The final evaluator therefore ran on the host with the same explicit
binaries, model, sampling, ordering, and scratch-isolation contract. Neither invalid attempt
contributes to correctness or performance claims.

This passes every locked Phase 4 performance gate without weakening exact-token reuse, stage tool
authority, independent review, checks, semantic commits, or clean-worktree completion.

## Current release-candidate full corpus

The current release binary then ran all 24 cases serially from clean, independently prepared Git
workspaces at
`/private/tmp/pb-prompt-root-phase5-production-16fd0e68-20260726`. The same independent auditor
reran every official check, verified immutable fixtures, derived the real Git delta, reconciled the
recorded commit and worktree, and only then compared pb's terminal state.

One raw result was excluded as an experiment error. `react_field_error_linkage` rendered a stable
error element with both `role="alert"` and `id="name-error"`, but the fixture required those two
attributes in one serialization order. Revision `49edee32` made that assertion order-independent
without weakening the semantic contract. The isolated replacement run at
`/private/tmp/pb-prompt-root-phase5-field-fixture-v2-16fd0e68-20260726/react_field_error_linkage`
independently passed, reached a clean semantic commit, and restored the exact root on all five
model calls. The original raw run remains preserved and excluded rather than being rewritten.

| Result | Qualified value |
| --- | ---: |
| Official task passes | 21/24 |
| pb verified-clean completions | 21/24 |
| False verified completions | 0 |
| Bounded model/control limits | 3 |
| pb defects | 0 |
| Model invocations | 111 |
| Rendered prompt tokens | 372,362 |
| Cached prefix tokens | 171,581 |
| Fresh-prefill tokens | 200,781 |
| Eligible stable-root tokens | 168,622 |
| Reused stable-root tokens | 166,831 |
| Root-hit invocations | 109/111 |
| Cache reconciliation failures | 0 |
| Durable checkpoints | 26 queued, 26 completed, 0 failed |
| Wall time | 6,814,799 ms |
| Measured energy | 53.18 Wh, complete coverage |

The two non-hit invocations were truthful first uses: one newly encountered `RepairMutation` root
and one 30-token sessionless `TaskArtifact` root. Both were persisted; unchanged populated roots did
not miss or falsely hit. The three unsuccessful cases were `rust_safe_relative_path`,
`python_word_frequency`, and `react_tabs_semantics`. Their official checks failed, pb kept their
contracts unsatisfied, and no task commit or verified completion was claimed. All unsuccessful work
remained within its bounded repair path.

Every invocation used the qualified Qwen layer-major matrix for its actual fresh suffix. Across the
qualified corpus, root lookup, disk decode, CPU validation/allocation, Metal hydration, and snapshot
capture totalled 28,682 ms; fresh-suffix prefill took 2,557,248 ms. This independently confirms the
paired profile: refill suffix work dominates, and the existing production graph already targets it.
No new FlashMoe data flow, expert cache, Q4-only path, alternate runtime, or hidden control is
justified.

## Release gates and exclusions

Formatting, strict compiler/correctness lints, all Rust targets, the 76 web tests, usability corpus
validation, the web build, documentation validation, the macOS arm64 release build, and the
required native smoke pass. The final Rust result was 1,436 passed and 23 ignored, plus two
auxiliary tests. The release binary SHA-256 remains
`dc320e6ceb99384d515bb246fbeb266bee2aac81c114fef7c0dbaa4241d8c2fd`; the one-token native
smoke exited zero and produced `5` for `2+2=`. That output remains a local-model quality warning,
not a cache or workflow correctness claim.

Two earlier scratch roots are explicitly excluded as experiment errors:

- `/private/tmp/pb-prompt-root-phase5-final-927198cb-20260725`
- `/private/tmp/pb-prompt-root-phase5-final2-927198cb-20260725`

Their wrapper execution could not acquire Metal. Direct release-binary runs used Metal normally;
neither excluded root contributes to correctness or performance claims.

## Production status and watchpoints

The plan's production gate is closed. This does not claim that a privacy-first local model solves
every bounded task: the current corpus retains three model/control limits, and the raw arithmetic
smoke remains poor. Production-ready here means cache and workflow control remain correct and safe
under those limitations, while the locked successful workload materially improves.

Future work should monitor generated-token and repair variability, the two truthful first-use root
misses, and full-corpus local-model quality. Any further optimization must repeat the paired protocol
and preserve exact-token identity, typed stage authority, independent checks and review, semantic
commit ownership, local-only cache state, and the existing FlashMoe scheduling and expert-I/O
invariants.
