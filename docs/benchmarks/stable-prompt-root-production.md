# Stable prompt-root release qualification

Status: **Release-candidate cache qualification passed; production performance promotion remains
open.**

Revision `bb3e89f9` completes the stable stage-root, shared cache-root, disk-retention, and
sessionless constrained-root work described by the
[stable prompt roots and FlashMoe refill plan](../stable-prompt-cache-and-refill-plan.md). The
machine-readable record is
`../../fixtures/harness-usability/baselines/2026-07-26-stable-prompt-root-production.json`.

This is deliberately not labelled production-ready. Exact-root correctness, restart reuse,
resource behavior, the 24-case safety corpus, fresh-prefill reduction, and energy reduction pass.
The locked three-language wall-time and call-shape gates do not.

## Candidate and protocol

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

## Full 24-case result

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
performance must remain separate. The cache work is correct and materially reduces prefill, but it
cannot be called production-ready under the plan's existing end-to-end gate.

## Release gates and exclusions

Formatting, strict compiler/correctness lints, all Rust targets, the 76 web tests, the web build,
documentation validation, the macOS arm64 release build, and the required native smoke pass. The
full Rust result was 1,425 passed and 23 ignored, plus two auxiliary tests.

Two earlier scratch roots are explicitly excluded as experiment errors:

- `/private/tmp/pb-prompt-root-phase5-final-927198cb-20260725`
- `/private/tmp/pb-prompt-root-phase5-final2-927198cb-20260725`

Their wrapper execution could not acquire Metal. Direct release-binary runs used Metal normally;
neither excluded root contributes to correctness or performance claims.

## Remaining promotion work

Do not weaken exact-token reuse, stage authority, review, checks, or completion gates to improve the
headline. The next qualification should first make the locked agent comparison less sensitive to a
single repair trajectory: retain per-call prefill and decode time, use repeated paired runs on a
locked machine, and report medians plus raw trials. Any new implementation should target the phase
that remains dominant in those paired profiles. A change still needs 15% end-to-end wall reduction,
the Rust/Python four-call floor, and the per-case regression bound before this record may be promoted
to production-ready.
