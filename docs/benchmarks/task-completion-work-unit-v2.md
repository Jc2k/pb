# Work-Unit Controller v2 qualification

Captured: 2026-07-22

Model: native `mlx-community/Qwen3-Coder-Next-4bit`, affine Q4, 48 layers, 512 experts per layer,
K=10, with the approximately 40.5 GiB resident expert corpus. Every model trial used temperature
0, top-k 1, seed 0, the checked-in task contract, and a distinct preserved scratch root.

## Claim boundary

This record qualifies TC1 and TC2 and records targeted TC3 failure conversions. It does not claim a
new locked 11-case TC3 aggregate: the seven W0 successes have not all been rerun on the final source.
The full W6 gate therefore remains open even though the targeted results reach the 9/11 functional
floor when combined with the unchanged W0 success set.

A success counts only when pb reports a satisfied contract and verified completion, the semantic
managed commit is `HEAD`, the worktree is clean, the exact allowed delta is present, and the
supervisor independently reruns the contract checks.

## TC1 — ordered creation

| Run | Scratch / run ID | Commit | Calls | Wall ms | Energy Wh |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `/private/tmp/pb-wucv2-tc1-fixed-1-20260722` / `1784711785477-68811-0` | `38fa9ed10698ddb4ce9947c047752728bffe4097` | 6 | 411,631 | 4.84 |
| 2 | `/private/tmp/pb-wucv2-tc1-fixed-2-20260722` / `1784712242786-70683-0` | `d62b4121b0e4ca85ba0b5690c5e7a61f08a09474` | 6 | 347,792 | 4.42 |
| 3 | `/private/tmp/pb-wucv2-tc1-fixed-3-20260722` / `1784712687561-72945-0` | `42d881f3bb968a26b5940f785adfa6d3bb466b8f` | 6 | 369,415 | 4.37 |

Result: **3/3 verified at six calls**. Exact bytes, two-path deltas, contract checks, fresh review,
semantic commits, and clean worktrees passed independent audit in every run.

## TC2 — useful coding

| Run | Scratch / run ID | Commit | Calls | Prompt / cached / fresh | Generated | Wall ms | Energy Wh |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | `/private/tmp/pb-wucv2-tc2-final-1-20260722` / `1784727519941-50969-0` | `1984675c1953e45b92939405d6c95ba33cc9ea9f` | 7 | 35,303 / 17,210 / 18,093 | 2,193 | 599,405 | 6.11 |
| 2 | `/private/tmp/pb-wucv2-tc2-final-2-20260722` / `1784728154875-52937-0` | `b3cc22e3f633033f871cecfd030c80fb97f12fe3` | 7 | 35,293 / 17,210 / 18,083 | 2,181 | 583,619 | 6.39 |
| 3 | `/private/tmp/pb-wucv2-tc2-final-3-20260722` / `1784728770556-55016-0` | `b8dd44eab58e44a9a533e2d94ad174fa2c306d08` | 7 | 35,288 / 17,210 / 18,078 | 2,153 | 586,630 | 5.98 |

Result: **3/3 verified at seven calls** with identical three-path artifacts and zero rejected
actions or repair turns. Median wall time was 586,630 ms and median energy was 6.11 Wh. Against the
qualified 1,167,352 ms, 12.90 Wh, 16-call baseline, that is 49.7% less wall time, 52.6% less energy,
and 56.3% fewer calls. The path-specific hint states the literal local expectation; it grants no
tool authority, check evidence, or acceptance credit.

## TC3 — targeted failure conversions

| W0 failure | Scratch / run ID | Result | Calls | Wall ms | Energy Wh | Independent audit |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `resume_partial_case_helpers` | `/private/tmp/pb-wucv2-tc3-resume-partial-case-helpers-1-20260722` / `1784729407708-57538-0` | verified, commit `42141d776d3cdc7b4e3c45ae9346d3165687476d` | 7 | 454,330 | 4.91 | adopted formatter bytes and created tests were accounted separately; behavior, tests, dependency check, exact delta, semantic `HEAD`, and clean status passed |
| `remove_legacy_marker` | `/private/tmp/pb-wucv2-tc3-remove-legacy-marker-4-20260722` / `1784732727735-57336-0` | verified, commit `2a0717536f23058be411f4fdb09ac4e21f661b20` | 8 | 431,485 | 4.56 | exact deletion and README bytes, required check, two-path delta, fresh review, semantic `HEAD`, and clean status passed; zero rejected workflow or tool actions |

These two conversions address the adopted-work accounting and delete/modify controller gaps from
W0. Combined with the seven independently audited W0 successes, the targeted evidence reaches the
9/11 floor. `mixed_counter_change` and `create_slugify_repair` remain the known unconverted W0
failures and are primarily constrained by generated-test/dependency quality rather than a false
completion or scope escape.

## Defects found during qualification

Preserved native runs exposed three pb defects that deterministic tests now cover:

1. carried planning evidence populated the ledger path set but not the executor's exact byte
   fingerprint, producing an impossible mutation-ready/read-required state (`823300ad`);
2. Git's tracked-deletion `missing` entry was treated as a present path, so a completed delete unit
   did not advance to the following modify unit (`e768b5bc`); and
3. that same synthetic entry changed the content fingerprint when the deletion was staged and
   committed, invalidating pb's own current successful check after managed commit (`c7e3000e`).

The superseded scratches are retained. The final deletion run used the expected stage sequence,
eight model invocations, nine successful tool calls, no replan, one authoritative check, fresh
inspection, and a managed commit. Full source validation after the fixes passed 1,259 library tests,
both environment-contract integration tests, the correctness/suspicious Clippy profile, corpus
validation, and the documentation build.

## Remaining gate

Run all 11 TC3 cases on one final source and report a new aggregate, then compare supported local
model tiers. Until that happens, W6 is partial rather than complete. The controller gap itself is
closed for ordered create, useful mixed creation, resumed/adopted work, and delete/modify work under
the tested contracts.
