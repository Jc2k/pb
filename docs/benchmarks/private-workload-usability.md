# Private-workload usability corpus

**Status:** Shipped corpus and tooling; 24/24 fixtures qualified; original and
post-optimization stratified samples plus a complete release-candidate run recorded.

This corpus exists to answer an internal question: is pb useful for the kinds of
small Rust, Python, and React/TypeScript repairs that occur in private local
repositories? It is not a public leaderboard, a claim that pb is better than
another agent, or a customer-acquisition artifact.

## Corpus

The checked-in corpus contains 24 dependency-bounded cases:

| Language         | Cases | Task shapes                                                             |
| ---------------- | ----: | ----------------------------------------------------------------------- |
| Rust             |     8 | compact algorithms, repository logic, state, parsing, path safety       |
| Python           |     8 | compact algorithms, repository logic, state, time boundaries, atomicity |
| React/TypeScript |     8 | component behavior and accessibility rendered with React 19             |

The cases are synthetic. They use the task shapes and evaluation discipline of
the [Aider polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark),
[SWE-bench](https://github.com/SWE-bench/SWE-bench),
[SWE-bench Multilingual](https://github.com/swe-bench/SWE-bench_Multilingual),
and [ReactBench](https://www.reactbench.com/). No upstream implementation, test,
issue text, or gold patch is copied into the corpus. Source-family metadata
records whether each case is shape-derived or behavior-derived.

Each case is a fresh local Git repository with immutable tests, one allowed
implementation path, a required offline behavior check, a required fresh review,
a semantic task-owned commit, and a clean-worktree terminal gate. Reference
solutions exist only in the corpus definition; the runner does not materialize
them into the model workspace.

Rust checks use Cargo offline with a locked dependency-free package. Python
checks use the standard library and remain compatible with Python 3.9. React
checks pin React, React DOM, and type packages in a Deno lockfile. Their exact
packages must be cached once before a machine is disconnected:

```bash
deno task cache:usability-react
```

Every subsequent React check uses `--cached-only --frozen`. The cache step is
the only part of the suite designed to contact a package registry.

## Qualification and scoring

`deno task test:usability` independently proves all 24 fixture pairs:

1. the seeded implementation fails its official behavior check;
2. running that check does not dirty the controlled repository;
3. the isolated reference implementation passes; and
4. the only resulting repository delta is the contract-allowed implementation
   path.

The model score deliberately keeps four layers separate:

| Layer                     | Meaning                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| Official task correctness | Immutable corpus behavior check passes, independently rerun after pb exits                    |
| pb completion             | pb records a satisfied contract and `verified_completed=true`                                 |
| Safe delivery             | Recorded commit is `HEAD`, semantic, allowed-path-only, and the worktree is clean             |
| Efficiency                | Wall time, model calls, prompt-cache split, generated tokens, tool calls, and measured energy |

`pb verified` with a failed independent task check, altered immutable fixture,
dirty worktree, forbidden path, or mismatched commit is
`pb_defect_false_verification`. A correct or nearly correct workspace that pb
does not verify remains a model-or-control limitation. This distinction prevents
artifact quality and controller safety from being averaged into a reassuring but
useless score.

## Running and auditing

List the cases:

```bash
deno run --allow-read scripts/run-harness-usability.ts --list
```

Run a selected case with a release binary and retain its scratch root:

```bash
deno run --allow-read --allow-write --allow-run \
  scripts/run-harness-usability.ts \
  --scratch-parent /private/tmp/pb-usability-run-1 \
  --case rust_registry_removal
```

Omit `--case` to run all 24 sequentially. The runner continues after nonzero
harness outcomes, keeps every workspace, event stream, journal, and run index
beneath the supplied scratch parent, and appends the process result for each
case to `run-results.jsonl`.

On a qualification host where a Deno child cannot see Metal, add `--prepare-only`. The runner
materializes the same immutable case and prints its typed paths, task, and limits as JSON without
starting pb. Invoke the release `pb harness agent` binary directly with those values; this preserves
the corpus contract while keeping Metal access outside the orchestration sandbox.

Independently audit one or more retained runs:

```bash
deno run --allow-read --allow-run scripts/audit-harness-usability.ts \
  /private/tmp/pb-usability-run-1/rust_registry_removal
```

The auditor reruns the official check, compares immutable seed files, derives
the actual task delta from Git, verifies the commit and clean worktree, reads
the preserved pb event stream, classifies the run, and emits per-run plus
aggregate JSON. It does not accept pb's completion verdict as evidence of task
correctness.

## 25 July 2026 qualification sample

The first model qualification used revision `03f1ddcc`,
`hf://mlx-community/Qwen3-Coder-Next-4bit`, the FlashMoe MLX Q4 backend,
temperature 0, top-k 1, and seed 0. The retained scratch root is
`/private/tmp/pb-usability-qualification-20260725-c`. The machine-readable
record is
`fixtures/harness-usability/baselines/2026-07-25-qwen3-coder-next.json`.

This is a stratified harness-compatibility sample, not a score for all 24 cases.

| Case                   | Official | pb verified clean | Terminal outcome            |  Calls | Fresh prefill | Generated |        Wall |       Energy |
| ---------------------- | -------: | ----------------: | --------------------------- | -----: | ------------: | --------: | ----------: | -----------: |
| Rust registry removal  |     pass |               yes | final                       |      7 |        26,503 |     1,260 |      8m 20s |      6.49 Wh |
| Python TTL boundary    |     pass |               yes | final after one failed edit |      8 |        16,171 |     1,541 |      7m 09s |      3.77 Wh |
| React accessible alert |     fail |                no | step limit                  |     20 |        57,575 |     5,675 |     24m 03s |     15.39 Wh |
| **Sample total**       |  **2/3** |           **2/3** | —                           | **35** |   **100,249** | **8,476** | **39m 32s** | **25.65 Wh** |

There were no false verified completions. Rust and Python each ended in a clean
semantic commit that changed only the intended implementation. Python initially
introduced invalid indentation, consumed the focused check failure, repaired it,
and then completed.

React added the alert semantics but composed `alert error` instead of the
required `alert alert--error`. It then requested four replans, repeated the same
failed check, once generated malformed repetitive structured output until the
token cap, and exhausted the cumulative planning budget without making the final
small repair. pb preserved the changed workspace and reported
`contract_status=unsatisfied`, `verified_completed=false`, and
`termination_reason=step_limit`.

Before the retained sample, one excluded trial found an experiment error: Cargo
and Deno checks created untracked lock/build artifacts during diagnostic
preview. pb correctly refused to certify the mutated workspace. The fixtures now
seed lockfiles, ignore build products, use frozen checks, and the qualifier
asserts that checks are side-effect-clean.

The release backend loaded and generated successfully, but a deterministic raw
`2+2=` smoke emitted `5` and then `5 for very large` at four tokens. That is a
model/runtime-quality warning; it is not counted as a corpus result.

## 25 July 2026 production-candidate rerun

Revision `a69239ce` repeated the same model, backend, cases, contract limits, and deterministic
sampling after the bounded-workflow call-reduction work. The machine-readable record is
`fixtures/harness-usability/baselines/2026-07-25-qwen3-coder-next-call-reduction.json`. Rust and
Python were rerun from fresh workspaces on the committed revision. React used the exact source tree
later committed as that revision; documentation-only commit metadata did not change its release
binary. The recorded release-binary SHA-256 is
`9cfc827dd00f8f86d09b41c9df531393790896a6eaf03770c76f7df055d302e2`; the required one-token
FlashMoe smoke exited zero and returned `5` for `2+2=`.

| Case                   | Official | pb verified clean | Calls | Fresh prefill | Generated |    Wall |  Energy |
| ---------------------- | -------: | ----------------: | ----: | ------------: | --------: | ------: | ------: |
| Rust registry removal  |     pass |               yes |     4 |         9,041 |       578 | 3m 31s | 2.78 Wh |
| Python TTL boundary    |     pass |               yes |     4 |        13,177 |       681 | 4m 38s | 2.92 Wh |
| React accessible alert |     pass |               yes |     6 |        25,053 |       967 | 7m 43s | 4.27 Wh |
| **Sample total**       |  **3/3** |           **3/3** | **14** |    **47,271** | **2,226** | **15m 52s** | **9.97 Wh** |

Against the original sample, model calls fell from 35 to 14 (60.0%), wall time from 39m 32s to
15m 52s (59.9%), rendered prompt tokens from 162,208 to 63,439 (60.9%), fresh prefill from 100,249
to 47,271 (52.8%), generated tokens from 8,476 to 2,226 (73.7%), and measured task energy from
25.65 Wh to 9.97 Wh (61.1%). Correct, verified-clean completion improved from 2/3 to 3/3; neither
sample contained a false verified completion.

Rust and Python each used exactly one model call for planning, plan review, mutation with inline
completion, and code review. React initially chose the wrong modifier class and then produced one
malformed JSX repair. Both failures remained in the accepted one-path repair workflow; the release
reached Ready after two focused repair calls instead of repeating planning and review. This directly
closes the replan-thrash failure recorded in the original sample without weakening the immutable
check, fresh code review, semantic commit, allowed-path, or clean-worktree gates.

The new prompt-cache attribution reported two cold starts and four prompt divergences across the 14
calls. It also showed 16,168 reused tokens, including a warm Rust run with a persisted-prefix hit in
all four stages. Reuse is therefore functioning, while cold stage prefixes and stage-schema
divergence remain the largest measured efficiency gap after redundant-call removal.

One excluded orchestration attempt launched the release binary as a Deno child inside the evaluation
sandbox and received no Metal device; direct `pb harness agent` runs used Metal normally. Those
three setup failures remain preserved as experiment errors and are not included in the qualification
sample.

## 26 July 2026 complete prompt-root qualification

Revision `bb3e89f9` ran all 24 cases with the release binary after stable-root canonicalization,
shared FlashMoe prompt-root persistence, byte-budgeted retention, access-recency repair, and
sessionless constrained-root persistence. The independent audit found 21 verified-clean passes,
three truthful bounded model/control limits, zero false verification, 127/127 root-hit invocations,
and 268,147/268,147 eligible root tokens reused. The three-language sample passed its fresh-prefill
and energy targets but failed its wall-time and call-shape promotion gates, so it is not labelled
production-ready. See the
[stable prompt-root release qualification](stable-prompt-root-production.md) and its checked-in
machine-readable baseline for the full result and exact exclusions.

## Engineering priorities

1. **Improve stable-prefix availability.** The call count is now bounded by useful stage work in the
   successful cases, but cold starts and stage-schema divergence still account for most fresh
   prefill. Use the typed miss reasons to improve prefix stability without weakening per-stage tool
   authority. Implementation and production qualification are tracked by the
   [stable prompt roots and FlashMoe refill plan](../stable-prompt-cache-and-refill-plan.md).
2. **Keep repair quality observable.** The React run proves the controller can contain weak local
   edits without replanning, but two repair cycles for a one-line JSX change remain a model-quality
   cost. Track repair cycles separately from controller overhead.
3. **Qualify inference quality separately.** Keep a tiny deterministic sanity
   set ahead of expensive agent runs so a degraded local backend is not mistaken
   for controller regression.
4. **Expand the model baseline deliberately.** The complete 24-case model run is
   useful for periodic qualification, but it should not be required on every
   change at the observed cost. Per-change work should run the fixture validator
   plus a rotating Rust/Python/React slice; retained full runs can be compared
   only when model, backend, parameters, corpus, and pb revision are locked.
