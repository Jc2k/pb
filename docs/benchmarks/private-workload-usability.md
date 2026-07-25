# Private-workload usability corpus

**Status:** Shipped corpus and tooling; 24/24 fixtures qualified; one stratified
three-language model sample recorded on 25 July 2026.

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

## Engineering priorities

1. **Reduce workflow cost before expanding routine coverage.** Seven or eight
   model calls and roughly eight minutes for a one-line successful repair are
   not acceptable for ordinary private work. Use this corpus to measure
   controller-call elimination and prefill improvements independently of pass
   rate.
2. **Stop replan thrash after focused check evidence.** The React trace shows a
   failing immutable check, a single allowed path, and an already accepted plan.
   Reproductions should determine whether a controller-owned repair continuation
   can retain plan authority while preventing repeated full planning/review
   cycles. This is not yet classified as a clear pb defect because the model
   itself repeatedly requested replanning and pb bounded it truthfully.
3. **Qualify inference quality separately.** Keep a tiny deterministic sanity
   set ahead of expensive agent runs so a degraded local backend is not mistaken
   for controller regression.
4. **Expand the model baseline deliberately.** The complete 24-case model run is
   useful for periodic qualification, but it should not be required on every
   change at the observed cost. Per-change work should run the fixture validator
   plus a rotating Rust/Python/React slice; retained full runs can be compared
   only when model, backend, parameters, corpus, and pb revision are locked.
