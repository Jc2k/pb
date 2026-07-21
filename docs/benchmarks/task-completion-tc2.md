# TC2 useful-coding qualification

Captured: 2026-07-21

Model: native `mlx-community/Qwen3-Coder-Next-4bit`, affine Q4, 48 layers, 512 experts per
layer, K=10, approximately 40.5 GiB resident expert corpus

Qualified source: `2d91a2e9561988c496161f17dddc2a13baf41880`
(`fix: reserve final implementation closure turn`)

Fixture: `fixtures/harness-task-completion/tc2-task.txt` with
`fixtures/harness-task-completion/tc2-contract.json`

## Hypothesis and contract

TC2 tests whether the native local model can complete a small useful coding task through strict
planning, fresh plan review, bounded implementation, trusted checks, repair, fresh code review,
managed commit, and independent verification. It must create a dependency-free JavaScript slugify
module, meaningful model-authored Deno tests, and documentation.

Success requires:

- only `slugify.mjs`, `slugify.test.mjs`, and `README.md` change;
- the independent behavior oracle passes;
- the trusted dependency gate finds no HTTP(S), JSR, or npm import;
- the model-authored Deno tests pass;
- fresh code review covers all three paths and all three current check receipts;
- pb creates a semantic task-owned commit;
- the final worktree is clean; and
- the run exits zero with `contract_status=satisfied` and `verified_completed=true`.

The locked model settings were temperature 0, top-k 1, seed 0, 1,536 maximum generated tokens,
131,072 context tokens, and six model-driven steps per strict stage. Every attempt used a preserved
scratch root.

## Failure-led controller improvement

Initial run `1784669878454-39823-0` on the original two-check fixture created all three requested
files and passed both declared checks. It nevertheless ended `step_limit`, unsatisfied, and
unverified because the model spent implementation steps 5 and 6 calling those checks and never
submitted its typed implementation artifact. The workspace remained uncommitted and dirty. This
was truthful containment of a model closure failure, not task completion. The run used eight model
calls, 49,995 prompt tokens, 3,680 generated tokens, 912,955 ms, and 9.64 Wh.

That preserved artifact also exposed an experiment error: its test file imported Deno's remote
assertion library even though the task said dependency-free, while the contract had no independent
dependency check. The check passed only because network access was available. The fixture now has
an explicit `dependency_free` gate and requires its receipt in fresh review.

Commit `2d91a2e9` reserves the final ready implementation or repair turn for
`submit_implementation`. Earlier turns keep their ordinary authority, and an unfinished final turn
still exposes edits and replan. The rule does not infer completion or skip checking: it only
prevents a redundant diagnostic from consuming the last opportunity to submit typed accounting.
Unit and end-to-end tests cover the ready and unfinished cases.

One attempted post-build run, `1784671540864-58032-0`, failed before inference because the command
was launched inside a workspace sandbox without a visible Metal device. Its scratch root is
preserved and the attempt is classified as an experiment-environment error, not model or pb
task-completion evidence.

## Qualified run

Run `1784671564495-58382-0` used the corrected release and strengthened three-check fixture.
Planning and plan review each passed on their first turn. Implementation created the three plan
paths in order. A 1,536-token README action decoded only an incomplete `<tool_call>` and was
rejected atomically; the following turn wrote a compact complete README. The fifth turn passed the
behavior check. On turn six, the new closure checkpoint exposed only the typed implementation
submission, which the model completed successfully.

Deterministic checking then rejected the remote assertion import through `dependency_free` and
entered repair. The model read the current files, replaced the test import with local assertions,
and passed behavior and dependency checks. The repair's final reserved turn submitted current
implementation accounting. Harness-owned checking then passed all three checks, fresh review
accepted the exact fingerprint, and pb created task commit
`7fd899ce771857bd39a3c6d0ec77e85e6b230607` (`feat: add dependency-free slugify module`).

| Result | Value |
| --- | ---: |
| Verified completion | yes |
| Model calls | 16 |
| Prompt tokens | 99,016 |
| Generated tokens | 4,903 |
| Tool calls | 16 |
| Wall time | 1,167,352 ms |
| Energy | 12.90 Wh |
| Repair cycles | 1 |
| Rejected workflow artifacts | 0 |
| Parse failures | 1 |
| Final changed paths | 3 |

Independent review reran the behavior oracle, dependency gate, and six Deno tests successfully from
the recorded commit. `HEAD` was the recorded OID, `git ls-files` contained only the three allowed
paths, `git diff --check` passed, and `git status --porcelain` was empty.

## Findings

1. **Positive evidence — useful task completion is achievable locally.** The resident open model
   completed planning, implementation, trusted checks, a real repair, fresh review, semantic
   commit, clean worktree, and external verification without cloud inference.
2. **P1 controller defect — fixed and natively qualified.** Reserving the last ready implementation
   and repair turn converted the observed “correct files but no stage submission” failure into a
   complete workflow while retaining strict validation and unfinished-work authority.
3. **Positive evidence — privacy requirements need executable gates.** The model twice selected a
   remote test dependency. Once the requirement became a trusted check, pb rejected it, supplied
   exact diagnostics, invalidated stale evidence after repair, and required every check again.
4. **P2 model limitation — output shaping remains unreliable.** Both TC2 runs spent the full output
   allowance on an action that collapsed to `<tool_call>`. The bounded controller recovered without
   a partial file, but the failure added several minutes and watt-hours.
5. **P2 efficiency — functional promotion passes; efficiency promotion does not.** The run stayed
   below the 30-minute and 15 Wh envelopes, but exceeded the 12-call target with 16 calls. Repair
   reread all three files even though only the test file caused the failure, and code review plus
   typed accounting remain expensive relative to an 80-line artifact.
6. **P3 artifact quality — acceptable but not polished.** The committed module satisfies the exact
   contract and the tests are meaningful. The test file retains one unused local helper and the API
   coerces non-string input without documenting that extension. These are non-blocking review
   quality signals for a broader corpus.

## Disposition

TC2 is functionally qualified and remains a repair-path regression gate. The next task-completion
work should expand the offline corpus and reduce calls through deterministic acceptance skeletons,
failure-focused carried evidence, and high-value independent tool batches. Gate strength, local
inference, and explicit dependency checks remain fixed.
