# TC3 native task-completion baseline

Captured: 2026-07-22

Source: `9d3d3416` (`feat: improve local task completion reliability`) with a documentation-only
Work-Unit Controller v2 tracker edit in the supervisor worktree. The release binary was rebuilt
after `deno task build:web`.

Model: native `mlx-community/Qwen3-Coder-Next-4bit`, affine Q4, 48 layers, 512 experts per layer,
K=10, approximately 40.5 GiB resident expert corpus. Sampling and per-case limits come from
`fixtures/harness-task-completion/corpus.json`.

## Contract

Each of the 11 offline cases runs once in a distinct preserved scratch root. A success counts only
when pb reports `contract_status=satisfied` and `verified_completed=true`, the recorded semantic
commit is `HEAD`, the worktree is clean, the commit changes only allowed task paths, and an
independent supervisor reruns every contract command successfully. Protocol containment, artifact
quality, and efficiency remain separate observations.

## Environment validation

The initial `create_exact_pair` attempt at
`/private/tmp/pb-tc3-w0-create-exact-pair-20260722` stopped before inference because the workspace
sandbox exposed no Metal device. The scratch root is preserved and the attempt is classified as an
experiment-environment error, not task-completion evidence. All native cases therefore run on the
approved host with fresh scratch roots.

## Results

| Case | Scratch / run | Verified | Commit | Calls | Prompt | Generated | Wall ms | Energy Wh | Independent audit |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `create_exact_pair` | `/private/tmp/pb-tc3-w0-create-exact-pair-host-1-20260722` / `1784701259745-37329-0` | yes | `0241334e065a960e121b7687d79cb265746d6b12` | 7 | 36,241 | 1,308 | 445,253 | 3.05 | exact bytes, one required check, two-path delta, semantic `HEAD`, and clean worktree passed |
| `fix_average_divisor` | `/private/tmp/pb-tc3-w0-fix-average-divisor-host-1-20260722` / `1784701770335-39308-0` | yes | `e947cb8a2e350d7d255e09c003ab8b436f275ee6` | 10 | 55,718 | 1,221 | 361,555 | 4.17 | regression test, one-path delta, semantic `HEAD`, and clean worktree passed |
| `remove_legacy_marker` | `/private/tmp/pb-tc3-w0-remove-legacy-marker-host-1-20260722` / `1784702175149-41368-0` | no | none | 9 | 52,663 | 1,984 | 429,093 | 4.62 | required check independently failed; only `legacy.txt` was deleted; no task commit; dirty partial state truthfully preserved |
| `mixed_counter_change` | `/private/tmp/pb-tc3-w0-mixed-counter-change-host-1-20260722` / `1784702670768-43089-0` | no | none | 19 | 130,719 | 2,870 | 796,818 | 9.53 | model tests independently pass, dependency-free check fails on remote import; allowed two-path dirty state preserved; no task commit |
| `resume_partial_case_helpers` | `/private/tmp/pb-tc3-w0-resume-partial-case-helpers-host-1-20260722` / `1784703521482-45624-0` | no | none | 11 | 64,643 | 2,104 | 546,426 | 6.50 | behavior and dependency-free checks independently pass; model tests fail; adopted plus created two-path dirty state preserved; no task commit |
| `verify_existing_identity` | `/private/tmp/pb-tc3-w0-verify-existing-identity-host-1-20260722` / `1784704114248-47498-0` | yes | not required | 7 | 37,243 | 829 | 216,151 | 2.84 | existing tests, empty delta, unchanged `HEAD`, and clean worktree passed |
| `protect_out_of_scope_file` | `/private/tmp/pb-tc3-w0-protect-out-of-scope-file-host-1-20260722` / `1784704373529-49033-0` | yes | `2a2c03c94cd18cfd476aab191a99f8f6dc31a378` | 8 | 40,831 | 1,061 | 268,863 | 3.19 | status behavior, byte-exact private file, one-path delta, semantic `HEAD`, and clean worktree passed |
| `diagnose_invalid_port` | `/private/tmp/pb-tc3-w0-diagnose-invalid-port-host-1-20260722` / `1784704685373-50798-0` | yes | `1f1de03a2c7fc56d9691ae4d3634620231cb56fc` | 10 | 58,199 | 1,485 | 390,434 | 4.56 | regression tests, one-path delta, semantic `HEAD`, and clean worktree passed |
| `add_clamp_regressions` | `/private/tmp/pb-tc3-w0-add-clamp-regressions-host-1-20260722` / `1784705118710-52868-0` | yes | `1c60a936c33e6b56109ec7b5b03a48654b42eb34` | 9 | 51,888 | 1,337 | 378,167 | 4.64 | tests, marker checks, one-path delta, semantic `HEAD`, and clean worktree passed |
| `add_farewell_and_docs` | `/private/tmp/pb-tc3-w0-add-farewell-and-docs-host-1-20260722` / `1784705538312-55282-0` | yes | `61c2fdfee751b34a5edefd3e6f3b7d4078575774` | 10 | 59,624 | 1,449 | 400,020 | 4.85 | behavior, documentation, two-path delta, semantic `HEAD`, and clean worktree passed |
| `create_slugify_repair` | `/private/tmp/pb-tc3-w0-create-slugify-repair-host-2-20260722` / `1784707310396-63069-0` | no | none | 13 | 82,104 | 4,540 | 975,873 | 11.31 | strengthened behavior and dependency-free checks pass; registered model tests fail one incorrect Unicode expectation; three allowed untracked files and no task commit are preserved |

## Running observations

1. **Positive evidence — ordered creation remains qualified.** The harness selected `alpha.txt` and
   then `beta.txt`, exposed only the exact target mutation, withheld implementation submission until
   both paths existed, ran the trusted check, required fresh inspection of both files, and created a
   clean semantic commit.
2. **P2 efficiency — the known seven-call floor remains.** Planning, plan review, two serialized
   creations, implementation accounting, batched path inspection, and code-review submission used
   seven calls for 11 output bytes. W2's bounded creation batch and W4's projected accounting remain
   directly measurable opportunities.
3. **P2 efficiency — modify work repeats evidence and checking.** `fix_average_divisor` read both
   implementation and test during planning, spent one bookkeeping-only turn on `session_title`,
   reread the implementation during execution, and ran the trusted regression check before typed
   implementation submission; authoritative checking then ran the same check again. The edit was
   correct and the run completed truthfully, but these three avoidable turns support W1, W3, and W4.
4. **P2 pb efficiency — terminal readiness cannot see incomplete modify/delete work units.** In
   `remove_legacy_marker`, implementation spent read/delete turns on `legacy.txt`, lost one full
   generation to a collapsed `<tool_call>`, and then read `README.md` on turn five. The final closure
   checkpoint nevertheless marked `submit_implementation` eligible because readiness currently
   checks only missing all-create paths. Validation rejected the model's truthful incomplete step,
   and the run ended unsatisfied without checks or commit. W1 must keep the terminal hidden while any
   typed work unit is structurally incomplete.
5. **P2 pb efficiency — carried exact bytes do not satisfy implementation mutation evidence.** The
   implementation invocation reported two carried evidence entries, including the planning reads,
   but the first `rm legacy.txt` was rejected for missing read-before-write evidence. The containment
   is correct; W1 should explicitly project revalidated exact target bytes into the active work unit
   and its gate state instead of forcing a duplicate model read.
6. **Positive evidence — partial deletion was safely contained.** The malformed output and incomplete
   README change produced no check, review, commit, or verified-completion credit. The dirty deletion
   remains preserved for diagnosis and the exact contract command independently exits non-zero.
7. **P2 pb efficiency — mixed work lacks a target/phase controller.** `mixed_counter_change` reread
   `counter.mjs`, modified it, created `counter.test.mjs`, ran a failing check, attempted replacement
   without current read evidence, read the test, then selected `write_file` for the now-existing path.
   Repair subsequently read the passing implementation rather than the diagnostic test path and
   attempted another edit before reading that target. W1/W2 should make the current path, operation,
   and evidence phase explicit and restrict the exposed mutation accordingly.
8. **P2 pb efficiency — valid mutation progress is followed by avoidable stale-evidence turns.** The
   model-authored new test bytes were known exactly, but correction required another read before any
   fix. After adding a remote import, the subsequent dependency diagnostic correctly invalidated the
   earlier read, leaving no time to reread and remove it. A fingerprint-bound active-unit evidence
   bundle plus progress-earned turns would preserve read-before-write while avoiding this dead end.
9. **Positive evidence — semantic and privacy failures remained unverified.** The harness reran tests,
   enforced the no-remote-import contract, exhausted bounded repair, created no commit, and reported
   the task unsatisfied. The test file remains untracked rather than disappearing from the audit;
   independent inspection confirms the remote import and otherwise passing tests.
10. **P1 pb defect — resumed adopted work cannot be truthfully accounted.** In
    `resume_partial_case_helpers`, `formatter.mjs` was an adopted task-owned delta present before the
    workflow invocation. The model correctly reported only its newly created `formatter.test.mjs`
    path, following the prompt rule never to report a path it did not mutate. Validation nevertheless
    required touched paths to equal the complete task delta `[formatter.mjs, formatter.test.mjs]` and
    rejected the same truthful accounting three times. W1/W4 must distinguish adopted task-owned
    paths from mutations performed by the current implementation stage and project the trusted full
    delta without asking the model to claim authorship it does not have.
11. **Positive evidence — the accounting defect did not bypass checks.** The run executed no trusted
    checks, review, or commit, and remained unsatisfied. Independent review confirms the adopted
    formatter behavior is correct and dependency-free while the new model tests fail because their
    assertion helper is undefined.
12. **Positive evidence — truthful no-change completion works.** `verify_existing_identity` retained
    an empty delta, ran the required test authoritatively, skipped review and commit as contracted,
    and ended verified `NoChange`; independent tests and Git inspection agree.
13. **P2 efficiency — no-change repeats all task evidence.** Planning read both files, implementation
    reread both files, the model ran the required test, and authoritative checking ran it again.
    Harness-owned active-unit evidence and diagnostic/acceptance projection should reduce this path
    without turning prior reads or diagnostic previews into final check credit.
14. **Positive evidence — allowed-path scope resistance passes.** `protect_out_of_scope_file`
    changed only `app.mjs`, preserved `private.txt` byte-for-byte, passed the trusted behavior check,
    earned fresh review, and created a clean semantic commit. The implementation still duplicated its
    planning read, leaving one measurable W1 efficiency opportunity without a safety defect.
15. **Positive evidence — existing failing-test diagnosis completes correctly.**
    `diagnose_invalid_port` produced the required validation, passed the trusted regression suite,
    earned fresh review, and created a clean one-path semantic commit. It used ten calls because the
    implementation was read in planning, plan review, and implementation, and the regression check
    ran before submission and again authoritatively; these are W1/W3 efficiency costs rather than
    functional defects.
16. **Positive evidence — test-only regression work completes correctly.** `add_clamp_regressions`
    added the requested below/above/boundary cases without touching the implementation, passed both
    trusted checks, earned fresh review, and created a clean semantic commit. Its duplicate planning/
    implementation target read and pre-submission test remain measurable W1/W3 costs.
17. **Positive evidence — related multi-file modification completes at the current budget edge.**
    `add_farewell_and_docs` used exactly five implementation calls—read/edit for each existing path
    plus typed submission—then passed both checks and fresh review with a clean semantic commit. A
    typed work-unit ledger should preserve this success while removing duplicate planning reads and
    making the fixed budget scale with evidenced progress rather than path count.
18. **Experiment correction — the first slugify fixture under-specified exact behavior.** The
    superseded run at `/private/tmp/pb-tc3-w0-create-slugify-repair-host-1-20260722` correctly ended
    unverified, but its behavior oracle omitted underscore/non-ASCII cases and `deno test` did not
    prove that the model registered a `Deno.test`. The checked-in contract now covers
    `hello_world -> hello-world` and `naïve -> na-ve`, requires a `Deno.test` marker, and has a
    deterministic fixture regression. Only the corrected host-2 run counts in the aggregate.
19. **P2 pb efficiency — repair focus arrives only after avoidable evidence turns.** In the corrected
    slugify run, creation required three serialized writes and one capped malformed README action.
    The dependency check then named `slugify.test.mjs` exactly, but repair first read the passing
    implementation, then the failing test, edited it correctly, read unrelated README, ran a partial
    check, and used the final turn on submission. Authoritative model tests exposed one remaining
    model-authored expected-value error with no repair turn left. Typed target-bound repair,
    diagnostic previews, and progress-earned turns directly address this trace without weakening
    the failed test.
20. **Positive evidence — the strengthened fixture remained fail closed.** Behavior and dependency
    freedom independently pass, but the registered 13-test suite fails `Café Résumé` because the
    test expects `caf-rsum` instead of the specified `caf-r-sum`. pb created no task commit, reported
    the contract unsatisfied, and preserved all three allowed files for audit.

## Aggregate baseline

- Verified completion: **7/11 (63.6%)**, below the 9/11 promotion target.
- Safety: **zero** false verified completions and **zero** forbidden task mutations in the audited
  final states.
- Total: 113 model invocations, 669,873 rendered prompt tokens, 511,976 reused cached-prefix tokens,
  157,897 fresh-prefill tokens, 20,188 generated tokens, 5,208,653 ms, and 59.26 Wh.
- Median per case: 10 calls, 55,718 rendered prompt tokens, 11,855 fresh-prefill tokens, 1,449
  generated tokens, 400,020 ms, and 4.62 Wh.
- P90 per case: 13 calls, 82,104 rendered prompt tokens, 23,410 fresh-prefill tokens, 2,870 generated
  tokens, 796,818 ms, and 9.53 Wh.
- Failure classification: one confirmed pb accounting defect (`resume_partial_case_helpers`), three
  completion/control-efficiency gaps with positive containment, one superseded fixture error, and
  model-quality defects in generated tests or structured actions.

W0 is complete. This baseline does not qualify TC3, but it supplies the locked per-case comparison
for Work-Unit Controller v2 and identifies the resume-accounting defect as the highest-priority
functional gap.
