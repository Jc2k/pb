# Verified task-completion reliability

Status: **Active; Work-Unit Controller v2 qualified for TC1/TC2 and targeted TC3 conversions; full
W6 TC3 rerun pending**

This record tracks pb's move from protocol-safe local agent runs to repeatable, externally verified
coding-task completion. It complements the [small-model reliability plan](small-model-agent-reliability-plan.md),
which established safe control behavior, and the
[Qwen3-Coder-Next performance follow-on](qwen3-coder-next-agent-follow-on.md), which owns the current
native model/runtime qualification.

## Outcome

pb should complete useful local coding tasks through accepted planning, fresh plan review, bounded
implementation, trusted checks, fresh code review, managed semantic commit, and a clean worktree.
Every claimed success must have an explicit trusted contract and independently reproducible evidence.

The primary metric is **verified task completion**, defined as all of:

1. the harness exits zero with `contract_status=satisfied` and `verified_completed=true`;
2. every required mutation, path restriction, named check, fresh-review read, and review check is
   satisfied against the final content fingerprint;
3. the recorded managed commit is the scratch workspace `HEAD` and is semantic;
4. the worktree is clean and contains no forbidden task delta; and
5. an independent supervisor reruns the contract checks successfully from that commit.

Protocol safety, task completion, artifact quality, and efficiency remain separate measurements. A
truthfully contained failure is positive harness evidence but never counts as task completion.

## Invariants

- No completion, check, review, content-fingerprint, allowed-path, or commit gate is weakened to raise
  the completion rate.
- Every model run uses a fresh or explicitly resumed preserved scratch root. Failed and superseded
  runs are retained.
- Locked comparisons keep model, model digest, template, sampling, context, stage limits, task,
  contract, and host conditions fixed.
- The agent may read trusted checks but cannot mutate their definition through the task delta.
- No network access or automatic cloud escalation is introduced.
- Deterministic reproductions precede expensive model reruns for an identified pb defect.
- The project-local `.pb/` directory is user state and is never benchmark scratch.

## Failure classification

Every unsuccessful run is classified before changes are proposed:

- **pb defect:** control, tool, persistence, check, review, commit, reporting, recovery, resource, or
  termination behavior violated the explicit contract;
- **model limitation:** the model produced weak code, an invalid action, drifted instructions, or
  failed to finish while pb bounded and reported it truthfully;
- **experiment error:** the fixture, prompt, profile, quoting, model, or budget could not test the
  intended hypothesis; or
- **positive evidence:** pb correctly rejected, contained, persisted, or reported unsafe or
  incomplete output.

## Qualification ladder

### TC1 — ordered creation

The checked-in `fixtures/harness-task-completion` TC1 fixture requires two exact small files, one
trusted check, fresh review of both paths, a semantic commit, and a clean worktree.

Hypothesis: the native Qwen runner follows the accepted all-create plan one harness-selected missing
path at a time, advances only after an atomic complete write, and reaches verified completion.

Promotion gates:

- first run: complete native stage/check/review/commit qualification;
- locked repeatability: 3/3 successful runs with unchanged settings;
- safety: zero out-of-order mutation, false check/review credit, false completion, or forbidden path;
- report functional and efficiency outcomes independently.

### TC2 — useful coding

The TC2 fixture requires a dependency-free slugify implementation, model-authored tests,
documentation, an independent behavior check, an explicit no-remote-import check, the model test
check, fresh review, semantic commit, and a clean worktree.

Initial functional gate: one fully verified completion after TC1's first qualification. The first
efficiency envelope is 12 model invocations, 30 minutes, and 15 Wh. Exceeding an efficiency bound
does not erase a functional success, but it blocks an efficiency promotion.

### TC3 — repository task corpus

The checked-in 11-case offline corpus covers ordered creation, repair after a failed check, one-file
fixes, regression tests, related multi-file changes, failed-check diagnosis, delete/modify work,
resume adoption, mixed create/modify work, truthful no-change completion, and out-of-scope mutation
resistance. Its manifest, safe scratch preparation, trusted contracts, and locked per-case bounds are
deterministically validated. The audited W0 native baseline is 7/11 verified with zero false
completion or forbidden mutation; see the [TC3 baseline](benchmarks/task-completion-tc3-baseline.md).

Initial promotion target:

- at least 80% verified completion on small tasks;
- zero false verified completions and forbidden mutations;
- independently reproducible checks from every successful commit; and
- median and tail wall time, energy, invocations, prompt/generated tokens, and repair cycles reported.

### TC4 — external comparison

After TC3, adapt a recognized edit corpus and compare pb with another local agent on the same model,
hardware, context, sampling, offline inputs, and task subset. pb success remains the checked-commit
contract, not the presence of a plausible patch.

## Improvement order

Work is evidence-gated in this order:

1. qualify the shipped next-missing-path controller with TC1;
2. **Implemented:** derive immutable required-check skeleton facts from the trusted contract,
   project them into the submitted plan, and recompute its digest instead of asking the model to
   reproduce them;
3. generalize deterministic work units to mixed create/modify/delete plans;
4. grant bounded additional work only after a content or evidence transition;
5. run declared cheap diagnostics after completed work units without granting final check credit;
6. make recovery prompt/tool prefixes byte-stable and measure fresh versus cached prefill;
7. add hash-bound task-owned staging and atomic publication only if complete bounded scaffolds and
   edits still cannot express required files;
8. compare supported local model tiers by verified completions per hour and Wh; and
9. consider protocol fine-tuning only after the preserved trace corpus identifies stable model
   failure classes.

Larger context windows, higher unconditional output caps, acceptance-gate relaxation, checkpoint-
specific runtime shortcuts, and silent hosted-model fallback are not promotion strategies.

## Work-Unit Controller v2 milestones

The existing small-model reliability plan already shipped bounded context, focused repository and
review evidence, outcome-aware loop recovery, structured tool errors, deterministic closure, and
authorized-subset tool exposure. This follow-on targets the remaining task-completion gap without
reimplementing those controls.

| ID | Priority | Status | Deliverable | Promotion proof |
| --- | --- | --- | --- | --- |
| W0 | P0 | complete | Execute and independently audit the locked 11-case TC3 native baseline | 7/11 verified; zero false completion/forbidden mutation; per-case audit plus aggregate invocation, token, cache, wall, and energy evidence published |
| W1 | P0 | implemented | Persist a typed harness-owned work-unit ledger for create, modify, delete, adopted/resumed, and no-change work | Ordered-state, checkpoint round-trip, fingerprint, adopted-work, and target fixtures pass |
| W2 | P0 | implemented | Expose stable target-bound workflow mutations and bounded atomic batches for independent creations | Harness-bound path insertion, operation-only tool exposure, batch validation/rollback, and malformed-batch fixtures pass |
| W3 | P1 | implemented | Grant bounded work only after unique progress and run explicitly eligible cheap diagnostics without final evidence credit | Per-unit/max-four progress credit and diagnostic-preview isolation/focus fixtures pass |
| W4 | P1 | implemented | Project trusted implementation and review skeleton fields while retaining model-authored completion and review judgments | Resume/adopted projection and existing artifact validators pass deterministic tests |
| W5 | P2 | qualified | Make work-unit prompt and tool prefixes byte-stable and measure cacheable versus fresh prefill | Cross-target tool schema identity passes; the locked TC2 series reports rendered, cached-prefix, and fresh-prefill tokens separately |
| W6 | P2 | partial | Qualify TC1, TC2, TC3, and supported local model tiers | TC1 is 3/3 at 6 calls; TC2 is 3/3 at 7 calls with the efficiency target passed; two targeted W0 failure conversions reach the 9/11 floor only when combined with unchanged W0 successes; a final-source 11-case rerun and model-tier comparison remain |

### W1 target state

The accepted plan, task baseline, current content snapshot, and trusted diagnostics produce a
checkpointed work-unit ledger. Each unit records its plan step, operation, path, baseline/current
fingerprints, and one of `evidence_needed`, `mutation_ready`, `structurally_complete`,
`diagnostic_failed`, `diagnostic_repair_ready`, or `blocked_for_replan`. Structural completion
records only the required filesystem transition; it never substitutes for configured checks or
fresh review. Diagnostic failure invalidates older reads of the exact named path before bounded
replacement/edit repair; a missing target requires replanning.

Modify and delete work requires exact fingerprint-bound target evidence. An adopted task-owned
delta may satisfy structural progress without being rewritten. No-change tasks have an empty
mutation ledger and retain their existing truthful no-change gates.

### W2 target state

Implementation and repair expose only the operation appropriate to the active unit. The model does
not copy a repository path or expected fingerprint into a scoped mutation; pb resolves both from
the checkpointed ledger and records them in the durable event. Independent creation units may be
submitted as a bounded atomic batch after every member and the combined mutation payload validate.
No member executes when any member is invalid or truncated.

### W3 target state

A distinct content/evidence transition may earn at most one additional implementation or repair
turn for its work unit, with at most four earned turns for the stage. Failed, rejected, no-op,
repeated, cached, and bookkeeping-only actions earn none. Existing global invocation and generated-
token budgets remain authoritative.

Only checks explicitly marked as diagnostic-eligible may run after the current work-unit queue is
structurally complete. Their results are fingerprint-bound diagnostic feedback, not selected-check
receipts. Passing checks run again through the authoritative checking stage; failing output may
focus a repair unit only when it names an exact task path.

### W4 target state

For implementation submission, pb supplies trusted plan identity/digest, current fingerprint, and
actual touched paths. The model still names the completed accepted steps, gives a semantic summary,
and proposes the commit subject. For review, pb supplies the checked fingerprint and successful
check identifiers while the fresh model remains responsible for substantive assessments, findings,
and verdict. Projection never synthesizes a passing completion or review.

### W5 and W6 measurement

Work-unit targets and fingerprints belong in the dynamic prompt suffix, not rewritten tool schemas.
Schema ordering, common guidance, and cache identities must be canonical and versioned. Report
rendered, cacheable, and fresh-prefill tokens separately.

The final locked comparison retains the current task contracts and independent audit procedure.
TC2 must also improve wall time and energy by at least 25% from its qualified 1,167,352 ms and
12.90 Wh run. Efficiency never compensates for a safety or functional regression.

## Evidence log

| Date | Gate | Evidence | Result | Next action |
| --- | --- | --- | --- | --- |
| 2026-07-21 | Fixture definition | TC1/TC2 tasks, contracts, and independent review procedure | Pending native execution | Build current release and run TC1 in a new preserved scratch root |
| 2026-07-21 | Reporting | `scripts/summarize-harness-completion.ts` and synthetic tests | Machine-readable harness claims and efficiency metrics remain distinct from independent verification | Validate against preserved field runs and use for TC1 |
| 2026-07-21 | TC1 native qualification 1 | scratch `/private/tmp/pb-tc1-ordered-create-1-20260721`, run `1784667572614-75379-0`, commit `f4da7a5a925f82a6c129850f284ff2731d8568d9` | Verified completion; exact check independently passed; clean worktree; 7 calls, 561,696 ms, 7.24 Wh; exposed two journal diagnostic defects | Fix strict planned-check audit and work-unit classification, then repeat TC1 with the locked settings |
| 2026-07-21 | TC1 locked post-fix series | runs `1784668516949-33851-0`, `1784668983662-35797-0`, and `1784669344810-37478-0`; [TC1 report](benchmarks/task-completion-tc1.md) | 3/3 verified; zero rejected or forbidden actions; median 317,781 ms and 3.97 Wh; reporting fix qualified | Retain TC1 as a regression gate and run TC2 |
| 2026-07-21 | TC2 initial run | scratch `/private/tmp/pb-tc2-slugify-1-20260721`, run `1784669878454-39823-0` | Both declared checks passed, but implementation exhausted six steps without typed submission; unsatisfied and unverified; fixture also failed to enforce dependency freedom | Reserve the last ready implementation turn for closure and add a trusted dependency gate |
| 2026-07-21 | TC2 useful-coding qualification | scratch `/private/tmp/pb-tc2-slugify-3-20260721`, run `1784671564495-58382-0`, commit `7fd899ce771857bd39a3c6d0ec77e85e6b230607`; [TC2 report](benchmarks/task-completion-tc2.md) | Verified completion after one real repair; all three checks independently passed; clean three-path semantic commit; 16 calls, 1,167,352 ms, 12.90 Wh | Retain as a repair regression; expand TC3 and reduce model calls without weakening gates |
| 2026-07-21 | Trusted acceptance projection | unit projection/digest test, scripted planning-stage test, exact-path repair-focus test, and complete Rust quality suite | Required contract check IDs are projected and rehashed before fresh review; bounded failed-check diagnostics focus repair without granting evidence or scope | Measure the native invocation and token effect in TC3 rather than claiming an efficiency win from deterministic tests |
| 2026-07-21 | TC3 candidate corpus | `fixtures/harness-task-completion/corpus.json`, `scripts/run-harness-task-corpus.ts`, Deno preparation tests, Rust contract normalization test | 11 safe offline cases across 11 categories; reproducible seeded and resumed scratch preparation; aggregate native results not yet claimed | Execute preserved cases, audit every claimed success, and publish aggregate functional and efficiency results |
| 2026-07-22 | W0 TC3 baseline | 11 corrected-contract preserved native runs; [aggregate report](benchmarks/task-completion-tc3-baseline.md) | 7/11 verified, zero false completion/forbidden mutation; median 10 calls, 400,020 ms, 4.62 Wh; one confirmed resumed-work accounting defect | Qualify Work-Unit Controller v2 against the same cases |
| 2026-07-22 | W1–W5 deterministic implementation | typed checkpoint ledger, target-bound/batched creates, unique progress credits, diagnostic previews, trusted submission projection, stable-schema and reporting fixtures | Controller path implemented without weakening authoritative check/review/commit gates; full quality and native W6 qualification pending | Run full quality suite, rebuild release, then execute locked TC1/TC2/TC3 comparisons |
| 2026-07-22 | W6 TC1 controller series | three preserved `pb-wucv2-tc1-fixed-*` runs; [qualification report](benchmarks/task-completion-work-unit-v2.md) | 3/3 verified at exactly 6 calls; exact bytes, checks, reviews, semantic commits, and clean states independently audited | Retain as the ordered-create regression gate |
| 2026-07-22 | W6 TC2 controller series | three preserved `pb-wucv2-tc2-final-*` runs; [qualification report](benchmarks/task-completion-work-unit-v2.md) | 3/3 verified at 7 calls; median 586,630 ms and 6.11 Wh, improving the qualified baseline by 49.7% wall time and 52.6% energy | Retain the literal local expectation as advisory-only active-unit guidance |
| 2026-07-22 | W6 targeted TC3 conversions | adopted resume run `1784729407708-57538-0`; final delete/modify run `1784732727735-57336-0`; [qualification report](benchmarks/task-completion-work-unit-v2.md) | Both prior W0 failures verified with clean semantic commits. Qualification exposed and fixed carried-fingerprint, tracked-deletion advancement, and post-commit content-identity defects | Run all 11 cases on one final source before claiming a new aggregate |

## Completion audit

- [x] TC1 and TC2 tasks state objective artifact requirements.
- [x] Trusted contracts restrict paths and require checks, fresh review, semantic commit, and a clean
  worktree.
- [x] TC2 has an independent behavior check separate from model-authored tests.
- [x] Current release is rebuilt from the source under evaluation.
- [x] TC1 native qualification passes and its journal, events, Git state, and check are reviewed.
- [x] TC1 repeats successfully 3/3 with locked settings.
- [x] TC2 reaches one independently verified completion.
- [x] TC3 candidate corpus and single-case runner exist.
- [x] TC3 aggregate native-model report exists.
- [x] Highest-value evidenced acceptance-projection gap is fixed and regression-tested.
- [x] Work-Unit Controller v2 passes locked TC1 and TC2 native repeatability gates.
- [x] Adopted/resumed and delete/modify W0 failures have independently audited verified conversions.
- [ ] All 11 TC3 cases are rerun on one final source for a current aggregate.
- [ ] External same-model comparison is recorded.
