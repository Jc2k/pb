# Verified task-completion reliability

Status: **Active; TC1/TC2 fixtures defined, native qualification pending**

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
documentation, an independent behavior check, the model test check, fresh review, semantic commit,
and a clean worktree.

Initial functional gate: one fully verified completion after TC1's first qualification. The first
efficiency envelope is 12 model invocations, 30 minutes, and 15 Wh. Exceeding an efficiency bound
does not erase a functional success, but it blocks an efficiency promotion.

### TC3 — repository task corpus

Build a 10–20 fixture offline corpus covering one-file fixes, regression tests, related multi-file
changes, failed-check diagnosis, review repair, resume, and out-of-scope mutation resistance.

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
2. derive immutable acceptance skeleton facts from the trusted contract instead of asking the model
   to reproduce them;
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

## Evidence log

| Date | Gate | Evidence | Result | Next action |
| --- | --- | --- | --- | --- |
| 2026-07-21 | Fixture definition | TC1/TC2 tasks, contracts, and independent review procedure | Pending native execution | Build current release and run TC1 in a new preserved scratch root |
| 2026-07-21 | Reporting | `scripts/summarize-harness-completion.ts` and synthetic tests | Machine-readable harness claims and efficiency metrics remain distinct from independent verification | Validate against preserved field runs and use for TC1 |
| 2026-07-21 | TC1 native qualification 1 | scratch `/private/tmp/pb-tc1-ordered-create-1-20260721`, run `1784667572614-75379-0`, commit `f4da7a5a925f82a6c129850f284ff2731d8568d9` | Verified completion; exact check independently passed; clean worktree; 7 calls, 561,696 ms, 7.24 Wh; exposed two journal diagnostic defects | Fix strict planned-check audit and work-unit classification, then repeat TC1 with the locked settings |

## Completion audit

- [x] TC1 and TC2 tasks state objective artifact requirements.
- [x] Trusted contracts restrict paths and require checks, fresh review, semantic commit, and a clean
  worktree.
- [x] TC2 has an independent behavior check separate from model-authored tests.
- [ ] Current release is rebuilt from the source under evaluation.
- [x] TC1 native qualification passes and its journal, events, Git state, and check are reviewed.
- [ ] TC1 repeats successfully 3/3 with locked settings.
- [ ] TC2 reaches one independently verified completion.
- [ ] TC3 corpus and aggregate report exist.
- [ ] Highest-value evidenced controller and efficiency gaps are fixed and regression-tested.
- [ ] External same-model comparison is recorded.
