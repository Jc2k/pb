# Goal-driven harness delivery audit

Captured: 2026-07-24

This audit used a live `pb harness agent` delivery as a stimulus, fixed clear pb defects, and stopped
only after a fresh prompt reached verified completion without another harness finding. Generated
visual polish was deliberately outside the contract. The harness hypothesis was narrower: pb must
bound malformed model actions, preserve truthful state, enforce an independent check and fresh
review, create the semantic commit itself, and report the outcome accurately.

## Experiment contract

The first fixture requested a dependency-free grocery-checklist website with three allowed files,
model-authored tests, a required Deno check, fresh review, semantic commit, and clean worktree. The
qualification fixture then reduced artifact size while retaining the complete strict workflow: one
allowed `slug.js`, an external behavior test, fresh inspection and code review, semantic commit, and
clean worktree. Accessibility, styling, framework choice, and unrequested product polish were not
acceptance criteria.

All runs used the release arm64 binary, FlashMoe Qwen3-Coder-Next 4-bit at temperature 0/top-k 1,
an isolated persistent scratch repository, and locally preserved event/journal evidence.

## Preserved runs

| Run | Outcome | Classification and evidence |
| --- | --- | --- |
| Grocery checklist | `step_limit`, contract unsatisfied | The model repeatedly exceeded the native mutation payload while creating `app.js`. pb rejected every truncated action, wrote no partial `app.js`, charged retries, stopped after three consecutive unparsable actions, preserved `index.html`, created no delivery commit, and reported the contract unsatisfied. This was positive containment of a model limitation. The Task transcript separately exposed a pb defect: clause extraction split `index.html`, `app.js`, and function arguments into fragments such as `js`, then falsely rejected reasonable partitions as duplicate ownership. |
| Slug module, verifier v1 | interrupted | The agent created a correct `slug.js`, but the supervisor-authored external Deno test used an invalid static template-literal import. pb surfaced the exact verifier parse error and granted no check credit. The run was stopped and preserved as an experiment error rather than asking the model to repair correct code against a broken contract. |
| Slug module, verifier v2 | `Ready`, contract satisfied, verified completed | The validated external check passed first as diagnostic feedback and then ran authoritatively after structured implementation submission. Fresh code review received a harness-owned full-file observation plus current check evidence. pb created commit `25ac865`, left a clean worktree, and reported verified completion. An independent post-run check passed and confirmed the recorded commit at `HEAD`. |

Preserved roots:

- `/private/tmp/pb-goal-grocery-20260724-c1hhHP/`
- `/private/tmp/pb-goal-slug-20260724-ZDqjsx/`
- `/private/tmp/pb-goal-slug-v2-20260724-2gFDk9/`

The successful run used six total model calls including the Task preflight, 25,569 total tokens, and
an estimated 4.06 Wh. Its Task transcript retained `slug.js` and `slugify(value)` as complete source
clauses, produced one Task, and therefore entered the ordinary Build workflow with the exact original
request.

## Fixed pb defects

### P1 — punctuation fragmented Task source clauses

- Classification: pb defect.
- Evidence: the first run's Task transcript contained `"index"`, `"html and app"`, `"js"`,
  `"Export pure addItem(items"`, and `"text)"`; both proposals were rejected because the fragment
  `js` appeared more than once.
- Impact: valid default decomposition silently fell back to one Build for ordinary prompts containing
  filenames, function arguments, or comma-delimited behavior lists.
- Disposition: fixed by `ef7af0ac`. Sentence extraction now preserves dotted paths and comma lists,
  ownership tolerates only punctuation-adjacent whitespace, the planner protocol was versioned, and
  focused plus full-suite tests pass.

### P2 — successful controls were reported as model limitations

- Classification: pb defect.
- Evidence: the successful run journal labeled active work-unit guidance, unique progress credit,
  and its passing diagnostic preview as `model_limitation` observations despite each being positive
  harness behavior.
- Impact: a clean audit looked as though it contained model failures and contradicted the documented
  observation taxonomy.
- Disposition: fixed by `e3ff6e66`. Those successful corrections now classify as
  `positive_evidence`; a failing preview remains a model limitation. Deterministic event tests and the
  full suite pass, so another expensive model run is not required to prove this reporting-only change.

## Positive evidence and limits

- Truncated native writes never produced partial target files or escaped the allowlist.
- Parse-loop, stage-step, invocation, token, and progress-credit bounds remained finite and truthful.
  The visible step ceiling may grow by one for each unique productive work unit, up to the documented
  maximum of four; failed or repeated actions earned no credit.
- Diagnostic previews did not satisfy the authoritative check gate.
- Checking, fresh review, and managed commit used current repository evidence, and journal/Git state
  agreed after completion.
- The multi-file grocery artifact did not complete because this model repeatedly ignored an explicit
  compact-payload correction. That is not evidence of a harness defect. A larger artifact or multi-Task
  completion qualification should use a separately budgeted/model-qualified corpus rather than
  weakening mutation or completion gates.

No unresolved P0-P2 pb defect remained in the final delivery path. The next useful work is broader
corpus coverage for dotted paths and code-like punctuation; another identical live slug run would
mostly remeasure the same model and is not required for this audit.
