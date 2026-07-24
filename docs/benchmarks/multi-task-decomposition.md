# Task decomposition feasibility probe

Captured: 2026-07-23

Plan: [Task decomposition workflow](../multi-task-workflow-plan.md)

The resulting design calls each high-level queue entry a Task. Child implementation planning is
reserved until each Task activates.

## Decision

Model-assisted Task decomposition is viable only as a proposal-and-review stage. It is not
reliable enough to let a model define executable task budgets or start child workflows directly.

The 7B and 14B models produced useful six-Task dependency graphs from one request brief, but neither
produced truthful aggregate budgets. The 7B plan also violated the per-task ceilings and introduced
vague testing/refinement catch-alls. The 14B plan kept every task below its individual ceiling, but
misordered storage migration after the service that needed it and misstated all three aggregate
budget dimensions. Qwen3 4B spent both bounded generations reasoning about the constraints and never
returned a final plan.

The implementation consequence is narrow:

- models may propose task boundaries, dependencies, acceptance facts, and qualitative effort;
- Rust validates the graph and requirement coverage;
- Rust compiles exact task budgets from policy presets and remaining parent budget;
- fresh plan review must challenge task size, ordering, and acceptance quality; and
- no task starts until the accepted, controller-projected artifact passes deterministic validation.

This is a model limitation and product-design constraint, not evidence of a pb safety defect. The
harness bounded the failed 4B run, preserved all outputs, made no workspace mutation, and reported
the step-limit outcome truthfully.

## Experiment contract

The self-contained fixture asked a planning profile to decompose a durable CSV-import request for an
existing Rust service and React UI. The request required persisted queued/running/completed/failed/
cancelled state, restart-safe idempotent resume, compatible start/status/cancel APIs, progress/error/
cancel/retry UI, schema migration, Rust and web tests, architecture and user documentation, and no
deployment work.

The numeric arm required three to six ordered tasks. Every task needed an ID, dependencies,
objective, deliverables, observable acceptance checks, and exact invocation/token/wall-time budgets.
Individual ceilings were 7 invocations, 6,000 generated tokens, and 35 minutes. Aggregate ceilings
were 30 invocations, 24,000 generated tokens, and 150 minutes. Declared totals had to equal the task
sums, and vague final integration/testing catch-alls were forbidden.

A follow-up 7B arm removed numeric arithmetic. It requested `small` or `medium` effort while saying
that the controller would assign exact budgets. It also required behavior-owning tasks to carry their
tests and documentation instead of adding final testing, integration, review, or documentation
catch-alls.

Every run used:

| Field | Value |
| --- | --- |
| Profile / intent | `plan` / `discuss` |
| Context | 8,192 tokens |
| Generation | 1,536 maximum new tokens |
| Visible steps | 1, with ordinary bounded truncation recovery |
| Sampling | temperature 0, top-k 1, seed 0 |
| Backend | llama.cpp CPU, `--gpu-layers 0` |
| Workspace | empty isolated harness repository; no mutation requested or observed |

This was one feasibility trial per arm, not a stability qualification.

## Results

| Arm | Terminal result | Structure | Task-size result | Budget result | Planning defects |
| --- | --- | --- | --- | --- | --- |
| Qwen3 4B, numeric | step limit after two 1,536-token generations | no final artifact | not scorable | not scorable | repeated constraint analysis never converged to output |
| Qwen2.5-Coder 7B, numeric | final in one call, 834 generated tokens | six-node acyclic graph | failed: one task requested 15,000 tokens and 60 minutes | failed: task sums were 20 invocations / 36,000 tokens / 150 minutes, declared as 21 / 45,000 / 150 | separate test catch-all, vague review/refine task, weak observable checks |
| Qwen2.5-Coder 14B, numeric | final in one call, 961 generated tokens | six-node acyclic graph | all declared task caps fit, though the core service task remained broad | failed: task sums were 34 invocations / 20,000 tokens / 140 minutes, declared as 30 / 24,000 / 150 | migration depended on the service instead of preceding it; several checks were subjective |
| Qwen2.5-Coder 7B, qualitative | final in one call, 563 generated tokens | six-node acyclic graph | plausible small/medium labels | numeric allocation correctly omitted | ignored the colocated-test instruction, repeated the migration ordering error, omitted restart and user-documentation coverage |

Removing arithmetic reduced output cost but did not fix semantic ordering, coverage, or catch-all
behavior. The product should therefore remove numeric allocation from the model contract without
mistaking that simplification for sufficient plan quality.

## Implementation qualification attempts

After implementing the typed proposal/review protocol, deterministic compiler, two-attempt limit,
and fail-closed recovery result, five additional live runs exercised the actual production entry
point. These are diagnostic attempts, not a completed qualification matrix. Each rejection happened
before a child Build or Goal began, and every workspace retained only its harness initialization
commit.

| Model / request | Template / policy | Calls / wall time | Terminal controller result |
| --- | --- | --- | --- |
| Qwen2.5-Coder 7B, cross-stack | v1 / 6 Tasks | 2 / 231 s | rejected: second proposal contained 9 Tasks (first contained 7) |
| Qwen2.5-Coder 7B, cross-stack | v2 / 8 Tasks | 2 / 216 s | rejected: requirement `req6` remained uncovered |
| Qwen2.5-Coder 14B, storage/API | v2 / 6 Tasks | 3 / 422 s | reviewer requested revision; replacement contained 8 Tasks |
| Qwen2.5-Coder 14B, storage/API | v2 / 8 Tasks | 3 / 416 s | reviewer requested revision; replacement exceeded aggregate controller budget |
| Qwen3 8B, storage/API | v2 / 8 Tasks | 2 / 298 s | rejected: replacement used unknown field `scope_hits` instead of `scope_hints` |

The v2 template asks for one to the policy maximum Tasks and explicitly permits combining closely
coupled behavior. Raising the default count ceiling from six to eight removed an artificial barrier
for smaller-model-sized cross-stack work, but did not cure coverage, aggregate sizing, or schema
conformance. The reviewer path also increased latency substantially for the 14B runs. None of these
models is promoted: the embedded qualification catalog remains empty, so ordinary Build dispatch is
unchanged in this release.

## Schema-constrained follow-up

Captured: 2026-07-24

The production harness path was then exercised while the planner and critic were constrained at
token selection time to exact controller-owned JSON schemas. llama.cpp used LLGuidance generated
from the JSON schema; the same schemas compiled through FlashMoe's native constraint validator.
This eliminated malformed dictionaries, misspelled keys, unknown IDs, model-authored budgets, and
invalid enum values as a class of trial failure. It did not make semantically weak plans useful by
itself.

The controller and protocol were tightened only in response to preserved failures:

- pb now retains the verbatim objective and supplies the only request evidence the model may select;
- punctuation-delimited compound clauses become separate reviewable facts;
- decomposition-wide testing, documentation, ordering, and sizing constraints are attached to
  every behavior-owning Task by Rust rather than copied by the model;
- every Task has separate outcome, test, and documentation facts;
- a multi-Task Build may own at most two behavioral clauses and needs an outcome and test fact for
  each;
- standalone testing/documentation/validation catch-alls and constraint-only Tasks are rejected;
- array order, IDs, dependencies, effort, and numeric budgets remain controller-owned; and
- the critic must assess every exact source clause once, then complete all six aggregate audits.

All llama.cpp trials below used temperature 0, top-k 1, seed 0, an 8,192-token context, Metal
offload, and the same storage/API request. Temporary qualification records existed only in the
trial builds and were removed afterward.

| Model / protocol | Calls / generated / wall | Controller result | Manual audit |
| --- | --- | --- | --- |
| Qwen2.5-Coder 14B, v8 | 2 / 1,188 / 64 s | accepted four Tasks | false pass: invented a new endpoint inside the existing endpoint and claimed missing per-Task documentation was present |
| Qwen3-Coder-Next GGUF, v8 | 6 / 8,488 / 294 s | rejected after three attempts | grounded final critique correctly found missing restart-idempotency detail and test/documentation catch-all ownership |
| Qwen2.5-Coder 14B, v9 | 2 / 2,677 / 132 s | accepted eight Tasks | false pass: standalone documentation/testing Tasks and reversed storage/consumer ordering |
| Qwen2.5-Coder 14B, v10 | 3 / 2,829 / 116 s | rejected after three attempts | safe rejection; model did not converge while copying a decomposition-wide clause |
| Qwen2.5-Coder 14B, v11 | 2 / 1,487 / 397 s debug wall | accepted three Tasks | false pass: schema, compatibility, and restart idempotency shared one small Task without restart-specific acceptance or tests |
| Qwen2.5-Coder 14B, v12 | 4 / 4,093 / 264 s | accepted five Tasks on the second attempt | first manually useful result: bounded schema, compatibility, restart-idempotency, existing-endpoint health, and cutover Tasks with owned tests/docs and correct prerequisite order |

The v11 debug wall time includes unoptimized full-model hashing and is not a production latency
measurement. The v12 record targets model SHA-256
`c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940`, template SHA-256
`64a2bea62f0933362cc7f30dee655bfe04c33eb21bc61732242416ab822770b0`, and protocol SHA-256
`fd8200c26f33b8b610367f1982ef687c3354d77fa2fb0be0821575a99b4a92d3`.

### Follow-up decision

The result establishes technical viability for schema-constrained Task planning and one useful
storage/API decomposition. It does not qualify a model. Earlier 14B accepted-plan false positives
and Qwen3-Coder-Next convergence failures show that structural validity and a self-reported critic
pass are not sufficient evidence. The embedded qualification catalog therefore remains empty.
Promotion still requires the repeatable three-shape matrix and 95% useful-boundary target below,
with manual scoring of every accepted plan and zero accepted invalid plans.

## Classification

### P2 — Exact budget arithmetic is not a model-owned control boundary

- Classification: model limitation.
- Evidence: both completed numeric arms declared totals different from the sums; the 7B arm also
  exceeded individual and aggregate ceilings.
- Impact: accepting model-authored limits could overspend the request or strand later Tasks.
- Disposition: planned controller-owned budget compilation and exact aggregate validation.
- Recommendation: accept only qualitative effort hints; persist exact controller-projected budgets
  in the reviewed plan digest.

### P2 — Plausible task lists still contain unsafe execution order and oversized scopes

- Classification: model limitation.
- Evidence: both 14B and qualitative 7B placed migration after dependent service work; 7B created
  forbidden catch-all tasks; core service tasks combined persistence, lifecycle, restart, and
  idempotency.
- Impact: a smaller executor could receive a task that is blocked by a missing predecessor or too
  broad to finish inside its allowance.
- Disposition: planned deterministic DAG checks plus fresh Task-plan review and bounded revision.
- Recommendation: require explicit requirement/acceptance traceability and scope hints, then reject
  or revise tasks that do not leave a buildable checkpoint.

### P2 — Qwen3 4B does not reliably close the planning artifact

- Classification: model limitation.
- Evidence: the initial generation and thinking-off recovery each reached 1,536 tokens without a
  valid final artifact.
- Impact: 4B cannot be the unattended default for this planning stage on current evidence.
- Disposition: keep explicit-only until the typed Task corpus passes; do not add a hidden stronger
  model escalation.
- Recommendation: expose one compact terminal schema, retain bounded truncation recovery, and stop
  truthfully when the artifact is absent.

### Positive evidence — Failed planning remained bounded and non-mutating

- Classification: positive evidence.
- Evidence: all four scratch repositories retained only the harness initialization commit; the 4B
  arm terminated at the visible step cap and every completed arm remained an unverified discussion
  final.
- Impact: weak planning did not become mutation, workflow progress, or a completion claim.
- Disposition: preserve the same authority boundary in the Task-planning stage.

## Preserved evidence

The scratch roots contain cumulative and immutable per-run events, journals, run indexes, and Git
workspaces:

- `/private/tmp/pb-decomposition-20260723/4b/`
- `/private/tmp/pb-decomposition-20260723/7b/`
- `/private/tmp/pb-decomposition-20260723/14b/`
- `/private/tmp/pb-decomposition-20260723/7b-controller-owned-budget/`
- `/private/tmp/pb-task-qualification-7b-cross-1/`
- `/private/tmp/pb-task-qualification-7b-cross-v2-1/`
- `/private/tmp/pb-task-qualification-14b-storage-v2-1/`
- `/private/tmp/pb-task-qualification-14b-storage-v2-2/`
- `/private/tmp/pb-task-qualification-qwen3-8b-storage-v2-1/`
- `/private/tmp/pb-task-schema-q25-14b-storage-v8-1/`
- `/private/tmp/pb-task-schema-q3next-storage-v8-1/`
- `/private/tmp/pb-task-schema-q25-14b-storage-v9-1/`
- `/private/tmp/pb-task-schema-q25-14b-storage-v10-1/`
- `/private/tmp/pb-task-schema-q25-14b-storage-v11-1/`
- `/private/tmp/pb-task-schema-q25-14b-storage-v12-1/`

The first 4B scratch run also records an experiment-environment lock failure before the bounded
host-authorized rerun. That setup failure is separate from the completed model result.

## Qualification gap

The typed runtime controls are shipped and the checked-in
`fixtures/task-decomposition/corpus.json` locks the deterministic, semantic-review, qualification,
and attempt-limit cases. `deno task test:task-decomposition` validates its schema and required
coverage.

Automatic rollout still requires a complete repeatable qualification run across at least three
request shapes: a cross-stack feature, a storage/API migration with rollback, and a scoped
multi-component refactor, with enough independent trials to establish the plan's 95% useful-boundary
target. Promotion requires zero accepted invalid graphs or budget overflows, zero false multi-Task
completion, and bounded convergence to an accepted or truthfully rejected plan within two attempts.
The exact model bytes, backend, planner template, protocol, and evidence digest must then be added to
the embedded catalog in source control; repository configuration cannot supply or override it.
