# Task decomposition workflow plan

Status: design record; not shipped

Assessment: [2026-07-23 decomposition feasibility probe](benchmarks/multi-task-decomposition.md)

## Decision summary

Add one high-level planning pass that turns the user's prompt into a reviewed `TaskPlanArtifact`.
The plan contains one or more outcome-shaped Tasks with dependencies, acceptance facts, execution
kind, and separate budgets. Each Task is either `build` or `goal`.

Only the active Task receives an implementation plan. When a Build Task activates, its Task
objective becomes the input to the existing Build planner, which decomposes it into the concrete
repository changes recorded in its ordinary `PlanArtifact`. After that Build or Goal has delivered,
passed its gates, and created its managed commit or commit range, pb reconciles that exact repository
state and starts planning the next queued Task as an independent request. Future Tasks therefore do
not carry stale path-level plans made before their predecessors changed the repository.

This is additive. Existing `PlanArtifact`, `PlanStep`, `WorkUnit`, `WorkflowRun`, `WorkspaceTask`,
`run_task`, Goal milestones, workflow stages, tools, events, and existing UI names keep their
current meaning and spelling. The new feature adds only Task-plan/Task-queue types and a conditional
multi-Task progress surface.

Model-generated decomposition is advisory. The feasibility probe found useful 7B/14B graphs but no
controller-safe one-shot plan: exact totals were false, dependency mistakes survived, Tasks were
sometimes oversized, and 4B did not close an artifact. The controller must own numeric allocation,
graph validation, state transitions, accounting, and Task activation. A fresh model critic may
recommend revision but cannot waive those checks.

Task count decides whether the new orchestration becomes visible:

```text
User prompt → high-level TaskPlanArtifact
             ├─ one Task  → existing Build or Goal path directly
             └─ 2+ Tasks  → MultiTaskRun → queued TaskRequest
                                           ├─ build → existing WorkflowRun
                                           │           └─ PlanArtifact (actual changes)
                                           └─ goal  → existing GoalRun
                                                       └─ existing milestones/child workflows
```

- A one-Task Build plan unwraps into the existing Build workflow, with no Task-list wrapper or new
  progress UI. A one-Task Goal plan likewise opens the existing Goal experience directly.
- A multi-Task plan owns one branch and runs Tasks sequentially. The active Task is delegated
  as an ordinary independent Build or Goal request. The Task list shows only aggregate progress.
- Build keeps its current local Ready boundary. Goal keeps its current approval, milestones,
  continuation, evidence, budget, pause/amendment, and completion semantics.
- A Goal Task is appropriate when that Task's outcome needs the existing Goal engine's several
  bounded Builds or evidence-driven replanning. It receives a Goal budget inside the parent Task
  budget; the surrounding multi-Task run does not rename or replace Goal milestones.

No Task can publish, push, open a pull request, increase authority, allocate more budget, or mark a
Goal complete.

## Existing baseline

The shipped implementation has the Build and Goal engines needed to execute Tasks, but not the
high-level decomposition and multi-Task queue:

- `PlanArtifact.steps` are ordered plan facts inside one `WorkflowRun`.
- `WorkUnitLedger` compiles those plan steps into one active path-bound mutation unit at a time.
- `WorkflowLimits` and `WorkflowCounters` apply across the complete strict workflow, not per plan
  step.
- `GoalRun` already supplies the complete Goal engine, including milestones, child workflows,
  approval, persistence, evidence, budgets, and completion.
- Build and Goal already share strict workflow authority, evidence freshness, managed commits, and
  the local Ready boundary. Multi-Task orchestration must dispatch into these controls rather than
  duplicate or alter them.

## Product contract

### Terms

- **Task plan** — the new high-level decomposition of the user's prompt.
- **Task** — one dependency-aware request with its own execution kind, budget, state, acceptance
  facts, and committed result.
- **Multi-Task run** — the new durable queue/controller created only when the Task plan contains
  more than one Task.
- **Build Task** — a Task delivered by one strict workflow planned just-in-time.
- **Goal Task** — a Task delivered by one bounded Goal that may coordinate several strict Builds;
  recursively decomposing that Goal into another Task plan is excluded from version one.
- **Plan step** — the existing ordered change inside a Build's accepted strict workflow plan.
- **Work unit** — the existing controller-selected path operation inside implementation or repair.
- **Goal criterion** — the existing completion claim inside a Goal Task. It retains its current Goal
  meaning and is not a Task boundary.

Use `task` consistently in new model schemas, prompts, events, API payloads, persistence, and UI
copy. Use `planned_task` or `TaskSpec` where code needs to distinguish the new concept from an
existing type. `PlanStep`, `WorkUnit`, configured `WorkspaceTask`, workflow stage/step, model turn,
and all existing event/UI names keep their current distinct names.

### Activation

Before repository delivery planning, a bounded high-level planner submits a `TaskPlanArtifact` with
one or more coarse Tasks. Submitting it grants no mutation or Goal authority.

- One Build Task dispatches the Task objective into the existing Build workflow. That workflow then
  inspects the current repository and creates its ordinary `PlanArtifact`: the concrete paths,
  changes, checks, and delivery plan for that part of the high-level plan. It then proceeds exactly
  as it does today.
- One Goal Task dispatches the Task objective, criteria, continuation policy, and budget into the
  existing Goal creation/approval flow. Each strict child workflow selected by Goal performs the
  same existing Build planning.
- Two or more Tasks create a durable `MultiTaskRun`; its Task list becomes visible and Task 1 starts
  automatically as soon as deterministic validation and review accept the plan. If Task 1 is a Goal
  Task, it still stops at the existing Goal approval boundary.

The controller accepts the Task plan only within the original workdir, capability, publication, and
aggregate budget envelope. A planner/model/template qualification record determines whether the
planner may choose `goal` automatically. An unqualified planner's schema permits only `build`; the
controller rejects rather than silently coercing an unauthorized Goal proposal. Explicitly
user-selected Goal mode keeps its current behavior. Activating any Goal Task still follows the
existing Goal authority and approval contract.

### Version-one scheduling

Execute one Task at a time. A DAG is still valuable for validation, amendment, visibility, and future
safe scheduling, but shared repository and Git-control state make parallel mutation unnecessary in
the first version.

A Task becomes runnable only when every dependency is `committed` or `no_change`. At activation, pb
turns the Task specification into a new independent request bound to the current HEAD, content,
authority, remaining budget, and predecessor evidence. A Build Task then enters ordinary workflow
planning and decomposes its assigned outcome into actual planned changes; a Goal Task enters bounded
Goal planning. No future Task has an accepted child plan yet.

Each completed Task must leave current delivery/Goal evidence and a safe managed commit, preserved
Goal commit range, or verified no-change result. Only then may pb reconcile the queue and activate
the next Task. A failed, blocked, cancelled, or budget-exhausted dependency prevents its dependents
from starting.

Previously completed commits are preserved if a later Task stops. Multi-Task cancellation and Goal
cancellation are preservation, not rollback.

### Just-in-time Task planning

The Task decomposition intentionally stays coarser than a workflow plan. Each Task records
the outcome, requirements, dependencies, acceptance facts, qualitative effort, execution kind, and
optional scope hints. It does not lock exact files, implementation operations, checks selected from
the future workspace, or a child plan digest.

Immediately before each Task:

1. reconcile the previous Task's terminal HEAD, managed commit or commit range, content, evidence,
   and queue digest;
2. derive a new `TaskRequest` from the accepted Task and current conversation context;
3. take a fresh repository baseline and focused brief;
4. run Build planning/review or Goal planning/approval against that current state;
5. deliver, check, review, and commit through the selected existing engine; and
6. store the result and actual usage before considering the next queued Task.

If the prior commit invalidates a future Task's assumptions, readiness review proposes a queue
revision. It cannot silently rewrite or skip the accepted Task. Completed Tasks and commits remain
immutable history; only pending Tasks are superseded.

### Planning failure and queue amendments

High-level decomposition gets two bounded attempts: an initial proposal and at most one reviewed
revision. If neither yields a valid artifact, pb records a typed `TaskPlanRejected` outcome and
offers `Retry planning`, `Edit request`, and `Run as one Build`. The last action is a new explicit
user decision and remains subject to all existing Build limits; pb never takes it silently.

The preflight has a typed wall-time and token allowance, reports activity through the existing
session presentation, and remains cancellable through the existing session control. Cancellation or
budget exhaustion before dispatch cannot create a `MultiTaskRun` or start a child. A one-Task result
still creates no Tasks panel.

After a Task is delivered, pb may automatically revise pending Tasks when current repository evidence
invalidates their assumptions, provided the revision preserves the source objective, authority,
publication boundary, aggregate budget, and any approved Goal contract. It cannot introduce a Goal
Task through amendment unless the planner is qualified and the existing Goal approval flow will
run. Any expansion pauses for an explicit user decision. The UI and event log show the old and new
Task-plan digests and the reason for revision before the next Task activates.

## Typed model

Introduce a focused module rather than placing Task orchestration inside `agent_core.rs`:

```text
src/task_queue/
├── mod.rs
├── artifacts.rs
├── config.rs
├── engine.rs
└── persistence.rs
```

The exact Rust names may change during implementation, but the persisted facts should be equivalent
to:

```rust
struct TaskPlanArtifact {
    version: u32,
    objective: String,
    requirements: Vec<TaskRequirement>,
    tasks: Vec<TaskSpec>,
    acceptance: Vec<TaskAcceptance>,
    risks: Vec<TaskRisk>,
}

struct TaskSpec {
    id: String,
    title: String,
    description: String,
    requirement_ids: Vec<String>,
    depends_on: Vec<String>,
    acceptance_ids: Vec<String>,
    scope_hints: Vec<String>,
    effort: TaskEffort,
    kind: TaskKind,
    goal_contract: Option<TaskGoalContract>,
    budget: TaskBudget,
}

enum TaskKind {
    Build,
    Goal,
}

struct TaskRun {
    spec_id: String,
    state: TaskState,
    attempts: usize,
    counters: TaskCounters,
    request: Option<TaskRequest>,
    workflow: Option<WorkflowCheckpoint>,
    goal: Option<GoalCheckpoint>,
    result: Option<TaskResult>,
}

struct TaskResult {
    base_head: String,
    terminal_head: String,
    commits: Vec<String>,
    no_change: bool,
    evidence_refs: Vec<String>,
    summary: String,
}

struct MultiTaskRun {
    version: u32,
    id: String,
    source_turn_id: String,
    stage: MultiTaskStage,
    plan: ArtifactEnvelope<TaskPlanArtifact>,
    policy_sha256: String,
    budget: MultiTaskBudget,
    counters: MultiTaskCounters,
    active_task_id: Option<String>,
    tasks: Vec<TaskRun>,
    outcome: Option<MultiTaskOutcome>,
}
```

`MultiTaskRun` is instantiated only for an accepted plan containing two or more Tasks. For an
accepted one-Task plan, pb projects that Task's budget and request directly into the selected
existing Build or Goal entry point; it creates neither `MultiTaskRun` nor `TaskRun`.

`scope_hints` guide just-in-time repository focus; they do not grant capabilities or make model
guesses authoritative. A Build Task's later `PlanArtifact` names exact paths against the current
repository and must remain within the Task, source request, and active-stage boundaries. A Goal
Task's accepted criteria and budget are part of the Task-plan digest, but its internal milestones
and Build plans retain their existing Goal representation and timing.

A normal Build Task result contains its existing managed commit. A Goal Task preserves the ordered
commits made by its existing child workflows and records their range from `base_head` through
`terminal_head`; the Task controller does not squash them or create a synthetic grouping commit.
The next Task binds to the reconciled `terminal_head` only after Goal completion and Task acceptance
are both current.

Persist the reviewed plan and exact compiled budgets in its digest. Reject duplicate IDs, unknown
dependencies/references, cycles, empty deliverables, uncovered requirements, impossible acceptance
mappings, invalid kind/Goal contracts, Tasks above policy ceilings, recursive decomposition, and
aggregate allocations above the source request budget before any child request starts.

Semantic quality still needs fresh Task-plan review. The critic must assess requirement coverage,
Task size, dependency order, commit boundaries, acceptance observability, migration/rollback order,
Build-versus-Goal choice, and budget fit. A `pass` verdict cannot override deterministic validation.

## Budget model

### Controller-owned allocation

The decomposition model supplies `TaskEffort` (`small`, `medium`, or `large`) plus scope and
evidence needs. It does not supply executable numeric limits. The controller maps effort through
typed policy presets, applies hard ceilings, reserves Task-plan coordination cost, and projects exact
`TaskBudget` values into the reviewed artifact.

`TaskBudget` covers the complete just-in-time planning and execution of that Task. It should
include the model-controlled dimensions that can reset accidentally across child
stages:

```rust
struct TaskBudget {
    max_workflows: usize,
    stage_steps: usize,
    model_invocations: usize,
    generated_tokens: usize,
    advisory_calls: usize,
    plan_cycles: usize,
    repair_cycles: usize,
    wall_time_minutes: u64,
}
```

For a Goal Task, this is the outer allocation for its Goal coordinator and every strict child
Build; the compiled inner Goal budget cannot exceed it. Review path and diff byte limits remain
workflow policy ceilings rather than spendable Task allocations. For a multi-Task plan, a
`MultiTaskBudget` adds `max_tasks`, `max_workflows`, total invocation/token/advisory and wall-time
limits, plus a bounded coordination reserve for Task planning, review, evaluation, and amendment.

The effective child allowance is the component-wise minimum of:

1. the accepted Task allocation;
2. the compiled strict-workflow policy;
3. the request-level step/token cap;
4. the remaining multi-Task budget, when present; and
5. the existing Goal budget for a Goal Task.

Every invocation, generated token, advisory call, stage step, repair/plan cycle, and elapsed-time tick
is charged once to the active Task and, for a multi-Task plan, rolled up once to
`MultiTaskCounters`. A Goal Task passes the same usage into the existing Goal counters without
creating a second Goal accounting model. High-level decomposition/review is charged to a bounded
coordination allowance. Retry, resume, replan, amendment, or a new child workflow never creates
fresh parent allowance.

Task allocation is a ceiling, not permission to consume the full amount. Unused capacity remains
unused. A later Task cannot borrow it above its accepted cap without a new reviewed Task-plan
digest. Any Goal Task increase also follows the existing user-controlled Goal budget/amendment
rules. Models may request review but cannot apply reallocation. The UI exposes allocations and
consumption as read-only facts. A user can explicitly approve an increase through the applicable
multi-Task or existing Goal decision flow, which creates a new reviewed digest and never resets
usage.

Add an optional typed `multi_task` policy and update `src/init.rs`; do not add an environment flag.
Omitting it must preserve current Build and Goal configuration, serialized checkpoints, and
effective limits unchanged.

## State and transitions

High-level planning is a bounded preflight:

```text
Decomposing → PlanReview → PlanRevision → Dispatch
                                      ↘ Rejected
```

Dispatch does one of two things:

```text
one Task   → existing Build or Goal state machine
2+ Tasks   → MultiTaskRun(RunningTask → Evaluating → Ready)
                                      ↘ Blocked / Failed / Cancelled
```

The one-Task path does not expose the preflight as a Task queue. Build retains its existing stages.
Goal retains `AwaitingPlanApproval`, final `AwaitingUserReview`, pause, and user decisions. The
multi-Task controller does not duplicate any child-engine state.

At every active Build or Goal checkpoint, the multi-Task controller:

1. validates the Task-plan digest, multi-Task policy, and active child binding;
2. reconciles expected HEAD, index, refs, and content with the prior Task result;
3. absorbs only new child usage into Task and multi-Task counters;
4. persists the exact active child checkpoint before another model stage;
5. evaluates Task and whole-plan acceptance against current evidence; and
6. selects the next dependency-ready Task or a typed terminal outcome.

The controller must be idempotent across event replay and restart. Usage absorption needs a stored
child-counter watermark so the same checkpoint cannot be charged twice.

## Build compatibility

- Preserve the existing `WorkflowRun` path and serialized behavior for ordinary single Builds.
- Keep current `PlanStep`/work-unit planning, stage tools, checks, review, commit ownership, and Ready
  semantics inside every Build Task.
- For one Build Task, dispatch directly and show the current Build UI; do not create or render a
  Task queue.
- A multi-Task run owns one branch and an ordered chain of Task commit results. It reports multi-Task Ready
  only after all required Tasks are committed/NoChange and whole-plan acceptance is
  current.
- A child final, commit, Goal completion, or Ready result cannot skip remaining Tasks.
- A Task plan cannot expand the source turn, allowed paths, tools, network, executor, secret, or
  publication envelope accepted by Build.
- If Task planning fails or cannot fit the configured aggregate budget, dispatch stops with a typed
  plan/budget outcome; it does not execute the broad original prompt as one oversized workflow.

## Goal Task compatibility

- A Goal Task creates or resumes an ordinary `GoalRun` from the Task objective, criteria,
  continuation policy, authority, and projected budget.
- Keep the existing Goal plan, `GoalMilestoneRun`, `milestones_for`, approval, continuation,
  persistence, evidence, amendment, and completion semantics unchanged.
- The multi-Task parent stores only the active Goal checkpoint reference, Task summary, budget
  watermark, and final Goal result. It does not interpret or replace Goal milestones.
- A Goal Task is delivered only after Goal reaches its existing accepted completion boundary and
  its terminal HEAD or no-change result is verified. Its Task result names the preserved child
  commit range; it does not require a new Task-level commit. Goal pause, rejection, block, failure,
  cancellation, or budget exhaustion stops the Task truthfully.
- For one Goal Task, dispatch directly and show the current Goal UI; do not create or render a Task
  queue.
- For a Goal Task inside a multi-Task run, show existing Goal progress only as detail beneath the
  active Task. The surrounding Tasks UI does not rename Goal states or criteria.
- Do not recursively invoke high-level Task decomposition from inside a Goal Task in version one.
  The Goal engine continues to choose its existing milestones and strict child workflows.
- Starting, replacing, pausing, resuming, or amending a Goal Task cannot reset Task, multi-Task, or
  existing Goal usage. Any budget or authority increase follows existing Goal user-decision rules.

## Persistence, API, and presentation

Only plans with two or more Tasks create new durable queue checkpoints. Those checkpoints include
the plan/policy digests, authority projection, Task queue and states, exact budgets/counters, active
child workflow/Goal reference, child counter watermark, Git/content fingerprints, summaries,
evidence references, and terminal reason. Tampering or impossible state fails validation before
model loading or mutation.

Add additive events for decomposition, review, activation, Task request/planning/checkpoint/commit,
Goal-Task progress, usage, budget stop, amendment, and terminal result. Keep every existing event
name and child event stream unchanged; link new events by multi-Task/Task/request/workflow/Goal ID.

When and only when the plan has two or more Tasks, the web/API projection should show:

- the ordered Tasks list and active Task;
- dependency/blocking state;
- each Task's Build/Goal kind, allocated and consumed budget;
- multi-Task totals;
- the accepted decomposition digest and revision;
- child workflow outcome, commit, checks, and review evidence; and
- a truthful next action for blocked, failed, paused, or budget-reached state.

Do not label a pending dependency as failed, a child Ready result as multi-Task completion, or
multi-Task Ready as external publication.

Add one durable inline Tasks panel without renaming, replacing, or reinterpreting existing UI
components. It appears only for multi-Task plans and the active Task expands into its existing child
Build/Goal presentation. Agent prompts and new serialized payloads use the same Task terminology.
Recommended states are `queued`, `planning`, `building`, `checking`, `reviewing`, `committed`,
`blocked`, `failed`, and `cancelled`. Committed Tasks show their commit or commit range and evidence
summary; queued Tasks remain concise and editable only through a reviewed queue revision. Budget
allocation and consumption are visible but not directly model-editable.

## Example breakdowns

These are Tasks, not child implementation plans. Exact paths and Build `PlanStep`s are
intentionally absent until each Task reaches the front of the queue.

### Durable CSV imports

| Task | Kind | Outcome and acceptance boundary | Illustrative controller budget |
| --- | --- | --- | --- |
| 1. Persist the import lifecycle | Build | Add durable queued/running/completed/failed/cancelled state and migration; storage tests pass and the service restarts with state intact | 6 invocations / 4,500 tokens / 30 min |
| 2. Make execution restart-safe and idempotent | Goal | Demonstrate interrupted imports resume without duplicate rows, durable cancellation works, and failure recovery meets the accepted criteria; the Goal may create several child Builds after Task 1's commit | 18 invocations / 14,000 tokens / 75 min |
| 3. Expose compatible import APIs | Build | Start/status/cancel behavior passes API tests and existing endpoint contracts remain unchanged | 6 invocations / 4,500 tokens / 30 min |
| 4. Add progress, error, cancel, and retry UI | Build | Web behavior tests cover every state and the user guide for the UI is current | 7 invocations / 5,000 tokens / 35 min |

Task 1 is planned against the initial repository. Task 2 is not planned until Task 1 is reviewed and
committed. Its Goal child Builds are then planned against the new schema. Task 3 sees the committed
recovery semantics, and Task 4 sees the final API contract. Documentation and tests live with the
behavior-owning Task rather than in a vague final cleanup Task.

### Implement Task decomposition in pb

| Task | Kind | Outcome and acceptance boundary |
| --- | --- | --- |
| 1. Define Task-plan artifacts and compatibility | Build | Add Task decomposition schemas, budget projection, DAG validation, the one-Task bypass, and compatibility fixtures |
| 2. Make the Task queue durable | Goal | Prove one-active-Task scheduling, restart recovery, exact counter rollup, no double charge, and commit-to-next-Task reconciliation across deterministic fixtures |
| 3. Integrate multi-Task Build | Build | Convert each active Task into a fresh strict workflow request while preserving the existing single-Build path |
| 4. Dispatch Goal Tasks | Build | Convert a Goal Task into an ordinary existing Goal request and checkpoint without changing Goal plans, milestones, accounting, or UI |
| 5. Add multi-Task progress UI | Build | Add the conditional Tasks list, budgets, active existing Build/Goal detail, evidence, and commits while preserving every current UI component |

Task 2 is a good Goal candidate for a stronger model because its desired result is a set of durable
controller guarantees, while the exact sequence of fixes may depend on failures found during
recovery testing. Tasks 1 and 3–5 have crisp deliverables and are better represented as Builds.

### UI projection

```text
Tasks  1 of 4 committed                         Overall 12% budget used

✓  1  Persist the import lifecycle     Build   committed  8fd31c2
●  2  Make execution restart-safe      Goal    criterion 2/3 · 41% budget
○  3  Expose compatible import APIs    Build   queued
○  4  Add import progress UI           Build   queued

Active Task 2 — Goal
[existing Goal progress and controls]
```

The top list is stable multi-Task state. The lower area is the existing UI for the active Build or
Goal. When Task 2 commits, pb collapses its child transcript into evidence/usage/commit summary,
reconciles the queue, and only then begins planning Task 3. A one-Task plan shows only that existing
Build or Goal UI, without the top list.

### Small requests stay small

For “Fix the average divisor and add its regression test,” the high-level planner returns one Build
Task. pb unwraps it into one ordinary Build and does not create a `MultiTaskRun` or show a Tasks
list.

## Implementation sequence

### Task 0 (Build) — Lock the decomposition and accounting corpus

- Add typed scripted fixtures for valid DAGs, cycles, unknown dependencies, missing coverage,
  oversized Tasks, false aggregate totals, model-authored numeric budgets, migration-order review,
  catch-all Task revision, Build-versus-Goal selection, two-attempt exhaustion, and unqualified Goal
  proposals.
- Add cross-stack, storage-migration, and multi-component-refactor real-model prompts.
- Record decomposition/review protocol, graph validity, accepted Task size, budget truth,
  revision count, runtime, and context separately from generated artifact quality.

Acceptance:

- every invalid deterministic artifact is rejected before Task activation;
- the model contract contains qualitative effort but no executable numeric allowance;
- the checked corpus reproduces the failure classes in the feasibility probe;
- the second invalid attempt produces `TaskPlanRejected` and no child mutation; and
- Goal selection is allowed only by a matching qualified planner record or explicit user intent.

Suggested commit: `test: baseline task decomposition control`

### Task 1 (Build) — Add Task-plan artifacts and validation

- Add `src/task_queue` artifacts, digest, DAG/coverage/scope validators, review artifacts, and policy
  types.
- Add a deterministic effort-to-budget compiler with coordination reserve and exact aggregate
  projection.
- Add planner capability records that independently qualify Task decomposition and automatic Goal
  selection for an exact model/template/protocol version.
- Add optional multi-Task configuration and update `src/init.rs` scaffolding without changing
  existing workflow/Goal configuration.
- Add the one-Task dispatch rule without creating a queue checkpoint.

Acceptance:

- serialization and digest order are stable;
- invalid graph, scope, coverage, authority, and budget states fail closed;
- compiled allocations are deterministic and never exceed parent/policy ceilings;
- unqualified Goal output fails validation rather than being coerced;
- existing Build and Goal config retains equivalent limits and serialized behavior.

Suggested commit: `feat: define bounded task plans`

### Task 2 (Goal) — Make the multi-Task controller durable

- Implement the reducer, one-active-Task scheduler, child checkpoint binding, counter watermark and
  rollup, Git/content reconciliation, persistence, resume, cancellation, and terminal outcomes.
- Create a fresh `TaskRequest` and repository baseline only when a dependency-ready Task activates.
- Reuse `WorkflowRun` unchanged as the Build executor and `GoalRun` unchanged as the Goal executor.
- Preserve completed child commits/evidence, record Build commits or Goal commit ranges, and block
  dependents after non-success.
- Add automatic pending-Task revision under the immutable objective/authority/budget/Goal-contract
  envelope; route expansions to an explicit user decision.

Acceptance:

- restart resumes the exact Task/request stage with exact remaining allowances;
- replay cannot double-charge usage or start a second child;
- retry/replan/new-Task transitions never reset multi-Task counters;
- stale Git/content/policy/plan state blocks before mutation;
- only dependency-ready Tasks can start; and
- future Tasks have no accepted child plan or path set before activation;
- completed Task history never changes during queue revision; and
- the next Task binds to the reconciled terminal HEAD of the prior Build commit or Goal commit range.

Suggested commit: `feat: orchestrate durable task queues`

### Task 3 (Build) — Integrate queued Build Tasks

- Add typed Task-plan proposal/review/revision before dispatch.
- Compile active Task requests into existing workspace, contract, check, review, and commit machinery.
- Preserve the direct one-Task workflow path and add truthful multi-Task summaries/journals to
  harness and daemon sessions.
- After two invalid attempts, expose retry/edit/explicit-single-Build actions without automatic
  fallback.

Acceptance:

- every existing single-Build workflow fixture retains its stage/outcome/evidence behavior;
- a Task cannot broaden authority or publication scope;
- child Ready cannot bypass remaining Tasks or whole-plan acceptance;
- an infeasible Task plan stops instead of running as one oversized fallback Task;
- the durable accepted-plan projection precedes Task 1's activation event so the UI can render the
  Tasks panel immediately; and
- choosing `Run as one Build` records a new user decision and enters the unchanged Build path.

Suggested commit: `feat: dispatch queued build tasks`

### Task 4 (Build) — Dispatch Goal Tasks

- Convert a Goal Task contract into the current Goal creation/resume input.
- Store and reconcile the existing `GoalCheckpoint` as the active Task child.
- Apply the Task ceiling through existing Goal limits without changing Goal accounting.
- Preserve the ordered child commit range and terminal HEAD without squashing or adding a synthetic
  Task commit.
- Preserve current Goal approval, milestones, continuation, pause, amendment, evidence, UI, and
  completion rules.

Acceptance:

- existing Goal checkpoints restore byte-compatibly with their existing plan digest and milestone
  behavior;
- a single Goal Task is indistinguishable from current Goal mode after dispatch;
- a Goal Task's usage rolls into Task and, when present, multi-Task totals exactly once;
- Goal completion still requires its existing verifier/acceptance basis;
- Task-plan amendments cannot expand Goal authority or budget without the existing user decision;
- Goal Task approval still occurs at the current Goal boundary; and
- restart and resume retain the exact child commit range and terminal HEAD.

Suggested commit: `feat: dispatch queued goal tasks`

### Task 5 (Build) — Add multi-Task UI, API, and documentation

- Add Tasks progress, active child, budgets, evidence, blocked reason, and decomposition revision to
  shared Rust/web event projections and responsive UI.
- Render it only for plans with two or more Tasks. Preserve all existing UI component names,
  behavior, and historical event rendering.
- Reuse the existing Build or Goal presentation beneath the active Task instead of recreating its
  stages.
- Add the complete state matrix for initial planning, automatic start, Goal approval, reload/resume,
  retry, pending-plan revision, blocked dependency, budget exhaustion/increase, cancellation,
  failure, no-change, and completion.
- Show allocation/consumption read-only and expose only explicit user decision controls for an
  increase.
- Add multi-Task configuration and lifecycle guidance under `docs/user/`.
- Update `docs/architecture/workflows.md`, `user-contracts.md`, `security.md`, and `local-privacy.md`
  when the behavior ships; retain this document as its design/evidence record.

Acceptance:

- CLI/API/web/restored-session projections agree on Task and budget state;
- desktop/mobile users can identify Build versus Goal ownership without opening raw events;
- the inline Tasks panel is keyboard and screen-reader operable, works at narrow and wide layouts,
  and respects existing safe-area/PWA requirements;
- a one-Task run has no Tasks panel, queue controls, or changed Build/Goal interaction;
- no UI control can apply a model-requested budget or authority increase; and
- docs distinguish shipped behavior, configurable limits, and design history.

Suggested commit: `feat: expose multi-task progress`

### Task 6 (Goal) — Qualify and roll out

- Run deterministic Task-plan, workflow, Goal, persistence, API, web, docs, and harness suites.
- Maintain at least 30 expert-reviewed prompts across the cross-stack, migration, refactor, small
  single-Build, and Goal-selection shapes; run three locked trials per prompt for every candidate
  model/template/protocol.
- Keep 4B Task planning opt-in unless it closes typed plan/review artifacts within the bounded
  attempts. Do not add hidden stronger-model or cloud escalation.
- Qualify basic Task planning and automatic Goal selection separately; record the exact
  model/template/protocol digest with the result.

Promotion gates:

- zero accepted invalid DAGs, authority escapes, budget overflows/resets, false Ready/Goal-complete
  claims, duplicate charges, or silent restart continuation;
- 100% deterministic Task order, budget, persistence, and compatibility fixtures;
- at least 95% of trials for a default-enabled planner produce an accepted, expert-acceptable valid
  plan within the two attempts, while every rejection exposes the bounded recovery actions and every
  accepted plan passes the same controller validator;
- zero automatic Goal selections from an unqualified planner and zero Goal activation without the
  existing approval contract;
- zero unjustified Goal selections on the locked negative corpus and at least 90% recall on the
  expert-labelled Goal-positive corpus for any profile qualified to select Goal automatically; and
- no regression in the existing small-model workflow and Goal-control corpora.

Suggested commit: `fix: harden task planning control`

## Verification policy

Each implementation Task needs focused reducer/artifact tests plus the repository gates in
`AGENTS.md`. User-visible or architectural behavior changes require the matching curated docs in the
same commit. Web state changes require web tests. Configuration additions require `src/init.rs`.

Before rollout, run at minimum:

1. `deno task build:web`
2. `cargo fmt --all -- --check`
3. `cargo clippy --all-targets --all-features -- -D warnings -A clippy::all -D clippy::correctness -D clippy::suspicious`
4. `cargo test --all-targets`
5. `deno task test:web`
6. `deno task test:docs`
7. `cargo build --release --target aarch64-apple-darwin`
8. the full scripted harness corpus and new Task-decomposition corpus
9. the locked 4B/7B/14B Task-decomposition matrix

The narrow FlashMoe smoke is required only if implementation changes FlashMoe data flow or backend
behavior.

## Explicit non-goals for version one

- parallel mutating Tasks in one repository;
- model-authored executable numeric budgets;
- automatic stronger-model or cloud escalation;
- arbitrary user-programmable workflow DAG stages;
- cross-repository projects;
- Task-owned network/publication authority;
- rollback of already committed Tasks on later failure; and
- treating multi-Task Ready as Goal acceptance or external publication.
