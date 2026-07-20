# Durable goal mode implementation plan

Status: **Shipped version-one implementation; G8 qualification recorded**

This document is the design and implementation record for durable **Goal** mode in pb. Goal UI,
typed proposal/control tools, hashed persistence, sequential strict-workflow continuation, API,
Unix RPC, CLI, safe pause, amendments, terminal evidence, and responsive monitoring are shipped.
The curated current behavior lives in the user and architecture chapters; the workstream detail
below preserves the original design target and rollout reasoning.

## Version-one implementation outcome

The shipped controller deliberately keeps the highest-risk choices deterministic:

- the initial linear Goal plan is derived from user-authored ordered criteria and must be approved
  by exact digest; each milestone still runs the existing repository-aware planning and independent
  plan-review stages before implementation;
- one milestone and one embedded strict workflow run at a time, with remaining Goal totals clamped
  into each child workflow policy;
- user/API changes use optimistic Goal digests, plan changes receive new plan digests, and accepted
  amendments preserve completed milestones plus retired/current criterion evidence;
- running pause is cooperative at model/tool/workflow boundaries; restart never silently resumes;
- prose criteria stop at Ready for review, while `workflow_ready` API criteria may machine-complete;
- model tools can propose/start an approval-gated Goal, read bounded status, or request pause,
  amendment, and budget review; they cannot resume, cancel, accept, publish, or rewrite state; and
- desktop and mobile render a persistent textual Goal identity, progress, plan review, details,
  budgets, decisions, edit/pause/resume/accept/stop controls, and terminal history.

Deterministic reducer, persistence, API, tool-surface, parser-recovery, restart, responsive-web, and
seven Goal-specific harness assertions are part of the implementation. The locked 4B/7B/14B field
matrix is recorded in the [G8 qualification report](benchmarks/goal-mode-g8.md). Explicit Goal
creation and digest approval remain the primary contract. Review-plan continuation is safe because
the deterministic controller, rather than the model, starts only accepted milestones. Automatic
continuation remains an explicit per-goal choice, 4B remains explicit-only, and one-turn Auto stays
an internal explicit intent rather than a normal web default.

## Decision summary

- A goal is a durable session-level controller above delivery workflows. It is not an
  `AgentProfile`, a longer prompt, or a new synonym for Build.
- A session may contain zero or more completed goals and at most one active goal. A goal may contain
  several bounded delivery workflows, but version one runs only one milestone and one workflow at a
  time.
- The user explicitly starts Goal mode by default. A model may propose a goal during discussion.
  Model-initiated start is available only after a user grants Auto authority for the cited turn.
- The goal controller, not the model, owns budgets, stage transitions, pause/resume, durable state,
  current evidence, and terminal claims.
- Goal planning produces a small linear milestone plan. Each implementation milestone is delivered
  by the existing strict workflow: planning, plan review, implementation, checks, code review, and
  managed commit.
- Goal updates are authority-monotonic. A model may narrow work, lower a budget, or request an
  amendment. It cannot expand paths, tools, integrations, network access, publication authority, or
  budgets without an explicit user or trusted-policy decision.
- While a goal is non-terminal, desktop and mobile surfaces show a persistent, textual **Goal**
  identity derived from durable server state. The UI never relies on colour, animation, or the
  transient activity feed to tell the user which mode controls the session.
- A user can change an active goal through a checkpointed amendment flow. Completed milestones and
  evidence remain immutable; the revised objective, criteria, and remaining milestones receive a
  new reviewed plan digest before work resumes.
- Machine-verifiable criteria may complete automatically. Prose-only criteria stop at **Ready for
  review** until the user accepts the result or requests another milestone.
- The web UI presents Goal beside Discuss and Build as a composer choice, while the API keeps
  `TurnIntent` unchanged. Goal creation is a separate lifecycle operation, not a conversational turn
  intent.
- Local Ready evidence remains local. Goal mode does not push, open pull requests, wait for remote
  CI, merge, or publish.

## Shipped foundation and resolved prerequisites

The following foundation is already shipped:

- `TurnIntent::Discuss`, `Deliver`, and the internal `Auto` transition;
- read-only discussion and the `propose_delivery` / `start_delivery` handoff pattern;
- strict stage capabilities and typed terminal submissions;
- challenged plans, deterministic affected checks, isolated code review, repair, and managed
  commits;
- workflow-wide step, invocation, generated-token, advisory, plan-cycle, and repair-cycle budgets;
- fingerprinted workflow checkpoints and Ready evidence;
- persisted daemon sessions, restart-to-paused recovery, and session event streaming; and
- web list/detail projections for active workflow stage and outcome.

The July 19, 2026 feasibility probes exposed two prerequisites that landed with Goal mode:

1. `pb harness agent --intent auto` now includes delivery and Goal proposal/start tools in its
   narrowed surface; Discuss still removes the start tools and Auto calls must cite the exact turn.
2. Malformed textual native-tool arguments now enter the bounded parse-correction path rather than
   escaping the completion adapter as `engine_error`. A 7B strict workflow had produced a
   valid plan, plan review, and exact file mutation, then stopped on a one-character JSON structure
   error in `submit_implementation`.

These were control-plane recovery/exposure defects, not reasons to move orchestration into model
prompts. Deterministic regressions now cover both.

## Outcome

A user can start a bounded goal from a project, the home screen, or an existing conversation; review
what pb plans to do; let pb work through several local milestones; and intervene without losing the
accepted plan, current workspace, evidence, or budget history.

The user can always answer:

1. What is the goal and what does done mean?
2. What did I authorize, including continuation and resource limits?
3. Which milestone and strict workflow stage is active?
4. What has been committed and what evidence supports it?
5. How much of the total goal budget remains?
6. Why is the goal paused or blocked, and what decision is required?
7. Is pb claiming machine-verified completion, or asking me to review a prose criterion?

Version one is successful when it can finish a small multi-milestone local repository objective,
survive daemon restart, stop safely on weak-model output, and render the same truthful state in the
web UI, CLI, API, persisted checkpoint, and harness journal.

## Product language

Use these user-facing terms:

- **Discuss** — investigate, explain, or shape work without mutation.
- **Build** — deliver one bounded repository change through the strict workflow.
- **Goal** — pursue a larger objective through one or more bounded Builds within a total budget.
- **Milestone** — one independently reviewable step in a goal.
- **Ready for review** — pb has finished its planned work, but at least one completion criterion
  still needs a person to accept it.
- **Complete** — every required criterion has machine evidence or an explicit user acceptance.
- **Needs input** — a user-owned decision blocks safe progress.
- **Needs authority** — the next proposal would broaden the approved authority envelope.
- **Budget reached** — the current approved total is exhausted; no automatic continuation occurs.
  It remains recoverable only when the user may approve more budget below a hard policy ceiling.

Keep internal terms such as checkpoint digest, content fingerprint, stage contract, artifact
envelope, and policy hash behind an expandable evidence view.

Use **Goal** consistently in product UI. Do not alternate between “Goal mode” and “Task mode” for
the same state: task is a generic product/conversation concept, while Goal names this durable
controller.

## Invariants

- Explicit user intent and answers remain above project configuration, harness policy, and model
  proposals in the authority hierarchy.
- Goal mode never turns a repository-owned file into permission to self-start, expand authority, or
  publish. Project configuration may narrow or cap a goal; it cannot grant user authority.
- A model final, a workflow Ready result, a goal Ready-for-review result, and goal completion remain
  distinct claims.
- Every mutating milestone uses the shipped strict workflow. Goal coordination does not receive a
  parallel edit, shell, commit, or Git-control path.
- Accepted goal plans, amendments, criteria, budgets, milestone results, and completion evidence are
  typed, hashed, and durable. Model prose cannot replace them.
- Goal-wide budgets include all coordinator, planner, reviewer, workflow, repair, advisory, and
  retry invocations. Starting another milestone does not reset the total.
- A goal can have at most one active milestone and one active workflow. Version one has no
  concurrent mutation or multi-branch merge problem.
- Every milestone begins from the committed content left by its predecessor. Uncommitted or external
  Git-control changes block continuation until reconciled.
- Pause is cooperative and checkpointed. It does not roll back changes, abandon evidence, or claim
  that an in-flight command was interrupted when it was allowed to finish.
- Cancel preserves workspace content and audit evidence. Version one does not offer automatic
  rollback.
- Restart never silently resumes mutation. An interrupted active goal restores as paused and
  requires an explicit Resume, even when its continuation policy was automatic.
- Publication remains a separate authority boundary.

## Non-goals for version one

- A user-programmable workflow DAG or arbitrary model-authored state machine.
- Parallel milestones, multiple active worktrees, or automatic merge/conflict resolution.
- Continuous background operation with no total step, token, invocation, workflow, retry, or wall
  time limit.
- Model-selected cloud escalation or automatic switching to a stronger model.
- Automatic path, integration, network, secret, or publication expansion.
- Cross-repository goals. A goal may use the existing normalized workspace graph within one
  repository, but it has one repository and task branch owner.
- Scheduled or recurring goals; those need a separate automation lifecycle.
- Automatic rollback, squash, rebase, push, PR creation, remote CI waiting, or merge.
- Treating a reviewer model's prose as machine verification of an inherently subjective criterion.

## User journeys

### Start from Home or a project

1. The user enters an objective and selects **Goal** beside Discuss and Build.
2. pb opens a Goal setup sheet instead of immediately queuing a model run. On Home, the user must
   select a registered project; version one does not start a repository-less goal.
3. The objective is prefilled from the composer. The user may add "Done when" criteria, choose a
   continuation policy, and choose a budget preset or advanced limits.
4. The sheet displays the repository/focus root, local authority summary, and an explicit statement
   that publishing is excluded.
5. **Plan goal** creates a session and starts a read-only goal-planning stage.
6. The user sees the proposed milestones and criteria. Under the recommended default, delivery does
   not begin until the user chooses **Approve and start**.

### Promote a conversation

1. During Discuss, the model may call `propose_goal` with a bounded objective and suggested criteria.
2. The conversation renders a proposal card with **Review goal** rather than a mutation button.
3. Opening the setup sheet carries the cited source turn and conversation decisions into the draft.
4. User edits are authoritative. The proposal does not self-start.

### Explicit Auto activation

1. An advanced action on the **Discuss** composer offers **Allow this turn to start a goal when the
   objective is clear**. This is separate from explicitly choosing Goal.
2. Sending that turn records an activation grant bound to the current session and source turn ID.
3. The read-only model may call `start_goal` once with that exact turn ID, objective, and criteria.
4. The call ends the read-only invocation. The daemon validates the grant and queues goal planning;
   it does not expose mutation in the discussion turn.
5. The grant is consumed on use and is not persisted as a reusable project permission.

### Run and monitor a goal

1. A persistent Goal banner shows that Goal, rather than Discuss or a one-off Build, currently owns
   the session. It names the state and current milestone on desktop and remains a compact sticky row
   on mobile.
2. The Goal header shows overall status, current milestone, and total budget use.
3. The main conversation continues to render teammate messages and concise workflow activity.
4. The Goal drawer shows the accepted plan, milestone history, evidence, and exact budget counters.
5. In automatic continuation, a successful milestone queues the next accepted milestone without a
   new model deciding whether it has authority.
6. In manual continuation, the UI pauses between milestones and shows **Start next milestone**.

### Intervene

- **Pause** records a pause request. pb finishes the current non-interruptible model/tool operation,
  checkpoints at the next safe boundary, and changes the label to Paused.
- **Resume** reconciles policy, Git control, content, remaining budget, and the active workflow
  checkpoint before another model invocation.
- A missing product decision renders a **Needs input** card with choices and consequences.
- A proposed scope, budget, path, tool, integration, or network expansion renders a separate request
  card. Denial leaves the original authority intact.
- **Stop goal** requires confirmation and states that completed commits and current workspace changes
  are preserved.

### Change an active goal

Changing an active goal is not a chat-message side effect and never rewrites accepted history:

1. The user chooses **Edit goal** from Goal details or the Goal overflow menu.
2. If work is running, the UI first requests a safe-boundary pause and labels the action **Pause and
   edit**. The amendment editor opens only after the durable Paused event.
3. The editor starts from the accepted objective, criteria, continuation policy, budget, and
   remaining milestones. Completed milestones, commits, checks, and evidence are read-only.
4. The user submits a typed amendment against the latest goal digest. The server classifies changes
   to the objective, success criteria, remaining scope, budget, paths, tools, and network authority.
   Removing a success criterion is a change to the completion contract, not an automatic narrowing.
5. pb replans only the unfinished portion in read-only context, reviews it, and presents an exact
   before/after summary. Pending milestones may be retained, replaced, or marked Superseded; they
   are never silently deleted from history.
6. **Approve changes and resume** accepts the revised plan digest and reconciles the workspace before
   the next workflow starts. **Discard changes** returns to the prior paused plan. **Stop and start a
   new goal** is offered when the new request changes repository or is materially unrelated.

User-authored expansion is allowed only through the existing explicit budget/authority confirmation
steps. A model-proposed amendment follows the same review UI and cannot approve itself. Discussing a
different idea while the goal is paused does not modify the goal.

### Finish

1. After the final milestone, the harness evaluates every accepted criterion against current durable
   evidence.
2. Fully machine-verifiable criteria may transition directly to Complete.
3. Any prose or user-confirmation criterion transitions to Ready for review.
4. The completion card lists commits, checks, criteria, remaining caveats, and changes outside the
   goal contract.
5. **Accept goal** records user acceptance of the displayed evidence digest and transitions to
   Complete. **Request another milestone** keeps the same goal active and enters reviewed plan
   revision; it does not end Goal mode.
6. **Stop goal** transitions to Cancelled after safe child-workflow shutdown. An unrecoverable
   controller failure or a hard policy ceiling transitions to Failed with a typed outcome.
7. On any terminal event—Completed, Cancelled, or Failed—the server clears the active-goal slot,
   retains the complete goal history/evidence, and emits the terminal session projection atomically.
   The Web UI removes active Goal controls and restores Discuss as the selected composer mode.
8. Ready for review, Paused, Blocked, and a recoverable Budget reached state are non-terminal. They
   continue to show Goal active until the user resolves them or explicitly stops the goal.

## Durable domain model

Add a focused `src/goal/` module rather than extending `agent_core.rs` with another large inline
state machine:

```text
src/goal/
  mod.rs
  artifacts.rs
  capabilities.rs
  config.rs
  engine.rs
  persistence.rs
  tools.rs
```

### Session ownership

```rust
struct ConversationSession {
    session_id: String,
    turns: Vec<ConversationTurn>,
    active_workflow: Option<WorkflowCheckpoint>, // non-goal Build only
    completed_workflows: Vec<WorkflowSummary>,
    active_goal: Option<GoalCheckpoint>,
    completed_goals: Vec<GoalSummary>,
}
```

There must be one canonical owner for an active workflow:

- a normal Build stores it in `active_workflow`; and
- a Goal milestone stores it in `active_goal.run.active_milestone.workflow`.

Session list/detail workflow fields remain compatibility projections. For a goal session they are
derived from the embedded milestone workflow rather than persisted twice.

### Goal run and checkpoint

```rust
struct GoalCheckpoint {
    sha256: String,
    run: GoalRun,
}

struct GoalRun {
    version: u32,
    id: String,
    session_id: String,
    source_turn_ids: Vec<String>,
    objective: String,
    stage: GoalStage,
    policy: CompiledGoalPolicy,
    policy_sha256: String,
    authority: GoalAuthorityEnvelope,
    authority_sha256: String,
    planning_snapshot: ContentSnapshot,
    git_control: WorkflowGitControlState,
    plan: Option<ArtifactEnvelope<GoalPlanArtifact>>,
    plan_review: Option<ArtifactEnvelope<GoalPlanReviewArtifact>>,
    milestones: Vec<GoalMilestoneRun>,
    active_milestone_id: Option<String>,
    criteria: Vec<GoalCriterionState>,
    counters: GoalCounters,
    amendments: Vec<GoalAmendment>,
    requests: Vec<GoalDecisionRequest>,
    ready_evidence: Option<GoalReadyEvidenceBundle>,
    outcome: Option<GoalOutcome>,
    blocked_reason: Option<String>,
}
```

The checkpoint digest covers every field that can affect continuation or terminal truth. Policy,
authority, accepted plan, criteria, budget counters, milestone workflow checkpoints, decisions, and
evidence are therefore tamper-evident together.

### Goal stages and outcomes

```rust
enum GoalStage {
    Planning,
    PlanReview,
    PlanRevision,
    AwaitingPlanApproval,
    RunningMilestone,
    Evaluating,
    AwaitingUserReview,
    Paused,
    Blocked,
    Completed,
    Failed,
    Cancelled,
}

enum GoalOutcome {
    Complete,
    BudgetExhausted,
    CriteriaUnsatisfied,
    MilestoneFailed,
    RepeatedNoProgress,
    AuthorityDenied,
    ContextLimit,
    EngineError,
    Cancelled,
}

enum GoalCompletionBasis {
    MachineVerified,
    UserAccepted,
}
```

`Paused` records its exact resumable prior stage. `Blocked` records a typed blocker and may resume
only after the matching decision or external condition changes. Terminal stages cannot resume.
`GoalReadyEvidenceBundle` records the completion basis separately from the terminal outcome, so
user acceptance cannot be mistaken for machine verification.

Terminalization must occur at a safe boundary and as one durable reducer operation: finalize or
reconcile the active child workflow, write the outcome and evidence, append the terminal event,
move the goal from the session's active slot to durable goal history, and publish the new session
summary. A missed event can therefore recover the same inactive state from `SessionDetails`.

### Goal plan

Version one uses a linear plan with at most the configured milestone ceiling:

```rust
struct GoalPlanArtifact {
    summary: String,
    requirements: Vec<GoalRequirement>,
    criteria: Vec<GoalCriterion>,
    milestones: Vec<GoalMilestone>,
    risks: Vec<GoalRisk>,
    assumptions: Vec<String>,
    open_questions: Vec<String>,
    resolved_challenge_ids: Vec<String>,
}

struct GoalMilestone {
    id: String,
    title: String,
    description: String,
    requirement_ids: Vec<String>,
    criterion_ids: Vec<String>,
    expected_components: Vec<String>,
}
```

Each milestone must be independently deliverable, reviewable, and small enough for one strict
workflow. Milestones cannot contain arbitrary commands or capabilities. Exact paths, changes, and
checks are resolved and validated by the milestone's ordinary strict planning stage.

Goal plan review runs in fresh read-only context. It checks requirement coverage, milestone size,
ordering, acceptance traceability, authority fit, budget fit, failure/recovery boundaries, and
whether a human decision is being disguised as an assumption.

### Goal amendments and plan versions

Every accepted initial or revised plan has an increasing version and content digest. An amendment
records its base goal/plan digests, requester, typed field changes, authority and budget deltas,
reviewed replacement-plan digest, decision, and timestamps. Completed milestone records and
evidence keep the plan version under which they ran.

Only draft goals may use the ordinary draft update operation. Once a plan is approved, all semantic
changes use an amendment. Applying an amendment invalidates stale completion evidence where its
criterion or repository expectation changed, but preserves still-valid evidence with an explicit
carry-forward receipt. The server—not the model or browser—calculates that distinction.

### Completion criteria

```rust
enum GoalVerifier {
    Check { component_id: String, check_id: String },
    PathState { path: String, expectation: PathExpectation },
    CommitEvidence,
    NoUnexpectedChanges,
    UserConfirmation,
    ReviewRequired,
}
```

- Machine verifiers require current evidence and may complete automatically.
- `UserConfirmation` is satisfied only by an explicit user action tied to the current evidence
  digest.
- `ReviewRequired` lets a user express a useful prose criterion without pb mislabelling a model
  judgment as verification. It always stops at Ready for review.
- The goal planner may propose a stronger machine verifier, but changing the verifier is part of
  the user-approved goal plan.

### Authority envelope

`GoalAuthorityEnvelope` snapshots:

- repository and focus roots;
- allowed path constraints from a trusted user/harness contract;
- current workflow, environment, MCP/LSP, and policy hashes;
- whether public research is allowed;
- continuation policy;
- explicit per-turn Auto activation grant, if any; and
- an explicit `publication = false` boundary.

The envelope does not duplicate stage capabilities. Every model invocation still receives the
intersection of its goal-stage capability, workflow-stage capability, request allowlist, project
policy, environment capability, and remaining authority. A goal amendment may only preserve or
narrow that intersection without approval.

### Goal policy and budget

Add `.pb/goal.toml` as a project-owned narrowing and ceiling document with its own version and
policy hash. `pb init` must create or preserve it when this workstream lands.

Representative version-one policy:

```toml
version = 1
max_milestones = 8
max_workflows = 12
max_milestone_attempts = 2
max_consecutive_failures = 2

[limits]
total_model_invocations = 120
total_generated_tokens = 100000
advisory_calls = 12
wall_time_minutes = 120
```

Repository configuration cannot enable Auto start or automatic continuation by itself. Those are
user/session choices constrained by the project ceilings.

`GoalCounters` records at least:

- coordinator and workflow model invocations;
- prompt and generated tokens;
- advisory calls;
- elapsed active wall time;
- milestone workflows and attempts;
- plan revisions;
- amendments and budget requests; and
- repeated no-progress failures.

Every child workflow receives limits bounded by both its normal compiled workflow policy and the
remaining goal budget. A retry never receives a fresh goal-wide allowance.

## Controller state machine

| Current stage | Accepted event | Next stage | Owner |
| --- | --- | --- | --- |
| Planning | valid goal plan | PlanReview | model submission, harness validation |
| PlanReview | pass | AwaitingPlanApproval or RunningMilestone | reviewer submission, policy |
| PlanReview | blocking findings | PlanRevision | harness reducer |
| AwaitingPlanApproval | user approves current digest | RunningMilestone | user/API |
| RunningMilestone | workflow Ready/NoChange | Evaluating | existing workflow engine |
| RunningMilestone | workflow Blocked | Blocked | workflow evidence |
| RunningMilestone | workflow Failed | PlanRevision or Failed | bounded goal policy |
| Evaluating | more accepted milestones | RunningMilestone or Paused | deterministic scheduler |
| Evaluating | all machine criteria pass | Completed | deterministic verifier |
| Evaluating | user/review criteria remain | AwaitingUserReview | deterministic verifier |
| AwaitingUserReview | user accepts evidence digest | Completed | user/API |
| AwaitingUserReview | user requests another milestone | PlanRevision | user/API |
| Active stage | user requests edit | Paused at safe boundary | user/API, harness checkpoint |
| Paused | valid amendment submitted | PlanRevision | user/API |
| AwaitingPlanApproval | revised plan approved | RunningMilestone | user/API, reconciliation |
| AwaitingPlanApproval | revised plan discarded | Paused with prior plan | user/API |
| Active stage | recoverable budget exhausted | Blocked | deterministic budget guard |
| Active stage | hard policy ceiling exhausted | Failed | deterministic budget guard |
| Active stage | pause reaches safe boundary | Paused | user request, harness checkpoint |
| Paused/Blocked | validated resume condition | prior stage | user/API, reconciliation |
| Non-terminal | cancel | Cancelled | user/API |

Invalid events fail closed. Reducer tests must cover every permitted edge and representative
forbidden edges without loading a model.

## Orchestration loop

1. Capture current repository, workspace graph, environment evidence, Git-control state, goal
   policy, workflow policy, and user authority.
2. Run goal planning and plan review with read-only goal-stage capabilities.
3. Pause for user plan approval unless the user explicitly chose automatic plan acceptance.
4. Select the first pending accepted milestone deterministically.
5. Compile a `ConversationHandoff` and optional harness contract from the milestone's requirements,
   criteria, and authority subset.
6. Start the existing strict workflow on the goal-owned branch/workspace. Do not create a second
   mutation path.
7. Persist both goal and child workflow checkpoint before each model invocation or deterministic
   stage.
8. On Ready or NoChange, bind the workflow summary and Ready evidence into the milestone record.
9. Reconcile HEAD, index, refs, content, policy, and total remaining budget before selecting the next
   milestone.
10. If a failure invalidates the accepted goal plan, request bounded plan revision. Do not blindly
    rerun the same milestone after two equivalent failures.
11. Evaluate completion criteria after every milestone and again at finalization.
12. Emit Complete, Ready for review, Blocked, Failed, or Cancelled with a durable evidence bundle.

The daemon queues each continuation as a new bounded run against the resident session/model runtime.
It must not recursively call the agent from an event callback or keep an uncheckpointed background
loop alive.

## Goal tools

All tools use native typed schemas and one-action workflow boundaries.

| Tool | Availability | Effect |
| --- | --- | --- |
| `propose_goal` | Discuss/Auto | Records a proposal card; grants no authority. |
| `start_goal` | Auto with matching one-use grant | Ends read-only turn and requests goal planning. |
| `goal_status` | Goal model stages and advisors | Returns a bounded authoritative goal brief. |
| `submit_goal_plan` | Goal Planning/PlanRevision | Submits the complete typed milestone plan. |
| `submit_goal_plan_review` | Goal PlanReview | Passes or challenges the exact plan digest. |
| `goal_checkpoint` | Goal coordinator/evaluation | Reports milestone result interpretation and next bounded proposal; grants no transition by itself. |
| `goal_request_amendment` | Active goal model stages | Proposes a typed objective, criterion, milestone, or authority change. |
| `goal_request_budget` | Active goal model stages | Requests a bounded increase with reason and expected payoff. |
| `goal_pause` | Active goal model stages | Requests a safe-boundary pause; cannot resume itself. |
| `submit_goal_completion` | Evaluating | Proposes criterion evidence; deterministic validation decides the outcome. |

Do not give the model direct `goal_update`, `goal_resume`, `goal_cancel`, or `goal_accept` authority.
Those names imply ownership the model does not have. A narrowing amendment may be accepted
automatically by deterministic policy; any broadening becomes a user decision request.

Tool result envelopes must name current goal ID, stage, plan digest, active milestone, remaining
budget, and whether the operation was applied, proposed, denied, or requires user input. Large goal
history is represented by deterministic receipts; raw events remain durable.

## Pause, cancel, and recovery semantics

Add a distinct cooperative pause signal rather than reusing cancellation:

- model inference may finish but no subsequent tool action starts after pause is observed;
- a running tool/check may finish or time out under its existing bound;
- the workflow/goal runner checkpoints before returning `paused`;
- managed commit either completes atomically or reconciles before pause is acknowledged; and
- the UI shows **Pausing after current action** until the checkpoint event arrives.

On daemon restart:

- a previously running goal restores as Paused;
- the exact goal and embedded workflow checkpoint digests are validated;
- policy, authority, repository root, workspace graph, Git control, and content are reconciled;
- an in-flight deterministic check or commit is reconciled from evidence rather than assumed; and
- the user must explicitly Resume.

Cancel records a terminal goal outcome and cancels any active child workflow through its existing
safe path. It preserves commits, uncommitted content, events, checkpoints, and evidence for audit.

## API design

Keep existing session endpoints compatible. Add goal operations as a distinct resource:

```text
POST   /api/goals
GET    /api/goals/{goal_id}
PATCH  /api/goals/{goal_id}/draft
POST   /api/goals/{goal_id}/approve-plan
POST   /api/goals/{goal_id}/amendments
POST   /api/goals/{goal_id}/amendments/{amendment_id}/approve
POST   /api/goals/{goal_id}/amendments/{amendment_id}/discard
POST   /api/goals/{goal_id}/pause
POST   /api/goals/{goal_id}/resume
POST   /api/goals/{goal_id}/cancel
POST   /api/goals/{goal_id}/accept
POST   /api/goals/{goal_id}/request-milestone
POST   /api/goals/{goal_id}/decisions/{request_id}
```

`POST /api/goals` accepts either an existing `session_id` or the same new-session project/model
fields needed to create a session transactionally:

```rust
struct StartGoalRequest {
    session_id: Option<String>,
    objective: String,
    criteria: Vec<GoalCriterionInput>,
    continuation: GoalContinuationPolicy,
    budget: GoalBudgetRequest,
    source_turn_ids: Vec<String>,
    auto_activation_turn_id: Option<String>,
    // New-session repository/model/attachment fields when session_id is absent.
}
```

Mutating actions include the last observed `goal_sha256` or an `If-Match` equivalent. A stale browser
cannot approve an obsolete plan, accept superseded evidence, or overwrite a newer decision.

`PATCH /draft` is valid only before the initial plan is approved. An amendment request contains the
base goal and plan digests plus explicit objective, criterion, continuation, budget, and remaining
scope changes. Approving it targets the reviewed replacement-plan digest. Discarding it leaves the
previous accepted plan paused and resumable. A repository change is rejected because version one
goals cannot cross repositories.

Extend `SessionListItem` and `SessionDetails` additively with `goal: Option<GoalSummary>` and
`active_goal: bool`. `GoalSummary` includes only bounded UI projections: ID, objective, stage,
outcome, milestone counts, current milestone title, continuation policy, budget percentages,
blocking request kind, and Ready evidence summary.

The existing session event stream remains the live transport. A separate goal stream would create
ordering and resume races. Goal events include `goal_id` and optional `milestone_id`, and session
details return the latest durable summary for recovery after a missed event.

## CLI design

Add a top-level command family rather than overloading `pb queue`:

```text
pb goal start "<objective>" --workdir <path> [--criterion <text>] [--automatic]
pb goal status <goal-id>
pb goal pause <goal-id>
pb goal resume <goal-id>
pb goal cancel <goal-id>
pb goal accept <goal-id>
```

`pb goal start` defaults to plan review before delivery. `--automatic` is explicit user authority for
bounded continuation, not model self-authorization. CLI output names current milestone, strict
workflow stage, counters, latest commit/evidence, and required decision. It never collapses Ready
for review into Complete.

The hidden harness gains deterministic goal fixtures and an explicit `pb harness goal` or
`pb harness agent --goal-contract` surface only after the core types stabilize. Harness runs record
goal policy, authority, checkpoint, plan, milestone workflows, budget, decisions, and terminal
evidence.

## Events and audit

Add bounded additive event variants:

- `GoalProposed`;
- `GoalStarted`;
- `GoalPlanSubmitted`;
- `GoalPlanReviewed`;
- `GoalPlanAwaitingApproval`;
- `GoalPlanApproved`;
- `GoalMilestoneStarted`;
- `GoalMilestoneCompleted`;
- `GoalMilestoneFailed`;
- `GoalCriterionSatisfied`;
- `GoalAmendmentRequested` and `GoalAmendmentResolved`;
- `GoalBudgetRequested` and `GoalBudgetResolved`;
- `GoalPauseRequested`, `GoalPaused`, and `GoalResumed`;
- `GoalBlocked`;
- `GoalReadyForReview`;
- `GoalCompleted`;
- `GoalFailed`; and
- `GoalCancelled`.

Events carry IDs, hashes, status, bounded labels, counters, and evidence IDs rather than complete
plans or diffs. `GoalCheckpoint` stores the resumable structured state. Existing workflow events
continue unchanged and include their normal workflow ID; the milestone record binds that workflow ID
to its goal and milestone.

Session metrics remain per invocation. Add a derived goal usage projection that sums persisted
records without duplicating energy/token counters in every event.

## Web UI specification

### Component map

Add focused components and pure mapping helpers:

```text
webui/src/components/GoalModeControl.tsx
webui/src/components/GoalModeBanner.tsx
webui/src/components/GoalStartSheet.tsx
webui/src/components/GoalAmendmentSheet.tsx
webui/src/components/GoalPlanReview.tsx
webui/src/components/GoalHeader.tsx
webui/src/components/GoalProgress.tsx
webui/src/components/GoalDrawer.tsx
webui/src/components/GoalDecisionCard.tsx
webui/src/components/GoalCompletionCard.tsx
webui/src/lib/goalUtils.ts
```

Keep network/state orchestration in `HomePage.tsx`, `ProjectsPage.tsx`, and `SessionPage.tsx`; keep
labels, progress calculations, budget formatting, and event reduction in testable helpers.

### Composer mode control

Replace the UI-only two-value state with:

```ts
type ComposerMode = "discuss" | "deliver" | "goal";
```

The segmented control renders **Discuss**, **Build**, and **Goal**. Discuss/Build continue mapping to
existing `TurnIntent` values. Goal opens `GoalStartSheet`; it is never serialized as
`intent: "goal"`.

When a goal is active, the control shows **Goal active** and does not permit a second goal. Ordinary
follow-up submission remains unavailable while mutation is running. Paused/blocked goal surfaces
offer the specific decision or resume action before a general composer.

### Persistent mode identity

The user must be able to identify Goal mode without scrolling to the first goal event or opening
details:

- On desktop and tablet, `GoalModeBanner` is sticky below the session title/navigation and above the
  scrollable transcript. It shows the textual state (**Goal running**, **Goal paused**, **Goal needs
  input**, or **Goal ready for review**), a truncated objective, milestone `n / m`, and the primary
  action. The Goal segment remains selected and reads **Goal active**.
- On mobile, the same durable state becomes a compact sticky top row: `Goal · Running · 2/4`, with
  **Details**. When a goal action is required, a safe-area-aware sticky bottom action shows Pause,
  Resume, Resolve, or Review. The transcript scrolls between these two controls; neither disappears
  when activity is appended.
- The document title and session-list badge include the bounded state, so returning to a background
  tab or session still reveals `Goal paused`, `Goal needs input`, or `Goal complete`.
- Text and accessible labels carry the meaning. Colour, spinner, and progress animation are only
  secondary cues. The banner is rendered from `GoalSummary`/`SessionDetails`, not inferred from the
  latest chat message.

After a terminal event the sticky active banner is replaced by a non-sticky completion card in the
transcript. Discuss becomes selected; the user may start a new Goal, but the completed Goal remains
available in history and the Goal drawer.

### Goal setup sheet

Use an accessible modal sheet on desktop and a full-height bottom sheet on small screens. It contains:

1. **Objective** — multiline, prefilled from the composer, required, bounded.
2. **Done when** — repeatable criterion rows with remove/reorder controls; an empty list is allowed
   only if the UI clearly says pb will propose criteria for user approval.
3. **How to continue**:
   - Review the plan, then continue automatically (recommended);
   - Ask before every milestone; or
   - Continue automatically within limits.
4. **Budget** — Compact, Standard, and Extended presets plus an Advanced disclosure showing maximum
   milestones, workflow attempts, model invocations, generated tokens, and wall time.
5. **Authority summary** — repository/focus path, local file/command authority inherited from Build,
   current network/integration policy, and **No publishing**.
6. Primary **Plan goal** and secondary **Cancel** actions.

Client validation is convenience only. The server normalizes every field and returns structured
field errors. Closing the sheet preserves the composer text. One-turn Auto activation is configured
separately from the Discuss composer and is not part of this sheet.

### Goal plan review

After read-only planning, render a plan card above the composer:

- objective and accepted Done-when criteria;
- ordered milestone cards with title, purpose, requirement/criterion badges, and expected
  components;
- assumptions and risks in collapsed sections;
- total budget and continuation policy;
- plan-review challenges, if any; and
- **Approve and start**, **Edit goal**, and **Cancel goal**.

Approval includes the displayed plan digest. Editing returns to a draft derived from the current
accepted user input, not arbitrary model prose. The UI must not show implementation as running until
the server emits `GoalPlanApproved` and `GoalMilestoneStarted`.

### Active goal header

Add a compact header below the session title:

```text
Goal · Running                                      Pause   •••
Improve parser recovery
Milestone 2 of 4 · Add bounded malformed-call retry
[Planning]—[Review]—[Implementing]—[Checks]—[Review]—[Commit]
Budget: 38% tokens · 3/8 workflows · 27 min active
```

The strict workflow progress indicator remains the source for the lower stage row. Goal progress is
the upper milestone context. This header is the expanded desktop form of the persistent mode banner.
On small screens, it collapses to the sticky status, `2 / 4`, and **Details** row specified above;
the objective and stage detail remain available in the mobile Goal sheet.

Pause changes immediately to **Pause requested** and disables duplicate clicks. Stop lives in an
overflow menu with confirmation because it is terminal. Resume is prominent only after the durable
Paused event.

### Goal drawer

Make Goal the first panel in the existing right-hand Session details drawer. On narrow layouts,
**Goal details** opens the same content as an overlay sheet.

The panel has four sections:

- **Overview** — objective, status, continuation mode, authority summary, current/next milestone.
- **Milestones** — ordered status list with workflow outcome, commit, checks, retry count, and reason
  for skipped/superseded entries.
- **Evidence** — criteria with Machine verified, Waiting for review, or Unsatisfied labels; linked
  commits/check receipts and current evidence digest.
- **Budget** — exact used/limit values for time, invocations, generated tokens, workflows,
  milestones, and advisory calls. Progress bars include text and never rely on colour alone.

Raw hashes, tool results, and detailed workflow artifacts remain in the existing Tools/Activity
details rather than duplicating them in the Goal overview.

Overview includes **Edit goal** for every non-terminal stage. Running work labels it **Pause and
edit**; Paused/Blocked work labels it **Edit goal**. `GoalAmendmentSheet` shows editable current
fields, immutable completed work, and a warning that changes are reviewed before resumption. After
replanning, the sheet becomes a before/after review with **Approve changes and resume**, **Discard
changes**, and, for incompatible scope, **Stop and start a new goal**.

### Intervention cards

Render typed cards immediately above the composer:

- **Needs input** — question, choices, free-text option where allowed, and impact of each choice.
- **Budget request** — used/limit, requested new limit, reason, expected next milestone, Approve,
  Edit, and Deny.
- **Authority request** — exact old/new path/tool/network/integration scope with no combined blanket
  approval.
- **Milestone failed** — truthful outcome, preserved commits/content, retry/replan recommendation,
  and user actions.
- **Ready for review** — completion criteria, commits, checks, caveats, Accept goal, Request another
  milestone, and Discuss result.

Only the server's current unresolved request ID is actionable. Cards from older events render as
resolved history and cannot resubmit a decision.

### Session and project lists

Extend `SessionCard` and table rows with bounded goal projections:

- active: `Goal · 2/4 · Implementing`;
- paused: `Goal paused`;
- blocked: `Goal needs input`, `Goal needs authority`, or `Goal budget reached`;
- ready: `Goal ready for review`; and
- terminal: `Goal complete`, `Goal stopped`, or `Goal stopped · budget exhausted`.

Add a **Goals** filter without replacing the existing lifecycle filters. Project and Home summary
counts continue to use session lifecycle status; a separate small count reports active/blocked goals
to avoid making one row belong to two incompatible status totals.

### Accessibility and responsive behavior

- Use an actual labelled dialog/sheet with focus moved on open, focus containment, Escape handling,
  and focus restoration to the Goal button.
- Every progress indicator exposes a text label and current/max values. Colour and icons are
  supplemental.
- Announce only meaningful durable transitions through `aria-live="polite"`; do not announce every
  token, tool result, or budget tick.
- Pause, cancel, approval, and authority buttons have distinct accessible names and disabled/busy
  states.
- Milestone reordering in draft mode supports buttons/keyboard, not drag-only interaction.
- Mobile sticky controls include `env(safe-area-inset-bottom)` and retain the existing
  `viewport-fit=cover`/PWA requirements.
- Respect reduced-motion preferences for progress transitions and spinners.

### Web UI state table

| Goal stage | Header label | Primary bottom action | General composer |
| --- | --- | --- | --- |
| Planning/PlanReview | Planning goal | Pause | hidden |
| AwaitingPlanApproval | Review goal plan | Approve and start | hidden |
| RunningMilestone | Milestone n of m | Pause | hidden |
| Evaluating | Checking goal progress | Pause | hidden |
| Paused | Goal paused | Resume | available only for Discuss after resume/cancel decision |
| Blocked | Needs input/authority/budget | Resolve request | replaced by typed decision form |
| AwaitingUserReview | Ready for review | Accept goal | Discuss also available |
| Completed | Goal complete | Discuss result | available, Discuss selected |
| Failed | Goal stopped | Start revised goal | available, Discuss selected |
| Cancelled | Goal cancelled | Discuss result | available, Discuss selected |

Discussing a Ready-for-review result is read-only and does not amend the accepted goal. The user
must choose **Request another milestone** or approve a typed amendment before goal work resumes.

### Ending Goal mode

Goal mode controls the session until the server publishes a terminal stage. UI navigation, closing
the browser, pausing, blocking, reaching Ready for review, or merely discussing the result do not end
it. The terminal paths are:

| Trigger | Terminal stage | What the user sees next |
| --- | --- | --- |
| All machine criteria verify | Completed | completion card; Discuss selected |
| User accepts current review evidence | Completed | accepted completion card; Discuss selected |
| User confirms Stop goal | Cancelled | stopped card with preserved-work summary; Discuss selected |
| Unrecoverable bounded failure or hard ceiling | Failed | reason/evidence card and Start revised goal; Discuss selected |

Terminal controls are disabled until the safe-boundary event is durable. The completion card states
whether completion was Machine verified or User accepted. Starting a revised goal creates a new goal
ID and may cite the prior goal's evidence, but cannot make a terminal goal resumable.

## Security and privacy implications

- Auto start is a one-use user/session grant, not a project-configured capability.
- Automatic continuation starts only already-approved milestones within the same authority envelope.
- Budget and authority requests are separate. Approving more tokens must not also approve more paths
  or network access.
- Goal summaries and notifications must not include secret environment values, raw tool output, or
  remote credentials.
- Public research and MCP/LSP use retain their existing network/privacy policy and are visible in the
  authority summary.
- Goal mode does not add an external publication tool or weaken the local Ready boundary.
- The persisted checkpoint contains user objectives and decisions and follows the same local data
  ownership and deletion rules as sessions.

When implementation lands, update `docs/architecture/security.md` for activation/delegation and
authority requests; `docs/architecture/local-privacy.md` for persistence and any background
continuation path; `docs/architecture/workflows.md` and `user-contracts.md` for lifecycle and
terminal claims; and the matching user guides for commands, configuration, data locations, and
cleanup.

## Deterministic and real-model evaluation

Build a goal-control corpus before broad web rollout. Deterministic fixtures must cover:

- explicit start, proposal-only, valid Auto start, wrong-turn Auto start, and consumed grant replay;
- plan approval bound to the current digest and stale-browser rejection;
- milestone sequencing with no concurrent workflow;
- total budget accounting across several workflows and retries;
- child workflow Ready, NoChange, Blocked, Failed, Cancelled, and malformed-action outcomes;
- pause requested during model inference, tool execution, checking, and managed commit;
- restart restoration to Paused with exact remaining budget;
- Git-control or content drift between milestones;
- narrowing amendments, broadening requests, denial, and stale decision IDs;
- user amendment from a running safe-boundary pause, revised-plan approval, discard-to-old-plan,
  superseded milestones, and carried/invalidated evidence;
- machine completion, Ready for review, user acceptance, and request-another-milestone;
- false completion, missing criterion evidence, stale evidence, and unexpected changes;
- goal cancellation preserving committed and uncommitted work; and
- legacy sessions restoring with no fabricated goal state.

Web tests cover setup submission, plan approval, persistent desktop/mobile mode identity, progress
labels, active drawer projections, amendment edit/review/discard, decision cards, budget formatting,
list/document-title badges, restart state, terminal composer restoration, stale-digest conflicts,
keyboard behavior, and safe-area CSS.

After deterministic success, use a locked small-model matrix with Qwen3 4B,
Qwen2.5-Coder 7B, and the available 14B coder model. Measure separately:

- activation/tool compliance;
- valid goal-plan and goal-review submissions;
- milestone artifact quality;
- complete-goal rate;
- false-completion and authority-escape rate;
- recovery turns and no-progress stops;
- total prompt/generated tokens, latency, and energy; and
- user-intervention count.

The rollout gate is zero false completion and zero authority escape. A low 4B completion rate may
justify explicit-only activation or simpler coordinator schemas; it must not justify weakening
evidence or authority gates.

## Workstreams and commit boundaries

### G0 — Close activation and parse-recovery gaps (P1)

- Make harness Discuss/Auto allowlists compatible with their required transition/proposal tools.
- Route malformed textual native-tool arguments into bounded model parse correction rather than
  backend `engine_error`.
- Add full-request tests that exercise actual harness allowlists and llama textual-tool parsing.
- Preserve zero execution for malformed calls and the existing parse/no-progress ceilings.

Acceptance:

- a valid `start_delivery` is exposed and accepted only in Auto;
- Discuss still cannot self-start;
- a malformed `submit_implementation` receives a correction and can recover on the next bounded
  turn; and
- no malformed argument is guessed, partially executed, or awarded evidence.

Commit: `fix: recover bounded workflow transitions`

### G1 — Add goal configuration and durable reducer types (P0)

- Add `src/goal/` types, artifact validation, pure reducer, policy hashing, checkpoint hashing,
  counters, summaries, criteria, decisions, and evidence bundles.
- Add `.pb/goal.toml` loading and hard ceilings; update `src/init.rs` and configuration/user docs.
- Extend persisted sessions additively with active/completed goals.
- Add compatibility tests for legacy session notes and unknown/future versions.

Acceptance:

- every transition table edge has reducer coverage;
- tampered policy, authority, plan, budget, child workflow, decision, or evidence fails validation;
- project configuration cannot enable Auto authority; and
- old sessions restore without goal claims.

Commit: `feat: add durable goal state`

### G2 — Enforce goal planning and approval (P0)

- Add goal-stage capabilities and typed plan/plan-review terminal tools.
- Produce bounded goal briefs from conversation handoff and repository/workspace evidence.
- Run fresh-context goal plan critique and bounded revision cycles.
- Implement manual/review-plan/automatic continuation policy, with user approval bound to the plan
  digest.
- Pause unresolved human decisions through typed decision requests.

Acceptance:

- planning/review cannot mutate or run shell commands;
- a rejected or stale plan cannot start a milestone;
- every milestone traces to requirements and criteria and fits configured ceilings; and
- model assumptions cannot replace unresolved user decisions.

Commit: `feat: enforce reviewed goal plans`

### G3 — Orchestrate milestone workflows and total budgets (P0)

- Embed exactly one active strict workflow in the active milestone.
- Compile milestone handoff/contract input and reuse the shipped workflow engine.
- Roll workflow usage into goal counters and cap child limits by remaining total budget.
- Reconcile commit/content/Git control between milestones.
- Implement bounded replan after milestone failure and deterministic repeated-no-progress stop.
- Evaluate criteria after every milestone.

Acceptance:

- no goal path can mutate, check, review, or commit outside the strict workflow;
- a second milestone cannot start while another workflow is active;
- starting another workflow never resets total budget; and
- stale or unexpected repository state blocks continuation truthfully.

Commit: `feat: orchestrate bounded goal milestones`

### G4 — Add goal proposals, Auto grants, amendments, and pause (P1)

- Add `propose_goal`, one-use `start_goal`, status, amendment, budget-request, pause, and completion
  tools with stage schemas.
- Validate Auto grants against session, source turn, expiry/use, and authority.
- Implement monotonic amendment classification and typed user decision requests.
- Add the safe-boundary user amendment lifecycle, plan versioning, superseded-milestone history, and
  deterministic evidence carry-forward/invalidation.
- Add cooperative pause checks and exact resume reconciliation.

Acceptance:

- proposal and start tools never expose mutation in discussion;
- a replayed, wrong-turn, expired, or absent Auto grant fails closed;
- narrowing may apply without broadening; broadening always pauses for a decision;
- changing a completion criterion always produces a reviewed completion-contract delta;
- Pause never reports success before a durable safe-boundary checkpoint; and
- the model cannot resume, cancel, accept, or directly rewrite the goal.

Commit: `feat: add bounded goal controls`

### G5 — Add API, CLI, events, and persistence recovery (P1)

- Add goal and amendment endpoints, optimistic digest checks, bounded summaries, and session event
  variants.
- Add `pb goal` commands and daemon client operations.
- Persist goal checkpoints in session notes and restore interrupted goals as Paused.
- Make session list/detail workflow projections goal-aware without duplicate canonical workflow
  state.
- Extend harness journals and audit summaries.

Acceptance:

- API, CLI, web, restored sessions, and journals agree on stage/outcome/budget;
- stale UI/API mutations return conflict without altering state;
- restart cannot silently resume mutation; and
- deletion removes goal state under the existing session data contract.

Commit: `feat: expose durable goal sessions`

### G6 — Build Goal creation and plan-review UI (P1)

- Add the three-mode composer control and Goal setup sheet to Home, Projects, and completed Session
  composers.
- Add the persistent Goal mode banner and compact mobile state row from durable session state.
- Add criterion editing, continuation choices, budget presets/advanced limits, authority summary,
  registered-project selection on Home, and the separate advanced one-turn Auto option on Discuss.
- Render goal proposals and the digest-bound plan approval card.
- Add responsive, focus, keyboard, reduced-motion, and safe-area behavior.

Acceptance:

- Goal is never serialized as `TurnIntent`;
- closing/reopening preserves objective text without preserving an unconfirmed grant;
- plan approval targets the exact displayed digest;
- a second active goal cannot be created; and
- desktop/mobile users can identify Goal mode and its durable state without opening the activity
  feed; and
- focused web tests and `deno task test:web` pass.

Commit: `feat: start goals from the web ui`

### G7 — Build active Goal monitoring and intervention UI (P1)

- Add Goal header, milestone/workflow progress, right/mobile drawer, evidence and budget sections,
  list badges/filter, and typed decision cards.
- Add Edit goal/Pause and edit, amendment before/after review, discard-to-prior-plan, and incompatible
  Stop-and-start-new flows.
- Add pause/resume/cancel, budget/authority decisions, Ready-for-review, Accept, and Request another
  milestone actions.
- Restore the normal Discuss composer only after a durable terminal outcome and retain a completion,
  stopped, or failure card in history.
- Keep raw details in existing drawer/activity surfaces.

Acceptance:

- every state in the UI table renders from API state without inferring authority from prose;
- actions disable while pending and stale request IDs cannot be resubmitted;
- accessibility and responsive tests cover desktop/mobile state; and
- Paused, Blocked, Budget reached, and Ready for review never visually masquerade as terminal; and
- `deno task test:web` passes.

Commit: `feat: show and control goal progress`

### G8 — Evaluate, document, and roll out (P0)

Status: **Complete for version one.** The deterministic Goal corpus passes, active-Goal context and
request auditing are available in `pb harness agent`, and the locked model evidence plus rollout
decision are recorded in the [G8 report](benchmarks/goal-mode-g8.md). The field result deliberately
does not promote one-turn Auto to a normal web default: proposal/tool compliance was not reliable
enough across the matrix, although deterministic authority containment held.

- Add the deterministic goal corpus and web/API/harness parity fixtures.
- Run the locked 4B/7B/14B matrix only after deterministic control passes.
- Update curated architecture, security, privacy, user, configuration, and cleanup documentation.
- Add goal status/progress to harness reports without calling subjective artifact quality verified.
- Roll out explicit/manual first, then review-plan continuation, then optional automatic
  continuation and Auto activation only when their separate gates pass.

Acceptance:

- zero false completion, authority escape, budget reset, and silent restart continuation;
- current docs distinguish Shipped, Configurable, and Design record behavior;
- `cargo test --all-targets`, `deno task test:web`, and `deno task test:docs` pass; and
- representative real-model runs preserve full events/checkpoints and report limitations
  separately from pb defects.

Commit: `test: validate bounded goal control`

## Rollout order

1. Land G0 independently and rerun the existing workflow small-model corpus.
2. Land G1-G3 behind API/harness-only feature availability with no web control.
3. Land G4-G5 and prove restart, pause, budget, amendment, and terminal truth deterministically.
4. Land G6 with explicit start and manual continuation only.
5. Land G7, then enable review-plan continuation after web/API parity passes.
6. Run G8 model evaluation. Keep 4B explicit-only if activation or structured coordination remains
   unreliable.
7. Offer automatic continuation as an explicit per-goal choice.
8. Offer one-turn Auto start last; never make it a repository-controlled default.

Each step must be independently revertible without making stored goal/checkpoint data claim a later
stage than the running binary understands.

## Verification

Run focused tests throughout, then before declaring the implementation complete:

```bash
cargo fmt --check
cargo test --all-targets
deno task test:web
deno task test:docs
deno task build:web
cargo build --release --target aarch64-apple-darwin
```

Goal orchestration changes do not by themselves require the FlashMoe architecture smoke. If a
workstream changes FlashMoe data flow, scheduling, model state, or Metal behavior, update the
FlashMoe architecture record first and run the smoke required by `AGENTS.md`.

## Completion contract

This design goal is implemented only when:

- a user can explicitly create, review, start, observe, pause, resume, cancel, and accept a goal in
  the web UI and CLI;
- one goal can complete at least two sequential strict-workflow milestones on one goal-owned branch;
- goal, child workflow, authority, policy, budget, decision, and evidence state survive restart and
  reconcile before resume;
- model tools can propose activation/amendment/completion but cannot self-grant authority, budget,
  acceptance, resume, cancellation, or publication;
- machine completion and Ready for review remain truthful and visibly distinct;
- total budgets bind all model work and cannot reset across milestones or retries;
- deterministic control and UI corpora pass with zero false completion or authority escape;
- the locked small-model evaluation is recorded with model limitations separated from pb defects;
- user-visible and curated architecture/security/privacy documentation describe the shipped result;
  and
- all required repository checks pass in the same semantic commit series.
