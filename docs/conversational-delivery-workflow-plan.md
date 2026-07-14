# Conversational sessions and enforced delivery workflow plan

Status: implemented through W8; W9 publication-seam verification remains

## Decision summary

Keep the project experience conversational while making repository delivery a
separate, durable, harness-owned workflow:

- A project session is a conversation that may contain zero or more delivery
  workflow runs.
- Discussion turns are read-only and require no branch, plan, checks, review, or
  commit.
- A delivery run follows a fixed version-one state machine: plan, plan critique,
  implementation, deterministic checks, isolated code critique, bounded repair,
  and managed commit.
- The harness, not the model, chooses stage transitions and exposes
  stage-specific capabilities.
- Implementation and repair retain model-supplied `run_command` as a practical
  escape hatch. Its activity is journaled and reconciled but is not treated as
  contained, reviewed, checked, or committed merely because the command exits
  successfully.
- Model-requested teammates remain useful for bounded fresh-context advice, but
  they are read-only, cannot advance workflow state, and cannot acquire
  authority their caller lacks.
- Structured artifacts and content fingerprints connect every accepted plan,
  check, review, and commit. Prose never overrides a missing or stale fact.
- Web, queue, and `pb harness agent` use the same workflow engine. Their
  presentation and model prompts may differ, but their transitions and terminal
  outcomes do not.
- Version one uses a typed fixed workflow rather than a user-programmable DAG.
  Project configuration selects policy and budgets without permitting invalid
  stage orderings.
- Push, pull-request/merge-request creation, CI waiting, and review-feedback
  repair are a later external-publication workflow. Version one must leave a
  durable `ready` result and provider seam that can support those stages without
  weakening local delivery gates.

This plan builds on the implemented workspace/check/handoff design in
`docs/agent-handoff-workspace-plan.md`. It replaces prompt-directed review and
unsafe generic delegation as sources of workflow control; it does not replace
the existing repository ownership, check evidence, executor, or managed-commit
machinery.

## Outcome

A user can remain in one project conversation to brainstorm, rubber-duck,
investigate, implement, discuss the result, and initiate another change.
Discussion stays lightweight and read-only. When delivery begins, pb can
truthfully show and enforce:

1. a structured plan was produced;
2. a fresh-context critic challenged that exact plan;
3. blocking plan challenges were answered or the plan was revised;
4. implementation occurred only after plan acceptance;
5. configured checks passed for the final task-owned content;
6. a fresh isolated reviewer challenged that exact final content with the check
   evidence;
7. any repair invalidated stale checks and review evidence;
8. a managed semantic commit contains only the accepted task-owned content; and
9. the workflow stopped with a specific non-success outcome if any required fact
   remained missing.

The workflow can prove that the configured process ran and that named executable
evidence is current. It cannot prove that a weak model reasoned deeply, that two
invocations of one model are independent experts, or that incomplete project
tests establish product correctness. UI and CLI language must preserve those
distinctions.

## Baseline and motivating defects (historical)

The defects in this section describe the pre-workflow baseline. New requests now use the
harness-owned workflow described below; prompt-owned gates survive only for restored persisted
requests that predate conversation intent.

The current implementation already provides strong building blocks:

- task and invocation baselines;
- task-owned path calculation and safe managed commits;
- a typed workspace graph with affected-component check selection;
- reusable fingerprinted check evidence;
- explicit harness acceptance contracts;
- isolated review workspaces;
- bounded parse, tool-loop, gate, and handoff repair behavior; and
- shared `run_agent` use by web and harness entry points.

The current sub-agent/control behavior is not an acceptable workflow boundary:

- every non-research profile can request a build or scout child, allowing a
  nominally read-only parent to cause source-workspace mutations;
- a build child at the one-level nesting limit cannot request its required
  review and is excluded from top-level deterministic handoff;
- deterministic handoff depends on the inferred root profile rather than the
  existence of a task-owned mutation or active delivery workflow;
- ordinary review credit can be earned with an arbitrary successful read and
  command plus a literal `REVIEW PASS` response;
- each child is individually bounded, but there is no global delegation,
  invocation, or generated token budget across a run;
- the prompt says a teammate result is summarized, while the implementation
  returns the child's unbounded final text directly to the parent context; and
- workflow state does not survive as a first-class persisted object or appear as
  a stage in the web API.

These are control-plane defects even when the underlying model behaves well.
Workstream W0 adds deterministic regressions before changing behavior.

## Product model

### Conversation and delivery are different scopes

`SessionStatus` continues to describe process lifecycle (`queued`, `running`,
`paused`, `completed`, or `failed`). Do not overload it with workflow stages.

A session contains conversation turns and an optional active delivery workflow:

```rust
struct ConversationSession {
    session_id: String,
    turns: Vec<ConversationTurn>,
    active_workflow: Option<WorkflowRun>,
    completed_workflows: Vec<WorkflowSummary>,
}

enum TurnIntent {
    Discuss,
    Deliver,
    Auto,
}
```

The current persisted event history remains the rendered conversation. The
workflow state is an additive persisted snapshot with structured stage events
for audit.

### Starting delivery

Delivery can begin in three ways:

1. the user selects **Build** when submitting a new or follow-up turn;
2. an API or `pb harness agent` request explicitly uses delivery intent; or
3. in `Auto`, the discussion agent requests a transition using
   `start_delivery(source_turn_id, task_summary)`.

`start_delivery` grants no mutation itself. It ends the discussion invocation
and asks the harness to create a workflow from the cited current user turn and
relevant prior decisions. A false positive therefore starts a stricter process
rather than granting direct write authority.

Explicit **Discuss** intent never auto-promotes. The model may emit a
`DeliveryProposal` card, but the user must select **Build** or send a later
`Deliver` turn. When an outstanding proposal exists, a natural follow-up such as
“go ahead” may cite that proposal and start delivery without another
confirmation round.

### Returning to conversation

A successful, failed, blocked, or cancelled workflow returns the session to the
ordinary composer. The user can ask questions about the result without starting
another workflow. A later delivery turn creates a new workflow ID, baseline,
plan, evidence set, and commit result while retaining the conversation as
non-authoritative context.

### User-facing language

Use **Discuss** and **Build** in the UI. Avoid exposing internal terms such as
gate, fingerprint, artifact hash, or sub-agent in the main conversation.

Show a compact durable stage label during delivery:

- Planning
- Challenging the plan
- Implementing
- Running project checks
- Reviewing the result
- Repairing findings
- Committing
- Ready / Needs another pass / Needs help

Raw artifacts, commands, evidence, and transition reasons remain available under
an expandable details surface and in harness journals.

## Invariants

1. Discussion cannot mutate repository content, the Git index, HEAD, remotes, or
   external systems through built-in tools.
2. Starting a delivery workflow is an explicit harness transition, not a
   model-granted capability.
3. The active stage is the only source of model tool authority.
4. Authority is transitive: a requested teammate cannot receive capabilities
   unavailable to its caller or stage.
5. Model-requested teammates cannot transition workflow state.
6. A stage does not advance on prose or a generic model final; it advances only
   through the named structured submission accepted by the harness.
7. Plan critique references the exact accepted plan content hash.
8. Check evidence and code-review evidence reference the current task-owned
   content fingerprint.
9. Any content mutation after checks or review makes affected evidence stale.
10. The model receives no dedicated `git_commit`, push, or provider-publication
    tool in a delivery stage, and shell activity never counts as the managed
    commit or a publication transition. The implementation escape hatch can
    still invoke external programs, including Git, unless the host restricts it;
    pb must block on detected HEAD, index, or ref mutation and must not claim it
    can reverse every external side effect. Accepted commit and future
    publication transitions remain harness-owned.
11. Arbitrary `run_command` is available only in implementation and repair as an
    explicit build escape hatch. Discussion, planning, plan review, checking,
    and code review do not expose it. Configured `run_task(id)` and
    `run_check(id)` remain the preferred auditable operations where practical.
12. A failed check or blocking review finding prevents managed commit.
13. Step, invocation, token, advisory-call, plan-cycle, and repair-cycle limits
    are global workflow facts and cannot be multiplied through delegation.
14. A daemon restart, harness resume, or paused user question preserves the
    exact workflow stage, artifacts, budgets, baseline, and evidence references.
15. Existing stored sessions and requests remain readable. Missing workflow
    fields deserialize to legacy behavior only for already persisted work; new
    requests use the current project policy.
16. Project-local configuration is snapshotted and hashed when a workflow
    starts. Mid-run config edits do not silently change its contract.
17. Web and harness terminal status is derived from the same `WorkflowOutcome`.
18. Discussion and delivery outcomes do not claim externally verified task
    completion without an explicit satisfied acceptance contract.

## Non-goals for version one

- Proving that a model genuinely considered every risk instead of producing
  plausible review text.
- Treating the same local model in a fresh context as an independent security
  boundary.
- Inferring a complete acceptance contract from arbitrary user prose.
- Supporting an arbitrary project-authored workflow graph or executable hooks at
  every stage.
- Claiming filesystem, process, or network containment for implementation while
  its `run_command` escape hatch executes in the host environment. Container,
  network, and isolated-workspace hardening are later defence-in-depth work.
- Automatically pushing, opening a PR/MR, resolving review threads, or merging.
  The local result must be ready for a later publication workflow, but those
  external mutations require their own implementation and approval policy.
- Replacing project-authored tests, CI, task runners, or human review for
  high-risk changes.

## Configuration

Add `.pb/workflow.toml`, separate from:

- `.pb/workspace.toml` for components, executors, checks, and dependencies;
- `.pb/environment.toml` for environment preparation;
- `.pb/policy.toml` for user/tool authorization; and
- harness acceptance contracts for task-specific externally supplied facts.

Version one is deliberately small:

```toml
version = 1

delivery = "strict"              # strict is the only v1 delivery policy
default_intent = "discuss"       # discuss | auto | deliver

[limits]
stage_steps = 8
total_model_invocations = 40
total_generated_tokens = 24000
advisory_calls = 4
plan_cycles = 2
repair_cycles = 2
review_paths = 40
review_diff_bytes = 200000
```

Validation rules:

- version one accepts only `delivery = "strict"`, which always means plan, plan
  critique, affected checks, code critique, and managed commit;
- all limits must be positive and subject to hard runtime ceilings;
- review scope limits must fit within the configured workflow token/invocation
  ceilings;
- unknown fields and enum values fail configuration loading with an actionable
  path; and
- version-one configuration cannot reorder or omit safety-critical stages
  through a custom DAG.

The built-in policy for new project delivery requests is the same strict policy.
There is no project-configured relaxed delivery mode in version one: users
either discuss without mutation or build through the complete workflow.
Compatibility behavior remains available only when restoring an already
persisted legacy request or running an explicitly legacy deterministic fixture.

`pb init` must inspect, explain, generate, and preserve `.pb/workflow.toml`. As
required by `AGENTS.md`, every future workflow schema addition must update
`src/init.rs` in the same change.

## Durable state model

Add a workflow module with serialized versioned types. Suggested layout:

```text
src/workflow/
├── mod.rs          # public engine and state projection
├── config.rs       # .pb/workflow.toml parsing and normalization
├── artifacts.rs    # plan and review schemas/validation
├── capabilities.rs # stage tool matrix and delegation authority
├── engine.rs       # transition loop and bounded repair
└── persistence.rs  # checkpoint/event projection helpers
```

Core types:

```rust
enum WorkflowStage {
    Planning,
    PlanReview,
    PlanRevision,
    Implementing,
    Checking,
    CodeReview,
    Repairing,
    Committing,
    Ready,
    Failed,
    Blocked,
    Cancelled,
}

enum WorkflowOutcome {
    Ready,
    NoChange,
    PlanRejected,
    PlanCyclesExhausted,
    ChecksFailed,
    ReviewFailed,
    RepairCyclesExhausted,
    ExecutorUnavailable,
    CommitBlocked,
    StepLimit,
    InvocationLimit,
    TokenLimit,
    EngineError,
    Cancelled,
}

struct WorkflowRun {
    version: u32,
    id: String,
    source_turn_id: String,
    task: String,
    stage: WorkflowStage,
    policy: CompiledWorkflowPolicy,
    policy_sha256: String,
    repository: RepositoryContext,
    plan: Option<ArtifactEnvelope<PlanArtifact>>,
    plan_review: Option<ArtifactEnvelope<PlanReviewArtifact>>,
    checks: CheckEvidenceLedger,
    code_review: Option<ArtifactEnvelope<CodeReviewArtifact>>,
    counters: WorkflowCounters,
    commit: Option<HandoffCommitSummary>,
    outcome: Option<WorkflowOutcome>,
    blocked_reason: Option<String>,
}
```

`WorkflowRun` is a checkpoint, not a replacement for events. Persist the latest
complete checkpoint in `PersistedSession` so history trimming cannot destroy
resumability. Emit additive immutable workflow events for audit. Harness scratch
runs persist the same checkpoint in the run directory and include its digest in
the run index.

Every transition validates a pure `transition(state, event) -> Result<state>`
function. Both live execution and restoration use the same reducer. Invalid,
repeated, or out-of-order transition events fail closed.

## Structured artifacts

### Plan artifact

The planning stage receives the current user task, an approved compact
conversation handoff, repository instructions, architecture documents, workspace
graph, and read-only repository tools. It must call `submit_plan(plan)` with:

```rust
struct PlanArtifact {
    summary: String,
    requirements: Vec<PlanRequirement>,
    steps: Vec<PlanStep>,
    acceptance: Vec<PlanAcceptance>,
    risks: Vec<PlanRisk>,
    assumptions: Vec<String>,
    open_questions: Vec<String>,
}

struct PlanRequirement {
    id: String,
    description: String,
    source: String,
}

struct PlanStep {
    id: String,
    requirement_ids: Vec<String>,
    component_ids: Vec<String>,
    paths: Vec<PlanPath>,
    description: String,
}

struct PlanPath {
    path: String,
    change: PlannedChange, // create | modify | delete
}

struct PlanAcceptance {
    id: String,
    requirement_ids: Vec<String>,
    check_ids: Vec<String>,
    description: String,
}
```

Deterministic validation requires:

- non-empty requirements, steps, and acceptance facts;
- unique stable IDs;
- every requirement mapped to at least one step and acceptance fact;
- every referenced component/check present in the normalized workspace graph;
- repository-contained normalized paths;
- modify/delete paths to exist at the plan snapshot, while explicitly marked
  create paths may be absent;
- no unresolved open question at acceptance; an open question must pause through
  `ask_user` or keep the plan unaccepted; and
- plan size within configured bounds.

The harness cannot validate whether prose meaning is correct. It can validate
coverage structure, repository references, and that later stages use the
accepted artifact.

### Plan critique artifact

The plan reviewer starts in a fresh read-only context with the task, plan,
workspace graph, repository instructions, and relevant source paths. It does not
receive the planner's reasoning transcript.

It must call `submit_plan_review(review)`:

```rust
struct PlanReviewArtifact {
    plan_id: String,
    plan_sha256: String,
    assessments: Vec<PlanAssessment>,
    challenges: Vec<ReviewChallenge>,
    verdict: ReviewVerdict,
}

enum PlanAssessmentKind {
    RequirementCoverage,
    Architecture,
    ComponentImpact,
    TestStrategy,
    FailureModes,
    Assumptions,
}

struct PlanAssessment {
    kind: PlanAssessmentKind,
    status: AssessmentStatus,
    evidence: Vec<EvidenceReference>,
    explanation: String,
}

struct ReviewChallenge {
    id: String,
    severity: ReviewSeverity,
    requirement_ids: Vec<String>,
    description: String,
    evidence: Vec<EvidenceReference>,
}
```

The harness requires exactly one assessment for every configured dimension,
verifies read evidence for referenced repository paths, and rejects a pass
containing blocking challenges. It does not require a fabricated negative
finding when the plan is sound.

Blocking challenges transition to `PlanRevision`. The revised plan must identify
the challenge IDs it resolves. The next critic receives the new plan plus
unresolved challenges; plan cycles are globally bounded.

### Implementation artifact

The implementer receives the accepted plan, accepted plan review, current
workspace state, and stage capabilities. It cannot alter the accepted plan. If
implementation discovers a material requirement or architecture change, it calls
`request_replan(reason)` and returns to planning. It must not silently diverge
and later explain the divergence in its final response.

`submit_implementation` records a concise implementation summary and any plan
step it could not complete. The artifact must account structurally for every
accepted plan step; missing or explicitly incomplete steps block advancement.
This does not prove semantic completion. The harness independently calculates
changed paths, verifies that reported touched paths agree with the actual delta,
and selects checks from actual changed content rather than the model's summary.

### Code critique artifact

The code reviewer receives a fresh isolated snapshot containing the exact final
content, plus:

- task and requirements;
- accepted plan and plan critique;
- final changed paths and diff;
- selected components;
- current check evidence and bounded outputs; and
- repository instructions and architecture documents.

It does not receive the implementer's reasoning transcript.

`submit_code_review(review)` contains structured findings and one assessment for
correctness, requirements, architecture, tests, regressions, and
maintainability. A finding includes severity, path/location, affected
requirement/plan IDs, evidence, and an actionable explanation.

A pass is accepted only when:

- the review references the current content fingerprint;
- all changed implementation paths were read, subject to a bounded exception for
  generated or binary outputs represented by producer/check evidence;
- all required assessment dimensions are present;
- all selected required checks are current and successful; and
- no blocking finding remains.

If changed source paths or diff bytes exceed the configured review scope, pb
must not silently sample a subset and call the result reviewed. Version one
blocks the workflow with an actionable request to split the task or obtain an
explicitly broader/human review. Partitioned multi-review coverage can be added
later with its own aggregate evidence contract.

Blocking findings transition to `Repairing`. The repair stage receives finding
IDs, not the entire review transcript. Any repair invalidates affected check and
review evidence, reruns affected checks, and starts a new isolated code review.
Repair cycles are globally bounded.

## Stage transition table

| From             | Required current facts                                    | Accepted action              | Next         |
| ---------------- | --------------------------------------------------------- | ---------------------------- | ------------ |
| Discuss          | Current user turn or accepted delivery proposal           | `start_delivery`             | Planning     |
| Planning         | Structurally valid plan, no open questions                | `submit_plan`                | PlanReview   |
| PlanReview       | Review references current plan hash                       | blocking challenges          | PlanRevision |
| PlanReview       | Complete assessments, no blockers                         | passing `submit_plan_review` | Implementing |
| PlanRevision     | Revised plan resolves or retains challenge IDs explicitly | `submit_plan`                | PlanReview   |
| Implementing     | Accepted plan hash still current                          | `request_replan`             | Planning     |
| Implementing     | Task-owned delta or explicit valid no-change result       | `submit_implementation`      | Checking     |
| Checking         | Missing/stale selected checks                             | harness execution            | Checking     |
| Checking         | Failed required checks and repair budget remains          | harness feedback             | Repairing    |
| Checking         | All selected checks current and successful                | harness result               | CodeReview   |
| CodeReview       | Review references current content fingerprint             | blockers                     | Repairing    |
| CodeReview       | Complete assessments, no blockers                         | passing `submit_code_review` | Committing   |
| Repairing        | Findings/check failures supplied, plan still applicable   | `submit_implementation`      | Checking     |
| Committing       | Current checks and review match current fingerprint       | managed commit               | Ready        |
| Any active stage | Global budget exhausted or deterministic repeated failure | harness stop                 | Failed       |
| Any active stage | Missing executor/user/external prerequisite               | harness stop/pause           | Blocked      |

No-change delivery still produces and reviews a plan. After implementation, if
the final task-owned content delta is empty, change-conditioned checks and code
review stay out of the way, `Always` checks may run, no commit is created, and
the workflow finishes with `NoChange`.

## Capability and delegation model

### Stage tool matrix

Use an explicit `StageCapabilities` value to construct schemas and re-check
every call at execution. Do not infer authority from `AgentProfile`.

| Capability                 |       Discuss |      Plan |   Plan review |  Implement/repair |   Code review |
| -------------------------- | ------------: | --------: | ------------: | ----------------: | ------------: |
| Repository read/search/LSP |           yes |       yes |           yes |               yes |           yes |
| Public research            |        policy |    policy |        policy |            policy |        policy |
| Built-in file mutation     |            no |        no |            no |               yes |            no |
| `run_task(id)`             |            no |        no |            no |        configured |            no |
| `run_check(id)`            |            no |        no | evidence only |        configured | evidence only |
| Arbitrary `run_command`    |            no |        no |            no | escape hatch: yes |            no |
| Managed commit             |            no |        no |            no |                no |            no |
| Workflow submission        | proposal only |      plan |   plan review |    implementation |   code review |
| Advisory teammate          |     read-only | read-only |     read-only |         read-only |     read-only |

`run_task(id)` is a new trusted project-configured operation for formatting,
generation, or bounded diagnostics that should not count as check evidence. Add
an explicit task declaration to `.pb/workspace.toml`:

```rust
struct WorkspaceTask {
    id: String,
    label: String,
    command: String,
    cwd: String,
    executor: String,
    allowed_changes: Vec<String>,
    timeout_seconds: u64,
}
```

The command is trusted project configuration; the model supplies only its ID.
Run it in an isolated task snapshot, validate the resulting content delta
against `allowed_changes`, and promote only declared output changes back through
controlled file operations. An empty `allowed_changes` means the task is
diagnostic and no source delta is promoted. Task execution never receives check
credit, and unexpected HEAD, index, ref, or out-of-scope content changes in the
task snapshot fail the task. Project task configuration is authority and must
receive the same path, timeout, executor, and cycle validation as checks; it
must not be generated from model-supplied shell text.

`run_command` remains available to the implementation and repair stages because
real builds routinely need project-specific commands that have not yet been
declared as tasks. Treat this as a deliberately broad authority grant, not as a
sandbox or a security boundary. Every invocation and bounded output must be
journaled. After it returns, pb recalculates repository state and the task-owned
content delta; unexpected HEAD, index, or ref changes block the workflow, while
ordinary content mutations invalidate affected check and review evidence.
`run_command` cannot submit a stage artifact, advance the state machine, award
check credit merely because it exited successfully, or perform the harness-owned
managed commit. Depending on the host environment it may still affect processes,
files outside the repository, the network, or other external state; version one
states that limitation plainly and relies on existing user policy/approval
controls. A later defence-in-depth workstream should constrain it with container
execution, isolated workspaces, network policy, and tighter filesystem mounts
without removing the useful escape hatch.

Checks remain harness-owned at the mandatory checking stage even when the
implementer ran a named check while iterating. Current reusable evidence may
avoid rerunning it only when command, executor, inputs, dependencies, outputs,
and fingerprint still match.

### Read-only isolation

Plan-review and code-review stages use isolated snapshots. Planning and
discussion use no generic command capability and therefore may inspect the
source workspace directly. If read-only stages later gain project tasks, run
them in an isolated snapshot and reject the artifact if the snapshot was mutated
unexpectedly.

External MCP/app tools are governed by `.pb/policy.toml`; strict read-only
defaults must not assume an arbitrary connector is non-mutating. Only explicitly
classified read capabilities should be available by default.

### Advisory teammates

Replace the generic target matrix with typed advisory roles:

- explore;
- research;
- focused read-only review; and
- monitor.

Planning is a workflow stage, not an advisory prerequisite. Build and scout are
not model-requested advisory targets. Environment scouting that changes
configuration is a separate explicit delivery task.

Each advisory request includes a concrete task, allowed evidence scope, maximum
steps, maximum tokens, and a required bounded result schema. The harness
decrements the workflow-wide budget before starting it. Advisory calls execute
sequentially by default on local models and return only their structured result,
never their full internal transcript.

For compatibility, restoration detects an old persisted request by the absence
of its serialized intent field and sets an in-memory legacy-control marker. Only
that marker enables the old profile completion/review/handoff gates. It is never
serialized as request authority. A new caller that omits intent is normalized
through the current project workflow policy, so field omission cannot select the
weaker path. Literal `REVIEW PASS` remains understood only inside that restored
compatibility path; current code-review credit is the structured artifact stored
on `WorkflowRun`.

## Model runner integration

Refactor the current monolithic primary/sub-agent flow into a reusable stage
runner:

```rust
struct StageContract {
    stage: WorkflowStage,
    profile: AgentProfile,
    capabilities: StageCapabilities,
    max_steps: usize,
    max_tokens_per_turn: i32,
    terminal_action: TerminalActionKind,
}

fn run_stage(
    engine: &mut dyn CompletionEngine,
    contract: &StageContract,
    context: StageContext,
    sink: &mut dyn EventSink,
) -> Result<StageOutcome>;
```

Important properties:

- every workflow stage starts with fresh messages assembled from structured
  artifacts;
- the loaded completion engine is reused across sequential same-model stages so
  context isolation does not reload model weights;
- stage metrics roll into a workflow-wide budget before another stage starts;
- direct final actions are accepted only in Discuss; delivery stages require
  their named terminal submission;
- parse/tool-loop correction remains bounded within the stage and globally;
- monitor advice cannot grant steps beyond the workflow's remaining budget; and
- future per-stage model selection is an additive model-pool feature, not a
  prerequisite for enforcing the state machine.

`AgentProfile` remains useful for prompt/persona selection and event
attribution. It no longer determines repository authority or whether handoff
occurs.

## Deterministic checks, review, and commit

Split the current combined deterministic handoff into explicit reusable phases:

1. `plan_handoff` calculates changed paths, affected components, and selected
   checks.
2. `run_handoff_checks` reuses or executes current check evidence and returns
   repair feedback.
3. The workflow runs isolated code review after checks pass.
4. `finalize_handoff_commit` verifies current evidence and calls the existing
   safe managed commit.

The commit predicate is exact:

```text
accepted code-review fingerprint == current workspace content fingerprint
and every selected required check has current scoped evidence
and the accepted task-owned delta has not changed since review
```

The managed commit's resulting task-owned paths must match the accepted reviewed
delta, and no task-owned uncommitted delta may remain afterward. Pre-existing
unrelated dirty paths remain outside the commit as today. HEAD/index changes
observed before the managed commit stage are treated as unexpected workflow
mutations and block commit rather than being silently accepted as model work.

The hard-coded fallback subject `feat: complete requested changes` should be
replaced with a semantic subject proposed in the implementation artifact and
validated by the harness, with a safe semantic fallback only when proposal
generation fails.

## Conversation handoff context

Do not paste an unbounded conversation into planning. Build a structured
`ConversationHandoff` from the current turn and explicitly recorded decisions:

```rust
struct ConversationHandoff {
    source_turn_ids: Vec<String>,
    task_summary: String,
    user_decisions: Vec<String>,
    constraints: Vec<String>,
    unresolved_questions: Vec<String>,
    proposal_id: Option<String>,
}
```

The discussion agent may propose this artifact, but planning independently maps
it into requirements. The source turn IDs remain visible so the user can audit
which conversation facts became delivery input. Memory remains data rather than
authority and cannot silently add a user decision.

## Persistence and recovery

Extend `PersistedSession` additively with:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
workflow: Option<WorkflowRun>,

#[serde(default)]
completed_workflows: Vec<WorkflowSummary>,
```

Add stable turn and workflow IDs. Persist a workflow checkpoint after every
accepted artifact, transition, budget change, check batch, review result, and
commit result. The checkpoint and latest transition event must be written before
the next model invocation.

Add events such as:

- `ConversationTurnStarted`;
- `DeliveryProposed`;
- `WorkflowStarted`;
- `WorkflowStageStarted`;
- `WorkflowArtifactAccepted`;
- `WorkflowChallengeRaised`;
- `WorkflowEvidenceInvalidated`;
- `WorkflowStageCompleted`;
- `WorkflowBlocked`; and
- `WorkflowCompleted`.

Events carry artifact/evidence IDs and hashes, not duplicated unbounded artifact
bodies. The checkpoint holds the current structured artifacts needed to resume.
Raw full evidence remains in the event/check streams already used by harness
journals.

On daemon restoration:

- a previously running discussion turn becomes paused as today;
- an active workflow restores its exact stage and remaining budgets;
- an in-flight model invocation is rerun only after an explicit resume;
- an in-flight deterministic check/commit stage reconciles current events, Git
  state, and evidence before retrying; and
- no transition is inferred from a model's last prose response.

## Web/API work

### API

Extend start and continue requests additively:

```rust
struct StartSessionRequest {
    // existing fields
    intent: Option<TurnIntent>,
}

struct ContinueSessionRequest {
    task: String,
    intent: Option<TurnIntent>,
}
```

Add `workflow_stage`, `workflow_id`, `workflow_outcome`, and `strict_workflow`
to session list/detail responses. Keep lifecycle `status` unchanged.

Provide explicit endpoints/actions for starting a displayed delivery proposal,
cancelling an active workflow, and resuming a blocked/paused workflow.
Cancellation must preserve work and evidence; it must not reset or clean the
repository.

### UI

Add a compact Discuss/Build intent control to new-session and follow-up
composers. Avoid a modal for ordinary use. An explicit build-oriented quick
action selects Build. Ambiguous free-form input defaults according to normalized
project policy.

During a workflow:

- show one current stage indicator and bounded cycle progress;
- render plan/review challenges as concise teammate messages;
- keep raw sub-stage model events and evidence behind details;
- disable ordinary follow-up submission while a stage is actively executing,
  while preserving `ask_user` answers and cancel/stop controls;
- show `Ready`, `No code changes`, `Needs another pass`, and `Needs help` from
  `WorkflowOutcome`; and
- after terminal outcome, restore the normal conversational composer with
  Discuss selected.

Add/update `webui/src/**/*.test.ts` tests for intent submission, stage
rendering, failure labels, restoration, proposal-to-build transitions, and
conversational continuation after workflow completion. Run `deno task test:web`
for every web behavior workstream.

## Harness and deterministic evaluation

`pb harness agent` remains a daemon-free exercise of the same core engine.
Preserve its current default as a delivery task for compatibility, with explicit
`--intent discuss|deliver` for targeted tests. Add trusted
`--workflow-config <PATH>` input analogous to `--workspace-config`; normalize it
before model loading and record source/hash in run metadata without copying it
into scratch.

Harness journals and run-index audit add:

- normalized workflow policy/hash;
- stage sequence and outcomes;
- plan/review artifact hashes;
- plan and repair cycles;
- global workflow budget use;
- rejected out-of-stage actions;
- evidence invalidations;
- final workflow outcome; and
- whether strict workflow requirements were enabled and satisfied.

Advance the deterministic harness-eval schema and corpus. Required fixtures
include:

1. discussion answers without creating/checking out a task branch;
2. discussion cannot use built-in mutations or a mutating teammate;
3. explicit delivery starts planning;
4. final/prose from planning cannot skip plan submission;
5. planning cannot mutate through direct tools, generic shell, or delegation;
6. malformed/incomplete plans are rejected structurally;
7. a plan review referencing the wrong plan hash is rejected;
8. a plan review missing an assessment or read evidence is rejected;
9. blocking plan challenges force a bounded revision cycle;
10. implementation cannot start before plan acceptance;
11. implementation can request replan without silently diverging;
12. implementation and repair can use `run_command`, its invocation is
    journaled, and its mutations cannot bypass later checks, review, or managed
    commit;
13. configured check failure forces repair and blocks review/commit;
14. mutation after checks makes affected evidence stale;
15. code review referencing the wrong content fingerprint is rejected;
16. code review missing changed-path evidence is rejected;
17. blocking code findings force repair, recheck, and rereview;
18. mutation after review blocks commit until checks/review recur;
19. a read-only parent cannot delegate build/scout authority;
20. multiple advisory calls cannot exceed global call/invocation/token limits;
21. managed commit contains only task-owned reviewed content;
22. no-change delivery performs no fictional edit or commit;
23. daemon/harness resume reconstructs the exact stage and remaining budget;
24. web and harness projections agree for the same scripted completions; and
25. legacy stored requests/events deserialize without acquiring
    strict-completion claims.

Real-model evaluation follows deterministic coverage. Run a bounded fixed-seed
matrix using the configured open-weight model to measure stage protocol
adherence, plan/review artifact validity, repair convergence, latency, tokens,
and energy. Real-model artifact quality is reported separately and does not
weaken deterministic workflow acceptance.

## External publication extension

Version one ends at a local `Ready` or `NoChange` outcome. Design the terminal
evidence bundle so a later provider-owned workflow can consume:

```rust
struct ReadyEvidenceBundle {
    workflow_id: String,
    commit_oid: String,
    plan_sha256: String,
    review_sha256: String,
    check_evidence_ids: Vec<String>,
    repository_remote: Option<String>,
}
```

Future stages are:

```text
Ready
  -> publication approval
  -> push
  -> create/update PR or MR
  -> wait for CI
  -> ingest unresolved review feedback
  -> fresh-context feedback triage
  -> implementation repair
  -> local checks and code review
  -> managed commit and push
  -> wait again
```

External operations require a provider abstraction, idempotency keys, approval
policy (`off`, `ask`, or `auto`), reconciliation after restart, and bounded
feedback rounds. They must never be implemented as unrestricted model shell
commands. This should be a separate follow-on goal once the local workflow is
complete and stable.

## Workstreams and commit boundaries

### W0 — Capture and close current authority gaps (P0)

Add deterministic tests demonstrating the current read-only-parent mutation
path, delegated-build dead end, root-profile handoff bypass, unbounded
delegation amplification, and unbounded child result. Then:

- enforce a target-profile/capability lattice at the `sub_agent` execution
  boundary;
- remove build/scout from advisory targets;
- bound and structurally truncate advisory results;
- introduce global advisory/invocation/token counters; and
- make any unexpected task-owned mutation visible to the terminal control path
  regardless of root profile.

Commit: `fix: enforce agent delegation authority`

### W1 — Add workflow configuration and durable types (P0)

- Add `.pb/workflow.toml` parsing, normalization, validation, and policy
  hashing.
- Extend `.pb/workspace.toml` with trusted bounded task declarations and update
  discovery/init.
- Add workflow stages, outcomes, counters, artifacts, checkpoint, and pure
  reducer types.
- Extend request/session/result/event schemas additively.
- Update `pb init` discovery/output and compatibility tests.
- Persist legacy old sessions without changing their claims.

Commit: `feat: add project delivery workflow model`

### W2 — Build the capability-scoped stage runner (P0)

- Extract a reusable fresh-context `run_stage` around the current completion
  engine.
- Separate profile/persona from executable capabilities.
- Add named terminal submission tools and execution-boundary checks.
- Preserve model-supplied `run_command` only for implementation and repair, with
  journaling, bounded output, state reconciliation, and no workflow credit.
- Reuse loaded same-model engines across sequential stages.
- Enforce cumulative workflow budgets and deterministic repeated-failure stops.

Commit: `feat: run bounded capability-scoped agent stages`

### W3 — Add conversational intent and delivery promotion (P1)

- Make discussion read-only and avoid task-branch creation for discussion-only
  sessions.
- Add turn IDs, intent fields, delivery proposals, and explicit start/cancel
  transitions.
- Assemble bounded auditable `ConversationHandoff` context.
- Preserve ordinary follow-up conversation after every terminal workflow
  outcome.

Commit: `feat: separate project discussion from delivery`

### W4 — Enforce plan and plan critique (P0)

- Implement plan and plan-review artifact schemas and validators.
- Run critique in a fresh read-only context without planner transcript.
- Track challenge IDs and bounded plan revision cycles.
- Pause unresolved human decisions through existing question handling.
- Prevent implementation capability until plan acceptance.

Commit: `feat: enforce challenged delivery plans`

### W5 — Enforce implementation, checks, code critique, and commit (P0)

- Implement implementation/replan submissions.
- Split deterministic handoff into check and commit phases.
- Run code critique from an isolated exact snapshot after checks.
- Track structured findings and bounded repair cycles.
- Invalidate evidence after every relevant mutation.
- Commit only when check/review/current fingerprints agree.

Commit: `feat: enforce reviewed delivery handoff`

### W6 — Persist, restore, and render workflow state (P1)

- Checkpoint before every next-stage model invocation.
- Reconcile in-flight deterministic stages on resume.
- Add session list/detail workflow projections.
- Add web intent controls, stage status, challenge messages, outcome labels, and
  details.
- Retain the conversational UI rather than exposing a compliance dashboard.

Commit: `feat: show durable delivery workflow progress`

### W7 — Expand harness contracts and evaluation (P0)

- Add harness workflow input/provenance and journal fields.
- Add the deterministic fixtures listed above and update the checked
  baseline/schema.
- Add web/harness parity tests using the same scripted stage completions.
- Run bounded real-model protocol experiments after deterministic success.

Commit: `test: cover enforced delivery workflows`

### W8 — Remove obsolete prompt-owned workflow paths (P1)

After parity is established:

- remove build-profile instructions that tell the model to decide whether to
  plan/review/commit;
- remove the generic model-requested build/scout delegation paths;
- replace ordinary `REVIEW PASS` gate state with structured workflow review
  evidence;
- retain compatibility behavior only for restored legacy requests; and
- update architecture, harness, and user-facing documentation to describe the
  final behavior.

Commit: `refactor: retire prompt-owned delivery control`

### W9 — Publication seam verification (P2)

- Produce the `ReadyEvidenceBundle` from successful local delivery.
- Add a no-op/mock publisher interface and idempotency tests without network
  mutation.
- Document the separate follow-on goal for push, CI, and PR/MR feedback
  automation.

Commit: `feat: expose reviewed delivery evidence`

## Rollout order

1. Land W0 before exposing any new workflow UI; it closes existing delegation
   holes.
2. Land W1 and W2 behind explicit request policy while deterministic stage tests
   stabilize.
3. Land W3 discussion/delivery intent without making strict delivery the default
   yet.
4. Land W4 and W5, then make explicit Build requests use the strict engine.
5. Land W6 and W7; prove web/harness parity and restore behavior.
6. Switch new project delivery requests to the normalized strict default.
7. Land W8 only after no supported new path depends on the old prompt-owned
   gates.
8. Land W9 and close this goal before starting external publication automation.

Do not use an undocumented environment toggle for rollout. During development,
use explicit serialized request/config policy so events and restored sessions
remain interpretable.

## Verification

Run focused tests after each workstream, then the repository-required final
suite:

1. `deno task build:web`
2. `cargo test --all-targets`
3. `deno task test:web`
4. `cargo build --release --target aarch64-apple-darwin`
5. `pb harness eval` with the updated checked schema/baseline
6. focused `pb harness agent` runs for discuss, successful delivery, plan
   rejection, check repair, review repair, no-change, budget exhaustion, commit
   block, and resume
7. bounded real-model runs that exercise at least discussion, successful strict
   delivery, plan revision, and code-repair paths

Ordinary workflow changes should not alter FlashMoe data flow. If engine reuse
or model lifecycle work changes the FlashMoe backend, first update
`docs/flashmoe-architecture-parity-plan.md`, then run the required release smoke
from `AGENTS.md` in addition to the suite above.

Every implementation commit uses a semantic subject. Preserve unrelated
working-tree changes and never add project-local `.pb/` state to implementation
commits.

## Completion contract for the delivery goal

Use the following objective when creating the implementation goal:

> Implement conversational project sessions with an enforced, durable delivery
> workflow shared by the web interface and `pb harness agent`: read-only
> discussion, explicit delivery promotion, challenged structured plans,
> capability-scoped implementation, deterministic affected checks, isolated
> structured code review, bounded repair, fingerprint-bound managed commit,
> truthful outcomes, persistence/recovery, UI visibility, deterministic
> evaluation, and a publication-ready evidence seam.

The goal is complete only when:

- W0-W9 are implemented with their semantic commits or consciously consolidated
  without losing independently reviewable boundaries;
- every invariant in this document has deterministic coverage;
- the 25 required harness fixtures pass and the checked baseline/schema are
  updated;
- web and harness scripted runs produce equivalent transition and terminal
  results;
- new strict Build requests with a task-owned delta cannot skip plan, plan
  critique, checks, code critique, or managed commit through final prose, shell,
  delegation, profile inference, or restart;
- strict Build requests with no task-owned delta still require a challenged plan
  and an explicit validated no-change result, and cannot invent edits, evidence,
  review, or a commit;
- Discuss sessions remain conversational and cannot mutate the project through
  available tools or teammates;
- repair, plan, invocation, advisory, token, and repeated-failure bounds stop
  deterministically;
- old stored sessions/events deserialize without false strict-workflow claims;
- the full verification suite passes;
- bounded real-model results and known model-quality limitations are recorded
  without weakening deterministic acceptance;
- architecture and user-facing documentation match the final implementation; and
- no unresolved P0/P1 workflow-control defect remains in the final review.

Do not mark the goal complete merely because the state machine compiles, the
happy path works, or a model produces an attractive artifact. Completion is the
enforced cross-entry-point behavior and evidence contract above.
