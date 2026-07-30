# Conversation and delivery workflows

pb treats conversation and delivery as different authority scopes. A conversation can build shared
understanding. Delivery can change a repository, but only through an enforced, reviewable sequence.

## Conversation

A discussion starts with read-only capabilities. The model can inspect repository evidence, use
public research tools, consult bounded read-only advisors, answer, or propose delivery. It cannot
edit, run project commands, or commit.

```text
user message → read-only investigation → answer
                                  └────→ delivery proposal → user chooses Build
```

The proposal is not self-authorization. The transition happens only after explicit user intent is
recorded by the calling surface.

### In-flight user messages

**Shipped.** A daemon-managed running session accepts bounded user messages independently of turn
intent. The web composer posts the message to the existing session; it does not create a new turn,
Task, delivery proposal, or Goal. The daemon appends a durable `user_message` event and keeps the
message pending until the primary agent harness takes it. At the next agent step boundary the
harness appends the whitespace-trimmed text as a normal `user` chat message and records
`user_message_applied`. Pending messages survive session persistence and daemon restart.

The boundary is intentionally between model/tool loops: pb does not interrupt a model inference or
partially executed tool batch. Message delivery is controller-owned and role-aware. Planning and
build authors may take pending messages; plan-review and code-review critics may not. Feedback
queued during plan review redirects the workflow to plan revision before a review verdict can be
accepted. Feedback queued after a build submission, during checks or code review, or before commit
invalidates the stale downstream artifacts and restarts planning from the current repository
snapshot. The next authoring stage then takes the message as ordinary user conversation input.

Every stage artifact and successful task-finalization boundary performs the same pending-message
check. This includes prose `final`, typed stage submissions, deterministic checks-to-review
transitions, review-to-commit, and the no-change or commit success path. Immediately before an
irreversible final response or managed commit, the daemon atomically closes that run's message
window. A message either wins that boundary and is routed, or its POST is rejected with conflict;
pb never acknowledges a message that the completed run cannot see. A containing Goal or multi-Task
controller reopens the window when it starts its next model stage.
Workflow stage capabilities, accepted-plan scope, checks, review, commit ownership, Goal controls,
budgets, and publication authority remain controller-owned; user conversation content alone cannot
mutate or widen those state machines.

## Delivery

Delivery is a harness-owned state machine:

```text
Planning ──→ Plan review ──→ Implementing ──→ Checking ──→ Code review ──→ Committing ──→ Ready
    ↑              │              ↑              │              │
    └─ revision ───┘              └──── repair ──┴──────────────┘
                                      bounded retries
```

Each model-driven stage has one permitted structured terminal action. Checking and committing are
deterministic harness stages and do not run a model.

**Shipped recovery contract.** A blocked checkpoint advertises its controller-selected recovery
instead of presenting every stop as resumable. `ExecutorUnavailable` retains the exact stage and can
resume after its prerequisite is repaired. `CommitBlocked` requires a fresh delivery from the
repository's current files: pb archives the blocked workflow summary, clears stage evidence and the
old repository baseline, captures a new baseline, and starts planning under a new turn id. The UI
labels that action **Restart with current files**. A generic resume request is rejected for these
content-sensitive blocks so it cannot queue an immediately stale checkpoint.

### High-level Tasks

**Shipped and on by default for new Builds.** Before an eligible explicit Build starts, pb decides
whether high-level partitioning can add useful coordination. A request containing at most three
behavior clauses stays one exact controller-owned Build when it does not explicitly request
decomposition or source ordering. pb records a `one_build_simple_request` decision with zero model
attempts. Larger requests, explicit separate-Task requests, and requests with `before`, `after`, or
`then` dependencies ask the selected local model for a compact high-level partition. This is
separate from the existing Build `PlanArtifact`. Its complete model-facing artifact is
`{"tasks":["request one","request two"]}`:
an ordered list of self-contained requests that will later enter normal Build planning one at a
time. The model preserves every controller source clause in exactly one request and may add only
boundary context. Sentence boundaries retain dotted paths and comma-delimited code or behavior
lists; pb ignores only whitespace adjacent to punctuation while recovering clause ownership. It
attaches request-wide constraints to every behavior-owning Task, derives the short UI title, creates
the sequential dependency/commit chain, and assigns the bounded `small` budget. The model cannot author repository
paths, IDs, dependencies, acceptance claims, effort, Goal authority, or numeric budgets.

The proposal is token-constrained to a controller-owned JSON schema in llama.cpp and FlashMoe.
Both engines stop when the first complete schema-valid JSON value is decoded instead of waiting for
the model to emit an end token. FlashMoe builds the same LLGuidance grammar against its active Hugging
Face or DeepSeek tokenizer. Unsupported tokenizer/schema pairs fail the optional partitioning
preflight; neither backend substitutes prompt-only JSON. The planner receives only the original
request, its exact source-clause strings, and a bounded component/dependency outline rather than the
full workspace graph. Rust owns source coverage, disjoint ownership, explicit
`before`/`after`/`then` order, IDs, dependency construction, budgets, and authority. There is no
model critic in default routing and at most one planner revision for a deterministic rejection.

If an attempted constrained proposal contains one Task, pb discards it immediately and runs the
exact original Build request. It does not summarize or project that generated request. Invocation,
schema, parse, source-coverage, ordering, and coordination-budget failures also fail soft to that
same original Build after the bounded attempts. Cancellation remains terminal. Every route has an
expandable Task-planning transcript. A bypass records its deterministic reason and no attempts; an
attempted partition also preserves the full local prompt, schema, raw constrained output, normalized
artifact, typed failure, token/runtime counters, and final controller decision. This evidence
explains routing without exposing or requesting hidden chain-of-thought.

Two or more accepted Tasks create a digest-bound `MultiTaskRun` and an ordered Tasks panel. The
controller activates only one dependency-ready Task. A Build Task is projected into the existing
strict workflow; a Goal Task is projected into the existing `GoalRun`, including its approval,
pause, amendment, evidence, and completion boundaries. The next Task receives a fresh request and
plans only after its predecessor reaches committed or verified-no-change delivery and pb reconciles
the exact repository state. A Goal Task preserves its child commit range rather than adding a
synthetic Task commit.

Every Task has controller-compiled ceilings for child workflows, stage actions, model invocations,
generated tokens, advisory calls, plan/repair cycles, and active runtime. Child checkpoints carry
monotonic usage watermarks, so retry, resume, Goal milestones, and restart cannot reset or
double-charge allowance. User-approved Goal amendments may add bounded criteria or change
continuation, but cannot remove an accepted Task criterion, change the Task objective or authority,
or exceed the parent Task budget.

The parent checkpoint persists the accepted plan and review, exact model/template/protocol
qualification, policy, budgets, queue revision, active request, native child checkpoint, repository
fingerprints, usage, results, and terminal reason in the existing repository-local session Git
note. It also persists the planning transcript and, before `Ready`, a whole-request completion audit
mapping every original requirement to successful Task results, acceptance IDs, evidence references,
commits, and the exact terminal repository. Missing ownership, result evidence, or repository
reconciliation prevents a `Ready` outcome. Restore pauses active work at a safe boundary.

The default partitioner can create Build Tasks only and cannot amplify authority. Explicit Goal
mode continues to enter the existing Goal approval and milestone controller without this Build
preflight. Automatic Goal-shaped Task selection remains a separate fail-closed promotion: it needs
an embedded qualification bound to exact model bytes, backend, template, protocol, and evidence.
The embedded promotion catalog is currently empty. `.pb/tasks.toml` supplies versioned budget
ceilings but cannot disable or enable default partitioning, grant Goal authority, or allow
publication.

### 1. Planning

The planning profile reads the repository and submits a structured plan with scope, implementation
steps, and checks. If a user-owned choice is missing, planning can ask a question. The stage cannot
edit files or run a shell.

The native `submit_plan` schema requires non-empty summary text plus at least one requirement,
implementation step, and acceptance fact; step and acceptance records must each name a requirement.
Constrained local decoding therefore cannot reproduce an empty plan skeleton. Schema constraints do
not attempt repository-dependent proofs: Rust still validates IDs, coverage, configured checks, and
ordered create/modify/delete path transitions before the workflow can advance. A model can still
fail to produce an acceptable plan, but an invalid artifact never becomes delivery authority.
Strict delivery does not expose `session_title`: the original task is already the stable title
fallback, so cosmetic rewriting cannot consume a planning inference. Planning guidance requires
local evidence before naming an existing path and reserves public web research for genuinely
external or current facts rather than repository-local file discovery. Removal plans are prompted
to account for state, imports, derived values, payload fields, and tests that can become obsolete
when a visible control disappears. The fresh plan critic receives the corresponding component-impact
check and is told to challenge markup-only removal plans that leave ambiguous behavior.

Accepted plan and plan-review artifacts remain in the workflow summary projection. The web
transcript renders the latest accepted plan at its acceptance event with requirements, steps, paths,
and acceptance facts. Reviewer prose remains a normal teammate message even when the next event is
the structured review submission; action grouping applies to instrumentation and tool calls, not
model-authored feedback.
The model-facing plan schema groups each step's paths by Create, Modify, and Delete. Create remains a
free repository-relative name. Modify and Delete are native-collar constrained to at most 32 exact
existing paths selected from task-relevant repository names, contract/current evidence, and exact
local glob/search/read results. With no candidate, those arrays are constrained empty until local
discovery establishes one. When the controller's baseline snapshot already has a task-relevant
existing candidate, a pathless plan is rejected as deferred discovery rather than accepted as an
implementation step. This keeps the schema bounded independently of repository size and
prevents a near-miss path spelling from consuming a rejected planning turn. The harness projects the
groups back to the durable ordered `PlanPath` artifact before validation. Preserved legacy array
submissions remain readable, but new native generation uses the grouped contract.
For controller-block runs, planning, plan revision, and plan review also receive one freshly
fingerprinted task-focused excerpt from the strongest matching existing path. A range observation
does not pretend to cover the whole file: the model can request a specific missing line range, but
an unbounded reread of the same file is rejected so generic continuation hints cannot consume the
stage budget by walking unrelated code.
Once plan revision has concrete repository work-unit paths, its tool surface drops public web
research: the revision must answer the critic from the accepted plan and current local evidence,
not begin an unrelated external evidence chain. Configured LSP tools are likewise exposed for a
file-scoped workflow only when their declared language IDs support at least one accepted task path.
Document-scoped LSP calls repeat that language check before lazy server startup, so a Rust analyzer
cannot consume time or service resources for a TypeScript path even if a caller bypasses model-facing
tool exposure.
If a non-native or preserved submission still names an absent Modify or Delete path, live validation
keeps the plan rejected and adds up to three extension-compatible, edit-distance-ranked existing
paths when a close name exists. Suggestions are fallback guidance only: pb never silently rewrites a
model-authored target.

For a trusted contract, pb may place the exact current contents of existing small UTF-8
`allowed_paths` directly into the planning context. Each automatic read is bound to the workspace,
path, and content hashes; it is admitted only when the complete set survives prompt preflight below
the conservative context target, and it is revalidated immediately before use. Missing, changed,
partial, binary, symlinked, oversized, or budget-displaced paths retain the ordinary model-driven
read path. Accepted observations enter the same read-before-write and complete-file evidence ledgers
as an equivalent successful built-in read, with explicit controller provenance.

When a trusted harness contract requires named checks, pb owns that immutable acceptance skeleton.
The planner may omit those check IDs; after submission, pb unions the missing required IDs into the
first acceptance fact and recomputes the plan digest before validation and fresh plan review. The
model still owns requirements, implementation steps, paths, descriptions, and any additional
configured checks it selects. Projection therefore removes transcription work without allowing the
model or controller to invent task scope.

### 2. Plan review

A review profile receives focused evidence in a fresh context and either accepts the plan or
returns actionable findings. Rejection returns the workflow to bounded plan revision. The reviewer
does not inherit mutation authority. Strict planning and review stages also exclude the legacy
`session_changes` summary tool: invocation-local model summaries are not repository evidence and
cannot help satisfy a typed plan or review gate.

pb projects the accepted plan ID and digest into the submitted review rather than asking the local
model to transcribe controller-owned identity. Every required assessment kind remains mandatory.
Assessment entries contain only their dimension and status. A concern or failure records its
specific explanation and evidence once in the corresponding typed challenge, and cited repository
evidence remains freshness-validated. Older checkpoints with assessment-local detail remain
readable, but new native submissions cannot generate that duplicate prose.
The deterministic validator rejects internally inconsistent reviews: a passing verdict cannot
contain a concern or failed assessment, and a revision verdict needs both a blocking challenge and
a corresponding concern or failed assessment. A blocker must describe a defect in the proposed
plan; the critic is explicitly forbidden from treating an unchanged implementation defect as a
plan defect when the exact plan already addresses it.
The terminal review tool prevalidates that self-contained verdict/challenge/assessment consistency
inside the live review turn. A malformed disposition therefore returns tool feedback without
discarding the reviewer's conversation or repository reads, and the next turn exposes only the
corrected terminal action. The outer workflow still validates accepted-plan identity, requirement
references, and observed-path evidence before accepting the artifact.
Observed-evidence validation also runs at that live tool boundary. Plan-review check references are
native-collar constrained to exact configured workspace check IDs, or the field is removed when no
checks exist; source path and line evidence must omit `check_id`. An invalid evidence label therefore
receives an in-context terminal retry while preserving the critic's verdict and findings instead of
restarting a fresh critic that could silently reverse them.
The critic need not re-read a task path merely to confirm code that the plan proposes to change when
the task, exact plan, bounded repository brief, and carried evidence already support every structural
assessment. Any repository fact it does cite must still come from current carried or model-read
evidence.
When fresh controller observations cover every existing path in the exact proposed plan and survive
the same prompt-capacity and fingerprint checks used by planning, plan review exposes only
`submit_plan_review` for that turn. Trusted contract-path coverage provides the same closure. The
critic can still reject the plan; a resulting plan revision gets ordinary read tools. If any proposed
existing path is unobserved, stale, unsafe, or displaced by prompt preflight, plan review keeps its
ordinary evidence tools. Bounded range observations count only for their exact bytes and remain
valid review evidence; they do not become complete-file mutation evidence.
Once the deterministic submission precondition is current, the turn exposes only the plan-review
terminal and focused repository evidence tools.

The workflow checkpoint carries a byte-bounded bundle of complete small-file reads between stages.
pb records the exact bytes, path/content hashes, source stage and normalized read arguments; it
revalidates every path before injecting the bundle or seeding the new stage's observed-read ledger.
A changed or oversized/partial file therefore still requires a fresh read. Model conclusions and
TODO prose are never promoted as evidence.

### 3. Implementation

The build profile receives the accepted plan and mutation capabilities. It can use built-in edits,
configured tasks and checks, or a journaled command escape hatch. It submits an implementation
artifact that identifies what changed and the evidence it produced. It never receives `git_commit`.

On the final unfinished controller-owned work unit, every mutation schema requires an inline
implementation-completion object. pb executes and records the mutation first, then independently
validates the completion against the resulting repository fingerprint and ordinary implementation
gate. Diagnostic-eligible checks run before the inline completion can close the work unit. A failed
preview reopens only the exact diagnosed task path and keeps the edit; a passing preview permits the
completion without a bookkeeping-only model turn. An invalid object does not roll back or misreport
the successful mutation; the ordinary bounded submission path stays available on the next turn.
If an `apply_patch` generation reaches a native constraint dead end before forming an executable
call, its one bounded recovery switches to the smaller `edit_file` operation when available. The
recovery cannot repeat the same irreparable patch branch and cannot claim inline completion. pb
re-observes the changed path, then exposes another bounded mutation turn for the remaining separated
edits under the same work-unit and read-before-write authority.

When a configured language server supports a changed accepted task path, pb proactively requests
diagnostics without waiting for the model to choose an LSP tool. While implementation is still in
progress, the syntax pass admits only error-severity diagnostics identified as parser or syntax
failures, avoiding expected semantic noise from half-finished code. Once the work-unit ledger is
structurally complete, pb ensures the current content version has received a settled pass before a
direct handoff or final-grace path; unchanged versions are deduplicated. That pass admits current
error diagnostics. A blocking result reopens only its exact current task paths for repair. Failed
check output is carried into a fresh repair stage so controller observations select the cited line
windows rather than forcing whole-file pagination. Once those exact diagnostic bytes are available,
the repair turn stays mutation-bound and cannot escape into a redundant replan. It does
not replace the checking stage, including for supported paths beyond the bounded pass selection.

Settled evidence is workspace-epoch scoped, not merely file-version scoped. Any content mutation
invalidates settled evidence for every supported task path. pb opens the whole bounded server/path
set before requesting each target. A server that implements LSP pull diagnostics completes a target
only with an explicit full diagnostic report. A push-only server can still provide a fresh
post-open publication after the document-version barrier, but that bounded snapshot is advisory:
even an empty snapshot cannot complete coverage or be presented as clean. Coverage is tracked per
server/path target. A path is settled only when every matching target completed for the same
workspace epoch; path, call, end-to-end pass time, startup, transport, push-only, and
stale-workspace limits are explicit incomplete outcomes.

Configured LSP servers can also participate in the mutation control collar through the typed
`semantic_enforcement` mode. This path is distinct from proactive repair hints. The controller
captures an immutable workspace identity and exact mutation snapshot, sends full-content base and
candidate overlays with monotonic LSP document versions to a fresh provider session rooted at an
exact bounded shadow workspace, requires explicit full pull diagnostics, and binds each diagnostic
snapshot to provider/configuration/dependency/workspace and exact overlay content digests. The
diagnostic-debt comparison rejects newly introduced classified errors while allowing an existing
error to be repaired. Incomplete, stale, push-only, unpinned, timed-out, unclassified, or
project-detached results are `Unknown` and cannot authorize required mode. Rust additionally needs
a loaded non-empty crate graph and analyzer-confirmed membership for every overlay document.

At a completed Qwen JSON or DeepSeek DSML payload-string close, FlashMoe and llama.cpp apply the same
portable mutation gate and may run the separately configured settled-transaction semantic provider.
That LSP diagnostic path does not supply token-by-token facts for incomplete code. `Reject` masks
only the completed payload boundary; `Defer` keeps it reachable. Native streaming language layers
instead query immutable request-local project state directly. The final executor independently
reconstructs the prepared write/edit/patch against the authorized base and revalidates before the
existing publication step. Generation and final-executor decisions have separate content-free
evidence; generation state is never executor authority. Current backends report
`candidate_probe_only`; model-state replay and complete-state restore remain explicitly
unimplemented.

### 4. Checks

The harness selects affected configured checks and records their current results. Named acceptance
checks cannot be replaced by an arbitrary command that happens to exit successfully. A mutation
after a check makes older evidence stale.

For a failed check, pb compares its bounded diagnostic text only with the exact current task paths.
Paths named as complete diagnostic tokens are listed as repair focus, so a local model can inspect
the implicated files first. The hint is not evidence, does not narrow or expand mutation authority,
and does not claim that unnamed paths are irrelevant. If a trusted contract authorizes exactly one
changed path, pb deterministically reopens that path even when an assertion reports only rendered
output or another symptom rather than repeating the filename. Multi-path failures still require an
explicit path match before the controller chooses a repair target.

When the bounded check output explicitly cites another regular UTF-8 file inside the workspace,
pb may carry at most two small excerpts into the focused repair as controller-read diagnostic
support. This commonly exposes the exact local test assertion or compiler source location that
explains the symptom. Support files remain read-only context: their paths do not become repair
targets, their bytes grant no read/check/review evidence, and paths outside the workspace, changed
task paths, symlinks, binary files, and oversized files are omitted.

When the trusted contract authorizes exactly one path and that same changed work unit has
failed-check evidence, the repair stage no longer exposes `request_replan`: another plan cannot grant
a different path, and the accepted work unit already binds the required repository transition, so
replanning would only repeat planning and review without changing authority. The model receives the
current read or repair action instead. Replan remains available when the contract allows multiple
paths, when no contract fixes the path set, when an uncovered failure has no unique contract target,
or when repository state has put the active unit in `blocked_for_replan`.

### 5. Code review

A fresh read-only reviewer inspects the delivered change and current focused evidence. An accepted
review is bound to the content fingerprint it saw. Findings enter a bounded repair cycle; repair
then refreshes checks and review.

Every required code-assessment kind remains mandatory and contains only its dimension and status.
Concerns and failures record their explanation and typed evidence once in findings. A passing
verdict cannot contain a concerning assessment; a revision verdict needs both a blocking finding
and a corresponding concern or failure. Review acceptance remains bound to the checked content
fingerprint. Once all fresh-review evidence is current, only focused inspection/read/search tools
and the code-review terminal remain exposed.

### 6. Managed commit

When the accepted plan, checks, and code review all describe the current content, the harness owns
the commit operation. It verifies the repository boundary and builds a durable evidence bundle that
binds the commit OID to the workflow artifacts and check receipts.

An optional trusted harness contract is an additional authority boundary, not a label applied after
the workflow. Required or forbidden mutation and allowed paths constrain the plan and final delta;
required checks must be selected by plan acceptance facts, and required checks and fresh-review
reads must have current evidence; commit semantics and workspace
cleanliness are checked before `Ready`. A strict workflow without such a contract can still be
locally ready, but its acceptance status remains unspecified and it is never reported as externally
verified. An unmet explicit contract has the distinct `contract_unsatisfied` outcome.

## Durable Goal controller

**Shipped.** Goal mode composes several strict delivery workflows without moving orchestration into
model prose:

```text
Goal draft → exact plan approval → milestone 1 strict workflow
                                      ↓ Ready
                              criterion evidence
                                      ↓
                              milestone 2 … → evaluate
                                                  ├─ machine criteria → Complete
                                                  └─ prose criteria → Ready for review → Accept
```

The version-one initial plan is deterministically derived from the user's ordered completion
criteria: one bounded sequential milestone per criterion. The user can edit that draft and approves
its exact content digest before mutation. Each milestone then receives the full planning and plan-
review stages of the existing strict workflow, so the simple Goal decomposition does not bypass
repository-aware implementation planning or critique.

`GoalRun` is the canonical controller state. It owns the objective, versioned plan, criteria,
milestones, continuation choice, project policy hash, authority envelope, total budget and counters,
pause/amendment state, child workflow checkpoint, outcome, and completion basis. A session has at
most one active Goal and retains completed Goal checkpoints. Exactly one milestone can be active;
parallel workflows are not a version-one capability.

The controller clamps each new child workflow's invocation and generated-token policy to the Goal's
remaining totals. Workflow completion rolls durable usage and evidence into the Goal before the
next milestone can be selected. Workflow, token, invocation, and wall-time exhaustion stop at a
typed budget outcome. Automatic continuation is a user-selected per-Goal choice and never comes
from `.pb/goal.toml`.

Pause is cooperative. The web/API mutation records a pause request immediately, but the agent loop
checks it before another model or tool action and the delivery loop returns its unchanged,
non-terminal workflow checkpoint. Only that durable safe-boundary transition becomes `GoalPaused`.
Restart recovery applies the same rule: active work restores paused, while an initial plan awaiting
approval or final evidence awaiting acceptance remains in that truthful review state.

Before initial approval, editing replaces the draft and plan digest. After work starts, editing is a
checkpointed amendment: current work pauses, a replacement plan receives a new version and digest,
unfinished milestones become superseded only after approval, and completed milestone/workflow and
criterion evidence moves to immutable history. Discarding records the prior stage so an explicit
resume can return there.

The model-facing Goal tools are deliberately asymmetric:

- `propose_goal` records a read-only discussion artifact;
- `start_goal` exists only in an explicit Auto turn, must cite that exact turn, and creates an
  approval-gated Goal rather than mutation authority;
- `goal_status` returns a bounded controller-owned brief;
- `goal_pause`, `goal_request_amendment`, and `goal_request_budget` can only stop and request human
  review; and
- there is no model `goal_resume`, `goal_cancel`, `goal_accept`, or direct Goal rewrite tool.

The daemon exposes digest-checked HTTP and Unix-RPC operations; `pb goal` uses the same controller.
Session list/detail projections embed Goal summaries while retaining the child workflow only inside
its milestone, avoiding two canonical copies of workflow state.

## Stage capability matrix

| Capability | Discuss | Plan/review | Implement/repair | Check/commit |
| --- | ---: | ---: | ---: | ---: |
| Read repository | yes | yes | yes | harness only |
| Public research | yes | yes | yes | no |
| Mutate repository | no | no | yes | harness only |
| Run commands/tasks/checks | no | no | yes | harness only |
| Advisory input | yes | yes | yes | no |
| Commit | no | no | no | harness only |

Tool schemas are derived from this matrix for every stage. A request-level allowlist and project
policy can narrow the set further. Neither can broaden it.
The legacy model-owned `todo`, `git_commit`, and `git_revert` schemas are retired from every current
surface. The accepted plan, typed stage artifacts, checkpoint, deterministic checks, and managed
commit already own that state and authority. Persisted legacy sessions remain readable, but a new
allowlist or policy cannot revive the retired tools.

## Agent-tool runtime contract

**Shipped.** Built-in tools have one central effect record for mutation, parallel safety,
deterministic caching, useful progress, and retirement. Exposure, batching, caching, progress
accounting, and execution all derive from that record instead of maintaining independent lists.
Dynamic MCP tools carry an equivalent discovered effect record; LSP tools are read-only.

The runtime applies these boundaries before a result can become evidence:

| Surface | Enforced behavior |
| --- | --- |
| File reads and discovery | `read_file` accepts bounded UTF-8 files; glob, regex, and skill discovery have time/input ceilings; results are prompt-bounded with explicit continuation or failure rather than silent partial authority. |
| File mutation | Existing files require an exact content fingerprint from the bytes actually read. Create and replace use synced temporary files and atomic no-clobber/replace operations; stale concurrent edits fail. `edit_file` requires one unique match. Real local backends prepare the exact virtual result through the control collar and refuse to close or execute a supported Rust, Python, TypeScript/TSX, JavaScript/JSX, HTML, or CSS mutation unless the pinned complete-file parser accepts it. Rust- and Python-edit-capable requests additionally require their ready exact native worlds before inference and independent final replay; only promoted exact contradictions can veto, while partial or dynamic facts remain unknown. Unsupported extensions retain the ordinary executor checks and are not reported as syntax constrained. For an initial Python or Python-stub modify whose complete controller-observed file consumes at most half the bounded replacement allowance, pb exposes only atomic `replace_file`; this avoids indentation-fragment errors while requiring the model to preserve unrelated bytes. Other languages, larger or range-only observations, and diagnostic repair retain exact-edit behavior. Diff events are bounded and identify truncation. |
| Patch, move, and remove | FlashMoe `apply_patch` accepts a bounded canonical text-only unified-diff subset: exact offsets, counts, context and deletion bytes; LF; unquoted `a/` and `b/` paths; optional matching `diff --git`; and exact `100644` create/delete metadata. It rejects recount-dependent hunks, mode changes, renames, copies, timestamps, index metadata, and binary patches, applies the patch to an immutable controller snapshot in memory, validates every supported resulting file, rechecks live base hashes, and then requires exact `git apply --check` parity before publication. When an accepted work unit binds one target, the collar rejects any patch file outside that target; a range-only observation further confines every old-side hunk to observed bytes. The llama.cpp compatibility path retains its broader Git/recount behavior and is never a fallback after collar rejection. `mv` moves only a file or symlink and cannot overwrite. `rm` operates on the final filesystem entry and removes only a file, symlink, or empty directory; recursive directory mutation is not an agent capability. |
| Configured tasks | `run_task` executes against an isolated snapshot, rejects Git-control or undeclared-path changes, bounds promoted path/file totals, validates symlinks, stages every output, and rolls back the destination set if promotion fails. It never earns named-check credit. |
| Commands and checks | Host and managed commands drain stdout/stderr concurrently, cap combined output, have explicit timeouts, observe user cancellation, and stop the owned process group or managed exec. `run_command` defaults to 120 seconds and is capped at 600; configured checks/tasks use their validated timeout. |
| Public research | Every URL and redirect is restricted to HTTP(S), resolved before connection, rejected if any answer is private/special-use, and pinned to the validated public address. Proxies are bypassed. Bodies are capped and oversized chunked responses fail instead of being returned as complete. |
| Vision | A vision path must be inside the workspace or exactly match a session attachment. Inputs are regular files with byte and pixel ceilings. |
| MCP | Only operator-declared `capabilities.read_only_tools` are exposed. Server annotations are untrusted hints. Unsupported schemas and normalized-name collisions become explicit status failures; `isError` is a failed call; only an operator-read-only, server-idempotent call is retried, and only after a typed transport failure. A configured service runtime must match the owning session runtime. |
| LSP | Discovery is schema-only and server startup is lazy behind one per-session/per-server initialization lock; no registry lock is held while an image or server starts. Names cannot collide silently; documents/config/responses and the open-document set are bounded; language IDs follow file extensions; pull diagnostics require an explicit full report, while fresh push publications remain advisory and cannot establish clean coverage; only typed transport failures restart a server. Proactive passes are deterministic, read-only, content-fingerprinted, limited to 8 supported task paths and 8 server/path calls, retain every unattempted target as deferred, and use one 12-second deadline for bounded workspace observation and revalidation, cancellable server launch, acknowledged stdin writes, initialization, diagnostics, shutdown, and a bounded restart. They wait at most 2 seconds for a push publication, retain at most 64 diagnostics, and collapse each message to 500 characters. Any workspace mutation during collection discards the entire result. Marketplace packages require a bounded typed manifest bound to an immutable OCI digest and cannot request write or network authority. Sidecars use the owning task runtime or a session-owned service-only lease for local/no-environment projects. |
| Durable memory | Agent writes are byte/count bounded and evidence-backed. The agent can record only facts, gotchas, procedures, and debt. Decisions and preferences require a future controller-owned approval record and cannot be self-approved in tool arguments. Supersession requires active source and replacement entries. |

Dynamic JSON schemas are admitted only when pb's recursive validator can enforce every supported
keyword. Project policy selectors are validated against the resulting exposed schema set at session
start, including unmatched wildcards, so misspellings fail closed.

## Freshness is part of correctness

Plans, checks, and reviews carry content or evidence fingerprints. A later mutation invalidates
evidence tied to the earlier state. This makes “reviewed” mean reviewed at the delivered content,
not merely that a review-shaped conversation occurred somewhere in the session.

The same rule applies during recovery. Persisted structured artifacts and fingerprints can restore
a stage; discarded or truncated prose cannot become a substitute source of authority. Resuming an
active durable workflow also reuses its current task branch before validating Git control state, so
a new invocation identifier cannot itself alter the checkpoint's refs fingerprint.

## Bounded failure and recovery

The workflow budgets stage steps, model invocations, generated tokens, advisory calls, plan cycles,
and repair cycles. It also detects repeated no-progress operations against unchanged state and
stops loops deterministically. A request-level step ceiling narrows every model-driven stage; the
lower of that ceiling and the compiled workflow policy is cumulative within a plan or repair
cycle. Each accepted cycle grants one additional stage-sized cumulative tranche while each model
invocation remains capped to one tranche. An implementation-requested replan consumes the same
bounded plan-cycle allowance as a critic-requested revision, so replanning cannot reset or bypass
the workflow-wide invocation and token budgets.
Work-unit target paths are bound before loop signatures are calculated, so omitting or redundantly
supplying the hidden fixed path cannot disguise an identical action. An exact deterministic read
cache hit remains fully preserved in durable evidence, but the next prompt receives only a compact
replay receipt, the required continuation when one exists, and the current fingerprint. Repeating
that exact read after a replay is blocked before another tool execution.

When a model response is truncated before producing a valid action, pb can retry with thinking
disabled and, within the same global budget, use a larger output cap. A capped native `write_file`
or `replace_file` also receives at most one target-bound atomic retry inside that same stage step.
pb distinguishes a mutation-schema boundary from real output-token exhaustion: FlashMoe reports the
exact constrained-decoding terminal state, while other backends may use the published schema size,
serialized output, and unused token allowance as a conservative inference. A payload-bound retry
doubles the string allowance while retaining the same token ceiling; a genuinely token-bound retry
requests a complete payload below half of the ordinary allowance and reduces its token ceiling to
match. Both expose only the attempted tool, start from the original authoritative messages rather
than carrying the rejected payload, and bind any recovered target path. The retry schema enforces
those constraints rather than relying on correction text, and pb rejects a parsed retry that changes
the bound tool or path. These mechanics help the model express an allowed action; they never expose
a new capability or waive a transition gate.
FlashMoe also bounds structural whitespace after entering a native tool envelope. Its sampler widens
past rejected whitespace candidates until it finds a schema-valid structural token, so a resident or
streamed model cannot spend the whole output allowance without advancing the call.
Retries consume model-invocation and generated-token budgets, but remain part of the same visible
stage step. Stage-step accounting is checkpointed at the `StepStarted` boundary, so an action-only
retry cannot prematurely terminalize a workflow while its result is still being recorded.
Equivalent max-token native actions are correlated by attempted tool name and the current workspace
and evidence fingerprints. A merely parsed action does not erase that history before its outcome is
known; a successful executed action clears it, while a real workspace or evidence transition also
changes the signature. For a capped file write, the correction explicitly says that no partial file
exists and requires complete content. Two repeated payload-bound recovery failures at unchanged
state terminate; other equivalent parse failures retain the general three-failure bound.

Compatibility edit actions are model-turn boundaries. Implementation prompts tell the model to
stop after the action and never imitate pb's transcript or invent later results. If a local model
still emits a fenced action followed by fabricated `Tool calls:` or role entries, pb executes only
the validated action and omits that untrusted pseudo-transcript from subsequent model context. The
real tool result and content fingerprint remain the only mutation evidence.

That boundary is per model completion, not one tool invocation per prompt. Native function-call
output may contain multiple calls, and the JSON compatibility protocol has an equivalent
`tool_calls` batch. pb validates every call against the same stage, allowlist, schema, policy, and
progress gates. Independent parallel-safe calls run concurrently and all authoritative results are
returned before the next model pass; a batch containing same-path read/write dependencies, a check
mixed with a mutation, or opaque `run_command` dependencies is rejected atomically, and a workflow
or delivery transition must be the only call in its batch. The prompt explicitly encourages
batching independent discovery reads and lookups so local inference is not spent on unnecessary
round trips. The FlashMoe control collar additionally permits at most one generated mutation call in
a batch; non-mutation batching is unchanged. Batch events record call, parallel-safe, useful,
bookkeeping-only, and dependency-rejection counts.
If a max-token native completion contains complete early calls followed by an incomplete call, pb
rejects the entire batch before execution and invokes the bounded truncation recovery. A complete
administrative call therefore cannot mask or partially commit an oversized file mutation.

The workflow capability set is intersected with the active event sink. Non-interactive harness
runs omit `ask_user`, while the web event sink exposes it because it can collect and return a real
answer. This prevents a local model from entering an unanswerable question loop without weakening
interactive workflows.

Implementation guidance gives symmetric concrete actions for missing and existing paths. Missing
paths use `write_file`; existing paths use a separate `read_file` turn followed by
`replace_file`, `edit_file`, or `apply_patch`. An attempted overwrite keeps failing closed but now
returns that exact recovery sequence instead of a generic suggestion.
Strict implementation schemas derive mutation-string `maxLength` from the turn's generated-token
cap after reserving native-envelope and closing overhead. The portable estimate budgets four payload
characters per remaining token, matching the observed constrained Qwen action ratio while the hard
token ceiling remains the final bound for other tokenizers. The same string bound is enforced again
by recursive executor-side schema validation and appears in the stage anchor and invocation
telemetry. Larger files are built from complete, loadable, atomic work units rather than partial JSON
or partial filesystem writes.
After plan review, pb persists an ordered, typed work-unit ledger in the workflow checkpoint. Every
planned create, modify, or delete records its plan step, operation, path, task/invocation/current
fingerprints, adopted-work provenance, and structural state. Modify and delete units require a
complete fingerprint-current read before mutation. Revalidated complete-file evidence carried from
planning seeds the same path and byte fingerprint in the implementation gate, so it does not force
a duplicate model read. A task-owned delta present at invocation can satisfy structural progress as
adopted work without requiring the current model to claim that it authored those bytes. A
forbidden-mutation contract has an initialized empty ledger.

Implementation and repair expose only the active unit's operation. The target path is omitted from
the model-required arguments and inserted by pb into the durable call immediately before schema,
policy, scope, fingerprint, and executor validation. Tool schemas are therefore byte-stable across
target paths. Creation also advances one controller-bound path at a time; it does not expose a
multi-file mutation that could make unknown file sizes compete for one model-turn budget. A complete
transition advances the ledger after a normal turn or checkpoint resume, while an incomplete, stale,
or out-of-order transition keeps the typed implementation terminal hidden.
An acceptance contract may attach a bounded advisory hint to an exact allowed path. pb repeats only
the active mutation-ready unit's hint in the dynamic prompt suffix; the canonical tool schema stays
unchanged. The hint cannot select or advance a unit, grant authority or evidence, or satisfy any
workflow gate. This is the supported way to give a weaker local model concise path-specific
constraints without projecting raw verifier commands into its action context.
On the final ordinary implementation or repair turn, pb exposes only the typed implementation
submission when its deterministic terminal preconditions are ready. This reserves closure capacity
for small local models instead of letting a redundant diagnostic consume the stage budget. If a
required creation path is still missing or another deterministic precondition is not ready, the
ordinary authorized surface remains available; pb never synthesizes implementation accounting or
waives its validation. Harness-owned checking still runs after the accepted submission.
An active unit may earn one additional stage turn after its first distinct content or exact-evidence
transition, with four earned turns maximum. Failed, rejected, cached, repeated, no-op, and
bookkeeping-only calls earn none; workflow-wide invocation and generated-token limits remain
authoritative. Identical replacements and edits fail without emitting a diff or invalidating
existing evidence.

Contract checks marked `diagnostic_eligible` run automatically after the queue is structurally
complete. When workspace discovery selected an affected project type-check or web check, pb may
also run at most one of those high-signal checks as an early preview even without contract opt-in;
it is capped at 60 seconds. Preview results are fingerprint-bound feedback only: they never enter
selected-check evidence and all required checks run again in the authoritative checking stage. A
failed preview can reopen only an exact current task path named as a complete diagnostic token.
Earlier read evidence for that path is invalidated; the repair controller supplies current focused
diagnostic ranges and exposes only path-bound mutation tools for that observed repair. A missing
diagnostic target requires a replan. The preview command must preserve repository content and Git
control state.

Proactive LSP diagnostics run immediately before these configured diagnostic previews. A syntax
pass is intentionally narrow during partial implementation; a settled pass includes every
error-severity diagnostic returned for the bounded path set. Results bind both the workspace and
each reported path's content fingerprint. A mutation during collection makes the pass stale and
discards its diagnostics. Transport, startup, timeout, and malformed-response failures are visible
Trinity warnings and cannot satisfy or bypass an ordinary check. A successful pass likewise grants
no selected-check, review, commit, progress, or completion evidence. pb never requests or applies
LSP edits, formatting, commands, or code actions automatically. Empty push-only publications are
shown as incomplete advisory evidence; only an explicit full pull-diagnostic response completes a
server/path target.

At implementation submission, pb projects the accepted plan identity/digest, current content
fingerprint, actual task-delta paths per named plan step, and no-change fact. The model still accounts
for every step and owns status, summaries, and the proposed semantic commit subject. Implementation
and per-step summaries are limited to 1,024 characters, and the semantic subject to 200 characters,
in both the exposed schema and artifact validator. A final mutation's inline completion schema omits
the controller-owned identity, fingerprint, touched-path, and no-change fields entirely; pb projects
them only after the mutation succeeds. At code review, pb similarly projects the checked fingerprint
while the fresh reviewer owns assessments, findings, and verdict. Existing artifact validators still
reject missing steps, paths outside the plan, stale state, incomplete work, or unsupported review
conclusions.

When a constrained mutation branch becomes irreparable, an `edit_file` recovery is treated as one
bounded replacement rather than proof that the path-level work unit is complete. Its retry schema
omits inline completion, and pb reopens and re-observes the unit before allowing closure. This is
true both for an `apply_patch`-to-`edit_file` fallback and for an `edit_file` retry after the same
constraint dead end.

### Deterministic controller actions

**Shipped, intrinsic.** pb executes a uniquely determined local observation before a model turn
when every structural, freshness, boundedness, and prompt-admission gate passes. Daemon, desktop,
web, queue, and direct harness workflows use the same behavior. It is not user configuration and
cannot be disabled or broadened by persisted/API request fields.

Production uses one explicit controller-labelled user/context block. pb never fabricates an
assistant tool call or tool result. The hidden evaluator retains only a native-read control arm and
the production controller-block arm. The controller records actual origin, exact full/range
coverage, fingerprints, and content-derived action identity. Model tools and controller actions
remain distinct durable events. Each new model tool event also records the active profile
character. Controller actions and deterministic corrections record Trinity Walker as the workflow
steward plus the profile she is assisting when that context is available. Older tool events without
an actor remain explicitly unattributed rather than inheriting a nearby profile.

New tool calls and results also carry a stable call id, optional batch id, and a typed result outcome
(`succeeded`, `failed`, `rejected`, `timed_out`, `cancelled`, or `cache_replay`). Consumers correlate
by id even when parallel results arrive out of order or a correction occurs between call and result.
Legacy events use a bounded tool/actor fallback only when both sides lack an id, and an absent legacy
outcome remains unknown rather than being displayed as success.

An exact active small-file observation may satisfy read-before-write only while its current
fingerprint and complete prompt bytes remain valid. For an oversized UTF-8 Modify work unit, pb may
instead select a small set of task-relevant line windows using exact, identifier-aware, one-edit,
or narrow UI-control synonym matches. It admits those windows only when a rare or multi-term match
provides a useful anchor, records their exact byte hashes, and exposes target-bound `edit_file` and
canonical `apply_patch`. The first ranged implementation turn is mutation-only: `read_file` remains
hidden while the controller-provided bytes are already actionable. If constrained mutation stops
after the teammate explicitly names a missing fact, Trinity gives one recovery invocation only a
target-bound `read_file` shape. Its native schema requires one center line and caps the optional
surrounding context at forty lines on each side. This keeps the schema compact while making a
whole-file call structurally impossible instead of spending an inference only to reject an
overbroad range; generic continuation reads are also rejected. A completed explicit excerpt does
not advertise a continuation into the rest of the file; continuation metadata appears only when
the excerpt itself exceeded the result budget. This recovery happens
instead of forcing another attempt at the same incomplete mutation. Planning and plan-review reads
never arm an implementation mutation merely because their hashes are still current: read authority
is stage-bound, so implementation receives a fresh controller observation or performs its own read
before the work unit becomes mutation-ready. This keeps the bytes that authorize a write present in
the same teammate context that performs it. A partial constraint-recovery edit also invalidates its
old controller ranges, forcing fresh current bytes before the next inference instead of letting a
repair repeat an already-applied hunk. That bounded continuation exposes only an exact edit, so it
cannot restart the multi-hunk patch branch that already proved irreparable. An edit's old text and every
patch hunk's complete old-side range must lie
wholly inside an included byte window. A range-bound patch may carry separated hunks, but the
generation collar and executor both bind every hunk to the active accepted-plan path; it cannot use
one work unit to touch another file. A work-unit observation is retained whenever it survives the real
context preflight; the lower prompt-share heuristic is reserved for optional multi-file planning
and review batches. This avoids falling back from a viable range to an impossible complete-file
read on a large accepted target. A failed-diagnostic range has the same edit boundary and is
selected from exact diagnostic locations. When no confident task window exists, the ordinary
model-driven read path remains available. Fresh review inspection is injected only
when every required changed path fits, and the reviewer still authors assessments, findings, and
verdict. Optional completion fields on a successful final mutation remain model-authored. Automatic
deletion is intrinsic but limited to a unique tracked, clean, unchanged, bounded plan deletion with
Git recovery. The accepted plan, active work unit, allowed path, current identity, and Git state are
all revalidated first; an attached contract must explicitly require mutation.

A successful planned create that later fails an exact-path diagnostic is now an existing repair
target. For a bounded text file, the controller injects its current fingerprinted bytes and advances
directly to target-bound repair instead of asking the model to repeat a deterministic `read_file`
turn. The observation carries the same read-before-write evidence as a planned modification; it
cannot reopen another path or observe a create before the file exists.

Terminal output and the web transcript present actions as work by teammates: the active profile
character owns a model-requested tool action, while Trinity owns automatic observation, closure,
mutation, correction, and handoff work. `Model` and `Harness` remain visible secondary provenance
labels, and the compact, initially collapsed **Actions** drawer preserves event chronology without
repeating actor identity in every row. Character attribution is
presentation over typed events; it never changes a controller event into a model tool call or
claims that a model requested an automatic action.

The web transcript treats stored execution detail as progressive disclosure. A standalone
inference keeps a stable one-line marker. Consecutive tool-directed inferences by the same teammate,
including their intermediate reasoning notes, are presented as one collapsed action run with one
avatar and author line; the notes remain available inside the expanded run. Stage changes, teammate
chat, user messages, and material Trinity feedback remain explicit boundaries. Internal bounded-turn
accounting does not split a run. A terminal repeat-limit error remains durable evidence but is not
rendered as a second actorless red card when Trinity's terminal delivery feedback already explains
it; an adjacent workflow stop is combined into that same feedback card without merging the
underlying durable causes.
A fixed-size information affordance
opens aggregate and per-call metrics without changing line layout. The session runtime summary uses
the same labelled metric sheet. Harness artifact corrections render as Trinity
feedback with human-readable context and expandable technical detail. Controller closure checkpoints
remain durable events but are excluded from chat. Terminal session summaries and web delivery
summaries derive commits and diffs from the task's captured repository baseline rather than an
arbitrary recent-log window. For older strict-workflow summaries that predate this baseline, the web
UI shows stored commit lines only when the event stream also contains a successful typed commit
receipt.

Proactive LSP collection uses the same typed action stream. The transcript shows Trinity inspecting
language diagnostics and summarizes clean, blocking, stale, or incomplete results; the durable
tool result retains the bounded structured report. This is truthful controller attribution, not a
fabricated model request or a synthetic assistant/tool-call exchange.

Content fingerprints identify present worktree entries and bytes, independently of Git index
bookkeeping. In particular, the synthetic tracked-`missing` entry visible before staging a deletion
is excluded. Staging and committing an already-reviewed deletion therefore preserve the checked
content identity, while restoring or changing any worktree path still invalidates its evidence.

Local and managed command failures retain their exit status and bounded stdout/stderr in structured
tool feedback. Output redirected from stderr to stdout is still preserved. Timeout and user
cancellation are distinct results and terminate the owned process group or managed exec. A failed
command therefore provides actionable diagnostics while receiving no check or completion credit.

The step-limit monitor parses explicit negative boolean fields as values, not keywords. A healthy
checkpoint containing `off_track: no` or `blocked: no` can therefore grant its bounded extra step;
actual loop evidence, a blocked/off-track status, or an explicit no-grant decision still stops.

A newly started workflow plans from the current invocation baseline. This is normally identical to
the task baseline, but on a persistent scratch resume it includes explicitly adopted partial work.
The original task baseline remains available for final task-delta ownership, while the current
snapshot prevents planning from receiving an impossible stale fingerprint.

Implementation and repair prompts also include an authoritative state for every planned path:
missing, present unchanged, created in this task, modified in this task, or deleted in this task.
A resumed model therefore does not need volatile TODO memory to know that an earlier `create`
already succeeded, and is explicitly told not to call `write_file` for an existing path.

During a strict workflow stage, pb may recover a complete unwrapped JSON object (plain or in one
JSON code fence) when the exposed schemas identify exactly one intended tool. The only ambiguity it
resolves is `write_file` versus `replace_file`, using whether the bounded workspace target already
exists. The recovered call still passes ordinary path, policy, schema, stage, and artifact
validation. A workflow terminal may also accept its declared typed artifact when the outer function
arguments are valid but the artifact field contains one complete JSON-encoded string. pb parses
that string exactly once, validates the resulting typed artifact normally, preserves the original
call in the transcript, and reports the normalization in the tool result. It does not repair,
complete, recursively decode, or invent artifact values. Prose, partial JSON, arrays, and any other
ambiguous object are not coerced. Stage prompts prefer native function calls but also state the
exact compatibility action shape for model runtimes that cannot emit them.

The preferred terminal wire form is flat (`id` beside the artifact fields) and omits fields with
safe empty defaults. pb reconstructs and validates the same durable `ArtifactEnvelope`; the older
nested artifact object remains a bounded compatibility form and produces the same artifact digest.
For native Qwen/GLM and DeepSeek FlashMoe stages, generation-time constraints compile the actually
exposed tool names and supported JSON-schema subset before inference. Qwen/GLM uses its JSON tool
envelope; DeepSeek uses its DSML control-token identity and ordered parameter dialect. Terminal-only
turns require their single terminal tool. Candidate filtering cannot grant authority or replace
executor validation, and an unsupported schema fails preflight rather than silently disabling
constraints. The supported scalar subset includes exact `const` values as well as `enum`; both
dialects reject a non-matching value while it is generated, and preflight rejects contradictory
`const`/`enum` declarations. Object and array `const` remain unsupported by native tool constraints.
The controller also identifies an exposed stage-submission tool as terminal. Once its constrained
JSON body is complete, native generation stops semantically; FlashMoe and both fresh and cached
llama.cpp requests supply the exact missing Qwen envelope close only to the structured parser.
They never synthesize or repair the JSON body. Ordinary tools remain batchable, so this is not a
one-call-per-prompt policy. At constrained structural frontiers pb deterministically emits a unique
validated closing suffix, requires every non-EOS token to increase decoded length, and blocks a
repeated 32-token continuation. A file-content string that reaches its declared limit while still
open is instead returned as a truncated named mutation, making it eligible for compact recovery rather
than force-closing and executing a cut-off file. Escaped string prefixes are measured after JSON
decoding, so schema `maxLength`
remains authoritative. These guards only select output accepted by the compiled schema; the normal
parser, capability checks, and executor validation still run afterward.

**Shipped.** FlashMoe mutation generation also uses the workspace-internal `pb-control-collar`
library. Immediately before each generated attempt, the controller copies only current,
fingerprint-matching complete-file reads into a bounded immutable snapshot. For a target-scoped
work unit, that snapshot also carries the accepted path out of band. The model-visible schema can
therefore omit `path`; the JSON prefix extractor, mutation gate, final parser, and executor all
normalize to the same controller-owned target without making target paths part of schema identity.
A model-supplied path cannot override that binding. The collar has no live
filesystem, Git, process, model, or capability access. It parses Qwen JSON or DeepSeek DSML, binds
mutation payload closure to the virtual result, and rejects an invalid supported complete file or
canonical patch while a repair token can still be sampled. Qwen retains full-vocabulary widening;
DeepSeek probes candidates from its full-logit output and widens the candidate frontier before
top-k truncation. A request that exposes a FlashMoe mutation tool without the required snapshot
fails closed before decode.
DeepSeek complete-state checkpoints for structured generation are namespaced by the exact rendered
stable-root token digest. Repeated turns with the same root reuse the exact prefix; a changed tool
schema or authority shape starts cold rather than restoring incompatible Metal state. Raw harness
prefix-extension sessions retain their explicit session identity.

If every vocabulary candidate is rejected after a mutation branch has already produced visible
bytes, FlashMoe returns a typed `constraint_dead_end` with the partial transcript and executes
nothing. The controller permits at most one fresh, same-tool, same-target mutation retry; a second
dead end follows the ordinary incomplete-workflow path. Candidate-only state is never repaired into
an executable call.

**Shipped Rust v2 behavior.** If a real local backend exposes a tool schema that can edit a Rust
file, pb captures the exact project, creates a verified immutable shadow, and loads/primes one
project-wide pinned rust-analyzer HIR/Salsa world before prompt reservation, durable invocation
accounting, or backend entry. Cargo metadata is offline. Preparation of one exact project identity
is process-wide single-flight, so concurrent edit-capable tasks wait for and reuse one load instead
of duplicating it. The owner hands the exact completed world directly to registered waiters before
releasing the flight; reuse therefore does not depend on the small process LRU retaining that world
while unrelated projects load. Cold loads execute in a request-independent in-process worker, and at most one
cold Rust world is constructed process-wide at a time. Exact worlds are reused across turns and
workflow stages. Changes to already
indexed `.rs` files use Salsa invalidation only after all request snapshots from the previous
revision are gone; Cargo/configuration/dependency changes,
file topology changes, overlapping requests, ambiguity, or refresh failure build an independent
world before the next inference. Any live workspace drift during a load aborts that invocation; it
cannot silently rebase the already captured mutation snapshot, and a later attempt must recapture
both identities. An exact controller-bound non-Rust target bypasses Rust
preparation; `apply_patch` remains conservative because one patch can touch several paths.
Cancellation observed while native preparation is running is checked again at the readiness
boundary and prevents prompt work, budget reservation, invocation accounting, and model entry. Both
the request that starts a detached cold load and requests queued behind the exact flight or the
global one-worker capacity poll cancellation at 100 ms intervals. Cancellation abandons that
request wait, while the already-started exact local load may finish and populate the bounded cache.
The pinned in-process rust-analyzer loader still cannot cooperatively stop every project-discovery,
Cargo, VFS, or Salsa operation; this is a bounded request-wait guarantee, not immediate reclamation
of analyzer CPU or memory. Initial snapshot capture and incremental refresh retain operation-level
cancellation boundaries.

The decoder receives only a request-local read snapshot. The Rust layer streams known/generated
bytes and deletions through a Rust parser and resolves target/import/public callable shapes directly
through HIR. Project-local facts remain repairable while later hunks or files can change them;
qualified immutable dependency facts can still reject exact token-time contradictions. During
pre-inference readiness, an exact profile also constructs a second independently writable Salsa
database from the loaded graph, roots, configuration, and VFS inputs. At tool closure, every
completed modification to an already indexed Rust source is applied to that database together.
The layer checks all local target modules and compares only its promoted unresolved-name,
supported-import, field/method, privacy, selected call/type/trait-bound, mutability, and
moved-from-reference categories against baseline debt. A later candidate file can therefore repair
an earlier one before the transaction is accepted or rejected; an API change can also expose a
promoted error in an untouched local module. The mutable database is restored before its serialized
request lock is released. Build scripts and procedural macros make the deep profile partial; source
creation/deletion, unsupported relative imports, complete ownership/borrow analysis,
trait-implementation completeness, compiler checks, and runtime behavior defer instead of
rejecting. Immediately before executor entry, pb reconstructs the completed virtual
write/edit/replacement/patch—including the final hunk and untouched base tail—and replays it through
a fresh stack bound to the same live world identity. Candidate checkpoints retain append-only
source lengths and parser/semantic state rather than copying the virtual file for every accepted
token. When a streamed tool string closes, the generation gate first synchronizes the remaining
known suffix or patch base tail, emits the final file boundary, and finalizes every participating
language layer; closing a candidate does not bypass a closure-only diagnostic. The enclosing sampler
checkpoint can restore that closure and probe a sibling token from the exact earlier parser,
virtual-file, and semantic state.

The hidden model-free Rust semantic qualifier uses an ephemeral offline Cargo workspace and the
ordinary production lifecycle. Its checked corpus compares generation closure, independent executor
replay, and direct content-free diagnostic/unknown results for all promoted classes, baseline debt,
cross-crate transactions, source-topology unknowns, and all four mutation tools. It probes every
logical UTF-8 payload boundary and 64 deterministic rollback/full-replay branches per case; after a
prefix becomes impossible, every longer prefix must remain impossible. It runs no model, LSP, build
script, procedural macro, or generated mutation. Passing proves only the pinned profile and corpus,
not rustc equivalence.

**Shipped Python v1 behavior.** If a real local backend exposes a tool schema that can edit or
create a Python file, pb creates an exact verified shadow—even if the project has no Python file
yet—and loads/primes exact-pinned Astral `ty` state
before inference. Python 3.12 is the fallback; when exactly one conventional project-local
`.venv` or `venv` can be qualified, its `pyvenv.cfg` selects the supported Python version and pb
copies bounded `.py`, `.pyi`, `.pth`, `py.typed`, and distribution-metadata inputs from its
`site-packages` into the immutable shadow. A static plain-path `.pth` target inside the repository is
accepted only when every relevant file is already controller-observed, then becomes an additional
first-party root so candidate edits and dependant closure use the same overlay. A repository-owned
`.pb/python.toml` may select one exact in-workspace environment but cannot authorize an external
read. External environments and exact external plain-path editable roots require user-owned
configuration bound to the canonical workspace; those static trees are copied into separate frozen
dependency roots. The runtime-owned grant is not serialized into an agent request.

The host platform, first-party roots, bundled typeshed, configuration inputs, dependency manifests,
external-root identity, and the separate dependency-image digest identify the world. Dependency
traversal is explicitly sorted, so identical bytes cannot acquire a different identity from
filesystem enumeration order. Ignored environment files are therefore covered without becoming
repository files. Each request gets an
independently writable Salsa database over shared frozen bytes; this avoids treating a read-only
Salsa snapshot as a mutable overlay. Every captured dependency module is interned and type-primed
before the request reaches token generation. Exact worlds use a bounded process cache and the same
direct-result single flight, with at most one cold
Python loader. Cancellation is polled while initiating or queued requests wait, and no model starts
after cancellation. When Rust and Python are both reachable, Rust preparation completes first and
the model starts only after both readiness receipts exist.

The Python stream keeps known/generated origins, verifies a complete structural boundary before
semantic pruning, and hard-rejects only a generated string-plus-integer literal addition confirmed
by the pinned checker. The per-code policy marks `invalid-argument-type`, `invalid-assignment`,
`invalid-return-type`, `unresolved-attribute`, and `unresolved-import` as closure-only: even a
parser-complete statement cannot hard-reject on those diagnostics because a later file in the same
patch can repair it. `unsupported-operator` has token-time authority only through the separately
qualified generated-literal proof; its broader checker diagnostic is still closure-scoped. Complete create/modify/delete candidates are overlaid together, with deletion
represented as absence, so new imports and callable signatures resolve coherently within the
transaction. Finalization checks every non-deleted frozen first-party file plus newly created Python
files and compares promoted diagnostic multisets against baseline debt. This catches promoted errors
introduced into untouched in-project dependants by a changed or deleted public symbol. Preserved
baseline suppressions remain baseline, while generated `ty: ignore` or `type: ignore` directives
reject. A complete captured static environment may prove a missing external import; resolved
dependency source/stub shapes can participate in the same promoted type checks. Multiple local
environments, undeclared or symlinked layouts, native modules, `.pth` import hooks, absent
environments, `Any`, dynamic behavior, unpromoted diagnostics, and dependants outside the frozen
project remain partial or unknown. Search roots and `.pth` line lengths are explicitly bounded;
missing configured roots, non-directory targets, nested-search artifacts, duplicate authority, and
resource overflow cannot produce a complete environment. A nonexistent plain-path `.pth` target is
ignored exactly as Python would ignore it. Without a complete qualified external search space, a missing
absolute third-party import cannot veto generation. Immediately before execution, pb recaptures the
dependency identity, reconstructs the mutation, and replays it through a fresh Python request
database against the same frozen world.

The hidden model-free native-world qualifier runs the exact production Python or Rust lifecycle in
separate processes over deterministic tiny, representative, and large project/dependency graphs. It
records native load/prime time, the complete cold pre-inference barrier, warm request construction,
exact process-cache reuse, invalid and valid execution replay, and peak resident memory. Multiple
host workers then submit alternating invalid and valid completed mutations to the same lifecycle;
the language-owned writable database must serialize them, return the exact decision for every
replay, and accept one final recovery replay. Current resident memory before and after that stress is
bounded separately from whole-process peak RSS. The command fails closed against explicit
language-specific ceilings and emits only content-free identities, counts, byte totals, and timings.
The matrices establish reproducible lifecycle, serialization, and reclamation baselines; they are
not universal latency or memory guarantees for projects with different syntax, macros, plugins,
dynamic environment behavior, or dependency shapes.

The separate Python semantic qualifier keeps this lifecycle outside inference while exercising
language behavior. Its checked-in corpus freezes first-party files and a typed third-party package
surface, then compares the production generation closure, independent executor replay, and the
language crate's content-free promoted diagnostic-code delta for every case. Annotated,
unannotated, baseline-debt, dynamic-unknown, dependency, and multi-file cases all use the same
native `ty` resolver and request overlays as production; no live LSP or incomplete-document
diagnostic stream participates. Corpus validation requires both valid and invalid examples for the
three primary profiles, every promoted diagnostic class, and write/replace/edit/patch coverage.
It also probes every logical UTF-8 payload boundary and 64 deterministic rollback/full-replay
branches per case under the same monotonic-rejection rule as Rust. This is differential evidence for
the existing profile, not authority for an additional token-time rule or a universal Python
soundness claim.

The syntax profiles are pinned Tree-sitter grammars for Rust, Python, TypeScript/TSX,
JavaScript/JSX, HTML, and CSS. They require UTF-8 and reject error or missing nodes; HTML additionally
checks explicit element closure and supported embedded JavaScript, TypeScript, JSON, and CSS. This
is the cross-language baseline, not a general name-resolution, type, borrow, module-resolution,
CSS-semantics, or Python runtime guarantee. Rust v2 adds exact immutable-dependency steering and its
explicitly promoted complete-transaction diagnostic allowlist; it is still not rustc or a full
borrow checker. The conservative prefix layer advances only over newly decoded
logical payload bytes, keeps constant-time persistent-stack checkpoints for candidate branches, and
probes canonical patch additions/context before line closure; final Tree-sitter validation still
decides complete-file acceptance. Qwen and DSML byte-BPE fragments that end inside a UTF-8 scalar
  remain eligible only when the active protocol/schema has at least one bounded viable scalar
  completion; candidate probing restores the same reusable protocol, patch, and language-layer
  batch base between sibling tokens. It does not allocate a replacement analyzer checkpoint per
  vocabulary entry. The checked-in tokenizer qualifier covers both production Qwen and DeepSeek
tokenizer boundaries without loading model weights. A hidden no-execution llama.cpp fixture runner
also loads an exact GGUF and reports complete calls plus content-free constraint counts; one pinned
Qwen profile passes valid write/patch and fail-closed malformed/truncation cases. This is profile
evidence, not a claim that every llama.cpp template or the DeepSeek llama.cpp dialect is promoted.
Deeper Rust/Python layers and a future native TypeScript/JavaScript crate use the same versioned
event/checkpoint interface only after each error class has its own soundness corpus and fail-closed
qualification.

Strict JSON artifacts use a separate tokenizer-neutral constraint session and cannot be combined with
native tools in one request. FlashMoe computes the LLGuidance allowed-token set before top-k. On
Qwen-family output heads, the bitset is uploaded to the resident Metal vocabulary kernel so invalid
tokens are excluded before candidate selection and full vocabulary logits are not read back to the
host. Resident and streamed experts share this output-head path. DeepSeek applies the same bitset to
its existing full-logit Metal output. Every sampled token is committed back to LLGuidance, and pb
rejects an incomplete or unparsable terminal artifact even if the model exhausts its token budget.

Implementation and repair turns keep their edit tools after a rejected prose final. They are not
narrowed to `submit_implementation` before the model has had another chance to make the repository
change that the correction requires.

Plan paths are validated in step order. `create` makes a missing path available to later `modify`
or `delete` steps, and `delete` removes it from the subsequent plan state. A modification before
creation and a duplicate creation while the path exists remain invalid. A path that starts and ends
missing is also rejected because no durable repository delta could prove the work happened.

After acceptance, conceptual operations on the same path compile to one path-transition work unit.
The unit retains every contributing plan step id but derives its authority from the task baseline and
required final state: create for missing-to-present, delete for present-to-missing, and modify for
present-to-present. Repository state can therefore complete only one unit for a path; an early create
cannot accidentally complete a later conceptual modification, and a temporary delete cannot become
an impossible outstanding unit. Legacy checkpoints containing repeated path units fail validation
and require replanning instead of inheriting completion under the new meaning.

When a trusted harness contract restricts paths, planning validates every proposed path against
that allowlist before implementation. Prompt examples use workspace-relative paths and explicitly
warn that `repo/` is not a magic prefix, so an impossible plan is corrected before any write turn.

## Context compaction and inference reuse

**Shipped.** pb renders and tokenizes every prompt before inference. It reserves the requested
generation space plus a fixed safety margin, begins compacting above 70% of usable prompt capacity,
and targets 60%. Only completed assistant/tool groups become deterministic receipts. The system
instructions, task, active contract/stage material, accepted artifacts, checks, fingerprints, and
terminal requirements remain byte-for-byte authoritative. If those anchors cannot fit, the call
ends with `context_limit` instead of asking the model to summarize or silently dropping evidence.

This is sufficient for correctness and bounded long-running agent use. A model-authored rolling
summary would weaken the authority and audit guarantees, so it is not the next compaction step.
Future compaction work should be driven by measured retrieval or quality failures and should retain
the same deterministic receipt and durable-event boundary.

Compaction is also cache-safe. Both local inference paths compare exact rendered tokens before
reusing attention state. llama.cpp keeps the live context across passes and saves byte-budgeted
restartable state. The service keeps FlashMoe model/Metal runtimes resident across managed turns; a
bounded LRU of logical sessions retains both a safe prompt checkpoint and an exact generated-token
head, while the stable first-system-message prefix is content-addressed for cross-session reuse.
FlashMoe also persists full-attention KV or MLA KV, linear-attention recurrence, the final hidden
state, and token ids in a model-fingerprinted local cache. A sessionless JSON-constrained Task call
may restore and publish only that stable first-system-message state; its constraint engine, dynamic
evidence, and generated artifact are fresh request state. The provisional DeepSeek V4 Flash path
does not use those snapshots: its four-stream hyperconnection and raw/compressed/indexer KV state is
reset as one request-scoped allocation, rather than partially restoring a snapshot whose schema was
designed for Qwen/GLM state.

Dynamic branch, recent-commit, environment-evidence, and per-turn discussion authority are rendered
as a second system message. The first system message and its authorized native-tool schema therefore
form a stable checkpoint without weakening tool authority. A changed token never reuses later state.
If compaction or a stage/tool-schema transition diverges before the newest checkpoint, pb falls back
to a matching stable prefix when one exists and evaluates the remaining suffix normally.

Every `llm_invocation` event attributes the call to conversation, Task partitioning, workflow
planning, review, evidence gathering, mutation, closure, or recovery, and records the active stage
and profile when present. Duration, prompt/generated tokens, prompt-cache usage, and energy estimates
therefore remain attributable even for constrained Task-planning calls outside the stage loop.

Each `llm_invocation` records total prompt tokens, cached tokens, actually-prefilled tokens, cache
source, disk-restore time, and a backend-owned reason when no prefix was reused. The typed reasons
distinguish disabled caching, a cold session, prompt divergence, a missing stable prefix, an
unreadable persisted cache, a required context reset, and a runtime path that cannot reuse state.
Successful partial or full reuse has no miss reason. These fields distinguish context size from work
performed without guessing at cache behavior from latency alone, and the terminal and web transcript
surface the same attribution.

When a backend can identify a reusable stable root, the cache record also carries a versioned exact
rendered-token digest, model-namespace digest, eligible and reused root-token counts, and a
controller-owned bounded authority class. Before inference, the controller binds an explicit system
instruction version, workflow stage, authority class, canonical rendered-tool-schema digest, and
decode-constraint mode to the request. Model-facing tools are sorted canonically. The versioned
first-system-message root contains the immutable stage role, terminal action protocol, and handoff
authority rule; these strings carry no task or repository material. Dynamic task, repository,
branch, environment, contract, plan, correction, and evidence content remains after the root.
Planning, review, implementation, repair, and closure use finite evidence/authorized/terminal
authority classes instead of incidental readiness combinations. Any managed workflow state without
a classified authority fails before inference. These labels are diagnostic only: the backend's
exact rendered-token comparison remains the reuse authority. A decode-only constraint can change
without invalidating identical prefetched KV state, but the fresh constraint remains recorded and
enforced.

Exact small contract-path evidence remains outside that root. Planning and isolated plan review each
receive a stage-bound controller rendering made from bytes re-read in that stage's exact workspace.
The controller rechecks workspace and path fingerprints after prompt preflight, records full byte
coverage, and skips unsafe or oversized candidates. Review keeps its normal repository-read tools,
so the optimization changes evidence placement rather than review authority.

Durable checkpoints and audit events retain the complete observation receipts. Later model stages
receive a compact path/content projection only after current-hash validation, while isolated plan
review avoids even that copy when the same eligible contract bytes will follow as stage-bound
controller blocks. This reduces fresh prompt work without weakening the structured evidence gate.

Initial planning has one additional bounded authority transition. If a trusted non-empty path scope
has fresh full controller observations for every existing allowed path, the first model generation
uses the task-independent `PlanningClosure` root and exposes only `submit_plan`. A missing allowed
path is already explicit in harness path-state evidence; any unsafe or unobserved existing path
keeps planning tools exposed. Independent plan review keeps repository reads unless the same exact
full observations satisfy its own fresh closure proof; any challenged plan revision is tool-enabled.
The controller therefore removes only duplicate reads, not semantic review or recovery authority.

FlashMoe native usage further separates memory lookup, disk open/read/decode, CPU validation and
allocation, Metal state hydration, actual fresh-suffix prefill, prompt-snapshot capture, and durable
write queueing. Session metrics report how many queued checkpoints completed, their wall time, and
any failure. A bounded lookup-detail enum distinguishes a missing session, a divergent session, a
missing exact root, and a divergent-session fallthrough that hit or missed the root. These
non-overlapping lifecycle counters keep a fast cache hit, slow disk decode, slow
Metal restore, slow model prefill, and slow persistence from being collapsed into one TTFT number.
The native record also names the selected scalar, cache-only, DeepSeek batch, or Qwen layer-major
command and a bounded selection reason derived from the actual remaining suffix and prepared
resource state.

## Where the workflow ends

Ready means local delivery is complete under the configured contract. Publication is a separate
workflow because pushing, opening a pull request, and responding to provider state are external
mutations with different approvals and idempotency needs.

Goal completion is one level higher. `WorkflowReady` criteria may be machine-verified from strict
workflow evidence. Review-required or user-confirmation criteria stop at Goal **Ready for review**
until the user accepts the exact current Goal checkpoint. Complete, user-accepted, budget-reached,
failed, and cancelled are durable distinct outcomes. None adds publication authority.

For the detailed implementation record and transition invariants, see the
[conversational delivery workflow plan](../conversational-delivery-workflow-plan.md).
