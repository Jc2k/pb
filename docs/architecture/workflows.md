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
does not inherit mutation authority.

pb projects the accepted plan ID and digest into the submitted review rather than asking the local
model to transcribe controller-owned identity. Every required assessment kind remains mandatory.
A passing assessment may omit repetitive evidence and explanation fields; a concern or failure
still requires a non-empty explanation, and cited repository evidence remains freshness-validated.
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
gate. A valid object closes implementation without a bookkeeping-only model turn. An invalid object
does not roll back or misreport the successful mutation; the ordinary bounded submission path stays
available on the next turn.

When a configured language server supports a changed accepted task path, pb proactively requests
diagnostics without waiting for the model to choose an LSP tool. While implementation is still in
progress, the syntax pass admits only error-severity diagnostics identified as parser or syntax
failures, avoiding expected semantic noise from half-finished code. Once the work-unit ledger is
structurally complete, pb ensures the current content version has received a settled pass before a
direct handoff or final-grace path; unchanged versions are deduplicated. That pass admits current
error diagnostics. A blocking result reopens only its exact current task paths for repair. It does
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

Every required code-assessment kind remains mandatory, while passing assessments may omit empty
evidence and explanation fields. Concerns and failures still require an explanation, findings retain
their typed evidence requirements, and review acceptance remains bound to the checked content
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
| File mutation | Existing files require an exact content fingerprint from the bytes actually read. Create and replace use synced temporary files and atomic no-clobber/replace operations; stale concurrent edits fail. `edit_file` requires one unique match. For an initial Python or Python-stub modify whose complete controller-observed file consumes at most half the bounded replacement allowance, pb exposes only atomic `replace_file`; this avoids indentation-fragment errors while requiring the model to preserve unrelated bytes. Other languages, larger or range-only observations, and diagnostic repair retain exact-edit behavior. Diff events are bounded and identify truncation. |
| Patch, move, and remove | `apply_patch` validates every path, checks the patch before applying it, and uses a bounded process. `mv` moves only a file or symlink and cannot overwrite. `rm` operates on the final filesystem entry and removes only a file, symlink, or empty directory; recursive directory mutation is not an agent capability. |
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
lower of that ceiling and the compiled workflow policy is cumulative across validation attempts.

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
round trips. Batch events record call, parallel-safe, useful, bookkeeping-only, and dependency-
rejection counts.
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
complete. Preview results are fingerprint-bound feedback only: they never enter selected-check
evidence and all required checks run again in the authoritative checking stage. A failed preview
can reopen only an exact current task path named as a complete diagnostic token. Earlier read
evidence for that path is invalidated; after a fresh complete read, repair uses a bounded replace or
edit even when the original plan operation created the file. A missing diagnostic target requires a
replan. The preview command must preserve repository content and Git control state.

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
for every step and owns status, summaries, and the proposed semantic commit subject. At code review,
pb similarly projects the checked fingerprint while the fresh reviewer owns assessments, findings,
and verdict. Existing artifact validators still reject missing steps, paths outside the plan, stale
state, incomplete work, or unsupported review conclusions.

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
fingerprint and complete prompt bytes remain valid. A failed-diagnostic range permits only an edit
whose old text lies wholly inside an included byte window. Fresh review inspection is injected only
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
mutation, correction, and handoff work. `Model-requested` and `Automatic` remain visible secondary
provenance labels, and the **Actions** drawer preserves event chronology. Character attribution is
presentation over typed events; it never changes a controller event into a model tool call or
claims that a model requested an automatic action.

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
For native Qwen FlashMoe stages, generation-time constraints compile the actually exposed tool
names and supported JSON-schema subset before inference. Terminal-only turns require their single
terminal tool. Candidate filtering cannot grant authority or replace executor validation, and an
unsupported schema fails preflight rather than silently disabling constraints.
The controller also identifies an exposed stage-submission tool as terminal. Once its constrained
JSON body is complete, native generation stops semantically; pb supplies a missing Qwen envelope
close only to the structured parser. Ordinary tools remain batchable, so this is not a one-call-per-
prompt policy. At constrained structural frontiers pb deterministically emits a unique validated
closing suffix, requires every non-EOS token to increase decoded length, and blocks a repeated
32-token continuation. A file-content string that reaches its declared limit while still open is
instead returned as a truncated named mutation, making it eligible for compact recovery rather
than force-closing and executing a cut-off file. Escaped string prefixes are measured after JSON
decoding, so schema `maxLength`
remains authoritative. These guards only select output accepted by the compiled schema; the normal
parser, capability checks, and executor validation still run afterward.

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
decode-constraint mode to the request. Model-facing tools are sorted canonically; dynamic task,
repository, branch, environment, plan, and evidence content remains after the root. Planning,
review, implementation, repair, and closure use finite evidence/authorized/terminal authority
classes instead of incidental readiness combinations. Any managed workflow state without a
classified authority fails before inference. These labels are diagnostic only: the
backend's exact rendered-token comparison remains the reuse authority. A decode-only constraint can
change without invalidating identical prefetched KV state, but the fresh constraint remains recorded
and enforced.

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
