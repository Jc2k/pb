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

### 1. Planning

The planning profile reads the repository and submits a structured plan with scope, implementation
steps, and checks. If a user-owned choice is missing, planning can ask a question. The stage cannot
edit files or run a shell.

### 2. Plan review

A review profile receives focused evidence in a fresh context and either accepts the plan or
returns actionable findings. Rejection returns the workflow to bounded plan revision. The reviewer
does not inherit mutation authority.

The workflow checkpoint carries a byte-bounded bundle of complete small-file reads between stages.
pb records the exact bytes, path/content hashes, source stage and normalized read arguments; it
revalidates every path before injecting the bundle or seeding the new stage's observed-read ledger.
A changed or oversized/partial file therefore still requires a fresh read. Model conclusions and
TODO prose are never promoted as evidence.

### 3. Implementation

The build profile receives the accepted plan and mutation capabilities. It can use built-in edits,
configured tasks and checks, or a journaled command escape hatch. It submits an implementation
artifact that identifies what changed and the evidence it produced. It never receives `git_commit`.

### 4. Checks

The harness selects affected configured checks and records their current results. Named acceptance
checks cannot be replaced by an arbitrary command that happens to exit successfully. A mutation
after a check makes older evidence stale.

### 5. Code review

A fresh read-only reviewer inspects the delivered change and current focused evidence. An accepted
review is bound to the content fingerprint it saw. Findings enter a bounded repair cycle; repair
then refreshes checks and review.

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
Strict stages do not expose the legacy conversational `todo` tool: the accepted plan, typed stage
artifacts and checkpoint already own durable progress.

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
stops loops deterministically.

When a model response is truncated before producing a valid action, pb can retry with thinking
disabled and, within the same global budget, use a larger output cap. A capped native `write_file`
or `replace_file` also receives at most one compact atomic retry inside that same stage step. That
retry exposes only the attempted mutation tool, starts from the original authoritative messages
rather than carrying the rejected oversized payload, and requests a complete loadable payload
below half of the ordinary mutation allowance. When the truncated call exposed its path, the retry
schema binds that exact original target, and its generation-token ceiling is reduced to the smaller
serialized payload allowance. The retry schema enforces those constraints rather than relying on
the correction text alone; pb also rejects a parsed retry that changes the bound tool or path. A
failed compact retry does not grow back to the original cap. These recovery mechanics help the
model express an allowed action; they never expose a new capability or waive a transition gate.
Retries consume model-invocation and generated-token budgets, but remain part of the same visible
stage step. Stage-step accounting is checkpointed at the `StepStarted` boundary, so an action-only
retry cannot prematurely terminalize a workflow while its result is still being recorded.
Equivalent max-token native actions are correlated by attempted tool name and the current workspace
and evidence fingerprints. A merely parsed action does not erase that history before its outcome is
known; a successful executed action clears it, while a real workspace or evidence transition also
changes the signature. For a capped file write, the correction explicitly says that no partial file
exists and requires materially shorter complete content. The third equivalent failure at unchanged
state terminates without another model turn.

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
Strict implementation schemas also derive a conservative mutation-string `maxLength` from the
turn's generated-token cap after reserving native-envelope and closing overhead. The same bound is
enforced again by recursive executor-side schema validation and appears in the stage anchor and
invocation telemetry. Larger files are built from complete, loadable, atomic work units rather than
partial JSON or partial filesystem writes.
An edit tool receives mutation and progress credit only when repository bytes actually change.
Identical replacements and edits fail without emitting a diff or invalidating existing evidence.

Local command failures retain their exit status and bounded stdout/stderr in structured tool
feedback. Output redirected from stderr to stdout is still preserved. A failed command therefore
provides actionable diagnostics while receiving no check or completion credit.

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

Implementation and repair turns keep their edit tools after a rejected prose final. They are not
narrowed to `submit_implementation` before the model has had another chance to make the repository
change that the correction requires.

Plan paths are validated in step order. `create` makes a missing path available to later `modify`
or `delete` steps, and `delete` removes it from the subsequent plan state. A modification before
creation and a duplicate creation while the path exists remain invalid. This lets a plan express
several conceptual phases on one new file without weakening filesystem-state checks.

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
state, and token ids in a model-fingerprinted local cache. The provisional DeepSeek V4 Flash path
does not use those snapshots: its four-stream hyperconnection and raw/compressed/indexer KV state
is reset as one request-scoped allocation, rather than partially restoring a snapshot whose schema
was designed for Qwen/GLM state.

Dynamic branch, recent-commit, environment-evidence, and per-turn discussion authority are rendered
as a second system message. The first system message and its authorized native-tool schema therefore
form a stable checkpoint without weakening tool authority. A changed token never reuses later state.
If compaction or a stage/tool-schema transition diverges before the newest checkpoint, pb falls back
to a matching stable prefix when one exists and evaluates the remaining suffix normally.

Each `llm_invocation` records total prompt tokens, cached tokens, actually-prefilled tokens, cache
source, and disk-restore time. These counts distinguish context size from work performed and make a
cache miss or invalidation visible in the web transcript.

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
