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

## Freshness is part of correctness

Plans, checks, and reviews carry content or evidence fingerprints. A later mutation invalidates
evidence tied to the earlier state. This makes “reviewed” mean reviewed at the delivered content,
not merely that a review-shaped conversation occurred somewhere in the session.

The same rule applies during recovery. Persisted structured artifacts and fingerprints can restore
a stage; discarded or truncated prose cannot become a substitute source of authority.

## Bounded failure and recovery

The workflow budgets stage steps, model invocations, generated tokens, advisory calls, plan cycles,
and repair cycles. It also detects repeated no-progress operations against unchanged state and
stops loops deterministically.

When a model response is truncated before producing a valid action, pb can retry with thinking
disabled and, within the same global budget, use a larger output cap. These recovery mechanics help
the model express an allowed action; they never expose a new capability or waive a transition gate.
Retries consume model-invocation and generated-token budgets, but remain part of the same visible
stage step. Stage-step accounting is checkpointed at the `StepStarted` boundary, so an action-only
retry cannot prematurely terminalize a workflow while its result is still being recorded.

Compatibility tool turns are single-action boundaries. Implementation prompts tell the model to
stop after the action and never imitate pb's transcript or invent later results. If a local model
still emits a fenced action followed by fabricated `Tool calls:` or role entries, pb executes only
the validated action and omits that untrusted pseudo-transcript from subsequent model context. The
real tool result and content fingerprint remain the only mutation evidence.

Implementation guidance gives symmetric concrete actions for missing and existing paths. Missing
paths use `write_file`; existing paths use a separate `read_file` turn followed by
`replace_file`, `edit_file`, or `apply_patch`. An attempted overwrite keeps failing closed but now
returns that exact recovery sequence instead of a generic suggestion.

The step-limit monitor parses explicit negative boolean fields as values, not keywords. A healthy
checkpoint containing `off_track: no` or `blocked: no` can therefore grant its bounded extra step;
actual loop evidence, a blocked/off-track status, or an explicit no-grant decision still stops.

A newly started workflow plans from the current invocation baseline. This is normally identical to
the task baseline, but on a persistent scratch resume it includes explicitly adopted partial work.
The original task baseline remains available for final task-delta ownership, while the current
snapshot prevents planning from receiving an impossible stale fingerprint.

During a strict workflow stage, pb may recover a complete unwrapped JSON object (plain or in one
JSON code fence) when the exposed schemas identify exactly one intended tool. The only ambiguity it
resolves is `write_file` versus `replace_file`, using whether the bounded workspace target already
exists. The recovered call still passes ordinary path, policy, schema, stage, and artifact
validation. Prose, partial JSON, arrays, and any other ambiguous object are not coerced. Stage
prompts prefer native function calls but also state the exact compatibility action shape for model
runtimes that cannot emit them.

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

For the detailed implementation record and transition invariants, see the
[conversational delivery workflow plan](../conversational-delivery-workflow-plan.md).
