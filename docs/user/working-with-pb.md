# Working with pb

pb separates discussion from delivery. That separation is a user-facing authority boundary, not
just a change of prompt wording.

## Discuss

Use **Discuss** in the web interface for explanation, brainstorming, repository inspection, and
planning. A discussion is read-only. It can consult bounded advisory agents and it can propose a
delivery, but it cannot edit the project or start delivery on its own.

Discussion is useful when the important output is understanding:

- explain a subsystem or unfamiliar diff;
- compare approaches and surface trade-offs;
- investigate a failure without changing anything;
- shape a delivery request before granting mutation authority.

Only your explicit **Build** choice promotes a web conversation into the delivery workflow.

A discussion may also offer a read-only Goal proposal. The proposal does not start repository work.
You can review and edit it in the Goal setup sheet. The internal Auto intent may ask pb to create a
Goal for the exact current turn, but the resulting milestone plan still waits for your approval.

## Build

Use **Build** when the desired outcome is a repository change. A queue task is already explicit:

```bash
pb queue "Implement the accepted retry behavior" --workdir /path/to/project
```

Delivery moves through enforced stages: planning, independent plan review, implementation,
configured checks, independent code review, bounded repair when necessary, and a managed commit.
The active stage controls which tools are visible and which structured submission can advance the
workflow.

Small complete files read during one stage may appear in the next stage as harness-carried evidence.
pb rechecks their hashes first; changed, partial, or oversized reads must be inspected again. Strict
Build stages use the accepted plan and checkpoint for progress instead of exposing a second TODO
protocol. Independent tool calls can share one model response, while dependent same-path or
mutation/check batches are rejected before any call runs.

When a trusted contract names existing small text files as allowed paths, pb can place their exact
current contents into the first planning turn automatically. The transcript labels these as
automatic controller observations. If a file or the bounded context is unsuitable, pb simply leaves
the normal read tool available.

During implementation, pb exposes one accepted-plan file operation at a time and inserts that
controller-owned target into the action before validation. Consecutive new files do not share one
model output budget. If constrained generation reaches a file-payload boundary well before its token
ceiling, pb retries once with more string room at the same token limit; real token exhaustion instead
gets one smaller complete-file retry. Neither case writes a partial file.

For the last remaining file operation, the local model returns the implementation summary with the
mutation. pb validates that summary only after the edit succeeds, so a valid edit can finish the
implementation stage without a separate bookkeeping turn. Review pass records are similarly
compact: pb supplies controller-owned plan identity and does not require empty explanations, while
concerns and failures still need specific reasons and fresh evidence.

If a required check fails against the only contract-allowed path, pb keeps the accepted plan and
offers a focused read or repair rather than another identical planning cycle, including when an
assertion reports rendered output without repeating the filename. Replanning remains available for
real scope choices and repository-state blockers; the single-path rule cannot widen or change the
trusted contract.

You may see pb pause for a planning question when a missing choice would materially change the
work. Answering that question updates the user-owned contract; it does not hand the model a general
permission to improvise.

### Message a running task

**Shipped.** The session composer remains available while a task is running. A message sent there
is added to the existing agent conversation at the next primary agent loop boundary; it does not
start a follow-up Task, select Discuss, Build, or Goal, or create a Goal. This lets you correct an
assumption, add context, or redirect the current approach without stopping the session.

Messages are recorded before the agent receives them and acknowledged when the harness adds them to
the prompt. If pb restarts before that boundary, the still-pending message is restored with the
paused session. A message cannot widen the current stage's tool authority, accepted plan, repository
scope, or publication boundary. During a long model inference or tool action, the current operation
finishes first; the message is picked up on the next loop.

The workflow routes feedback to an author, never to a fresh-context critic. A message sent during
plan review sends the workflow back to plan revision. A message sent after the implementation
submission, during checks or code review, or just before commit invalidates the stale downstream
evidence and restarts planning from the current repository state. A typed stage submission is
deferred in the same way as an ordinary final response when a message is already waiting.

Immediately before the task emits its final response or creates its managed commit, pb atomically
stops accepting in-flight messages. A send that loses that final race reports a conflict instead of
appearing accepted and then being ignored. Goal and multi-Task sessions reopen the message window
when their next model stage begins.

### Actions in web and terminal

pb performs narrowly safe routine work—such as an eligible file read or changed-path inspection—
without asking the model to choose the obvious next action. This is intrinsic workflow behavior.
The model receives one explicit pb-owned context block, never a fabricated assistant tool call.

The web transcript and **Actions** drawer show this work as messages and actions from the team. The
active profile character owns tools the model requested—for example, Kate owns Build tool calls.
Trinity Walker, the team steward, owns automatic reads, changed-path inspection, safe deletion,
no-change closure, corrections, and handoff work. Each action also says **Model-requested** or
**Automatic**, so the friendlier character metaphor does not hide provenance. Reads show their full
or bounded-range coverage; deletion states that the path was tracked and Git-recoverable. The
terminal uses the same character-first attribution and provenance.

Model-inference rows show why the call was needed, prompt and generated token counts, energy when it
is measurable, and prompt-cache work. A zero-reuse call includes the backend's reason—such as a cold
session, changed prompt, unavailable stable prefix, unreadable cache, required context reset,
disabled cache, or an unsupported runtime path. A cache hit reports reused and fresh token counts and
does not carry a miss reason. When a stable root is available, the row additionally shows how many
eligible root tokens were reused and the bounded workflow authority class. The stored diagnostic
uses local digests and token counts rather than prompt or repository text.

Sessions saved before actor attribution remain readable. Their model tool actions appear as
**Legacy action** rather than being guessed from the closest chat message.

If a file is missing, binary, symlinked, stale, oversized, or cannot fit the bounded prompt safely,
pb does not pretend it was read. The normal model/tool path remains available. Automatic deletion
has stricter gates and applies only to a unique accepted tracked-clean path that is unchanged and
recoverable from Git.

## Tasks

**Shipped and on by default for new Builds.** pb first decides whether a Build request could benefit
from outcome-shaped **Tasks**. A bounded request with at most three behavior clauses stays one exact
Build without calling the model unless it explicitly requests separate work or ordering. Larger or
explicitly coordinated requests ask the selected local model for a partition. This sits above
ordinary Build planning:
after the previous Task commits, each Build Task starts a fresh normal plan against the repository it
received and decides its actual changes, checks, documentation, review, and commit boundary.

The high-level model output is deliberately small: `{"tasks":["first request","second request"]}`.
Each string is the actual outcome text later passed into a fresh Build workflow. For multiple Tasks,
the model must preserve every controller source clause in exactly one request; it may add only the
context needed to make the boundary independently deliverable. File extensions, function arguments,
and comma-delimited behavior lists stay intact when pb finds those clauses, and insignificant
whitespace beside punctuation does not defeat ownership matching. pb retains the exact
original request and owns Task IDs, derived UI titles, dependencies, acceptance records, Build
budgets, and authority. Both llama.cpp and FlashMoe constrain generation to this JSON schema and stop
as soon as the complete value is decoded. Rust proves disjoint ownership and explicit `before`,
`after`, and `then` order. There is no extra model critic, and there is at most one revision for a
deterministic rejection.

The number of accepted Tasks determines the experience:

- A bounded simple request records a zero-attempt single-Build decision and starts normal planning.
- One Task is discarded and the exact original request continues as the existing Build workflow.
- Invalid planning also continues with that unchanged Build request.
- Two or more accepted Tasks show a **Tasks** panel with ordered state, Build kind, active Task, budget
  use, commits, and any blocked reason. An active Goal's existing controls appear inside that Goal
  Task.

Only one Task runs at a time. pb plans a Task against the repository left by its predecessor,
requires the predecessor's managed commit or verified no-change result, checks the exact repository
state, and only then queues the next Task. Stopping or exhausting a Task preserves completed local
commits and prevents dependent Tasks from starting. A restart restores the exact active request and
remaining allowance, paused at a safe boundary.

The expandable **Task planning details** record explains every route. A simple-request bypass has a
reason and zero attempts. An attempted partition also preserves the local prompt, schema, raw
constrained JSON, normalized artifact, failure, token/runtime usage, and controller decision; it is
model I/O, not hidden reasoning. Before a multi-Task run becomes Ready, pb also
checks every original requirement against successful Task evidence, acceptance IDs, commits, and
the exact terminal repository.

Default decomposition creates Build Tasks only. Explicit **Goal** mode continues unchanged and still
waits at its normal approval boundaries. Automatic Goal-shaped Tasks require a separately promoted,
exact model qualification and are not enabled by `.pb/tasks.toml`.

## Goal

**Shipped.** Use **Goal** when one durable objective needs several sequential, bounded Build
workflows. Goal is a controller above Build, not another model profile. It is available beside
Discuss and Build on Home, Projects, and a finished session. The setup sheet asks for:

- the objective and completion criteria;
- whether pb asks before each milestone or continues an approved plan within limits;
- a Compact, Standard, Extended, or custom total budget; and
- the registered local repository. Goal mode never adds publishing authority.

Creating a Goal produces a linear milestone plan and stops at **Review goal plan**. You may edit the
draft, approve the exact displayed plan digest, or stop it. Every approved milestone runs the normal
strict Build workflow, including planning, plan critique, implementation, affected checks, code
review, and a managed local commit. Only one milestone and one child workflow run at a time.

The web session keeps a textual Goal banner visible while the Goal is active. On a phone the banner
becomes a compact sticky `Goal · state · progress` row; **Details** opens the same milestone,
criterion, evidence, budget, Pause/Resume, edit, accept, and stop controls. Session lists also carry
a Goal badge, so colour or activity animation is never the only mode indicator.

You can use the CLI against the running service:

```bash
pb goal start "Ship the durable migration" --workdir /path/to/project \
  --criterion "Old sessions restore" --criterion "Mobile controls remain usable"
pb goal status GOAL_ID
pb goal pause GOAL_ID
pb goal resume GOAL_ID
pb goal cancel GOAL_ID
pb goal accept GOAL_ID
```

`start` defaults to reviewing the plan and then continuing within the selected limits. Pass
`--manual` to stop between milestones or `--automatic` for explicit bounded automatic
continuation. Plan approval and goal editing are available in the web UI and HTTP API.

### Changing or stopping a Goal

Before initial approval, **Edit goal** replaces the draft and creates a new plan digest. After work
starts, **Pause and edit** records a pause request; pb finishes the current non-interruptible model
or tool operation, persists the child workflow at the next safe boundary, and only then opens the
editor. An accepted amendment creates a new plan version. Completed milestones, workflows, and
criterion evidence remain in history, while unfinished milestones are marked superseded. Budget
changes remain subject to `.pb/goal.toml` ceilings.

Inside a Goal Task, the same amendment UI remains available, but the enclosing Task contract also
applies: an amendment may add bounded criteria or change continuation, but cannot remove an accepted
criterion, change the Task objective or repository authority, or exceed the Task budget.

**Stop goal** preserves commits, uncommitted workspace content, events, and evidence. It does not
roll back repository work. A daemon restart likewise never silently resumes Goal mutation: active
work restores paused and requires an explicit Resume.

### How Goal mode ends

- Criteria marked `workflow_ready` by an API client may reach **Complete** automatically when all
  current strict-workflow evidence is Ready.
- Ordinary prose criteria stop at **Goal ready for review**. **Accept goal** records explicit user
  acceptance and ends the Goal.
- A failed milestone, exhausted total budget, context/engine failure, or cancellation records a
  distinct terminal outcome instead of claiming completion.

After a durable terminal checkpoint, the normal Discuss/Build/Goal composer returns. The completed,
stopped, or failed Goal remains visible in session history and details. Local completion still does
not push, open a pull request, merge, or publish.

## Profiles

The primary profile changes how pb approaches the request:

| Profile | Intended use |
| --- | --- |
| `build` | Deliver a concrete, scoped change. |
| `scout` | Inspect the repository and derive an appropriate development environment before delivery. |
| `explore` | Read-only investigation. |
| `plan` | Read-only implementation planning. |
| `review` | Read-only critique of a change or proposal. |
| `ask` | Read-only explanation and questions. |
| `research` | Public research with bounded repository context. |

For example:

```bash
pb queue --profile build "Add the missing test" --workdir /path/to/project
```

Advisory profiles can give the primary session fresh-context input. They cannot mutate the primary
workspace, delegate again, or advance its workflow stage.

## Inspecting local inference caches

Run `pb cache status` to see the exact local llama.cpp and FlashMoe session-cache namespaces,
format versions, byte budgets, and aggregate usage. Cleanup is deliberately two-step:

```bash
pb cache clean --backend flash-moe
pb cache clean --backend flash-moe --yes
```

The first command is a dry run. The confirmed command removes only the selected versioned session
namespace; model artifacts and the other backend's cache are outside its target.

## Reading outcomes

A delivery result is deliberately more precise than “the model stopped talking.”

- **Ready** means the enforced workflow reached its terminal success state. For a change-bearing
  build, pb owns the commit and binds it to accepted plan, review, and check evidence.
- **No change** means the request was resolved without a repository delta. It is not a hidden commit.
- **Blocked** means a required user decision, executor, check, or safety gate prevented progress.
- **Failed** means the bounded workflow could not satisfy its stage or repair contract.
- **Cancelled** means the run was explicitly stopped.

A contract-free harness final can still be useful, but it is not called externally verified. See
[Contracts with the user](../architecture/user-contracts.md) for the distinction between a model
answer, a Ready workflow, and a satisfied acceptance contract.

## What Ready does not publish

Ready is local delivery evidence. pb does not automatically push the branch, open a pull request,
merge, wait for provider CI, or respond to remote review comments. Those actions cross a separate
external authority boundary and remain follow-on work. The detailed proposal is preserved in the
[external publication record](../external-publication-workflow-follow-on.md).
