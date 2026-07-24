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

You may see pb pause for a planning question when a missing choice would materially change the
work. Answering that question updates the user-owned contract; it does not hand the model a general
permission to improvise.

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

Sessions saved before actor attribution remain readable. Their model tool actions appear as
**Legacy action** rather than being guessed from the closest chat message.

If a file is missing, binary, symlinked, stale, oversized, or cannot fit the bounded prompt safely,
pb does not pretend it was read. The normal model/tool path remains available. Automatic deletion
has stricter gates and applies only to a unique accepted tracked-clean path that is unchanged and
recoverable from Git.

## Tasks

**Shipped controller; no planner currently qualified.** For a larger explicit Build request, pb can
first create a high-level plan of outcome-shaped **Tasks** once an exact qualification is shipped.
Each Task is either **Build** or
**Goal**, has its own read-only budget display, and runs in order. This is above ordinary Build
planning: when a Build Task becomes active, its normal planner still inspects the current repository
and proposes the actual changes, checks, and commit boundary for that Task.

The full selected local model artifact must exactly match pb's embedded
model/backend/template/protocol qualification record; split GGUF qualifications bind every shard.
pb constrains both the proposed plan and its fresh review to exact JSON
schemas while each token is selected. pb splits compound request sentences at punctuation
boundaries; the model orders Tasks and selects their behavioral requirements from those verbatim
request clauses. pb automatically attaches
decomposition-wide constraints to every Task. Each Task separately states outcome acceptance, test
work, and documentation work or impact. pb retains the original objective and assigns IDs,
dependencies, qualitative effort, and budgets so summarization and bookkeeping mistakes cannot
enter the queue. A Build Task in a multi-Task queue can own at most two behavioral clauses and needs
an outcome acceptance fact and test fact for each. pb rejects a queue entry that claims only decomposition constraints, so
testing, documentation, ordering, and generic final validation cannot become catch-all Tasks.
Deterministic validation and review still decide whether the result is complete and useful. The
embedded catalog is empty in this release, and `.pb/tasks.toml`
cannot turn the feature on. With an unqualified model, pb
starts the current Build workflow directly, so existing projects do not pay an extra planning turn
or see a new UI.

The fresh critic must first assess every supplied source-request clause exactly once, identifying
the Tasks that preserve it. It must then separately audit request coverage, Task boundaries,
dependency order, observable acceptance, test/documentation ownership, and effort/Goal authority.
pb accepts a pass only when all six audits pass; a revision must identify a failed audit, select
verbatim evidence from the original request, and provide a blocking correction grounded in that
evidence.

The number of accepted Tasks determines the experience:

- One Build Task looks and behaves like the existing Build workflow.
- One Goal Task looks and behaves like the existing Goal workflow and still waits at its normal
  Goal plan approval boundary.
- Two or more Tasks show a **Tasks** panel with ordered state, Build/Goal kind, active Task, budget
  use, commits, and any blocked reason. An active Goal's existing controls appear inside that Goal
  Task.

Only one Task runs at a time. pb plans a Task against the repository left by its predecessor,
requires the predecessor's managed commit or verified no-change result, checks the exact repository
state, and only then queues the next Task. Stopping or exhausting a Task preserves completed local
commits and prevents dependent Tasks from starting. A restart restores the exact active request and
remaining allowance, paused at a safe boundary.

If high-level planning cannot produce a valid reviewed plan in two attempts, pb does not silently
run the broad request. The session shows three choices: **Retry planning**, **Edit request**, or
**Run as one Build**. The final choice is a new explicit decision and enters the unchanged Build
workflow.

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
