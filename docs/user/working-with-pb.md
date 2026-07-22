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

The web transcript and **Actions** drawer show model tool calls and deterministic pb actions in one
place. Every item carries an actor badge: **Model** for a model-issued tool and **pb** for controller
work. Reads show their full or bounded-range coverage; deletion states that the path was tracked and
Git-recoverable. The terminal uses the corresponding `tool` and `pb action` labels.

If a file is missing, binary, symlinked, stale, oversized, or cannot fit the bounded prompt safely,
pb does not pretend it was read. The normal model/tool path remains available. Automatic deletion
has stricter gates and applies only to a unique accepted tracked-clean path that is unchanged and
recoverable from Git.

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
