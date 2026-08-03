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

### Browse session history

**Shipped.** The home and project pages show the six most recent matching sessions first. Use
**Show more sessions** to reveal the next batch without making a long history dominate the page.
Changing the status filter starts that filtered history from its first batch again. No sessions are
deleted or hidden from the filter totals.

Session notes use v5 session state and v5 event envelopes. A note written with an incompatible
development schema, or missing required current state, is ignored during restore. pb does not guess
at missing teammate speech, reconstruct old transcript metadata, or derive absent status and usage
records from neighboring events.

Starting work from a project page leaves branch and managed-workspace selection to pb. The page does
not offer a synthetic list of branch names. Session rows show the branch reported by pb or identify
the managed workspace when no branch applies; the overview shows only the latest session's branch.

### Speak a prompt

**Shipped where the browser supports the Web Speech API.** Prompt composers on the home and project
pages, running-session messages, planning answers, follow-up Tasks, and Goal objectives include a
microphone button. Press it and speak normally: interim words appear live in the same controlled
field. Submission stays disabled while the microphone is active. Press stop—or let the browser end
the utterance—to turn the last visible preview into ordinary editable, submittable text. Pressing the
microphone again appends another utterance instead of replacing text already in the field.

pb creates a fresh browser recognizer for every recording and permits only one prompt control to own
the microphone. Hiding or leaving the page, closing a sheet, changing composer state, or unmounting
the control aborts both active and recently stopped browser sessions. This prevents a stale callback
or an animated panel transition from stranding the microphone and blocking the next recording. A
browser without speech recognition simply omits the button; typing remains unchanged.

Safari also requires the page itself to be a secure context before it exposes microphone capture.
`http://localhost:8311` qualifies on the Mac that runs pb, but a plain LAN URL such as
`http://mac-name.local:8311` does not qualify on another Mac or iPhone. On the host Mac, open
**Settings → Secure remote access** to let pb create a private Tailscale HTTPS address, then use the
shown address from the other tailnet device. pb checks, repairs, and removes only its own Serve
endpoint, so routine use does not require managing Tailscale commands.

The browser owns speech capture and recognition. pb requests on-device recognition when the browser
exposes that option, never records or stores audio, and receives only the live text placed in the
prompt field. Browsers without an on-device implementation may use their configured speech service,
which can send audio outside pb; using the microphone is therefore an explicit browser-managed
network and permission choice.

## Build

Use **Build** when the desired outcome is a repository change. A queue task is already explicit:

```bash
pb queue "Implement the accepted retry behavior" --workdir /path/to/project
```

Delivery moves through enforced stages: planning, independent plan review, implementation,
configured checks, independent code review, bounded repair when necessary, and a managed commit.
The active stage controls which tools are visible and which structured submission can advance the
workflow. Planning and review use current repository evidence and do not expose prior model-session
summaries as a substitute for inspecting the project. When pb has already identified an existing
path relevant to the task, planning must name an executable change instead of deferring file
discovery to implementation. For a larger matching file, pb can show the planner and plan reviewer
a freshly checked task-focused excerpt up front. They may request a specific missing line range,
but cannot spend the stage walking the whole file through generic continuation links.

The accepted delivery plan appears in the main session transcript with its requirements,
implementation steps, target paths, and acceptance facts. A reviewer verdict is additional chat
from the review teammate rather than content hidden inside the following submission action.

Small complete files read during one stage may appear in the next stage as harness-carried evidence.
pb rechecks their hashes first; changed, partial, or oversized reads must be inspected again. Strict
Build stages use the accepted plan and checkpoint for progress instead of exposing a second TODO
protocol. Independent tool calls can share one model response, while dependent same-path or
mutation/check batches are rejected before any call runs.

For a large accepted Modify target, pb can provide a small, task-relevant, fingerprinted range and
open exact-path mutation actions immediately. One canonical patch can update several separated
hunks when every old-side hunk remains inside the shown ranges; it cannot touch another path.
Replayed reads remain available in the session's
technical evidence, but the agent receives a compact replay receipt instead of another full copy of
the same file contents; repeating that exact read is then stopped before it runs again.

When a trusted contract names existing small text files as allowed paths, pb can place their exact
current contents into the first planning turn automatically. The transcript labels these as
automatic controller observations. If a file or the bounded context is unsuitable, pb simply leaves
the normal read tool available.

During planning, new paths and existing paths have different controls. Create paths remain ordinary
repository-relative names. Modify and Delete choices come from a small, per-turn list of exact
existing paths assembled from the task, trusted path evidence, and successful local discovery. The
generation collar therefore prevents small filename typos without putting the repository's entire
path list into every schema. If the intended existing file is not in that list, the planner must use
glob, search, or read first; the next turn incorporates those exact results.

During implementation, pb exposes one accepted-plan file operation at a time and inserts that
controller-owned target into the action before validation. For a large existing text file, pb can
also place a few task-relevant, fingerprinted line windows into context and allow an exact edit or
multi-hunk patch only within those windows. Identifier forms such as `branchPicker`, `branch-picker`,
and nearby `<select>` controls can contribute conservative anchors. If no confident task anchor
exists, the normal read flow remains
available. A failed path lookup can show similarly named existing files, but pb never silently
changes the requested path. Consecutive new files do not share one model output budget. If
constrained generation reaches a file-payload boundary well before its token
ceiling, pb retries once with more string room at the same token limit; real token exhaustion instead
gets one smaller complete-file retry. Neither case writes a partial file.

With FlashMoe, supported source files also have a generation-time completion gate. Rust, Python,
TypeScript/TSX, JavaScript/JSX, HTML, and CSS mutations cannot finish or execute unless the exact
virtual file is valid under pb's pinned syntax parser. Existing files still need current exact read
authority, either complete-file evidence or a fingerprinted controller range that contains the edit.
`apply_patch` uses exact hunk offsets, counts, and context and accepts only pb's canonical text
patch form; stale, recount-dependent, rename/copy, mode-changing, timestamped, quoted-path, binary,
or syntax-breaking patches fail without falling back to a looser parser. For real local backends,
Rust and Python additionally have narrow native semantic layers prepared before inference and
replayed before execution. Rust keeps project-local facts repairable while output is streaming,
then overlays every completed modification to an already indexed `.rs` file in one independently
writable rust-analyzer database. In projects without build scripts or procedural macros, it can
veto newly increased debt for its promoted name/import, field/method, privacy, call/type, mutability,
moved-from-reference, and trait-bound diagnostics. New/deleted Rust files, unsupported relative
import contexts, and partial project profiles stay open rather than being rejected. Python applies
all candidate files—including deletions—as one overlay and can veto
complete generated string-plus-integer literal additions and selected newly introduced pinned-`ty`
diagnostics across the frozen first-party project, including untouched in-project dependants of a
changed or deleted symbol. Before inference, one unambiguous safe project-local `.venv`/`venv` is
also copied into the frozen world and its dependency modules are type-primed; this can constrain
imports and callable shapes from installed Python source or stubs. Plain-path editables inside the
repository join the frozen first-party search roots. An exact external environment or editable root
can participate only through user-owned, canonical-workspace-bound configuration; repository files
cannot grant that read. pb never runs the environment's interpreter. Ambiguous, undeclared,
symlinked, native, or hook-bearing environment facts remain partial, and an unqualified missing
external import stays open. `Any`, dynamic/runtime behavior,
dependants outside the frozen project, and other diagnostics remain unknown. The Rust layer is not
rustc or a complete borrow checker, and the Python layer is not a runtime proof. Neither promises
general type correctness, compilation, passing checks, or safe runtime behavior; configured checks
and review still decide those claims.

For the last remaining file operation, the local model returns the implementation summary with the
mutation. pb validates that summary only after the edit succeeds, so a valid edit can finish the
implementation stage without a separate bookkeeping turn. Review pass records are similarly
compact: pb supplies controller-owned plan identity, each assessment records only its kind and
status, and concerns or failures put their specific reasons and fresh evidence once in challenges
or findings.

If a required check fails against the only contract-allowed path, pb keeps the accepted plan and
offers a focused read or repair rather than another identical planning cycle, including when an
assertion reports rendered output without repeating the filename. Replanning remains available for
real scope choices and repository-state blockers; the single-path rule cannot widen or change the
trusted contract.

If the failed output cites a local test or other small support file, pb can show that bounded source
to the repair model automatically. The cited file remains read-only and does not become an allowed
change; all required checks still rerun after the repair.

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
no-change closure, corrections, and handoff work. Each action also says **Model** or **Harness**, so
the friendlier character metaphor does not hide provenance. Reads show their full or bounded-range
coverage; deletion states that the path was tracked and Git-recoverable. The terminal uses the same
character-first attribution and provenance. Trinity-owned rows use one consistent lilac identity
accent while retaining the same neutral message surfaces as the rest of the team; her avatar has no
extra outline. Other teammates use the same narrow accent and tinted provenance label in a colour
drawn from their avatar background. A stars glyph and a **Predicted…** action label call out
repository reads or inspections Trinity completed ahead of the next model call. Action bubbles keep
only the primary action, optional compact summary, and disclosure control visible until expanded.

The work drawer appears only when a session has a Goal, recorded actions, or a managed plan. Actions
use a compact chronological row instead of repeating the teammate identity already established in
the transcript, and the drawer starts collapsed. Routine lifecycle activity is not repeated in a
separate activity feed. A simple request that skipped model-based Task partitioning does not add an
empty zero-call planning disclosure above the transcript.

Model timing appears immediately below the work it measures. A text-only model turn places the
elapsed-time row after its prose; a tool-directed turn places the elapsed time under its collapsed
action run. A model call that produces no visible prose or action does not create a detached timing
row. When the same teammate makes consecutive tool-directed calls, the transcript folds the actions
into one collapsed run with one avatar and name. Model prose remains visible as ordinary teammate
chat; it is never moved into the action disclosure merely because a tool submission follows. Model
prose, accepted delivery plans, Trinity feedback, and delivery summaries use the same chat-row width
and responsive card treatment, so their author and content remain readable on narrow screens.
Consecutive messages from the same teammate form one visual run: the avatar, name, and role appear at
the start and return when another speaker intervenes. A time divider appears after a five-minute gap.
On touch screens, pull the transcript to the left to reveal the exact time beside every message until
you release it. Model and Trinity prose render headings, lists, fenced code, emphasis, and inline
code; single-tilde file paths from local models are treated as inline code too. User input, stage
changes, and material Trinity feedback also start a new run. Internal turn-credit accounting stays
out of chat, and a repeat-limit failure is explained by Trinity instead of appearing again as an
unattached red error. When the repeat stop and delivery outcome are adjacent, Trinity presents one
conversational handoff while the stored evidence keeps both causes. If the repeated action produced
the same failure twice, chat keeps the first explanation and the second teammate action, then
suppresses the stale duplicate explanations around the terminal outcome. Trinity closes the pass
with two ordinary chat bubbles: she first tells the responsible teammate what went wrong and that
their task is on hold, then addresses you by your local username and asks for the follow-up context or
recovery action the team needs. There is no separate terminal-status badge inside the transcript.
On a pointer device, hover reveals a fixed-size information button that does not move or reflow the
row; click it to open aggregate and per-call metrics. Touch devices show the action-run button and
also support long-press on standalone inference rows. The detail sheet presents purpose, token
counts, energy, prompt-cache work, and native-runtime timings as labelled fields. The session's
runtime total uses the same interaction instead of an expandable block in the transcript. When
Trinity completed eligible repository actions early, the summary counts them and reports up to that
many potentially avoided model turns. This is deliberately an upper bound; the UI does not claim to
know the counterfactual model trajectory, and the displayed energy remains measured session use
rather than energy saved. The detail sheet also calls out duplicate or dependent actions prevented
before execution, no-progress loops stopped, and invalid generation candidates filtered by the
control collar. Candidate counts are not presented as blocked tool calls. A
zero-reuse call retains the backend's reason—such as a cold session, changed
prompt, unavailable stable prefix, unreadable cache, required context reset, disabled cache, or an
unsupported runtime path. When a stable root is available, the stored diagnostic retains its reused
token count and bounded workflow authority class. These local diagnostics use digests and token
counts rather than prompt or repository text.

Harness validation feedback appears as a message from Trinity. The message identifies which
teammate's artifact needs another pass and explains the problem in ordinary language; raw validation
data stays out of the normal transcript. On a pointer device, hover reveals its information button;
on a touch device, long-press the feedback to open **Technical details**. Tool failures name the
failed action, summarize the actionable cause without exposing the temporary workspace path, and
tell the teammate to correct the action, choose another action, or report the blocker. Internal
closure checkpoints do not appear in chat. Trinity's visible sentence is derived from the event that
caused it: proactively supplied repository evidence is described as code Trinity already found and
inspected, while invalid artifacts, failed tools, repeated actions, diagnostics, and terminal pauses
each explain their own concrete cause. The visible wording describes ordinary teammate actions such
as reading the relevant lines; it does not expose prompt context or tell a teammate to ask the
harness for code. A message that asks for action addresses the responsible teammate in second
person, so both its owner and requested next step remain clear. A terminal session never leaves an
accepted plan labelled **Awaiting review**: it reports an incomplete, cancelled, invalidated, or
completed review state as appropriate.
Delivery summaries list only commits and changes made after the session's captured repository
baseline, so earlier repository history is not presented as work from the current delivery. Older
strict-workflow sessions show their stored commit list only when they also contain a successful commit
receipt.

Planning is fail-closed. The constrained plan tool requires a non-empty summary, requirements,
steps, acceptance facts, and the required requirement references. pb then checks repository-aware
facts—such as coverage and whether a path exists before a proposed modification—before accepting the
plan. These controls prevent an invalid plan from advancing; they cannot guarantee that a model will
produce a valid plan within its retry budget. Delivery planning uses the original task as its stable
title fallback instead of spending a model call on cosmetic title rewriting. A structurally
inconsistent plan-review verdict is rejected inside the same live review turn, preserving the
reviewer's reads and restricting the correction turn to the review submission rather than starting
another evidence pass.
If a plan reviewer reads a path that does not exist, pb marks that exact read as non-retryable and
will not run it again unchanged. The next review turn is deliberately small: the reviewer can submit
a revision challenge from the evidence already collected, or use one repository search to find the
actual symbol or path before continuing. This recovery still consumes the session's normal bounded
review budget.

If repository content changes while a read-only plan or code review is running, pb keeps the review
bound to its earlier exact snapshot and stops before implementation or commit. The session names the
changed paths and offers **Restart with current files**. That action preserves the earlier plan and
review in history, captures the current repository as a new baseline, and starts fresh planning;
generic **Resume** is reserved for a preserved stage whose prerequisite can actually be repaired in
place, such as an unavailable executor.

Apple-container-backed work uses a pb-owned session worktree. Local execution is intentionally a
compatibility mode and works in the registered repository itself, so another editor or session can
change those files while a task is running. Content fingerprints prevent stale review or commit
evidence from being accepted, but they are conflict detection rather than filesystem isolation.

Persisted sessions and their events must use the current v5 schemas. Incompatible development-era
notes are skipped during restoration instead of being migrated or shown with guessed attribution.
The current schema requires every session-state key explicitly, using `null` rather than omission
for inactive state and rejecting unknown compatibility fields. It stores the session start time,
Trinity's complete authored chatter (including the username-addressed request to the local user),
structured commit summaries, typed check/commit evidence,
registered-project identity, pending proposals, and Goal
change requests directly so the terminal and browser show the same team conversation and actions
without reconstructing them from nearby events. Session snapshots and live events share a monotonic
revision, preventing a slower snapshot response from reverting a newer title or running state.
The browser addresses registered projects by durable ID rather than copying their names or filesystem
paths into session, Goal, usage, notification, or integration requests. The service resolves that ID
and returns structured failures for invalid or stale mutations. Successful controls for an existing
session return and apply the same revisioned snapshot used by the session stream; the stream then keeps
other clients synchronized. The browser therefore does not guess lifecycle state or wait for SSE before
clearing a completed control. A requested cancellation is explicit while work winds down, and the
runner's resolved branch and focus root replace requested workspace values as soon as its `started`
event is published. Live effects at or below
the accepted snapshot revision are replay,
while the SSE service sends revisioned session snapshots after state-changing events. When a browser
reconnect cursor is no longer retained, the service marks the snapshot as a history reset rather
than joining non-contiguous transcript windows. Project pages consume a server-sent registry,
session, usage-summary, and terminal-transition snapshot instead of polling separate endpoints. The
browser supplies its local calendar-day bounds, and pb returns total and today summaries for all
sessions and for every registered project; the browser does not download per-turn usage records to
recalculate those cards. Each service process gives that stream a new identity and monotonic
revision, and only the live stream may
change the browser's accepted process identity, so a delayed HTTP response cannot restore an older
process after restart. The service records a subscription-specific terminal-transition floor under
the same publication lock as revisions and retained transitions. The first snapshot includes only
retained transitions above that floor, and each subsequent event advances it before producing the
next delta. Projects added or removed through
the CLI therefore appear without a browser reload, a session that finishes while the first snapshot
is being built still produces one finish notification, and project/session identity and usage cannot
drift between separate requests.

The browser validates the complete v5 session envelope, event variant fields, authored chatter,
evidence, and transcript metadata before applying an update. A replaced EventSource connection no
longer owns callbacks, so a queued message from the previous calendar-day usage stream cannot roll
the page back after midnight. A registered project with no session is shown as having “No sessions”;
it is not presented as if queued work exists.

Deleting a session continues its authoritative durable removal, usage update, and live publication
even if the browser or terminal disconnects while the request is in flight. A durable deletion error
leaves the session visible. Environment or managed-workspace cleanup failures after deletion are
reported as warnings on the successful result.

The service records an event before making it live. Terminal reattachment atomically captures
history and subscribes, recovers a lagged receiver from sequence-numbered history, and drains final
events before reporting that the session finished. Retention may exceed its nominal event count to
preserve the prior records required to validate server-authored chatter, evidence, supersession,
and transcript projections on strict restore.

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
