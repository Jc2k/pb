# Agent handoff and workspace execution plan

Status: proposed

## Outcome

Move the reliable control-plane behavior learned through `pb harness agent` into the shared agent
runtime without turning the web interface into a compliance dashboard. A model final becomes a
request for the team to hand work back, not proof that the task was achieved. A deterministic
handoff teammate identifies the affected parts of the repository, reuses current check evidence,
runs missing checks, prevents a commit after a failed check, creates one safe repository commit when
needed, and reports that work in the group chat.

The same flow must work for:

- a single-package repository;
- a monorepo containing independently testable services in different languages;
- a generated web asset bundle that must be built before the application that serves it;
- one Cargo workspace with many packages, including `default-members` and shared crates;
- multiple independent Cargo workspaces below one Git root; and
- a task whose final repository content is unchanged and therefore has nothing to test or commit.

The deterministic runtime can establish that named checks passed for specific inputs, the intended
paths were committed, and the session-owned workspace delta is clean. It cannot prove that an
underspecified product request was achieved. Explicit task contracts remain the strongest source of
task-specific acceptance facts, while ordinary web sessions receive a useful project-level handoff
policy.

## Product language

The web experience remains a group chat with the existing team. Internal terms such as contract,
gate, fingerprint, and verification do not appear in normal handoff copy.

Representative messages are:

- `I’m checking the affected parts before we wrap this up: the web bundle and Rust tests.`
- `The API tests passed, but the web bundle failed. I’ve sent that back to Kate for another pass.`
- `Kate changed the Rust service after its tests passed, so I’m rerunning those checks.`
- `Everything affected passed. Kate committed the changes as 3f42c81.`
- `There’s no repository change to hand off, so I don’t have anything to test or commit.`
- `I couldn’t run the payments checks because its Node environment is unavailable. Ramon may need
  to set that up before we can finish.`

The deterministic handoff participant has a stable internal actor ID and is not an `AgentProfile`:
models cannot delegate to it and it is never represented as a model invocation. Its user-facing
name and avatar can be selected with the rest of the cast without changing persisted event data.

## Non-goals

- Do not infer task-specific acceptance requirements from arbitrary prompt prose.
- Do not claim that passing project checks proves product or visual quality.
- Do not replace Cargo, npm, Deno, CI, task runners, or project-authored orchestration.
- Do not require every repository to declare a complete component graph before pb remains useful.
- Do not create one commit per component by default; handoff is repository-scoped.
- Do not make arbitrary shell execution a trusted substitute for a named check.
- Do not auto-commit after a step limit, parse loop, engine failure, or other exit without an
  accepted final/handoff intent.

## Invariants

- Harness, queue, and web sessions continue to call the same core agent loop.
- Explicit repository configuration and documented canonical commands override inferred commands.
- A later change to a check's declared inputs makes its prior evidence stale.
- An unrelated component change does not invalidate narrowly scoped check evidence.
- Unknown or shared paths select a conservative broader check set rather than no checks.
- No final session-owned content delta means change-conditioned checks, review, and commit stay out
  of the way; explicitly `always` checks may still run.
- A failed required check prevents the managed commit path.
- Pre-existing user changes are never staged or committed by managed handoff.
- A contract-free `pb harness agent` run may retain its compatibility exit of zero after a
  successful handoff or valid no-change outcome, but it must not claim `verified_completed`
  without an explicit satisfied contract.
- Harness resume preserves one task baseline across runs and records a fresh invocation baseline
  for each run; prior scratch changes remain task-owned rather than becoming invisible
  pre-existing changes.
- Successful check evidence is reusable only while its component, command, inputs, dependency
  inputs, and declared outputs still match.
- `pb harness infer`, `pb harness bench`, and `pb harness cache-clean` retain their current behavior
  and data formats.
- Raw structured evidence is preserved even when the web UI renders conversational summaries.
- Stored requests, sessions, and events remain readable through additive/defaulted fields.
- Existing flat `guard_commands` remain supported as a repository-wide compatibility fallback.

## Target model

### Repository and focus

Keep the Git/security boundary separate from the part of the repository selected in the UI:

```rust
struct RepositoryContext {
    repo_root: PathBuf,
    focus_root: PathBuf,
    task_baseline: WorkspaceBaseline,
    invocation_baseline: WorkspaceBaseline,
}

struct WorkspaceBaseline {
    head: String,
    status: WorkspaceStatus,
    content: ContentSnapshot,
}
```

`repo_root` owns branch selection, path containment, diffs, commits, and session persistence.
`focus_root` provides the initial task context, default component selection, and UI association. A
session started for `services/payments` may inspect shared code elsewhere, but it must not be
silently reclassified as an undifferentiated repository-wide session after `find_git_root` runs.

Capture the task baseline once when a logical task is created. It defines every change owned by
that task, including changes produced during earlier runs of a resumed harness scratch workspace.
Capture the invocation baseline immediately after branch selection at the beginning of each run.
It supports per-run auditing without redefining ownership. For a new web task, both baselines
initially describe the same tree. For a resumed harness task, restore the task baseline from
scratch metadata and capture the invocation baseline from the scratch tree as it exists at resume
time.

Both baselines precede environment session commands or model tools for their respective scope.
Runtime-owned mutations are then observable rather than silently treated as pre-existing work. If
a user or supervisor explicitly adopts changes after task creation, record that provenance rather
than silently folding the changes into a new task baseline.

### Workspace graph

Introduce a typed graph, separate from tool authorization policy:

```rust
struct WorkspaceGraph {
    components: BTreeMap<ComponentId, Component>,
    executors: BTreeMap<ExecutorId, Executor>,
    checks: BTreeMap<CheckId, Check>,
}

struct Component {
    id: ComponentId,
    root: PathBuf,
    include: Vec<PathPattern>,
    exclude: Vec<PathPattern>,
    executor: ExecutorId,
    depends_on: Vec<ComponentId>,
}

struct Check {
    id: CheckId,
    label: String,
    command: String,
    cwd: PathBuf,
    executor: ExecutorId,
    trigger: CheckTrigger,
    inputs: Vec<PathPattern>,
    outputs: Vec<PathPattern>,
    depends_on: Vec<CheckId>,
    timeout_seconds: u64,
}

enum CheckTrigger {
    Changed,
    Always,
    Needed,
}
```

- `Changed` runs when a declared input or an affected dependency changed.
- `Always` is opt-in for checks required even when there is no final content delta.
- `Needed` is satisfied only when its input fingerprint and declared output evidence are current;
  this is suitable for generated bundles consumed by later checks.

Use a new `.pb/workspace.toml` for explicit components, executors, checks, and dependency edges.
Keep `.pb/environment.toml` responsible for how execution environments are prepared. Extend
`pb init` and scout discovery whenever the workspace schema gains fields, as required by the
repository conventions.

### Polyglot executors

Replace the single `CommandBackend` assumption with an executor registry. Start executors lazily
for the selected check plan:

- a composite devcontainer can remain the default executor for all components;
- otherwise Rust, Node/Deno, Python, Go, or service-specific components may name different local
  or container executors;
- every executor sees the same repository mount so declared generated outputs can cross executor
  boundaries; and
- setup/session commands belong to an executor rather than being eagerly flattened across the
  whole repository.

The initial implementation may execute a stable topological order serially. Parallel independent
checks can follow once event ordering, resource limits, and cancellation are deterministic.

### Affected components and checks

At handoff, calculate paths whose final content differs from the task baseline, including committed
changes from any resumed invocation. Use the invocation baseline only to attribute actions to the
current run. Do not use a historical `wrote_file` boolean.

1. Map each changed path to its directly owning components.
2. Add affected reverse dependencies from the workspace graph.
3. Expand the required check dependency DAG.
4. Use conservative repository/workspace checks for unowned shared paths.
5. Remove checks whose current evidence already matches their scoped inputs and dependency outputs.
6. Execute the remaining plan in a stable topological order.

Root manifests, lockfiles, shared toolchain configuration, workspace dependency declarations, and
unknown paths should fan out conservatively. Documentation-only paths may select a documentation
component with no code checks when explicitly configured.

### Check evidence

Generalize the current named harness check evidence into a shared ledger:

```rust
struct CheckEvidence {
    check_id: CheckId,
    input_fingerprint: String,
    dependency_outputs: BTreeMap<CheckId, String>,
    output_fingerprint: Option<String>,
    exit_status: i32,
    duration_ms: u64,
    executor: ExecutorId,
    source: EvidenceSource,
}

enum EvidenceSource {
    AgentTool,
    ExactGuardCommand,
    CommitTool,
    Handoff,
}
```

The model-facing `run_check(id)` tool, the managed `git_commit` tool, and the deterministic handoff
engine use the same executor and ledger. An exact `run_command` match for a known canonical check
may be routed through this executor and recorded; unknown commands never receive check credit.

Evidence is persisted in the session event stream and may be reused by a resumed/continued session
only when all scoped input and dependency-output fingerprints still match.

### Generated outputs

A producing check records the fingerprint and existence of declared outputs. A consuming check
includes that evidence in its dependency key. For example:

```text
web source + deno config
    -> web-bundle
    -> webui/dist output evidence
    -> rust-build
```

If the web inputs and bundle output are current, the bundle is not rebuilt. If the bundle is
missing or stale, it runs before the Rust build even when the current model edit touched only Rust.

### Cargo workspaces

Use `cargo metadata --no-deps --format-version 1` instead of treating every recursively discovered
`Cargo.toml` as an independent project. Record:

- the Cargo workspace root;
- `workspace_members` and `workspace_default_members`;
- each package name, manifest path, root, dependencies, and targets; and
- enough package dependency information to calculate an initial reverse-dependency closure.

Rules:

- One Cargo workspace is a component group; its packages are components for impact analysis.
- Multiple unrelated Cargo workspace roots under one Git repository become separate groups.
- A narrow affected set may produce one combined command such as
  `cargo test -p shared-auth -p api -p worker --all-targets`.
- Workspace manifests, `Cargo.lock`, shared profiles/configuration, ambiguous ownership, or broad
  project instructions select `cargo test --workspace` or the documented canonical workspace
  command.
- Never assume plain `cargo test` means all workspace packages; represent default-member and
  whole-workspace intent explicitly.
- Project-authored commands such as this repository's `cargo test --all-targets` override inferred
  package commands when they are declared as canonical.

## Deterministic handoff flow

An accepted primary-model final enters the handoff engine before the session receives its terminal
outcome:

1. Recompute the final session-owned delta from the task baseline, retaining per-run attribution
   from the invocation baseline.
2. If it is empty and no `Always` check applies, emit a no-change team message and finish without a
   review, check, or commit.
3. Resolve affected components and the required check DAG.
4. Emit one conversational message describing what the handoff teammate is checking.
5. Reuse current evidence and execute only missing or stale checks.
6. On failure, emit raw `CheckResult` evidence plus a conversational failure message. If the model
   has steps remaining, add the structured failure to Kate's context and allow another bounded
   implementation turn.
7. If the same check fails with the same input fingerprint repeatedly, stop the bounce
   deterministically and leave the session needing another pass.
8. Once every required check is current, use the managed safe-commit path if a session-owned delta
   remains uncommitted.
9. Emit the commit evidence and a final handoff teammate message.

The handoff engine is not entered after a parse loop, engine/resource failure, ordinary step limit,
or an unaccepted final. Existing final-only grace may still obtain the explicit handoff intent after
all work evidence is ready.

Explicit harness contracts compile into the same completion policy. Contract-only facts such as
allowed paths or required review reads remain enforced, but missing executable checks and a missing
managed commit can be satisfied by the handoff engine instead of merely rejecting the first final.

## Safe commit ownership

Do not use the current `git add -A` behavior for automatic web handoff.

Track separately:

- paths dirty before task creation;
- paths whose content changed during the task, with invocation-level provenance;
- the task-start, invocation-start, and current index;
- commits created since the task baseline; and
- paths already included in those commits.

The managed commit path must:

1. refuse any session change that overlaps a path dirty at task creation unless explicit ownership
   was granted and recorded;
2. refuse unexpected staged paths not owned by the session;
3. stage only session-owned paths;
4. preserve unrelated staged, unstaged, and untracked user work;
5. validate the semantic commit message;
6. create no commit when the final content delta is empty; and
7. define cleanliness as no remaining uncommitted **session-owned** delta, while an explicit
   contract may still require a globally clean repository.

Unchanged project-local `.pb/` state and other pre-existing untracked files must not enter the
commit. The isolated harness scratch remains the simplest safe case. If path-scoped index handling
cannot meet these invariants reliably, move build sessions to isolated worktrees before enabling
automatic web commits.

## Events and web rendering

Preserve raw check evidence and add conversational and terminal handoff events:

```rust
enum TeamActor {
    Agent(AgentProfile),
    Automation(AutomationActor),
}

enum AutomationActor {
    Handoff,
}

struct TeamMessage {
    actor: TeamActor,
    tone: TeamMessageTone,
    message: String,
    evidence_ids: Vec<String>,
}

struct HandoffSummary {
    outcome: HandoffOutcome,
    affected_components: Vec<ComponentId>,
    checks: Vec<CheckSummary>,
    commit: Option<CommitSummary>,
    changed_paths: Vec<String>,
}
```

Keep internal outcome detail machine-readable while rendering conversational labels:

- ready: `The team wrapped this up`;
- no change: `The team left the code untouched`;
- check failure: `This needs another pass`;
- missing executor/user action: `The team needs help to continue`;
- non-final termination: `The task stopped before handoff`.

Render `TeamMessage` in the ordinary chat stream with the handoff teammate's avatar. Combine
parallel/related successes into one message, show a pending handoff as one updateable chat item, and
put commands, outputs, fingerprints, component selection, and dependency details behind `What I
ran`. Browser notifications should use the same outcome vocabulary rather than treating every model
final as completed.

The web TypeScript event union must add all existing structured completion/check events as well as
the new team messages and summaries. Persisted sessions must restore the same conversation and
handoff state after daemon restart.

## Configuration and discovery precedence

Use this order when constructing the workspace graph and canonical checks:

1. explicit `.pb/workspace.toml` declarations;
2. exact commands and ordering requirements in AGENT.md/AGENTS.md and other supported project docs;
3. CI/task-runner metadata and root scripts;
4. ecosystem-native discovery such as Cargo metadata and JavaScript workspace manifests; and
5. conservative language defaults.

Discovery writes or proposes explicit normalized configuration through `pb init`; it should not
silently replace user-authored configuration. If discovery is incomplete, retain a repository-wide
legacy guard rather than inventing narrow safety.

## Harness subcommand behavior

The shared handoff implementation changes `pb harness agent` and `pb harness eval`. It does not
change the inference or FlashMoe diagnostic surfaces: `pb harness infer`, `pb harness bench`, and
`pb harness cache-clean` keep their current arguments, execution paths, output formats, and exit
semantics.

### Trusted workspace input

Add an optional hidden `--workspace-config <PATH>` argument to `pb harness agent` and the equivalent
optional field to evaluation fixtures. Parse, validate, and normalize this trusted configuration
before model loading, then pass the normalized `WorkspaceGraph` directly in `AgentRequest`. Record
the source path and content hash in run metadata, but do not copy the file into the scratch
workspace or count it as a model-produced mutation.

Keep workspace topology separate from the acceptance contract:

- workspace configuration says what components, executors, checks, inputs, outputs, and dependency
  edges exist;
- the contract says what task-specific facts must be true before the run can be externally
  verified; and
- the absence of workspace configuration retains the current single local executor and synthetic
  repository-wide component.

Normalize contract schema v1 into that synthetic component, its local executor, and a whole-content
fingerprint. This preserves v1 allowed-path, required-mutation, check, review, commit, and clean-tree
semantics. Workspace-aware acceptance fields can be introduced later as contract v2; they are not a
prerequisite for executor/topology support.

Reject an invalid graph, missing executor kind, cycle, escaping path, or unsupported runtime before
model loading with structured setup failure. The model tool allowlist continues to govern model
tool calls only. Handoff checks are trusted runtime operations, not injected model tools, so they do
not require `run_check` to be in the allowlist and do not consume model tool-step allowance.
Model-based repair turns do consume the normal step/token budget.

### Scratch creation and resume

Persist task-baseline metadata when the scratch workspace is created. On resume:

1. restore the original task baseline and prior ownership/adoption records;
2. capture a new invocation baseline before environment preparation or model execution;
3. classify every delta from the task baseline as task-owned unless an ownership record says it was
   adopted or pre-existing;
4. load prior successful check evidence from immutable run events; and
5. reuse that evidence only when the current fingerprints, command, executor, dependencies, and
   declared outputs still match.

This prevents a partially completed resumed run from treating its own earlier work as unrelated
dirty state. The invocation baseline still makes the current run's actions auditable. Existing
`--resume-scratch` behavior may adopt externally edited scratch content, but the adoption must be an
explicit provenance record and must never rewrite older run journals.

### Agent exit semantics

`verified_completed` remains an acceptance-contract result, not a synonym for handoff readiness.
Add a separate structured `handoff_outcome` and extend termination reasons where needed so callers
can distinguish workflow failure from parse, engine, or resource failure.

| Situation | Process exit | `verified_completed` | Structured outcome |
| --- | ---: | ---: | --- |
| No contract; checks pass and handoff succeeds | 0 | false | `ready` |
| No contract; no final content change | 0 | false | `no_change` |
| Explicit contract satisfied and handoff succeeds | 0 | true | `ready` |
| Contract requires mutation but final content is unchanged | nonzero | false | `contract_unsatisfied` |
| A required check still fails after bounded repair | nonzero | false | `checks_failed` |
| A required executor is missing or cannot start | nonzero | false | `executor_unavailable` |
| A safe required commit cannot be created | nonzero | false | `commit_blocked` |
| Repair allowance is exhausted with stale/failed requirements | nonzero | false | `repair_exhausted` |
| Parse loop, step/resource limit, model engine error, or setup error | current nonzero behavior | false | existing termination reason |

A final with no contract therefore remains CLI-compatible while becoming truthful in persisted
data. No-change is successful only when mutation is not required and there are no applicable
`always` checks left to run.

### Journals and run index

Extend each immutable harness run journal and its run-index entry with additive/defaulted fields:

- handoff outcome and termination reason;
- task-baseline and invocation-baseline identifiers;
- workspace-configuration source/hash and affected components;
- checks planned, reused, executed, passed, failed, and skipped, including evidence IDs and output
  artifact fingerprints;
- executor preparation/start outcomes;
- bounded-repair turns and the failure evidence returned to the model;
- conversational `TeamMessage` events from guard rails and handoff;
- no-change classification; and
- commit requested/reused/created/blocked, commit hash, and ownership/safety result.

Continue storing raw check/process output in the event stream rather than duplicating large output
in the run index. Preserve older journals through serde defaults or an explicit additive schema
reader, and never rewrite previous run files during resume. Suppress the current `completed run
produced no commits` observation when the structured outcome is a valid `no_change`.

Guard-rail feedback uses the same `TeamMessage` event as the web UI, with a stable runtime actor and
machine-readable evidence references. The CLI may render it as ordinary progress text, while the
journal preserves enough structure for the group-chat UI and evaluation tooling.

### Evaluation schema and accounting

Bump the checked fixture corpus and JSONL result schema when these fields land rather than silently
changing the meaning of existing metrics. The evaluator must count deterministic handoff
`CheckResult` events as executions; it must not infer `executed_checks` solely from model
`ToolCall(run_check)` events.

Add per-fixture expectations and aggregate metrics for:

- affected components and selected check IDs;
- checks planned, executed, reused, failed, and prevented by a dependency failure;
- executor starts and avoided executor starts;
- handoff and guard-rail team messages;
- repair-turn count and repeated-failure cutoff;
- no-change outcome and contract-free compatibility exit;
- commit request, reuse, creation, safety refusal, and resulting hash; and
- output-artifact evidence and stale-output rebuilds.

Evaluation fixtures may name a workspace graph/config plus an optional contract. Real-model run
configuration records the workspace-config hash and executor policy so results are comparable.
Update scripted completion counts, protocol comparisons, checked baselines, and stable JSONL field
ordering together. Deterministic fixtures must pass before real-model harness experiments are used
to judge the new control plane.

## Workstreams

### W0 — Deterministic baseline fixtures (P0)

Extend the scripted completion engine and harness evaluation corpus before behavior changes.

Cover:

- web currently labels an uncontracted final as completed;
- ordinary build completion does not require tests or a commit;
- an unsatisfied contract rejects rather than executes a missing check;
- a model-run named check is current for the same content;
- a post-check mutation makes evidence stale;
- a build final with no mutation cannot currently finish; and
- current `git add -A` would include unrelated dirty paths.

Likely files: `src/agent_core.rs`, `src/harness_eval.rs`, `fixtures/`, `webui/src/**/*.test.ts`.

Acceptance:

- fixtures run without a model or container runtime;
- each current limitation has one explicit expected outcome; and
- artifact quality remains separate from control-plane pass/fail.

### W1 — Repository/focus and workspace schema (P0)

Add `RepositoryContext`, preserve `focus_root` through web/session persistence, and introduce the
additive workspace graph/config types. Load legacy `guard_commands` into a synthetic repository
component/check.

Likely files: `src/agent_core.rs`, `src/web.rs`, `src/session_store.rs`, `src/projects.rs`,
`src/environment.rs`, new `src/workspace.rs`, and `src/init.rs`.

Acceptance:

- registering or selecting a nested service preserves both its focus and Git root;
- existing project/session records deserialize;
- existing environment files need no edits; and
- a flat guard configuration produces the same command behavior as before.

### W2 — Workspace discovery and Cargo topology (P1)

Implement explicit-config precedence, component ownership, Cargo metadata discovery, multiple Cargo
workspace roots, package dependency edges, and conservative shared-path handling. Detect root and
nested JavaScript/Deno/Python/Go services without treating setup commands as a flat repo-wide list.

Likely files: `src/init.rs`, `src/environment.rs`, `src/workspace.rs`, and focused discovery fixture
directories.

Acceptance:

- a virtual Cargo workspace is discovered once rather than once per member manifest;
- `default-members` and whole-workspace scope remain distinct;
- two independent Cargo workspaces are represented separately;
- a shared crate change selects its configured/dependent consumers;
- a service-local change does not select an unrelated service; and
- root lockfile/workspace changes select the conservative workspace command.

### W3 — Executor registry, check DAG, and evidence ledger (P1)

Generalize named check execution to project checks, add lazy per-component executors, scoped input
fingerprints, output evidence, stable topological execution, timeout/cancellation, and persisted
evidence events. Route exact canonical `run_command` matches through the named executor.

Likely files: `src/agent_core.rs`, `src/environment.rs`, `src/events.rs`, `src/container.rs`, and new
`src/handoff.rs` or `src/checks.rs`.

Acceptance:

- an already-current agent-run check is not repeated at handoff;
- unrelated changes do not stale narrow evidence;
- a relevant later mutation does stale it;
- a missing/stale generated output reruns its producer;
- a consuming check never runs before its dependencies; and
- only affected executors are prepared.

### W4 — Handoff teammate and bounded repair loop (P1)

Intercept accepted finals, resolve and execute the check plan, emit raw plus conversational events,
return actionable failures to Kate while budget remains, and stop repeated identical failures.
Implement the no-change path here.

Likely files: `src/agent_core.rs`, `src/events.rs`, `src/harness.rs`, `src/harness_eval.rs`, and
`src/cli_ui.rs`.

Acceptance:

- a missing required check is executed rather than rejected as merely missing;
- a failed check prevents handoff and returns bounded evidence to the build agent;
- a successful repair causes only stale checks to rerun;
- an identical repeated failure stops at a fixed threshold;
- no final delta runs no change-conditioned checks and requests no commit; and
- step-limit/engine exits never auto-finalize work.

### W5 — Safe managed commit (P1)

Replace automatic `git add -A` with session-owned staging, current-check enforcement, commit reuse,
and commit evidence. Preserve the existing model `git_commit` tool but route it through the same
handoff/check policy.

Likely files: `src/agent_core.rs`, `src/handoff.rs`, `src/events.rs`, and git fixture helpers.

Acceptance:

- checks must be current before the managed commit path succeeds;
- an agent-created applicable commit is not duplicated;
- no-change creates no empty commit;
- failed checks create no managed commit;
- pre-existing staged, unstaged, and untracked files remain byte-for-byte and index-for-index
  unchanged; and
- overlapping dirty paths cause a clear team message and no unsafe commit.

### W6 — Web group-chat handoff (P1)

Add the deterministic team actor, chat rendering, expandable evidence, pending-to-final handoff
updates, outcome-aware cards, notifications, component/focus display, and restored-session support.

Likely files: `webui/src/types/index.ts`, `webui/src/components/Session.tsx`,
`webui/src/pages/SessionPage.tsx`, `webui/src/components/SessionDashboard.tsx`,
`webui/src/lib/helpers.ts`, `webui/src/lib/sessionUtils.ts`, and their tests.

Acceptance:

- no normal UI copy says `contract verified`;
- check feedback appears as a message from the handoff teammate;
- failures and suggested ownership are visible without opening raw logs;
- successful check noise is grouped;
- `What I ran` exposes exact evidence;
- no-change, ready, needs-work, and incomplete outcomes have distinct natural-language rendering;
  and
- restored sessions reproduce the same handoff conversation.

### W7a — Harness agent CLI and scratch lifecycle (P1)

Compile harness contracts into the shared policy, add trusted workspace-topology input, implement
the dual task/invocation baseline, and make handoff outcomes part of `pb harness agent` exit
handling. Keep handoff checks outside the model tool allowlist while charging model repair turns to
the existing budget.

Likely files: `src/lib.rs`, `src/harness_contract.rs`, `src/harness.rs`, `src/agent_core.rs`,
`src/handoff.rs`, `docs/harness.md`, and the contract examples.

Acceptance:

- explicit v1 contracts retain allowed-path, mutation, review, commit, and cleanliness semantics;
- an optional workspace config is parsed outside scratch and passed as trusted runtime input;
- executable missing facts can be satisfied by handoff;
- resume restores the original task baseline and records a new invocation baseline;
- prior scratch changes remain eligible for checks and a safe task-owned commit;
- contract-free ready/no-change runs exit zero without setting `verified_completed`;
- required-mutation no-change, check failure, unavailable executor, repair exhaustion, and unsafe
  commit each produce their documented structured nonzero result; and
- `infer`, `bench`, and `cache-clean` have CLI parsing and behavior regression coverage.

### W7b — Harness journals and run-index migration (P1)

Persist the handoff lifecycle, evidence, team messages, executor starts, no-change classification,
and commit result in immutable per-run artifacts. Add defaults/version handling for existing
journals and make resume append a new run rather than rewriting old evidence.

Likely files: `src/harness.rs`, `src/events.rs`, journal/run-index fixtures, and `docs/harness.md`.

Acceptance:

- run index and journal expose every field listed under `Journals and run index`;
- raw output remains in events and summaries reference it by evidence ID;
- older persisted runs remain readable;
- resumed runs reuse only fingerprint-current evidence and leave prior run files unchanged;
- guard-rail feedback is persisted as the same stable `TeamMessage` shape used by web; and
- valid no-change runs do not emit the misleading no-commit observation.

### W7c — Harness evaluation schema and rollout (P2)

Bump the fixture/result schema, extend deterministic metrics for the full handoff lifecycle, update
checked baselines and protocol comparisons, document migration, and run real-model experiments only
after deterministic fixtures pass.

Likely files: `src/harness_eval.rs`, `fixtures/`, checked baseline artifacts, `docs/harness.md`, and
this plan.

Acceptance:

- deterministic handoff checks are counted from check events rather than only model tool calls;
- evaluation reports affected-check selection, duplicate-check avoidance, executor starts, repair
  loops, team feedback, no-change exits, output evidence, and commit safety;
- fixture expectations cover contract-free, contracted, resumed, and multi-executor runs;
- real-model result configuration includes the normalized workspace-config hash and executor
  policy; and
- old and new schemas are either explicitly migrated or rejected with a clear version error rather
  than being compared as if equivalent.

## Milestone order

| ID | Priority | Depends on | Deliverable |
| --- | --- | --- | --- |
| W0 | P0 | — | Deterministic current-behavior fixtures |
| W1 | P0 | W0 | Repository/focus split and additive workspace schema |
| W2 | P1 | W1 | Workspace and Cargo discovery |
| W3 | P1 | W1, W2 | Executor/check DAG and shared evidence |
| W4 | P1 | W3 | Deterministic handoff teammate and no-change flow |
| W5 | P1 | W3, W4 | Safe check-aware managed commit |
| W6 | P1 | W1, W4, W5 | Web group-chat handoff experience |
| W7a | P1 | W1, W3, W4, W5 | Harness CLI, contracts, dual baselines, and exit semantics |
| W7b | P1 | W7a | Immutable harness journals and run-index migration |
| W7c | P2 | W0, W2-W5, W7a, W7b | Evaluation schema, deterministic corpus, migration, and experiments |

W2 may land incrementally by ecosystem, but Cargo workspace topology and explicit config must be
available before affected-check selection becomes the default. W6 can begin its event/rendering
scaffolding after W1, but should not replace current completion labels until W4/W5 provide truthful
terminal outcomes. W7a can proceed alongside W6 once the shared handoff and commit policies exist;
W7c waits for deterministic harness lifecycle data, not for web presentation work.

## Required deterministic scenarios

The completed implementation needs regression coverage for at least:

1. single Rust package, changed and unchanged;
2. web-only change selecting web tests but not an unrelated service;
3. Rust build depending on a current web bundle;
4. missing web bundle causing the producer to run before Rust;
5. two independent microservices in different executors;
6. one Cargo workspace with multiple members and `default-members`;
7. a shared Cargo package affecting multiple dependants;
8. two unrelated Cargo workspaces under one Git root;
9. root `Cargo.lock` or workspace manifest selecting broad checks;
10. model runs the exact named check before final and handoff does not duplicate it;
11. mutation after success reruns only stale checks;
12. failed check returns to Kate, then passes after a repair;
13. identical repeated failure stops deterministically;
14. final edit is reverted and therefore produces the no-change path;
15. no-change skips review/check/commit unless a check is explicitly `Always`;
16. agent already committed applicable content and no duplicate commit is created;
17. pre-existing user-owned `.pb/` and dirty paths never enter the managed commit;
18. nested UI project selection preserves its focus after Git-root resolution;
19. handoff messages survive session persistence and daemon restart; and
20. old events/configuration continue to deserialize and render conservatively;
21. contract-free `pb harness agent` ready and no-change outcomes exit zero while
    `verified_completed` remains false;
22. a contract that requires mutation rejects no-change with `contract_unsatisfied`;
23. persistent check failure, unavailable executor, exhausted repair, and unsafe required commit
    each return their distinct structured nonzero result;
24. resumed scratch changes remain task-owned, checkable, and committable relative to the original
    task baseline;
25. resume reuses prior check evidence only while all relevant fingerprints and outputs match;
26. a valid harness no-change run does not produce a misleading no-commit observation;
27. older run-index and journal records deserialize while new per-run records remain immutable;
28. harness evaluation counts runtime handoff checks independently of model `run_check` tool calls;
29. trusted workspace configuration is recorded but does not appear in the scratch diff; and
30. `pb harness infer`, `bench`, and `cache-clean` retain their CLI and runtime behavior.

## Verification and commit policy

Use one semantic commit per workstream. For any web behavior change, build and test the web UI as
required by `AGENTS.md`. For Rust changes, run focused tests first and the full repository suite
before marking the plan complete. The final audit must include:

1. `deno task build:web`
2. `cargo test --all-targets`
3. `deno task test:web`
4. `cargo build --release --target aarch64-apple-darwin`
5. focused `pb harness agent` tests for contract-free, contracted, no-change, failed-check,
   unavailable-executor, commit-blocked, and resumed-scratch outcomes
6. parsing/runtime regression tests for `pb harness infer`, `bench`, and `cache-clean`
7. the expanded scripted `pb harness eval` suite, including checked schema-v2 JSONL output
8. focused monorepo/Cargo workspace fixture tests
9. one bounded real-model run for each supported handoff outcome after deterministic coverage passes

FlashMoe smoke/architecture work is required only if these workstreams change the FlashMoe backend
or its data flow; ordinary agent-loop/event/UI changes should not manufacture unrelated backend
work.

## Completion audit

- [ ] Model final is treated as handoff intent rather than achievement proof.
- [ ] The web UI presents deterministic feedback as a team member in the group chat.
- [ ] Repository root and selected project focus remain distinct throughout a session.
- [ ] Affected checks are selected from a typed workspace graph.
- [ ] Polyglot components can use separate lazy executors.
- [ ] Cargo workspaces use metadata-aware package/dependency selection.
- [ ] Generated bundle dependencies are ordered and fingerprinted.
- [ ] Current check evidence is reused and stale evidence is rerun.
- [ ] Failed checks cannot produce a managed commit.
- [ ] No-change sessions run no unnecessary change-conditioned work.
- [ ] Managed commits contain only session-owned changes.
- [ ] Explicit harness contracts converge on the same handoff implementation.
- [ ] Harness resume preserves task ownership with distinct task and invocation baselines.
- [ ] `pb harness agent` exits and `verified_completed` follow the documented matrix.
- [ ] Harness journals and run index preserve handoff, evidence, team-message, and commit outcomes.
- [ ] Harness evaluation accounts for runtime checks, repair, executors, no-change, and safe commit.
- [ ] Non-agent harness subcommands remain behaviorally unchanged.
- [ ] All new events/configuration retain backward-compatible reads.
- [ ] Deterministic and full repository verification passes.
