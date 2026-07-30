# Internal harness

`pb harness` is a hidden CLI surface for exercising pb internals. It is intentionally omitted from
top-level help because it is a testing tool rather than a supported end-user workflow.

## Blocking agent runs

Run a complete agent task without starting `pb serve`, connecting to the Unix socket, or creating a
daemon session:

```bash
pb harness agent "Build a small Rust CLI that prints a greeting"
```

`--intent deliver` is the default and runs the strict workflow. Use `--intent discuss` for a
read-only conversational experiment or `--intent auto` to allow a read-only turn to request an
explicit transition into delivery.

The command blocks until `agent_core::run_agent` completes or fails. Existing web and `pb queue`
session paths continue to use their normal daemon lifecycle. `journal.md` is initialized before model
loading, so an interrupted run still leaves the scratch location and raw-event recovery guidance.
`--max-steps` caps each model-driven strict stage as well as a direct agent run; when a workflow
policy declares a lower stage limit, the lower bound wins.

For a bounded active-Goal control experiment, a trusted caller may inject the same read-only model
projection used by the daemon:

```bash
pb harness agent --intent discuss \
  --goal-context fixtures/harness-goal-context.json \
  "Call goal_status, then request a safe-boundary pause"
```

The JSON is parsed and validated before model loading. Impossible stages, counters above their
budget, malformed plan digests, and inconsistent milestone totals are rejected. The projection
contains no authority. It exposes only `goal_status`, `goal_pause`, `goal_request_amendment`, and
`goal_request_budget`; the last three record controller requests in JSONL and the Goal audit section
of the journal. They cannot apply an amendment, increase a budget, resume, cancel, accept, publish,
or rewrite a Goal. Without `--goal-context`, those active-Goal tools remain hidden.

### Deterministic controller-action qualification

`pb harness agent` uses the same intrinsic deterministic actions as daemon, desktop, web, and queue
runs. There is no rendering or deletion option. Controller observations are one truthful
`controller_block` user/context message and emit typed durable events and stage evidence with their
actual origin, exact coverage, fingerprints, byte counts, and authority effects. They never create
model `tool_call` or `tool_result` events.

Admission uses the active model's prompt renderer and tokenizer. A candidate observation must stay
unchanged through compaction and keep the complete prompt at or below 55% of usable capacity. Full
small-file observations may seed read-before-write evidence; failed-diagnostic ranges authorize
only edits wholly inside the included byte windows. Fresh review inspection is all-or-none and
never supplies assessments or a verdict. A successful final mutation can carry optional
model-authored completion fields, while a controller no-change close is limited to structurally
empty, mutation-forbidden work. Inline completion exposes only the model-owned step status and
bounded accounting text: summaries are limited to 1,024 characters and the semantic commit subject
to 200. Plan identity, fingerprints, touched paths, and the no-change fact are projected from trusted
current state only after the mutation succeeds.

After a planned create succeeds, an exact-path diagnostic failure makes that file an existing repair
target. The controller can inject its current bounded bytes and read-before-write receipt directly,
avoiding a model-authored read turn before the target-bound repair. It cannot do this before the
create exists or for a different path.

Automatic deletion requires a unique accepted delete of a tracked, clean, unchanged file or
symlink. It never applies to directories, dirty, untracked, adopted, stale, oversized, forbidden,
or ambiguous content. Harness summaries separately report controller observation count and prompt
bytes, coverage, controller closures, and controller mutations. The safety proof is recorded in
[Deterministic controller actions](controller-action-elision-plan.md).

Run the locked native/controller experiment against an explicit local model and an empty,
persistent output directory outside the pb source tree:

```bash
pb harness action-elision-eval \
  --model hf://mlx-community/Qwen3-Coder-Next-4bit \
  --output-dir /private/tmp/pb-action-elision-e1
```

The evaluator creates one dedicated Git fixture, resets it to the same baseline before each arm,
and begins after an accepted plan and fresh plan review. It records configuration, source-tree,
running-executable, model-artifact, and fixture digests; every semantic generation input and actual
rendered prompt digest; events; final bytes; and a machine-readable summary for native and truthful
controller-block arms. Protocol version 2 requires byte-identical read results, identical fixture
inputs, a durable controller receipt matching the generation input, and exactly one
controller-owned user/context block with no model tool call or tool-call ID. Behavioral and
artifact outcomes remain reported separately, so a weak model is not mislabeled as a provenance
failure.

### Acceptance contracts

An optional trusted JSON contract makes completion externally verifiable:

```bash
pb harness agent --contract docs/harness-contract-v1.example.json \
  "Build the requested game and satisfy the supplied contract"
```

Version 1 can require a final mutation, restrict changed paths, define named checks, require
semantic commits, require a clean worktree, and state the exact paths/checks a review must inspect.
See `docs/harness-contract-v1.example.json` for the complete shape.

The optional `work_unit_guidance` object maps exact normalized task paths to short trusted hints.
pb repeats only the active path's hint when that typed work unit is ready to mutate. Guidance is
limited to 64 entries, 512 bytes per path, and 4,096 bytes total; when `allowed_paths` is nonempty,
every guided path must be allowed. It is advisory prompt material only: it cannot select a path,
grant read or mutation authority, satisfy a check or review, advance a stage, or earn progress. This
gives a small local model a concise implementation constraint without copying verifier commands or
opaque check identifiers into every turn.

An individual check may set `"diagnostic_eligible": true`. After all typed work units are
structurally complete, pb may run that check once as fingerprint-bound repair feedback. The preview
cannot satisfy any required-check or review gate; authoritative checking reruns it after the model's
typed implementation submission. A failing preview focuses repair only when its bounded output
names an exact current task path. Previous reads of that path are invalidated before a bounded
replace/edit repair, and the preview must not change repository or Git control state.
Diagnostic-eligible checks have a 60-second timeout ceiling even though authoritative checks may use
the general one-hour contract maximum.

The agent receives `run_check(id)` only when the contract defines checks. The check command is
trusted caller input; the model supplies only its ID. Each run records its exit status, bounded
stdout/stderr, duration, timeout state, executor, command/input/output fingerprints, and evidence
source. In strict delivery and trusted contracts, only `run_check(id)` earns named-check credit;
`run_command` and `run_task` remain diagnostic or mutation tools even if their command text
resembles a configured check. A restored legacy/direct request may route an exact configured guard
through the check runtime for compatibility, but cannot use that path to satisfy a strict contract.
Evidence becomes stale when its declared inputs,
dependencies, command, executor, or outputs change. Contracts are parsed before model loading and
their checks compile into the same workspace check graph used by deterministic handoff.

An optional trusted topology can describe polyglot components, separate executors, generated
outputs, and check dependencies without copying configuration into the scratch repository:

```bash
pb harness agent --workspace-config /absolute/path/workspace.toml \
  "Update the API and its generated web assets"
```

The file is parsed, validated, and normalized before model loading. Its canonical source path,
SHA-256, and executor policy are recorded in run metadata. Without it, the harness loads repository
workspace configuration when present and otherwise discovers manifests, components, executors, and
checks from the isolated workspace before model loading. This matches ordinary delivery check
selection: a changed path automatically runs its affected discovered checks even when the model did
not copy check IDs into the plan. An explicit `--workspace-config` remains authoritative and skips
discovery. Workspace configuration defines how work is checked; the v1 contract remains the
separate source of task-specific acceptance facts.

An empty `allowed_paths` list means unrestricted paths. Otherwise built-in write tools reject a
path outside the list before mutation, while `run_command` and final validation detect indirect
forbidden changes. Check timeouts are limited to one hour and terminate the local command process
group. Contracts add task-specific facts to the strict workflow; they do not restore the former
prompt-owned review or commit gates.

Terminal output and the `session_summary` event distinguish `reached_final`, `handoff_outcome`,
`contract_status`, `verified_completed`, and `termination_reason`. In strict delivery, a model final
cannot advance a stage. The named structured submission must pass harness validation before pb
moves through planning, fresh plan review, implementation, checking, fresh code review, repair, or
managed commit. pb computes the task delta, selects affected checks, starts only their executors,
reuses only current fingerprint-bound evidence, and creates or reuses a safe task-owned commit only
after the complete workflow passes.

A contract-free `ready` or `no_change` workflow exits zero while retaining
`contract_status=unspecified` and `verified_completed=false`. A satisfied explicit contract exits
zero with `contract_status=satisfied` and `verified_completed=true`. Strict delivery applies the
contract to planned and final mutation, allowed paths, required checks, named fresh-review reads and
check evidence, semantic commit requirements, and final workspace cleanliness; attaching a contract
does not make `Ready` verified by itself. During planning, pb projects immutable required contract
check IDs into the first submitted acceptance fact and recomputes the plan digest before validation
and fresh review. The model selects only additional configured checks; direct or restored plans that
still omit a required check are rejected before implementation. Required mutation with no delta becomes
`contract_unsatisfied`; persistent check failure, missing executor, repeated repair failure, and an
unsafe required commit remain distinct nonzero `checks_failed`, `executor_unavailable`,
`repair_exhausted`, and `commit_blocked` outcomes. Step, parse, runtime-engine, and resource-limit
exits keep their existing structured reasons. A prompt whose authoritative anchors and exposed
schemas cannot fit is rejected before inference with the distinct `context_limit` reason. Older
stored summaries without these additive fields remain readable with conservative defaults.

### Task-completion qualification

The deterministic control corpus proves harness behavior, not generated artifact quality. The
checked-in `fixtures/harness-task-completion` directory adds two explicit offline artifact
qualifications:

- TC1 requires two exact small files and qualifies native next-missing-path execution through checks,
  fresh review, managed commit, and a clean worktree; and
- TC2 requires a small dependency-free module, model-authored tests, documentation, an independent
  behavior check, and the same complete strict workflow.

A task-completion fixture passes only with a satisfied explicit contract, verified completion, an
independently reproduced check, the recorded commit at `HEAD`, and a clean worktree. Protocol
containment, task completion, artifact quality, wall time, and energy are reported separately. The
[reliability plan](task-completion-reliability-plan.md) defines the locked repeatability and corpus
promotion gates.

`scripts/summarize-harness-completion.ts` reports rendered prompt tokens, reused cached-prefix
tokens, actual fresh-prefill tokens, cache-hit invocation count, and observed native tool-schema
digests separately. The usability auditor also aggregates cache misses by reason. These fields
measure prefix reuse without treating cached tokens as work avoided by a weaker contract.

Each `llm_invocation.prompt_cache` also records a typed `miss_reason` when no prefix was reused:
`cache_disabled`, `cold_session`, `prompt_diverged`, `stable_prefix_unavailable`,
`cache_unreadable`, `context_reset`, or `runtime_unsupported`. A partial or full hit omits the field.
This taxonomy comes from the active inference backend, so preserved-run audits can separate expected
cold starts and unsupported paths from prompt instability or broken persisted state.

The same directory contains a schema-validated 11-case TC3 candidate corpus spanning ordered
creation, repair after a failed check, a one-file fix, regression tests, related multi-file work,
failed-check diagnosis, delete-and-modify work, out-of-scope resistance, mixed create/modify work,
adopted work from a resumed scratch baseline, and truthful no-change completion. List it with:

```bash
deno run --allow-read scripts/run-harness-task-corpus.ts --list
```

Run one case at a time with a new scratch root and the current release binary:

```bash
deno run --allow-read --allow-write --allow-run \
  scripts/run-harness-task-corpus.ts \
  --case fix_average_divisor \
  --scratch-dir /private/tmp/pb-corpus-fix-average-1
```

`--prepare-only` materializes the exact seeded repository, trusted contract, task, and optional
resumed-task baseline without loading a model. Every scratch root is retained and audited with the
same independent procedure as TC1/TC2. The manifest and runner establish reproducible inputs; they
do not constitute the pending aggregate TC3 model result.

The run audit records every observed strict named check in its planned and executed check evidence,
even though strict delivery does not emit the legacy handoff summary. Harness-owned next-path work
unit guidance is recorded as positive control evidence rather than a model limitation.

### Private-workload usability qualification

The checked-in `fixtures/harness-usability` corpus exercises 24 synthetic Rust, Python, and
React/TypeScript repository repairs derived from public benchmark task shapes. It exists for local
engineering decisions, not agent comparisons. Reference solutions never enter model workspaces.
Each task request states every behavior enforced by its official check; checks may be stricter about
exact values, but they do not hide an additional product requirement behind shorthand. This keeps
small-model failures attributable to implementation or control rather than an ambiguous stimulus.

Run `deno task test:usability` to prove that every seed fails, every isolated reference repair
passes, and every official check is side-effect-clean. React packages require the one-time
`deno task cache:usability-react` bootstrap; actual corpus checks are cached-only and frozen.

`scripts/run-harness-usability.ts` runs selected cases or the full corpus sequentially and preserves
each scratch root. `scripts/audit-harness-usability.ts` reruns task checks and verifies immutable
fixtures, actual changed paths, the recorded commit, worktree cleanliness, and retained event metrics
without trusting pb's own verdict. Official correctness, pb verification, safe clean delivery, and
efficiency remain distinct fields. See the
[private-workload usability record](benchmarks/private-workload-usability.md) for the exact protocol,
commands, and current internal sample.

`scripts/run-paired-harness-usability.ts` compares two explicit release binaries without discarding
run variance. It requires an odd repeat count of at least three, rotates case order, alternates which
binary runs first, locks the explicit model and sampling settings, independently audits every
scratch root, checkpoints partial progress atomically, and retains raw values plus per-case and
aggregate medians. Its result fails the production
performance gate unless correctness, clean verification, the Rust/Python four-call floor, exact
candidate root reuse, the locked fresh-prefill ceiling, paired-median energy and wall time, and
per-case regression requirements all pass. A harness process that ends nonzero is still
independently audited and retained rather than aborting later trials. A non-passing completed
comparison exits with status 2 while preserving the report.

```sh
deno run --allow-read --allow-write --allow-run \
  scripts/run-paired-harness-usability.ts \
  --scratch-parent /private/tmp/pb-paired-qualification \
  --baseline-binary /path/to/baseline/pb \
  --baseline-revision BASELINE_REVISION \
  --candidate-binary target/aarch64-apple-darwin/release/pb \
  --candidate-revision CANDIDATE_REVISION \
  --evaluator-revision EVALUATOR_REVISION \
  --model hf://mlx-community/Qwen3-Coder-Next-4bit \
  --repeats 3
```

### Prompt budget and bounded results

Every model call is rendered and tokenized before it is charged to the invocation budget. The
llama.cpp and FlashMoe paths use the same chat-template and tokenizer path for preflight as for
generation; the scripted engine uses a deterministic counter. The usable prompt capacity is the
declared context minus the current generation reserve and a fixed 32-token safety margin. pb begins
compacting completed assistant/tool exchanges above 70% of that usable capacity and targets 60%.
The task, current stage/contract material, accepted workflow artifacts, fingerprints, checks, and
terminal requirements remain authoritative anchors and are not summarized away.

Compacted exchanges become deterministic context receipts containing the tool name, canonical
argument hash, success state, bounded excerpt, exact omission counts, workspace fingerprint, and
evidence effects. A receipt grants no authority of its own. Full tool results are emitted to the
durable event stream before their prompt representation is shortened, so prompt budgeting never
removes audit evidence. The `llm_invocation.context` snapshot records preflight and backend prompt
tokens, usable capacity, safety margin, schema tokens, compacted messages, and omitted result
content. Runtime parity checks reject any disagreement between preflight and the backend-reported
prompt count.

Each `llm_invocation` also records a controller-classified purpose, plus the active workflow stage
and profile when present. The purpose distinguishes conversation, Task partitioning, planning,
review, evidence, mutation, closure, and recovery calls. Constrained high-level Task-planning calls
use the same event and runtime/token/energy accounting as stage-loop calls.

Built-in `read_file` results are whole-line bounded against the active context and generation
reserve. An omitted or oversized range ends with a machine-generated continuation containing the
exact next line and `next_call` JSON, plus a targeted `ripgrep` suggestion. Other oversized prompt
results use a deterministic prefix/suffix representation with a raw SHA-256 and exact omitted
character, byte, and line counts. If compaction cannot make the prompt fit without altering anchors,
pb emits `context_limit` with the measured token count and largest prompt sections; no model
invocation or generation budget is consumed.

### Focused planning and review evidence

Planning and fresh plan review receive a deterministic `RepositoryBrief` rather than the complete
normalized workspace graph. The brief is capped at 16,000 characters and includes the focus root,
component/executor/check/task identifiers while space permits, manifests, likely entry points,
bounded project-instruction excerpts, top-level paths, and task-dirty paths. It always carries the
SHA-256 of the complete normalized graph and explicit omitted counts. The complete graph—not the
brief—continues to validate component and check IDs in submitted plans. Repository instructions in
the brief are labeled evidence-only and cannot add user authority.

When a trusted contract names `allowed_paths`, planning may additionally receive complete existing
small UTF-8 files as controller observations. Fresh plan review receives a new stage-bound rendering
of the same eligible exact bytes instead of relying only on the serialized carried-evidence bundle;
ordinary read tools remain available for absent or partial evidence. The complete candidate set must
fit below 55% of usable prompt capacity and survive a second workspace/path hash check. Each accepted
observation records controller origin, stage, exact byte coverage and hashes, seeds the
read-before-write gate, and persists a complete stage-evidence entry. A stage never injects the same
path observation twice. Missing, symlinked, binary, partial, changed, oversized, or displaced files
are skipped so the model can use the ordinary read tool.

Read-before-write authority is consumed only in its source stage. In particular, planning or plan
review evidence cannot make an implementation work unit mutation-ready without a fresh
implementation-stage observation or read. This prevents a current but no-longer-visible review
excerpt from authorizing a write in the next teammate's context.

Checkpointed evidence retains its complete controller receipts, hashes, sizes, source stage/tool,
and ordering. Model prompts receive only a compact path/content projection after pb revalidates the
current path hash. When isolated plan review will receive eligible contract-path bytes as fresh
controller blocks, its initial user message does not duplicate those bytes or receipt metadata; if a
block does not survive preflight, the reviewer's ordinary read tools remain available.

For initial planning only, when fresh full controller observations cover every existing path in a
non-empty trusted `allowed_paths` set, pb narrows that first generation to the stable
`PlanningClosure` authority containing only `submit_plan`. Missing paths remain represented by the
trusted planning path-state projection. Unsafe, partial, oversized, binary, symlinked, or unobserved
existing paths keep the ordinary planning evidence tools. Fresh plan review always retains its
independent repository-read authority; a challenged plan revision also returns to tool-enabled
planning. This prevents a local model from spending another invocation rereading bytes pb just
supplied without converting controller evidence into a review verdict.

Fresh code review receives a changed-path manifest capped at 16,000 characters, selected check IDs, and bounded check
evidence instead of a complete diff followed by duplicate complete current files. Each manifest
entry states added, modified, deleted, or renamed status; prior path when applicable; text, binary,
symlink, or other content kind; and whether focused inspection is required. Reviewers call
`inspect_change(path)` for each current reviewable text path. The tool returns bounded relevant diff
hunks, numbered current context around those hunks, the exact checked content fingerprint, and
path-relevant current check summaries. New files use bounded current context when Git has no index
diff; deleted and binary paths report that text is unavailable rather than claiming it was read.

A successful text `inspect_change` earns the same fresh, invocation-local path evidence as
`read_file`. `submit_code_review` remains hidden until every current changed text path has earned
that evidence, and final artifact validation rechecks the complete trusted graph, selected check
ledger, changed-path set, and checked fingerprint. `limits.review_diff_bytes` now caps each focused
inspection representation; `limits.review_paths` continues to bound the complete task-delta path
set. Full checked content remains bound by workspace and commit fingerprints and available in the
isolated review workspace and resulting managed commit.

### Outcome-aware progress and deterministic read reuse

The existing exact-call guard still blocks a third consecutive identical tool action before it can
run. A bounded post-result progress guard additionally fingerprints the tool family, normalized
call and outcome, repository content, and harness evidence state. Two failed outcomes without a
content or evidence transition produce one correction with a concrete alternative. An unchanged
A-B-A call cycle is blocked before the proposed third call; two equivalent stale workspace edits
also prevent another edit until the model gathers different evidence. Other third equivalent
failures terminate the sequence before another model turn. A real content or evidence transition
clears the relevant failure window.

Successful deterministic built-in reads use a 64-entry run-local cache. The initial cacheable set
is `read_file`, `glob`, `ripgrep`/`search`, and `git_log`; only successful results enter it. Keys bind
normalized arguments to current tracked/untracked repository content, Git control state, active
request/contract/tool policy, context-dependent result bounds, and the exact target bytes for
`read_file` (including explicitly read ignored files). Non-Git temporary workspaces use a
deterministic filesystem fingerprint that excludes `.git` and `.pb` control state.

A hit replays the exact original result and only the original read-path, contract-read, and legacy
review-read effects. It cannot replay check, write, workflow-transition, or broader review
authority. Commands, status that includes external session state, network, MCP, LSP, memory,
focused review inspection, and every mutation-capable tool remain uncached. Cache hits consume zero
tool runtime/energy and are counted in the context snapshot for the model invocation that receives
the replayed result.

Built-in tool effects come from one runtime record used by exposure, batching, caching, progress,
and execution. The retired `todo`, `git_commit`, and `git_revert` schemas are absent from current
sessions and cannot be restored by an allowlist or policy. Dynamic MCP tools are exposed only from
an operator-audited read-only name list; server annotations cannot add authority. Parallel MCP
batches require every member to be both read-only and server-marked idempotent.

Deterministic recovery is bounded. Repeated parse, artifact validation, identical-tool, plan-cycle,
repair-cycle, invocation, token, stage-step, and advisory failures stop with explicit outcomes
before another model turn can reinterpret the same fact. Strict workflow stages never use the old
prose-final grace path. Literal `REVIEW PASS` text and model-requested commits have no current
workflow meaning; code-review credit comes only from the structured, fingerprint-bound review
artifact. The former profile gate/final-grace/handoff behavior is retained only in memory for an
actually restored persisted request that predates conversation intent, so old sessions remain
readable without letting a new request opt into weaker control by omitting fields.

For max-token native calls, repeated-failure identity includes the attempted tool and current
workspace/evidence fingerprints. A parsed tool action that subsequently fails without progress does
not erase the earlier capped-action history. Capped `write_file` and `replace_file` corrections say
that no partial file was created and direct the model to produce materially shorter complete content.
The preserved DeepSeek V4 Flash field evidence is recorded in the
[agent-harness field run](benchmarks/deepseek-v4-flash-agent.md).

Implementation turns whose accepted plan consists only of creating currently missing paths start
in action-first mode: reasoning is disabled for the first bounded turn so the model spends that
turn on a native edit action. Later implementation turns, every repair turn, planning, and both
fresh review stages retain normal reasoning. The decision is derived once from the validated plan
and current workspace; it is not a model option or environment toggle.

After fresh plan review, implementation and repair persist a typed ledger for every planned create,
modify, and delete. The ledger records plan step, operation, target, task/invocation/current
fingerprints, adopted provenance, progress credit, diagnostic focus, and structural state. Only the
active unit's read or mutation tool is exposed. Target-bound tools do not require the model to copy
the path: pb inserts the ledger path into the durable call before validation and execution. Existing
modify/delete targets require a current complete read; adopted task-owned deltas can already be
structurally complete without false model authorship. Exact-path diagnostic failures invalidate
older reads and reopen only that path as a repair; they do not repeat the original create operation.
Under a trusted one-path contract, a failed check also reopens that sole changed path when its
assertion text describes only the observed symptom and omits the filename. Multi-path work does not
receive that inference.

For the last unfinished controller-rendered work unit, mutation schemas require the typed
implementation completion beside the mutation payload. The executor applies the mutation before
validating that completion against the new fingerprint. Acceptance advances immediately; rejection
reports `mutation_succeeded=true` and keeps the bounded implementation-submission fallback.

A diagnostic-failed work unit under a trusted one-path `allowed_paths` contract exposes only its
focused read and then target-bound repair tools. `request_replan` is omitted because another plan
cannot select a different authorized path and would discard current review/check state without
changing authority. Multi-path or contract-free work, uncovered failures without a unique contract
target, and `blocked_for_replan` filesystem transitions retain the existing replan route.

For a ranged controller observation, the initial implementation turn hides `read_file` and exposes
the controller-authorized mutation directly. If a constrained mutation explicitly reports one
missing fact, its single recovery turn exposes only `read_file` with one center line and at most
forty surrounding lines on each side. This compact generated schema makes a whole-file read
structurally impossible instead of spending a turn and rejecting it only after generation. A
completed explicit excerpt has no continuation into the remainder of the file.
After a partial mutation recovered from a constraint dead-end, pb discards the pre-mutation
controller ranges and observes the current target again before another teammate inference. That
continuation is restricted to `edit_file`; the rejected multi-hunk patch grammar is not re-entered.

Creation units execute one controller-bound path per model action, so an unknown multi-file payload
cannot consume the turn before the first accepted-plan path is complete. One real content/evidence
transition can earn one extra turn per unit, at most four per stage. No failed, rejected, cached,
repeated, no-op, or bookkeeping action earns budget. When the ledger is complete, pb projects plan identity,
fingerprint, touched paths, and no-change into implementation accounting; the model still supplies
step status, summaries, and commit subject.

Each run creates a persistent scratch root under the system temporary directory unless
`--scratch-dir` selects a new path. The layout is:

```text
pb-harness-.../
├── workspace/       # isolated git repository used by the agent
├── task-baseline.json # immutable original task baseline
├── adoptions.jsonl  # explicit provenance for resumed external changes
├── multi-task-checkpoint.json # latest durable parent checkpoint, when Task planning activates
├── events.jsonl     # cumulative compatibility AgentEvent stream
├── journal.md       # latest-run compatibility view
├── run-index.jsonl  # append-only started/finished run records
└── runs/<run-id>/
    ├── events.jsonl # immutable event stream for this invocation
    ├── task-planning-transcript.json # deterministic bypass or constrained attempts and route
    └── journal.md   # final journal, or running recovery journal if interrupted
```

An explicitly supplied scratch directory may already exist when it is empty; the harness
initializes it as a new scratch root. A non-empty existing directory is treated only as a resume
candidate and is rejected unless it contains the expected Git workspace, so unrelated contents
are never adopted or overwritten.

The optional `task-planning-transcript.json` records the final controller decision. A bounded
single-Build bypass has a deterministic reason and no attempts. An attempted partition additionally
preserves every compact planner prompt, schema, raw and normalized artifact, typed failure, and usage
record. Historical transcripts can also contain the retired advisory-critic role. The
optional `multi-task-checkpoint.json` mirrors the latest accepted multi-Task parent checkpoint
for qualification and recovery inspection. It contains the accepted plan, controller-owned
budgets, active child checkpoint, usage watermarks, repository boundary, and terminal reason. It is
written only when an accepted proposal creates a multi-Task run; a one-Task or fail-soft Build route
does not fabricate it.

The harness allocates the run ID and writes the per-run `running` journal plus a `started` index
record before model loading. Events are flushed to both streams. Final journals are atomically
replaced and a `finished` index record captures the structured outcome. Resuming a scratch root
creates a new run directory and never rewrites a prior run; any partial dual-write failure is
surfaced instead of reporting verified completion.

Resume restores the original task baseline, captures a new invocation baseline, and treats earlier
scratch work as part of the same task. Prior successful evidence is restored from cumulative
events and reused only while all current fingerprints match; externally edited resumed content is
appended to `adoptions.jsonl`. An active workflow resumes on its existing task branch; allocating a
new run ID does not create or switch to another branch before checkpoint validation.

Finished journals and run-index records include the handoff outcome, both baseline identifiers,
workspace-config provenance, affected components, planned/executed/reused/failed/skipped checks,
output fingerprints, executor starts, repair turns, team feedback evidence, no-change
classification, commit disposition/hash, and the publication-ready evidence digest and sanitized
repository remote. The full evidence bundle remains in the durable workflow checkpoint and binds
the commit to its accepted plan, fresh code review, and selected check evidence. No publisher or
network operation is invoked by the harness. Raw process output stays in event JSONL. A valid
no-change run does not produce the old misleading no-commit observation.

Failed local `run_command` actions return the exit status and bounded combined stdout/stderr to the
model, including diagnostics that the command redirected to stdout. This is the same bounded output
recorded in the event stream, so a failed compiler or test invocation remains actionable without
granting it completion evidence.

The same bounded command runner covers host and managed execution. `run_command` defaults to 120
seconds and accepts a maximum of 600 seconds. Timeout and user cancellation are distinct results;
pb terminates the owned process group or managed exec while draining stdout and stderr concurrently.

File mutations require an exact current-content fingerprint, use atomic no-clobber or replace
operations, and reject stale concurrent edits. `mv` is limited to files and symlinks. `rm` operates
on the final entry and can remove a file, symlink, or empty directory, not a recursive tree.
`run_task` validates a bounded isolated snapshot, stages its complete declared
output set, rejects Git-control/undeclared-path/unsafe-symlink changes, and rolls back destination
paths if promotion fails.

For a trusted contract, named check evidence is earned only by `run_check({"id":"..."})`. A raw
`run_command` that exactly matches or extends the declared command remains ordinary shell evidence;
its result explicitly steers the model to the named function but never marks the check current.
The checked-in `fixtures/harness-browser-game` example supplies a dependency-free named check that
rejects HTTP(S), protocol-relative, npm, and JSR references in generated browser assets.

The workspace starts on `main` with one empty baseline commit. The agent receives the same
conversation/workflow engine as web and queue, a local command backend rooted in the scratch
repository, and the build profile by default. Strict stage capabilities—not the profile prompt—
decide which tools are available. Changes stay reviewable as managed commits on the generated task
branch.

The journal is an initial audit aid, not a substitute for review. A supervising Codex run should:

1. Inspect P0/P1 observations and the raw event stream.
2. Review and test the committed workspace changes.
3. Fix clear harness or agent-runtime bugs in pb itself and commit those fixes.
4. Add ranked manual observations that automatic event classification could not infer.
5. Replace or extend the scaffold with a concrete plan for non-blocking improvements after the task
   succeeds.

Automatic observations include an explicit `pb_defect`, `model_limitation`, `experiment_error`, or
`positive_evidence` classification. Bounded model stops, missing evidence, dirty experimental
workspaces, and an unfinished run are P2/P3 observations rather than P0/P1 defects. P0/P1 is
reserved for severe, evidenced pb failures that require immediate supervisor action.
Successful active-work-unit guidance, unique progress credit, and passing diagnostic previews are
positive control evidence. In addition to explicitly eligible contract checks, pb can preview at
most one affected discovered project type-check or web check before accepting inline completion;
the preview is capped at 60 seconds. A failing preview remains a model limitation and never
receives check, review, commit, or completion credit.

The automatic missing-commit observation applies only to a `ready` delivery. Engine errors,
exhausted repairs, failed checks, and other incomplete outcomes retain their actual terminal cause
without being mislabeled as completed work.

FlashMoe inference, benchmark, and cache-clean utilities also live beneath the hidden harness, for
example `pb harness infer ...`. `infer` and `bench` accept
`--metal-working-set-limit-mib <MiB>` to lower the device-derived safety limit. The override can
only make the default policy stricter and is applied before expert-access graph resolution, so a
lower limit can select the existing positioned-read implementation instead of resident expert
mapping. `--resource-summary` prints the opt-in JSON resource ledger; normal runs keep it disabled
and emit tracing only for high-water changes, pressure recovery, or a resource-limit abort. The
ledger includes cumulative Metal command submissions and actual bytes copied from host to Metal or
read back through tracked runtime buffers. Native generation telemetry records the corresponding
per-prefill deltas, allowing the scalar reference and promoted device-resident layer-major graph to
be compared without inferring traffic from model geometry. `infer --no-thinking`
asks a checkpoint's chat template to suppress emitted reasoning; it cannot be combined with
`--raw`, which bypasses the chat template entirely.
`infer --json-schema PATH` loads a JSON Schema before model generation, disables emitted reasoning,
and exercises the production LLGuidance constraint session. It is text-only, cannot be combined
with prefill parity, and fails closed if the active Hugging Face or DeepSeek tokenizer cannot
compile the schema. This is the qualification surface for checking real-model structured output;
it does not change production configuration.
`infer --tool-fixture PATH` similarly exercises the production native-tool constraint without
starting the multi-stage agent workflow. The version 1 JSON fixture contains `tools` in the native
`name`/`description`/`input_schema` shape, optional `terminal_tool_names`, and up to 32 immutable
UTF-8 snapshot entries under `files`, each with `path` and `content`. The harness forces
`tool_required`, supplies even an empty snapshot for mutation tools, disables emitted reasoning,
prints the accepted tool-call array as JSON, and fails if no call closes. It cannot be combined with
JSON Schema, raw or image input, prefill parity, session reuse, or repeat generation. Snapshot paths
are canonical repository-relative paths and the same 32 MiB production bound applies. This explicit
fixture surface is for deterministic Qwen/DeepSeek tokenizer and mutation-gate qualification; it is
not a configuration switch and never reads or writes the named workspace files.
Reproducible Python create and patch examples are checked in under `fixtures/control-collar/`.
`pb harness llama-infer PROMPT --model MODEL.gguf --tool-fixture FIXTURE.json` is the corresponding
exact-GGUF qualification surface for the production llama.cpp adapter. It loads the named local
model, forces the fixture's native tool call through the full-vocabulary collar, never executes the
returned mutation, and prints a JSON report containing token/timing counts, the content-free
constraint report, and complete parsed calls. It exits nonzero for an empty candidate frontier,
token exhaustion without a complete call, a malformed call, or fixture/model setup failure.
`--show-transcript` is an explicit local debugging opt-in that can print generated tool arguments or
source fragments to the terminal on failure; normal runs omit that sensitive preview. The command is
bounded by `--max-tokens`, does not persist the transcript, and may report the existing CPU-only
correctness fallback when accelerated context creation fails. It qualifies one exact
model/tokenizer/template profile and does not promote unrelated GGUFs or dialects.
`pb harness collar-qualify` is the model-free promotion surface for the source-prefix layer. It
loads the exact tokenizer and tokenizer-configuration artifacts selected by `--model`, replays the
versioned `fixtures/control-collar/prefix-language-v1.json` corpus at every real token boundary,
checks prefix-monotonic decoding, rollback equivalence, deterministic random chunkings per case,
and the final expected validity/rule. `--random-chunk-replays` defaults to 64 and is bounded to
65,536 per case so scheduled high-replay qualification remains explicit and finite. It prints only
artifact digests, counts, and p50/p95/p99/max
probe timings, and fails when p95 exceeds `--latency-budget-micros` (1,000 microseconds by default).
The command does not load model weights, read a workspace, enable a production feature, or persist
source. Run it separately for every tokenizer profile being promoted; live `infer --tool-fixture`
runs remain the end-to-end model/template/throughput gate.
`pb harness semantic-qualify --server <name>` is the provider-profile promotion surface. It creates
an ephemeral Git workspace from the versioned corpus, runs the configured digest-pinned no-network
provider against a fresh exact read-only semantic shadow for every generation and final-executor
case, and requires complete authoritative receipts plus exact allow/reject and diagnostic-class
matches. The report contains only corpus/provider/configuration digests, counts, and latency
percentiles. Provider unavailability, an incomplete project graph, detached documents, an unknown
result, or a latency-budget breach fails qualification and does not enable semantic enforcement.
The checked-in Rust corpus covers valid standard-library and forward references, type/call/name/
field/method/mutability/ownership failures, and a canonical multi-file patch.
`pb harness native-world-qualify --language python|rust` is the model-free native lifecycle,
serialization, and resource qualification surface. It starts a separate pb process for each
deterministic tiny, representative, and large graph. Python uses 4/7, 1,024/515, and 10,000/5,003
first-party/dependency files; Rust uses two Cargo targets with 6/6, 258/130, and 2,050/1,026
workspace/dependency files. Each arm crosses the ordinary production pre-inference lifecycle and
requires one cold readiness barrier, one warm request, one exact process-cache hit, a rejected
invalid final replay, and an accepted valid final replay. It then drives 4/4/2 host workers through
65/33/9 alternating invalid/valid replays against the serialized writable analyzer overlay and
requires a final valid recovery replay. Current resident memory before and after stress provides a
bounded retained-growth check; whole-process peak RSS remains a separate bound.

The report contains only profile/world identities, counts, byte totals, analyzer load/prime timings,
complete lifecycle/stress timings, and memory values; no fixture source is printed or persisted.
Language-specific defaults fail Python above 60 seconds cold, 20 seconds per final replay, 120
seconds aggregate stress, or 1 GiB peak RSS; Rust uses 180 seconds cold, 30 seconds per replay, 240
seconds stress, or 4 GiB peak RSS. Both use 20 seconds warm/cache and 512 MiB retained-growth
ceilings. These are qualifier failure bounds, not latency promises for arbitrary projects, and may
be overridden only on this hidden measurement command. Peak measurement requires Unix; current RSS
measurement requires macOS or Linux. The qualifier never loads a model, and passing it does not
substitute for live backend fixtures or semantic false-rejection corpora.
`pb harness python-semantic-qualify --corpus
fixtures/control-collar/semantic-python-v1.json` is the language-owned Python semantic promotion
surface. It materializes the digest-locked first-party files and static dependency image in an
ephemeral Git workspace, prepares and primes the ordinary native `ty` world before any case, and
runs every complete mutation through the production generation gate, an independent
execution-time replay, and a separate promoted-diagnostic delta. The checked-in version 1 corpus
requires allow and reject cases for annotated, unannotated, and frozen third-party code; all six
promoted diagnostic classes; baseline debt, dynamic unknowns, and multi-file transactions; and all
four write/replace/edit/patch tools. The report contains only corpus/world/configuration/dependency
digests, category and diagnostic counts, parity counts, exhaustive UTF-8 prefix-probe and
deterministic rollback-replay counts, and timings. Every case probes every logical UTF-8 boundary and
64 reproducible rollback/full-replay branches; a hard rejection must remain hard for every longer
prefix. It never invokes an LSP, network, package installer, or model. Frozen dependency imports and
symbol shapes are resolved directly by the language crate before inference. Passing this corpus does
not promote another diagnostic to token-time hard rejection: that still requires a separately named
monotonic proof.
`pb harness rust-semantic-qualify --corpus fixtures/control-collar/semantic-rust-v2.json` applies the
same production generation, independent final-replay, direct diagnostic-delta, exhaustive prefix,
and rollback checks to the exact native Rust profile. Corpus validation requires all ten promoted
diagnostic classes, baseline debt, cross-crate repair and failure, conservative unknowns for import
context and create/delete topology, and all four mutation tools. The fixture is an ephemeral offline
Cargo workspace; no build script, procedural macro, model, LSP, network, or generated mutation is
executed. The content-free report additionally records the prepared Rust target count and native
world identity. This qualifier proves the checked profile and corpus, not rustc equivalence.
For Qwen prefill qualification, `infer --prefill-mode auto|scalar|layer-major` selects the promoted
policy, exact scalar reference, or an explicit layer-major request. `auto` promotes only a prepared
Qwen3-Coder-Next affine-Q4 graph with at least 32 fresh tokens and sufficient live Metal reserve.
`--prefill-state-summary` opts into complete hidden/KV/router/recurrent fingerprints; ordinary
generation does not construct the diagnostic router/recurrent trace.
`--prefill-parity` loads the model once and fails unless scalar and layer-major content and state
match exactly; `--prefill-parity-prefix-tokens N` additionally warms and genuinely restores an
exact raw-token prefix before comparing the remaining suffix. `--prefill-chunk-tokens N` forces a
smaller layer-major boundary and is accepted only with parity or explicit layer-major mode. These
are hidden harness controls, not production configuration or environment toggles.
`infer --session-id ID --repeat N` runs text inference through the exact production session cache,
prints cached and actually-prefilled token counts for each pass, and persists the final checkpoint
for families with a versioned disk-session format. `--repeat 2` verifies live reuse without a
second model load. The pinned DeepSeek V4 profile uses bounded complete-state Metal checkpoints in
memory only. Structured agent stages retain an exact first-message base and current-prompt
checkpoint under a stage identity that includes the system prompt and tool schema, allowing tool
results and bounded correction retries to extend the base without silently resetting or crossing
stage contracts. Structured DeepSeek checkpoints are additionally keyed by the exact rendered
stable-root token digest: unchanged roots reuse complete Metal state, while tool-schema or authority
narrowing starts cold instead of colliding with an incompatible prefix. Raw prefix-extension
qualification keeps its explicit session identity. A second process starts fresh because DeepSeek
has no partial or alternate disk format.

The agent/evaluation handoff plumbing does not apply to `pb harness infer`, `pb harness bench`, or
`pb harness cache-clean`; they remain direct diagnostic utilities rather than workflow stages.

Agent `llm_invocation` events now separate native fresh-prefill and decode tokens, wall time and
rates, and record the model family, active expert count, resident/streamed strategy, prefill command,
effective thinking state, tool-constraint mode/digest/rejections, serialized tool-schema tokens,
largest serialized action, mutation payload allowance, and carried-evidence bytes. `tool_batch`
events add call, parallel-safe, useful, bookkeeping-only, and dependency-rejection counts. These
fields are optional/defaulted so existing journals remain readable.

Cache-capable agent invocations also report a privacy-safe `prompt_cache.root` record. It includes
the local backend, model-namespace and exact rendered-token digests, eligible/reused root tokens,
explicit system-instruction version and workflow stage, bounded stage-authority class, tool-schema
digest, and output-constraint mode. The descriptor is bound before inference rather than inferred
from model output. The completion
summarizer and usability auditor aggregate eligible roots, reused roots, complete root hits, and
authority-class counts without storing prompt or source content.
The optional bounded `lookup_detail` separates missing/divergent session state from exact-root
fallthrough hits and misses; it contains no prompt or repository content. Aggregate reports retain
those detail counts and miss counts grouped by stage and authority class. Token reconciliation
violations are classified as pb telemetry defects rather than accepted as performance evidence.

`native.refill` records memory lookup, disk read/decode, CPU validation/allocation, state hydration,
fresh-suffix prefill, snapshot capture, and persistence-queue milliseconds separately. Terminal
session metrics record queued and completed durable checkpoints, durable wall time, and failures.
Native usage records the selected prefill command and a bounded reason such as complete-root
restore, below-threshold suffix, qualified layer-major suffix, forced scalar reference, unsupported
graph, or resource limit. The auditor aggregates command and reason counts.
The usability summary and aggregate preserve each total, so a cache-policy change cannot claim a
prefill improvement by moving work into restore, checkpoint capture, or persistence.

### Managed prompt-cache lifecycle qualification

`scripts/run-harness-cache-scenario.ts` prepares six clean copies of one bounded usability case and
invokes the hidden `pb harness cache-eval` surface. The arms run in a fixed order: cold empty
storage, warm same logical session in the same process, new session in the same process, changed
planning-tool authority, matching original authority after that incompatible probe, and a fresh pb
process using the persisted root. For example:

```bash
deno run --allow-read --allow-write --allow-run \
  scripts/run-harness-cache-scenario.ts \
  --scratch-parent /private/tmp/pb-cache-eval \
  --binary target/aarch64-apple-darwin/release/pb \
  --case rust_registry_removal
```

Use `--prepare-only` to print the exact binary argument vector without starting inference. The
evaluator requires six distinct, previously unused absolute scratch roots and matching contracts,
an absolute empty or absent cache root, and a new absolute report path. It refuses to replace an
existing report or use a non-empty cache root. The first five arms share one loaded FlashMoe
runtime; pb then drops its unleased pooled runtime and starts the final arm as a child process, so
restart evidence cannot come from live Metal state.

The preparer retains `cache-eval-scenario.json` with the source revision and dirty-state flag,
binary path and SHA-256, operating-system/architecture class, exact argument vector, sampler limits,
arm paths, timestamps, and final exit code. The evaluator report independently fingerprints the
executable and records the resolved model, model directory, sampler settings, cache root, model
namespace, backend/cache format, and per-arm evidence.

The JSON report fails closed unless the planning root is cold initially, completely reused by the
same and new sessions, changed by the narrower tool schema with zero old-root reuse, reusable again
when the original schema returns, and completely restored from `disk_prefix` by the child. All
comparisons use the backend's exact rendered-token digest and token counts. Each arm also records
whether the agent satisfied its task contract, but generated-artifact quality is deliberately not
a cache gate: an incomplete local-model edit must not hide a cache defect, and a cache hit must not
turn an incomplete artifact into a pass.

`--cache-dir`, `--session-id`, and `--exclude-tool` on `pb harness agent` are hidden evaluator
plumbing. They are explicit CLI arguments rather than production configuration or environment
switches. Exclusions can name only the existing harness tool set and can narrow, never broaden, the
active stage capability. Strict managed stages normally ignore a direct-run allowlist so it cannot
accidentally remove their typed terminal tools. The evaluator therefore carries its exclusions in a
separate non-serializable harness field, applies them only after deriving stage capabilities, and
rejects any attempt to remove a required terminal action.

## Control evaluation

Run the deterministic, model-free control suite with:

```bash
pb harness eval --jsonl harness-eval.jsonl
```

The command writes one schema-v3 JSON object per fixture and prints a compact table covering valid
actions, named-check compliance, false completion, recovery loops, turns, latency, tokens, energy,
and termination. JSONL additionally records selected components/checks, runtime executions versus
model `run_check` calls, reuse and dependency skips, started/avoided executors, team messages,
repair turns, no-change, commit disposition, output fingerprints, workflow stages, and artifact
hashes. The complete control suite also contains seven deterministic Goal assertions for exact plan
approval, model-tool authority, sequential milestones, pause/checkpoint/resume, amendment evidence,
completion basis, and budget/cancellation accounting. Goal records carry the stage, outcome,
completion basis, plan/checkpoint hashes, progress, and cumulative usage without labeling
subjective artifact quality verified. The bounded open-weight protocol matrix and its known
limitations are recorded in
[Enforced workflow open-weight model evaluation](harness-workflow-model-evaluation.md). Without `--jsonl`, JSONL
goes to stdout and the table goes to stderr so the machine stream stays parseable. The scripted
report contains no timestamps, scratch paths, or nondeterministic commit IDs and is stable enough
to diff directly; real-model records retain the resulting commit hash. A protocol mismatch exits
non-zero. `artifact_quality` remains separate from protocol scoring. Schema-v1 fixture/result data
is rejected explicitly rather than compared under changed metric meanings.
The checked fixture corpus also models resumed scratch work: `resumed_files` are applied after the
original task baseline is captured, so evaluation can prove that inherited uncommitted work remains
task-owned, checkable, and committable without requiring a model or a second process.

Real-model matrices are opt-in and must name the local model explicitly:

```bash
pb harness eval --model model.gguf --model-dir /path/to/models \
  --max-tokens 512 --ctx-size 32768 --temperature 0 --top-k 1 --seed 0 \
  --jsonl model-eval.jsonl
```

Use `--suite small-model` to run the stable subset used by
[the small-model reliability plan](small-model-agent-reliability-plan.md) and its checked
[S0 baseline](benchmarks/small-model-agent-baseline.md) and
[S1 prompt-budget checkpoint](benchmarks/small-model-agent-s1.md). S2's deterministic prompt
reduction is recorded in the [focused-evidence checkpoint](benchmarks/small-model-agent-s2.md).
S3's no-progress and read-cache boundaries are recorded in the
[progress-recovery checkpoint](benchmarks/small-model-agent-s3.md).
S4's structured tool failures and bounded truncation retry are recorded in the
[action-recovery checkpoint](benchmarks/small-model-agent-s4.md).
S5's deterministic closure checkpoint and conservative schema pruning are recorded in the
[workflow-closure checkpoint](benchmarks/small-model-agent-s5.md). The repeated 8K/16K rollout
decision and final local-model comparison are recorded in the
[S6 report](benchmarks/small-model-agent-s6.md).
Goal-control qualification, the 4B/7B/14B matrix, and its rollout decision are recorded in the
[Goal G8 report](benchmarks/goal-mode-g8.md).
Each model-invocation record
includes an optional backward-compatible context snapshot covering capacity, generation reserve,
prompt high-water utilization, preflight/backend token counts, safety margin, message/schema size,
thinking mode, retry reason, and the compaction/cache/closure counters introduced by later
milestones. Evaluation summaries count `thinking_off_truncation_retries`,
`bounded_read_constraint_retries`, `expanded_mutation_payload_retries`,
`compact_mutation_truncation_retries`, and `larger_cap_truncation_retries` separately. Runtime
setup failures are reported separately from artifact quality so a backend experiment error is not
scored as model reasoning.

Real-model records also include a bounded `tool_trace` for diagnosis. Each entry stores a tool name
capped at 120 characters plus the normalized argument SHA-256, a 600-character JSON preview, and a
truncation flag. Full write contents or patch arguments therefore cannot make the evaluation record
unbounded.

Every real-model record repeats the backend, model, resolved model directory, token/context/thread/
GPU settings, sampling values, seed, FlashMoe resource-policy version, normalized workspace-config
hash, and executor policy. One loaded engine is reused across the corpus. FlashMoe evaluation
refuses to start if the versioned bounded-resource policy is inactive.

Failed `apply_patch` checks now return a bounded mismatch diagnostic with the target file, expected
hunk line, a few old-side patch lines, and numbered current content around that location. Large
lines and the full diagnostic are capped; binary targets are identified without dumping bytes. The
check path never mutates the file, and the correction recommends `edit_file` for exact current text
or `replace_file` when the target has drifted substantially.

Built-in tool failures are returned to the model as JSON `tool_failure` envelopes capped at 2,400
characters. Each envelope has a stable `reason_code`, the requested tool, a bounded message,
`retryable`, the exact `valid_signature` when that tool was exposed, and a
`suggested_next_action`. A close miss may include `suggested_tool`, selected only from the tools
exposed for that invocation; pb never rewrites or executes the misspelled call. Missing, unknown,
or wrongly typed top-level arguments are rejected against the exposed JSON schema before policy or
runtime execution, and pb does not guess, coerce, or add values.

When a generation reaches its token cap without a valid action, pb first retries once at the same
cap with thinking disabled and an action-only correction. A visibly truncated workflow terminal
call keeps the existing terminal-only schema on that retry. If the result is still truncated, pb
gives a capped native `write_file` or `replace_file` at most one compact retry when applicable,
with only that tool exposed and without the rejected payload in model context. The requested
payload is below half the normal mutation allowance, the retry schema enforces that exact smaller
bound, its token ceiling is reduced accordingly, and its path is bound to the original target when
that path was present before truncation. pb rejects a parsed retry that changes the compact tool or
bound path. The payload must remain complete and loadable, and a failed compact retry does not grow
back to the original cap. When compact recovery does not apply, pb may grow a truncated action cap
once within the request limit; existing parse-loop thresholds remain authoritative.
Every attempt reserves an invocation and records generated tokens before another retry is allowed,
so the global invocation and token budgets cannot be bypassed by recovery.
An action-only retry remains within its original stage step: durable stage-step accounting advances
once per visible `step_started`, while every backend attempt is still charged to the separate model
invocation and generated-token limits.

Each compatibility edit action ends its model turn. The prompt forbids invented tool results, and
pb does not replay fenced compatibility actions followed by fabricated transcript markers into the
next model context. Only the validated first action is executed; its real result and harness content
fingerprint determine the next state.

One model turn is not limited to one independent tool. Native function-call output and the JSON
compatibility `tool_calls` form may carry a batch; every member is validated before execution,
parallel-safe members run together, and all real results return before the next inference pass.
Calls whose outcome is needed by a later call stay in separate turns, and authority-changing
workflow or delivery transitions must be the only call in a batch.

The native runner can stop as soon as the constrained JSON body of an exposed stage-submission
tool is complete. A missing `</tool_call>` suffix is added only to the parser input; it is not
invented as model evidence. Non-terminal tool calls continue decoding so independent calls can
still share a response. Constraint recovery can force only a schema-valid closing suffix, rejects
non-EOS tokens that do not increase decoded length, and prevents a repeated 32-token continuation
from consuming the cap. If `write_file` or `replace_file` content reaches its schema limit while
still open, generation stops as a truncated named call instead of force-closing a cut-off file;
the normal bounded compact recovery then applies.

Successful edit transport is not sufficient for progress credit. `replace_file`, `edit_file`, and
`apply_patch` must produce a real byte transition; an identical result returns a structured failure,
emits no diff, and does not invalidate review evidence. A dirty workspace preserved after an
incomplete delivery is reported as model-limitation evidence, while the same state after a claimed
ready or verified result remains an experiment error and still fails a clean-workspace contract.

Tool exposure also follows the event sink's actual interaction capability. The non-interactive
CLI harness does not advertise `ask_user`, because it cannot deliver an answer; interactive web
sessions do. A model therefore cannot spend bounded workflow turns waiting on an unavailable
question channel.

A native batch is atomic with respect to truncation. If generation reaches its token cap after one
or more complete calls but leaves a later call incomplete, none of those calls execute; the same
bounded shorter-action/larger-cap recovery used for a single truncated action runs instead. This
prevents a small bookkeeping call from hiding a cut-off write later in the completion.

Implementation prompts include exact compatibility examples for both creation and replacement.
`write_file` still refuses an existing target; its failure directs the model to call `read_file`,
wait for the authoritative result, and only then use `replace_file` in a later turn. The
read-before-write gate is unchanged.

Step-limit monitor decisions distinguish explicit false fields such as `off_track: no` and
`blocked: no` from an actual unhealthy status. A healthy `grant more steps: yes` checkpoint can use
the configured bounded extension, while loop evidence and explicit no-grant decisions still stop.

When a persistent scratch directory starts a new workflow over adopted partial work, its planning
snapshot is the current invocation baseline rather than the original empty task baseline. The
original baseline still defines the task-owned delta, but plan submission is checked against the
workspace the model can actually inspect.

On implementation or repair, the stage prompt lists the current state of every planned path
relative to the original task baseline. This state survives process restarts even though advisory
TODOs do not: already-created or modified paths are explicit, missing paths remain explicit, and
the model is warned not to retry `write_file` against an existing path.

An all-create accepted plan additionally turns that durable state into a deterministic next-path
schema. The first missing path in plan order is the only `write_file` target for the turn, and the
implementation terminal stays unavailable until every planned creation exists. This is a
model-control narrowing over the existing executor and checkpoint; it introduces no alternate
scheduler or mutation authority.

During a strict workflow stage, pb can recover a complete unwrapped JSON object, either plain or in
one JSON code fence, when the exposed schemas identify exactly one tool. The sole special-case tie
is `write_file` versus `replace_file`, selected from whether the bounded target currently exists.
The recovered action still goes through normal path, policy, schema, capability, and artifact
validation. For workflow terminal tools only, an otherwise valid outer argument object may contain
its typed artifact field as one complete JSON string. pb decodes that field exactly once, applies
the same typed artifact validation, preserves the original call, and records the normalization in
the result. It never repairs partial JSON, recursively unwraps strings, or invents missing fields.
Prose, arrays, and other ambiguous objects are not coerced. Implementation and repair also retain
their edit tools after a rejected prose final instead of exposing only a submission that cannot
truthfully advance.

Plan paths are checked in step order. Creating a missing path makes it available to later modify or
delete steps, while modification before creation and duplicate creation of an existing path remain
invalid. This ordered state is deterministic and the artifact validator remains the authority.
Contract `allowed_paths` are also checked when the plan is submitted, not only when an edit is
executed. All examples and corrections use workspace-relative paths; `repo/` has no special
meaning.

On each of the final two ordinary turns of planning, plan review, plan revision, or code review, pb
adds a bounded JSON `workflow_closure_checkpoint` derived from harness state. It reports ordinary
steps remaining, the current and harness-expected content fingerprints, the exact schema-derived
terminal signature, whether that terminal is eligible or hidden, and the missing deterministic
facts. The checkpoint is refreshed for each invocation instead of becoming durable authority in
the transcript. Its invocation context increments `closure_checkpoints`; the existing `tool_count`,
`tool_schema_chars`, and `tool_schema_tokens` fields record the narrowed schema surface.

`ToolExposureState` only filters an already authorized set. During those closure turns, planning
and plan review retain focused repository evidence tools; code review retains `inspect_change` and
targeted reads/search. If the final turn has no missing precondition, only the exact terminal tool
is exposed. Missing path reads, stale fingerprints, or a direct request allowlist keep the terminal
hidden, and the execution boundary repeats the same deterministic check if a model hallucinates the
call anyway. Implementation and repair schemas are not pruned by this policy.

Review stages narrow earlier than the final-two-turn closure checkpoint. As soon as the deterministic
plan-review or code-review terminal precondition is current, ordinary turns expose only that terminal
plus the stage's focused evidence tools. This removes unrelated discovery, network, memory, and
delegation schemas without granting terminal eligibility or weakening the executor-side check.

A challenged plan revision with concrete work-unit paths also excludes public web tools and LSP
tools whose configured language IDs do not match any accepted path. Document-scoped LSP execution
rechecks the path language before starting a lazy provider. Built-in web tools run their async client
on a dedicated bounded runtime thread, including when the harness itself is already inside Tokio;
a model-requested web action therefore cannot panic the owning agent runtime through nested
`block_on`.

Plan-review evidence check IDs are constrained to exact configured check names and observed-evidence
validation happens inside `submit_plan_review`. A typo therefore returns terminal-tool feedback in
the same review context rather than discarding honest reviewer findings during an outer stage retry.
Fresh task-focused observations that cover every existing proposed plan path make the review turn
terminal-only, preventing generic file pagination while preserving a blocking verdict. For
implementation, a native constraint dead end while forming `apply_patch` gets one fresh `edit_file`
recovery instead of retrying the identical irreparable patch construction. That smaller recovery
cannot close the path-level work unit: pb re-observes the result and exposes the remaining bounded
edit. Inline completion also runs diagnostic-eligible checks before handoff; failures carry their
exact output into target-bound repair, where controller-selected line windows replace whole-file
rediscovery and `request_replan` is withheld for the already-authorized diagnostic path.

Direct bounded runs now derive every instruction from the actual allowlist. A restricted prompt
never orders an unexposed setup, command, review, or commit tool. When those general discovery
tools are absent, the prompt includes at most 32 sorted top-level repository paths, each capped at
120 characters; `.git` and `.pb` are excluded. The path hint is orientation only and earns no read
or review evidence.

A `read_file` request beyond EOF remains a non-mutating result, but the progress guard records that
range as known-empty. If unchanged content is followed by another known-empty request on the same
path—either after an empty result or an exact deterministic cache replay—the next request is
blocked before tool execution. Valid continuation ranges, different paths, and state transitions
remain available.

S6 does not add a model-family control override: 8K and 16K had identical protocol outcomes, no
overflow, low schema/prompt utilization, and no result compaction pressure. Existing context,
result, thinking, and truncation defaults remain authoritative. There is no automatic stronger
local-model or cloud escalation; selecting another local model is an explicit `--model` choice.
