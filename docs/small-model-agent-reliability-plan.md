# Small-model agent reliability plan

Status: active; S0-S4 complete

This document is the tracking source for improving the quality and completion rate of small local
models in pb's agent and enforced-delivery workflows. Update the milestone table, evidence log, and
completion audit as work lands.

This plan complements, and does not replace:

- [Harness reliability improvement plan](harness-improvement-plan.md), which owns truthful
  completion, durable evidence, bounded control recovery, and harness evaluation;
- [Conversational sessions and enforced delivery workflow plan](conversational-delivery-workflow-plan.md),
  which owns workflow stages, authority, capabilities, checks, review, and managed commits; and
- [FlashMoe architecture parity plan](flashmoe-architecture-parity-plan.md), which remains the
  source of truth for FlashMoe data flow, scheduling, resource ownership, and model-family parity.

The earlier harness plan deliberately did not treat weak artifact quality as a harness defect. This
plan takes the next layer: make the same safe harness easier for a small model to use without
weakening any acceptance gate.

## Outcome

A small local model should receive a prompt it can understand and finish within its real context
window. It should get focused repository evidence, bounded and actionable tool results, early help
when it is repeating an unproductive strategy, and a deterministic closure checkpoint before its
turn budget expires.

Success means all of the following:

1. No accepted workflow fact, check, review requirement, content fingerprint, or path restriction
   can be removed or overridden by context compaction.
2. Prompt construction reserves enough room for generation and refuses an impossible request before
   backend inference begins.
3. Large reads, command results, workspace graphs, and review material cannot consume the context
   window without a bounded continuation path.
4. Reviewers inspect the exact checked content without receiving both complete files and a duplicate
   complete diff.
5. Repeated failures are recognized from outcomes and workspace progress, not only identical tool
   arguments.
6. Truncated reasoning gets one bounded action-oriented recovery attempt rather than immediately
   doubling the generation budget.
7. Near the step limit, the model is told exactly what remains and cannot start an unnecessary new
   tool chain after all terminal preconditions are satisfied.
8. Deterministic fixtures prove the control behavior; reproducible real-model trials measure model
   effectiveness separately from protocol correctness.

## Influences and local decisions

The concrete influences are SmallCode's context caps, semantic compression, tool routing, plan
anchors, patch-first editing, and early-stop detection, plus Little Coder's read guard, context
watchdog, bounded quality corrections, thinking budget, and pre-cap finalization warning:

- <https://github.com/Doorman11991/smallcode>
- <https://github.com/Doorman11991/smallcode/blob/master/ARCHITECTURE.md>
- <https://github.com/itayinbarr/little-coder>
- <https://github.com/itayinbarr/little-coder/blob/main/.pi/extensions/thinking-budget/index.ts>
- <https://github.com/itayinbarr/little-coder/blob/main/.pi/extensions/quality-monitor/index.ts>
- <https://github.com/itayinbarr/little-coder/blob/main/.pi/extensions/finalize-warn/index.ts>

pb will borrow narrow mechanisms, not framework claims or self-reported benchmark scores. In
particular:

- stage capabilities remain harness-owned and deterministic;
- read-before-write stays strict; there is no permissive second-attempt bypass;
- compaction produces deterministic receipts, not a model-authored summary that can become hidden
  authority;
- raw tool output and workflow artifacts remain durable even when their prompt representation is
  compacted;
- tool-name recovery may suggest a valid schema but must not execute arbitrary prose or YAML;
- automatic cloud escalation is out of scope; and
- persistent hidden shell state and automatic worktree rollback are out of scope.

## Invariants

- Preserve all safety and truthfulness invariants in the plans linked above.
- Do not change strict workflow stage order or let prompt text advance a stage.
- Do not let a context receipt satisfy evidence that the underlying tool execution did not earn.
- Evidence effects are fingerprint-bound. A cache hit or compacted receipt is valid only for the
  exact content/evidence fingerprint under which it was produced.
- Preserve full raw events for audit. Prompt truncation is not event truncation.
- New event fields are additive and deserialize conservatively when absent.
- A context failure terminates explicitly; it must not be reported as model refusal, ordinary step
  exhaustion, or verified completion.
- Deterministic transcript and prompt-assembly tests precede real-model experiments.
- Real-model comparisons use the same model, template, seed, sampling, context, fixtures, and
  machine conditions before and after a change.
- Artifact quality, protocol compliance, context efficiency, and runtime performance remain
  separate measurements.
- Do not touch project-local `.pb/` state while implementing or evaluating this plan.
- If implementation adds a per-project configuration field, update `src/init.rs` in the same
  milestone.
- FlashMoe backend or model-family changes require the architecture plan update and smoke required
  by `AGENTS.md`; prompt/control-only work does not justify a FlashMoe architecture change.

## Baseline facts

The implementation already has several strong controls that this plan must extend rather than
replace:

- `StageCapabilities` provides stage-specific allowlists and terminal actions.
- Terminal submission tools are hidden until deterministic preconditions are satisfied.
- Existing files must be read before built-in mutation tools can overwrite them.
- Checks and review evidence are content-fingerprint-bound.
- Strict workflow stages use structured submissions and fresh contexts.
- Parse, gate, exact-tool, stage, invocation, generated-token, plan-cycle, and repair-cycle failures
  are bounded.
- `pb harness eval` separates protocol results from artifact quality and supports deterministic and
  explicitly configured local-model runs.

The effectiveness gaps are:

- `RunBudget` limits invocations and generated tokens but not accumulated input tokens.
- `read_file` returns the entire remaining file when no end range is supplied.
- planning and plan review embed the complete pretty-serialized workspace graph.
- code review embeds a complete diff followed by complete current contents of each changed text
  file, then requires the reviewer to read those files again.
- the default bounded review material limit is 200,000 bytes, independent of the active tokenizer
  and usable prompt budget.
- `ToolLoopGuard` recognizes exact call signatures but not repeated outcomes, alternating loops, or
  lack of workspace/evidence progress.
- a truncated unparsable completion first retries with a larger token cap; thinking is disabled only
  after that retry path fails.
- tool errors are unstructured strings and do not consistently identify the valid next action.
- strict stages have no proactive remaining-turn checkpoint.

## Target architecture

### Exact prompt budgeting

Extend `CompletionEngine` with a prompt-measurement operation that renders the same messages, tool
schemas, chat template, and thinking mode used by generation and returns the exact token count.
Implement it with the active llama.cpp tokenizer and FlashMoe tokenizer/template. The scripted
engine receives a deterministic counter so tests remain model-free.

For each generation:

```text
context capacity
  - reserved generation tokens
  - fixed safety margin
  = usable prompt capacity
```

The prompt assembler targets at most 70% of usable prompt capacity before generation. If it crosses
that soft limit, it compacts old completed tool exchanges to a 60% target. It may use the remainder
when authoritative current-stage material requires it, but it must never exceed usable capacity.
Ratios should begin as internal policy constants. Expose configuration only after deterministic and
real-model evidence shows a legitimate need.

The assembler classifies prompt material as:

- **authoritative anchor** — current task, stage contract, accepted plan/review, current content
  fingerprint, selected checks, unresolved findings, terminal requirements; never summarized away;
- **current evidence** — recent reads, diffs, diagnostics, and tool results still needed for the next
  action; retained while space allows;
- **replaceable history** — completed assistant/tool exchanges; replaceable by deterministic
  receipts; and
- **discardable duplicate** — content already represented by an authoritative artifact or current
  receipt; removable.

A receipt records tool name, normalized argument hash, success/failure, bounded result excerpt,
omitted bytes/lines, workspace fingerprint, and explicit evidence effects. Receipts are prompt
material only; the durable event stream retains the original result.

If anchors plus required schemas and generation reserve cannot fit, terminate with a structured
`context_limit` reason that reports measured tokens and the largest prompt sections. Do not silently
drop an anchor or retry inference with an impossible prompt.

### Bounded repository evidence

`read_file` should honor explicit small ranges. An omitted or excessively large range returns the
largest whole-line slice allowed by the current tool-result budget, followed by a machine-generated
continuation instruction containing the next line number and a `ripgrep` suggestion. Binary files
remain excluded.

Introduce two deterministic representations:

- `RepositoryBrief`: a capped view derived from `WorkspaceGraph` and repository state containing
  focus root, components, manifests, entry points when known, executor/check IDs, relevant project
  instructions, top-level paths, and dirty paths. It carries the full graph checksum.
- `inspect_change(path)`: a code-review read tool returning file status, relevant diff hunks,
  bounded current context around each hunk, selected-check diagnostics relevant to the path when
  available, and the checked content fingerprint. A successful call records the same path-read
  evidence required by code review.

Planning receives the brief instead of the complete graph. Code review receives a changed-path
manifest, brief check evidence, and the exact fingerprint; it calls `inspect_change` for changed text
paths. Deleted, renamed, binary, and newly created paths need explicit bounded representations and
tests.

### Progress-aware recovery

Retain the current pre-execution exact-call guard and add a post-result `ProgressGuard`. Its key is
based on tool family, normalized outcome/error fingerprint, workspace content fingerprint, and
evidence-state fingerprint. It keeps a bounded recent window so it can identify A-B-A-B cycles and
different arguments that produce the same unchanged failure.

- First occurrence: no intervention.
- Second no-progress occurrence: emit one correction containing the repeated outcome and concrete
  alternatives.
- Third occurrence without workspace/evidence progress: block the call and terminate or require a
  stage-appropriate replan.
- A real content or evidence transition resets the relevant no-progress sequence.

Add a small cache only for deterministic built-in reads such as `read_file`, `glob`, `ripgrep`, and
repository-local status/history reads. Key it by normalized arguments, content fingerprint, and
policy scope. A cache entry stores its evidence effects so replay cannot accidentally earn more
authority than the original execution. Do not initially cache commands, MCP calls, LSP calls,
network results, memory tools, or any mutation-capable tool.

### Action-oriented recovery and closure

Render built-in tool failures as a bounded `ToolFailureEnvelope` with a stable reason code, concise
message, retryability, valid signature, and one suggested next action. Unknown tool names may receive
a nearest valid tool suggestion. Missing or mistyped arguments receive schema-derived suggestions,
but pb does not silently reinterpret or execute them.

For a max-token completion without a valid tool action:

1. retry once at the same token cap with thinking disabled and an action-only correction;
2. if that attempt is also truncated, use the existing bounded cap-growth policy once; and
3. charge every attempt to the existing global invocation/generated-token budget.

Do not claim exact thinking-token enforcement until a backend reports thinking tokens separately.
The initial policy controls thinking mode and recovery attempts, which are observable across both
backends.

When two ordinary steps remain, append a deterministic closure checkpoint listing remaining steps,
missing terminal preconditions, current fingerprint, and the exact terminal tool/schema. With one
step remaining and all terminal preconditions satisfied, expose only the terminal submission tool.
An unmet precondition never becomes satisfiable merely because the step limit is close.

### State-aware tool exposure

Stage capability checks remain the authority. After the preceding metrics exist, add a narrower
`ToolExposureState` that chooses schemas from the already-authorized set based only on deterministic
workflow state. Initial reductions should be conservative:

- planning and plan review: repository evidence tools plus the stage terminal tool when eligible;
- code review: `inspect_change`, targeted repository reads/search, and `submit_code_review` when
  eligible;
- terminal-only closure: only the required submission tool; and
- no classifier-based hiding in implementation/repair until evaluation proves a safe rule.

Do not use a model or regex classifier to grant tools. A schema-budget optimization may remove an
authorized tool from one turn, but it can never make an unauthorized tool available.

## Workstreams

### S0 — Baseline and measurement foundation (P0)

Add deterministic measurements before changing prompt behavior.

Implementation:

- Extend scripted completion fixtures so they can record rendered prompt size, exposed tool names,
  thinking mode, and closure messages for each invocation.
- Add additive context metrics/events: exact prompt tokens, usable capacity, high-water ratio,
  schema tokens, compacted messages, omitted tool-result bytes, read-cache hits, and closure uses.
- Add a `small_model` fixture group to the existing harness evaluation rather than creating a
  separate competing evaluator. Bump its schema only if the new fields cannot be additive.
- Capture a checked-in baseline at `docs/benchmarks/small-model-agent-baseline.md` using the fixture
  matrix below and one explicitly named available local model.
- Record model path identifier/hash, backend, chat template source, context, seed, temperature,
  top-k, token caps, build commit, and machine conditions.

Likely files: `src/agent_core.rs`, `src/events.rs`, `src/harness_eval.rs`, harness fixture data,
`docs/harness.md`, and the new baseline report.

Acceptance:

- scripted metrics are deterministic and require no model, network, Metal, or container;
- old stored events deserialize with conservative defaults;
- the existing control fixture corpus retains identical protocol outcomes; and
- the baseline report distinguishes protocol, task quality, context efficiency, and runtime.

### S1 — Prompt budget and bounded tool results (P0)

Implement exact measurement, prompt assembly, deterministic receipts, and bounded `read_file`.

Implementation:

- Add a focused module such as `src/agent_context.rs`; avoid further growing prompt-policy logic
  inline in `agent_core.rs`.
- Add exact prompt measurement to all completion engines using the same render/tokenize path as
  generation.
- Assemble and measure every generation request before model invocation.
- Compact replaceable history to deterministic receipts at the soft limit.
- Add `context_limit` termination and additive event/journal reporting.
- Apply a per-result budget to reads and other large built-in results. Preserve existing raw event
  evidence when the prompt view is shortened.

Acceptance:

- fixture CB1 reads a 5,000-line file in an 8K context without inference overflow and receives an
  exact continuation line;
- CB2 compacts repeated large successful tool results while preserving their durable raw events;
- CB3 proves task, accepted artifacts, fingerprints, checks, and terminal requirements survive
  compaction byte-for-byte;
- CB4 fails before generation with `context_limit` when anchors alone cannot fit;
- exact preflight measurement agrees with the backend's reported prompt tokens for representative
  llama.cpp and FlashMoe templates; and
- all existing read-before-write and review-read gates still pass their deterministic tests.

### S2 — Repository brief and focused review evidence (P0)

Remove duplicated graph/review material and add focused inspection.

Implementation:

- Build a deterministic capped `RepositoryBrief` from the normalized graph and repository state.
- Replace complete graph JSON in planning/plan-review prompts with the brief and full-graph hash.
- Add `inspect_change(path)` to code-review capabilities and read evidence.
- Replace `workflow_review_material` prompt embedding with a bounded changed-path manifest and
  focused inspection calls.
- Keep full checked bytes and diffs available to deterministic validation and durable audit; only
  their prompt representation changes.

Acceptance:

- RV1 reviews three changed 5,000-line files without duplicate full-file/diff prompt content;
- RV2 covers new, deleted, renamed, and binary paths without claiming unavailable text evidence;
- RV3 proves `submit_code_review` remains unavailable until every changed text path was inspected;
- RB1 produces a stable bounded brief for a large synthetic polyglot workspace;
- planning artifacts still resolve only real component/check IDs from the complete trusted graph;
- the constructed large-review prompt is at least 40% smaller than the S0 baseline; and
- no check, review, path, or fingerprint gate changes semantics.

### S3 — Outcome-aware progress guard and safe read cache (P1)

Detect repeated lack of progress across non-identical calls.

Implementation:

- Add post-result outcome/error fingerprints and evidence-state fingerprints.
- Detect exact, alternating, and same-outcome loops in a bounded window.
- Emit one actionable correction at threshold two and block at threshold three.
- Add the fingerprint-scoped deterministic built-in read cache and explicit cache-hit metrics.
- Keep command, network, MCP, LSP, memory, and mutation tools uncached.

Acceptance:

- PG1 preserves existing exact-call loop behavior;
- PG2 blocks A-B-A-B unchanged failures without a fourth model-driven retry;
- PG3 allows the same operation after a real workspace/evidence fingerprint transition;
- PG4 groups different stale patch arguments that return the same unchanged failure;
- PG5 proves a cached read replays only its original evidence effects at the same fingerprint; and
- no blocked call mutates the repository or consumes tool runtime.

### S4 — Structured errors and action-oriented truncation recovery (P1)

Make failures cheap for a weak model to understand and recover from.

Implementation:

- Add stable built-in tool error reason codes and bounded envelopes.
- Derive valid signatures and argument hints from the exposed schema.
- Suggest, but do not automatically execute, the nearest exposed tool name.
- Retry a truncated non-action once with thinking disabled at the same cap before cap growth.
- Record thinking mode and retry reason in additive events/metrics.

Acceptance:

- AR1 turns a one-character tool-name hallucination into one valid correction without executing the
  invalid call;
- AR2 reports a missing/incorrect argument with the exact valid signature and no guessed mutation;
- AR3 reaches a valid tool call after one same-cap thinking-off retry;
- AR4 proves all attempts count against invocation and generated-token limits;
- correction loops remain bounded at the existing deterministic thresholds; and
- error envelopes remain below their declared prompt budget for long error chains.

### S5 — Closure checkpoint and conservative schema pruning (P1)

Help the model finish the work it has already proved.

Implementation:

- Emit the two-step closure checkpoint from deterministic stage/gate state.
- At the last step, expose only the terminal tool when all preconditions are current.
- Add `ToolExposureState` for planning, plan review, and code review using only authorized tools.
- Record exposed schema count/tokens per invocation.
- Do not prune implementation/repair schemas beyond existing rules in this milestone.

Acceptance:

- CL1 completes a valid near-cap submission without starting a new tool chain;
- CL2 cannot submit while a read, mutation, check, review, or fingerprint precondition is missing;
- CL3 reports the exact current terminal schema and fingerprint;
- SE1 proves every exposed tool is a subset of `StageCapabilities` and the request allowlist;
- SE2 proves terminal tools remain hidden before their preconditions; and
- scripted workflow fixtures retain their expected stage outcomes.

### S6 — Real-model evaluation, model policy, and rollout (P2)

Use evidence from S0-S5 to select defaults and decide whether model-specific policies are needed.

Implementation:

- Run the fixed small-model fixture matrix at 8K and 16K contexts with the same available local
  model/configuration used for S0.
- Repeat enough trials to distinguish deterministic failures from sampling variance; temperature 0,
  top-k 1, and a fixed seed remain the default comparison.
- Compare protocol pass rate, accepted-stage/task completion, context overflow, prompt high-water,
  prompt/generated tokens, tool calls, corrections, cache hits, wall time, and energy when available.
- Introduce a typed `ModelControlPolicy` only for differences supported by the matrix: context
  reserve, thinking mode/retry, maximum schema budget, and bounded result size.
- Keep optional stronger-model escalation explicit, local by default, stage-scoped, and disabled by
  default. Do not add automatic cloud escalation.
- Document defaults and reproduction commands in `docs/harness.md` and the evaluation report.

Acceptance:

- all deterministic control and small-model fixtures pass;
- protocol pass rate and false-completion behavior do not regress from S0;
- context overflow is zero for fixtures whose anchors fit the declared context;
- the large-review prompt reduction remains at least 40%;
- each targeted recovery fixture completes within its declared correction/invocation bound;
- real-model results improve at least two previously failing target fixtures without regressing any
  previously passing protocol fixture; and
- defaults are justified by the checked-in before/after report rather than framework benchmark
  claims.

## Acceptance fixture matrix

| ID | Stimulus | Required observation | Owner |
| --- | --- | --- | --- |
| CB1 | 5,000-line default `read_file` at 8K context | bounded whole-line slice and exact continuation | S1 |
| CB2 | multiple large tool results | deterministic receipts; raw events preserved | S1 |
| CB3 | compaction with active workflow artifacts | authoritative anchors unchanged | S1 |
| CB4 | anchors exceed usable context | no inference; explicit `context_limit` | S1 |
| RV1 | three large changed source/test files | focused review under budget; no duplicate full content | S2 |
| RV2 | new/delete/rename/binary delta | correct bounded representation for each status | S2 |
| RV3 | incomplete changed-path inspection | review submission remains gated | S2 |
| RB1 | large polyglot workspace graph | stable capped brief with valid IDs/hash | S2 |
| PG1 | third exact identical call | existing pre-execution block retained | S3 |
| PG2 | alternating calls with same unchanged failure | warn once, then block without a fourth retry | S3 |
| PG3 | retry after content/evidence change | retry allowed | S3 |
| PG4 | varied stale-patch attempts, same outcome | one no-progress sequence | S3 |
| PG5 | repeated deterministic read | cache hit with identical scoped evidence | S3 |
| AR1 | misspelled exposed tool | one bounded suggestion; no auto-execution | S4 |
| AR2 | invalid tool arguments | exact signature and corrective hint | S4 |
| AR3 | thinking consumes truncated turn | same-cap thinking-off action retry | S4 |
| AR4 | repeated truncation | global budgets remain authoritative | S4 |
| CL1 | all evidence current with two steps left | checkpoint then terminal-only closure | S5 |
| CL2 | missing deterministic precondition near cap | no terminal bypass | S5 |
| CL3 | stale fingerprint near cap | checkpoint names stale fact; submission hidden | S5 |
| SE1 | every workflow stage/state | schemas are deterministic authorized subsets | S5 |

## Milestone tracker

Status values are `not started`, `in progress`, `blocked`, or `complete`. A milestone is complete
only when its production path, deterministic proof, documentation, and semantic commit all exist.

| ID | Priority | Depends on | Status | Required proof | Evidence/commit |
| --- | --- | --- | --- | --- | --- |
| S0 | P0 | — | complete | deterministic metrics, fixture group, checked baseline | `test: baseline small-model agent control` |
| S1 | P0 | S0 | complete | exact preflight, receipts, bounded reads, CB1-CB4 | `feat: budget agent prompt context`; [S1 checkpoint](benchmarks/small-model-agent-s1.md) |
| S2 | P0 | S0, S1 | complete | brief, focused review, RV1-RV3, RB1, ≥40% reduction | `feat: focus workflow repository evidence`; [S2 checkpoint](benchmarks/small-model-agent-s2.md) |
| S3 | P1 | S0, S1 | complete | progress guard/cache, PG1-PG5 | `fix: stop no-progress agent tool loops`; [S3 checkpoint](benchmarks/small-model-agent-s3.md) |
| S4 | P1 | S0, S1 | complete | error envelopes/retry policy, AR1-AR4 | `fix: guide truncated agent actions`; [S4 checkpoint](benchmarks/small-model-agent-s4.md) |
| S5 | P1 | S0, S3, S4 | not started | closure/schema rules, CL1-CL3, SE1 | — |
| S6 | P2 | S1-S5 | not started | before/after matrix, justified defaults, rollout docs | — |

S2, S3, and S4 may proceed independently after their dependencies are complete. S5 must integrate
the recovery state from S3/S4. Do not begin S6 rollout until all deterministic milestones pass.

## Verification and commit policy

Use one reviewable semantic commit per milestone. Suggested commit shapes are:

- `test: baseline small-model agent control`
- `feat: budget agent prompt context`
- `feat: focus workflow repository evidence`
- `fix: stop no-progress agent tool loops`
- `fix: guide truncated agent actions`
- `feat: add deterministic workflow closure`
- `docs: record small-model reliability rollout`

For each milestone:

1. Mark the tracker row `in progress` before implementation.
2. Add or enable its deterministic fixtures with the production change.
3. Run focused tests for every touched module.
4. Run `cargo fmt --check` and `cargo test --all-targets` before the milestone commit.
5. Run `deno task test:web` only when web behavior or shared serialized behavior affects the web UI.
6. Update `docs/harness.md`, event/schema documentation, configuration scaffolding, and this tracker
   in the same milestone when applicable.
7. Record commands, results, fixture IDs, prompt metrics, and the semantic commit in the evidence log.
8. Mark the row `complete` only after reviewing the committed diff and evidence.

Before the overall goal is complete, run:

1. `deno task build:web`
2. `cargo test --all-targets`
3. `deno task test:web`
4. `cargo build --release --target aarch64-apple-darwin`
5. the full scripted `pb harness eval` suite
6. the documented S0/S6 local-model comparison
7. the narrow FlashMoe smoke only if a FlashMoe path changed

The existing untracked `.pb/` directory is user state and is never test scratch or plan evidence.

## Evidence log

Append one row per significant checkpoint. Keep raw bulky reports under `docs/benchmarks/` and link
them rather than pasting them here.

| Date | Milestone | Commit/worktree | Verification | Metrics/result | Remaining risk |
| --- | --- | --- | --- | --- | --- |
| 2026-07-15 | Plan | uncommitted document | architecture cross-check and `git diff --check` passed | implementation not started | baseline and numeric rollout decisions require S0 evidence |
| 2026-07-15 | S0 | `test: baseline small-model agent control` | `cargo fmt --check`; `cargo test --all-targets` (912 passed, 8 ignored); `deno task test:web` (47 passed); full and small-model scripted evals passed | scripted subset 4/4; Qwen2.5-Coder-7B exact protocol 0/4; maximum context 6.10%; no overflow | S1 context-pressure fixtures must prove budgeting and compaction |
| 2026-07-16 | S1 | `feat: budget agent prompt context` | `cargo fmt --check`; `cargo test --all-targets` (922 passed, 8 ignored); `deno task test:web` (47 passed); full and small-model scripted evals; CB1-CB4; FlashMoe and fixed llama.cpp preflight parity | scripted corpus 41/41 and subset 4/4; Qwen2.5-Coder-7B exact protocol remains 0/4; maximum context 6.80%; zero overflow | S2 must remove duplicate planning/review material without changing gates |
| 2026-07-16 | S2 | `feat: focus workflow repository evidence` | `cargo fmt --check`; `cargo test --all-targets` (927 passed, 8 ignored); `deno task test:web` (47 passed); full and small-model scripted evals; RV1-RV3 and RB1 | scripted corpus 41/41 and subset 4/4; constructed large-review prompt reduction ≥40%; 16,000-character brief/manifest and byte-exact focused inspection bounds | S3 must detect outcome-equivalent no-progress cycles and cache only deterministic reads |
| 2026-07-16 | S3 | `fix: stop no-progress agent tool loops` | `cargo fmt --check`; `cargo test --all-targets` (939 passed, 8 ignored); `deno task test:web` (47 passed); full and small-model scripted evals; PG1-PG5 | scripted corpus 41/41 and subset 4/4; A-B-A and varied stale edits pre-blocked with no fourth model retry; one scoped read-cache hit in the repeated-read fixture | S4 must make ordinary tool/schema and truncation failures cheaper for a small model to correct |
| 2026-07-16 | S4 | `fix: guide truncated agent actions` | `cargo fmt --check`; `cargo test --all-targets` (945 passed, 8 ignored); `deno task test:web` (47 passed); full and small-model scripted evals; AR1-AR4 | scripted corpus 41/41 and subset 4/4; typo and argument failures are non-executing structured envelopes; same-cap thinking-off recovery succeeds and global retry budgets stop deterministically | S5 must add closure help without exposing terminal actions before every current precondition is satisfied |

After each checkpoint, report:

```text
Active milestone:
Production behavior changed:
Safety/truthfulness invariants checked:
Deterministic fixtures:
Focused verification:
Prompt/model metrics:
Evidence recorded:
Remaining acceptance criteria:
Next bounded slice:
```

## Goal prompts

These prompts are deliberately milestone-bounded. Start a fresh goal for one prompt at a time. Do
not combine milestones merely because later work looks adjacent.

### Master completion goal

> Implement `docs/small-model-agent-reliability-plan.md` through S6. Preserve pb's deterministic
> stage capabilities, structured workflow transitions, fingerprint-bound checks/review, strict
> read-before-write behavior, truthful outcomes, and managed commits. First establish the S0
> baseline; then add exact prompt budgeting and deterministic compaction, focused repository/review
> evidence, outcome-aware recovery, structured action guidance, and deterministic closure; finally
> run the fixed real-model comparison and justify rollout defaults. Treat the plan's milestone table
> and evidence log as the source of progress. Complete one semantic milestone commit at a time,
> update documentation and deterministic fixtures with production behavior, never use `.pb/` as
> scratch, and do not declare completion until the final audit and required repository verification
> pass.

### S0 goal

> Complete only S0 in `docs/small-model-agent-reliability-plan.md`. Add deterministic prompt/context
> measurements and a small-model fixture group to the existing harness evaluator, preserve event
> compatibility, and capture the checked S0 baseline with one explicitly identified available local
> model. Do not change prompt compaction, tool results, gates, or model policy yet. Mark S0 in
> progress before edits, run its focused and repository tests, record exact evidence and metrics,
> make a semantic commit, and mark S0 complete only after reviewing the committed diff.

### S1 goal

> Complete only S1 in `docs/small-model-agent-reliability-plan.md`, assuming S0 is complete. Add an
> exact backend-aware prompt budget, pre-generation measurement, deterministic receipt compaction,
> bounded read/tool results with continuation hints, and explicit `context_limit` termination.
> Preserve authoritative anchors and raw events byte-for-byte, and do not weaken read, review,
> check, path, or fingerprint gates. Prove CB1-CB4 plus tokenizer preflight parity, update tracking
> evidence and docs, make one semantic commit, and stop after S1 is complete.

### S2 goal

> Complete only S2 in `docs/small-model-agent-reliability-plan.md`, assuming S0-S1 are complete.
> Introduce a deterministic capped repository brief and focused `inspect_change` review evidence;
> remove duplicate full graph/diff/file prompt material without removing durable checked evidence.
> Preserve changed-path read gating for new, deleted, renamed, binary, source, and test files. Prove
> RV1-RV3 and RB1, including the 40% constructed review-prompt reduction, update docs/evidence, make
> one semantic commit, and stop after S2 is complete.

### S3 goal

> Complete only S3 in `docs/small-model-agent-reliability-plan.md`, assuming S0-S1 are complete. Add
> a bounded post-result progress guard that recognizes exact, alternating, and same-outcome failures
> using content/evidence fingerprints, plus a narrowly scoped deterministic built-in read cache.
> Never cache commands, network, MCP, LSP, memory, or mutation tools, and never let cache replay earn
> new authority. Prove PG1-PG5, update tracking evidence, make one semantic commit, and stop after S3
> is complete.

### S4 goal

> Complete only S4 in `docs/small-model-agent-reliability-plan.md`, assuming S0-S1 are complete.
> Add bounded structured tool failures, schema-derived corrective hints, non-executing nearest-tool
> suggestions, and the same-cap thinking-off recovery attempt before token-cap growth. Charge every
> attempt to existing global budgets and preserve deterministic correction limits. Prove AR1-AR4,
> update events/docs/evidence compatibly, make one semantic commit, and stop after S4 is complete.

### S5 goal

> Complete only S5 in `docs/small-model-agent-reliability-plan.md`, assuming S0 and S3-S4 are
> complete. Add the deterministic two-step closure checkpoint, terminal-only last-step exposure when
> all preconditions are current, and conservative authorized-subset schema pruning for planning and
> review. Do not weaken a missing/stale precondition and do not add classifier-based implementation
> pruning. Prove CL1-CL3 and SE1 plus existing workflow fixtures, update docs/evidence, make one
> semantic commit, and stop after S5 is complete.

### S6 goal

> Complete only S6 in `docs/small-model-agent-reliability-plan.md`, assuming S1-S5 are complete. Run
> the fixed deterministic and real-model before/after matrix under the recorded S0 configuration,
> analyze protocol, task, context, recovery, runtime, and energy metrics separately, and introduce
> only model-control defaults supported by the results. Keep stronger-model escalation explicit and
> automatic cloud escalation disabled. Check in the comparison report and reproduction commands,
> run the full completion audit, update every tracker/evidence entry, make the final semantic commit,
> and declare the plan complete only if every acceptance criterion passes.

## Completion audit

- [ ] S0-S6 are complete with semantic commit and verification evidence.
- [ ] Existing harness protocol fixtures have no outcome regressions.
- [ ] Authoritative prompt anchors cannot be compacted or silently truncated.
- [ ] Backend-aware preflight prevents context overflow while preserving generation reserve.
- [ ] Large reads provide exact bounded continuation and preserve raw audit evidence.
- [ ] Planning uses a bounded repository brief tied to the complete graph hash.
- [ ] Code review no longer receives duplicate full diff and full file contents.
- [ ] Every changed text path must still earn fresh fingerprint-bound review-read evidence.
- [ ] Exact, alternating, and same-outcome no-progress loops are bounded.
- [ ] Read-cache hits cannot earn stale or broader evidence.
- [ ] Tool correction never auto-executes guessed prose, tools, or mutation arguments.
- [ ] Truncated reasoning recovery is bounded and fully charged to global budgets.
- [ ] Near-cap closure cannot bypass a missing or stale precondition.
- [ ] Exposed tools are always a deterministic subset of existing authority.
- [ ] S0/S6 reports are reproducible and separate protocol from artifact quality.
- [ ] `deno task build:web`, `cargo test --all-targets`, `deno task test:web`, release build, and
  required harness evaluations pass.
- [ ] Documentation describes shipped behavior and the tracker status matches the repository.
- [ ] `.pb/` user state was not read, changed, or used as scratch.
