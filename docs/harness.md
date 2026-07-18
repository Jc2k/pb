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

### Acceptance contracts

An optional trusted JSON contract makes completion externally verifiable:

```bash
pb harness agent --contract docs/harness-contract-v1.example.json \
  "Build the requested game and satisfy the supplied contract"
```

Version 1 can require a final mutation, restrict changed paths, define named checks, require
semantic commits, require a clean worktree, and state the exact paths/checks a review must inspect.
See `docs/harness-contract-v1.example.json` for the complete shape.

The agent receives `run_check(id)` only when the contract defines checks. The check command is
trusted caller input; the model supplies only its ID. Each run records its exit status, bounded
stdout/stderr, duration, timeout state, executor, command/input/output fingerprints, and evidence
source. An exact canonical `run_command` invocation shares the same evidence ledger; unrelated
commands do not satisfy named checks. Evidence becomes stale when its declared inputs,
dependencies, command, executor, or outputs change. Contracts are parsed before model loading and
their checks compile into the same workspace check graph used by deterministic handoff.

An optional trusted topology can describe polyglot components, separate executors, generated
outputs, and check dependencies without copying configuration into the scratch repository:

```bash
pb harness agent --workspace-config /absolute/path/workspace.toml \
  "Update the API and its generated web assets"
```

The file is parsed, validated, and normalized before model loading. Its canonical source path,
SHA-256, and executor policy are recorded in run metadata. Without it, the harness uses a
synthetic repository-wide component and local project executor. Workspace configuration defines
how work is checked; the v1 contract remains the separate source of task-specific acceptance facts.

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
zero with `verified_completed=true`. Required mutation with no delta becomes
`contract_unsatisfied`; persistent check failure, missing executor, repeated repair failure, and an
unsafe required commit remain distinct nonzero `checks_failed`, `executor_unavailable`,
`repair_exhausted`, and `commit_blocked` outcomes. Step, parse, runtime-engine, and resource-limit
exits keep their existing structured reasons. A prompt whose authoritative anchors and exposed
schemas cannot fit is rejected before inference with the distinct `context_limit` reason. Older
stored summaries without these additive fields remain readable with conservative defaults.

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

Deterministic recovery is bounded. Repeated parse, artifact validation, identical-tool, plan-cycle,
repair-cycle, invocation, token, stage-step, and advisory failures stop with explicit outcomes
before another model turn can reinterpret the same fact. Strict workflow stages never use the old
prose-final grace path. Literal `REVIEW PASS` text and model-requested commits have no current
workflow meaning; code-review credit comes only from the structured, fingerprint-bound review
artifact. The former profile gate/final-grace/handoff behavior is retained only in memory for an
actually restored persisted request that predates conversation intent, so old sessions remain
readable without letting a new request opt into weaker control by omitting fields.

Each run creates a persistent scratch root under the system temporary directory unless
`--scratch-dir` selects a new path. The layout is:

```text
pb-harness-.../
├── workspace/       # isolated git repository used by the agent
├── task-baseline.json # immutable original task baseline
├── adoptions.jsonl  # explicit provenance for resumed external changes
├── events.jsonl     # cumulative compatibility AgentEvent stream
├── journal.md       # latest-run compatibility view
├── run-index.jsonl  # append-only started/finished run records
└── runs/<run-id>/
    ├── events.jsonl # immutable event stream for this invocation
    └── journal.md   # final journal, or running recovery journal if interrupted
```

The harness allocates the run ID and writes the per-run `running` journal plus a `started` index
record before model loading. Events are flushed to both streams. Final journals are atomically
replaced and a `finished` index record captures the structured outcome. Resuming a scratch root
creates a new run directory and never rewrites a prior run; any partial dual-write failure is
surfaced instead of reporting verified completion.

Resume restores the original task baseline, captures a new invocation baseline, and treats earlier
scratch work as part of the same task. Prior successful evidence is restored from cumulative
events and reused only while all current fingerprints match; externally edited resumed content is
appended to `adoptions.jsonl`.

Finished journals and run-index records include the handoff outcome, both baseline identifiers,
workspace-config provenance, affected components, planned/executed/reused/failed/skipped checks,
output fingerprints, executor starts, repair turns, team feedback evidence, no-change
classification, commit disposition/hash, and the publication-ready evidence digest and sanitized
repository remote. The full evidence bundle remains in the durable workflow checkpoint and binds
the commit to its accepted plan, fresh code review, and selected check evidence. No publisher or
network operation is invoked by the harness. Raw process output stays in event JSONL. A valid
no-change run does not produce the old misleading no-commit observation.

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

The automatic missing-commit observation applies only to a `ready` delivery. Engine errors,
exhausted repairs, failed checks, and other incomplete outcomes retain their actual terminal cause
without being mislabeled as completed work.

FlashMoe inference, benchmark, and cache-clean utilities also live beneath the hidden harness, for
example `pb harness infer ...`. `infer` and `bench` accept
`--metal-working-set-limit-mib <MiB>` to lower the device-derived safety limit. The override can
only make the default policy stricter. `--resource-summary` prints the opt-in JSON resource ledger;
normal runs keep it disabled and emit tracing only for high-water changes, pressure recovery, or a
resource-limit abort. `infer --no-thinking` asks a checkpoint's chat template to suppress emitted
reasoning; it cannot be combined with `--raw`, which bypasses the chat template entirely.

The agent/evaluation handoff plumbing does not apply to `pb harness infer`, `pb harness bench`, or
`pb harness cache-clean`: their arguments, dispatch paths, output, and exit behavior are unchanged.

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
hashes. The bounded open-weight protocol matrix and its known limitations are recorded in
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
Each model-invocation record
includes an optional backward-compatible context snapshot covering capacity, generation reserve,
prompt high-water utilization, preflight/backend token counts, safety margin, message/schema size,
thinking mode, retry reason, and the compaction/cache/closure counters introduced by later
milestones. Evaluation summaries count `thinking_off_truncation_retries` and
`larger_cap_truncation_retries` separately. Runtime
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
may grow the cap once within the request limit; existing parse-loop thresholds remain authoritative.
Every attempt reserves an invocation and records generated tokens before another retry is allowed,
so the global invocation and token budgets cannot be bypassed by recovery.
An action-only retry remains within its original stage step: durable stage-step accounting advances
once per visible `step_started`, while every backend attempt is still charged to the separate model
invocation and generated-token limits.

Each compatibility edit action ends its model turn. The prompt forbids invented tool results, and
pb does not replay fenced compatibility actions followed by fabricated transcript markers into the
next model context. Only the validated first action is executed; its real result and harness content
fingerprint determine the next state.

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

During a strict workflow stage, pb can recover a complete unwrapped JSON object, either plain or in
one JSON code fence, when the exposed schemas identify exactly one tool. The sole special-case tie
is `write_file` versus `replace_file`, selected from whether the bounded target currently exists.
The recovered action still goes through normal path, policy, schema, capability, and artifact
validation. Prose, partial JSON, arrays, and other ambiguous objects are not coerced. Implementation
and repair also retain their edit tools after a rejected prose final instead of exposing only a
submission that cannot truthfully advance.

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
