# Internal harness

`pb harness` is a hidden CLI surface for exercising pb internals. It is intentionally omitted from
top-level help because it is a testing tool rather than a supported end-user workflow.

## Blocking agent runs

Run a complete agent task without starting `pb serve`, connecting to the Unix socket, or creating a
daemon session:

```bash
pb harness agent "Build a small Rust CLI that prints a greeting"
```

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
group. Contract-free invocations retain the existing profile gates and daemon/socket workflows.

Terminal output and the `session_summary` event distinguish `reached_final`, `handoff_outcome`,
`contract_status`, `verified_completed`, and `termination_reason`. A model final is handoff intent.
For top-level build/scout work the runtime then computes the task delta, selects affected checks,
starts only their executors, reuses only current evidence, offers one bounded model repair after a
failure, and creates or reuses a safe task-owned commit only after checks pass.

A contract-free `ready` or `no_change` handoff exits zero while retaining
`contract_status=unspecified` and `verified_completed=false`. A satisfied explicit contract exits
zero with `verified_completed=true`. Required mutation with no delta becomes
`contract_unsatisfied`; persistent check failure, missing executor, repeated repair failure, and an
unsafe required commit remain distinct nonzero `checks_failed`, `executor_unavailable`,
`repair_exhausted`, and `commit_blocked` outcomes. Step, parse, runtime-engine, and resource-limit
exits keep their existing structured reasons. Older stored summaries without these additive fields
remain readable with conservative defaults.

Deterministic recovery is bounded. Repeated parse, completion-gate, and identical-tool signatures
stop with `parse_loop` or `gate_loop` at their fixed thresholds, before another model or monitor
turn can reinterpret the same fact. If the ordinary last step establishes all required evidence,
the runtime emits `final_grace` events and permits exactly one 256-token generation with an empty
tool schema. Only one exact JSON final action is accepted. Executable contract facts may be
satisfied by deterministic handoff during that grace path; any remaining mutation, review,
commit, or cleanliness fact still rejects the final and cannot be converted into verified
completion by prose.

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
classification, and commit disposition/hash. Raw process output stays in event JSONL. A valid
no-change run does not produce the old misleading no-commit observation.

The workspace starts on `main` with one empty baseline commit. The agent receives the normal full
agent runtime, a local command backend rooted in the scratch repository, and the build profile by
default. Its changes therefore stay reviewable as commits on the generated task branch.

The journal is an initial audit aid, not a substitute for review. A supervising Codex run should:

1. Inspect P0/P1 observations and the raw event stream.
2. Review and test the committed workspace changes.
3. Fix clear harness or agent-runtime bugs in pb itself and commit those fixes.
4. Add ranked manual observations that automatic event classification could not infer.
5. Replace or extend the scaffold with a concrete plan for non-blocking improvements after the task
   succeeds.

FlashMoe inference, benchmark, and cache-clean utilities also live beneath the hidden harness, for
example `pb harness infer ...`. `infer` and `bench` accept
`--metal-working-set-limit-mib <MiB>` to lower the device-derived safety limit. The override can
only make the default policy stricter. `--resource-summary` prints the opt-in JSON resource ledger;
normal runs keep it disabled and emit tracing only for high-water changes, pressure recovery, or a
resource-limit abort.

The agent/evaluation handoff plumbing does not apply to `pb harness infer`, `pb harness bench`, or
`pb harness cache-clean`: their arguments, dispatch paths, output, and exit behavior are unchanged.

## Control evaluation

Run the deterministic, model-free control suite with:

```bash
pb harness eval --jsonl harness-eval.jsonl
```

The command writes one schema-v2 JSON object per fixture and prints a compact table covering valid
actions, named-check compliance, false completion, recovery loops, turns, latency, tokens, energy,
and termination. JSONL additionally records selected components/checks, runtime executions versus
model `run_check` calls, reuse and dependency skips, started/avoided executors, team messages,
repair turns, no-change, commit disposition, and output fingerprints. Without `--jsonl`, JSONL
goes to stdout and the table goes to stderr so the machine stream stays parseable. The scripted
report contains no timestamps, scratch paths, or nondeterministic commit IDs and is stable enough
to diff directly; real-model records retain the resulting commit hash. A protocol mismatch exits
non-zero. `artifact_quality` remains separate from protocol scoring. Schema-v1 fixture/result data
is rejected explicitly rather than compared under changed metric meanings.

Real-model matrices are opt-in and must name the local model explicitly:

```bash
pb harness eval --model model.gguf --model-dir /path/to/models \
  --max-tokens 512 --ctx-size 32768 --temperature 0 --top-k 1 --seed 0 \
  --jsonl model-eval.jsonl
```

Every real-model record repeats the backend, model, resolved model directory, token/context/thread/
GPU settings, sampling values, seed, FlashMoe resource-policy version, normalized workspace-config
hash, and executor policy. One loaded engine is reused across the corpus. FlashMoe evaluation
refuses to start if the versioned bounded-resource policy is inactive.

Failed `apply_patch` checks now return a bounded mismatch diagnostic with the target file, expected
hunk line, a few old-side patch lines, and numbered current content around that location. Large
lines and the full diagnostic are capped; binary targets are identified without dumping bytes. The
check path never mutates the file, and the correction recommends `edit_file` for exact current text
or `replace_file` when the target has drifted substantially.
