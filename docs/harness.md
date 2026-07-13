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
stdout/stderr, duration, timeout state, and the current worktree content fingerprint. `run_command`
is still available for exploration but never satisfies a named check. A successful check or review
becomes stale after any later content mutation, and finalization reports all currently missing
contract facts together. Contracts are parsed and normalized before model loading.

An empty `allowed_paths` list means unrestricted paths. Otherwise built-in write tools reject a
path outside the list before mutation, while `run_command` and final validation detect indirect
forbidden changes. Check timeouts are limited to one hour and terminate the local command process
group. Contract-free invocations retain the existing profile gates and daemon/socket workflows.

Terminal output and the `session_summary` event distinguish four independent facts:
`reached_final`, `contract_status`, `verified_completed`, and `termination_reason`. A
contract-free final keeps the historical zero exit behavior, but is reported as
`contract_status=unspecified` and `verified_completed=false`. A final rejected by a supplied
contract is recorded as `contract_unsatisfied` and exits non-zero; only a final with a satisfied
contract is `verified_completed=true`. Step, parse, runtime-engine, and resource-limit exits use
their own structured termination reasons. Older stored summaries without these fields remain
readable with conservative defaults.

Each run creates a persistent scratch root under the system temporary directory unless
`--scratch-dir` selects a new path. The layout is:

```text
pb-harness-.../
├── workspace/    # isolated git repository used by the agent
├── events.jsonl  # complete typed AgentEvent stream
└── journal.md    # ranked observations, committed fixes, and review-plan scaffold
```

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
