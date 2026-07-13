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
