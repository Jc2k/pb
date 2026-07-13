# Harness reliability improvement plan

Status: active

This document turns the July 2026 `pb harness agent` supervision findings into an
implementation plan. It is the tracking source for the goal described at the end of this
document. Update the status and evidence columns as work lands; do not use generated artifact
quality as a proxy for harness correctness.

## Outcome

The daemon-free harness should be able to state, with durable evidence, whether an agent:

1. produced a final response;
2. satisfied an explicit task contract after its last content mutation; and
3. completed the required review and checks.

It should preserve every run for audit, stop deterministic failure loops without spending more
inference, and provide a repeatable control-plane evaluation that does not depend on a model making
high-quality application code. FlashMoe-backed runs must also have observable and bounded Metal
resource use before they are used for long experiments.

This plan does **not** try to make a small local model produce Codex-quality CSS, JavaScript, or
product design. An artifact defect is a harness finding only when pb skipped, misclassified, or
misreported evidence required by the task contract.

## Baseline evidence

The primary supervised experiment accumulated 71 agent starts, 317 tool calls, 308 tool results,
129 corrections, 39 errors, 26 final actions, and 21 session summaries. It demonstrated that the
harness can preserve useful work and reject some malformed actions, but also exposed these
control-plane problems:

- a generic successful command or irrelevant read can count as review evidence;
- `reached_final` is printed as `completed=true` even though no external acceptance contract exists;
- resumed runs append events but overwrite the human journal;
- gate corrections and monitor extensions can spend more large-model turns on a deterministic
  blocker;
- a final action that needs one more gate step can be lost at the step boundary;
- artifact quality and model protocol compliance are not measured separately;
- patch mismatch feedback does not give enough current-file context for recovery; and
- Metal allocation failures report a snapshot, but there is no run-level resource ledger,
  high-water evidence, or pre-failure limit.

## Invariants

- Keep `pb harness` hidden and preserve the existing daemon, socket, web, and queue paths.
- Keep `pb harness agent "<task>"` usable without a contract. Such a run may reach a final response,
  but it must not be labelled externally verified.
- Do not infer acceptance requirements from arbitrary task prose. Required checks and paths are
  structured, trusted input supplied by the harness caller.
- Evidence belongs to a workspace content fingerprint. A later content mutation invalidates it.
- New event fields and variants must remain readable by older stored sessions where practical;
  additive optional fields are preferred.
- Deterministic transcript tests come before real-model reruns. A model run is warranted only when
  the property under test cannot be exercised by a scripted completion engine.
- Do not add hidden FlashMoe environment switches. Update
  `docs/flashmoe-architecture-parity-plan.md` before changing the backend ownership or resource
  model.
- Do not touch project-local `.pb/` state while implementing or evaluating this plan.

## Target contract

Add `--contract <path>` to `pb harness agent`. Version 1 is a JSON document, parsed and validated by
the harness before model loading. A representative contract is:

```json
{
  "version": 1,
  "mutation": "required",
  "allowed_paths": ["game.js", "game-logic.test.mjs"],
  "checks": [
    {
      "id": "logic",
      "command": "deno test game-logic.test.mjs",
      "cwd": ".",
      "required": true,
      "timeout_seconds": 60
    }
  ],
  "commit": { "required": true, "semantic": true },
  "review": {
    "required": true,
    "read_paths": ["game.js"],
    "check_ids": ["logic"]
  },
  "workspace_clean": true
}
```

The command text is trusted supervisor input, not model input. The model invokes a native
`run_check` tool with only the check ID; it cannot substitute a cheaper command and receive credit.
Each result records the check ID, exit status, bounded output, duration, and current worktree content
fingerprint. A successful command run before the last mutation is stale. `run_command` remains
available for exploration but never satisfies a named check.

At finalization, the harness validates allowed paths, required mutations, named checks, commit
policy, review evidence, and cleanliness against the current fingerprint. The review sub-agent gets
the same normalized contract and can only satisfy the explicitly named read/check requirements.
Contract validation should report all missing facts in one correction rather than uncovering them
one turn at a time.

Without `--contract`, current profile gates remain as compatibility guards. The outcome is
`contract_status=unspecified`, never `verified_completed=true`. If a contract is present and
unsatisfied, the command exits non-zero.

## Workstreams

### H0 — Deterministic evaluation foundation (P0)

Add a scripted `CompletionEngine` test driver around `run_agent_steps`. Keep fixtures small and
human-readable. The initial corpus must cover:

- prose containing multiple future tool actions executes only the first accepted action;
- a false final claim after inspection only;
- irrelevant `read_file` and `run_command` calls;
- a successful check followed by a content mutation;
- a repeated blocked or gate-rejected action;
- a valid final action arriving exactly at the ordinary step limit; and
- a review pass that omits a required path or check.

Record deterministic metrics for action validity, gate corrections, false completion, executed
checks, model invocations, and termination reason. These fixtures are the regression suite for H1,
H2, H4, and H5; they must not load llama.cpp, FlashMoe, Metal, or a container runtime.

Likely files: `src/agent_core.rs` tests and a focused fixture module or fixture directory under
`tests/`.

Acceptance:

- the corpus runs in `cargo test` on non-macOS CI;
- each fixture has one explicit expected outcome; and
- a checked-in baseline report captures current behavior before completion semantics change, while
  target assertions are enabled as the corresponding milestones land.

### R0 — Bound FlashMoe Metal resources (P0)

Start this after H0 and complete it before using FlashMoe for long harness evaluations.

Add an RAII resource ledger owned by `MetalExecutionContext`. It should distinguish resident model
resources, reusable pool buffers, transient expert staging, recurrent state, and in-flight command
resources. Track live bytes/counts and high-water marks at allocation, pool transfer, purge/release,
command submission, and completion. Sample Metal's `currentAllocatedSize` and
`recommendedMaxWorkingSetSize` at token boundaries and on pressure events.

Define an explicit fail-safe from the device-reported recommended working set, with headroom for
driver allocations. The limit must be a documented runtime policy or visible CLI argument, not a
hidden environment toggle. Before an allocation that would exceed the limit, drain idle pooled
buffers, resample once, and abort with a structured diagnostic if it still cannot fit. Keep the
existing allocation-failure diagnostic as the last line of defence.

Emit lightweight tracing only on high-water changes, pressure recovery, and abort; do not add
per-buffer or per-token info logging in production. Detailed snapshots may be enabled explicitly for
the hidden harness benchmark.

Likely files: `src/inference/flashmoe/metal.rs`, the token/runtime boundary in
`src/inference/flashmoe/runtime.rs`, harness benchmark arguments in `src/lib.rs`, and
`docs/flashmoe-architecture-parity-plan.md`.

Acceptance:

- unit tests prove every ledger ownership transition balances on success and injected error paths;
- a release-mode, 128-generated-token fixed-prompt soak leaves transient and in-flight ledger counts
  at zero on every token boundary, remains below the declared cap, and exits before OS allocation
  failure when the cap is intentionally constrained;
- the normal narrow FlashMoe smoke still exits 0 with a sensible token; and
- detailed resource tracing is opt-in and the default throughput passes the comparison protocol
  below.

### H1 — Explicit acceptance contracts (P1)

Introduce a serialized harness contract type and a smaller normalized runtime type on
`AgentRequest`. Add `run_check` only when named checks exist. Store evidence by check ID and content
fingerprint in `GateState`; content mutation invalidates check and review evidence. Final validation
must use the worktree fingerprint already used to invalidate review results.

Allowed paths are validated from the final diff and untracked-file set. Built-in write tools should
reject an out-of-contract path immediately. Because `run_command` may mutate files, compare the
before/after path set and issue a contract correction if forbidden paths changed; do not silently
award mutation evidence.

Likely files: `src/lib.rs`, `src/harness.rs`, `src/agent_core.rs`, `src/events.rs`, and
`docs/harness.md`.

Acceptance tests:

- `git status` or a generic successful command cannot satisfy a named check;
- failed named checks do not count;
- a successful named check at the current fingerprint counts;
- a later mutation makes that result stale;
- an irrelevant read cannot satisfy a required read path;
- forbidden changed paths keep the contract unsatisfied; and
- a review pass cannot satisfy the parent contract without its named evidence.

### H2 — Truthful run outcomes (P1)

Separate these concepts in `StepRunOutcome`, `AgentRunResult`, events, journals, and CLI output:

- `reached_final`: the model emitted an accepted final action;
- `contract_status`: `unspecified`, `unsatisfied`, or `satisfied`; and
- `verified_completed`: a final action was reached and the explicit contract is satisfied.

Add a structured termination reason such as `final`, `step_limit`, `gate_loop`, `parse_loop`,
`contract_unsatisfied`, `resource_limit`, or `engine_error`. Preserve serde compatibility with old
events using optional/default fields. Do not use the old `completed=true` label for a contract-free
run.

Acceptance:

- an uncontracted final reports `reached_final=true`, `contract_status=unspecified`, and
  `verified_completed=false` while retaining the current compatibility exit behavior;
- a contracted but unsatisfied final exits non-zero and is journalled accurately;
- a satisfied contracted final reports verified completion and exits 0; and
- stored older `SessionSummary` events still deserialize and render.

### H3 — Immutable per-run journals (P1)

Change the scratch layout to preserve every invocation:

```text
scratch/
├── workspace/
├── events.jsonl             # cumulative compatibility stream
├── journal.md               # latest-run compatibility view
├── run-index.jsonl          # append-only run metadata
└── runs/<run-id>/
    ├── events.jsonl         # immutable events for this invocation
    └── journal.md           # immutable final or interrupted journal
```

Generate a collision-resistant run ID before model loading. Write the per-run journal as `running`
first and append a `started` index record before loading the model, then replace the journal
atomically and append a `finished` record on completion. The event sink writes both the per-run
stream and the existing cumulative stream. If one write fails, surface the partial audit failure
rather than presenting the run as verified.

Acceptance:

- two resumed invocations preserve two independent journals and event streams;
- the cumulative stream contains both runs and `journal.md` identifies the latest run;
- an interrupted run remains discoverable from the index; and
- a prior run's files are never rewritten by resume.

### H4 — Bounded deterministic recovery and final grace (P2)

Replace repeated gate/blocked corrections with a keyed deterministic failure tracker. Once the same
failure signature reaches its threshold, stop with `gate_loop` or `parse_loop`; do not ask the model
monitor whether more steps are useful. The monitor may summarize ambiguous progress, but it cannot
grant completion or override a structured contract fact.

When the ordinary step budget ends immediately after all required evidence is current, allow one
small final-only grace turn. No tools are available, its token cap is small, and it accepts only an
exact final action. The grace turn never bypasses an unsatisfied contract. Emit its use and result as
events.

Acceptance:

- a repeated deterministic blocker causes no model invocation after the configured threshold;
- monitor text cannot change a structured gate result;
- a final-only grace fixture completes without reopening tools; and
- an unsatisfied contract cannot become verified through the grace path.

### H5 — Model-control evaluation command (P2)

Add a hidden `pb harness eval` command that runs the H0 fixture corpus against either the scripted
engine or an explicitly selected local model. Produce machine-readable JSONL and a concise table
covering valid-action rate, named-check compliance, false completion, recovery-loop incidence,
turns, latency, tokens, and energy when available.

Keep artifact-quality observations in a separate field and out of the pass/fail score. The default
CI mode uses only the scripted engine. Real-model matrices are opt-in, record model/config/seed, and
must refuse FlashMoe runs until the R0 resource policy is active.

Acceptance:

- the scripted suite is deterministic and CI-safe;
- two reports can be compared without parsing terminal prose;
- model protocol regressions fail independently of generated artifact quality; and
- every real-model result records enough configuration to reproduce the run.

### H6 — Patch mismatch recovery (P3)

When `apply_patch` rejects a hunk, include the file, expected location, and a small bounded excerpt
of the current nearby content in the correction. Recommend `edit_file` or `replace_file` when the
target has drifted substantially. Do not mutate the file during diagnosis and do not dump large file
contents into the prompt.

Acceptance:

- a deterministic stale-hunk test returns useful nearby context;
- the rejected patch leaves the workspace fingerprint unchanged; and
- feedback remains bounded for large files and binary paths.

## Milestones and dependency order

| ID | Priority | Depends on | Status | Required proof | Evidence |
| --- | --- | --- | --- | --- | --- |
| H0 | P0 | — | complete | Scripted fixture corpus and baseline metrics | `test: add deterministic harness control fixtures`; 7 fixtures; 705 Rust tests passed, 7 ignored |
| R0 | P0 | H0 | complete | Ledger tests, constrained abort, release soak, throughput comparison | `fix: bound FlashMoe Metal resources`; `docs/benchmarks/harness-r0-after.md`; 10×32 and 128-token soaks passed; 1.661 tok/s median |
| H1 | P1 | H0 | complete | Contract parser, `run_check`, fingerprinted evidence tests | `feat: add harness acceptance contracts`; deterministic named-check, stale-evidence, path, review, and timeout tests |
| H2 | P1 | H1 | complete | Outcome/event/CLI compatibility tests | `fix: distinguish final and verified harness outcomes`; scripted uncontracted, rejected-contract, satisfied-contract, engine-error, CLI-exit, and legacy-event tests |
| H3 | P1 | H0 | complete | Two-run resume and interruption tests | `feat: preserve per-run harness journals`; dual event streams, append-only run index, atomic compatibility journal, resume/interruption tests |
| H4 | P2 | H0, H1, H2 | complete | Loop-stop and final-grace transcript tests | `fix: bound harness recovery loops`; keyed parse/gate/tool thresholds, no-monitor contract proof, and capped tool-free grace tests |
| H5 | P2 | R0, H0-H4 | proposed | Deterministic eval report and optional model mode | — |
| H6 | P3 | H0 | proposed | Bounded stale-patch diagnostic tests | — |

H1 and R0 may proceed in parallel after H0, but long FlashMoe model evaluations remain blocked until
R0 passes. H3 and H6 are independent after H0. Update a row to `in progress`, `blocked`, or
`complete` and link the commit/test evidence before starting dependent work.

## Commit and verification policy

Use one reviewable semantic commit per milestone. Expected commit shapes are:

- `test: add deterministic harness control fixtures`
- `feat: add harness acceptance contracts`
- `fix: distinguish final and verified harness outcomes`
- `feat: preserve per-run harness journals`
- `fix: bound harness recovery loops`
- `feat: add harness control evaluation`
- `fix: improve harness patch mismatch feedback`
- a separate semantic FlashMoe commit with the architecture document updated in the same milestone

For every milestone, run focused Rust tests first. Before declaring the goal complete, run:

1. `deno task build:web`
2. `cargo test --all-targets`
3. `deno task test:web`
4. `cargo build --release --target aarch64-apple-darwin`
5. the scripted `pb harness eval` suite
6. the narrow FlashMoe smoke required by `AGENTS.md` after backend changes
7. the documented bounded FlashMoe soak and before/after throughput comparison

Capture the R0 performance baseline before its implementation. Use the same model files, prompt,
sampling parameters, token count, machine power mode, and release target for before/after runs. Run
three warmups followed by at least seven measured trials and compare median generated tok/s and
elapsed time. A median tok/s regression above 3% that is also larger than the combined median
absolute deviation fails the milestone and must be investigated. Keep the raw benchmark summaries
as tracking evidence. Do not claim an improvement unless the same protocol supports it; R0's goal is
bounded, observable resource use with no material throughput regression.

## Goal package

Objective:

> Implement `docs/harness-improvement-plan.md` end to end. Build the deterministic control fixture
> foundation; add explicit structured acceptance contracts with native named checks and
> post-mutation evidence; distinguish final responses from verified completion; preserve immutable
> per-run journals; make recovery and final closure deterministic and bounded; add the model-control
> evaluation command and bounded patch diagnostics; and implement and prove bounded FlashMoe Metal
> resource accounting according to the architecture plan. Preserve daemon/socket workflows and
> contract-free harness compatibility. Use deterministic fixtures before model reruns, keep artifact
> quality separate from harness correctness, update tracking evidence as milestones land, make
> semantic commits, and run the full repository, release, FlashMoe smoke, soak, and performance
> verification before completion.

Completion audit:

- [ ] H0 through H6 and R0 are marked complete with commit and test evidence.
- [ ] Contracted runs cannot be verified when a required post-mutation check was skipped.
- [ ] Uncontracted or merely final runs are not labelled verified.
- [ ] Resume preserves immutable audit files for every invocation.
- [ ] Deterministic blockers stop without unbounded inference.
- [ ] Evaluation separates protocol compliance from artifact quality.
- [ ] FlashMoe resource use is bounded and default release throughput has no material regression.
- [ ] Existing daemon/socket agent workflows still pass their tests.
- [ ] Documentation and the FlashMoe architecture plan describe the shipped behavior.
- [ ] Every implementation change is included in a semantic commit and the worktree contains no
      unintended changes.
