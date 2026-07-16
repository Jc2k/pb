# Small-model agent S3 progress-recovery checkpoint

Captured: 2026-07-16

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

S3 bounds repeated failure strategies across non-identical calls and reuses deterministic read
results without changing stage authority. This checkpoint is deterministic and model-free;
real-model effectiveness remains an S6 measurement.

## Production behavior

- A run-local progress guard keeps at most eight failed outcomes, keyed by tool family, normalized
  call/outcome, repository content, and harness evidence state.
- The second failure without a state transition emits one actionable correction. The existing
  exact-call guard still blocks a third consecutive identical action; the new guard pre-blocks an
  unchanged A-B-A proposal and a third stale workspace-edit strategy, or stops after a third
  outcome-equivalent failure before another model turn.
- Content or fingerprint-bound evidence progress resets the failure sequence. Successful tool
  outcomes also clear earlier failures.
- A 64-entry FIFO cache stores only successful `read_file`, `glob`, `ripgrep`/`search`, and
  `git_log` results. Its key binds normalized arguments, repository and Git control fingerprints,
  request/policy scope, context-dependent result bounds, and exact `read_file` target bytes. Glob
  and search candidates are path-sorted before a result limit is applied.
- A cache hit returns the byte-identical result with zero tool runtime/energy and replays only the
  original read-path, contract-read, and legacy review-read effects. It cannot earn check, write,
  transition, or unrelated review authority.
- Commands, network, MCP, LSP, memory, `session_changes`, `inspect_change`, and mutation tools are
  explicitly excluded.

## Deterministic acceptance

| Fixture | Proof | Result |
| --- | --- | --- |
| PG1 | Three identical missing-file reads preserve the existing pre-execution block: three model invocations, two executed tool calls, no content mutation, and one scripted completion unused | pass |
| PG2 | Missing read A, invalid search B, then A is warned once and the proposed A is blocked before execution: three model invocations, two executed tool calls, and no fourth retry | pass |
| PG3 | A successful evidence read changes the evidence fingerprint, so the same earlier missing-file operation is allowed again and the run reaches its scripted final | pass |
| PG4 | Different stale patch arguments normalize to one stale-context outcome; after a read and two failed patches, the third patch proposal is blocked with the repository fingerprint unchanged | pass |
| PG5 | A repeated deterministic read emits the exact original result and one cache-hit metric; content (including an explicitly read ignored file) and policy-scope changes invalidate the entry | pass |
| Scope | Cache-effect replay excludes named checks and write state; non-cacheable command/network/MCP/LSP/memory/review/mutation families are asserted explicitly | pass |

The focused integration fixtures additionally assert that every pre-execution block consumes zero
tool runtime and cannot mutate the repository. A different recovery read remains allowed after two
equivalent read failures, so the correction's path/query alternative is actionable rather than
being hidden by a family-wide block.

## Reproduction

```bash
cargo test agent_progress::tests
cargo test pg
cargo test agent_core::tests
cargo test harness_eval::tests
cargo run --quiet -- harness eval --jsonl /tmp/pb-harness-s3-scripted.jsonl
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s3-scripted.jsonl
```

The scripted control corpus remains 41/41 and its small-model subset remains 4/4. The
`repeated_blocked_action` fixture now reports one deterministic read-cache hit; the other three
small-model fixtures report zero. The checked S0/S1 real-model run is not repeated here because S3
does not select rollout defaults; the fixed before/after model matrix remains S6 work.

Milestone-wide verification passed `cargo fmt --check`, `cargo test --all-targets` (939 passed, 8
ignored), and `deno task test:web` (47 passed). The first sandboxed Rust run exposed only the suite's
expected macOS meter-lock and Application Support write restrictions; the approved host-capability
rerun passed in full.
