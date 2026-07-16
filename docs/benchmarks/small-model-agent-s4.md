# Small-model agent S4 action-recovery checkpoint

Captured: 2026-07-16

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

S4 makes ordinary action failures cheaper for a small model to correct without weakening tool
exposure, policy, workflow, or global budget boundaries. This checkpoint is deterministic and
model-free; real-model effectiveness and rollout defaults remain S6 work.

## Production behavior

- Built-in failures use a valid JSON `tool_failure` envelope capped at 2,400 characters. Stable
  reason codes distinguish unknown or unexposed tools, argument errors, unmet preconditions,
  read-before-write/target failures, policy or approval denial, timeout/unavailability, and generic
  execution failure.
- Valid signatures and top-level argument checks come from the exact schema exposed on that model
  turn. Missing, unknown, and wrongly typed arguments are rejected before execution; values are
  never guessed, coerced, or added.
- A bounded edit-distance hint may suggest one nearest exposed tool. The original call remains a
  failed, zero-runtime result and is never rewritten or executed.
- A truncated non-action gets one same-cap retry with thinking disabled and an action-only
  correction. A truncated workflow terminal attempt retains its existing terminal-only recovery
  schema. Only one later cap-growth retry is permitted.
- Every retry reserves a global model invocation and records generated tokens. The optional
  invocation context now records `retry_reason` as `thinking_off_after_truncation` or
  `larger_token_cap_after_truncation`; evaluation summaries count both independently.
- The existing exact parse and no-progress thresholds still stop repeated failures.

## Deterministic acceptance

| Fixture | Proof | Result |
| --- | --- | --- |
| AR1 | `read_fil` receives exactly one `unknown_tool` correction suggesting exposed `read_file`; the invalid call has zero runtime, is not executed, and the next exact read succeeds | pass |
| AR2 | `write_file` without `content` returns `missing_argument` and `write_file(path: string, content: string)`; the target path is not created and no value is invented | pass |
| AR3 | A truncated prose turn is followed by a valid read at the identical 256-token cap with thinking disabled; invocation context records the retry reason and the run reaches final | pass |
| AR4 | A two-invocation budget stops before a third recovery attempt after charging both generated tokens; a one-token budget stops before the same-cap retry | pass |
| Envelope bound | A 10,000-character tool name, error chain, signature, and hint serialize as valid JSON below the declared 2,400-character cap | pass |
| Existing bounds | Repeated truncated parse signatures still terminate at the structured threshold, and truncated terminal submission recovery still exposes only its required workflow tool | pass |

## Reproduction

```bash
cargo test agent_tool_errors::tests
cargo test 'agent_core::tests::ar'
cargo test agent_core::tests
cargo test events::tests
cargo test harness_eval::tests
cargo run --quiet -- harness eval --jsonl /tmp/pb-harness-s4-scripted.jsonl
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s4-scripted.jsonl
```

The scripted control corpus remains 41/41 and its small-model subset remains 4/4. S4 does not
change their expected protocol outcomes. The checked S0/S1 real-model run is not repeated here
because rollout selection is explicitly deferred to the fixed S6 before/after matrix.

Milestone-wide verification passed `cargo fmt --check`, `cargo test --all-targets` (945 passed, 8
ignored), and `deno task test:web` (47 passed).
