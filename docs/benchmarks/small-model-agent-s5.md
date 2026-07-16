# Small-model agent S5 workflow-closure checkpoint

Captured: 2026-07-16

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

S5 helps a small model finish a strict stage near its step cap without granting new authority or
hiding a fact it still must prove. This checkpoint is deterministic and model-free; real-model
effectiveness and rollout defaults remain S6 work.

## Production behavior

- The final two ordinary turns of planning, plan revision, plan review, and code review receive a
  bounded JSON `workflow_closure_checkpoint`. It records remaining turns, current and expected
  content fingerprints, the exact terminal tool/signature, eligibility, and missing deterministic
  preconditions.
- Checkpoints are prompt overlays refreshed from current state. They emit durable correction events
  for audit but do not become evidence or stale transcript authority. Each model-invocation context
  records one closure use alongside the exact exposed tool count and schema token measurement.
- A harness-only, non-serialized expected fingerprint binds planning/review closure to the stage
  snapshot. A concurrent or resumed stale snapshot hides the terminal before generation and again
  at execution.
- `ToolExposureState` intersects the existing stage-authorized tools and any direct request
  allowlist. Closing planning/review turns retain only focused evidence tools plus an eligible
  terminal. The last ready turn exposes only the terminal tool.
- Code review keeps `inspect_change` and targeted repository reads/search until every changed text
  path has fresh read evidence. Nearness to the step limit never satisfies that gate.
- Implementation and repair retain their existing schemas; submission validation, checks, fresh
  review, fingerprints, and managed commit authority remain unchanged.

## Deterministic acceptance

| Fixture | Proof | Result |
| --- | --- | --- |
| CL1 | With two turns left, code review first exposes focused evidence without `submit_code_review`; after `inspect_change`, the last turn exposes exactly that terminal tool and the submission succeeds | pass |
| CL2 | An irrelevant search leaves the required changed-path read missing; the final turn still hides the terminal, and a hallucinated submission returns `precondition_unmet` and cannot advance the stage | pass |
| CL3 | A stale harness-expected fingerprint is named with exact expected/current values and terminal signature in the checkpoint; the terminal remains hidden after path inspection | pass |
| SE1 | Every planning, revision, plan-review, and code-review closure state is a subset of `StageCapabilities`; an explicit direct allowlist cannot be broadened | pass |
| SE2 | Ready final turns are terminal-only, unready turns omit the terminal, and implementation/repair resolve to the unchanged authorized state | pass |
| Metrics/bounds | Both CL1 invocations report one closure checkpoint; the final context reports one tool and measured schema tokens; a 100-fact checkpoint stays below 4,000 characters | pass |

## Reproduction

```bash
cargo test agent_closure::tests
cargo test 'agent_core::tests::cl'
cargo test se1_
cargo test agent_core::tests
cargo test harness_eval::tests
cargo run --quiet -- harness eval --jsonl /tmp/pb-harness-s5-scripted.jsonl
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s5-scripted.jsonl
```

The scripted control corpus remains 41/41 and its small-model subset remains 4/4. S5 does not
change expected protocol outcomes or prune implementation/repair schemas. The checked S0/S1
real-model run is not repeated here because rollout selection is explicitly deferred to the fixed
S6 before/after matrix.

Milestone-wide verification passed `cargo fmt --check`, `cargo test --all-targets` (953 passed, 8
ignored), and `deno task test:web` (47 passed).
