# Small-model agent S0 baseline

Captured: 2026-07-15

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

This is the pre-S1 behavioral baseline for the stable `small_model` harness-evaluation subset. S0
adds observation only: it does not compact prompts, bound reads, change tool exposure, alter
corrections, or change workflow gates.

## Configuration

| Field | Value |
| --- | --- |
| Repository base | `da5b913c` plus the S0 observation changes recorded by this report |
| Host | macOS 26.5.1, arm64 |
| Model identifier | `Qwen_Qwen2.5-Coder-7B-Instruct-GGUF` |
| Model file | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` |
| Model SHA-256 | `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Model size | 4,683,073,536 bytes |
| Backend | llama.cpp, CPU (`--gpu-layers 0`) |
| Chat template | llama.cpp plain-chat fallback; the cached model directory has no sidecar `tokenizer_config.json` |
| Context | 8,192 tokens |
| Maximum new tokens | 256 per turn |
| Sampling | temperature 0, top-k 1, seed 0 |
| Fixture group | `false_final_after_inspection`, `repeated_blocked_action`, `final_at_step_limit`, `review_missing_check` |
| Machine conditions | interactive development run; no controlled power mode; energy is observational only |

The initial Metal attempt with `--gpu-layers 999` failed before inference because llama.cpp could
not create either its accelerated context or CPU K/Q/V fallback context. The evaluator preserved
the diagnostic:

```text
failed to create llama context, including CPU K/Q/V fallback after accelerated context error:
null reference from llama.cpp: null reference from llama.cpp
```

That setup failure is classified as experiment/backend compatibility evidence, not model quality.
The recorded behavioral baseline uses `--gpu-layers 0` and completed normally.

## Reproduction

Scripted control subset:

```bash
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s0-scripted.jsonl
```

Local-model subset:

```bash
cargo run --quiet -- harness eval --suite small-model \
  --model Qwen_Qwen2.5-Coder-7B-Instruct-GGUF \
  --model-dir /Users/john/.local/share/pb/models \
  --ctx-size 8192 --max-tokens 256 --temperature 0 --top-k 1 --seed 0 \
  --gpu-layers 0 \
  --jsonl /tmp/pb-small-model-s0-qwen2.5-coder-7b-cpu.jsonl
```

The local-model command exits non-zero when model behavior differs from the exact checked control
expectations. Its JSONL is still the intended S0 measurement artifact.

## Scripted control result

All four selected fixtures pass their exact protocol expectations. Each invocation reports the
same deterministic context accounting:

- context capacity: 1,024;
- generation reserve: 256;
- usable prompt capacity: 768;
- observed prompt high-water: 1 scripted token, 0.14% of usable capacity;
- no compaction, omitted result content, cache hit, or closure checkpoint; and
- exposed schema high-water: 398-870 characters depending on the fixture.

This proves that S0 observation did not change the selected control outcomes.

## Local-model result

| Fixture | Exact protocol | Termination | Invocations | Tool calls | Corrections | Prompt tokens | Generated tokens | Context high-water | Schema chars | Latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `false_final_after_inspection` | fail | `final` | 3 | 2 | 2 | 1,215 | 75 | 6.10% | 555 | 19,417 ms |
| `repeated_blocked_action` | fail | `final` | 3 | 2 | 2 | 904 | 108 | 4.96% | 555 | 16,134 ms |
| `final_at_step_limit` | fail | `final` | 3 | 2 | 2 | 891 | 95 | 4.74% | 398 | 15,499 ms |
| `review_missing_check` | fail | `gate_loop` | 2 | 0 | 2 | 817 | 133 | 6.03% | 870 | 15,892 ms |

Aggregate observations:

- Exact fixture protocol: 0/4. This is the model baseline, not a deterministic harness regression;
  the scripted subset remains 4/4.
- Context overflow: 0/4. Authoritative material fit comfortably at 8K.
- Maximum observed context utilization: 6.10% of usable prompt capacity.
- The first three fixtures reached a final response but did not follow the exact expected action and
  turn sequence.
- The review fixture produced no valid tool call and stopped in a bounded gate loop after two
  corrections.
- Every fixture required two corrections. Recovery guidance and action production are therefore
  measurable improvement targets even before long-context fixtures are added.
- Because these four fixtures are small, they establish protocol/action behavior rather than the
  context-pressure benefit targeted by S1/S2. CB1-CB4 and RV1-RV3 will provide that proof.

## S1-S6 comparison lock

Use the same model bytes, fixture IDs, context, token cap, sampling, and CPU backend for the final S6
comparison unless a documented backend defect makes that impossible. Report any alternate backend
as a separate row rather than replacing this baseline.

The minimum non-regression rules are:

- scripted protocol remains 4/4 for this subset and green for the complete control corpus;
- no false-completion or workflow-authority regression;
- no context overflow when authoritative anchors fit;
- corrections, invocations, and task outcomes are reported separately from prompt reduction; and
- a runtime/backend failure is not scored as model reasoning quality.
