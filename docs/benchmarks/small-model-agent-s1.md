# Small-model agent S1 prompt-budget checkpoint

Captured: 2026-07-16

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

This checkpoint verifies S1's prompt-control behavior. It is not the S6 effectiveness comparison:
the stable real-model subset is intentionally small and does not create enough context pressure to
trigger compaction. Deterministic CB1-CB4 supply the context-pressure proof, while the fixed S0
model run proves real llama.cpp tokenizer parity and records incidental protocol movement.

## Production policy

| Policy | S1 value |
| --- | ---: |
| Safety margin | 32 tokens |
| Compaction threshold | 70% of usable prompt capacity |
| Compaction target | 60% of usable prompt capacity |
| Prompt tool-result maximum | 16,000 characters |
| Prompt tool-result minimum | 512 characters |
| Deterministic receipt excerpt | 320 characters |

Usable capacity is `context - generation reserve - safety margin`. Preflight renders and tokenizes
the exact generation request. Model invocation accounting begins only after preflight succeeds.

## Deterministic acceptance

| Fixture | Proof | Result |
| --- | --- | --- |
| CB1 | Default read of a 5,000-line file returns only whole lines, exact omitted-line accounting, and an exact next-line/`next_call` continuation | pass |
| CB2 | A successful result larger than 10,000 characters remains complete in `ToolResult`; its later prompt view records omitted content | pass |
| CB3 | System/task-stage anchors remain byte-for-byte identical while completed tool exchanges become deterministic receipts | pass |
| CB4 | Oversized anchors emit `context_limit` before generation, consume zero model invocations, and leave the scripted completion unused | pass |

Additional regression proof covers canonical JSON argument hashes, deterministic receipt content,
bounded prefix/suffix result views, additive event round trips, and the existing fingerprint-bound
read-before-write and review-read gates.

FlashMoe parity is model-free and deterministic: the parity fixture measures a structured prompt
with `measure_structured_prompt`, renders it through the same Qwen chat template, and asserts the
same token IDs and count used by generation. The llama.cpp real-model matrix completed without the
production preflight/backend parity assertion firing; every invocation reported a prompt count
identical to its preflight count.

## Fixed local-model checkpoint

Configuration is unchanged from S0: Qwen2.5-Coder-7B-Instruct Q4_K_M, SHA-256
`509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`, llama.cpp CPU,
8,192-token context, 256 maximum new tokens, temperature 0, top-k 1, seed 0. The cached model has no
sidecar chat template, so llama.cpp uses the same plain-chat fallback recorded by S0. That fallback
does not render native tool schemas, hence exact schema-token delta is zero; tool signatures remain
present in the system prompt.

| Fixture | Exact protocol | Termination | Invocations | Tool calls | Corrections | Prompt tokens | Generated | Preflight high-water | Usable utilization | Latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `false_final_after_inspection` | fail | `final` | 3 | 2 | 2 | 1,212 | 57 | 483 | 6.12% | 21,098 ms |
| `repeated_blocked_action` | fail | `final` | 3 | 2 | 2 | 840 | 56 | 325 | 4.12% | 15,418 ms |
| `final_at_step_limit` | fail | `final` | 3 | 2 | 2 | 1,048 | 244 | 537 | 6.80% | 23,514 ms |
| `review_missing_check` | fail | `gate_loop` | 2 | 0 | 2 | 819 | 117 | 481 | 6.09% | 15,845 ms |

All records report context 8,192, reserve 256, safety margin 32, and usable capacity 7,904. There
were no context overflows, compacted messages, or omitted results in these short fixtures. Exact
protocol remains 0/4, matching the S0 baseline category; generated text and exact action paths vary
despite fixed sampling, so effectiveness claims remain deferred to the repeated S6 matrix. The
scripted subset remains 4/4 under exact preflight accounting.

## Reproduction

```bash
cargo test agent_context::tests
cargo test agent_core::tests
cargo test inference::flashmoe::text_parity_tests::flashmoe_prompt_preflight_uses_the_generation_template_and_tokenizer
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s1-scripted.jsonl
cargo run -- harness eval --suite small-model \
  --model Qwen_Qwen2.5-Coder-7B-Instruct-GGUF \
  --model-dir /Users/john/.local/share/pb/models \
  --ctx-size 8192 --max-tokens 256 --temperature 0 --top-k 1 --seed 0 \
  --gpu-layers 0 \
  --jsonl /tmp/pb-small-model-s1-qwen2.5-coder-7b-cpu.jsonl
```

The local-model command exits nonzero when model behavior differs from exact scripted fixture
expectations; completed JSONL records and absence of runtime diagnostics distinguish that expected
protocol result from backend or context failure.
