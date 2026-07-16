# Small-model agent S6 real-model comparison and rollout decision

Captured: 2026-07-16

Plan: [Small-model agent reliability plan](../small-model-agent-reliability-plan.md)

S6 repeats the locked S0 small-model corpus before and after the final evidence-driven control
changes. Protocol compliance, safe containment, context use, recovery, runtime, and energy remain
separate measurements. The local-model command is expected to exit nonzero when any exact fixture
expectation differs; complete JSONL records without a runtime diagnostic are valid measurements.

## Locked configuration

| Field | Value |
| --- | --- |
| Host | macOS 26.5.1 (25F80), arm64 |
| Model | `Qwen_Qwen2.5-Coder-7B-Instruct-GGUF` / `qwen2.5-coder-7b-instruct-q4_k_m.gguf` |
| Model SHA-256 | `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Backend | llama.cpp CPU, `--gpu-layers 0` |
| Chat template | llama.cpp plain-chat fallback; the cached directory has no sidecar template |
| Contexts | 8,192 and 16,384 tokens |
| Generation | 256 maximum new tokens per turn |
| Sampling | temperature 0, top-k 1, seed 0 |
| Trials | three complete four-fixture trials per context and phase |
| Fixtures | `false_final_after_inspection`, `repeated_blocked_action`, `final_at_step_limit`, `review_missing_check` |
| S6 before candidate | `3775349d` (`feat: add deterministic workflow closure`) |

The model bytes match S0 exactly. Runs were sequential on the same interactive development host;
power mode was not controlled, so energy remains observational.

## Aggregate before/after matrix

Each row contains 12 records: four fixtures repeated three times. Context high-water is the maximum
single preflight prompt; utilization is relative to usable prompt capacity after the 256-token
generation reserve and 32-token safety margin.

| Context | Phase | Protocol | False completion | Overflow | Invocations | Tool calls | Corrections | Prompt tokens | Generated | Prompt high-water | Max utilization | LLM latency |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | before | 0/12 | 0 | 0 | 39 | 24 | 42 | 24,070 | 3,924 | 1,775 | 22.46% | 497,100 ms |
| 8K | after | 6/12 | 0 | 0 | 24 | 9 | 11 | 8,091 | 1,713 | 506 | 6.41% | 180,581 ms |
| 16K | before | 0/12 | 0 | 0 | 39 | 24 | 42 | 23,676 | 3,774 | 1,773 | 11.02% | 478,968 ms |
| 16K | after | 6/12 | 0 | 0 | 28 | 11 | 13 | 10,806 | 2,416 | 1,053 | 6.55% | 276,445 ms |

At 8K, the final candidate reduces invocations by 38.5%, tool calls by 62.5%, corrections by
73.8%, prompt tokens by 66.4%, generated tokens by 56.3%, prompt high-water by 71.5%, and measured
LLM latency by 63.7%. The 16K direction is the same; one `review_missing_check` trial exercised the
bounded same-cap thinking-off truncation retry, increasing its local totals without changing the
stable 2/4 protocol outcome.

Energy samples were incomplete and noisy. Before/after sums were respectively 1.334e-3/5.352e-4
kWh at 8K (8/7 records sampled) and 1.137e-3/2.094e-4 kWh at 16K (7/8 sampled). They are recorded
for observability and are not used to claim a controlled power improvement.

## Final per-fixture stability

Means are over three trials. `Pass` is the exact checked protocol expectation, not artifact prose
quality.

| Context | Fixture | Pass | Invocations | Tools | Corrections | Prompt | Generated | Max utilization | Latency |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | `false_final_after_inspection` | 3/3 | 2.00 | 1.00 | 0.00 | 627 | 66 | 4.43% | 11,028 ms |
| 8K | `repeated_blocked_action` | 3/3 | 3.00 | 2.00 | 1.67 | 988 | 347 | 5.86% | 26,752 ms |
| 8K | `final_at_step_limit` | 0/3 | 1.00 | 0.00 | 0.00 | 246 | 27 | 3.13% | 4,716 ms |
| 8K | `review_missing_check` | 0/3 | 2.00 | 0.00 | 2.00 | 836 | 131 | 6.41% | 17,698 ms |
| 16K | `false_final_after_inspection` | 3/3 | 2.00 | 1.00 | 0.00 | 627 | 66 | 2.19% | 11,115 ms |
| 16K | `repeated_blocked_action` | 3/3 | 3.00 | 2.00 | 1.67 | 987 | 345 | 2.85% | 31,779 ms |
| 16K | `final_at_step_limit` | 0/3 | 1.00 | 0.00 | 0.00 | 246 | 27 | 1.54% | 5,070 ms |
| 16K | `review_missing_check` | 0/3 | 3.33 | 0.67 | 2.67 | 1,742 | 367 | 6.55% | 44,184 ms |

The two S0 failures targeted by the diagnosed harness defects now pass in all six final trials.
`final_at_step_limit` still finalizes without creating the fixture's hidden `result.txt` artifact;
`review_missing_check` remains safely contained by the review evidence gate. Neither produces a
false completion. Adding fixture-specific action arguments or weakening review evidence would make
the score look better without making the agent safer or more generally capable, so S6 does neither.

## Evidence-driven production changes

1. A restricted direct-run prompt no longer orders tools absent from its actual allowlist. The
   before trace showed `session_title` and `run_command` attempts even when only `read_file` was
   exposed. Restricted prompts now say to use only `Available tools` and include at most 32 sorted
   top-level repository paths, each capped at 120 characters. `.git` and `.pb` are excluded.
2. A read whose range is beyond EOF remains an ordinary non-mutating tool result, but progress
   classification treats it as known-empty. On unchanged content, another known-empty range on the
   same path is blocked before execution after either an empty result or an exact cache replay.
   Valid pagination, another path, and any workspace/evidence transition remain allowed.
3. Real-model JSONL now includes an additive `tool_trace`: tool names are capped at 120 characters;
   arguments are represented by their normalized SHA-256 plus a 600-character preview and explicit
   truncation flag. This made wrong-action diagnosis reproducible without allowing a large write or
   patch argument to bloat the evaluation report.

These changes do not add tools, broaden a request allowlist, execute guessed actions, turn a prompt
hint into evidence, or weaken read/review/check/fingerprint gates. A cache replay still applies only
its original scoped evidence effects.

## Model-control and escalation decision

S6 does **not** introduce a `ModelControlPolicy`. The matrix does not support a model-family
override:

- 8K and 16K have identical final protocol outcomes, and all authoritative anchors fit;
- pre-fix 8K utilization peaked at only 22.46%, with zero context overflow;
- exposed schemas peaked at 870 characters, so a smaller schema budget is not the limiting factor;
- bounded tool-result policy did not compact or omit these short fixtures; and
- the existing same-cap thinking-off retry handled the single observed truncation without a larger
  token cap.

The shipped S1/S4 defaults therefore remain: a 32-token safety margin, 70% compaction threshold,
60% target, context-derived result bounds up to 16,000 characters, thinking enabled for ordinary
turns, one same-cap thinking-off truncation retry, then at most one bounded larger-cap retry. There
is no automatic stronger-model or cloud escalation. Selecting another local model remains an
explicit `--model` choice; no hidden or default escalation path was added.

## Acceptance and reproduction

- Scripted protocol remains green for the complete control corpus and the stable small-model
  subset.
- False completion and context overflow remain zero in every real-model record.
- `false_final_after_inspection` and `repeated_blocked_action` improve from S0 failures to stable
  passes without regressing any previously passing real-model protocol fixture (S0 had none).
- The deterministic RV1 constructed review-prompt reduction remains at least 40%.
- CB1-CB4, RV1-RV3/RB1, PG1-PG5 plus the S6 empty-read case, AR1-AR4, CL1-CL3, and SE1-SE2 remain
  under their deterministic bounds.

Scripted and focused checks:

```bash
cargo test agent_progress::tests
cargo test s6_repeated_known_empty_read_range_is_blocked_before_execution
cargo test direct_allowlist_prompt_never_orders_unexposed_tools_and_names_bounded_paths
cargo test real_model_tool_trace_hashes_and_bounds_arguments
cargo test rv1_large_review_prompt_uses_manifest_and_focused_inspection
cargo run --quiet -- harness eval --jsonl /tmp/pb-harness-s6-scripted.jsonl
cargo run --quiet -- harness eval --suite small-model \
  --jsonl /tmp/pb-small-model-s6-scripted.jsonl
```

Repeat the local-model command three times with distinct output paths for each `--ctx-size` value:

```bash
target/debug/pb harness eval --suite small-model \
  --model Qwen_Qwen2.5-Coder-7B-Instruct-GGUF \
  --model-dir /Users/john/.local/share/pb/models \
  --ctx-size 8192 --max-tokens 256 --temperature 0 --top-k 1 --seed 0 \
  --gpu-layers 0 --jsonl /tmp/pb-small-model-s6-8k-trial-1.jsonl

target/debug/pb harness eval --suite small-model \
  --model Qwen_Qwen2.5-Coder-7B-Instruct-GGUF \
  --model-dir /Users/john/.local/share/pb/models \
  --ctx-size 16384 --max-tokens 256 --temperature 0 --top-k 1 --seed 0 \
  --gpu-layers 0 --jsonl /tmp/pb-small-model-s6-16k-trial-1.jsonl
```

The final repository-wide verification results are recorded in the plan evidence log.
