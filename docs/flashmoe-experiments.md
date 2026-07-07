# FlashMoe Experiment Ledger

This file is for agents working on the pb FlashMoe backend. Keep it current whenever an experiment changes model quality, token latency, cache layout, Metal kernels, routing, expert I/O, tokenizer behavior, or `pb flashmoe` CLI behavior.

The goal is not to preserve every idea forever. The goal is to prevent accidental regressions into experiments that were already measured and found harmful, while keeping useful upstream `danveloper/flash-moe` lessons close to the pb implementation.

## Agent Rules

- Add or update a row whenever you try a FlashMoe optimization or correctness change.
- Record the command, prompt, token count, hardware-sensitive observation, and whether the output was sane.
- Keep discarded experiments in the table. A discarded row is a guardrail, not clutter.
- Prefer isolated math tests for kernels and routing before full-model smoke tests.
- If an experiment comes from upstream `danveloper/flash-moe`, say whether pb mirrors it exactly or deviates to support generic Qwen MoE / Qwen3-VL.
- Avoid hidden behavior toggled by environment variables. Prefer explicit CLI flags only when users need them, otherwise keep a single measured path.

## Upstream Reference

Source: https://github.com/danveloper/flash-moe

Upstream target:

- Model: Qwen3.5-397B-A17B.
- Hardware reference: M3 Max MacBook Pro, 48 GB unified memory, Apple SSD.
- Production quality target: 4-bit experts, not 2-bit, because 2-bit was faster but broke JSON/tool calling quality.
- Reported best 4-bit path: about 4.36 tok/s with an FMA dequant kernel.
- Architecture: 60 transformer layers, 45 GatedDeltaNet linear-attention layers, 15 full-attention layers, 512 experts, K=4 active experts per token, plus a shared expert, hidden size 4096.
- Expert storage: about 209 GB at 4-bit; each active expert is about 6.75 MB; each token reads K=4 experts per layer.

Upstream design principles to mirror when possible:

| Principle | Upstream lesson | pb guidance |
| --- | --- | --- |
| Trust the OS page cache | Custom Metal/malloc/LZ4 expert caches lost to memory pressure or overhead. | Do not add a custom expert cache unless a benchmark beats OS page cache on sustained generation and preserves quality. |
| Stream active experts only | Parallel `pread` of K active experts per layer is the intended I/O shape. | Keep expert I/O scoped to active routed experts; avoid broad speculative reads. |
| 4-bit is production | 2-bit achieved higher tok/s but damaged JSON/tool calling. | Treat Q4 as the quality-preserving baseline. Any lower-bit path must pass structured output tests. |
| FMA dequant matters | Rearranging dequant matvec to `fma(nibble, scale*x, bias*x)` was a kept speedup. | Preserve FMA-style Q4 math and parity tests against synthetic reference data. |
| Accelerate for linear attention | BLAS delta-net was materially faster than scalar CPU recurrence upstream. | Prefer Accelerate/Metal/structured kernels over scalar Rust for hot linear-attention loops. |
| Defer GPU expert work | Submit expert forward without waiting so CPU can prepare the next layer. | Keep deferred expert phases unless a measured simplification beats them. |
| Do not overlap SSD DMA with bandwidth-saturated GPU work casually | Upstream found unified memory contention canceled prefetch gains. | Treat prefetch/overlap experiments as suspicious until measured with per-bucket timings. |

## Upstream Experiments To Preserve

| Status | Experiment | Result | Keep-out / keep-in guidance |
| --- | --- | --- | --- |
| Keep | Q4 FMA dequant kernel | Improved sustained Q4 from about 3.90 to 4.36 tok/s upstream. | Keep FMA-style nibble dequant math covered by unit tests. |
| Keep | Trust OS page cache | Removing custom caches was foundational upstream. | Avoid Metal LRU, malloc expert caches, and hidden cache toggles. |
| Keep | GPU combine + norm in deferred expert command | Removed a CPU round trip in the upstream pipeline. | Do not split combine/norm back to CPU without timing evidence. |
| Keep | BLAS delta-net | Upstream reports large attention recurrence improvement over scalar code. | Keep recurrence math testable and avoid scalar-only rewrites in hot paths. |
| Keep | C/native tokenizer startup path | Upstream optimized tokenizer startup heavily. | Keep tokenizer loading measured; avoid slow startup regressions. |
| Discard | LZ4 expert compression | Full pipeline regressed despite smaller disk reads. | Do not compress experts for the main Q4 path without a fresh sustained benchmark. |
| Discard | F_RDADVISE / prefetch during GPU work | I/O improved but GPU waits regressed from unified memory contention. | Treat prefetch as a risky experiment; document any retry carefully. |
| Discard | Temporal/MLP expert prediction | Hit rates were too low for K=4; misses waste SSD bandwidth. | Avoid speculative routing reads unless all-expert-hit probability is addressed. |
| Discard | GPU LUT dequant | Indirect register access serialized and regressed upstream. | Do not replace FMA dequant with LUT without a focused kernel benchmark. |
| Discard | GPU private buffer compression | Shared-to-private blits cost more than matvec savings. | Avoid extra blit stages for expert payloads in the main path. |
| Discard | Spin-poll GPU wait | CPU contention hurt GPU throughput. | Keep blocking/wait behavior boring unless measured otherwise. |
| Discard | `dispatch_io` expert I/O | Dispatch data overhead was far slower than `pread`. | Prefer direct `pread`/FileExt style reads for expert payloads. |
| Discard | mmap expert files | Cold page faults were disastrous for 7 MB expert reads. | Do not mmap expert packs for cold streaming. |
| Discard | MTP speculative decoding | MoE verification still needs per-token expert I/O. | Do not assume dense-model speculative decode economics apply. |

## pb Current Baseline

Last verified: 2026-07-07 on local Apple Silicon release build.

Required smoke:

```bash
target/aarch64-apple-darwin/release/pb flashmoe infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="
```

Observed:

- Exit code: 0.
- Output: `4`.
- Backend load: about 0.5 s.
- One generated token: about 9.0 s.
- Quality: sane for the smoke prompt.
- Performance: not acceptable yet compared with upstream 4+ tok/s.

Validation commands that passed for the current cleanup commit:

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
deno task build:web
deno task test:web
cargo build --release --target aarch64-apple-darwin
```

## pb Experiment Log

| Date | Status | Area | Experiment / change | Evidence | Follow-up |
| --- | --- | --- | --- | --- | --- |
| 2026-07-07 | Keep | CLI | Added `pb flashmoe infer` and `pb flashmoe bench` so the backend can be tested without the agent framework. | Release smoke `2+2=` exited 0 and printed `4`. | Keep the smoke command in AGENTS.md. |
| 2026-07-07 | Keep | Cache | Canonical cache version is `flashmoe-v1-densebf16`; old automatic migrations were removed. | Local cache works with the smoke command; tests pass. | Do not add implicit migrations while this is single-user/local-cache only. |
| 2026-07-07 | Keep | Quality | Routing unit test matches upstream softmax-then-topK behavior. | `routing_weights_match_flashmoe_softmax_then_topk_reference` passes. | Extend parity tests before changing routing. |
| 2026-07-07 | Keep | Q4 math | BF16 scale/bias expert pack test mirrors upstream uint32/nibble packing expectations. | `q4_bf16_expert_pack_matches_flashmoe_uint32_nibble_reference` passes. | Preserve when touching pack format or Metal Q4 kernels. |
| 2026-07-07 | Keep | Dense runtime | Existing dense-Q4 reader/math tests remain, but new dense-Q4 manifest writing is disabled. | Focused FlashMoe tests and full `cargo test --all-targets` pass. | Re-enable only with measured benefit and cache-format docs. |
| 2026-07-07 | Discard | Config surface | FlashMoe-specific environment toggles were removed. | `rg PB_FLASHMOE src/inference/flashmoe/mod.rs src/lib.rs` returned no matches after cleanup. | Prefer deterministic behavior and explicit CLI flags. |
| 2026-07-07 | Discard | Migration | Old cache migration helpers/subcommand were removed. | `rg migrat src/inference/flashmoe/mod.rs src/lib.rs` returned no matches after cleanup. | Rebuild the local cache instead of carrying compatibility code unless users expand beyond one local cache. |
| 2026-07-07 | Discard | Routing | GPU/Metal route topK env-gated path is disabled; CPU routing is the stable path. | Smoke output sane; prior GPU routing experiments were not faster enough and added risk. | Retry only with isolated parity tests and timing evidence. |
| 2026-07-07 | Discard | Attention policy | Short-context Metal attention benchmarking path was removed; policy is fixed and simple. | Tests/clippy pass; upstream results note short-context GPU attention can be slower. | Revisit only with long-context benchmark rows. |
| 2026-07-07 | Open | Performance | Current pb backend is correct enough for smoke but far slower than upstream. | 1 token in about 9 s vs upstream 4-bit target above 4 tok/s. | Profile per-bucket timing from `pb flashmoe bench`; first suspect is dense projection / expert I/O pipeline shape. |
| 2026-07-07 | Keep | Refactor | Moved the historical FlashMoe monolith behind `src/inference/flashmoe/legacy.rs` and kept `mod.rs` as the stable facade. | `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, and `cargo test --all-targets` passed after the move. | Continue extracting modules behind the facade without changing cache/runtime behavior. |
| 2026-07-07 | Keep | Refactor | Extracted public FlashMoe request/output/timing constants and structs to `types.rs`. | Same per-step fmt, clippy, and full Rust test suite passed. | Keep public API types out of execution/platform modules. |
| 2026-07-07 | Keep | Refactor | Extracted pure routing/Q4 math helpers to `math.rs`. | Same per-step fmt, clippy, and full Rust test suite passed. | Move more pure math only with parity tests nearby. |
| 2026-07-07 | Keep | Refactor | Added `inference::backend::InferenceBackend` so llama.cpp and FlashMoe can share text/chat/vision request and output shapes. | Same per-step fmt, clippy, and full Rust test suite passed. | Gradually migrate agent/CLI call sites onto the shared interface after behavior tests cover token accounting and tool-call parsing. |
| 2026-07-07 | Discard | Expert parsing | Tried moving each expert read buffer into shared parsed records so packed/raw scale-bias payload bytes were borrowed instead of copied. | Baseline `target/aarch64-apple-darwin/release/pb flashmoe bench --raw --prompt "2+2=" --max-tokens 2 --temperature 0 --top-k 1 --status kept --results /private/tmp/pb-flashmoe-baseline.tsv`: output `4是`, decode row 2262.641 ms, expert_io 392.775 ms, expert_read 1103.986 ms, expert_compute 55.500 ms. Experiment `/private/tmp/pb-flashmoe-shared-pack-move.tsv`: output `4是`, decode row regressed to 2309.050 ms even though expert_io dropped to 248.866 ms and expert_read to 697.569 ms; expert_compute rose to 236.139 ms, likely from allocation/retention pressure around deferred Metal expert phases. Code reverted. | Do not retry shared backing buffers unless the allocator/retention side is redesigned, e.g. reusable owned expert buffers or direct Metal upload, and benchmark both expert I/O and total decode wall time. |
| 2026-07-07 | Keep | Metal dense matvec | Replaced per-dispatch scalar Metal buffers for dense mmap matvec parameters with `setBytes:length:atIndex:` so input/output buffers are the only temporary Metal buffers. This mirrors upstream's bias toward simpler command setup and avoids tiny allocation churn without changing routing, expert I/O, or model math. | Baseline `/private/tmp/pb-flashmoe-baseline.tsv` from `target/aarch64-apple-darwin/release/pb flashmoe bench --raw --prompt "2+2=" --max-tokens 2 --temperature 0 --top-k 1 --status kept --results /private/tmp/pb-flashmoe-baseline.tsv`: output `4是`, model 11615.972 ms, decode row 2262.641 ms, attention_projection 622.069 ms. Experiment `/private/tmp/pb-flashmoe-dense-setbytes.tsv`: output `4是`, model 11434.461 ms, decode 2205.926 ms, attention_projection 615.578 ms. Repeat `/private/tmp/pb-flashmoe-dense-setbytes-2.tsv`: output `4是`, model 10713.548 ms, decode 2043.682 ms, attention_projection 617.310 ms. | Keep, but treat as an incremental setup cleanup rather than the main bottleneck fix. Next suspects remain expert I/O scheduling and the dense projection pipeline shape; compare steady decode rows, not prefill-inclusive model tok/s. |
| 2026-07-07 | Keep | LM-head sampling | For deterministic, unpenalized sampling on resident dense mmap weights, compute LM-head logits into a GPU buffer and run the existing `topk_vocab` kernel before reading back only the top candidates. This keeps the BF16 resident mmap matvec math unchanged and removes full-vocab CPU readback/top-k from the hot `top_k=1` benchmark path. | Prior kept setBytes run `/private/tmp/pb-flashmoe-dense-setbytes-2.tsv`: output `4是`, model 10713.548 ms, decode 2043.682 ms, sampling 188.649 ms. Experiment `/private/tmp/pb-flashmoe-resident-lmhead-topk.tsv`: output `4是`, model 10586.388 ms, decode 1775.777 ms, sampling 36.255 ms. Repeat `/private/tmp/pb-flashmoe-resident-lmhead-topk-2.tsv`: output `4是`, model 10675.003 ms, decode 1876.723 ms, sampling 42.274 ms. Trimmed final source `/private/tmp/pb-flashmoe-resident-lmhead-topk-trimmed.tsv`: output `4是`, model 10643.710 ms, decode 1734.607 ms, sampling 39.278 ms. | Keep. This improves deterministic decode but does not solve the 4 tok/s target; attention projections remain about 618-622 ms and expert I/O/read noise remains a major source of wall time. For stochastic sampling with repeat penalty, keep the existing CPU-adjusted candidate path unless a GPU penalty-aware top-k is added and tested. |
| 2026-07-07 | Discard | Expert parsing | Tried skipping eager BF16 scale/bias decode for expert packs in release builds, leaving raw BF16 bytes for the deferred Metal expert phase and decoding only for CPU fallback. A focused synthetic unit test passed before reverting, but the full model did not improve. | Prior kept LM-head topK trimmed run `/private/tmp/pb-flashmoe-resident-lmhead-topk-trimmed.tsv`: output `4是`, model 10643.710 ms, decode 1734.607 ms, expert_io 293.041 ms, expert_read 723.255 ms, expert_compute 47.978 ms. Experiment `/private/tmp/pb-flashmoe-lazy-bf16-scale-bias.tsv` from `target/aarch64-apple-darwin/release/pb flashmoe bench --raw --prompt "2+2=" --max-tokens 2 --temperature 0 --top-k 1 --status kept --results /private/tmp/pb-flashmoe-lazy-bf16-scale-bias.tsv`: output `4是`, model 10680.982 ms, decode 1791.973 ms, expert_io 298.361 ms, expert_read 749.804 ms, expert_compute 49.445 ms, sampling 42.313 ms. Code reverted. | Do not retry lazy release-only BF16 decode in the current packed-record shape; parse decode is not the limiting wall-clock cost. If revisiting scale/bias handling, test direct Metal upload or reusable expert buffers against total decode wall time, not just parser allocation counts. |
| 2026-07-07 | Discard | Dense cache | Tried re-enabling dense-Q4 cache writing for attention projection matrices with a bumped local cache version (`flashmoe-v2-denseq4`), leaving `lm_head`, embeddings, router, norms, vision tensors, and shared experts unquantized. Dense-Q4 cache build required regenerating `model_weights.bin` and reused unchanged packed experts via local symlink for the benchmark. | Prior kept LM-head topK trimmed run `/private/tmp/pb-flashmoe-resident-lmhead-topk-trimmed.tsv`: output `4是`, model 10643.710 ms, decode 1734.607 ms, attention_projection 617.687 ms, attention_input_projection 184.356 ms, attention_output_projection 364.279 ms, sampling 39.278 ms. Experiment `/private/tmp/pb-flashmoe-denseq4-attn-projections.tsv` from `target/aarch64-apple-darwin/release/pb flashmoe bench --raw --prompt "2+2=" --max-tokens 2 --temperature 0 --top-k 1 --status kept --results /private/tmp/pb-flashmoe-denseq4-attn-projections.tsv`: output `5`, model 12996.110 ms, decode 1421.127 ms, attention_projection 628.068 ms, attention_input_projection 188.597 ms, attention_output_projection 366.672 ms, sampling 39.290 ms. Initial broader attempt that also quantized shared experts failed before generation with `shared expert tensors for layer 0 do not match config`. Code reverted. | Do not revive dense-Q4 cache writing in this form: it did not reduce the projection bucket and it broke the deterministic smoke answer. If dense quantization is revisited, start with isolated quality/parity checks and a faster Metal Q4 dense kernel microbenchmark before changing the production cache version. |
| 2026-07-07 | Discard | Metal dense matvec | Tried computing two BF16 dense output rows per SIMD group in `dense_mmap_matvec_bf16_simd` and dispatching 16 rows/threadgroup so the 4096-float input cache is reused across more rows. | Prior kept LM-head topK trimmed run `/private/tmp/pb-flashmoe-resident-lmhead-topk-trimmed.tsv`: output `4是`, model 10643.710 ms, decode 1734.607 ms, attention_projection 617.687 ms, attention_input_projection 184.356 ms, attention_output_projection 364.279 ms, routing 19.802 ms, expert_io 293.041 ms, expert_read 723.255 ms, expert_compute 47.978 ms, sampling 39.278 ms. Experiment `/private/tmp/pb-flashmoe-bf16-two-rows-per-simd.tsv` from `target/aarch64-apple-darwin/release/pb flashmoe bench --raw --prompt "2+2=" --max-tokens 2 --temperature 0 --top-k 1 --status kept --results /private/tmp/pb-flashmoe-bf16-two-rows-per-simd.tsv`: output `4是`, model 17858.003 ms, decode 1980.922 ms, attention_projection 649.432 ms, attention_input_projection 217.599 ms, attention_output_projection 364.931 ms, routing 15.272 ms, expert_io 366.184 ms, expert_read 1083.220 ms, expert_compute 47.281 ms, sampling 39.281 ms. Code reverted. | Do not increase rows per SIMD/threadgroup for this BF16 kernel; it worsened input projection and total decode, likely reducing useful parallelism/occupancy and not helping output projection. |

## Required Regression Notes For Future Agents

When changing FlashMoe code, include these in the final response and update the table above:

- Exact command used for correctness smoke.
- Exit code and model output.
- Tokens generated and elapsed time.
- Any per-token/per-layer timing file path from `pb flashmoe bench`.
- Whether the result should be kept, discarded, or retried.
- If deviating from upstream `danveloper/flash-moe`, why pb needs to deviate for generic Qwen MoE or Qwen3-VL support.

## Source Links

- Upstream repository: https://github.com/danveloper/flash-moe
- Upstream Q4 optimization notes: https://github.com/danveloper/flash-moe/blob/main/docs/optimization-experiments-q4.md
- Upstream experiment TSV: https://github.com/danveloper/flash-moe/blob/main/results.tsv
