# FlashMoe Architecture Parity Plan

This document is the source of truth for bringing pb's MoE backend into architectural parity with
`danveloper/flash-moe` while retaining pb's production shell: Rust pull/conversion, production
tokenizers, structured requests, Qwen-family model support, Qwen-VL inputs, and the shared inference
facade with llama.cpp.

The target is one scheduled execution graph for supported MoE model families. FlashMoe owns
buffer lifetime, expert scheduling, command-buffer sequencing, read scheduling, state transitions,
and CPU/GPU placement. Model variants provide typed stage implementations without creating another
runtime.

This plan replaces experiment-led development. Progress is measured by production ownership moving
to the target modules, fallback paths being eliminated, and correctness being preserved. Commit
count, new descriptor count, microbenchmark results, and isolated tok/s changes are not measures of
architectural completion.

## Non-Negotiable Invariants

- One path means one graph shape and one scheduler-owned layer lifecycle.
- Q4, BF16, F16, F32, Qwen3.5, Qwen MoE, and Qwen-VL differences are selected typed
  implementations of graph stages.
- Every supported model resolves a complete concrete graph before inference starts.
- Resolution includes model family, actual manifest dtype/layout, expert layout, device/kernel
  availability, stage placement, and execution policy.
- A missing implementation is an explicit unsupported-capability error. It never causes a silent
  change of dtype, layout, buffer ownership, CPU/GPU placement, scheduler, runtime, or backend.
- CPU execution is valid only when it is the resolved implementation for that graph stage. CPU code
  is not a production fallback for missing Metal support.
- Q4-specific offsets, packing, and kernels are valid inside the shared graph. They must be selected
  through typed layouts and leave the scheduler, state lifecycle, and command topology unchanged.
- PBQ4 is an import/build compatibility format, not a runtime execution layout.
- Production expert execution consumes scheduler-owned reusable whole-expert slots.
- Fixed whole-expert slots use page-aligned reusable backing. On Apple Silicon, scheduled CMD3
  lazily binds one non-owning shared Metal buffer to each backing allocation and reuses that wrapper
  whenever the identity-free slot returns from the scheduler pool. The submission retains the slot
  lease through GPU completion, and the wrapper is released before its backing allocation;
  non-aligned compatibility payloads use copied staging. This handoff does not retain expert
  identity or introduce an application expert cache.
- Direct component-buffer execution, upload-heavy reconstruction, and fused-to-unfused substitution
  are unsupported unless each is an explicitly resolved graph-stage implementation.
- Reference CPU implementations belong in tests, diagnostics, and explicit development tools.
- No new production behavior belongs in `legacy.rs` except a narrow temporary call-through needed
  in the same ownership slice to move its owner.
- If an architecture-aligned change makes `2+2=` or a parity fixture wrong, debug math, logits, and
  state through the unified path. Do not restore the old architecture to hide the error.
- Synchronous Metal completion retains the bounded timeout and terminal-status diagnostics while
  polling finely enough that host-side wait granularity does not become a per-layer decode stage.

## Upstream Parity Contract

The behavioral source of truth is `danveloper/flash-moe`, especially `metal_infer/infer.m`,
`metal_infer/shaders.metal`, `repack_experts.py`, and the Q4 optimization notes.

The Qwen3.5 Q4 implementation must preserve these upstream-shaped properties:

- Non-expert weights live in one 64-byte-aligned binary blob, mapped once.
- Experts are fixed records in per-layer files. The Qwen3.5-A17B Q4 record is 7,077,888 bytes with
  fixed gate/up/down packed-weight, scale, and bias offsets.
- Active experts are issued in parallel with positioned reads into scheduler-owned reusable
  whole-expert slots.
- Page-aligned fixed slots are handed directly to Metal CMD3 under their scheduler leases. Their
  non-owning Metal wrappers follow backing-allocation lifetime rather than expert identity, avoiding
  repeated VM wrapper creation while preserving the same positioned-read and lease lifecycle.
  Copied staging remains the compatibility path for payloads that cannot satisfy Metal's no-copy
  memory contract.
- The OS page cache is the expert cache. There is no application expert LRU, LZ4 main path, mmap
  expert read path, broad prefetch, speculative read path, `dispatch_io`, or hidden scheduling
  toggle.
- A layer follows this topology:
  `CMD3(previous) -> CMD1 -> declared attention math -> CMD2 -> declared routing -> expert reads -> CMD3`.
- CMD2 performs output projection, residual update, post-attention norm, router projection, and
  shared gate/up work.
- Qwen3.5 routing uses CPU softmax/topK after router score production.
- CMD3 performs active expert gate/up/SwiGLU/down, shared down, weighted combine, residual update,
  and next-layer input norm.
- CMD3 is deferred and retains the next layer's GPU-resident hidden/normed inputs when the resolved
  graph permits it.
- Expert Q4 matvec uses the upstream FMA dequant form.
- Throughput comparison uses sustained decode/generation throughput. TTFT and prefill are reported
  separately.

## GLM-5.2 Extension Contract

Status: **Shipped baseline**. Indexed MLX MXFP4 and unindexed Colibri import, dense lead-in layers,
compressed-KV MLA, sigmoid/noaux routing, shared-expert execution, and streamed routed experts are
implemented. The full `mlx-community/GLM-5.2-mxfp4` checkpoint has completed cache construction,
load, prefill, decode, and deterministic text generation through FlashMoe. DSA and MTP remain
design-record follow-ons rather than shipped guarantees.

GLM-5.2 support extends the same scheduler and runtime rather than embedding a checkpoint producer
or adding an alternate engine. Source-container concerns end at the pull/cache boundary; concrete
quantization remains explicit in pb's typed resident-dense and fixed-slot expert layouts before
graph resolution.

- The preferred source is `mlx-community/GLM-5.2-mxfp4`. Its indexed U32 weights hold low-nibble-
  first E2M1 values and its U8 scales hold E8M0 powers of two for groups of 32. Pull validates that
  layout and publishes routed experts without requantization as fixed native-MXFP4 slots. Each
  projection retains the source E2M1 nibbles and one E8M0 byte per group with no affine bias. The
  native Metal matvec decodes that typed representation directly; capability resolution fails if
  its kernel is absent. Resumable expert validation hashes the exact source weight/scale slice, and
  rejects the E8M0 `0xff` non-finite encoding before publishing a slot.
- Pull preserves a repository-level `chat_template.jinja` beside the tokenizer. Text rendering uses
  an embedded `tokenizer_config.json` template first and otherwise loads that external template, so
  GLM role/control tokens are applied in ordinary chat generation while `--raw` remains an explicit
  diagnostic completion path.
- The Colibri adapter remains supported for unindexed `out-*.safetensors`: packed offset-binary int4
  or signed int8 tensors plus F32 `.qs` row/group scales. It preserves int4 nibbles, converts
  symmetric scales into fixed affine-Q4 records, and preserves int8 input/output precision as
  resident BF16.
- Both adapters publish the normal aligned dense blob and per-layer fixed expert files under the
  GLM-specific cache version. Distinct fixed-MXFP4 and fixed-affine-Q4 metadata bind the runtime
  kernel without source probing; runtime code sees the canonical manifest and never parses MLX or
  Colibri source containers.
- Dense attention, embeddings, norms, the first three dense MLPs, and one shared expert per sparse
  layer remain resident. Routed experts use the existing scheduler-owned parallel positioned-read
  path, reusable whole-expert slots, Metal CMD3 implementation, and OS page cache; GLM does not add
  an application expert cache or speculative expert I/O path.
- GLM's attention implementation is MLA with q/kv LoRA, partial interleaved RoPE, and a compressed
  KV record containing the normalized KV latent plus rotary key. It is a typed attention-stage
  implementation in the shared graph, not a separate layer loop. The scheduler declares the
  unequal latent/rotary widths as MLA state rather than weakening the equal-width K/V contract used
  by full attention. Q/KV LoRA projection norms retain GLM's fixed `1e-6` epsilon independently of
  the decoder RMSNorm epsilon in config. Runtime weight absorption accepts either the original
  two-dimensional `kv_b_proj` or MLX-LM's equivalent pre-absorbed per-head `embed_q` and
  `unembed_out` tensors; both execute against the same compressed cache without materializing
  per-token keys and values. The capability graph binds MLX's pre-absorbed Q4 multilinear tensors,
  input `q_a`/`kv_a`/projection-RMSNorm/`q_b`, interleaved-to-split-half RoPE, current-record append,
  causal absorbed scores, softmax, compressed-context reduction, output unembedding, `o_proj`,
  residual/post-attention RMSNorm, and router projection into one ordered Metal command for sparse
  layers. The compressed-KV records remain scheduler-declared CPU-visible session state: prior
  records are uploaded before the command, while the current normalized latent and rotated key
  return alongside the CPU routing candidates after the command completes. The CPU cache therefore
  remains authoritative for prefix reuse and session snapshots without intermediate query/KV or
  attention-output readback and re-upload. The fused `kv_b_proj` adapter remains on its declared CPU
  transpose implementation until it has an equivalent resident kernel; runtime probing does not
  switch between them.
- Sparse layers select experts with sigmoid scores, `e_score_correction_bias` for selection only,
  top-K over the corrected scores, selected raw sigmoid weights normalized when
  `norm_topk_prob=true`, and `routed_scaling_factor` applied after normalization. Qwen softmax
  routing remains unchanged.
- `first_k_dense_replace` selects a resident SwiGLU feed-forward stage for the leading layers.
  Sparse layers retain the shared CMD2/CMD3 lifecycle and always-active shared expert.
- The baseline correctness graph uses full causal MLA. DSA sparse selection and the MTP head are
  follow-on typed accelerators and must not silently alter baseline precision, routing, or output.
  Until DSA is implemented, support must report its validated context/correctness boundary rather
  than claiming token-exact long-context parity.
- Focused validation covers indexed MLX MXFP4 dense conversion and aggregate-expert preservation,
  exact CPU and local-Metal E2M1/E8M0 decoding, malformed E8M0 rejection, metadata resolution,
  missing-kernel capability failure, Colibri int4/int8 import, unindexed shards, logical-shape recovery, MLA weight
  absorption with fused and pre-absorbed weights, distinct compressed-KV scheduler state, RoPE
  layout, sigmoid/noaux routing, graph capability binding, and sparse-layer expert metadata. Real
  checkpoint evidence covers all 78 decoder layers, 75 streamed sparse layers, first-token sampling,
  an eight-token decode, and the checkpoint's full external chat template. A deterministic
  no-thinking chat request for `What is 2+2?` completes as `2 + 2 = 4`; the diagnostic raw prompt
  `The capital of France is` begins with `Paris`. Model loading rejects incomplete MLA,
  router-bias, dense-lead-in, or expert artifacts before allocating generation state.

### GLM-5.2 Performance Evidence

The 2026-07-17 performance pass used the exact deterministic raw request
`2+2=` with three generated tokens and detailed per-layer timings against the local
`mlx-community/GLM-5.2-mxfp4` cache. The output remained token-identical as `5 2`.

- The shipped CPU-absorption baseline decoded at `0.170 tok/s` (`5.880 s/token`). Moving only
  pre-absorbed resident-Q4 `embed_q` and `unembed_out` multilinear work to Metal decoded at
  `0.205 tok/s` (`4.876 s/token`), a 20.6% throughput increase and 17.1% lower decode latency.
- Attention wall time fell from `2.506` to `1.592 s/token`; its absorbed-attention bucket fell from
  `1.397` to `0.480 s/token`. Expert I/O remained essentially unchanged at `1.983` versus
  `1.939 s/token`, which is the expected control for an attention-only change.
- Each GLM token still selects eight 21,233,664-byte slots across 75 sparse layers: 12,740,198,400
  bytes (`11.865 GiB`) of expert traffic before staging. The measured result therefore does not
  support a 3-4 tok/s SSD-only target for this layout.

The comparison sources establish different regimes. The
[MLX-LM lazy-loading issue](https://github.com/ml-explore/mlx-lm/issues/1438) is a design request for
on-demand expert loading, LRU residency, and optional next-layer prefetch, not published performance
evidence. [Colibri](https://github.com/JustVugg/colibri) reports 4+ tok/s with six RTX 5090 GPUs and
hot expert tiers, while its README estimates only 0.05-0.1 tok/s for a cold 1 GB/s SSD path reading
about 11 GB/token. FlashMoe already coalesces each selected expert into one positioned read and
issues the active set in parallel. Colibri's application LRU/hot pinning and speculative lookahead
remain outside pb's OS-page-cache and non-speculative I/O contract; DSA and MTP remain the larger
typed follow-ons for reducing attention work or forwards per accepted token.

A second pass on the same date fused the MLX `q_a`/`kv_a`/projection-norm/`q_b` input chain into one
Metal command. A real-Metal reference test covers distinct Q/KV projections, in-place Q/KV RMSNorm,
the chained `q_b` projection, and preservation of the non-normalized rotary suffix. The deterministic
three-token raw output remained `5 2`.

- In a back-to-back checkpoint/new-binary comparison, MLA attention fell from `1.581` to
  `1.332 s/token`; its CPU setup/miscellaneous bucket fell from `0.246` to `0.004 s/token` while the
  absorbed-attention kernel stayed flat at `0.486` versus `0.487 s/token`.
- Observed decode improved from `0.202` to `0.226 tok/s` (`4.946` to `4.421 s/token`). Expert I/O also
  varied from `1.945` to `1.694 s/token`; holding that unrelated bucket at the checkpoint value gives
  `4.672 s/token`, or about `0.214 tok/s`. The fused command therefore accounts for roughly a 5.9%
  end-to-end improvement under equal I/O, while `0.226 tok/s` is the measured whole-run result.
- A separate run reached `0.248 tok/s` when expert I/O fell to `1.290 s/token`. That is useful evidence
  for the remaining storage-side opportunity, but is not attributed to the attention change.

A third pass moved the absorbed causal scores, softmax, and compressed-context reduction between
the existing Q4 `embed_q` and `unembed_out` projections into their shared Metal command. A focused
real-Metal test compares the complete command with a CPU reference across distinct heads and two KV
records. The same three-token request remained token-identical as `5 2`.

- The absorbed-attention kernel bucket fell from `0.487` to `0.237 s/token`, a 51.3% reduction.
  Total decode measured `0.222 tok/s` (`4.513 s/token`) versus the preceding `0.226 tok/s`
  (`4.421 s/token`) run because expert I/O rose from `1.694` to `1.947 s/token`. Replacing only that
  noisy I/O bucket with the preceding value gives `4.261 s/token`, or about `0.235 tok/s`: a
  `0.161 s/token` intrinsic whole-token improvement for this pass.
- Diagnostic K=4 and K=1 runs reached `0.410` and `0.667 tok/s`, respectively, but changed the raw
  output to `4 5` and `л2`. They are lower-bound measurements, not supported quality modes. Even K=1
  leaves a `1.499 s/token` floor, proving that expert traffic alone cannot reach the 1 tok/s goal;
  resident graph/command work remains necessary alongside any future typed storage optimization.

A fourth pass replaced per-layer Metal expert-staging allocation and purge churn with the bounded,
identity-free staging pool described above. The buffers remain staging allocations: all bytes are
overwritten on checkout, no expert or layer key survives recycling, and pressure/error paths still
purge them. The same three-token request remained token-identical as `5 2`.

- Expert compute fell from `0.832` to `0.197 s/token`, a 76.3% reduction. Total decode fell from
  `4.513` to `2.644 s/token`, raising observed throughput from `0.222` to `0.378 tok/s` (70.3%).
  The absorbed-attention kernel was effectively unchanged at `0.237` versus `0.238 s/token`, which
  is the expected control for a staging-allocation change.
- The whole-run expert-I/O and MLA-input buckets also fell from `1.947` to `1.366 s/token` and from
  `0.986` to `0.404 s/token`. Those reductions are consistent with removing repeated driver and
  memory-allocation pressure, but include OS-page-cache and run-to-run effects and are not treated
  as isolated storage or attention improvements.
- The post-run resource ledger reported 60 pooled buffers totaling 171,832,752 bytes, zero
  transient expert buffers, 60 allocations, 30,012 reuses, and no pressure recovery. Eight
  whole-expert staging allocations were sufficient for the active K=8 width; the 16-buffer bound
  permits size variation without retaining a token- or expert-identity cache.

A fifth pass joined the GLM MLA input projections, exact CPU-precomputed RoPE factors, current
compressed-KV append, absorbed attention, and output unembedding into one ordered Metal command.
The CPU compressed-KV cache remains authoritative: previous records are uploaded before submission,
and the current record is copied back only after the command completes. A focused real-Metal test
matches the former two-command path's attention output, normalized latent, and rotated key. The same
three-token request again remained token-identical as `5 2`.

- Against a serial control captured under the same high-I/O conditions, attention fell from `0.725`
  to `0.502 s/token` and total decode from `3.158` to `2.948 s/token`. Expert I/O was flat at `1.931`
  versus `1.934 s/token`, isolating a 30.8% attention reduction, 6.6% lower total latency, and a
  throughput increase from `0.317` to `0.339 tok/s`.
- A repeat reached `0.345 tok/s` (`2.901 s/token`) with attention at `0.474 s/token` and expert I/O
  again at `1.935 s/token`. Replacing only that noisy I/O bucket with the fourth pass's best
  `1.366 s/token` measurement gives `2.332 s/token`, or about `0.429 tok/s`; this remains below the
  `1 tok/s` objective and identifies expert traffic as the dominant measured bucket.
- The fused timing is recorded in the attention-kernel bucket because its internal projection,
  RoPE, attention, and unembedding phases share a single command boundary. The resource ledger
  reported 66 pooled buffers totaling 171,953,056 bytes, zero transient expert buffers, 66
  allocations, 30,942 reuses, and no pressure recovery.

A sixth pass extended the sparse-layer command through `o_proj`, residual addition, post-attention
RMSNorm, and router projection. The CPU routing algorithm and correction-bias semantics are
unchanged, but they now consume logits at the same wait that returns the current compressed-KV
record. A real-Metal reference test compares routes plus retained residual/normed buffers with the
former MLA-command-then-CMD2 sequence. The deterministic raw output remained `5 2`.

- In a back-to-back checkpoint/new-binary comparison, total decode fell from `2.487` to
  `2.059 s/token`, raising throughput from `0.402` to `0.486 tok/s` (20.9%). Expert I/O increased
  slightly from `1.287` to `1.308 s/token`; replacing only that bucket with the control value gives
  `2.037 s/token`, or about `0.491 tok/s`.
- The separate CMD2 combine/norm/router bucket fell from `0.391` to `0.0003 s/token`; the enclosing
  attention wall bucket also fell from `0.571` to `0.512 s/token`. The fused attention-kernel bucket
  now includes post-attention projection and routing work and is not additive with the enclosing
  attention bucket.
- The resource ledger reported 71 pooled buffers totaling 172,092,064 bytes, zero transient expert
  buffers, 71 allocations, 30,487 reuses, and no pressure recovery. The five additional small pooled
  buffers retain no expert identity and remain within the existing general-buffer bound.

A seventh pass removed the second whole-expert memory copy. Worker `pread` now fills page-aligned
anonymous reusable slots; Metal CMD3 wraps each completed fixed slot without copying while the
scheduler lease retains its backing through command completion. Non-aligned compatibility payloads
keep the copied staging path. A real-Metal pointer-identity test verifies that the wrapped buffer
exposes the original slot address. The deterministic three-token raw output remained `5 2`.

- Two consecutive detailed runs both decoded at `0.512 tok/s`. The second measured `1.952 s/token`,
  down from the sixth pass's `2.059 s/token` and up from `0.486 tok/s` (5.3%). Expert compute fell
  from about `0.193` to `0.027 s/token`; expert I/O measured `1.277` versus `1.308 s/token`, while
  the enclosing attention bucket varied from `0.512` to `0.603 s/token`.
- Holding the sixth pass's non-expert buckets constant, the removed copy is worth about
  `0.166 s/token`, implying `1.893 s/token` or `0.528 tok/s`. The smaller measured whole-run gain is
  consistent with the opposing attention variation rather than residual expert staging work.
- The resource ledger fell from 71 pooled buffers and 172,092,064 pooled bytes to 63 buffers and
  2,222,752 bytes, with zero transient expert buffers, 63 allocations, 26,895 reuses, and no
  pressure recovery. The missing eight 21,233,664-byte allocations are exactly the former K=8 GLM
  staging set; the scheduler's reusable host slots retain no expert identity.

An eighth pass reduced synchronous Metal command-status polling from 2 ms to 100 us while retaining
the 120-second timeout, terminal-status validation, and detailed Metal error reporting. This changes
only host completion granularity: command construction, GPU work, state ownership, and model math
are unchanged. The deterministic three-token raw output remained `5 2`.

- Two consecutive detailed runs decoded at `0.549` and `0.542 tok/s`, up from the seventh pass's
  repeated `0.512 tok/s`. Their mean `1.833 s/token` is 6.1% below the `1.952 s/token` checkpoint.
- Resident attention averaged `0.509 s/token` across the two new runs versus `0.603 s/token` in the
  checkpoint, a 15.6% reduction. Expert-command completion averaged `0.012 s/token` versus
  `0.027 s/token`, a 54.4% reduction. Both buckets previously accumulated one coarse polling sleep
  at many layer boundaries rather than measuring additional GPU work.
- Expert I/O was the control: it averaged `1.278 s/token` in both the checkpoint and new runs, with
  the same 12,740,198,400 bytes read per token. The resource ledger also remained at 63 pooled
  buffers and 2,222,752 pooled bytes with no pressure recovery.

A ninth pass made each anonymous scheduler slot own its no-copy Metal wrapper for the lifetime of
that backing allocation. The wrapper has no layer or expert key, and the scheduler's existing lease
still prevents a worker from overwriting the slot before GPU completion. Wrapper creation observes
the Metal working-set limit, is recorded by the resource ledger, and releases the Metal object before
the host allocation is dropped. Non-aligned compatibility payloads retain copied staging. The
deterministic three-token raw output remained `5 2`.

- Under a matched physical-I/O regime, the restored eighth-pass control decoded at `0.394 tok/s`
  (`2.537 s/token`). Two persistent-wrapper runs decoded at `0.422` and `0.417 tok/s`; their mean
  was `0.420 tok/s` (`2.382 s/token`), a 6.6% throughput increase and 6.1% lower latency.
- Attention averaged `0.434 s/token` across the two new runs versus `0.580 s/token` in the control,
  a 25.2% reduction. Expert-command completion fell from `0.012` to `0.009 s/token`, a 29.0%
  reduction. The savings come from avoiding hundreds of repeated driver/VM wrapper creations and
  releases per token, not from changing command math or expert bytes.
- Expert I/O isolated the change: it averaged `1.907 s/token` in the new runs and `1.908 s/token` in
  the control, with the same 12,740,198,400 bytes read per token. This slower storage regime also
  explains why these absolute throughput figures are below the eighth pass's warm-cache evidence;
  the back-to-back comparison, rather than cross-regime tok/s, measures this pass.
- A final ledger-aware warm-cache run decoded at `0.585 tok/s` (`1.708 s/token`) versus the eighth
  pass's comparable `0.542 tok/s` (`1.846 s/token`). Expert I/O was effectively identical at `1.279`
  versus `1.279 s/token`; attention fell from `0.518` to `0.393 s/token`, and expert completion from
  `0.012` to `0.008 s/token`. The resource snapshot reported exactly eight resident expert wrappers
  totaling 169,869,312 bytes, zero active or transient buffers, 63 ordinary pooled buffers, and no
  pressure recovery or resource-limit abort.

A tenth pass tested storage-side strategies without retaining a new runtime path. All comparisons
used the same deterministic three-token request and preserved the raw output `5 2`.

- A positioned-read calibration against the checkpoint's actual 5.1 GB sparse-layer packs sustained
  about 7.1 GB/s with 21,233,664-byte whole-expert reads at 4, 8, and 16 workers. Quarter-slot reads
  likewise sustained about 7.0-7.2 GB/s. Four workers already saturated the device under this access
  pattern, so increasing scheduler width or splitting whole-expert reads did not expose additional
  storage throughput.
- Bounded `F_RDADVISE` lookahead for the preceding token's next-layer routes decoded at `0.598` and
  `0.603 tok/s`, while the immediately restored control reached `0.687 tok/s` and the established
  warm-cache band was `0.574-0.585 tok/s`. The advice neither reduced the selected experts' logical
  read volume nor produced a repeatable latency improvement, so speculative advice was removed.
- A global history cache retained no exact route reuse at its viable roughly 8 GB tier and decoded
  at `0.569 tok/s`; a 32 GB tier collapsed to `0.082 tok/s` under memory pressure. A five-slot-per-
  layer cache did find reuse and reduced positioned-read bytes from 12.74 GB/token to 9.79-9.92
  GB/token, but expert-I/O time increased from the `1.279 s/token` baseline to `1.312-1.314 s/token`
  and total decode remained `0.585-0.595 tok/s`. The eliminated reads were already OS-page-cache
  hits, while the application cache duplicated those pages and added whole-expert copies. It was
  removed in favor of the existing OS-cache contract.
- The local MLX checkpoint declares one next-token-prediction layer but contains no MTP layer
  weights. MTP therefore cannot be enabled as a runtime optimization from these artifacts. Reaching
  `1 tok/s` with supported K=8 quality now requires a typed reduction in physical expert traffic
  (for example, a validated native packed format) or a complete MTP checkpoint that reduces full
  model forwards per accepted token; queue depth, speculative page advice, and duplicate caches are
  closed for this storage layout.

An eleventh pass introduces typed native MXFP4 expert storage for the preferred MLX checkpoint.
The prior affine conversion expanded each routed expert to 21,233,664 bytes. Preserving the source
E2M1 weights and group-32 E8M0 scales reduces a slot to 20,054,016 bytes (5.56%), so K=8 across 75
sparse layers falls from 12,740,198,400 to 12,032,409,600 logical bytes per token. The scheduler's
parallel positioned reads, reusable whole-slot leases, CMD3 handoff, and OS-page-cache policy are
unchanged. Focused import, storage, capability, scheduler, CPU projection, and real-Metal kernel
tests pass. A complete rebuild published all 75 sparse layers and 256 experts per layer without
fallback or retained source shards; each layer pack is 5,133,828,096 bytes and the runtime occupies
about 368 GiB. The deterministic raw output remained `5 2`.

- One detailed run decoded at `0.650 tok/s`; three back-to-back non-instrumented prompts stabilized
  at `0.644-0.645 tok/s`. The latter is a 5.9% throughput increase over the preceding affine-Q4
  `0.609 tok/s` result, matching the 5.56% logical-byte reduction closely enough to identify storage
  volume—not native decode arithmetic—as the gain.
- The detailed native run measured `1.539 s/token`: expert I/O averaged `1.041 s/token`, the fused
  attention/post-attention command averaged `0.459 s/token`, and routed-expert submission averaged
  `0.007 s/token`. The resource ledger retained exactly eight anonymous expert wrappers totaling
  160,432,128 bytes and reported no pressure recovery or resource-limit abort.
- The 12,032,409,600 logical bytes per token leave a best observed effective rate near 7.5 GB/s.
  Native MXFP4 therefore improves the shipped baseline but cannot by itself reach `1 tok/s` on this
  storage device.

A twelfth pass evaluated Colibri's gate/up fusion and macOS direct-read strategies without retaining
a new runtime path. The same deterministic request preserved `5 2` throughout.

- A native-MXFP4 gate/up/SwiGLU kernel removed two dispatches and two temporary-buffer acquisitions
  per routed expert. A real-Metal reference test passed, but detailed routed-expert submission only
  fell from `0.00732` to `0.00677 s/token`, while the GPU-side bucket stayed effectively flat at
  `0.459` versus `0.458 s/token`. The kernel and staged-buffer changes were removed because the
  saving was immaterial beside expert I/O.
- A 12,032,409,600-byte read-only calibration found buffered positioned reads varying between
  6.4 and 7.6 GiB/s and `F_NOCACHE` reads holding about 7.3-7.4 GiB/s. The integrated all-direct
  reader nevertheless decoded at only `0.445 tok/s`, well below the buffered `0.587-0.645 tok/s`
  regimes, because it forced repeated routes back to physical storage. The direct reader was
  removed; buffered `pread` and the OS page cache remain authoritative.
- Colibri's reported 2.24 tok/s Apple-Silicon result uses an approximately 98 GiB process with a
  110 GB expert-cache budget and 74-75% application-cache hit rate on a 128 GB M5 Max. This host's
  roughly 56 GB recommended Metal working set cannot reproduce that hot-tier regime, and pb's prior
  bounded application-cache experiment already showed that duplicating warm OS pages raises I/O
  latency. Direct reads are useful for Colibri's cold misses, not as a replacement for pb's buffered
  repeated-route path.

A thirteenth pass profiled the remaining Colibri-style hot-tier and lower-K options without
retaining either. A deterministic 16-token chat generation produced 38 complete forwards (23
prompt/prefill positions and 15 decode positions) and 22,800 real layer/expert accesses. Simulating
independent per-layer LRU tiers against that exact trace gave:

| Experts retained per sparse layer | Resident native-MXFP4 bytes | Decode hit rate | Remaining logical expert traffic/token |
| ---: | ---: | ---: | ---: |
| 5 | 7.00 GiB | 6.36% | 10.49 GiB |
| 10 | 14.01 GiB | 24.77% | 8.43 GiB |
| 20 | 28.02 GiB | 48.31% | 5.79 GiB |
| 40 | 56.03 GiB | 65.13% | 3.91 GiB |
| 80 | 112.06 GiB | 77.19% | 2.56 GiB |

The 20-entry tier is the first point that could approach `1 tok/s` by byte-count extrapolation, but
its 28 GiB is already near the prior 32 GiB cache that collapsed to `0.082 tok/s` under memory
pressure once resident dense weights and working buffers were present. The 40-entry tier exceeds
this host's recommended Metal working set before accounting for the 10.5 GB resident dense graph.
No application hot tier was reintroduced.

Explicit lower-K routing did cross the numerical target but failed the model-quality contract. K=4
decoded three raw prompts at `1.027-1.078 tok/s` (`1.050 tok/s` aggregate) and a longer chat decode
at `1.248 tok/s`, but the no-thinking arithmetic request `What is 2+2?` answered `40, 1,` instead of
`4`. K=5 decoded at `0.975 tok/s` and also produced an incorrect arithmetic continuation. These
remain diagnostic lower bounds; GLM's configured K=8 is still the only supported routing width.

## DeepSeek V4 Flash Extension Contract

Status: **Design record; not shipped.** DeepSeek V4 Flash must remain outside
`is_flashmoe_hf_model` and normal FlashMoe planning until Gate 8 closes. This is an explicit
unsupported capability, not a llama.cpp fallback or a partially supported family.

The reviewed references are `danveloper/flash-pi-dsv4` at
`3f4741838b567a7b2e562333f90ea6e48637ab2a` and its upstream MIT `antirez/ds4` engine at
`80ebbc396aee40eedc1d829222f3362d10fa4c6c`. The published runtime checkpoint is
`antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
Reference hashes pin the analysis; pb must still validate against the current published checkpoint
and official continuation/logit vectors before claiming support.

DeepSeek V4 Flash is not a GLM MLA adapter. Its production graph requires all of these semantics:

- 43 decoder layers, hidden width 4,096, vocabulary 129,280, 64 attention heads, one 512-wide KV
  head, partial 64-wide RoPE, Q LoRA rank 1,024, grouped output LoRA, 256 routed experts, top-6
  routing, one shared expert, and routed intermediate width 2,048;
- four hyperconnection streams around both attention and FFN, with learned pre/post mixing and a
  20-iteration Sinkhorn split; the output head performs a learned hyperconnection collapse rather
  than consuming an ordinary single residual stream;
- dense raw attention for layers 0-1, then alternating compression ratios 4 and 128; ratio-4 layers
  use the 64x128 indexer and top-512 compressed-row selection, while ratio-128 layers use the
  checkpoint's non-indexed compressed path;
- learned attention sinks, layer-specific compressed RoPE/YaRN parameters, a 128-token sliding
  raw window, and persistent raw plus compressed KV state;
- token-id hash-selected experts for the first three layers; later layers compute
  `sqrt(softplus(router_logit))`, add correction bias for top-6 selection only, renormalize the
  uncorrected selected weights, and apply the checkpoint's 1.5 routed scale;
- Q8_0 resident attention/shared/output tensors, F16/F32 hyperconnection/compressor/indexer state,
  IQ2_XXS routed gate/up tensors, Q2_K routed down tensors, and the exact GGUF tensor/metadata
  contract. Q4 expert checkpoints are a separate typed layout, not a substitution fallback.

The Raspberry Pi profile intentionally reduces routing from the model's top 6 to top 2 for speed.
pb will not adopt that quality tradeoff as its default, hide it behind active-expert configuration,
or use it as correctness evidence. Gate 8 targets top-6 model semantics first. Any later reduced-K
experiment needs an explicit non-production policy, its own quality evidence, and no effect on the
supported graph.

The target remains one FlashMoe-owned scheduler and layer lifecycle. DeepSeek may add typed source,
state, attention, routing, expert-layout, and command implementations; it may not shell out to
`ds4`, depend on a separately running HTTP server, or place a second engine behind
`FlashMoeEngine`. The reference runtime is an oracle and design input, not a vendored production
backend.

The DeepSeek source adapter will validate GGUF metadata and tensor directories before publishing a
source-independent runtime manifest. Routed records keep their typed IQ2_XXS/Q2_K or Q4 layouts in
fixed, page-aligned per-layer expert files so scheduler-owned parallel positioned reads, reusable
whole-expert slots, and the OS page cache remain authoritative. Resident tensors enter one aligned
dense store. Runtime code must never rediscover GGUF types or offsets after graph resolution.

DeepSeek state extends the scheduler declaration with four residual streams plus raw/compressed KV
records and indexer state. Prefix snapshots must capture all of that state atomically. Reusing only
attention KV while rebuilding hyperconnections or compressor state is invalid. Disk persistence is
a follow-on after the in-memory state passes official-vector and prefix-reuse parity; it must not be
inferred from llama.cpp or the reference DS4 server's separate cache implementation.

## Scheduled Graph

Every supported variant resolves the same conceptual stages:

1. Token and position preparation, including an optional vision adapter output.
2. Completion or retained-buffer handoff from the previous layer's deferred CMD3.
3. CMD1 attention projections.
4. Declared attention math and KV/recurrent-state transition.
5. CMD2 output projection, residual, post-attention norm, router projection, and shared gate/up.
6. Declared routing softmax/topK.
7. Scheduler-owned active expert reads into whole-expert slots.
8. CMD3 active experts, shared down, combine, residual, and next-layer input norm.
9. LM head and sampling output.

The resolved graph must select one concrete implementation for every stage. A family-level label is
not sufficient. Resolution must bind actual storage, device, and kernel capabilities, for example:

```text
ResolvedFlashMoeGraph {
    family: Qwen35A17B,
    dense_layout: ResidentQ4,
    expert_layout: FixedQ4(Qwen35A17BLayout),
    attention: CpuQwen35,
    routing: CpuSoftmaxTopK,
    expert_reads: ParallelPositionedWholeSlots,
    cmd1: MetalResidentQ4,
    cmd2: MetalResidentQ4,
    cmd3: MetalFixedQ4,
    state_policy: DeferredGpuNextLayer,
}
```

Closed enums, generics, and layout structs should express variation without `dyn` dispatch in hot
loops. Candidate layout families include:

- `DenseLayout::{ResidentQ4, ResidentBf16, ResidentF16, ResidentF32}`.
- `ExpertLayout::FixedQ4(FixedQ4Layout)` for the first production target.
- Future BF16/F16 expert layouts only when they implement the same slot, scheduler, and CMD3
  contracts.
- `AttentionImplementation` and `RoutingImplementation` values that include placement rather than
  allowing helpers to discover placement at runtime.
- `FlashMoeInputAdapterExecutor::{QwenText, QwenVl}` with a retained concrete capability descriptor;
  the Qwen-VL implementation emits the same typed runtime inputs as text.

Loading fails before allocating runtime state if any selected stage cannot be resolved. Errors name
the family, concrete layout/device, stage, and missing implementation, for example:

```text
FlashMoe unsupported Qwen3-VL FixedQ4/Metal path: CMD3 shared-down implementation is missing.
```

## Ownership Boundaries

- `model_family`: raw Qwen/Qwen-VL config parsing and validation, family-name resolution, model
  dimensions, tensor names, attention schedule, RoPE/MRoPE metadata, K policy, shared-expert shape,
  and typed dense/expert layouts. It must not assign a Qwen3.5 Q4 layout to unrelated families.
- `planning`: backend/model selection, canonical model identity, artifact paths, requested routing
  policy, cache readiness, and cache cleanup. It chooses no runtime implementation; concrete graph
  stages remain the responsibility of `capabilities`.
- `cache`: the pull-time artifact transaction: validate source artifacts, invoke dense conversion
  and expert packing, and atomically publish one runtime cache. Format-specific conversion and
  packing implementations belong to `weights` and `experts`, not to runtime or `legacy`.
- `safetensors`: validated source-header parsing and absolute source-range metadata shared by
  cache importers. It owns no runtime storage or execution policy.
- `capabilities`: concrete model/device/storage capability resolution and precise unsupported
  errors. It returns a fully resolved graph, not a list of family-level stage labels.
- `weights`: resident aligned dense blob, manifest, typed tensor/projection handles, dtype/layout
  validation, manifest-resolved attention runtime layouts, dense caches, and pull-time conversion.
- `experts`: metadata-resolved fixed-slot layouts, validation, PBQ4 import/build compatibility and
  upgrade, layer readers, positioned reads, reusable whole-expert slots, and read metrics.
- `scheduler`: per-token/per-layer execution state, selected implementations, CMD sequencing,
  routing placement, active expert issue/finish, slot leases, and GPU -> SSD -> GPU ordering.
- `metal`: Metal device/queue/pipelines, concrete CMD1/CMD2/CMD3 builders, LM-head builders, KV and
  recurrent-state operations, Obj-C buffer ownership, command diagnostics, and wait policy.
- `math`: placement-independent attention preprocessing, rotary/MRoPE, normalization, vector
  combination, sampling primitives, and CPU reference math. It selects no graph implementation.
- `state`: hidden, residual, normed, KV, recurrent, next-layer, position, and session state with
  explicit CPU-visible and GPU-resident transitions.
- `text`: tokenizer loading and validation, Qwen chat-template rendering, tool-call parsing,
  candidate sampling, and text-generation progress diagnostics. It owns no model execution or
  graph-stage selection.
- `runtime`: engine state, the load transaction, and the model-family-agnostic generation loop that
  asks the scheduler to execute resolved graph stages. It does not inspect storage formats or probe
  optional implementations.
- `vision`: Qwen-VL preprocessing, embeddings, MRoPE, and multimodal position adaptation. It emits
  typed token/position inputs to `runtime` and never branches through the MoE layer loop.
- `legacy`: test-only compatibility and parity fixtures. It is not compiled into production and
  owns no target-architecture behavior.

## Reviewed Baseline

Baseline reviewed on 2026-07-11:

- `experts.rs` owns fixed-slot metadata, pull-time direct/aggregate expert grouping and validation,
  PBQ4/native-Q4/fixed-dense import conversion, source-range fingerprints, atomic layer
  publication, cache compatibility, layer readers, positioned reads, reusable whole-expert
  buffers, raw payloads, and the worker pool. Pull-time packing receives an explicit model,
  expert-directory, and quantization policy instead of depending on the runtime plan.
- The scheduler owns routed expert read issue/finish, read metrics, normalized routes, pending read
  sets, and whole-slot handoff into CMD3.
- Q4 fixed-slot execution rejects PBQ4/component records at the scheduler boundary.
- Production load now asks `ExpertSlotStore` to resolve one Q4/BF16/F16 slot implementation from
  the actual layer metadata. It validates the same format, slot size, expert coverage, and dense
  dtype across every layer before capability resolution. PBQ4 layers are upgraded there to the
  dimension-derived fixed-Q4 layout. The model-name quantization flag no longer selects runtime
  expert storage in `legacy.rs`; mixed or undeclared fixed-dense metadata is a load-time error.
- `model_family.rs` now owns `QwenModelConfig`, including nested Qwen-VL text configuration, RoPE
  metadata precedence, dimension/dtype validation, family-name detection, and derived MoE/attention
  dimensions. `legacy.rs`, `runtime.rs`, `vision.rs`, and expert fixtures consume that one type;
  the duplicate 400-line config and family-name implementation has been removed from `legacy.rs`.
- `planning.rs` now owns `FlashMoePlan`, family-aware routing requests, backend/model selection,
  canonical cache paths, readiness checks, and cleanup. Runtime and vision import the plan directly
  from that owner. Cache readiness no longer substitutes Qwen3.5's 60x512 expert shape when config
  parsing fails; malformed model metadata is an explicit planning error before graph construction.
- `runtime.rs` now owns `FlashMoeEngine` and the complete ordered load transaction from cache status
  through config/layout, metadata-resolved expert storage, dense bindings, typed input adapter,
  required Metal executor, capability graph, scheduler, and tokenizer. The stored capability plan
  was removed once it had produced the scheduler graph. `legacy.rs` no longer constructs the engine
  or selects load-time implementations.
- State descriptors, resident projection descriptors, and many CMD1/CMD2/CMD3 input/output/layout
  records have been extracted.
- Several missing CMD2/CMD3 continuations now produce explicit unsupported errors.
- The Metal shader source, pipeline surface, command diagnostics, and many buffer descriptors live
  in `metal.rs`.
- `metal.rs` is now the sole owner of the Metal/Objective-C runtime bridge: device discovery,
  selectors and message sends, pipeline compilation, mmap buffer wrapping, buffer binding and
  readback, dispatch primitives, command submission, diagnostic waits, retain, and release. The
  duplicate FFI and command-wait implementation has been removed from `legacy.rs`.
- Routing now has only the implementation selected by the resolved graph: Qwen3.5 uses declared
  CPU softmax/top-k. The dormant `route_top4` bool probe, optional Metal pipeline, shader, encoder,
  and CPU substitution branch have been deleted.
- `metal.rs` now owns the reusable transient-buffer pool and the complete resident top-k builder:
  typed Q4/BF16/F16/F32 projection/range validation, logical versus padded output rows, allocation,
  resident logits encoding, vocabulary top-k, submission, readback, and cleanup. Router-score and
  LM-head callers share that builder, and the resolved LM-head path returns a concrete result
  instead of probing `Option` availability.
- The fused resident CMD2 post-attention builder is also Metal-owned end to end. One
  `Cmd2ResidentPostAttentionPrepProjections` binding resolves Q4/BF16/F16/F32 output and router
  projections from manifest metadata, encodes both through the shared resident projection
  dispatcher around residual-add/RMS norm, performs declared CPU top-k readback, and returns
  GPU-resident residual/normed state. `legacy.rs` does not own production behavior for that stage.
- `MetalRuntime` owns device discovery, shader-library compilation, the command queue, every
  required pipeline, partial-construction cleanup, and final release. `MetalExecutionContext`
  owns that runtime together with resident dense and recurrent state; `legacy.rs` no longer
  compiles or releases Metal pipelines, queues, devices, or buffers.
- Production scheduled CMD3 now consumes typed GPU-resident post-attention state, scheduler-owned
  whole-expert slots, resident Q4/BF16/F16/F32 shared projections, routing weights, and typed output
  state in a
  single `MetalScheduledCmd3Builder`. Its submitted result owns the command buffer, slot leases,
  transient/borrowed buffers, deferred GPU handoff, wait/readback, and cleanup. CPU
  normed/residual upload and dense-CPU shared expert substitutions are precise unsupported-stage
  errors. The duplicate legacy CMD3 encoder, shared dense upload/cache, deferred-command type, and
  its model-specific direct-executor test have been removed; typed builder tests plus the required
  real-model smoke cover the production command.
- `LinearAttentionLayout` now lives with recurrent-state contracts in `state.rs`. The active
  Qwen3.5 fused linear-attention CMD1/CMD2 command is owned by
  `MetalFusedLinearAttentionBuilder`: resident Q4/BF16/F16/F32 input/output/router projections,
  typed resident convolution/decay/norm static tensors, recurrent state mutation, gated
  normalization, residual/RMS norm, CPU top-k readback, and GPU-resident CMD3 handoff move
  together. `weights.rs` resolves all ten per-layer bindings once during load; the token loop only
  obtains the already-resolved layer binding and invokes the builder.
- The previous split linear-attention runtime has been deleted: intermediate projection buffers,
  separate recurrence encoders, CPU recurrence and static-weight caches, CPU recurrent session
  state, and `Ok(None)` retries no longer sit beside the fused implementation. Qwen-VL deepstack
  and diagnostic expert skipping now fail as undeclared graph behavior at the layer boundary.
  Deferred full-attention CMD1 and fused linear attention both use the shared typed resident
  projection dispatcher and cannot select an alternate encoder through runtime probing.
- Single Q4 projections now use the same resident `MetalResidentProjectionBatchBuilder` as projection
  batches. The packed/component upload retry, direct single-mmap encoder, and their tests have been
  removed. The unused application-owned F32 LM-head buffer cache, command, and stale dedicated
  F32-only Metal shader/pipeline were also deleted; sampling has only the resolved resident topK
  builder. As a result, `legacy.rs` no longer creates command buffers or encoders, dispatches
  pipelines, or waits for Metal commands.
- `MetalExecutionContext` now owns runtime compilation, device/queue/pipelines, resident mmap
  binding, recurrent-state allocation/reset/release, and the reusable buffer pool. The former
  `MetalExecutorInner` and all recurrent Obj-C lifecycle helpers have left `legacy.rs`; its
  `MetalExecutionFacade` only validates construction policy and calls typed Metal APIs.
- The runtime scopes every token forward pass and vocabulary-sampling command to an Objective-C
  autorelease pool. Retained
  device/queue/pipeline, mmap, recurrent-state, and reusable buffers keep their declared owners,
  while transient command objects are drained after each token instead of accumulating across long
  CLI prefill/decode runs and eventually exhausting Metal allocations. One RAII encoding owner ends
  the compute encoder on every success, error, and early-return path before releasing it or recycling
  referenced buffers; deferred CMD3 explicitly transfers its ended command buffer to the submission.
  The reusable pool uses best-fit allocation so tiny transients cannot consume cached whole-expert
  buffers, and under capacity pressure it retains larger buffers instead of the earliest small ones.
  Command buffers use Metal's unretained-reference mode because pb explicitly owns every encoded
  resource through synchronous completion or deferred-submission transfer. The RAII owner claims
  autoreleased command-buffer and encoder return values with Objective-C's return-value handshake,
  so its explicit releases deallocate completed command objects immediately. Deferred CMD3 command
  creation and its later completion/status polling each run inside a nested autorelease pool. The
  completion pool drains Metal's committed-command auxiliaries at the layer boundary instead of
  leaving their expert-buffer references alive until the outer token autorelease pool drains.
  Scheduler `pread` expert payloads use page-aligned reusable whole-slot storage. Scheduled CMD3
  wraps aligned fixed slots as non-owning shared Metal buffers; its submission owns the slot leases
  until the command completes and releases the wrappers before the bytes return to the worker pool.
  Non-aligned compatibility payloads are copied into transient Metal staging buffers. Successful
  copied commands return those allocations to a separate, bounded 16-buffer staging pool; every
  checkout overwrites the whole payload, the pool carries no layer/expert key, and it is not an
  application expert cache. It cannot enter or evict the general reusable pool. Error cleanup and
  working-set pressure still mark copied staging resources purgeable-empty and release them. A
  long-prefill VM profile showed that unbounded ordinary release left one 2,654,208-byte
  IOAccelerator mapping per expert projection binding; separating and bounding the copied fallback
  limits that mapping set while the aligned path avoids the second whole-expert memory copy.
  The session cache removes a non-prefix-matching entry before allocating the replacement KV cache.
  Harness workflow stages intentionally share a logical session id while changing their system
  prompts, so stale planning/review state cannot remain resident throughout a fresh stage prefill
  and transiently double CPU KV memory. Structured agent requests also enforce their declared
  `ctx_size` against prompt plus generation capacity before allocating KV state.
  Structured FlashMoe output carries the actual rendered prompt-token count and distinguishes EOS
  from exhaustion of the requested generation cap. Capped, unparsable workflow turns therefore
  enter bounded retry/recovery instead of being misreported as prose finals. Recovery and
  terminal-submission-only turns disable emitted Qwen reasoning so their budget remains available
  for the required native function call. Workflow prompts use that single native tool-call dialect;
  unterminated Qwen tool-call envelopes are accepted only while a capped completion is being
  surfaced for retry, and rejected as malformed on a claimed end-of-generation result. If a capped
  output already attempted the stage's terminal function, the next recovery turn exposes only that
  function and requests the smallest valid minified artifact instead of regenerating a verbose plan.
  If a new allocation still fails, the pool releases all currently idle buffers and retries once,
  reporting the requested and released byte counts plus Metal's current and recommended working-set
  sizes on failure. The target runtime also owns a `MetalResourceLedger` beside the reusable pool.
  It accounts for resident dense mappings, recurrent state, active general buffers, idle pooled
  buffers, transient expert staging, and in-flight commands, and samples Metal's
  `currentAllocatedSize` at token boundaries. The default fail-safe reserves 10% of
  `recommendedMaxWorkingSetSize` (at least 1 GiB, capped at half the recommendation on small
  devices); hidden harness `infer`/`bench` callers may lower, but not raise, that limit with an
  explicit CLI argument. Idle pooled buffers are drained and the device is resampled before a
  structured resource-limit abort. Successful token boundaries require zero active transient and
  in-flight ownership. Detailed ledger snapshots are opt-in harness output; production logging is
  limited to pressure recovery and abort diagnostics rather than per-buffer or per-token messages.
- `MetalResidentProjectionBatchBuilder` now owns the Q4/BF16/F16/F32 resident projection batch
  used by full-attention CMD1 and deferred state. `weights.rs` resolves one
  `ResidentMmapMatvecProjection` from manifest dtype/quantization metadata; CPU or GPU input
  binding, fused compatible-Q4 dispatch, typed per-projection dispatch otherwise, packed GPU
  output handoff, optional readback, timing, and cleanup remain one implementation. The three
  duplicate encoders and fused-batch helper have been removed from `legacy.rs`.
- `weights` now resolves every configured shared-expert gate/up/down/router projection into one
  per-layer `SharedExpertWeightTable` during model load. The generation loop no longer discovers or
  caches Q4 shared bindings per token. CMD3 consumes the same `ResidentMmapMatvecProjection` type
  and Metal encoder as CMD1, CMD2, and LM-head sampling, so Q4/BF16/F16/F32 differ only by the
  selected resident projection implementation.
- `runtime.rs` now owns `forward_token_input`, the production token/layer loop, deferred CMD3 handoff,
  attention execution, CMD2/routing composition, active expert issue/finish, CMD3 submission,
  final norm/state recording, and public text/session/multimodal generation orchestration. The
  dead CPU dense shared-expert runtime branch was removed; supported CMD3 preparation requires
  load-resolved resident shared projections.
- `text.rs` now owns tokenizer loading/validation, Qwen chat templates, tool-call rendering and
  parsing, candidate sampling, and generation progress diagnostics. Runtime and weights consume
  that owner directly. The duplicate hand-rolled word-level/byte-BPE test adapters were deleted
  rather than carrying `legacy.rs`'s broad dead-code allowance into the extracted module.
- `weights.rs` now derives the dense transformer runtime layout from config plus the concrete
  tensor registry: per-layer full-versus-linear attention, standard-versus-gated Q projection,
  head/KV/rotary dimensions, linear-attention dimensions, and runtime tensor storage/shape
  validation move as one load-time responsibility. Runtime consumes the resolved layouts and does
  not infer them in the generation loop.
- `math.rs` now owns full-attention Q/gate splitting, Q/K normalization, rotary and multimodal
  rotary application, vector combination, sigmoid, and the linear-attention CPU reference helpers.
  Runtime and weights import those operations directly; `legacy.rs` no longer owns production
  full-attention preprocessing or normalization math.
- `cache.rs` now owns the complete public pull-time transaction from required HuggingFace artifacts
  through config/tokenizer validation, manifest construction, aligned dense-store publication,
  expert-layer publication, and final cache plan. CLI callers use that owner directly; the
  transaction and its safetensors/source-range helpers no longer live behind `legacy.rs`.
- Pull, infer, bench, and cache-clean now carry one explicit `q4`/`bf16`/`f16` expert-storage
  policy into planning and cache construction for every Qwen family. Each layout has a distinct
  cache namespace, and fixed-BF16/F16 cache import rejects a source tensor whose declared dtype
  does not exactly match the selected layout. Q4 remains the default production policy; the
  official Qwen3.5 BF16 model URI retains its compatibility default without imposing that choice
  on other Qwen-family models.
- Production load binds that requested policy to the metadata-resolved whole-expert slot layout
  before PBQ4 compatibility upgrade or graph construction. Q4, BF16, and F16 resolve only against
  their matching fixed-slot layout; all six mismatches are precise expert-storage-policy errors
  naming both the requested and actual layout, with no cache mutation or runtime fallback.
- `legacy.rs` is deleted and no module declares or imports it. Runtime owns timing aggregation; weights
  owns required-manifest and resident-Q4 graph-binding validation plus the F32 Accelerate matvec;
  and the unused runtime expert-store adapter has been deleted. The test-only `ExpertWeights`
  decoded/component execution adapter and CPU expert-phase substitute are also gone. PBQ4 tests
  inspect import records through experts-owned helpers; fixed-slot execution tests use
  `FixedQ4ExpertPayload` or scheduler-owned `ScheduledExpertSlot` directly. This completes Gate 7's
  legacy-removal criterion; only sustained equivalent upstream comparison remains open.
- Before deletion, the test-only legacy boundary stopped suppressing dead-code warnings. Unused synthetic-runtime
  gates, decoded-expert methods, tokenizer data, and an unregistered duplicate state test were
  deleted; deterministic synthetic dense-weight hashing moved to the `weights.rs` fixture that
  consumes it. The remaining active parity contracts now compile only under their target owners.
- Gate 7 removed the duplicate Accelerate-backed gated-delta execution path, CBLAS bridge, and
  scratch dispatch from the test-only legacy boundary. The math owner now retains the direct
  value-major recurrence oracle together with QK normalization/rotary, grouped-query attention,
  causal convolution ordering, and routing parity. Their legacy tests and scalar recurrence helper
  were removed; production recurrence remains solely the resolved Metal stage.
- Exact duplicate PBQ4 metadata parser/error tests and Qwen3Next norm-offset detection coverage
  were deleted from `legacy.rs`; their registered `experts.rs` and `weights.rs` owner tests remain.
- Planning now owns its model alias, typed cache layout, missing-artifact, stale-runtime cleanup,
  source-shard cleanup, and MLX shard-name tests. The duplicate planning/cache-policy block and its
  stale model-family import were removed from `legacy.rs`; partial expert-layer readiness now uses
  expert metadata directly under the planning owner. Expert buffer recycling and stale temporary
  file cleanup are tested beside the expert store, and all three legacy copies have been removed.
- Scheduler-owned positioned reads now have an owner-local PBQ4 import-store integration fixture
  that proves routed order, normalization and scale, warm-read accounting, and worker failure
  metrics. Six overlapping coordinator tests and the duplicate expert-I/O guardrail test were
  removed from `legacy.rs`; the lower-level scheduler and expert policy tests retain each contract.
- Text now owns the shared tokenizer fixtures and its sampling, chat-template, Qwen/Qwen-VL tool
  serialization, parser, and byte-level BPE parity suite in `text_parity_tests.rs`. Cross-owner
  integration fixtures borrow those test-only owner fixtures; the 782-line tokenizer/text block
  and a duplicate non-FlashMoe backend-selection test have left `legacy.rs`.
- Vision now owns Qwen-VL image preprocessing, placeholder expansion, multi-image span, MRoPE,
  config validation, and patch-order parity in `vision_parity_tests.rs`. The 741-line adapter test
  block and its broad legacy imports were removed from `legacy.rs`; these tests remain explicitly
  upstream of the shared decoder graph.
- State now owns tool-call rerender prefix reuse in `state_parity_tests.rs`; its existing tests
  already cover cache transfer, shallow CPU KV snapshots, recurrent placement rejection, and prefix
  boundaries. The five-test legacy session block and the now-dead state imports were removed rather
  than duplicating those owner contracts.
- Model-family tests now own validated runtime dimensions and model-config routing defaults;
  planning owns explicit K overrides and the Qwen3.5 K<4 force guard. The synthetic legacy family
  fixture and eight overlapping config/routing tests were removed, so family selection is no longer
  restated inside the compatibility boundary.
- Shared safetensors byte builders, typed tensor fixtures, expert triplets, and numeric assertions
  now live in the explicit `cfg(test)` `test_fixtures` module. Cache, expert, and weights tests can
  move independently without importing helper ownership from `legacy.rs`; the duplicate helper
  block has been removed there.
- Experts now owns PBQ4 quantization/projectability, native-Q4 fixed-slot adaptation, on-disk PBQ4
  rewrite, import-only record projection, and metadata encoding parity in
  `experts_parity_tests.rs`. Reusable expert-layer fixture construction moved to `test_fixtures`;
  the corresponding legacy tests, helpers, and dead imports were removed.
- Cache now owns runtime artifact emission, Qwen-VL artifact readiness, safetensors-index ingestion,
  and Qwen3 QK-norm/shared-expert manifest classification in `cache_parity_tests.rs`. These assembly
  tests no longer sit between unrelated math, weights, and expert tests in `legacy.rs`.
- Expert parity now also owns aggregate Qwen3.5 and MLX switch-MLP splitting, native-Q4 passthrough,
  fixed-slot layout, and source-reuse invalidation. The contiguous packer block and its model-layout
  imports were removed from `legacy.rs`; cache manifest construction is imported only by the one
  expert import test that exercises that boundary.
- Weights now owns the dense registry/manifest validators, attention and linear-attention layouts,
  resident BF16/Q4 projections, Metal batch and post-attention parity, dense-store caching, runtime
  shape, and LM-head integration suite in `weights_parity_tests.rs`.
- The final sampling/topK, routing/Q4 math, expert edge-case, dense validation, and rope tests moved
  to text, math, experts, weights, cache, and vision parity modules. Duplicate tokenizer/routing and
  Qwen3Next norm tests were deleted, as was the ignored legacy-only Metal compile fixture; release
  build plus the required smoke are the active Metal construction proof. `legacy.rs` and its module
  declaration are now gone. The post-removal suite passes 658 tests with six device-dependent tests
  ignored; web assets and release rebuilt, and the required default smoke printed `4`.
- Whole-slot raw reads no longer carry test-only slot-spec or recycle-pool fields used by the
  deleted adapter. Fixed-Q4 payloads no longer retain an optional decoded scale/bias component
  cache; CPU reference projection decodes the authoritative whole-slot bytes on demand, while the
  scheduled Metal path continues to consume typed offsets into those same bytes.
- Pull-time dense conversion is now weights-owned: MLX native-Q4 companion resolution, logical
  shape and quantization metadata, aligned offsets/padding, mmap source reads, post-hoc Q4
  conversion, and dense/vision store publication moved together. `safetensors.rs` owns shared,
  bounds-checked source-header parsing. Pull-time expert conversion is likewise experts-owned:
  direct and aggregate layer splitting, fixed-slot selection, PBQ4/native-Q4/fixed-dense record
  construction, source hashing and range reads, reuse validation, and atomic layer publication
  moved together. `cache.rs` only coordinates those owners.
- `FlashMoeExecutionScheduler` now owns the resolved graph and the sole production expert-read
  coordinator. `runtime.rs` resolves CMD1, attention placement, CMD2, and routing through that
  owner; it no longer calls graph builders or expert issue/finish APIs directly. CMD3 uses a typed
  scheduler transaction that issues routed reads, permits shared/next-norm preparation while they
  are pending, finishes whole-slot leases, builds and submits CMD3, and returns read metrics plus
  recurrent mix inputs. The duplicate engine-owned expert-store clone has been removed.
- The scheduler layer transaction now starts from an explicit initial, CPU-visible, or deferred-GPU
  previous-CMD3 handoff and consumes its phase through CMD1, CMD2, routing, pending whole-slot
  reads, CMD3 submission, and a scheduler-selected complete/defer output handoff. Per-layer full
  CPU-KV versus fused-linear-Metal attention is resolved once from the tensor manifest, validated
  layer-by-layer against the model-family layout by capability resolution, stored in the concrete
  graph, and carried by the transaction. A count or kind mismatch is a named CMD1 capability error.
  `runtime.rs` no longer constructs a parallel scheduler table, and scheduler construction cannot
  accept one. The tiny-fixture expert-skip runtime and dormant Qwen-VL layer loop were deleted. Text
  lookup and exact precomputed visual embeddings enter the same typed token-input boundary.
  Declared per-layer additions force a scheduler-owned complete-here CMD3 handoff, omit stale
  next-norm output, and are applied before the next shared layer transaction.
- Gate 4 has moved `DenseStore` as one owner into `weights.rs`: the mmap/blob and registry,
  resident tensor and norm caches, Q4 projection bindings, decoded/raw tile accounting, projection
  batches, CMD2/routing preparation, LM-head candidates, dtype decoding, and focused cache/read
  diagnostics moved together. Load, runtime, and sampling use that owner directly. Synthetic
  projection, CPU dense router-topK, raw-tile, and alternate projection helpers are test-only; they
  are no longer dormant production continuations hidden by `legacy.rs`'s broad dead-code allowance.
- `state.rs` now owns the CPU-visible `KvCache`, prompt/generated/recurrent records, full-attention
  KV insertion and causal lookup, capacity growth, and shallow session snapshots. The pure causal
  attention math moved to `math.rs`; `runtime.rs` owns generation/session orchestration without
  direct access to cache internals. Focused state tests prove shared snapshot storage, independent
  growth, and explicit rejection of undeclared GPU recurrent state.
- `FlashMoeGenerationState` and `FlashMoeSessionCache` now own the complete CPU-visible generation
  lifecycle: rendered prompt tokens, KV cache, reusable-prefix position, cached final hidden state,
  prompt snapshot, generated tokens, decode position, EOS/length stop state, and session-map
  commit. The engine generation loop borrows typed prefill/sample/decode views from that owner;
  its raw session map, local generated/stop state, and `store_session_cache` helper were removed.
- Session reuse now captures a typed CPU-visible snapshot of every resolved Metal linear-attention
  conv/SSM recurrent buffer at the prompt boundary and restores it before cached prefill or decode.
  `metal.rs` validates the complete layer table before writing any buffer, rejects missing/extra or
  mismatched recurrent layers, restores recurrent bytes, and clears transient conv/delta/gate
  outputs. Reset lock failure is also an error rather than a silent no-op. State tests cover typed
  snapshot shape/order and lifecycle transfer; a local-Metal test proves the real buffer round trip.
- Managed inference now obtains FlashMoe through a process-level per-model runtime pool. The pool
  locks only for model calls, so tool execution and nested advisory work do not retain a global
  inference lock; idle runtimes are reaped and the resident model count is bounded. A logical
  session keeps a safe prompt checkpoint plus an evaluated generated-token head and selects the
  longest exact match. The first stable system/tool prefix is a separate content-addressed
  checkpoint, so another session can skip that prefill without sharing later conversation state.
- `flashmoe-session-v1` serializes populated full-attention KV and compressed MLA records, typed
  linear-attention conv/SSM snapshots, final hidden state, and exact token ids. Model/runtime and
  token hashes reject stale state; owner-only atomic files and an explicit byte budget bound local
  persistence. Durable session state commits only the canonical prompt boundary; the speculative
  generated head remains memory-only. Agent events and the web transcript report cache source, reused tokens,
  actually-prefilled tokens, and restore latency. Hidden `harness infer --session-id --repeat`
  exercises live and restart restoration through the production path.
- Deferred CMD3 ownership is now Metal-native. `MetalScheduledCmd3Submission` owns command/buffer
  lifetime and exposes validated `MetalStateBuffer` views for hidden and next-layer normed state;
  `runtime.rs` no longer wraps raw pointers or carries a non-Metal ready continuation. The
  unsupported-platform submission representation is uninhabited, while token execution has one
  Apple-Silicon scheduled body and one entry-point unsupported error. Required resident-Q4 facade
  calls return concrete results/errors instead of boolean probes or `Ok(None)` continuations.
- Full-attention placement is now resolved by the scheduled graph rather than supplied by the
  runtime call site. Qwen3.5 selects its declared upstream-parity CPU KV implementation; an
  attention stage without a matching scheduled executor is a named unsupported capability. The
  dormant Metal KV cache allocation, runtime branches, four pipelines, shaders, and legacy
  encoders have been deleted, so CPU placement cannot silently switch to an undeclared GPU path.
- Qwen3.5 full-attention Q/K normalization and rotary preprocessing now match that declared CPU
  placement directly. The dead optional Metal RoPE/RMS probes, unary dispatcher, shaders,
  pipelines, and false CMD1 kernel requirements have been removed. The same ownership cleanup
  deleted standalone F32/BF16 dense Metal matvec, mmap, batch, and router-topK encoders; their
  supported layouts now enter the same resident projection and topK builders as Q4.
- Qwen3.5 Q4 capability planning now consumes one resident dense layout resolved by `weights`, a
  fixed-Q4 execution descriptor validated against every expert layer's metadata and file size, and
  the kernel surface from a successfully compiled Metal executor. The live load path builds the
  scheduled graph only after those concrete facts resolve.
- A production `FlashMoeEngine` now owns a required `MetalExecutionFacade` backed by a
  Metal-owned `MetalExecutionContext`. Metal-disabled and non-Apple construction fail explicitly,
  so a graph that selects required Metal stages cannot be represented by an engine with an absent
  executor.
- Scheduler-owned Q4 CMD3 payloads now validate BF16 scale/bias layout and aligned gate/up/down
  views into one whole-expert slot before encoding. The active-expert encoder has one fused
  whole-slot implementation; its component-upload and unfused SwiGLU substitution have been
  removed.
- Q4 CMD2 post-attention projection, residual/norm, and routing prep now return one required Metal
  result for both CPU-visible and Metal-resident attention values. Missing resident weights,
  projections, norm weights, or buffers are named errors instead of `Ok(None)` continuations.
- Normal Qwen3.5 linear-attention layers now require one fused resident CMD1/recurrence/CMD2
  implementation. A weights-owned table resolves four input projections, convolution/A-log/delta
  bias/norm static tensors, output projection, and router before capability resolution. Missing or
  incompatible bindings are named load errors; the live graph no longer performs per-token tensor
  lookup or retries through an intermediate Metal-values path or CPU recurrence.
- Load now binds every Qwen3.5 Q4 CMD1, CMD2, shared-CMD3, linear static-state, and LM-head
  projection against the actual manifest and resident byte ranges before capability resolution.
  Deferred next-layer state, CPU full-attention placement, resident projection batches, and
  resident Q4 LM-head sampling are fixed graph policy rather than runtime probes.
- Fixed-Q4 offsets are no longer model-family metadata. `experts` derives a checked BF16-scale/bias
  whole-slot layout from the concrete hidden/intermediate dimensions and fixed storage group size;
  native Q4 aggregate packing, store construction, and capability validation consume that same
  descriptor. A Qwen-family 4096x1536 expert shape therefore resolves a 10,616,832-byte record
  instead of inheriting Qwen3.5's 7,077,888-byte record. Shapes that cannot satisfy the runtime
  fixed-slot contract remain PBQ4 import data rather than entering CMD3.
- Routing-weight normalization and routed-expert scaling are now resolved graph policy. The
  scheduler constructs its expert-read coordinator from that policy, so production cannot pair a
  graph with different route math. Qwen3.5 and Qwen MoE use the upstream contract: softmax over all
  router logits, fixed-slot top-K selection, selected-route renormalization, and routed scale 1.0.
  Qwen MoE advances only when `norm_topk_prob=true`, while a missing value or `false` reports the
  routing stage as unsupported.
- Norm-weight semantics are concrete model metadata rather than a value-based runtime probe.
  Qwen3.5, Qwen, and Qwen-VL cache tensors are multiplicative RMSNorm weights; an actual
  `qwen3_next` config selects the typed `1 + weight` implementation. Removing the former
  mean-below-0.75 heuristic fixed the first real-checkpoint divergence at Qwen3.5 layer 35: its
  Q/K/V projections, gated attention, CMD2 residual, and generated continuation now match
  upstream.
- The declared CPU-KV attention stage is now a Qwen-family full-attention implementation rather
  than a Qwen3.5 label. Full-attention manifests and runtime CMD1 execution require both per-head
  Q and K RMSNorm bindings; an absent tensor is an explicit load/runtime error instead of silently
  running unnormalized attention.
- CMD3 treats a model configuration with zero shared experts as the declared no-shared-expert
  implementation. Models that declare shared experts must resolve every resident
  Q4/BF16/F16/F32 shared projection; a missing projection or invalid declared shape is an
  unsupported binding error and cannot collapse into the no-shared case. The supported single
  shared expert applies its sigmoid router gate after the down projection, matching upstream;
  configurations with multiple shared experts remain an explicit unsupported capability.
- Text-only Qwen MoE Q4 now resolves the same nine-stage scheduler graph as Qwen3.5 Q4. Its model
  metadata selects full attention, configured K, routed scale 1.0, selected-route normalization,
  dimension-derived fixed-Q4 slots, and the declared no-shared CMD3 source. Production load applies
  the same manifest/range binding validation to both text families. Metal kernel requirements are
  resolved from layer and shared-expert metadata: hybrid Qwen3.5 requires its linear kernels, while
  no-shared Qwen MoE does not falsely require the shared-activation kernel.
- Capability coverage now resolves Qwen3 text and Qwen3-VL through the same graph for every
  supported BF16/F16/F32 resident-dense layout paired with fixed-BF16 or fixed-F16 whole-expert
  slots. Family, input adapter, dense dtype, and expert dtype change typed stage inputs only; all
  combinations retain the same attention, positioned-read, CMD3, and LM-head stage contracts.
- Expert count and active K are concrete graph values rather than generation-loop inputs. The
  resolved CLI/model routing policy is applied to model metadata before capability resolution, and
  production CMD1/CMD2/router construction consumes the resulting scheduler-owned values.
- A linked Qwen MoE Q4 fixture resolves the Qwen3 graph and follows one K=8 full-attention
  transaction through selected-route softmax at scale 1.0, eight scheduler-owned positioned reads,
  whole-slot typed Q4 payloads through staged gate/up, SiLU product, and down projection, declared
  no-shared CMD3 combine, and deferred hidden/next-norm state. Its route weights and output state
  are checked against independent golden values.
- The production Q4 `mlx-community/Qwen3-30B-A3B-4bit` checkpoint now builds through the same cache
  transaction: all 48 layers publish 128 fixed-Q4 whole-expert slots, load resolves the nine-stage
  graph with K=8 and no shared expert, and real one-token inference exits successfully. Raw
  `1+1=` emits `2` in both FlashMoe and upstream MLX-LM. Raw `2+2=` emits `5` in both engines for
  this checkpoint, proving that result is checkpoint behavior rather than a reason to alter the
  unified math or restore a fallback. After the upstream route and norm contracts were aligned,
  the raw Qwen3.5 `2+2=` smoke selects token 10992 (`？`) in both engines; the earlier pb-only `4`
  was evidence of divergent math and is not a parity target.
- `vision.rs` now owns Qwen-VL image decoding, smart resize, normalization, block-major patch
  packing, the serialized ViT configuration contract, visual encoding output, M-RoPE positions,
  placeholder-span validation, multi-image embedding/DeepStack assembly, and the typed
  `QwenVlRuntimeInputs` emitted before decoder execution. Its cursor validates exact placeholder,
  embedding, M-RoPE, and DeepStack cardinality and emits resident-text or precomputed-visual
  `FlashMoeTokenInput` values. The multimodal prefill facade feeds those values through the same
  scheduler/CMD runtime as text; there is no Qwen-VL decoder loop or embedding fallback.
- `VisionEncoder` construction, image-to-patch entry, patch projection, learned position
  interpolation, block-major coordinate policy, spatial rotary math, transformer residual blocks,
  self-attention, ViT MLP execution, DeepStack/spatial merging, dense bias application, and
  normalization are owned by `vision.rs`. `legacy.rs` no longer defines or implements the image
  preprocessor, vision config, or vision encoder.
- The duplicate legacy Qwen-VL placeholder expansion and multimodal position implementation has
  been deleted. Existing single/multi-image parity fixtures now call narrow test adapters around
  `vision.rs`'s production `expand_image_placeholders` and `multimodal_mrope_positions`, so their
  span, grid, wrapping, and position assertions exercise the same algorithm used by typed runtime
  inputs.
- Load now retains exactly one `FlashMoeInputAdapterExecutor` instead of probing an
  `Option<VisionEncoder>` at request time. Qwen-VL construction requires concrete weight and
  manifest artifacts, validates required vision tensor names plus adapter dimensions/DeepStack
  metadata, and contributes a `QwenVlTypedInput` implementation to the same nine-stage Q4 graph.
  The already resolved model family selects the executor; incidental vision metadata in a
  Qwen3.5 config cannot redirect its text graph into Qwen-VL.
  A text adapter bound to Qwen-VL, a VL adapter bound to text, absent artifacts, invalid metadata,
  or missing required vision tensors is a token-input-stage error before inference.
- Qwen3-VL planning now recognizes real MoE repository names such as
  `Qwen3-VL-30B-A3B-*`, while dense names such as `Qwen3-VL-8B-*` remain outside FlashMoe. The
  production Q4 evidence target is
  `hf://mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`; the former placeholder
  `Qwen3-VL-MoE-Instruct` name no longer determines whether vision artifacts are planned. Pull
  classification coverage moved from `legacy.rs` to the planning owner.
- The real Qwen3-VL MLX Q4 repository carries a stale BF16 13-shard index beside four actual Q4
  shards. `safetensors.rs` now resolves one typed pull-time manifest source: it uses the declared
  index only when every referenced shard exists, otherwise it deterministically builds the tensor
  map from all actual validated shard headers and rejects duplicate tensor ownership. Weights-owned
  canonicalization maps the real `language_model.*` and `vision_tower.*` namespaces into the same
  runtime tensor names used by Qwen text and the typed vision executor.
- `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` now builds all 48 layers of 128 fixed-Q4 expert
  slots plus separate dense and vision stores, loads a concrete Qwen-VL adapter and the same
  nine-stage scheduler graph, emits `2` for a real text-only `1+1=` inference, and completes a real
  image request through preprocessing, vision encoding, placeholder/M-RoPE/DeepStack preparation,
  and the shared decoder. The CLI `--image` option calls the existing structured multimodal API;
  it does not introduce a decoder loop or runtime probe.
- Focused parity/reference tests cover expert layout and math, routing contracts, attention and
  recurrence primitives, state descriptors, and Metal buffer-plan contracts.
- Gate 5 now has a linked Qwen3.5 Q4 per-layer golden derived independently from the upstream
  equations. It follows CPU-KV scaled-dot-product attention, deferred CMD1 state, Metal-placed CMD2
  descriptors, fixed router scores, K=4 IDs/weights, four real scheduler-owned positioned reads of
  whole fixed-Q4 slots, typed gate/up/down offsets, SwiGLU expert output, shared output,
  residual/hidden state, declared CMD3 output, and next-layer RMSNorm. The stale ignored test that
  mislabeled a tiny Qwen3 model as the production Qwen3.5 URI was deleted rather than preserved as
  an alternate runtime fixture.
- A second deterministic fixture carries a valid two-layer Qwen3.5 linear-attention prefix through
  the resolved graph. It proves deferred hidden/next-normed handoff, recurrent mixing, eight
  scheduler-owned K=4 positioned reads, terminal hidden state, logits, and top candidates without
  inventing a full-attention layer outside the family schedule.
- Local-Metal reference tests exercise both the resident-Q4 fused projection batch and one mixed
  BF16/F16/F32 batch, Q4 and BF16/F16/F32 fused CMD2 preparation, and scheduler-issued CMD3
  active-expert plus Q4/BF16/F16/F32 shared-expert combine against independent CPU math. Route IDs
  are exact and
  floating-point route scores use an explicit numerical tolerance rather than bitwise equality
  across CPU and Metal implementations.

The architecture implementation is at the target and Gate 7 comparison evidence is complete:

- The engine container, load path, and public generation/session orchestration are now in
  `runtime.rs`, tokenizer/sampling ownership is in `text.rs`, and dense runtime-layout resolution
  is in `weights.rs`. The extracted `cache.rs` transaction delegates dense conversion to
  `weights.rs`, expert packing to `experts.rs`, and source-header parsing to `safetensors.rs`.
  Owner-local parity modules and the shared `cfg(test)` binary-fixture module replace the deleted
  test-only compatibility boundary.
- General caches and explicitly diagnostic/test helpers still use `Option` for data availability,
  but no supported graph-stage implementation or CPU/GPU placement is selected from those values.
- Text-only Qwen MoE Q4 and Qwen-VL Q4 now have linked fixtures plus real-checkpoint cache, load,
  and inference evidence through the shared runtime. Qwen-VL additionally has a real image request
  through its typed pre-MoE adapter; it has no alternate decoder loop.
- BF16/F16/F32 full-attention CMD1, CMD2, and LM-head sampling now use the same typed resident
  projection handle, Metal dispatch, CPU/GPU input bindings, residual/norm transition, router
  readback, state handoff, padded-vocabulary policy, and topK command as Q4. Qwen3/Qwen3-VL
  non-Q4 dense plus fixed-Q4 expert graphs resolve all nine stages; real-checkpoint validation is
  future hardening rather than a separate runtime implementation. Qwen3.5 hybrid BF16/F16/F32 now
  resolves all nine stages, including its resident
  shared-expert CMD3 implementation. Typed fixed-BF16 and fixed-F16 active-expert slots use the
  same positioned reads, scheduler leases, CMD3 handoff, and Metal builder as fixed-Q4 slots. The
  explicit expert-storage cache policy emits fixed-BF16 or fixed-F16 metadata for compatible
  source checkpoints, which production load resolves without a model-name branch. Source dtype
  mismatches are cache-build errors rather than conversion or runtime fallback. Additional real
  non-Q4 checkpoints remain useful validation, but their implementations already resolve through
  the same graph and have direct storage, capability, scheduler, and local-Metal reference evidence.

At this checkpoint, Gates 1 through 7 are complete. Qwen3.5, Qwen3 text, and Qwen3-VL Q4 have
resolved production graphs and real-checkpoint evidence; typed non-Q4 implementations use the same
contracts. The equivalent sustained upstream comparison is recorded below, and focused tests, the
all-target suite, release build, required smoke, and exact 32-token continuation were rerun after
the final parity correction.

## Completion Gates

Architecture progress is reported against the gates below. A gate is complete only when all of its
exit criteria hold in production code and tests.

| Gate | Status | Architectural result |
| --- | --- | --- |
| 1. Concrete Graph Resolution | Complete | Support is resolved from the real model and device. |
| 2. Concrete Metal Builders | Complete | Metal command execution is owned by `metal`. |
| 3. Scheduler-Owned Runtime | Complete | The scheduler executes the layer lifecycle. |
| 4. Weights And State Ownership | Complete | Runtime storage and state have single owners. |
| 5. Qwen3.5 Q4 Correctness Closure | Complete | The first production graph has parity evidence. |
| 6. Unified Variant Implementations | Complete | Other variants use the same graph/runtime. |
| 7. Legacy Removal And Benchmarking | Complete | Compatibility boundary removed; equivalent sustained comparison and post-comparison correctness recorded. |

### Gate 1: Concrete Graph Resolution

Objective: make support a load-time fact rather than a runtime guess.

Required work:

- Replace family-only capability declarations with a concrete execution specification built from
  model config, tensor manifest, expert metadata, platform/device, available pipelines, and policy.
- Select exactly one dense layout, expert layout, attention implementation, routing implementation,
  CMD1/CMD2/CMD3 implementation, state policy, and input adapter.
- Parameterize or relocate Qwen3.5 constants so unrelated families do not inherit its expert layout.
- Make a missing Metal executor or kernel fail graph resolution when a selected stage requires it.
- Keep Qwen MoE, Qwen-VL, and unimplemented BF16/F16 expert graphs as precise unsupported errors.

Exit criteria:

- A supported Qwen3.5 Q4 model resolves all nine concrete stages against its real manifest and
  device.
- The supported engine cannot contain an optional implementation for a required stage.
- Capability tests cover successful Qwen3.5 Q4 resolution and named family/dtype/layout/device/stage
  failures.
- No graph-stage choice depends on trying an implementation and observing `None` or `false`.

Completion evidence:

- The live Qwen3.5 Q4 load resolves real dense byte ranges, fixed expert metadata/files, compiled
  Metal kernels, routing policy, input adapter, state policy, and all nine stage implementations.
- Required Metal is a construction invariant; unsupported family, dtype, layout, device, and
  missing-stage combinations fail before inference.
- Production Qwen3.5 execution has one declared attention placement, deferred-state policy,
  CMD1/CMD2/CMD3 flow, whole-slot expert source, and resident Q4 LM-head implementation.
- Focused capability/parity tests, all-target tests, release build, and the required smoke provide
  the checkpoint verification recorded by the Gate 1 completion commit.

### Gate 2: Concrete Metal Builders

Objective: move command execution, not just command descriptions, behind the Metal boundary.

Required work:

- Move `MetalExecutor`, `MetalExecutorInner`, queue/device/pipeline ownership, Obj-C buffer lifetime,
  and complete encoder functions from `legacy.rs` to `metal.rs`.
- Expose concrete CMD1, CMD2, CMD3, KV/recurrent, and LM-head builders selected by the resolved
  graph.
- Make builders consume typed state plus weight/expert handles and return submitted/deferred typed
  state.
- Replace fused/unfused and component-upload discovery with resolved implementations or explicit
  unsupported errors.

Exit criteria:

- `legacy.rs` contains no Metal command encoder, pipeline dispatch, Obj-C buffer lifecycle, or
  command-buffer wait implementation.
- Runtime code cannot call low-level `encode_*` helpers.
- CMD1/CMD2/CMD3 builder tests cover bindings, state transitions, command topology, and missing
  implementation errors.

Completion evidence:

- `MetalExecutionContext` is the sole production owner of the Metal runtime, resident dense
  binding, reusable buffers, and GPU recurrent-state lifecycle.
- Concrete resident projection/topK, fused CMD1/recurrent/CMD2, resident post-attention CMD2, and
  whole-slot CMD3 builders own command creation, encoding, submission, waits, and cleanup.
- Undeclared Metal KV, RoPE/RMS, dense F32/BF16, component/upload Q4, split recurrence, and F32
  LM-head cache paths have been deleted rather than retained as fallbacks.
- Source audits show no command encoder, pipeline dispatch, Obj-C lifecycle, command wait, or
  low-level Metal encode call in production `legacy.rs`; focused builder tests, all-target tests,
  release build, and the required smoke provide checkpoint verification.

### Gate 3: Scheduler-Owned Runtime

Objective: make the scheduled graph the executable layer lifecycle.

Required work:

- Create `runtime.rs` and move the model-family-agnostic generation/layer loop out of `legacy.rs`.
- Give the scheduler ownership of previous CMD3 handoff, CMD1, attention placement, CMD2, routing,
  expert reads, CMD3 submission, and deferred output handoff.
- Replace loose helper composition with one resolved per-layer execution API.
- Keep sampling and tokenizer boundaries outside the hot layer loop.

Exit criteria:

- `forward_token_input` and the layer loop no longer live in `legacy.rs`.
- Runtime does not branch on Q4/BF16/F16, Qwen3.5/Qwen/Qwen-VL, or optional Metal helpers.
- The scheduler is the only production caller that sequences CMD1/CMD2/routing/reads/CMD3.
- Direct legacy execution paths are deleted or test-only.

Completion evidence:

- `runtime.rs` owns `forward_token_input` and the sole production token/layer loop; generation,
  tokenizer, rendering, sampling, and typed model input adaptation stay outside it.
- `FlashMoeExecutionScheduler` owns the resolved graph, per-layer attention implementation table,
  sole expert store/read coordinator, and an allocation-free phase transaction from explicit
  previous-CMD3 handoff through CMD1, CMD2, routing, whole-slot reads, CMD3 submission, and
  complete/deferred output handoff.
- Runtime no longer branches on family, dtype, Qwen-VL, optional executors, or implementation
  probes. The expert-skip fixture runtime and Qwen-VL layer-loop branch were deleted. A generic
  typed token input selects resident lookup or an exact precomputed embedding and exposes declared
  per-layer additions without identifying the source family; Qwen-VL graph selection requires its
  load-time typed adapter and manifest bindings.
- The concrete required-Metal facade and CMD3 input adapter moved with production execution into
  `runtime.rs`; deferred submission and typed GPU handoff ownership live in `metal.rs`. The
  undeclared CPU CMD3 input is test-only, and source audits show no production graph builder,
  expert issue/finish, or layer execution adapter in `legacy.rs`.
- Focused scheduler/runtime tests cover phase order, previous-handoff validation, whole-slot CMD3
  completion, metrics, and output policy. FlashMoe tests, all-target tests, release build, and the
  required real-model smoke provide checkpoint verification.

### Gate 4: Weights And State Ownership

Objective: finish the CPU/GPU ownership boundaries used by the scheduler.

Required work:

- Move `DenseStore`, the resident aligned blob, tensor registry, projection execution, and dense
  caches into `weights.rs`.
- Move KV, recurrent, hidden/residual/normed, deferred buffers, next-layer transitions, and session
  state ownership into `state.rs` and `metal.rs` as appropriate.
- Make CPU readback/upload occur only at declared graph boundaries.
- Remove duplicated expert-store ownership and temporary legacy caches.

Exit criteria:

- `legacy.rs` owns no dense store/cache and no generation state/cache.
- Each runtime buffer has one owner and an explicit placement transition.
- Qwen3.5 Q4 dense bindings use the scheduler/command-builder flow; unresolved BF16/F16/F32
  bindings are named load-time errors and Gate 6 must add them through those same contracts.
- State tests cover full-attention KV, linear recurrence, deferred CMD3, and session reuse through
  the resolved graph.

### Gate 5: Qwen3.5 Q4 Correctness Closure

Objective: prove the first production graph before broadening support or benchmarking.

Required work:

- Eliminate remaining component, CPU, and alternate scheduling fallbacks from the Qwen3.5 Q4 graph.
- Compare router scores, K=4 routes/weights, expert outputs, shared output, residual/norm state, and
  logits against reference/upstream fixtures.
- Debug any `2+2=` drift through the resolved graph without restoring old data flow.
- Add at least one per-layer golden fixture and one multi-layer state-transition fixture.

Exit criteria:

- Qwen3.5 Q4 loads only with a fully resolved graph and executes without legacy runtime branches.
- Supported-path tests prove no undeclared implementation is selected.
- Focused parity tests, `cargo test --all-targets`, and the required smoke pass.
- No performance benchmark has been used to choose architecture during Gates 1-5.

Completion evidence:

- The supported Qwen3.5 Q4 load resolves one resident-Q4/fixed-Q4 graph and the runtime contains no
  component expert upload, CPU expert substitution, alternate scheduler, or implementation probe
  that can change that graph.
- Independent single-layer and multi-layer fixtures cover attention math, router scores, K=4
  routes/weights, whole-slot expert reads, active/shared expert output, residual/norm transitions,
  recurrent state, deferred CMD3 handoff, terminal hidden state, logits, and candidates.
- Local-Metal reference tests cover resident-Q4 and BF16/F16/F32 projection/CMD2 preparation plus
  the scheduler-owned CMD3 active/shared combine. The current full all-target suite passed with
  654 tests and seven device-dependent tests ignored, the release build completed, and the required
  smoke printed `4` on 2026-07-11. After expert packing moved to its target owner, the full suite
  passed with 678 tests and seven device-dependent tests ignored; web assets and the release binary
  rebuilt, and the required smoke again printed `4` on 2026-07-11. After removing the legacy
  decoded/component execution adapter and its duplicate fallback tests, 675 tests passed with seven
  ignored; the release binary rebuilt and the required smoke still printed `4` on 2026-07-11.
  After capability resolution took ownership of the manifest-validated per-layer attention
  schedule, 676 tests passed with seven ignored; web assets and the release binary rebuilt, and the
  required smoke again printed `4` on 2026-07-11. The subsequent Qwen3/Qwen3-VL typed dense-expert
  matrix raised the suite to 677 passing tests with seven ignored without changing production
  execution. After explicit Q4/BF16/F16 production storage selection was generalized across Qwen
  families and its Qwen3 text/VL namespace policy was covered, 680 tests passed with seven ignored;
  web assets and the release binary rebuilt, and the required smoke printed `4` on 2026-07-11.
  After production load was made to reject every requested-policy/metadata-layout mismatch before
  cache upgrade or graph construction, 681 tests passed with seven ignored; the release binary
  rebuilt and the required smoke again printed `4` on 2026-07-11.
- No FlashMoe benchmark or tok/s experiment was run during Gates 1-5.

### Gate 6: Unified Variant Implementations

Objective: add variants as stage implementations, not runtimes.

Required work:

- Keep typed BF16/F16 whole-expert storage and Metal implementations on the existing positioned
  read, scheduler-lease, and CMD3 handoff while exercising the explicit production storage policy
  against compatible checkpoints and adding checkpoint evidence.
- Preserve the completed real-checkpoint correctness evidence for Qwen3 text and Qwen3-VL Q4.
- Extend capability and parity fixtures until every supported family/dtype/expert-layout
  combination has direct evidence.

Exit criteria:

- Every supported variant uses the same runtime and scheduler lifecycle.
- Variant differences are confined to model metadata, typed layouts/adapters, and selected stage
  implementations.
- Unsupported combinations fail at graph resolution with precise missing-stage errors.

Current capability matrix:

| Family | Dense/expert layout | Graph/load status | Correctness evidence |
| --- | --- | --- | --- |
| Qwen3.5 MoE | Resident Q4 / fixed-Q4 slots | Supported | Linked parity, all-target, release, real smoke |
| Qwen3 MoE text | Resident Q4 / fixed-Q4 slots | Supported through the unified graph with `mlx-community/Qwen3-30B-A3B-4bit` | Linked K=8 parity, 48-layer/128-expert cache build, real load/infer, and raw-output parity with upstream MLX-LM |
| Qwen3-VL MoE | Resident Q4 / fixed-Q4 slots | Supported through the unified graph and typed vision executor with `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` | Adapter/capability parity, stale-index header import, 48-layer/128-expert cache build, real text inference, and real image request through the shared decoder |
| Qwen3/Qwen3-VL full attention | Resident BF16/F16/F32 dense / fixed-Q4 slots | Resolved unified graph | Descriptor/capability parity plus mixed CMD1, per-layout CMD2, and padded-row LM-head local-Metal parity; real checkpoint pending |
| Qwen3.5 hybrid | Resident BF16/F16/F32 dense / fixed-Q4, fixed-BF16, or fixed-F16 slots | Resolved unified graph through metadata-selected typed active and resident shared CMD3; explicit storage policy emits fixed-BF16/F16 slots from matching source dtypes | Load-resolved expert metadata, linear/shared tables, typed whole-slot offsets, scheduler leases, and Q4/BF16/F16 active plus Q4/BF16/F16/F32 shared-CMD3 local-Metal parity; real checkpoint pending |
| Qwen3/Qwen3-VL | BF16/F16 expert slots with BF16/F16/F32 dense | Explicit storage policy emits fixed-BF16/F16 slots from matching source dtypes; load requires the selected policy to equal metadata-resolved slots before capability resolution | Cross-family 12-combination graph matrix, CLI/planning selection and 3x3 policy/layout rejection coverage, storage, scheduler, and local-Metal fixtures; real checkpoint pending |
| GLM-5.2 | Canonical resident Q4 plus BF16 Colibri input/output / fixed native-MXFP4 or affine-Q4 slots from layer 3 | Shipped baseline through the unified runtime and expert scheduler; indexed MLX preserves typed E2M1/E8M0 expert storage while unindexed Colibri adapts to affine Q4, full causal MLA is bounded by `index_topk`, and DSA/MTP are unimplemented | Native MXFP4 import/storage/capability/CPU/local-Metal parity plus Colibri import, MLA/RoPE/KV, routing, expert-boundary, all-target, release, complete 75-layer native cache build, and real text/performance evidence |
| DeepSeek V4 Flash | GGUF Q8/F16/F32 resident tensors / fixed IQ2_XXS+Q2_K top-6 expert slots | Design record only; deliberately excluded from FlashMoe model selection until Gate 8 closes | Reference shape/data-flow audit only; no pb runtime, cache, vector, or real-checkpoint evidence yet |

Completion evidence:

- Qwen3.5, Qwen3 text, and Qwen3-VL execute through one `runtime.rs` generation/layer loop and one
  scheduler transaction lifecycle. Family, dense/expert dtype, attention kind, shared-expert
  presence, and input modality are resolved metadata, typed layouts, or selected stage
  implementations; none selects a second runtime.
- Qwen3 text Q4 has real deterministic output parity with upstream MLX-LM. Qwen3-VL Q4 has real
  cache/load/text evidence plus a real image request through the typed adapter and shared decoder.
- Q4/BF16/F16 expert storage has one explicit production policy, separate namespaces, exact source
  dtype checks, metadata binding before graph construction, and direct coverage of affine-Q4,
  native-MXFP4, BF16, and F16 resolved layouts against every requested policy. Dense
  Q4/BF16/F16/F32 and fixed-Q4/MXFP4/BF16/F16 combinations retain one scheduler/CMD contract with
  capability and local-Metal reference evidence.
- The final Gate 6 suite passed with 684 tests and seven device-dependent tests ignored. Web assets
  and the release binary rebuilt, the required default Qwen3.5 smoke printed `4`, Qwen3 text real
  smokes matched upstream, and Qwen3-VL text and image requests exited successfully.
- Unsupported family/layout/dtype/device/kernel/adapter combinations remain named load-time or
  graph-resolution errors; no CPU/component/layout/scheduler fallback was restored.

### Gate 7: Legacy Removal And Benchmarking

Objective: finish the migration and only then evaluate sustained decode performance.

Required work:

- [x] Remove `legacy.rs` and relocate active compatibility/parity contracts to target owners.
- [x] Delete stale test-only adapters that no longer protect an active compatibility contract.
- [x] Run sustained decode comparisons against upstream using equivalent generation settings.
- [x] Report decode throughput separately from TTFT/prefill.

Final equivalent comparison, 2026-07-12:

- Checkpoint: `mlx-community/Qwen3.5-397B-A17B-4bit`, resident MLX Q4 dense weights and fixed-Q4
  expert slots.
- Prompt: raw `Hello, what is`, token IDs `[9419, 11, 1092, 369]`; 32 generated tokens; greedy
  sampling; routing K=4; application expert cache disabled; warm OS page cache retained in both
  runs.
- pb command:
  `target/aarch64-apple-darwin/release/pb harness infer --model hf://mlx-community/Qwen3.5-397B-A17B-4bit --raw --max-tokens 32 --top-k 1 --temperature 0 "Hello, what is"`
- Upstream command:
  `/private/tmp/flash-moe-reference/metal_infer/infer --model /private/tmp/upstream-model --weights /private/tmp/upstream-model/model_weights.bin --manifest /private/tmp/upstream-model/model_weights.json --vocab /private/tmp/upstream-model/vocab.bin --prompt-tokens /private/tmp/upstream-model/prompt_hello.bin --tokens 32 --k 4 --cache-entries 0`
- Both engines generated the same continuation through `...fierce debates about whether to`.
  Upstream recorded TTFT 993 ms and 31-token decode at 11.11 tok/s. pb recorded TTFT 2082 ms and
  31-token decode at 2.512 tok/s. The gap is follow-up performance work on the unified graph; it is
  not an incomplete architecture gate or justification for an alternate runtime.
- Post-comparison verification: norm-semantics and routing fixtures pass; `cargo test --all-targets`
  passes with 661 tests and six device-dependent tests ignored; web assets and the release binary
  rebuild; the required raw `2+2=` smoke exits zero and matches upstream token 10992 (`？`).

Exit criteria:

- `legacy.rs` is absent and no production or test module depends on it.
- The bulk-refactor benchmark lock below is satisfied.
- Correctness remains green before and after performance tuning.

### Gate 8: DeepSeek V4 Flash Typed Extension

Status: **Active design gate; implementation not started.** The preceding Qwen and GLM gates remain
closed and their supported paths must not change while this family is added.

Objective: support DeepSeek V4 Flash through the existing resolved graph and scheduler without
claiming that superficially similar hidden width or MoE tensors make it a Qwen/GLM variant.

Required ownership slices, in order:

1. Add a bounded GGUF directory reader and DeepSeek source adapter that validates every semantic
   metadata field, tensor name, type, dimension, compression-ratio entry, and per-layer SwiGLU
   clamp before publishing canonical resident and fixed-expert stores. Partial or arbitrary GGUF
   files fail before graph construction.
2. Extend model-family, manifest, state, and capability resolution for the four-stream
   hyperconnection lifecycle; dense/ratio-4/ratio-128 attention schedule; raw/compressed/indexer KV;
   hash and scored routing modes; shared expert; and typed IQ2_XXS/Q2_K expert slots. Qwen and GLM
   capability matrices must gain direct regression rows proving unchanged resolution.
3. Add owner-local CPU reference fixtures for hyperconnection split/collapse, compression,
   indexer selection, sink attention, RoPE, both routing modes, mixed-quant expert math, and complete
   single-layer state transitions. Reference code is diagnostic only.
4. Add Metal stage implementations inside the existing CMD/scheduler topology. Routed expert reads
   remain top-6 parallel positioned reads into scheduler slots. No DS4 subprocess, application
   expert cache, alternate generation loop, or CPU fallback is permitted.
5. Bind tokenizer/chat/tool-call rendering and exact prompt preflight to the same production path,
   then prove prompt-boundary session reuse captures all hyperconnection and compressed-attention
   state.
6. Validate tiny deterministic fixtures, official continuation/logit vectors, a full Q2 imatrix
   cache build, real load/prefill/decode, structured tool use, long-context compression boundaries,
   all-target tests, release build, the existing Qwen/GLM smokes, and a DeepSeek smoke.

Exit criteria:

- DeepSeek resolves a complete concrete graph before inference; unsupported GGUF types, shapes,
  kernels, devices, or state layouts produce named capability errors.
- The live path uses the shared scheduler/layer lifecycle and has no hidden top-2 profile, second
  runtime, runtime tensor probing, or source-container dependency.
- Top-6 routes/weights, hyperconnection state, raw/compressed/indexer KV, layer outputs, final
  collapse, logits, and generated continuations match independent reference evidence.
- Existing Qwen3.5, Qwen3, Qwen3-VL, and GLM capability/parity suites, real smokes, and cache
  namespaces remain green and unchanged in meaning.

The first implementation commit must complete source validation plus canonical cache publication;
adding only a family enum, descriptor, or optimistic model-name match is not progress and must not
make `is_flashmoe_hf_model` return true.

## Goal Execution System

This system governed the migration. Gates 1-7 are closed. Gate 8 is the active DeepSeek extension
and must not change the shipped meaning of those earlier gates or create a second runtime or
speculative fast path.

For a future active gate, completion is a checkpoint rather than permission to weaken the target.
In the same semantic commit that records its final evidence, update the status table and follow-up
guardrail. The architecture goal recorded here completed only when Gate 7 completed.

Do not mark a gate complete based on types, tests, or call-throughs alone while its production owner
remains unchanged. If an exit criterion is discovered to be wrong, update the plan explicitly before
working around it in code.

### What Counts As Progress

A semantic commit counts toward the active gate only when it does at least one of the following:

- Moves a complete production responsibility to its target owner and routes the live path through
  that owner.
- Deletes a production fallback or changes it into a resolved implementation/unsupported error.
- Replaces runtime implementation discovery with load-time concrete graph resolution.
- Moves a complete command builder, store, cache, state transition, or execution-loop segment out of
  `legacy.rs`.
- Adds parity/capability coverage required to permit one of those ownership changes safely.

Adding a descriptor, wrapper, field, layout record, or test alone does not count as architectural
progress unless it is part of the same commit or immediate commit pair that moves the production
execution boundary using it.

### Commit Discipline

- Prefer coherent ownership slices over one-field micro-commits.
- Keep commits semantic and independently compiling.
- Name the production symbol or responsibility moved in the commit message when useful.
- Production behavior in `legacy.rs` must not increase. Temporary shims must be narrow and removed
  within the active gate.
- Do not combine unrelated cleanup or formatting churn with a migration slice.
- Treat all pre-existing worktree changes as live user work.

### Required Status Report

After each significant commit or checkpoint, report:

```text
Active gate:
Ownership moved:
Legacy production removed:
Fallback eliminated or capability resolved:
Correctness verification:
Remaining exit criteria:
Next ownership slice:
```

Do not report descriptor count, commit count, or tok/s as progress toward a gate.

### Verification Loop

- Run focused tests for the module and behavior moved in every slice.
- Add capability tests for every supported or rejected family/dtype/layout/device/stage combination
  touched.
- Run `cargo test --all-targets` after significant behavior changes.
- Build web assets before a release build: `deno task build:web`.
- After FlashMoe backend changes, build the release binary and run:
  `target/aarch64-apple-darwin/release/pb harness infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="`
- A passing smoke is necessary but does not replace K=4 routing/logit/state parity fixtures.

### Benchmark Lock

Do not run FlashMoe benchmarks or optimize tok/s until all of these are true:

- Gate 1 concrete graph resolution is complete.
- Gate 2 concrete Metal builders are complete.
- Gate 3 scheduler-owned runtime is complete.
- Gate 4 weights/state ownership is complete for Qwen3.5 Q4.
- The Qwen3.5 Q4 graph has no silent CPU/component/layout/scheduler fallback.
- Focused parity tests, all-target tests, and the required smoke are green.

Microbenchmarks, isolated kernel experiments, hidden environment toggles, Q4-only alternate
runtimes, and experiment-led reversions are prohibited before this lock opens.

Lock status: open. Gates 1-7 are complete, the compatibility boundary is deleted, all-target tests,
release build, required smoke, and equivalent sustained comparison are recorded. Future
performance work must remain on the unified graph.

## Completed Qwen/GLM Goal Guardrail

The Qwen/GLM architecture migration completed on 2026-07-12. Use this guardrail for follow-up
performance work and while Gate 8 adds DeepSeek:

```text
Preserve the completed architecture in docs/flashmoe-architecture-parity-plan.md while improving
the unified FlashMoe graph. Make focused changes, compile them, and keep parity tests, all-target
tests, release build, and real-model smokes green. Treat existing worktree changes as live user
work and do not revert them.

North star:
- One resolved scheduler-owned execution graph for supported Qwen-family MoE models.
- Q4/BF16/F16/F32 and Qwen3.5/Qwen/Qwen-VL differences are typed stage implementations.
- Missing support is an explicit load-time unsupported-capability error.
- CPU/GPU placement is resolved policy, never a fallback.
- Performance work is now permitted, but it must modify the unified graph and preserve correctness.

For follow-up work:
1. Preserve owner-local parity modules and do not recreate a broad compatibility module or
   alternate runtime.
2. Attribute the recorded pb/upstream gap to concrete unified-graph stages before changing code.
3. Report TTFT/prefill separately from steady decode throughput with equivalent model, prompt, K,
   token count, sampling, and cache conditions.
4. Keep focused tests, `cargo test --all-targets`, the release build, the required default smoke,
   and the real Qwen3/Qwen-VL checkpoints green.
5. Do not add hidden toggles, a Q4-only runtime, silent fallback, or experiment-led reversion.

Do not spend a commit on another isolated descriptor or buffer wrapper. A descriptor is useful only
inside the ownership slice that routes production execution through it.

After each significant checkpoint report the active gate, ownership moved, legacy production
removed, fallback eliminated or capability resolved, verification, remaining exit criteria, and
the next ownership slice.

If a correctness check fails after an architecture-aligned move, debug math/logits/state through
the resolved graph. Do not restore the fallback or revert the ownership change to make the symptom
disappear.

The Qwen/GLM goal is complete. DeepSeek remains an unsupported combination until Gate 8's typed
stage implementations and evidence are complete; all other unsupported combinations must likewise
remain precise capability errors.
```
