# FlashMoe Architecture Parity Plan

This document is the source of truth for bringing pb's MoE backend into architectural parity with
`danveloper/flash-moe` while retaining pb's production shell: Rust pull/conversion, production
tokenizers, structured requests, Qwen-family model support, Qwen-VL inputs, and the shared inference
facade with llama.cpp.

The target is one scheduled execution graph for supported Qwen-family MoE models. FlashMoe owns
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
- Direct component-buffer execution, upload-heavy reconstruction, and fused-to-unfused substitution
  are unsupported unless each is an explicitly resolved graph-stage implementation.
- Reference CPU implementations belong in tests, diagnostics, and explicit development tools.
- No new production behavior belongs in `legacy.rs` except a narrow temporary call-through needed
  in the same ownership slice to move its owner.
- If an architecture-aligned change makes `2+2=` or a parity fixture wrong, debug math, logits, and
  state through the unified path. Do not restore the old architecture to hide the error.

## Upstream Parity Contract

The behavioral source of truth is `danveloper/flash-moe`, especially `metal_infer/infer.m`,
`metal_infer/shaders.metal`, `repack_experts.py`, and the Q4 optimization notes.

The Qwen3.5 Q4 implementation must preserve these upstream-shaped properties:

- Non-expert weights live in one 64-byte-aligned binary blob, mapped once.
- Experts are fixed records in per-layer files. The Qwen3.5-A17B Q4 record is 7,077,888 bytes with
  fixed gate/up/down packed-weight, scale, and bias offsets.
- Active experts are issued in parallel with positioned reads into scheduler-owned reusable
  whole-expert slots.
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
  so its explicit releases deallocate completed command objects immediately. Each deferred CMD3
  submission is also created inside a nested autorelease pool, which drains Metal's committed-command
  auxiliaries at the layer boundary instead of leaving their expert-buffer references alive until the
  outer token autorelease pool drains. Scheduler `pread` expert payloads are copied into bounded,
  reusable Metal staging buffers for only the active expert slots. This avoids the cumulative Metal
  accounting observed for no-copy mappings during long prefills; nested command draining prevents the
  copied staging path from retaining tens of gigabytes of completed-command resources before jetsam.
  If a new allocation still fails, the pool releases all currently idle buffers and retries once,
  reporting the requested and released byte counts plus Metal's current and recommended working-set
  sizes on failure.
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

Completion evidence:

- Qwen3.5, Qwen3 text, and Qwen3-VL execute through one `runtime.rs` generation/layer loop and one
  scheduler transaction lifecycle. Family, dense/expert dtype, attention kind, shared-expert
  presence, and input modality are resolved metadata, typed layouts, or selected stage
  implementations; none selects a second runtime.
- Qwen3 text Q4 has real deterministic output parity with upstream MLX-LM. Qwen3-VL Q4 has real
  cache/load/text evidence plus a real image request through the typed adapter and shared decoder.
- Q4/BF16/F16 expert storage has one explicit production policy, separate namespaces, exact source
  dtype checks, metadata binding before graph construction, and direct 3x3 match/rejection
  coverage. Dense Q4/BF16/F16/F32 and fixed-Q4/BF16/F16 combinations retain one scheduler/CMD
  contract with capability and local-Metal reference evidence.
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

## Goal Execution System

This system governed the migration. All completion gates are now closed; future architecture
extensions should add an explicit new gate before changing ownership boundaries and must not
create a second runtime or speculative fast path.

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

## Completed Goal Guardrail

The architecture migration completed on 2026-07-12. Use this guardrail for follow-up performance
work:

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

The architecture goal is complete. Any new unsupported combination must remain a precise
capability error until its typed stage implementation is added to this graph.
```
