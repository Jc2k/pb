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

- `model_family`: model dimensions, tensor names, attention schedule, RoPE/MRoPE metadata, K policy,
  shared-expert shape, and typed dense/expert layouts. It must not assign a Qwen3.5 Q4 layout to
  unrelated families.
- `capabilities`: concrete model/device/storage capability resolution and precise unsupported
  errors. It returns a fully resolved graph, not a list of family-level stage labels.
- `weights`: resident aligned dense blob, manifest, typed tensor/projection handles, dtype/layout
  validation, dense caches, and pull-time conversion.
- `experts`: fixed-slot layouts, validation, PBQ4 import/build compatibility, layer readers,
  positioned reads, reusable whole-expert slots, and read metrics.
- `scheduler`: per-token/per-layer execution state, selected implementations, CMD sequencing,
  routing placement, active expert issue/finish, slot leases, and GPU -> SSD -> GPU ordering.
- `metal`: Metal device/queue/pipelines, concrete CMD1/CMD2/CMD3 builders, LM-head builders, KV and
  recurrent-state operations, Obj-C buffer ownership, command diagnostics, and wait policy.
- `state`: hidden, residual, normed, KV, recurrent, next-layer, position, and session state with
  explicit CPU-visible and GPU-resident transitions.
- `runtime`: model-family-agnostic generation loop that asks the scheduler to execute resolved
  graph stages. It does not inspect storage formats or probe optional implementations.
- `vision`: Qwen-VL preprocessing, embeddings, MRoPE, and multimodal position adaptation. It emits
  typed token/position inputs to `runtime` and never branches through the MoE layer loop.
- `legacy`: temporary facade and compatibility shims only. It is not an owner in the target
  architecture.

## Reviewed Baseline

Baseline reviewed on 2026-07-10:

- `experts.rs` substantially owns fixed-slot metadata, cache compatibility, layer readers,
  positioned reads, reusable whole-expert buffers, raw payloads, and the worker pool.
- The scheduler owns routed expert read issue/finish, read metrics, normalized routes, pending read
  sets, and whole-slot handoff into CMD3.
- Q4 fixed-slot execution rejects PBQ4/component records at the scheduler boundary.
- Model-family metadata, state descriptors, resident projection descriptors, and many CMD1/CMD2/CMD3
  input/output/layout records have been extracted.
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
- `metal.rs` now owns the reusable transient-buffer pool and the complete resident-Q4 top-k
  builder: typed projection/range validation, allocation, Q4 logits encoding, vocabulary top-k,
  submission, readback, and cleanup. Router-score and LM-head callers share that builder, and the
  resolved LM-head path returns a concrete result instead of probing `Option` availability.
- The fused Q4 CMD2 post-attention builder is also Metal-owned end to end: it validates typed
  projection/state widths, encodes output projection, residual-add/RMS norm, and router projection
  in one command, performs declared CPU top-k readback, and returns GPU-resident residual/normed
  state. `legacy.rs` now only invokes this concrete builder for that stage.
- `MetalRuntime` owns device discovery, shader-library compilation, the command queue, every
  required pipeline, partial-construction cleanup, and final release. `MetalExecutionContext`
  owns that runtime together with resident dense and recurrent state; `legacy.rs` no longer
  compiles or releases Metal pipelines, queues, devices, or buffers.
- Production scheduled CMD3 now consumes typed GPU-resident post-attention state, scheduler-owned
  whole-expert slots, resident-Q4 shared projections, routing weights, and typed output state in a
  single `MetalScheduledCmd3Builder`. Its submitted result owns the command buffer, slot leases,
  transient/borrowed buffers, deferred GPU handoff, wait/readback, and cleanup. CPU
  normed/residual upload and dense-CPU shared expert substitutions are precise unsupported-stage
  errors. The duplicate legacy CMD3 encoder, shared dense upload/cache, deferred-command type, and
  its model-specific direct-executor test have been removed; typed builder tests plus the required
  real-model smoke cover the production command.
- `LinearAttentionLayout` now lives with recurrent-state contracts in `state.rs`. The active
  Qwen3.5 fused linear-attention CMD1/CMD2 command is owned by
  `MetalFusedLinearAttentionBuilder`: resident Q4 input projections, convolution and recurrent
  state mutation, gated normalization, output projection, residual/RMS norm, router projection,
  CPU top-k readback, and GPU-resident CMD3 handoff move together. `legacy.rs` retains only the
  model-weight lookup and one builder invocation for this resolved stage.
- The previous split linear-attention runtime has been deleted: intermediate projection buffers,
  separate recurrence encoders, CPU recurrence and static-weight caches, CPU recurrent session
  state, and `Ok(None)` retries no longer sit beside the fused implementation. Qwen-VL deepstack
  and diagnostic expert skipping now fail as undeclared graph behavior at the layer boundary.
  Deferred full-attention CMD1 likewise requires resident Q4 projections and cannot select the
  removed dense BF16/F32 encoder through runtime probing.
- Single Q4 projections now use the same resident `MetalQ4ProjectionBatchBuilder` as projection
  batches. The packed/component upload retry, direct single-mmap encoder, and their tests have been
  removed. The unused application-owned F32 LM-head buffer cache and command were also deleted;
  sampling has only the resolved resident-Q4 topK builder. As a result, `legacy.rs` no longer
  creates command buffers or encoders, dispatches pipelines, or waits for Metal commands.
- `MetalExecutionContext` now owns runtime compilation, device/queue/pipelines, resident mmap
  binding, recurrent-state allocation/reset/release, and the reusable buffer pool. The former
  `MetalExecutorInner` and all recurrent Obj-C lifecycle helpers have left `legacy.rs`; its
  `MetalExecutionFacade` only validates construction policy and calls typed Metal APIs.
- `MetalResidentProjectionBatchBuilder` now owns the Q4/BF16/F16/F32 resident projection batch
  used by full-attention CMD1 and deferred state. `weights.rs` resolves one
  `ResidentMmapMatvecProjection` from manifest dtype/quantization metadata; CPU or GPU input
  binding, fused compatible-Q4 dispatch, typed per-projection dispatch otherwise, packed GPU
  output handoff, optional readback, timing, and cleanup remain one implementation. The three
  duplicate encoders and fused-batch helper have been removed from `legacy.rs`.
- `runtime.rs` now owns `forward_token_input`, the production token/layer loop, deferred CMD3 handoff,
  attention execution, CMD2/routing composition, active expert issue/finish, CMD3 submission, and
  final norm/state recording. Generation, prefill, tokenizer, and sampling entry points remain
  outside that hot loop. The dead CPU dense shared-expert runtime branch was removed; supported
  CMD3 preparation requires the resolved resident-Q4 shared projections.
- `FlashMoeExecutionScheduler` now owns the resolved graph and the sole production expert-read
  coordinator. `runtime.rs` resolves CMD1, attention placement, CMD2, and routing through that
  owner; it no longer calls graph builders or expert issue/finish APIs directly. CMD3 uses a typed
  scheduler transaction that issues routed reads, permits shared/next-norm preparation while they
  are pending, finishes whole-slot leases, builds and submits CMD3, and returns read metrics plus
  recurrent mix inputs. The duplicate engine-owned expert-store clone has been removed.
- The scheduler layer transaction now starts from an explicit initial, CPU-visible, or deferred-GPU
  previous-CMD3 handoff and consumes its phase through CMD1, CMD2, routing, pending whole-slot
  reads, CMD3 submission, and a scheduler-selected complete/defer output handoff. Per-layer full
  CPU-KV versus fused-linear-Metal attention is resolved once at load and carried by the
  transaction; `runtime.rs` no longer discovers it from model layout. The tiny-fixture expert-skip
  runtime and dormant Qwen-VL layer loop were deleted. Text lookup and exact precomputed visual
  embeddings now enter the same typed token-input boundary. Declared per-layer additions force a
  scheduler-owned complete-here CMD3 handoff, omit stale next-norm output, and are applied before
  the next shared layer transaction.
- Gate 4 has moved `DenseStore` as one owner into `weights.rs`: the mmap/blob and registry,
  resident tensor and norm caches, Q4 projection bindings, decoded/raw tile accounting, projection
  batches, CMD2/routing preparation, LM-head candidates, dtype decoding, and focused cache/read
  diagnostics moved together. Load, runtime, and sampling use that owner directly. Synthetic
  projection, CPU dense router-topK, raw-tile, and alternate projection helpers are test-only; they
  are no longer dormant production continuations hidden by `legacy.rs`'s broad dead-code allowance.
- `state.rs` now owns the CPU-visible `KvCache`, prompt/generated/recurrent records, full-attention
  KV insertion and causal lookup, capacity growth, and shallow session snapshots. The pure causal
  attention math moved to `math.rs`; `legacy.rs` retains generation/session orchestration but no KV
  storage implementation or direct access to cache internals. Focused state tests prove shared
  snapshot storage, independent growth, and explicit rejection of undeclared GPU recurrent state.
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
  deleted F32/BF16 dense Metal matvec, mmap, batch, and router-topK encoders; the resolved graph
  requires resident Q4 projections and reports non-Q4 dense or router layouts as unsupported.
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
- Normal Qwen3.5 linear-attention layers now require one fused resident-Q4 CMD1/recurrence/CMD2
  implementation. Missing projections, static offsets, recurrent state, norm weights, or compatible
  dimensions are named errors; the live graph no longer retries through an intermediate
  Metal-values path or CPU recurrence.
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
  graph with different route math. Qwen3.5 uses selected-route softmax with scale 0.9; Qwen MoE
  advances only when `norm_topk_prob=true`, while a missing value or `false` reports the routing
  stage as unsupported. Moving this ownership exposed and corrected parity fixtures that had
  accidentally injected scale 1.0 beside a Qwen3.5 graph.
- The declared CPU-KV attention stage is now a Qwen-family full-attention implementation rather
  than a Qwen3.5 label. Full-attention manifests and runtime CMD1 execution require both per-head
  Q and K RMSNorm bindings; an absent tensor is an explicit load/runtime error instead of silently
  running unnormalized attention.
- CMD3 treats a model configuration with zero shared experts as the declared no-shared-expert
  implementation. Models that declare shared experts must resolve every resident Q4 shared
  projection; a missing projection is an unsupported binding error and cannot collapse into the
  no-shared case.
- Text-only Qwen MoE Q4 now resolves the same nine-stage scheduler graph as Qwen3.5 Q4. Its model
  metadata selects full attention, configured K, routed scale 1.0, selected-route normalization,
  dimension-derived fixed-Q4 slots, and the declared no-shared CMD3 source. Production load applies
  the same manifest/range binding validation to both text families. Metal kernel requirements are
  resolved from layer and shared-expert metadata: hybrid Qwen3.5 requires its linear kernels, while
  no-shared Qwen MoE does not falsely require the shared-activation kernel.
- Expert count and active K are concrete graph values rather than generation-loop inputs. The
  resolved CLI/model routing policy is applied to model metadata before capability resolution, and
  production CMD1/CMD2/router construction consumes the resulting scheduler-owned values.
- A linked Qwen MoE Q4 fixture resolves the Qwen3 graph and follows one K=8 full-attention
  transaction through selected-route softmax at scale 1.0, eight scheduler-owned positioned reads,
  whole-slot typed Q4 SwiGLU payloads, declared no-shared CMD3 combine, and deferred hidden/next-norm
  state. Its route weights and output state are checked against independent golden values.
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
- Load now retains exactly one `FlashMoeInputAdapterExecutor` instead of probing an
  `Option<VisionEncoder>` at request time. Qwen-VL construction requires concrete weight and
  manifest artifacts, validates required vision tensor names plus adapter dimensions/DeepStack
  metadata, and contributes a `QwenVlTypedInput` implementation to the same nine-stage Q4 graph.
  The already resolved model family selects the executor; incidental vision metadata in a
  Qwen3.5 config cannot redirect its text graph into Qwen-VL.
  A text adapter bound to Qwen-VL, a VL adapter bound to text, absent artifacts, invalid metadata,
  or missing required vision tensors is a token-input-stage error before inference.
- Focused parity/reference tests cover expert layout and math, routing contracts, attention and
  recurrence primitives, state descriptors, and Metal buffer-plan contracts.
- Gate 5 now has a linked Qwen3.5 Q4 per-layer golden derived independently from the upstream
  equations. It follows CPU-KV scaled-dot-product attention, deferred CMD1 state, Metal-placed CMD2
  descriptors, fixed router scores, K=4 IDs/weights, four real scheduler-owned positioned reads of
  whole fixed-Q4 slots, typed gate/up/down offsets, SwiGLU expert output, shared output,
  residual/hidden state, declared CMD3 output, and next-layer RMSNorm. The stale ignored test that
  mislabeled a tiny Qwen3 model as the production Qwen3.5 URI was deleted rather than preserved as
  an alternate runtime fixture.
- A second deterministic fixture now carries the resolved graph across a fused linear-attention
  layer and a declared CPU-KV full-attention layer. It proves deferred hidden/next-normed handoff,
  recurrent mixing, eight scheduler-owned K=4 positioned reads, terminal hidden state, logits, and
  top candidates without entering another runtime.
- Local-Metal reference tests exercise both the resident-Q4 fused projection batch and one mixed
  BF16/F16/F32 batch, fused CMD2 preparation, and scheduler-issued CMD3 active-expert plus
  shared-expert combine against independent CPU math. Route IDs are exact and floating-point route
  scores use an explicit numerical tolerance rather than bitwise equality across CPU and Metal
  implementations.

The architecture is not yet at the target:

- The legacy engine shell still stores the weights-owned `DenseStore`, state-owned session-cache
  owner, and runtime layout metadata even though their implementations live in focused modules.
- General caches and explicitly diagnostic/test helpers still use `Option` for data availability,
  but no supported graph-stage implementation or CPU/GPU placement is selected from those values.
- Text-only Qwen MoE Q4 has a resolved unified graph and linked parity fixture but still needs
  real-checkpoint smoke evidence. Qwen-VL preprocessing now emits exact typed inputs consumed by
  the shared runtime, including scheduler-compatible DeepStack handoff policy. Its Q4 capability
  and load path resolve only from a concrete vision executor and manifest bindings, but a real
  Qwen-VL checkpoint smoke is still required before claiming production correctness.
- BF16/F16/F32 full-attention CMD1 now uses the same typed resident projection handle, Metal batch
  builder, CPU/GPU input bindings, and state handoff as Q4. Full-attention non-Q4 graphs stop at the
  named CMD2 stage; Qwen3.5 hybrid non-Q4 graphs stop at fused linear-attention CMD1. Non-Q4 CMD2,
  shared expert, LM-head, and expert-slot implementations remain incomplete.

At this checkpoint, Gates 1 through 5 are complete. Qwen3.5 Q4 has a resolved production graph and
correctness closure. Typed implementations for additional variants and final legacy removal remain;
the amount of work is governed by those two exit gates rather than a percentage estimate.

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
| 6. Unified Variant Implementations | Active | Other variants use the same graph/runtime. |
| 7. Legacy Removal And Benchmarking | Pending | Migration is complete and benchmarking may begin. |

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
- Concrete resident-Q4 projection/topK, fused CMD1/recurrent/CMD2, Q4 post-attention CMD2, and
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
  per-layer additions without identifying the source family; Qwen-VL graph selection remains a
  named unresolved capability until its load-time bindings are proven.
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
- Local-Metal reference tests cover resident-Q4 CMD1 projection batches, CMD2 preparation, and the
  scheduler-owned CMD3 active/shared combine. The full all-target suite passed with 635 tests and
  five device-dependent tests ignored, the release build completed, and the required smoke printed
  `4` on 2026-07-11.
- No FlashMoe benchmark or tok/s experiment was run during Gates 1-5.

### Gate 6: Unified Variant Implementations

Objective: add variants as stage implementations, not runtimes.

Required work:

- Add typed BF16/F16 dense and expert implementations where model support requires them.
- Add Qwen MoE attention/layout/shared-expert implementations to the same graph.
- Create `vision.rs` and make Qwen-VL produce typed runtime inputs before the MoE graph starts.
- Add capability and parity fixtures for every newly supported combination.

Exit criteria:

- Every supported variant uses the same runtime and scheduler lifecycle.
- Variant differences are confined to model metadata, typed layouts/adapters, and selected stage
  implementations.
- Unsupported combinations fail at graph resolution with precise missing-stage errors.

Current capability matrix:

| Family | Dense/expert layout | Graph/load status | Correctness evidence |
| --- | --- | --- | --- |
| Qwen3.5 MoE | Resident Q4 / fixed-Q4 slots | Supported | Linked parity, all-target, release, real smoke |
| Qwen3 MoE text | Resident Q4 / fixed-Q4 slots | Resolved unified graph | Linked K=8 parity; real checkpoint pending |
| Qwen3-VL MoE | Resident Q4 / fixed-Q4 slots | Resolved unified graph with required typed vision executor | Adapter/capability parity; real checkpoint pending |
| Qwen3/Qwen3-VL full attention | Resident BF16/F16/F32 dense / fixed-Q4 slots | CMD1 implemented; unsupported at CMD2 | Descriptor and local-Metal mixed-batch parity |
| Qwen3.5 hybrid | Resident BF16/F16/F32 dense / fixed-Q4 slots | Unsupported at fused linear-attention CMD1 | Precise capability failure |
| Any supported family | BF16/F16 expert slots | Unsupported at active expert/CMD3 stage | Import/reference tests only |

### Gate 7: Legacy Removal And Benchmarking

Objective: finish the migration and only then evaluate sustained decode performance.

Required work:

- Reduce `legacy.rs` to a facade/compatibility boundary or remove it.
- Delete stale test-only adapters that no longer protect an active compatibility contract.
- Run sustained decode comparisons against upstream using equivalent generation settings.
- Report decode throughput separately from TTFT/prefill.

Exit criteria:

- No production inference owner remains in `legacy.rs`.
- The bulk-refactor benchmark lock below is satisfied.
- Correctness remains green before and after performance tuning.

## Goal Execution System

Use one active completion gate at a time. Work may prepare the next gate only when required to
complete the current one; it must not create a second runtime or a speculative fast path.

Completing an active gate is a checkpoint, not completion of the architecture goal. In the same
semantic commit that records the final evidence for a gate, update the status table, make the next
gate active, update the Current Active Goal Prompt, and continue. The overall architecture goal is
complete only when Gate 7 is complete.

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
  `target/aarch64-apple-darwin/release/pb flashmoe infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="`
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

## Current Active Goal Prompt

Use this prompt for implementation work:

```text
Implement docs/flashmoe-architecture-parity-plan.md. Do not stop at another plan.

Active gate: Gate 6, Unified Variant Implementations.

Move production ownership quickly, make it compile, and make focused tests, all-target tests, and
the required smoke work. Make regular semantic commits. Treat existing worktree changes as live
user work and do not revert them.

Start by inspecting git status, the active gate's exit criteria, and the production call sites that
still own the behavior. Work in coherent ownership slices, not one-field or one-wrapper commits.

North star:
- One resolved scheduler-owned execution graph for supported Qwen-family MoE models.
- Q4/BF16/F16/F32 and Qwen3.5/Qwen/Qwen-VL differences are typed stage implementations.
- Missing support is an explicit load-time unsupported-capability error.
- CPU/GPU placement is resolved policy, never a fallback.
- No experiments, microbenchmarks, tok/s work, hidden toggles, or Q4-only alternate runtime before
  the benchmark lock opens.

For Gate 6:
1. Make expert layout a concrete storage-resolved input to graph resolution. Generalize fixed-Q4
   slot validation around typed dimensions and offsets so Qwen MoE can use its own layout without
   inheriting Qwen3.5 constants.
2. Resolve the text-only Qwen MoE Q4 graph first. Add its full-attention, Q/K norm, routing-scale,
   shared-expert, and K policy as typed stage metadata while keeping the scheduler/runtime/CMD
   lifecycle unchanged. Missing tensors or kernels must name the exact unresolved stage.
3. Continue resident BF16/F16/F32 from the completed shared full-attention CMD1 binding into CMD2,
   fused linear attention, shared experts, and LM-head. Add BF16/F16 whole-expert slots only through
   the same scheduler leases and CMD3 handoff. Do not revive removed dense CPU/component paths as
   production continuations.
4. Run the resolved Qwen-VL Q4 graph against a real checkpoint and debug any manifest/math mismatch
   through its typed vision executor and shared scheduler path. Do not add request-time probing or
   an alternate decoder loop to make the smoke pass.
5. Add a capability matrix and focused parity fixture for each supported family/dtype/layout. Keep
   every not-yet-implemented combination as a precise load-time unsupported-stage error while the
   matrix is filled in.

Do not spend a commit on another isolated descriptor or buffer wrapper. A descriptor is useful only
inside the ownership slice that routes production execution through it.

After each significant checkpoint report the active gate, ownership moved, legacy production
removed, fallback eliminated or capability resolved, verification, remaining exit criteria, and
the next ownership slice.

If a correctness check fails after an architecture-aligned move, debug math/logits/state through
the resolved graph. Do not restore the fallback or revert the ownership change to make the symptom
disappear.

Gate 6 is a checkpoint, not the end of the goal. When every Gate 6 exit criterion is proven, update
the gate status and this prompt to Gate 7 in the same semantic commit, then continue implementing
the plan. Do not mark the overall goal complete until Gate 7 is complete.
```
