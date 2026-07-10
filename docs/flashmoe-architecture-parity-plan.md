# FlashMoe Architecture Parity Plan

This is the implementation plan for bringing pb's MoE backend into architectural parity with
`danveloper/flash-moe` while keeping pb's production shell: Rust pull/conversion, production
tokenizers, structured request APIs, Qwen model-family flexibility, Qwen-VL image support, and the
shared inference facade with llama.cpp.

The target is one MoE execution architecture for Qwen-family MoE models. FlashMoe owns buffer
lifetime, expert scheduling, command-buffer structure, read scheduling, and CPU/GPU handoff policy
for every supported Qwen MoE variant. Model-family traits describe dimensions, tensor naming,
quantized layouts, vision adapters, and dtype variants without forking that execution flow.

"One path" means one scheduled execution graph. Q4, BF16, F16, Qwen3.5, Qwen MoE, and Qwen-VL may
provide different typed implementations for graph stages, but they must not create separate runtimes,
silent fallbacks, or alternate data-flow paths. If a required implementation is missing for a model
variant, loading or planning must fail with an explicit unsupported-capability error so the gap can be
implemented deliberately.

This document replaces the old experiment ledger. Work should be judged by movement toward the
target architecture and correctness parity, not by isolated tok/s experiments.

## Principles

- FlashMoe is the backend for supported Qwen-family MoE models. llama.cpp is reserved for
  unsupported or intentionally delegated models.
- The MoE backend owns expert buffers, resident dense buffers, scheduler state, KV and recurrent
  state, command-buffer sequencing, and read scheduling.
- Model variants supply metadata through typed layouts and traits. They do not get a separate slow
  execution pipeline.
- Rust generics, typed layout structs, and closed enums should express dtype/layout variation without
  changing the execution flow. Avoid `dyn` traits in hot loops.
- CPU and GPU placement is part of the execution policy, not a fallback mechanism. CPU attention math
  or CPU softmax/topK is valid when it is the declared upstream-parity placement for that graph
  stage; CPU execution because a Metal implementation is missing is an unsupported capability.
- Q4-specific kernels, offsets, and packing are acceptable only when they fit the shared scheduler,
  state, and command topology and have a clear route for BF16/F16 or future quantized variants.
- Missing variant support must be an error, not a silent fallback. Reference CPU code belongs in
  tests, diagnostics, and explicit unsupported-development tools, not in production execution.
- Qwen3.5-A17B parity is the first concrete target, but the shape must also carry Qwen MoE and
  Qwen-VL through the same scheduler and buffer lifecycle.
- Correctness must be proven with parity tests before full-model throughput claims. If a parity
  alignment causes `2+2=` to produce the wrong answer, debug the math/state drift instead of
  treating the alignment as a failed speed experiment.

## Upstream Behaviors To Mirror

Source of truth: `danveloper/flash-moe`, especially `metal_infer/infer.m`,
`metal_infer/shaders.metal`, `repack_experts.py`, and the Q4 optimization notes.

- Throughput comparison uses decode/generation throughput, not prefill-inclusive wall time.
- Non-expert weights are stored in one 64-byte-aligned binary blob and mapped once.
- Experts are packed as fixed per-layer files. For Qwen3.5-A17B Q4 each expert is exactly
  7,077,888 bytes with fixed gate/up/down packed-weight, scale, and bias offsets.
- Active experts are read with parallel positioned reads into scheduler-owned reusable whole-expert
  slots.
- The OS page cache is the cache. Avoid application LRU caches, LZ4, mmap expert reads, broad
  prefetch, speculative expert reads, and dispatch_io.
- Each layer follows one command-buffer topology:
  CMD3(previous layer) -> CMD1 attention projections -> CPU attention math where upstream uses CPU
  -> CMD2 output projection, residual, post-attention norm, routing projection, shared gate/up
  -> CPU softmax/topK and expert reads -> CMD3 active experts, shared down, combine, residual, next
  input norm.
- CMD3 is deferred and produces the next layer's GPU-resident input when possible.
- Expert Q4 matvec uses the upstream FMA dequant form. Gate/up/SwiGLU and down are encoded in the
  upstream-shaped expert command, not as independent upload-heavy component paths.

## Execution Graph And Capabilities

Every supported model resolves to a single scheduled graph before inference starts. The graph
contains the same conceptual stages for all variants:

1. Token/position input preparation, including any vision adapter output.
2. CMD3 completion from the previous layer when deferred work exists.
3. CMD1 attention projections.
4. Declared attention math placement.
5. CMD2 output projection, residual update, post-attention norm, router projection, and shared
   gate/up work.
6. Declared routing placement, including softmax/topK.
7. Scheduler-owned active expert reads into whole-expert slots.
8. CMD3 active expert gate/up/down, shared down, combine, residual update, and next-layer input norm.
9. LM-head/topK or sampling output.

Variant-specific behavior is expressed as typed capabilities attached to those stages. Examples:

- `ExpertLayout::FixedQ4` can provide fixed packed-weight/scale/bias offsets and a Q4 FMA Metal
  kernel for CMD3.
- `DenseLayout::ResidentQ4` and `DenseLayout::ResidentBf16` can provide different projection
  descriptors while sharing the same `weights` ownership and scheduler binding flow.
- `RoutingPlacement::CpuSoftmaxTopK` and a future `RoutingPlacement::GpuSoftmaxTopK` can be two
  implementations of the same scheduler stage, not separate execution paths.
- Qwen-VL can provide a `vision` adapter and MRoPE/position plan, then hand the same runtime graph a
  typed token sequence.

Before loading, FlashMoe should validate that every graph stage has an implementation for the model
family, dtype/layout, quantization, device placement, and execution policy. Missing support should
produce a precise error, for example:

```text
FlashMoe unsupported Qwen3-VL Q4 path: CMD3 shared expert down projection is not implemented for
FixedQ4 experts.
```

The validator should reject silent fallbacks such as:

- Falling from fixed whole-expert slots to component-buffer expert execution.
- Falling from a missing Metal kernel to production CPU math.
- Falling from GPU-resident state to CPU vector clones/readbacks except at declared graph boundaries.
- Falling from the unified scheduler to legacy direct calls.
- Falling from FlashMoe to llama.cpp for a model that claims FlashMoe support.

## Current State

- `src/inference/flashmoe/mod.rs` is only a facade. Most production behavior is still exported from
  `legacy.rs`, including planning, cache build, runtime, Metal command construction, dense-store
  execution, state, tests, and Qwen-VL preprocessing.
- `model_family.rs` now contains useful parity metadata: Qwen family detection, layer schedule,
  configured K versus scheduled K, shared expert dimensions, Qwen-VL metadata, Qwen3.5 Q4 expert
  offsets, and an `UPSTREAM_PARITY` execution policy.
- `math.rs` now owns Q4 quantization and Q4 dequant/matvec math used by both dense cache packing
  and expert pack import/build compatibility. This removes one more math primitive from the legacy
  runtime while preserving the same byte layout and affine-MSE group metadata.
- `experts.rs` now owns fixed-slot metadata, the fixed-slot packed-record test fixture, the runtime
  `ExpertSlotStore`, layer reader opening, positioned reads, reusable whole-expert buffers, raw
  expert payload responses, and the expert read worker pool. Production store opening resolves
  through the model layout; default Qwen3.5 raw multi-read helpers are kept test-only. Packed expert
  tensor records, PBQ4 pack parsing, PBQ4 pack builder input records and writers, PBQ4 wire-size
  accounting, scale/bias payload decoding, shared scale/bias dtype sizing, and PBQ4-to-fixed-Q4
  compatibility conversion now live with expert storage instead of the legacy runtime. Expert layer
  rewrite/reuse policy, positioned slot writes, fixed-slot reuse validation, and atomic expert
  metadata commits are also owned by `experts`; legacy conversion code now supplies build callbacks
  while the storage module decides reuse and layout. PBQ4-to-fixed-Q4 layer upgrades, stale temp
  cleanup, per-expert completeness checks, and missing-pack detection now also live under expert
  storage. Aggregate expert tensor classification, split-layout sizing, slice planning, shape
  validation, native-Q4 consistency checks, direct gate/up/down expert group validation, expert
  source byte-range validation, native-Q4 slice byte-range planning, expected pack metadata
  construction, and Q4 record byte/group accounting now live in `experts` behind a typed
  manifest-tensor adapter, leaving legacy cache build code to bridge safetensor byte decoding for
  now. PBQ4 remains import/build compatibility; execution reads are moving toward fixed
  whole-expert slots.
- `scheduler.rs` now owns graph-stage resolution, CMD1 resolved input-state handoff, CMD2/CMD3
  descriptors, typed CMD2 attention/residual input descriptors, CMD2 command construction from
  declared CPU/GPU input placement, CMD2 typed input-state validation and resolved input-state
  handoff, CMD2 post-attention prep output resolution tied to that input state, CMD3 input validation, retained
  input-state handoff, and scheduled deferred output resolution carried to the Metal helper
  boundary, routing topK placement validation with retained routing-output handoff into route
  selection, declared router-score batches for CPU routing output, scheduler-built preselected
  routing commands from CMD2 post-attention prep output, declared CMD1 input-state validation, CMD2
  routing-output validation, scheduled CMD3 output readback validation with declared token-state
  application, full-attention KV placement and execution-shape validation for the declared attention
  math implementation, active expert read issue and finish metrics, route normalization, pending
  read sets, shared-expert source/shape validation, scheduler-built CMD3 commands from typed
  input/shared/next-norm descriptors, and the scheduled whole-slot handoff.
  Shared-expert scheduling now carries width, shared-expert count, per-expert intermediate width,
  and total intermediate width as one validated graph shape, so CMD2/CMD3 shared work no longer has
  to infer those dimensions from legacy phase structs alone.
  `scheduler.rs` also now owns the scheduled expert read coordinator that combines `ExpertSlotStore`,
  positioned-read worker submission, active read IDs, warm-read metrics, route normalization, and
  completion into scheduler-owned whole-expert slots. The historical runtime keeps a coordinator
  handle only as a bridge while command encoding is extracted.
  Shared expert source and shape are now carried by the runtime-built CMD3 scheduler descriptor, so
  a Q4 or dense shared expert implementation cannot be accepted without its declared graph shape.
  Scheduler-owned fixed-Q4 slots now resolve typed CMD3 expert payloads directly, runtime CMD3
  submission retains those scheduled slots instead of adapting them into `ExpertWeights`, and the
  old direct active-expert read, `ExpertWeights` scheduled-CMD3 adapter, and `ExpertWeights` Metal
  submission path are now test-only instead of production fallbacks. The scheduler now builds resolved
  CMD1/attention/CMD2/routing/CMD3 command objects before the legacy Metal encoder or runtime
  helpers are called. CMD3 next-layer norm weights are declared as typed scheduled inputs with
  source, tensor name, and width before the Metal encoder receives the CPU slice; when the graph
  expects the next-layer norm transition, a missing norm tensor is now an unsupported scheduled-CMD3
  error rather than a silent no-next-norm path. Fused Metal
  post-attention prep now resolves through the scheduler-owned CMD2 output before recording the
  approved preselected routing command for the runtime, so CMD2-to-routing handoff is not just a
  loose active-expert vector.
  Scheduled CMD3 Metal submission now consumes only a scheduler-resolved CMD3 command, returns a
  concrete deferred phase from both the wrapper and inner scheduled helpers, and treats "no
  submitted phase" as an unsupported implementation gap instead of returning to a fallback-shaped
  caller path.
- `metal.rs` now owns Metal command labels, status decoding, timeout/failure formatting, and
  release-policy detection for command-buffer failures, plus command-buffer wait policy and
  status-to-result resolution. It also owns the Metal shader source, declared kernel-name surface,
  pipeline-name plan, typed pipeline handle set, release ordering, and kernel-surface coverage.
  Metal post-attention prep now materializes as a Metal-owned typed payload carrying scheduler CMD3
  input state, declared prep state, buffers, active routes, and the optional resolved routing
  command. That routing command is attached through a Metal-owned validation method rather than a
  public mutable field, so CMD2 fused-prep routes must match the scheduler command before expert
  reads can be issued. Metal attention placement policy, batch-projection inputs, projection batches,
  attention-value outputs, and linear-attention static offsets now use Metal-owned descriptors for
  CMD1/CMD2 handoff. Metal KV cache layer layout/range validation and linear-attention GPU state
  cache descriptors now live with the Metal module, while allocation is still bridged through
  legacy Obj-C helpers. CMD3 submission now starts from a Metal-owned phase plan that validates
  active expert count, routing-weight count, payload count, width, output-state declaration, and
  next-norm presence before the encoder bridge allocates command-local buffers. CMD3 deferred
  hidden/next-normed output buffers now carry a Metal-owned descriptor that validates GPU-resident
  output state before the legacy wait/readback bridge can consume them. CMD3 expert-phase temporary
  buffers and command-local fixed-Q4 source-buffer reuse now carry Metal-owned lifecycle/cache markers, and reusable transient Metal buffer entries
  are typed in the Metal module. Resident dense and shared-expert Metal buffers now have
  Metal-owned descriptors, including mmap retention for resident dense weights; LM-head Metal
  buffer cache budget, shape matching, eviction, and release-all policy also live in `metal.rs`.
  Legacy still owns the concrete encoders and Obj-C buffer lifecycle, but command submission
  diagnostics, wait policy, attention placement policy, kernel definitions, post-attention prep
  payloads, projection/attention buffer descriptors, resident dense/shared buffer descriptors,
  KV/linear-state layout validation, reusable buffer records, LM-head cache policy, Q4
  source-buffer reuse, dispatch geometry, and phase buffer lifecycle markers now have a module
  boundary for the CMD1/CMD2/CMD3 builders to move behind.
- Existing code has moved fixed-slot and Q4 handling toward whole-expert payload ownership, but
  runtime behavior still lives in the historical monolith and still has fallbacks and component
  pathways that can bypass the target data flow.
- Production behavior can still silently use older CPU/component paths when a Q4 or Metal-shaped
  implementation is missing. Those paths need to become either typed graph-stage implementations or
  explicit unsupported-capability errors.
- Dense weights are closer to the upstream resident-blob model than experts are. `weights` now owns
  typed resident projection descriptors for dense, Q4, shared expert, and router score bindings plus
  the router score batch data model. The generic resident Q4 mmap projection cache now keys and
  builds descriptors through `weights` with a registry callback. Shared expert dense/Q4 descriptor
  groups now validate and expose their graph shape before the scheduler accepts them. Canonical
  router, shared expert, full-attention, linear-attention, and CMD3 next-layer norm tensor naming,
  Qwen3Next norm-offset policy, and prepared scheduled next-norm descriptors also live in
  `weights`; full-attention and linear-attention CMD1 projection request groups are weight-owned
  descriptors used by the legacy bridge.
  Dense shared-expert weight assembly and Q4 shared-expert projection assembly now use the same
  weight-owned shape validation with runtime lookup callbacks, and shared-expert phase cache/reuse
  policy is owned by `weights` instead of the engine. CMD2 Q4 post-attention prep projection
  assembly now also resolves as a typed weight-owned out-projection/router bundle and resident
  execution plan before the Metal helper runs, including dense-store bounds validation and declared
  active-route count. Invalid resident CMD2 bindings are now explicit descriptor errors rather than
  duplicated helper-local fallback checks. Router score projection descriptor lookup also lives in `weights` behind a
  registry callback, and Metal router topK now consumes that
  same declared dense/Q4 binding instead of re-resolving router layouts in the legacy helper. The
  generation loop supplies lookup closures instead of owning those weight-policy branches. Command
  construction and much of runtime score execution still flows through `legacy.rs` shims.
- `state.rs` owns CPU-visible hidden/residual/normed/next-normed buffers and now also describes
  GPU-resident hidden, residual, normed, and next-layer normed buffers with typed roles and lengths.
  CMD1 now declares its actual input as either CPU-visible normed state or GPU-resident
  next-layer-normed state before attention projection. CMD2 now declares submitted attention-value
  and residual inputs with CPU/GPU placement and lengths before the post-attention phase is
  accepted. CMD2 post-attention prep also declares its CPU-visible routing output as either router
  scores or preselected topK, with layer, expert count, active count, source, and placement. CMD3
  now declares its actual CPU-visible normed/residual input or GPU-resident post-attention-prep input
  before scheduled submission, and declares GPU-resident hidden plus optional next-layer normed
  output before Metal readback or next-layer reuse.
  Full-attention KV records now expose CPU-visible or GPU-resident state descriptors with layer,
  position, width, and placement, and the runtime resolves those descriptors through the scheduler
  before using CPU attention or writing the Metal KV cache.
  Recurrent per-layer records now expose CPU-visible recurrent state descriptors before they are
  recorded into the session/cache state. Linear-attention cache state now exposes CPU-visible and
  GPU-resident descriptors for conv state, SSM state, conv output, and value output lengths, and
  `state` owns the CPU-visible linear-attention recurrent buffers for those declared shapes before
  cache mutation or Metal allocation. The Metal object handles still live in `legacy.rs`, but
  deferred GPU inputs, post-attention prep, CMD3 inputs, deferred expert outputs, declared CMD3
  token-state application modes, full-attention KV updates, recurrent layer records, and
  linear-attention cache updates now carry state descriptors instead of raw anonymous lengths.
- Timing, benchmark, cache cleanup, pull-time conversion, and smoke tooling exist. They are useful
  verification tools, not the work queue.

## Architectural Gaps

- `legacy.rs` remains the center of gravity. New architecture work should extract ownership
  boundaries from it instead of adding more production paths inside it.
- The runtime has both fixed-slot Q4 concepts and older PBQ4/component concepts. PBQ4 should become
  import/build compatibility only; its packed tensor data model and fixed-slot conversion now live in
  `experts`, while execution should consume typed whole-expert slots.
- Fallbacks are not yet consistently modeled as errors. Any branch that silently changes dtype,
  buffer ownership, scheduler, or CPU/GPU placement hides missing implementation work and must be
  made explicit.
- Expert reads now follow the upstream positioned-read policy under `experts`, and the runtime holds
  a scheduler-owned read coordinator instead of a legacy store/pool wrapper. The scheduler owns
  issue/finish metrics, worker submission, route normalization, and scheduled slot completion.
  Scheduler-owned slots now expose fixed-Q4 CMD3 payloads with typed offsets and reject
  PBQ4/component payloads for execution. Active expert reads are now issued from the validated
  `ScheduledRoutingCommand` produced by CPU router scores or CMD2 preselected routes. The scheduler
  turns that routing command into a typed
  `ScheduledExpertReadSet` containing normalized `ScheduledExpertRoutes` plus issued read IDs/warm
  flags before positioned file reads are submitted, and the pending read set carries those
  scheduler-normalized routes instead of a loose layer plus route vector. The old direct
  `read_active_experts`/`ExpertWeights` submission route remains only as test/diagnostic scaffolding;
  production CMD3 submission retains scheduler-owned whole-expert slots. CMD1/CMD2/CMD3 submission is
  now built through the scheduled graph, and CMD3 now enters the Metal bridge as a scheduler-owned
  command object instead of a legacy-built descriptor/submission pair. The remaining gap is to move
  the Metal command encoding and legacy CPU diagnostic helpers out of `legacy.rs` behind the
  explicit CMD builder APIs.
- Command-buffer topology is still implicit inside the runtime and Metal helpers. CMD1/CMD2/CMD3
  descriptor and submission construction now flows through the scheduled graph, but the concrete
  command encoders should become explicit command builders used by every supported variant.
- GPU residency is partial. CMD1 CPU normed versus deferred GPU next-layer-normed inputs,
  post-attention residual/normed prep, scheduler-visible routing outputs, CMD3 CPU/GPU input state,
  CMD3 deferred hidden/next-normed outputs, and full-attention KV records now carry typed state
  descriptors. CMD3 next-norm weight resolution now comes from a prepared `weights` descriptor
  before the scheduler builds the stage.
  Full-attention CPU versus Metal KV execution is resolved as a declared attention graph-stage
  implementation, so Metal KV writes are not an implicit fallback path. Per-layer recurrent records
  now declare CPU-visible placement before cache/session recording. Linear-attention cache state now
  declares CPU-visible or GPU-resident placement before cache mutation or Metal allocation, but the
  actual Metal cache object ownership and command encoding still live in `legacy.rs`. Many
  next-layer buffer transitions still cross CPU/GPU boundaries in places that should become
  explicit state transitions.
- Routing topK placement is now represented as a scheduler graph stage and resolves score-based or
  fused-prep preselected routes into a scheduler-owned routing command. CPU router scores and fused
  CMD2 prep topK now submit declared routing-output state before route selection is accepted, and
  Metal post-attention prep must resolve as a scheduler-owned CMD2 output before its routes can feed
  topK validation. That resolved CMD2 output now builds the preselected routing command directly
  instead of delegating command construction to a legacy Metal prep helper. The live path now asks
  the scheduled graph to build CMD2 submissions, rejecting stale descriptors that do not match the
  current graph capability before post-attention helpers execute, and the live loop now asks the
  scheduled graph to build the CMD2 command directly from typed attention/residual input
  descriptors. CPU router score production now asks the scheduled graph to build the projection
  command carrying declared routing state, hidden width, and the optional resident projection
  descriptor before the dense store executes it, and the
  scheduled command now finalizes raw scores into the declared `RouterScoreBatch`. Active expert
  issue now consumes the resulting
  `ScheduledRoutingCommand` directly and resolves it into scheduler-owned `ScheduledExpertRoutes`
  before reads are issued. The CMD2 command now owns the handoff from declared Metal
  post-attention prep state plus preselected routes into a scheduled routing command, so the runtime
  no longer stitches those graph stages together manually. The Q4 post-attention prep helper now
  receives weight-owned CMD2 projection bindings for output projection and router projection instead
  of resolving those bindings inside the legacy helper. The scheduled router score command now owns
  raw score finalization into a declared batch, routing-output validation, and score-based topK
  selection before the runtime receives a `ScheduledRoutingCommand`. Router projection execution,
  Accelerate score production, and Metal router topK now require a weight-built declared resident
  dense/Q4 descriptor before the legacy bridge can run them, so missing router storage is an
  unsupported implementation error rather than a synthetic fallback. The
  remaining gap is score production ownership: projection execution still flows through legacy
  dense/runtime helpers instead of a typed CMD2 builder boundary, even though the inputs and
  capability checks are now descriptor-backed.
- Shared experts now enter CMD3 through a scheduler descriptor that carries source and graph shape,
  and the live CMD3 builder derives next-norm source from typed `ScheduledNextNormWeights` instead
  of a separate legacy-owned branch. The live path now asks the scheduled graph to build resolved
  CMD3 commands from typed input/shared/next-norm descriptors, rejecting stale descriptors that do
  not match the current graph capability. CPU
  upload inputs for CMD3 now materialize as scheduler-typed whole-phase inputs before Metal buffer
  allocation, so mismatched normed/residual state is an explicit unsupported input instead of a
  silent no-op/fallback. Metal post-attention prep inputs for CMD3 now materialize as a Metal-owned
  payload carrying scheduler-typed GPU-resident state plus the declared active-route count before
  the Metal expert phase consumes buffers. CMD3 phase buffers now carry Metal-owned lifecycle
  intent before release/recycle. The Metal CMD3 bridge now builds one `MetalCmd3ExecutionPlan` that
  validates phase shape, shared source/shape, active expert payloads, next-norm, and combine
  dispatch before Obj-C encoding begins. The actual shared gate/up/down, next-norm application, and
  shared down execution still live in the older Metal/CPU phase helpers. That execution should
  become part of the same CMD2/CMD3 builder model as routed experts.
- Qwen-VL needs a typed pre-MoE adapter: image preprocessing, vision embeddings, MRoPE, and position
  plans should feed the same text/MoE runtime rather than spreading multimodal branches through the
  execution loop.

## Target Modules

The refactor should break the current monolith by ownership boundary, not by "fast" versus
"generic".

- `model_family`: Qwen-family traits and structs for tensor names, dimensions, attention schedule,
  RoPE/MRoPE, expert layout, shared expert layout, and dtype/quantized layout support.
- `weights`: resident dense weight blob, manifest, typed tensor handles, alignment validation, and
  pull-time conversion.
- `experts`: fixed-slot expert pack layouts, validation, cache build, layer file handles, reusable
  whole-expert read buffers, optional Metal-backed slot storage, and read metrics.
- `capabilities`: graph-stage capability resolution and explicit unsupported errors for missing
  family/dtype/layout/device implementations.
- `scheduler`: per-token/per-layer execution scheduler that owns the serial GPU -> SSD -> GPU policy,
  attention/KV implementation placement, active expert issue/finish, routing placement,
  capability-selected stage implementations, and CMD sequencing.
- `metal`: command-buffer builders and kernels for CMD1, CMD2, CMD3, LM-head/topK, KV cache, and
  recurrent-state operations. Command-context/status/failure handling, shader-source ownership, and
  the declared command wait policy, kernel-name/pipeline-plan, and pipeline-handle release surfaces
  are now extracted here; the next step is to move the concrete encoder builders behind the same
  module boundary.
- `state`: KV cache, linear-attention recurrent state, hidden/residual/normed buffer ownership,
  token position state, and multimodal position plans.
- `runtime`: model-family-agnostic generation loop that composes scheduler, weights, experts, metal,
  tokenizer output, and sampling.
- `vision`: Qwen-VL preprocessing and image/token position adaptation that hands typed inputs to
  `runtime`.

## Implementation Phases

1. Lock the typed model layout surface.
   Keep `QwenMoeModelLayout` as the source for dimensions, K policy, layer kind, shared expert
   shape, and Q4 fixed-slot offsets. Add only generic/typed extensions needed by BF16/F16/future Q
   variants. Do not add new Qwen3.5-only execution branches.

2. Add graph capability validation.
   Define the stage graph and validate every required stage at plan/load time. Unsupported
   family/dtype/layout/device combinations must fail with named missing stages instead of silently
   falling back to older CPU/component/direct-call paths.

3. Extract expert storage and readers from `legacy.rs`.
   Move fixed-slot layout validation, layer reader opening, positioned read helpers, reusable
   buffers, read metrics, fixed-slot packed-record fixture coverage, PBQ4 parsing/writing/wire-size
   accounting, scale/bias decoding, layer rewrite/reuse policy, cache completeness checks, fixed-Q4
   upgrade compatibility, and PBQ4 import/build compatibility into `experts`. Runtime reads should
   return typed `ExpertSlot` or `ExpertMetalSlot` handles with offsets, not split gate/up/down
   owners.

4. Introduce the scheduler API before optimizing it.
   Create a scheduler that owns per-token/per-layer state: active expert IDs, pending read jobs,
   reusable slot leases, routing placement, graph-stage implementations, and the order of
   CMD1/CMD2/CMD3. Initially call existing Metal/runtime helpers through it, then shrink the old
   direct calls. Direct calls that cannot be represented as graph stages should become unsupported
   errors until implemented.

5. Extract dense weights into `weights`.
   Make the resident dense blob and typed projection descriptors a module-level data model. Keep Q4
   and BF16/F32 projection descriptors parameterized by dtype/layout while sharing the same runtime
   binding flow.

6. Make CMD1/CMD2/CMD3 explicit.
   Move Metal command construction behind builders whose inputs are typed state and typed weight or
   expert handles. Remove component-upload expert commands once fixed-slot and dtype/layout
   capability tests cover the shared path.

7. Move state ownership out of the generation loop.
   Represent hidden, residual, normed, KV, recurrent, and next-layer buffers as state objects with
   clear CPU-visible and GPU-resident transitions. CPU vectors are allowed for tokenizer input,
   sampling output, diagnostics, and declared CPU graph stages. Full-attention KV records now carry
   typed CPU/GPU placement and are resolved by the scheduler; recurrent layer records now carry
   CPU-visible placement before session/cache recording; linear-attention cache state now carries
   CPU/GPU placement and declared lengths before cache mutation or Metal allocation. Continue by
   moving actual Metal cache ownership and remaining next-layer transitions into the same model.

8. Reconcile routing through the scheduler.
   For Qwen3.5 parity, model CPU softmax/topK after router projection. Route selection and score
   source submission now have scheduler descriptors, and router score production now resolves a
   `weights`-owned projection descriptor before scores are submitted. Continue by moving the actual
   projection/readback execution into typed CMD2 builders. If GPU routing is kept or later proven
   better, it must be selected through the same scheduler policy for all variants, with parity tests
   proving equivalent logits/topK.

9. Fold shared experts into the same command model.
   Treat shared gate/up/down and shared down as scheduled work inside CMD2/CMD3, not as a side cache
   or separate phase with different ownership.

10. Isolate Qwen-VL as an adapter.
   Move image preprocessing, vision tensor handling, MRoPE, and multimodal position planning into a
   `vision` boundary that emits the same runtime inputs used by text-only MoE.

11. Add parity and capability tests before benchmark claims.
    Required coverage: fixed expert layout, Q4 FMA dequant, active expert gate/up/down, shared
    expert, routing logits/topK, residual/norm handoff, linear attention recurrence, full attention,
    KV/recurrent state updates, CMD buffer inputs/outputs, and per-layer golden fixtures where
    available. Capability tests should prove supported models fully resolve and unsupported
    variants fail with precise missing-stage errors.

12. Benchmark sustained decode last.
    Compare pb `decode_tok_s` against upstream generation tok/s only after correctness and data-flow
    parity are in place. Report TTFT/prefill separately.

## Immediate Work Queue

1. Stop adding runtime behavior to `legacy.rs` except small call-through shims needed during
   extraction.
2. Add the graph capability validator and start converting silent fallbacks into explicit
   unsupported errors.
3. Move the remaining expert conversion/build call sites from `legacy.rs` into `experts`, keeping
   tests with the moved code.
4. Add a scheduler module with a narrow API for "issue active expert reads" and "finish active
   expert slots", then route current runtime expert reads through it.
5. Convert active expert execution to consume scheduler-owned whole-expert slots only. Keep PBQ4 as
   a conversion/input format, not a runtime branch.
6. Extract dense projection descriptors and resident blob ownership into `weights`.
7. Turn the current fused post-attention prep and expert phase helpers into named CMD2/CMD3 builder
   calls with typed inputs. CMD1 input state, CMD2 routing outputs, CMD3 CPU/GPU inputs, and CMD3
   deferred hidden/next-normed outputs now have typed state metadata and scheduler validation;
   continue by moving the Metal command encoding behind builder calls that produce those state
   objects directly.
8. Move state ownership for full-attention KV, recurrent, and next-layer buffers out of the
   generation loop. Full-attention KV now has typed CPU/GPU placement and scheduler validation;
   recurrent layer records now have typed CPU-visible placement; linear-attention cache records now
   have typed CPU/GPU state descriptors. Continue with Metal cache ownership and remaining
   next-layer buffer transitions.
9. Add parity and capability tests around every extraction so behavior moves without hidden semantic
   changes or fallback paths.
10. Revisit the `2+2=` K=4 drift by comparing logits/state through the unified path. Do not revert
   architecture-aligned data flow because the current math is wrong; fix the math.

## Verification Discipline

- Run focused unit/parity tests for the module being extracted.
- Add capability tests whenever a new model family, dtype, layout, CPU/GPU placement, or graph stage
  is introduced. Supported combinations must fully resolve; unsupported combinations must fail with
  precise errors.
- Run `cargo test --all-targets` after behavior-affecting Rust changes when feasible.
- After FlashMoe backend changes, run:
  `target/aarch64-apple-darwin/release/pb flashmoe infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="`
- Treat benchmarks as final confirmation, not as the driver for what to build next. Do not run
  FlashMoe microbenchmarks until the bulk of the ownership-boundary refactor is complete and the
  unified graph is compiling with parity/capability tests.

## Non-Goals

- No second-class Qwen MoE execution path with different buffer ownership, scheduling, or CPU/GPU
  handoff behavior.
- No silent fallback from missing implementation to a different dtype, buffer shape, scheduler,
  runtime, CPU/GPU placement, or backend.
- No hidden environment toggles for scheduler behavior.
- No application expert cache, LZ4 main path, mmap expert reads, dispatch_io, broad prefetch, or
  speculative expert reads unless a fresh architectural reason invalidates upstream's design.
- No Qwen3.5-only runtime fork that prevents Qwen/Qwen-VL from using the same scheduler and buffer
  lifecycle.
- No isolated microbenchmark work that does not move data flow, scheduling, I/O, or CPU/GPU
  ownership toward this plan.

## Goal Prompt For Follow-On Work

Use this prompt to drive implementation quickly without drifting back into experiments:

```text
We are implementing docs/flashmoe-architecture-parity-plan.md now. Move code toward the target
architecture quickly, make it compile, and make the tests/smoke work. Do not stop at another plan.

North star:
- One scheduled FlashMoe execution graph for Qwen-family MoE models.
- Q4/BF16/F16 and Qwen3.5/Qwen/Qwen-VL differences are typed implementations of graph stages, not
  separate runtimes.
- Missing support is an explicit unsupported-capability error, never silent fallback.
- No isolated tok/s experiments, microbenchmarks, or Q4-only fast paths until the bulk of the
  refactor is complete.

Start by inspecting git status and the current FlashMoe modules. Treat existing changes as live user
work. Do not revert them. Avoid adding new production behavior to src/inference/flashmoe/legacy.rs
except narrow shims needed to extract code safely.

Implement in this order:
1. Add the graph/capability model: define the scheduled stages, supported placements, and explicit
   unsupported errors for missing family/dtype/layout/Metal/CPU-GPU implementations.
2. Move expert storage/read policy out of legacy.rs into experts: fixed-slot layouts, positioned
   reads, reusable whole-expert slots, read metrics, and PBQ4 import compatibility.
3. Add scheduler as the owner of active expert issue/finish, slot leases, routing placement, and
   CMD1/CMD2/CMD3 sequencing. Route the current runtime through it.
4. Convert active expert execution to consume scheduler-owned whole-expert slots with typed offsets.
   Component-buffer or CPU/component fallback should become an unsupported error unless represented
   as a declared graph-stage implementation.
5. Extract resident dense weight/projection descriptors into weights with one binding flow across
   Q4/BF16/F32 layouts.
6. Turn the current fused post-attention prep and expert phase helpers into explicit CMD2/CMD3
   builder calls with typed inputs.
7. Move state ownership out of the generation loop: hidden, residual, normed, KV, recurrent, and
   next-layer buffers should have clear CPU-visible and GPU-resident transitions.
8. Keep Qwen-VL as a pre-MoE adapter that emits typed runtime inputs instead of branching through
   the MoE execution loop.

For every extraction, add focused parity or capability tests. Supported combinations must fully
resolve. Unsupported combinations must fail with precise missing-stage errors. If a parity-aligned
change makes `2+2=` wrong, debug logits/state/math through the unified path instead of reverting the
architecture.

Verification:
- Run focused tests for each moved module.
- Run cargo test --all-targets after significant behavior changes.
- After FlashMoe backend changes, run:
  target/aarch64-apple-darwin/release/pb flashmoe infer --raw --max-tokens 1 --top-k 1 --temperature 0 "2+2="
- Do not run FlashMoe benchmarks or chase tok/s until the main ownership-boundary refactor,
  capability validation, scheduler path, and CMD graph are in place and compiling.
```
