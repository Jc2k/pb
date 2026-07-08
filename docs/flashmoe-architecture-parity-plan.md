# FlashMoe Architecture Parity Plan

This is the implementation plan for bringing pb's MoE backend into architectural parity with
`danveloper/flash-moe` while keeping pb's production shell: Rust pull/conversion, production
tokenizers, structured request APIs, Qwen model-family flexibility, Qwen-VL image support, and the
shared inference facade with llama.cpp.

The target is not a special "fast path" beside a generic slow path. The target is one MoE execution
architecture for Qwen-family MoE models. Model-family traits describe dimensions, tensor naming,
quantized layouts, vision adapters, and dtype variants, but they must not fork buffer ownership,
expert scheduling, command-buffer structure, or CPU/GPU handoff policy.

## Principles

- FlashMoe is the primary backend for supported MoE models. llama.cpp is the fallback for unsupported
  or intentionally delegated models.
- The MoE backend owns expert buffers, resident dense buffers, scheduler state, KV and recurrent
  state, command-buffer sequencing, and read scheduling.
- Model variants supply metadata through typed layouts and traits. They do not get a separate slow
  execution pipeline.
- Rust generics and enums should express dtype/layout variation without changing the execution flow.
- Qwen3.5-A17B parity is the first concrete target, but the shape should also carry Qwen MoE and
  Qwen-VL through the same scheduler and buffer lifecycle.
- Correctness must be proven with parity tests before full-model throughput claims.

## Upstream Behaviors To Mirror

Source of truth: `danveloper/flash-moe`, especially `metal_infer/infer.m`,
`metal_infer/shaders.metal`, `repack_experts.py`, and the Q4 optimization notes.

- Throughput comparison uses decode/generation throughput, not prefill-inclusive wall time.
- Non-expert weights are stored in one 64-byte-aligned binary blob and mapped once.
- Experts are packed as fixed per-layer files. For Qwen3.5-A17B Q4 each expert is exactly
  7,077,888 bytes with fixed gate/up/down packed-weight, scale, and bias offsets.
- Active experts are read with parallel positioned reads into reusable whole-expert buffers.
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

## Current pb Architectural Gaps

- PBQ4 record parsing and per-component upload still sit inside runtime expert reads. This copies and
  splits what should be one whole-expert buffer owned by the MoE scheduler.
- The fixed upstream expert layout was tested only as a cache-format/parser change; it did not change
  runtime buffer ownership, so it was not a real parity implementation.
- The current execution loop still has CPU-owned hidden/residual vectors and clone/readback seams in
  places where the backend should carry GPU-owned buffers between command buffers.
- Routing has GPU topK machinery in some paths, while upstream kept CPU softmax/topK for the measured
  Qwen3.5 path. The unified engine should make this an explicit scheduling decision, not an
  accidental side branch.
- Shared-expert Q4 handling is closer to upstream, but it is still grafted onto the existing
  component-buffer expert phase instead of living inside a single MoE command structure.
- Qwen-VL support currently encourages conditional flow in the runtime. Vision-specific work should
  happen before text/MoE execution, then hand the same execution engine a typed sequence, position
  plan, and model layout.

## Proposed Modules

The refactor should break the current monolith by ownership boundary, not by "fast" versus "generic".

- `model_family`: Qwen-family traits and structs for tensor names, dimensions, attention schedule,
  RoPE/MRoPE, expert layout, shared expert layout, and quantized dtype support.
- `weights`: resident dense weight blob, manifest, typed tensor handles, and pull-time conversion.
- `experts`: fixed-slot expert pack layouts, validation, cache build, layer file handles, reusable
  whole-expert read buffers, and read metrics.
- `scheduler`: per-token/per-layer execution scheduler that owns the serial GPU -> SSD -> GPU policy.
- `metal`: command-buffer builders and kernels for CMD1, CMD2, CMD3, LM-head/topK, KV cache, and
  recurrent-state operations.
- `state`: KV cache, linear-attention recurrent state, token position state, and multimodal position
  plans.
- `runtime`: model-family-agnostic generation loop that composes scheduler, weights, experts, metal,
  tokenizer output, and sampling.

## Implementation Phases

1. Define typed Qwen MoE layout traits.
   Capture Qwen3.5-A17B constants as one implementation. Add tests for tensor names, layer schedule,
   dimensions, expert offsets, shared expert dimensions, and Qwen-VL position-plan compatibility.

2. Replace runtime PBQ4 parsing with whole-expert layout handles.
   Keep PBQ4 only as an import/build compatibility format if needed. Runtime expert reads should
   return typed `ExpertSlot` handles containing one owned/reusable byte buffer or Metal buffer plus
   layout offsets.

3. Move expert reads into scheduler-owned reusable buffers.
   Allocate K reusable whole-expert slots per layer step, read directly into their storage, and bind
   gate/up/down by offsets. The scheduler, not ad hoc layer code, decides when reads are issued and
   finished.

4. Port the upstream command topology as the only MoE topology.
   Build explicit CMD1/CMD2/CMD3 command builders. Remove alternate component-upload expert command
   paths once parity tests cover dtype/layout variants.

5. Make GPU residency the normal handoff.
   Carry hidden, residual, normed, KV/recurrent state, and next-layer normed buffers through the MoE
   runtime with explicit ownership. CPU vectors should exist for tokenizer input, sampling output,
   diagnostics, and fallback-only unsupported operations.

6. Reconcile routing with upstream.
   For Qwen3.5 parity, use CPU softmax/topK after the router projection. If later GPU routing wins
   with the unified command structure, it must replace the policy for all supported variants through
   the same scheduler interface.

7. Add parity tests before benchmark claims.
   Synthetic fixed expert layout, Q4 FMA dequant, active expert gate/up/down, shared expert, routing,
   residual/norm handoff, linear attention recurrence, full attention, and per-layer golden fixtures.

8. Benchmark sustained decode.
   Compare pb `decode_tok_s` against upstream generation tok/s. Report TTFT/prefill separately.

## Non-Goals

- No second-class generic MoE path that keeps excessive copies because it is "not the fast path".
- No hidden environment toggles for scheduler behavior.
- No application expert cache, LZ4 main path, mmap expert reads, dispatch_io, broad prefetch, or
  speculative expert reads unless a fresh architectural reason invalidates upstream's measured
  failures.
- No Qwen3.5-only runtime fork that prevents Qwen/Qwen-VL from using the same scheduler and buffer
  lifecycle.
