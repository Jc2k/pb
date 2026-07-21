# Qwen3-Coder-Next device-resident prefill graph

Status: **Design record; implementation active.**

This plan continues the shipped Qwen3-Coder-Next layer-major matrix command into a genuinely
device-resident layer-major prefill graph. The built-in FlashMoe runner is the implementation
target. Explicit GGUF requests must remain functional through llama.cpp, but llama.cpp is a
compatibility surface rather than a fallback, performance target, or correctness oracle for a
native request.

## Why another graph is justified

The 2026-07-20 command is real and valuable: a 4,354-token resident prompt fell from the preserved
689.6-second scalar baseline to 59.685 seconds, exact state parity passed for resident and forced
streamed experts, and request allocation stayed within the 5% gate. It traverses each prompt chunk
layer-first, batches dense and expert projections, preserves token-order recurrence, and asks the
existing scheduler for one sorted unique expert union per layer.

It is not yet the final graph geometry. The current layer loop stores every row's hidden state in
`Vec<f32>`. Dense projection, attention, post-attention/router, expert, and next-layer norm helpers
accept CPU slices and return CPU vectors. Their Metal builders commit and wait, read complete
matrices back to the host, then a later builder uploads them again. CPU top-10 routing does require
one bounded host decision per layer, but full hidden and normalized matrices do not.

For this plan, “device-resident layer-major graph” therefore has an observable meaning:

- the chunk's hidden and normalized matrices have one typed Metal owner;
- those matrices are not read back or re-uploaded between transformer layers;
- the host receives only data required by an authoritative CPU boundary: routing candidates,
  persistent KV/session state while that remains CPU-owned, opt-in parity fingerprints, and the
  final hidden row;
- queued Metal commands consume prior typed outputs directly; a graph error is terminal and never
  switches precision, expert policy, model family, scheduler, or backend; and
- resident and streamed expert graphs share the same layer command. Only scheduler-owned expert
  acquisition differs.

The intended steady data flow is:

```text
token embeddings
      │ one chunk upload
      ▼
typed device hidden + next-norm matrices
      │
      ├─ full attention: QKV/causal attention/KV append
      │  or linear attention: projections/token-order recurrence
      ▼
device post-attention residual + norm + router scores
      │ bounded router readback
      ▼
existing scheduler → resident mapped union OR parallel positioned reads
      │
      ▼
device routed/shared expert matrices → next layer device state
      │
      └─ repeat; read final row only after the final norm
```

## Invariants

- The prepared resident/streamed expert decision remains binary and immutable for the loaded
  graph. There is no partial application cache, migration policy, or alternate scheduler.
- Streamed expert I/O remains scheduler-owned parallel `pread`; the OS page cache remains the
  cache. Resident execution issues zero expert reads.
- The common affine-Q4 Qwen expert command serves every graph for which capability preparation
  declares it. A Qwen3-Coder-Next-only experiment may prove the geometry, but no hidden
  checkpoint-name branch or Q4 micro-path may bypass typed graph resolution.
- Full-attention KV records, linear-attention convolution/SSM state, router order and weights,
  recurrent trace, final hidden state, and greedy continuation retain exact scalar parity.
- Prefix restore and chunk boundaries are graph inputs, not reasons to reset state or retry the
  scalar path.
- Resource owners release every transient buffer and command on success and every error path.
- Native FlashMoe load or execution failure stays native and explicit. llama.cpp is selected only
  by its existing explicit model/GGUF resolution path.

## Phase 0 — Make the host boundary measurable

Status: **Implemented and natively qualified.**

Add cumulative Metal command-submission, host-upload, and host-readback counters to the resource
ledger. Capture per-prefill deltas in native generation telemetry and the harness journal. The
current scalar and layer-major commands must report their actual traffic; counters may not infer
bytes from model geometry.

Gate: a deterministic ledger fixture proves upload/readback accounting and backward-compatible
event decoding. A short real Qwen run records the shipped matrix command baseline before buffer
ownership changes.

The 2026-07-21 resident baseline used a 40-token raw prompt and the forced shipped layer-major
matrix command. It completed prefill in 793 ms at 50.41 token/s, created 239 Metal command buffers,
uploaded 348,977,168 bytes, and read back 186,818,560 bytes. It ended with zero active general
buffers, zero transient expert buffers, and zero in-flight commands. The continuation was `a`.
This closes the measurement gate and proves that the remaining graph work is cross-layer device
state ownership rather than traversal order or expert scheduling.

## Phase 1 — Introduce a typed chunk owner

Define row-aware GPU matrix descriptors rather than reusing scalar `len` descriptors. A
`QwenPrefillChunkState` owns hidden and optional prepared-next-norm buffers, row count, width,
start position, current layer, and resource-ledger leases. State transitions validate role,
geometry, layer, placement, and ownership before encoding.

Embeddings are gathered once into the initial hidden matrix. The final owner can read the last row
for sampling and can materialize complete state only for session capture or explicit parity output.
Drop and error paths recycle ordinary buffers and purge transient expert staging through the
existing resource owner.

Gate: state-transition tests reject wrong roles, rows, widths, layers, and double release. The
resource snapshot returns to its pre-request live counts after success and injected failure.

## Phase 2 — Keep dense, attention, and post-attention state on Metal

Extend resident projection matrix builders to accept typed input buffers and return typed output
buffers. Q/K normalization, rotary application, causal attention, output projection, residual add,
post-attention RMSNorm, and router projection consume and produce graph-owned buffers.

Full-attention KV remains exact. Initially, only the newly appended K/V rows may cross to the
CPU-owned session cache; queries, attention output, hidden, and normed matrices stay on device.
Once exact restored-prefix coverage exists, move KV storage behind its own typed Metal owner and
materialize CPU snapshots only at the durable session boundary.

Linear-attention projections consume the device norm matrix directly. Its convolution and SSM
updates remain token ordered in the existing Metal command, and its output becomes the device
post-attention input without a full matrix readback.

Gate: every layer kind passes zero-prefix, restored-prefix, one-row, threshold, and multi-chunk
state parity. Telemetry shows no hidden/normed host transfer between layers.

## Phase 3 — Join router, scheduler, and expert output

Post-attention returns a typed residual/norm owner plus only the router score matrix to the host.
CPU softmax/top-10 preserves the current row-local order and exact weights. The existing scheduler
reduces routes to the sorted unique expert union and resolves resident mapped slots or streamed
scratch exactly as it does today.

The layer-major expert builder consumes the post-attention device buffers directly. It must not
rebuild grouped hidden rows on the CPU. A gather kernel forms expert-grouped rows, affine-Q4 expert
matrices and the shared expert execute, and the combine/next-norm command produces the next typed
chunk state. Command-queue ordering replaces host waits wherever the next host-visible fact is not
yet needed.

Gate: resident execution performs zero expert reads; forced 32 GiB execution reads every unique
expert once per layer request and retains no expert identity after it. Route order, combine bits,
and all state fingerprints match the scalar and shipped matrix references.

## Phase 4 — Resolve one production graph

Capability preparation gains a distinct device-resident graph capability only when every required
dense, attention, recurrent, expert, and state binding exists. `auto` continues to use the promoted
shipped matrix command until the new graph passes all gates. Hidden harness modes can force the
shipped matrix reference and device-resident candidate inside one loaded runtime.

There is no runtime probe/fallback. An explicit candidate request that cannot reserve or encode its
declared graph fails before mutation of prompt state. After promotion, `auto` selects the new graph
for the same resource-resolved long-suffix geometry and retains the scalar path for short suffixes
and unsupported prepared layouts.

## Promotion gates

Correctness:

- exact final hidden, per-layer KV, router/recurrent, and linear-state fingerprints;
- identical greedy continuations for zero prefix, restored prefix, 17/7-row forced chunks, the
  31/32/33 threshold, and a multi-chunk suffix;
- resident and forced-streamed parity in one loaded runtime; and
- unchanged upstream MLX-LM raw-token parity and structured `2+2=` smoke.

Graph geometry:

- zero hidden/normed host upload or readback between layers;
- at most one required router synchronization per MoE layer, plus explicit KV/session boundaries;
- final-row-only hidden readback in a non-session, non-parity request; and
- zero live transient buffers and zero in-flight commands at the request boundary.

Performance and resources:

- no regression from the 59.685-second/72.95-token/s 4,354-token shipped matrix result;
- target at least 1.5x over that result, with a stretch target below 30 seconds;
- no more than 5% additional request/session allocation over the resident model baseline; and
- equivalent graph behavior under the forced 32 GiB streamed policy.

Compatibility:

- `cargo test --all-targets`, documentation checks, and local Metal compilation pass;
- the required native one-token smoke exits zero with a sensible answer; and
- an explicit GGUF request still resolves and executes through llama.cpp without sharing the new
  FlashMoe graph state.

## Planned commits

1. `perf: measure flashmoe prefill host boundaries`
2. `refactor: own qwen prefill matrices on metal`
3. `perf: chain qwen layer-major graph state`
4. `perf: promote device-resident qwen prefill`

Each commit keeps `docs/flashmoe-architecture-parity-plan.md` aligned. Promotion evidence belongs
in a new dated section of the existing prefill qualification record; a plan or unit test alone is
not evidence that the production graph has changed.
