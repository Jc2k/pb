# How pb fits together

pb is a local control plane around a coding model. Its architecture is easiest to understand by
separating the interfaces a person uses, the harness that owns work, and the execution resources
where tools run.

## The system at a glance

```text
terminal (`pb queue`) ─────┐
                          ├── local pb service ── typed session events ──┐
browser (`pb serve`) ─────┘          │                                  │
                                    ├── agent runner + local inference  │
                                    │          │                        │
                                    ├── workflow state machine          │
                                    │          │                        │
                                    ├── policy + capability gates       │
                                    │          │                        │
                                    └── project execution environment ──┘
                                               │
                                  repository, checks, LSP, MCP, caches
```

The terminal and web application are adapters over the same session model. The service exposes an
HTTP/SSE interface to the browser and a local Unix-socket RPC interface to the CLI. Both surfaces
render the same typed events and server-authored teammate chatter, and can reconnect to the same
persisted session.

**Shipped.** The daemon appends each envelope to authoritative history before broadcasting it.
Terminal attachment subscribes while that history is locked, then switches from the captured
snapshot to live delivery without a publication gap. A lagged terminal receiver replays every
missing sequence from history before continuing, and terminal completion is announced only after
the final replay. History retention is dependency-aware: the nominal bound is soft when a retained
projection needs an earlier tool call, superseded entry, final/block reason, check, or commit.
The browser SSE stream begins with a revisioned session snapshot and sends another snapshot after
state-changing events. A reconnect cursor that has fallen outside retained history produces an
explicit history reset instead of silently splicing two non-contiguous transcript windows.
Every durable lifecycle transition, including queued follow-up and recovery work, publishes a typed
state event before its mutation response completes; the browser never has to refetch to discover it.

The browser starts repository-backed work with a registered project ID; resolving that identity to a
display name and filesystem path belongs to the daemon. CLI commands may still supply an explicit
workdir. Registered projects have durable IDs, so renaming or moving a repository does not detach its
restored sessions or usage from the project page. A single server snapshot refreshes the registry and
its reconciled session list while project pages remain open.
Mutating HTTP responses use a typed error code and server-authored message, so the web adapter does
not infer domain state from bare HTTP status numbers. Integration mutations return the full
authoritative installed collection rather than requiring a second read after a successful write.

## Component responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| CLI and web UI | User intent, questions, session controls, event rendering | Workflow transitions or tool authority |
| Agent runner | Prompt construction, local inference, bounded tool loop, metrics | Permission to invent tools or declare a stage complete |
| Workflow engine | Stage contracts, structured artifacts, checks, review, repair, commit, terminal outcome | Product intent not supplied by the user |
| Policy engine | First-match allow/deny/ask decisions over exposed tool calls | Adding capabilities to a stage |
| Environment control plane | Workspace, container/local executor, services, networks, cache leases, cleanup | Deciding that a model request deserves broader access |
| Session store | Events, workflow checkpoints, metrics, restoration | Remote publication |
| Local inference backends | Tokenization and generation for supported models | Acceptance, review, or repository authority |

This ownership split is the central architectural choice. A local model can reason about the work,
but durable state transitions are made by Rust code against explicit contracts.

## Control flow and data flow

The **control flow** carries intent, stage, capabilities, policy decisions, structured submissions,
and terminal outcomes. The harness treats these values as authority-bearing.

The **data flow** carries repository evidence, model prompts and completions, tool results, diffs,
check output, and events. Data can inform a transition, but raw text does not itself grant one. A
model saying “tests pass” is data; a current check receipt accepted by the workflow is evidence.

Keeping those flows separate prevents common agent failure modes:

- prompt text cannot expose a forbidden tool;
- a teammate response cannot advance the primary workflow;
- a successful unrelated command cannot satisfy a named check;
- a stale review cannot approve a later mutation;
- a final response cannot create a managed commit.

## Repository and workspace

pb resolves the repository root and the task focus independently. Multi-component projects can
describe workspace topology, executors, generated outputs, and affected checks. Apple-container
delivery happens in a task-owned worktree so the harness can compare the accepted baseline with the
delivered change and promote only declared outputs. Explicit local execution instead uses the
registered repository itself; pb captures exact content and Git-control baselines and stops on
concurrent drift, but does not claim filesystem isolation from another local session or editor.

The selected environment can be:

- an Apple-container-backed session with pb-owned resources and capability-scoped services; or
- explicit local execution for projects whose toolchain genuinely requires the host.

Local execution is a compatibility mode, not filesystem or process containment. The
[security model](security.md) makes that boundary explicit.

## Persistence and recovery

Daemon sessions are serialized into repository-local Git notes. On restart, sessions that were
queued, running, or paused are restored as paused rather than silently resumed. Workflow
checkpoints retain structured state, counters, and accepted evidence so recovery does not depend on
replaying a model's prose as authority.

Environment resources have separate lease records. The supervisor reconciles abandoned resources
at startup and reaps expired idle sessions. Persistent images and approved caches may be reused;
task containers, services, networks, and ephemeral workspaces remain session-owned.

## Host power lifecycle

**Shipped.** The web service owns one process-wide macOS idle-sleep assertion. Queue dispatch marks
it active before starting a running session and releases it only after dispatch finds no next item.
Pausing for user input releases the assertion; answering reacquires it before model work resumes.
The live global web preference updates the same owner, so toggling it during a task starts or stops
a native IOKit `PreventUserIdleSystemSleep` assertion immediately instead of waiting for the next
queue transition. pb releases the assertion by its IOPM assertion ID, while process exit provides a
final OS-owned cleanup boundary. No helper subprocess is used.

**Shipped.** The installed LaunchAgent declares one HTTP socket using the configured address and
port. pb adopts that descriptor through `launch_activate_socket`; a direct development process
falls back to an ordinary Tokio bind. When the configured address is not loopback, launchd
advertises `_http._tcp` through Bonjour for the installed service, while a direct server owns an
equivalent native `DNSServiceRegister` lease. Loopback listeners are never advertised. This keeps
launchd out of the development request path without creating a second server implementation.

**Configurable.** The Bonjour registration is eligible for the macOS Sleep Proxy path because pb
does not set `kDNSServiceFlagsWakeOnlyService`. Actual wake-on-HTTP remains a host and network
capability: **Wake for network access**, compatible hardware and power state, and a reachable Sleep
Proxy or local-subnet wake path are outside pb's control. Network-visible HTTP also crosses the
default trust boundary because pb does not add authentication or TLS.

## Local MoE inference

**Shipped.** FlashMoe runs supported Qwen MoE families through one typed runtime, resident dense
weights, scheduler-owned whole-expert acquisition, and Metal command builders. Graph resolution uses
parallel positioned reads, reusable whole-expert slots, and the operating-system page cache for
large expert corpora. When a supported Qwen/GLM variant's complete fixed-slot expert corpus fits
beneath the sampled Metal working-set limit with explicit transient/session headroom, it instead
maps, prefaults, and retains the complete expert table for the graph lifetime. The mode never changes
during inference and does not introduce a partial cache or eviction policy. Metal expert staging
allocations use a separate bounded pool whose bytes are overwritten on every checkout; it is
allocation reuse rather than an expert-identity cache and is drained under working-set pressure.

**Configurable.** Qwen3-Coder-Next uses a distinct native FlashMoe family and the indexed
`mlx-community/Qwen3-Coder-Next-4bit` source by default. Its graph preserves the published
48-layer hybrid attention schedule, 512 experts, and top-10 routing instead of applying Qwen3.5's
top-4 profile. Cache construction keeps its large affine-Q4 matrices quantized, splits grouped
linear-attention projections into canonical row order, and expands only the small affine-int8
router and shared-gate projections to resident BF16. It widens recurrent `A_log` vectors once to
the existing F32 decay-kernel contract. Its prepared norm tensors are already sanitized by the MLX
conversion and are consumed directly instead of receiving a second `1 + weight` transform. The
same load-time memory calculation chooses complete resident expert slots when the whole corpus
fits, or scheduler-owned positioned reads when it does not. Its chat contract is
non-thinking-only, so prompt measurement and generation both disable thinking for this family.
For the exact affine-Q4 capability, fresh suffixes of at least 32 tokens use the shipped device-
resident layer-major graph selected from the live Metal working-set budget. One typed owner carries
hidden and prepared-next-norm matrices across dense projection, hybrid token-order recurrence or
causal full attention, post-attention, routed/shared experts, combine, and the following layer.
CPU top-10 routing forms one sorted unique expert union per layer; router candidates, authoritative
full-attention KV/session records, opt-in parity fingerprints, and the final normalized row are the
host-visible semantic boundaries. Ordinary generation does not compute diagnostic router/recurrent
fingerprints.

The prepared resident graph clones mapped slots with zero expert reads, while the prepared streamed
graph acquires the same union through scheduler-owned parallel positioned reads. Both use the same
two-input-row affine-Q4 weight traversal and exact scalar dot/combine order. Exact hidden, KV,
routing/recurrent, and linear-state fingerprints gate the graph; other family/layout combinations
remain on their prepared scalar graph, and execution failure never changes model, precision,
scheduler, or backend. The implementation record and qualification are documented in the
[device-resident graph plan](../qwen3-coder-next-device-resident-prefill-plan.md).
An explicit GGUF URI still selects llama.cpp; failure to load an already-selected native FlashMoe
graph is terminal and never changes the backend behind the user's request.

**Configurable.** GLM-5.2 extends that runtime at typed boundaries. Pull accepts indexed MLX MXFP4
or unindexed Colibri tensors through source adapters, normalizes either representation into pb's
canonical resident-dense and fixed-slot Q4 layouts, and records a source-format-independent runtime
manifest. MXFP4 E2M1/E8M0 groups are decoded row-wise and requantized once; Colibri packed int4 and
int8 input/output tensors retain their existing import path. Pull also preserves an external
`chat_template.jinja` when the tokenizer configuration does not embed one, with embedded templates
taking precedence at load. Its first three dense MLPs remain
resident; sparse layers reuse the existing expert scheduler and always-active shared-expert
command. MLA stores a normalized 512-value latent plus a 64-value rotary key per token, with Metal
projections and a typed weight-absorption stage. For MLX's pre-absorbed Q4 per-head projections,
`q_a`, `kv_a`, both projection RMSNorms, `q_b`, RoPE, current-record append, causal absorbed scores,
softmax, latent-context reduction, output unembedding, `o_proj`, residual/post-attention RMSNorm, and
router projection execute in one ordered Metal command on sparse layers. The compressed-KV cache
remains authoritative CPU-visible session state: prior records are uploaded before the command, and
the current normalized latent and rotated key return with the CPU routing candidates after it
completes. Fused Colibri `kv_b_proj` weights retain their CPU transpose implementation. Sigmoid/noaux
routing uses the correction bias only for expert selection and keeps the checkpoint's routed scaling
factor.

**Shipped for the pinned request-scoped profile.** On Apple Silicon, the exact published
DeepSeek V4 Flash IQ2_XXS/Q2_K GGUF profile extends the same FlashMoe runtime. Pull validates the
bounded GGUF directory and publishes one resident tensor store plus 43 page-aligned expert packs.
Load binds a fixed 43-layer graph before inference and compiles its specialized Metal surface:
four-stream hyperconnections, dense/ratio-4/ratio-128 attention, raw and compressed KV, the ratio-4
indexer, grouped output, native JoyAI tokenization, and the output head. Prompt geometry selects one
of two commands in that already-resolved graph: prompts shorter than one 32-row matrix tile retain
the exact token accumulation order, while longer zero-prefix prompts execute layer-major Metal
prefill. The batch command routes every prompt row, reduces the result to one sorted unique expert
working set per layer, and asks the shared scheduler for at most eight parallel positioned reads;
each expert is staged once for that layer and the OS page cache remains authoritative. Decode keeps
the ordinary six-slot scheduled read path. There is no top-2 quality profile, runtime error
fallback, tensor probing, DS4 subprocess, application expert cache, or second inference engine.

DeepSeek state is request-scoped. pb resets its raw/compressed/indexer KV and hyperconnection state
before each request. A request that asks for a logical FlashMoe session fails with a named
capability error because the existing Qwen/GLM snapshot cannot represent the complete typed state;
pb never silently restores a partial snapshot or downgrades a session request. The agent backend
reads that load-resolved capability and issues ordinary request-scoped DeepSeek turns, so native
tool loops do not falsely request session restoration.
Source/cache/graph/routing/ABI tests and local compilation of the specialized Metal library are
recorded. The published 86.72 GB imatrix GGUF has completed canonical cache publication and real
Metal load/prefill/decode. All four enforced upstream continuation vectors match, including a
3,844-token case that crosses the top-512 indexed-attention frontier. On an M4 Max, that exact long
case improved from 4.41 to 153.8 prefill tok/s after layer-major migration; pinned `antirez/ds4`
recorded 234.3 tok/s under its one-expert cold SSD-streaming control. A real two-turn DSML loop
executed a native tool call. Complete-state DeepSeek snapshots remain a separate unsupported future
capability rather than part of this request-scoped profile. DSA sparse attention and the MTP
speculative head likewise remain GLM follow-on implementations, and GLM requests beyond
`index_topk` remain explicit unsupported paths.
The detailed implementation and evidence boundary lives in the
[FlashMoe architecture parity plan](../flashmoe-architecture-parity-plan.md).

## Deliberate seams

Several boundaries are intentionally visible rather than folded into one “agent” abstraction:

- [conversation versus delivery](workflows.md);
- [stage capability versus user policy](security.md);
- [local computation versus external data paths](local-privacy.md);
- [model completion versus verified acceptance](user-contracts.md);
- local Ready evidence versus provider-owned publication.

Those seams make it possible to change models, UI surfaces, execution backends, and future
publication providers without moving the source of authority into prompt text.
