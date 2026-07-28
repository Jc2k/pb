# Local privacy model

Local-first is an architectural property of pb's default path: inference, orchestration, session
state, repository inspection, and the user interfaces live on the user's machine. Privacy comes
from making every non-local edge visible and controllable, not from describing the entire product
as offline.

## The local core

```text
local model weights ──→ local inference
                            │
repository ──→ local tools ─┼─→ local workflow + evidence ──→ terminal / browser
                            │
local config + Git notes ───┘
```

Normal prompts are generated inside the pb process and sent to the selected local inference
backend. Session events, workflow checkpoints, durable Goal checkpoints, and supported multi-Task
controller checkpoints are persisted into repository-local Git notes.
Workflow checkpoints may include a bounded bundle of exact bytes from complete small-file reads,
together with their local path/content hashes and provenance. The bundle remains local session
evidence, is revalidated before reuse, and follows the same session-deletion lifecycle; it is not
model memory or a new network edge.
Intrinsic deterministic actions can source such prompt evidence directly from the local controller.
The bytes, hashes, coverage receipts, truthful controller blocks, and audit events stay on the same
local inference and repository-evidence path. There is no assistant-tool transcript emulation,
hosted inference fallback, or added network disclosure.
The web UI is embedded in the binary and served by the local Rust process. The listener stays on
loopback by default. When the user selects a non-loopback address, an installed service advertises
the HTTP socket through launchd and Bonjour; a direct development server creates an equivalent
native DNS-SD registration. That advertisement reveals the service's presence on the local network
but does not send repository or task content by itself.

Inference acceleration is local as well. llama.cpp keeps a live exact-prefix context during an
agent run and stores byte-budgeted restartable state under the platform cache root. FlashMoe keeps
loaded model/Metal runtimes in a bounded idle pool, holds safe prompt, generated-head, and shared
system-prefix checkpoints in memory, and writes compatible KV/MLA/recurrent checkpoints to its own
byte-budgeted cache. State files contain token ids and derived attention or recurrent state, so
they can reflect repository content that appeared in the prompt. Model, tokenizer, template,
runtime-layout, and token hashes prevent incompatible restoration; session filenames are hashed;
directories and files are owner-only on Unix; writes use temporary files and atomic replacement.
Sessionless JSON-constrained Task planning writes only the stable first-system-message checkpoint;
dynamic Task evidence, the JSON constraint engine, and generated artifact are not captured in that
shared root. Managed workflow stage roots likewise contain only the immutable role, terminal action
protocol, handoff authority rule, and exact tool schema. Task text, repository data, contracts,
plans, corrections, and evidence remain in the fresh suffix and in session-owned state.
Carried exact-file evidence is projected to model prompts as path and content only; controller
receipts and their workspace/path hashes remain in local checkpoint and audit state rather than
being duplicated into later prompt suffixes.
FlashMoe mutation constraints reuse those current local reads as a request-scoped immutable virtual
workspace. Source and patch bytes stay inside the pb process, are not persisted as a control-collar
cache, and are discarded with the request. The collar never opens a repository file or starts an
external analyzer. Its durable inference fields are limited to a dialect name, schema digest,
content-free rejection counts, snapshot file/byte counts, and terminal state; paths, source, patch
bodies, argument values, symbols, prompts, and logits are excluded.
Incremental source and patch branch caches are request-local memory only. Qualification reports
persist artifact digests, counts, and timing percentiles rather than corpus source or decoded
payloads.

Rust-edit-capable real-backend requests also build a local native semantic world before inference.
The controller copies the exact repository snapshot to an ephemeral verified shadow;
`pb-control-rust` runs pinned rust-analyzer internals in the pb process, invokes Cargo metadata with
`--offline`, starts no procedural-macro server, and does not run build scripts. It may read already
installed sysroot and dependency source caches, but it adds no network edge and sends no dependency
or repository source to the model. Exact warmed worlds are retained only in bounded process memory;
request snapshots and candidate branches are in-memory leases. Shadows are temporary and durable
events contain only identities, readiness/decision classes, counts, and timings. A cancelled
request may leave its already-started cold analyzer worker running locally until that exact build
finishes; at most one such cold worker runs process-wide, and its shadow/world is then dropped or
retained only by the same bounded in-memory cache. Cancellation adds no network or durable source
persistence.

Python-edit-capable real-backend requests use the same local-only lifecycle. The controller creates
an ephemeral verified shadow, and `pb-control-python` copies relevant Python/stub/package-marker
bytes into a frozen in-memory filesystem before running exact-pinned Astral `ty` internals in the pb
process. The controller also inspects conventional project-local `.venv`/`venv` directories—even
when Git ignores them—and, for one unambiguous safe layout, copies bounded Python source, stubs,
package markers, selected distribution metadata, and `pyvenv.cfg` into that same temporary shadow.
It never executes the environment's interpreter, processes dynamic path injection, contacts a live
LSP or package index, or reads an out-of-project environment. Dependency bytes and identities are
used only by the local analyzer, are type-primed before inference, and are recaptured before
execution; they are not added to prompts or durable events. Generated overlays, diagnostic debt,
and request databases stay in memory; bounded process caches retain only local analyzer worlds and
durable events remain content-free. A cancelled request may leave its one already-started local
Python preparation worker finishing into that bounded cache, but adds no network or durable source
persistence.
`inference.llamacpp_session_cache_enabled=false` and
`inference.flashmoe_session_cache_enabled=false` independently disable disk persistence without
disabling in-process reuse. These controls, their byte budgets, and the optional cache root are
typed user configuration rather than process environment variables.

The hidden managed-cache qualification harness can point FlashMoe at an explicit experiment-only
cache root. It accepts only an absolute empty or absent directory, keeps every inference and
checkpoint local, and refuses to overwrite its JSON report. The report retains the explicitly
selected scratch, cache, model, and output locations plus bounded cache sources, miss reasons,
stage/authority labels, token counts, and digests; it does not copy repository-relative paths,
prompt text, task text, tool arguments, or source content. Its restart arm is a new local pb process;
it adds no hosted inference, telemetry, synchronization, or network edge. The cache itself still
contains token ids and derived model state and therefore remains sensitive local data under the
ordinary cache ownership and cleanup model.

The hidden exact-GGUF tool-fixture runner likewise keeps model loading, prompts, snapshots, and
constraint probing local and does not execute fixture mutations. Its normal JSON report contains
complete calls because the fixture is an explicit local test input; it does not add telemetry or
remote persistence. The separate `--show-transcript` debugging opt-in may print an incomplete tool
argument or source fragment to the invoking terminal on failure. That preview is not persisted by
the runner and is disabled by default.

The DeepSeek V4 Flash extension keeps its model, resident tensors, streamed expert packs, tokenizer,
Metal state, and prompts on the same local path. Its typed hyperconnection/compressed-attention state
is not currently written to the FlashMoe session cache or reused across requests; the state is reset
in process when a request begins, and requests for logical session reuse are rejected explicitly.
Pulling the checkpoint is still an explicit external download,
after which inference and SSD expert reads require no hosted model service.

No hosted model API is required by the core workflow.

Deterministic controller actions do not add another data path. Eligible file bytes, diff
inspections, typed receipts, prompt blocks, and evaluator artifacts remain on the same local
controller, local model, session, and explicitly selected scratch/output roots. The intrinsic
behavior adds no telemetry, hosted inference fallback, remote persistence, or network authority.
Local inference events may retain versioned model-namespace, rendered-token-root, and tool-schema
digests plus an instruction version, bounded workflow stage/authority labels, token counts,
control-collar dialect/reason/count fields, and
local refill/persistence timings. Those values make cache reuse
auditable without copying prompt, task, path, tool-argument, or source content into diagnostic
fields. Bounded lookup details identify only cache lifecycle outcomes such as session divergence or
an exact-root miss. They remain in the same local session history and are not transmitted by this
behavior.
Durable actor fields contain only a local profile or workflow-steward identity and optional
assisting profile. Character presentation adds no network request and does not send repository
content to a separate service.

Durable Goal orchestration remains in this local core. Objectives, criteria, accepted and retired
plan versions, milestone evidence, budgets, model-requested changes, and user acceptance are stored
with the local session. The responsive Goal UI calls the same loopback daemon API as other session
controls. Goal mode adds no cloud scheduler, hosted inference fallback, analytics edge, or automatic
publication path.

High-level Task planning and the multi-Task controller follow the same local path. Planner
inference uses the selected local model; there is no hosted fallback or automatic model escalation.
Every compact planning attempt also remains local: its prompt, JSON schema, raw constrained output,
normalized artifact, typed failure, usage, and controller routing decision are stored in the
containing session Git note. This is diagnostic model I/O, not hidden chain-of-thought. The accepted
plan, qualification/identity digests, exact budgets and consumption, Task requests and results,
whole-request completion audit, repository fingerprints, and native active Build or Goal checkpoint
use the same local store. The Tasks UI reads that state through the loopback daemon. The feature adds
no hosted scheduler, telemetry, remote persistence, or publication edge.

## Explicit external edges

```text
model/update downloads ────────────────→ registries and GitHub releases
public research tool ──────────────────→ search/fetch service
remote MCP tool ───────────────────────→ configured provider
container service with egress ─────────→ external network
future publication approval ───────────→ source-control provider
```

Each edge exists for a distinct reason and should carry the minimum data it needs.

### Downloads

Model pulls and application updates are user-invoked network operations. Downloaded model and
runtime artifacts remain local for reuse after the operation.

Opening integration discovery queries the fixed public `crunchy-pb` GitHub organization;
configuring a selected package reads its container metadata from the registry. Installing an LSP
package resolves the selected tag to an immutable digest and pulls that exact image when a local
runtime is available. Registry metadata, redirects, and authentication realms must all be public
HTTPS targets: pb rejects local names, credentials, and any DNS set containing a private or
special-use address, pins each request to a validated answer, and bypasses ambient proxies. Those
requests disclose the
requested public package name and the ordinary network metadata of a download, but no repository
source or prompt content. Downloaded
images remain local reusable artifacts. A task never contacts the registry to fill in a missing
digest-pinned package; it reports the integration unavailable and asks for an explicit online
install or upgrade. At task runtime, packaged LSP networking is independently
controlled by its validated service capability; the rust-analyzer package uses an internal network
without egress and performs Cargo analysis offline.

Once configured, proactive diagnostics stay on the same local path: pb sends bounded current file
content to the session-owned LSP over stdio and records a bounded local report in the normal task
event stream. No prompt, source, or diagnostic is sent to the package registry or an external LSP
service. The read-only workspace mount, offline Cargo configuration, and no-egress service network
remain in force; pb does not apply server-proposed edits or commands.

Opt-in semantic mutation analysis uses that same local stdio sidecar path with controller-provided
base and candidate overlays. Each transaction copies only the exact captured in-scope regular files
to an ephemeral shadow directory, mounts that shadow read-only into a fresh isolated LSP session,
checks it for drift, and deletes it when the transaction ends. It may read repository content plus
offline dependency sources, declarations, sysroots, typeshed/stubs, and package metadata already
mounted into the session, but does not add egress and does not expose those inputs to the model.
Durable constraint telemetry and separate generation/final receipts are content-free: exact content
is represented only by hashes, versions, counts, and timings; paths, source, patch bodies, diagnostic
messages, prompts, and symbol indexes are not persisted. Any future language-specific serialized
symbol/type index must remain content-addressed, local, lockfile/configuration scoped, and proven
equivalent to that language's native resolver before it can change this data boundary. Mutable
external LSP cache attachments cannot currently authorize required semantic mode; provider-embedded
offline dependencies remain bound by the image digest.

### Public research

Planning, review, discussion, implementation, and repair may expose public research tools. A query
or fetched URL necessarily reaches the built-in search endpoint or fetched destination. The client
does not use ambient proxies: it validates every URL and redirect, rejects credentials and any DNS
set containing a private or special-use address, and pins the connection to one validated public
answer. Responses are byte-bounded. These SSRF controls do not make a public query private.

### MCP and services

A remote MCP server receives tool arguments and returns tool data. A container service receives
only the workspace, network, cache, and secret capabilities declared for it, but egress still means
the service can communicate outside the session. Capability declaration makes the edge auditable;
it does not make a third party private.

Individual MCP tools are exposed only when their raw names match the operator-audited
`capabilities.read_only_tools` list. Server annotations cannot authorize themselves, and the current
workflow exposes no external MCP mutation. “Read-only” describes the operator's intended effect;
it is not a confidentiality guarantee—arguments and reachable service data still cross that
integration's boundary.

Local host-command MCP servers stay on the machine but inherit the user's host permissions. Safari
browser automation is one example: pb exposes Safari Technology Preview's own MCP tools when the
server is configured and its desired raw tool names are explicitly classified read-only, rather
than owning a parallel WebDriver session. Page content, screenshots, console data, and other browser
diagnostics flow from Safari to the local agent process. They do not become a network disclosure
unless another enabled edge sends them elsewhere.

### Commands

A command can contain its own network client. pb's stage and policy checks govern whether the
command tool can run, while actual network containment depends on the selected environment. An
explicit local backend inherits host connectivity.

## Data classification by owner

| Class | Examples | Owner and expected handling |
| --- | --- | --- |
| User-global | Model preferences, project registry, OAuth token | Stored below the user's config/data roots; not checked into a project. |
| Repository-owned | `.pb/` configuration, source, acceptance facts | Visible to project collaborators if committed; secret values should not appear here. |
| Session-owned | Task workspace, container, services, network, event stream, active/completed Goal and multi-Task checkpoints | Reconciled or removed when terminal/expired, except persisted history. |
| Durable project memory | Evidence-backed Markdown entries under local `refs/pb/memory` | Outside the working tree and ordinary branch pushes; bounded and retained until that ref is changed or removed. |
| Reusable local artifact | Model weights, images, declared caches, llama.cpp and FlashMoe session state | May survive sessions; managed separately from task cleanup. |
| External disclosure | Search query, remote tool arguments, provider mutation | Occurs only through an enabled edge; governed by its provider and local policy. |

## Persistence is not publication

Git notes keep enough session history to reconnect and recover workflow state, but the pb notes
namespace is not included in an ordinary branch push. It remains repository-local unless someone
explicitly transfers that ref or copies the repository metadata.

Goal restart safety is local persistence behavior: interrupted active work restores paused and
requires an explicit Resume. Stopping a Goal preserves local commits, workspace content, and
evidence; deleting the containing finished session through the existing session-delete operation
removes its pb session note and Goal state under the same cleanup contract.

An interrupted active multi-Task controller is likewise restored paused with the exact active Task
request and counters. Its completed local commits are preserved; deleting the containing session
removes the Task plan and queue checkpoint with the same Git-note cleanup.

If verified container or network cleanup fails, pb keeps the local lease record in a failed state
and retries it later together with the session workspace. It does not erase the only inventory of
resources that may still need removal; a dirty or otherwise unremovable workspace keeps that
recovery record intact.

Likewise, a managed commit is local evidence. pb intentionally stops before remote publication. A
future publication flow must have its own approval, idempotency, provider, and audit contracts.

## Privacy choices that remain with the user

pb cannot know whether a repository fact is sensitive to your organization or whether a remote
integration is acceptable. The user or project owner chooses:

- which repositories pb may inspect;
- which model and cache locations persist;
- whether the server is reachable beyond loopback;
- which MCP/LSP services are enabled and what capabilities they receive;
- whether public research is allowed;
- whether execution uses a container or the local host;
- whether local commits are later published.

The practical controls and cleanup commands are in [Your data and privacy](../user/data-and-privacy.md).
