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
backend. Session events, workflow checkpoints, and durable Goal checkpoints are persisted into
repository-local Git notes.
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
`inference.llamacpp_session_cache_enabled=false` and
`inference.flashmoe_session_cache_enabled=false` independently disable disk persistence without
disabling in-process reuse. These controls, their byte budgets, and the optional cache root are
typed user configuration rather than process environment variables.

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
Durable actor fields contain only a local profile or workflow-steward identity and optional
assisting profile. Character presentation adds no network request and does not send repository
content to a separate service.

Durable Goal orchestration remains in this local core. Objectives, criteria, accepted and retired
plan versions, milestone evidence, budgets, model-requested changes, and user acceptance are stored
with the local session. The responsive Goal UI calls the same loopback daemon API as other session
controls. Goal mode adds no cloud scheduler, hosted inference fallback, analytics edge, or automatic
publication path.

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
package pulls its published image. Those requests disclose the
requested public package name and the ordinary network metadata of a download, but no repository
source or prompt content. Downloaded
images remain local reusable artifacts. At task runtime, packaged LSP networking is independently
controlled by its validated service capability; the rust-analyzer package uses an internal network
without egress and performs Cargo analysis offline.

Once configured, proactive diagnostics stay on the same local path: pb sends bounded current file
content to the session-owned LSP over stdio and records a bounded local report in the normal task
event stream. No prompt, source, or diagnostic is sent to the package registry or an external LSP
service. The read-only workspace mount, offline Cargo configuration, and no-egress service network
remain in force; pb does not apply server-proposed edits or commands.

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
| Session-owned | Task workspace, container, services, network, event stream, active/completed Goal checkpoints | Reconciled or removed when terminal/expired, except persisted history. |
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
