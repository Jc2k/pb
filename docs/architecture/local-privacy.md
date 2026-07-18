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
backend. Session events and workflow checkpoints are persisted into repository-local Git notes.
The web UI is embedded in the binary and served by the local Rust process.

Inference acceleration is local as well. llama.cpp keeps a live exact-prefix context during an
agent run and stores byte-budgeted restartable state under the platform cache root. FlashMoe keeps
loaded model/Metal runtimes in a bounded idle pool, holds safe prompt, generated-head, and shared
system-prefix checkpoints in memory, and writes compatible KV/MLA/recurrent checkpoints to its own
byte-budgeted cache. State files contain token ids and derived attention or recurrent state, so
they can reflect repository content that appeared in the prompt. Model, tokenizer, template,
runtime-layout, and token hashes prevent incompatible restoration; session filenames are hashed;
directories and files are owner-only on Unix; writes use temporary files and atomic replacement.
`PB_LLAMA_SESSION_CACHE=off` and `PB_FLASHMOE_SESSION_CACHE=off` independently disable disk
persistence without disabling in-process reuse.

No hosted model API is required by the core workflow.

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

### Public research

Planning, review, discussion, implementation, and repair may expose public research tools. A query
or fetched URL necessarily reaches the configured search/fetch service. Project policy can deny or
ask for those tools when that is inappropriate.

### MCP and services

A remote MCP server receives tool arguments and returns tool data. A container service receives
only the workspace, network, cache, and secret capabilities declared for it, but egress still means
the service can communicate outside the session. Capability declaration makes the edge auditable;
it does not make a third party private.

Local host-command MCP servers stay on the machine but inherit the user's host permissions. Safari
browser automation is one example: pb exposes Safari Technology Preview's own MCP tools when the
server is configured, rather than owning a parallel WebDriver session. Page content, screenshots,
console data, and other browser diagnostics flow from Safari to the local agent process. They do
not become a network disclosure unless another enabled edge sends them elsewhere.

### Commands

A command can contain its own network client. pb's stage and policy checks govern whether the
command tool can run, while actual network containment depends on the selected environment. An
explicit local backend inherits host connectivity.

## Data classification by owner

| Class | Examples | Owner and expected handling |
| --- | --- | --- |
| User-global | Model preferences, project registry, OAuth token | Stored below the user's config/data roots; not checked into a project. |
| Repository-owned | `.pb/` configuration, source, acceptance facts | Visible to project collaborators if committed; secret values should not appear here. |
| Session-owned | Task workspace, container, services, network, event stream | Reconciled or removed when terminal/expired, except persisted history. |
| Reusable local artifact | Model weights, images, declared caches, llama.cpp and FlashMoe session state | May survive sessions; managed separately from task cleanup. |
| External disclosure | Search query, remote tool arguments, provider mutation | Occurs only through an enabled edge; governed by its provider and local policy. |

## Persistence is not publication

Git notes keep enough session history to reconnect and recover workflow state, but the pb notes
namespace is not included in an ordinary branch push. It remains repository-local unless someone
explicitly transfers that ref or copies the repository metadata.

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
