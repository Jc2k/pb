# Session environment control plane

Status: implemented production baseline and active source of truth (2026-07-16).

This document is the source of truth for the second production environment milestone. It extends
the runtime foundation in [apple-container-environment-architecture.md](apple-container-environment-architecture.md)
into a daemon-owned control plane. A milestone is complete only when its code, persisted schema,
failure behaviour, and conformance tests satisfy this document.

## Outcome

For Linux-capable work, one logical pb session owns an isolated development environment across
model turns. The environment includes a task-owned workspace, a primary command container,
language servers, MCP services, a private network, and cache attachments. Images and approved
caches may outlive the session. Workspaces, containers, processes, networks, secret material, and
ephemeral mounts may not.

Host execution remains an explicit capability for Xcode, Apple SDKs, simulators, signing, Metal,
host UI, hardware access, and other work that cannot run correctly in a Linux VM. An LLM may
propose requirements, but it cannot grant host command authority.

## Non-negotiable properties

1. The user's original checkout is never mounted into an agent container.
2. An image reference is inspected and locked before it is reused. A changed local image identity
   invalidates its lock and executable caches.
3. The daemon, not an individual agent turn, owns session resources.
4. Every runtime resource is discoverable by deterministic name and `dev.pb.*` ownership labels;
   worktrees and attachments have atomic ownership records. Adoption requires the complete managed,
   project, and session label set; name-only inventory never grants ownership.
5. Desired state is persisted outside the repository before a resource transition is acknowledged.
6. Startup reconciliation adopts valid resources and removes expired or orphaned resources.
7. Cleanup is idempotent. `Drop` is only a best-effort backstop.
8. Long-lived protocols use supervised streaming processes with bounded shutdown and stderr.
9. Command LSPs share the primary environment; sidecar LSPs see the same task-owned files and only
   explicitly declared caches/network access.
10. MCP workspace, network, cache, and secret capabilities default to none; service port
    publishing is unsupported.
11. Bootstrap egress is temporally separate from agent execution and runs only declared commands.
12. Container-local workspaces are not enabled until diff promotion is lossless and crash-safe.

## Control plane

```text
project evidence
  -> EnvironmentResolver
  -> ResolvedEnvironment + EnvironmentLock
  -> WorkspaceManager
  -> EnvironmentSupervisor
       -> SessionLease ledger
       -> CacheManager
       -> primary task container
       -> ManagedProcess registry (commands and LSP)
       -> ServiceLease registry (MCP and other services)
       -> session networks
```

The web daemon owns `EnvironmentSupervisor`. Daemon-backed runs acquire a lease before entering
the blocking agent loop and reuse it on continuation. Direct CLI and harness runs use the same
supervisor interface but request terminal cleanup instead of retaining the lease after the turn.

Conversation state remains in Git notes. Runtime desired/observed state is local machine state and
is stored below pb's state directory. Repository history must never contain container identifiers,
host cache paths, secret references resolved to values, or lease heartbeats.

## Environment resolution and authority

### Evidence

Environment evidence is structured data:

```text
EnvironmentEvidence {
  source_path,
  component,
  requirement,
  detail,
}
```

Requirements include language/toolchain constraints, dependency inputs, setup and guard commands,
container signals, and host capabilities. Deterministic inspection covers at least:

- devcontainer, Dockerfile/Containerfile, Nix, mise/asdf, `.tool-versions`, and language toolchain
  files;
- Rust, Python, Node, Deno, Go, and Swift manifests and lockfiles;
- CI platform matrices and documented setup/guard commands;
- Xcode projects/workspaces, Apple-platform Swift packages, Apple framework imports, SDK commands,
  simulators, signing, entitlements, Metal, and macOS CI. Source inspection covers Swift imports
  (including declaration-qualified imports) and Objective-C/C/C++ framework imports, including
  Virtualization, Containerization, Security, UI, media, hardware, and platform service frameworks.

Resolution is component-aware. A repository containing a Linux service and an Xcode client may use
a container executor for the former and an explicit local executor for the latter. When component
boundaries are ambiguous, positive host evidence conservatively applies to the repository.
Automatic `pb init` writes the conservative repository plan to `.pb/environment.toml` and missing
top-level component suggestions to `.pb/environments/<component>.toml`. A run focused beneath that
component loads its scoped plan before authority validation; repository-root work uses the
repository plan, and an explicitly supplied run configuration keeps precedence. Scoped plans are
never allowed to override positive host evidence, and initialization never overwrites an existing
scoped plan.

Every primary, workflow-stage, monitor, and sub-agent model invocation receives a runtime-owned
summary of the selected component, resolved backend, host capabilities, dependency inputs,
toolchain constraints, resolved image/mode/network phases, declared cache ids, bootstrap commands,
and validation commands. The summary is capped at 24 evidence items and 4,096 characters, is
rebuilt from inspection rather than accepted from an API/persisted request, omits environment
values and cache locations, and explicitly carries no host, network, cache, secret, or service
authority. The LLM may propose missing configuration; the resolver validates it and the model
cannot silently select the local backend.

`pb init` writes the typed, component-scoped evidence atomically to
`.pb/environment.evidence.json`. Agent startup also re-inspects the selected repository so stale or
missing generated evidence cannot grant container authority. Positive host evidence forces the
effective backend to local and emits an audit/correction event; a repository-root focus applies
all positive host evidence conservatively.

### Lock

`.pb/environment.lock` is generated, deterministic, safe to commit, and contains no host resource
IDs. It records:

- lock and resolver versions;
- canonical hash of `.pb/environment.toml`;
- backend, platform, configured image, and local image identity;
- Dockerfile/context input hash for built images;
- runtime binary and exact validated version;
- dependency key hashes and cache-plan hash.

For a pull environment, pb inspects the local image immediately before launch. Reuse requires the
identity to match the lock. A missing or changed identity is resolved deliberately and rewrites the
lock; it never silently reuses executable caches from the previous identity. Registry digest
resolution may strengthen the identity when the backend exposes it, but local identity comparison
is mandatory on every acquisition. Pull/build/inspect/container-create is serialized by a
kernel-backed lock keyed by runtime and configured image reference, so two sessions cannot mutate a
shared tag between inspection and launch. Candidate resolution is non-persisting; the project lock
is replaced only after preparation succeeds. Attached LSP/MCP launch keeps the same image lock until
the named container is observable with the expected project/session labels; spawning the runtime CLI
process alone is not treated as successful creation.

## Workspace strategy

The default strategy is `worktree_bind`:

1. Resolve the repository root and task branch.
2. Create or adopt a Git worktree in pb's state directory, keyed by project and session.
3. Bind that worktree to `/workspace` in the primary container.
4. Overlay declared high-I/O paths with named volumes.
5. Run host-side file tools and Git operations against the worktree, never the original checkout.

Paused, completed, and failed daemon sessions retain their worktree for the continuation TTL.
Expiry removes a clean worktree after runtime cleanup; the branch remains. Cancellation preserves
uncommitted content unless the user explicitly deletes the session. Adoption validates repository
identity, branch, and path before reuse. Worktree intent is recorded before `git worktree add`, and
an interrupted creation is replayed under a cross-process session lock.

`container_volume` is an ineligible experimental strategy. The benchmark implementation exercises
the same deterministic small-file workload through a task bind and an Apple named volume, checks
content equality, records runtime/image identity, and verifies cleanup. A future volume workspace
must copy source into ext4 and promote an inspected patch. It is eligible only after transactional
copy-in/incremental sync/copy-out exists and rename, deletion, binary, symlink, permission,
cancellation, conflict, and crash tests pass.

### Filesystem decision record — 2026-07-16

Apple `container` 1.0.0 with `alpine:3.20`, two iterations, 2,000 generated Rust-like small files:

| strategy | samples | checksum |
| --- | --- | --- |
| `worktree_bind` | 978 ms, 1128 ms | `1843554495 69780` |
| `container_volume` | 286 ms, 265 ms | `1843554495 69780` |

The named volume was about 3.8x faster in this deliberately metadata-heavy sample. The selected
strategy remains `worktree_bind`: pb does not yet have the transactional synchronization protocol
needed to make a container-volume workspace observable, reviewable, conflict-safe, and recoverable.
`pb env benchmark` records a machine-specific JSON report in
`.pb/benchmarks/workspace-filesystem.json`; the ignored real-runtime test verifies both strategies
and zero leaked benchmark containers/networks.

## Session lease

### Persisted state

Each lease has an atomic JSON record in the pb state directory:

```text
SessionLeaseRecord {
  schema_version,
  lease_id,
  session_id,
  project_id,
  environment_lock_hash,
  workspace,
  desired_state,
  observed_state,
  resources[],
  created_at_ms,
  last_used_at_ms,
  expires_at_ms,
}
```

Resource records contain kind, deterministic name, role, and persistence class. Runtime identity
and desired/observed state are lease-level fields. Records never contain secret values.

### State machine

```text
new -> preparing -> ready -> in_use -> idle
                  |          |         |
                  v          v         v
                failed <- stopping -> stopped

persisted active state + missing process -> reconciling -> ready | failed | stopped
```

Only the supervisor mutates state. A transition writes desired state before external mutation and
observed state after verification. Replaying any transition is safe.

### Lifetime

- New turn: acquire or create the lease and increment its in-process use count.
- End of turn: mark idle; do not stop resources.
- Continuation: validate fingerprints and reuse the lease.
- Pause/question: retain until the 30-minute baseline idle TTL.
- Complete: mark idle so the UI can continue the session; TTL expiry gracefully stops services and
  the primary container, detaches cache leases, deletes networks, and removes a clean worktree.
- Cancel: signal processes, preserve workspace content, then clean runtime resources.
- Delete: clean runtime resources and remove the worktree after explicit confirmation semantics.
- Daemon startup: list pb-labelled resources, compare with ledgers, adopt valid active leases, and
  reap expired/orphaned resources.

Shutdown order is protocol shutdown, TERM/graceful stop, deadline, KILL, container deletion,
ephemeral attachment cleanup, then network deletion.

## Managed processes

`ManagedProcess` is the common primitive for container exec and attached service clients. It owns
piped stdin/stdout/stderr, process identity, exit status, and bounded shutdown. LSP and MCP clients
continuously drain stderr into a bounded 64 KiB diagnostic tail. LSP Content-Length frames and MCP
newline-delimited messages are parsed by their respective bounded background readers, so a silent
or flooding protocol server cannot defeat the request deadline by blocking the agent thread. The
primitive supports:

- streaming reads and writes;
- graceful protocol shutdown;
- signal/stop with deadline and forced termination;
- health and exit inspection;
- idempotent close.

Apple uses `container exec -i <container> ...`; OCI runtimes use the equivalent backend command.
One-shot agent checks may use captured execution, but LSP/MCP processes may not.

## Cache manager

Caches have trust classes:

- `download`: immutable/checksummed archives eligible for content-addressed sharing;
- `toolchain`: runtime-owned compiler/package-manager data;
- `project_executable`: dependency installs and build outputs scoped to project and environment;
- `lsp_index`: base declaration scoped to project/environment; image sidecars derive a physical
  volume scoped to inspected image identity, complete server configuration/initialization options,
  and service role.

Metadata records cache identity, volume, trust class, provenance hash, size estimate, last use,
active attachment count, and preparing owner. Preparation uses an exclusive lock. Attach/detach is
leased and crash-reconciled. GC enforces age and size budgets and never deletes an attached or
preparing cache. Discovery never mounts an empty cache volume over an image-owned language
toolchain directory: Apple documents persistence and sharing for named mounts but does not promise
Docker-style population from the underlying image. Toolchains baked into the image therefore remain
part of the locked image identity; declared download, dependency, build-output, and LSP caches are
mounted only at paths they own. An existing runtime volume is reusable only when its complete pb
project, role, trust, and provenance labels match the durable cache record; a matching name alone
is never ownership evidence.

## LSP supervision

Command-configured LSPs run as managed exec processes in the primary task container for exact
toolchain/cache parity. Image-configured LSPs run as named, labelled session sidecars with the
task-owned worktree mounted at the same absolute path used in `file://` URIs. Sidecar workspace,
network, and cache access are explicit configuration. A configured service working directory must
exist beneath the task worktree after canonical path resolution; absolute, `..`, and symlink
escapes are rejected. Canonical paths are converted with standards-compliant file-URL encoding, so
macOS state paths such as `Application Support` and project filenames containing URI delimiters are
represented correctly.

The daemon session registry owns LSP clients and tool registries across turns. The client implements
initialize/initialized, didOpen/didChange/didClose, configuration/workspace server requests,
diagnostics, shutdown/exit, stderr capture, health polling, and one bounded restart per failed tool
call. Index caches are declared explicitly and resolved only from caches already attached to the
environment lease. Sidecar index-cache volumes include the inspected server image and serialized
server configuration in their provenance, while command LSPs use the primary environment's locked
cache identity.

Marketplace sidecars add a typed OCI manifest before this launch boundary. pb bounds and validates
the manifest, admits only read-only workspace and no-network packages, and copies only the package's
bounded arguments, language IDs, initialization options, and cache IDs into the ordinary
`LspServerConfig`. The manifest cannot choose its runtime; absent a retained compatibility
assertion, the owning session runtime launches and cleans up the service.

## MCP service capabilities

Container MCP configuration declares:

```toml
[mcp.servers.example.capabilities]
workspace = "none" # none | read_only | read_write
network = "none"   # none | session | egress
cache_ids = []
secret_env = { TOKEN = "EXAMPLE_TOKEN" }
```

Container MCP is required for container-backed sessions; host-command MCP configuration is refused
there because its capabilities cannot be isolated. STDIO services are named, labelled container
processes and never use `run --rm`. Workspace mounting follows the declared access exactly. The
explicit local-host fallback resolves its MCP working directory against the canonical project
workspace and rejects absolute, parent, or symlink escapes. The
client negotiates MCP through the current `2025-11-25` revision (while accepting the three prior
compatible revisions), uses the specification's bounded newline-delimited stdio transport, and
enforces real request deadlines. Remote HTTP MCP fails closed until authenticated Streamable HTTP
session supervision is implemented; package production servers as capability-declared containers.
Secret environment references resolve from named host variables at launch; values are passed
through the runtime client's environment, never argv or the lease ledger. Static non-secret
environment values may remain in `env`. See the official MCP
[versioning](https://modelcontextprotocol.io/docs/learn/versioning) and
[stdio transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
contracts.

## Network policy

- Bootstrap: default egress, declared setup commands only, no model tools and no secrets.
- Primary runtime: pb-owned internal network; pb grants no egress or host forwarding.
- LSP: command LSPs inherit primary policy; image LSP network access is declared and defaults to
  none/isolated.
- MCP: declared per service; default none/isolated and workspace none.
- Local host executor: explicit configuration and policy audit event.

Apple documents host-service access as an administrator-created localhost DNS/packet-forwarding
capability; pb never creates that capability. Conformance probes both public egress and the standard
`host.container.internal` alias. Apple isolated networks can still expose an internal default
route, so route-table shape is not used as proof of egress. A stronger claim is not made merely
because a network is named `internal`. See Apple's
[container networking guide](https://github.com/apple/container/blob/main/docs/how-to.md#access-a-host-service-from-a-container).

## Failure and reconciliation matrix

Deterministic tests cover pre-create worktree intent replay, dirty-worktree preservation, original
checkout exclusion, cache preparation exclusion and active-GC denial, lease
reuse/expiry/ordered cleanup, image and dependency lock invalidation, managed streaming processes,
inventory filtering, service working-directory containment, and service argument capability/secret
isolation. Runtime, workspace, and cache mutations use kernel-backed cross-process file locks on
Unix, so process death releases ownership without a guessed stale timeout. Retry and daemon
reconciliation converge to one valid lease or no lease.

The real Apple conformance gate currently covers:

- image inspect/identity comparison;
- isolated workspace bind behavior (the task-worktree/original-checkout selection is covered by
  deterministic integration tests before runtime launch);
- read-only root plus tmpfs/workspace writes, with named-volume writes covered by the benchmark;
- public and standard host-alias reachability denial on the isolated network;
- streaming exec, named attached-service stdio/removal, and bounded container cancellation;
- zero ephemeral pb resources after terminal cleanup.

LSP turn reuse and MCP capability behavior are deterministic unit/integration gates; adding
fixture images for an end-to-end real-runtime protocol matrix remains hardening work, not a reason
to weaken the enforced ownership/capability model.

## Session requirements audit — 2026-07-16

| Session requirement | Enforced production baseline |
| --- | --- |
| Capture how to build a development environment | Deterministic component evidence, human-owned plans, immutable image preparation, setup/guard commands, and an identity lock are generated and supplied to every model invocation. |
| Know when containers are invalid | Apple-only projects, sources, SDK commands, signing, simulation, Metal, UI, and hardware evidence force an audited local executor; mixed repositories use scoped plans and repository-root work is conservative. |
| Reuse images but not session containers | Inspected image identities and provenance-labelled caches persist; primary, bootstrap, LSP, MCP, network, process, and workspace resources are session-owned, reconciled, and terminally cleaned. |
| Default to no agent network while still bootstrapping | Only the declared bootstrap phase receives egress by default. Agent commands and service containers use isolated networks unless explicit service capability policy grants more. |
| Choose bind mount versus container-local copy | The measured named-volume path remains ineligible despite its speed advantage because transactional source synchronization and lossless diff promotion do not yet exist; a task-owned worktree bind is the correctness boundary. |
| Give LSP and MCP services code plus reusable caches | Command LSPs share the primary container; image LSPs and MCP servers are named session sidecars with explicit worktree/capability mounts and provenance-keyed persistent cache attachments. |
| Survive concurrency, crashes, and stale state safely | Kernel-backed session/image/cache locks, desired/observed ledgers, complete ownership-label validation, startup reconciliation, bounded process shutdown, and fail-safe cleanup prevent name-only adoption or deletion. |

The production baseline is complete only while the deterministic suite, the real Apple lifecycle
gate, the bind-versus-volume correctness benchmark, and the post-run zero-leak inventory all pass.

## Delivery milestones

| Milestone | Complete when |
| --- | --- |
| Resolver and lock | implemented: typed evidence and deterministic lock drive launch/cache identity |
| Worktree boundary | implemented: original checkout is never mounted and recovery preserves task content |
| Runtime process API | implemented: streaming, inventory, stop, and Apple/OCI argument gates |
| Session supervisor | implemented: leases persist, reuse, reconcile, expire, cancel, and clean idempotently |
| Cache manager | implemented: trust, preparation locks, attachments, recorded-estimate quotas, reconciliation, and GC |
| LSP | implemented: session registry, managed exec/sidecar, document lifecycle, restart, declared caches |
| MCP | implemented: named lease-owned services and deny-by-default capability schema |
| Filesystem decision | implemented: real benchmark selects bind on the correctness gate |
| Production gate | passed: full Rust suite and real Apple lifecycle/benchmark conformance |

The goal is not complete while any row is represented only by this document.
