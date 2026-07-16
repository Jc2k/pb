# Apple container environment architecture

Status: active source of truth for pb environment isolation work.

## Purpose

pb treats development environments as a safety boundary, not as a convenient place to run shell
commands. Linux-capable agent commands, language servers, and container-backed MCP servers should
run in Apple containers by default. Host execution is an explicit capability for work that requires
macOS, Xcode, Apple SDKs, signing, simulators, Metal, host UI, or hardware access.

Apple's `container` CLI is the first supported backend. It uses the Containerization Swift package,
where each Linux container runs in its own lightweight virtual machine. pb may add a small Swift
runtime helper later when it needs framework capabilities that the CLI cannot expose reliably, but
the Rust orchestration model must not depend on Docker-only concepts such as committing a mutable
container.

This document defines the target architecture and records the migration boundary implemented by
each milestone. Code and tests must stay aligned with it.

The initial implementation targets Apple `container` 1.0.0 or newer. Its CLI surface was checked
against the installed 1.0.0 command reference and Apple's signed
[1.0.0 release](https://github.com/apple/container/releases/tag/1.0.0). Apple's
[technical overview](https://github.com/apple/container/blob/main/docs/technical-overview.md)
documents the per-container VM architecture, while the
[Containerization package](https://github.com/apple/containerization) is the lower-level Swift API.

## Safety contract

1. An absent or unresolved project environment never grants host command execution.
2. Host execution is selected only by explicit configuration or deterministic host requirements.
3. Runtime containers have no external network by default.
4. Networked dependency resolution is a distinct bootstrap phase.
5. Images are immutable build inputs. A running container is never committed as an environment.
6. Reusable caches have explicit trust scope and invalidation inputs.
7. Every ephemeral container and network has a pb owner, project, session, and role label.
8. A logical session owns its task, LSP, MCP, network, and ephemeral container lifecycles.
9. Cleanup is idempotent and recoverable after process failure; `Drop` is a last line of defence,
   not the resource ledger.
10. The user's original checkout is not the long-term workspace boundary.

The initial migration enforces items 1 through 7 at the runtime interface. Durable cross-turn
leases, task worktrees, service adoption, and crash reconciliation remain follow-on milestones and
are listed explicitly below.

## Control plane

The target control plane has five layers:

```text
project evidence
    -> environment resolver
    -> EnvironmentPlan + EnvironmentLock
    -> image/cache manager
    -> SessionLease
         |- task workspace
         |- agent execution container
         |- LSP services
         |- MCP services
         |- service containers
         `- session network
```

### Environment resolver

The resolver consumes trusted project evidence in this order:

1. `.pb/environment.toml`;
2. devcontainer, Dockerfile, Nix, mise/asdf, and tool-version configuration;
3. language manifests and lockfiles;
4. platform-specific CI jobs;
5. documented setup and guard commands.

It produces typed requirements with evidence, rather than asking the model to infer safety from a
prompt. The model may explain or propose a configuration, but the resolver validates it before the
environment gains authority.

Host requirements include:

- `macos`, `xcode`, `apple_sdk`, `simulator`, `keychain`, and `codesigning`;
- `metal`, host UI, USB/hardware access, and Virtualization.framework;
- Apple SDK targets, `xcodebuild`, `xcrun`, `simctl`, notarization, entitlements, and Xcode project
  or workspace files.

A repository can eventually assign different executors to different components. Until component
requirements are complete, any positive host requirement conservatively selects the local backend
for the whole project. The presence of a generic CI workflow is not a container requirement.

### Environment plan and lock

`.pb/environment.toml` is the human-owned plan. It declares:

- backend and base image or Dockerfile;
- setup, session, and guard commands;
- bootstrap and runtime network policy;
- CPU and memory bounds;
- persistent cache mounts and their invalidation inputs;
- human-readable discovery provenance.

A later `.pb/environment.lock` will record resolved image digests, platform, runtime compatibility,
cache fingerprints, and resolver version. Mutable tags such as `latest` may be accepted as plan
input, but a reusable build or cache must be keyed by the resolved digest.

### Runtime contract

The runtime interface is capability-oriented. It models:

- runtime identity and version validation;
- image pull, inspect, and deterministic image builds;
- explicit container launch specifications;
- internal networks;
- named cache volumes;
- process execution and cancellation;
- idempotent resource cleanup.

Apple CLI command construction is separate from Docker/Podman construction. Backend-specific
operations must not be added to the shared interface merely because Docker provides them.

Every launch receives:

- a unique pb-owned resource name;
- labels for `dev.pb.managed`, project, session, and role;
- an explicit workspace mount and working directory;
- resource bounds;
- a read-only root filesystem with writable tmpfs mounts for transient OS paths;
- an explicit network attachment or explicit egress policy;
- an entrypoint that is independent of the image's application entrypoint.

### Bootstrap and runtime phases

System packages and language runtimes belong in a Dockerfile/Containerfile image build. Project
dependency setup belongs in a bootstrap container and must write only to the workspace or declared
persistent cache mounts. This restriction replaces the old mutable-container commit behaviour.

The default phases are:

1. Pull or build the immutable toolchain image.
2. Start an egress-enabled bootstrap container when setup commands exist.
3. Run setup from `/workspace` using declared caches.
4. Remove the bootstrap container.
5. Start the session container on a pb-owned internal network.
6. Run per-session commands without external network access.

Projects that genuinely require network during agent execution opt into runtime egress explicitly.
An Apple internal network is host-only, not proof of a physically absent network device. Runtime
conformance tests therefore verify both external and host reachability before pb claims a stronger
property.

### Cache trust

Cache declarations have a stable logical ID, an absolute container target, and project-relative key
files. The physical volume name is derived from:

- project identity;
- resolved local image metadata and setup commands;
- cache logical ID and target;
- contents of declared key files.

Missing key files participate in the fingerprint, so creating or removing a lockfile invalidates
the cache. Cache volumes survive session cleanup. They are never shared across unrelated projects.

Download caches can later use a wider content-addressed scope after checksum verification. Compiler
outputs, virtual environments, package install trees, and other executable artifacts remain
project/environment scoped. LSP indexes use a separate server-version-scoped cache.

### Workspace boundary

The current migration continues to bind the selected workspace so host editing and Git workflows
remain coherent. The next workspace milestone replaces the original checkout with a task-owned Git
worktree and overlays high-I/O paths such as `target`, `node_modules`, `.venv`, compiler caches, and
LSP indexes with named volumes.

The stronger follow-on mode copies source into an ext4 workspace volume and promotes only an
inspected diff. That mode requires repository reads, writes, diffs, review, LSP file events, and Git
handoff to use a common `WorkspaceFs` abstraction. It must not be introduced as a command-only
special case.

### Session and service lifecycle

The target `SessionLease` is owned by the daemon and survives individual model turns. It records:

- environment and project fingerprints;
- container, network, volume attachment, and service identifiers;
- creation, last-use, cancellation, and expiry timestamps;
- desired and observed resource state.

Agent continuations reuse the lease. Images and cache volumes may outlive it; task, LSP, MCP, and
other service containers do not. Shutdown order is protocol shutdown, graceful process stop,
forced stop after a deadline, container deletion, ephemeral volume detach, and network deletion.

At daemon startup, pb lists resources with `dev.pb.managed=true`, reconciles them against persisted
session state, adopts valid active leases, and reaps expired or orphaned resources. Resource names
and labels introduced in the initial migration are the compatibility boundary for this ledger.

LSP services receive the same canonical `/workspace` view as commands. They start lazily, implement
the full document lifecycle, respond to server requests, expose stderr and health, and use explicit
index caches. Initially an LSP may run through `exec` in the task container to guarantee toolchain
parity; sidecars are appropriate once shared workspace and cache semantics are reliable.

MCP services declare workspace access (`none`, `read_only`, or `read_write`), network access,
secrets, cache mounts, and ports. Workspace access defaults to none. Secret references are resolved
at launch and are not stored as project TOML values.

## Configuration version two

The migration extends the existing compatible TOML rather than introducing a second environment
file. Omitted new fields receive safe defaults.

```toml
version = 2
mode = "pull"
backend = "apple_containers"
image = "docker.io/library/rust:1.88"
bootstrap_network = "egress"
runtime_network = "isolated"
setup_commands = ["cargo fetch --locked"]
session_commands = []
guard_commands = ["cargo test --all-targets"]

[resources]
cpus = 4
memory_mb = 4096

[[caches]]
id = "cargo-registry"
target = "/usr/local/cargo/registry"
key_files = ["Cargo.lock"]
```

An optional `[env]` table carries non-secret runtime variables such as `DENO_DIR` or a virtual
environment `PATH`. Secret values are never placed there.

Version-one files continue to deserialize. Saving writes the current version. Unknown fields remain
errors. Local environments retain the same command lists but ignore container-only resource,
network, and cache fields.

## Initial migration acceptance criteria

The first implementation backed by this document is complete when:

- Apple pull uses `container image pull`, and no runtime exposes container commit;
- Apple version 1.0 or newer is validated before use;
- Apple and OCI launch commands are covered by exact argument tests;
- launches have unique names, ownership labels, `/workspace` working directory, and resource bounds;
- runtime roots are read-only with explicit tmpfs and workspace/cache write locations;
- isolated runtime networks are created and removed with the container handle;
- setup uses a separate bootstrap launch, runs from `/workspace`, and is never committed;
- named cache volumes are content-keyed and retained across container cleanup;
- language discovery records dependency inputs and emits known runtime caches and environment paths;
- no environment means no implicit local command backend;
- project checks do not silently manufacture a local backend;
- Apple/Xcode signals take precedence over generic container or CI signals;
- LSP and MCP integration installation defaults to the detected preferred runtime rather than
  hard-coded Docker;
- configuration, discovery, runtime construction, and cleanup behaviour have unit coverage;
- an ignored Apple conformance test exercises pull/run/exec/cleanup when the Apple service and a
  small test image are available.

## Follow-on milestones

1. Task-owned Git worktrees and cache overlays; never mount the original checkout.
2. Durable daemon `SessionLease` persistence, cross-turn reuse, startup reconciliation, and TTL GC.
3. Environment lockfile with registry-resolved image digests and cache provenance.
4. LSP supervisor with file-event bridging, cache mounts, health, restart, and graceful shutdown.
5. MCP capability declarations, secret references, scoped egress, and session-owned services.
6. Container-local ext4 workspace with inspected diff promotion.
7. Optional Swift Containerization helper for streaming process I/O, vsock services, and framework
   lifecycle primitives that cannot be made reliable through CLI subprocesses.

Each milestone must add daemon-free conformance coverage and failure-injection tests before the
safety contract is strengthened in user-facing documentation.
