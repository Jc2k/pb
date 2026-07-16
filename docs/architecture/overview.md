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
render the same typed events and can reconnect to the same persisted session.

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
describe workspace topology, executors, generated outputs, and affected checks. Delivery happens in
a task-owned workspace so the harness can compare the accepted baseline with the delivered change
and promote only declared outputs.

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

## Deliberate seams

Several boundaries are intentionally visible rather than folded into one “agent” abstraction:

- [conversation versus delivery](workflows.md);
- [stage capability versus user policy](security.md);
- [local computation versus external data paths](local-privacy.md);
- [model completion versus verified acceptance](user-contracts.md);
- local Ready evidence versus provider-owned publication.

Those seams make it possible to change models, UI surfaces, execution backends, and future
publication providers without moving the source of authority into prompt text.
