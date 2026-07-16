# Security model

pb's primary security concern is authority control: a probabilistic model, untrusted repository
content, or a remote tool response must not be able to grant itself a more powerful operation or
falsely advance delivery.

This model is defence in depth. It does not claim that every execution mode is an operating-system
sandbox.

## What pb treats as untrusted

- model completions, including confident claims about completion;
- repository text that may contain prompt-like instructions;
- command, check, LSP, browser, research, and MCP output;
- advisory-agent responses;
- stale evidence produced before the latest mutation;
- project configuration until it passes parsing, path, and authority validation.

These inputs can supply facts. Rust-owned gates decide whether the facts satisfy the current
contract.

## Shipped authority controls

### Stage-scoped capabilities

Every workflow stage has a fixed capability record and terminal action. The set of exposed tools is
a deterministic subset of that record and any direct request allowlist. Unknown tools and tools
outside the active stage are rejected even if the model asks for them by name.

Read-only stages do not receive mutation or shell tools. Implementation and repair receive edits
and commands but not commit authority. Checking and committing are deterministic harness stages.

### Structured, fingerprinted transitions

A model advances a model-driven stage only with the expected structured submission. Accepted plans,
reviews, implementation artifacts, checks, and commits are tied to current repository or evidence
fingerprints. Later mutations invalidate stale approvals.

### Delegation cannot amplify authority

Advisory work runs with bounded fresh context against an isolated workspace snapshot. Advisors are
read-only, cannot delegate again, and cannot advance the caller's workflow. Multiple advisors share
global call and token limits.

### Policy can narrow calls

`.pb/policy.toml` can allow, deny, or ask for an exposed call based on profile, tool, and arguments.
Rules are first-match. An unmatched call is allowed, so policy should not be mistaken for a default-
deny boundary. It is also subordinate to stage capabilities: it can narrow authority, never add it.

### Managed repository boundary

The workflow owns commit creation and checks the task workspace and expected repository state before
committing. Task contracts can restrict allowed paths and require specific mutations or checks.
Configured task outputs are promoted through controlled file operations; unexpected changes do not
gain check or commit credit.

### Bounded execution

Step, invocation, token, advisory, plan, and repair budgets cap model-driven work. The no-progress
guard fingerprints repeated failed operations against unchanged workspace/evidence state and can
block a call or terminate the run without spending another model turn.

## Environment isolation

Apple-container-backed sessions give pb ownership of the primary container, task workspace,
services, network, and cache attachments. Lease records support startup reconciliation and terminal
cleanup. Runtime and bootstrap phases are distinct, and service capabilities for workspace,
network, caches, and secrets default to none.

Container-backed MCP/LSP services must declare read-only or read-write workspace access and none,
session-only, or egress network access. Secret values are injected at launch and excluded from
project configuration, arguments, and the session ledger.

Images and approved caches may outlive a session. The primary container, services, networks, and
ephemeral workspaces are session-owned.

## Important limits

| Surface | Boundary |
| --- | --- |
| Local execution backend | Commands run on the host with the user's OS permissions. Tool gating and policy still apply, but this is not filesystem, process, or network containment. |
| Broad configured task or `run_command` | Shell authority is intentionally broad within its execution environment. Review tasks and add policy where needed. |
| Unmatched policy call | Allowed. Add explicit rules for operations that need deny or approval behavior. |
| Public research and remote MCP | Data can leave the machine when the tool is invoked. Stage gating is not a confidentiality proof. |
| Web listener | Loopback by default. A non-loopback listener is not automatically protected by authentication or TLS. |
| Container runtime | Isolation depends on the selected runtime and host configuration. Persistent images/caches remain outside the ephemeral resource lifecycle. |
| Publication | Local Ready evidence does not authorize a push, pull request, merge, or provider-side mutation. |

## Credentials

The GitHub OAuth flow uses PKCE and a loopback callback. Its token is stored below the user
configuration directory and created with mode `0600` on Unix. Project files can name host
environment variables for service secret injection; they should not contain the secret values.

pb does not turn a secret-bearing host command into a contained operation. Prefer capability-scoped
container integrations when repository, egress, and secret boundaries matter.

## Security contract in one sentence

The model may propose an operation only from the capabilities of the current stage; policy may
narrow that proposal; the environment determines where it executes; and only harness-validated,
current evidence can change workflow state.

The [session environment control-plane record](../session-environment-control-plane.md) describes
the implemented resource model and remaining hardening in detail.
