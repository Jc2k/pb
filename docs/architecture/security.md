# Security model

pb's primary security concern is authority control: a probabilistic model, untrusted repository
content, or a remote tool response must not be able to grant itself a more powerful operation or
falsely advance delivery.

This model is defence in depth. It does not claim that every execution mode is an operating-system
sandbox.

## What pb treats as untrusted

- model completions, including confident claims about completion;
- repository text that may contain prompt-like instructions;
- command, check, LSP, research, and MCP output, including browser MCP results;
- advisory-agent responses;
- stale evidence produced before the latest mutation;
- project configuration until it passes parsing, path, and authority validation;
- downloaded model containers and metadata until their selected backend validates the exact
  bounded source contract.

These inputs can supply facts. Rust-owned gates decide whether the facts satisfy the current
contract.

## Shipped authority controls

### Stage-scoped capabilities

Every workflow stage has a fixed capability record and terminal action. The set of exposed tools is
a deterministic subset of that record and any direct request allowlist. Unknown tools and tools
outside the active stage are rejected even if the model asks for them by name.

Read-only stages do not receive mutation or shell tools. Implementation and repair receive edits
and commands but not commit authority. Checking and committing are deterministic harness stages.

The hidden prompt-cache qualification harness can remove a named tool from its ordinary direct-run
allowlist to prove that rendered authority invalidates an old cache root. It validates the name
against that fixed allowlist and carries the exclusion in a non-serializable harness-only request
field. Strict stage capabilities are derived first, after which the evaluator can only subtract;
removing a stage's typed terminal action is rejected. The experiment therefore cannot expose a tool
or operation that the stage did not already permit. Generated artifact success is recorded
separately and never substitutes for the exact-root cache gates.

### Structured, fingerprinted transitions

A model advances a model-driven stage only with the expected structured submission. Accepted plans,
reviews, implementation artifacts, checks, and commits are tied to current repository or evidence
fingerprints. Later mutations invalidate stale approvals.

### Task decomposition cannot amplify authority

Default high-level partitioning can produce Build Tasks only. Its constrained model artifact
contains Task titles and controller-issued source-clause IDs, so it cannot add requirements, paths,
acceptance claims, budgets, Goal authority, or publication authority. Rust requires exact source
coverage, creates dependencies and budgets, and discards one-Task or invalid output before running
the exact original Build. Repository configuration cannot change that authority boundary.
Automatic Goal-shaped Task selection remains a separate promotion bound to exact model bytes,
backend, prompt-template version, artifact protocol, and checked-in evidence.

The multi-Task controller carries the source repository, workflow and Goal policies, request cap,
no-publication boundary, and aggregate allowance as immutable authority. A child receives only its
own projection and current repository baseline. One child runs at a time, and the controller checks
the exact Git/content state before starting the next. Child counters are monotonic watermarks;
retry, resume, restart, or Goal milestones cannot reset or double-spend them. Task runtime deadlines
stop at the existing safe workflow boundary before the over-budget parent fails closed.

A user can use the existing Goal amendment flow inside a Goal Task, but the parent admits only a
bounded revision that preserves the accepted objective and criteria, stays in the same workdir,
keeps publication disabled, and fits the Task budget. Default Build partitioning may fail soft only
to the exact original Build request; it cannot substitute a broader summary or gain authority.

### Delegation cannot amplify authority

Advisory work runs with bounded fresh context against an isolated workspace snapshot. Advisors are
read-only, cannot delegate again, and cannot advance the caller's workflow. Multiple advisors share
global call and token limits.

### Policy can narrow calls

`.pb/policy.toml` can allow, deny, or ask for an exposed call based on profile, tool, and arguments.
Rules are first-match. An unmatched call is allowed, so policy should not be mistaken for a default-
deny boundary. It is also subordinate to stage capabilities: it can narrow authority, never add it.

Policy is compiled against the complete exposed tool-schema set before the session starts. Exact
selectors must name an exposed tool and valid argument paths. Wildcard selectors must match at
least one exposed tool, and every constrained argument must exist on at least one matching schema.
Regular expressions are compiled eagerly. A stale or misspelled rule therefore stops the session
instead of silently failing to protect the intended call.

### Managed repository boundary

The workflow owns commit creation and checks the task workspace and expected repository state before
committing. Task contracts can restrict allowed paths and require specific mutations or checks.
Existing files can be changed only after a read of the exact current bytes, recorded by content
fingerprint. Writes use synced temporary files and atomic no-clobber or replace operations, with a
final stale-content check immediately before replacement. Recursive deletion is not exposed.

FlashMoe mutation generation receives a bounded immutable snapshot containing only complete files
whose controller read fingerprints are still current. The `pb-control-collar` crate is pure and
unpublished: it cannot read the live workspace, follow a path, run Git or an analyzer process,
publish bytes, grant a tool, or choose a backend. Its token filtering narrows a call already exposed
by the controller. Missing snapshots, unsupported schemas, unavailable pinned parsers, incomplete
envelopes, invalid virtual results, and empty candidate frontiers fail closed. Durable rejection
telemetry contains only dialect, reason codes, counts, digests, terminal states, and snapshot byte/
file counts—never source, patch, argument, symbol, or path content.

Request-local prefix and patch checkpoints retain only the bounded in-memory state needed to compare
candidate branches. They are stream-scoped, discarded with the constraint session, grant no new
path or tool authority, and are never accepted as executor evidence.
The active work-unit target is carried in that immutable snapshot when a target-bound schema omits
`path`. The collar may inject only this controller-owned value for validation; a model value cannot
replace it or widen the accepted-plan ledger.

`pb-control-rust` is likewise unpublished and has no live-workspace or publication authority. The
controller gives it a verified immutable shadow plus content/configuration/dependency identities.
Rust Cargo metadata runs offline, build scripts are not loaded, and no procedural-macro server is
started in the safe profile. Missing cached dependencies fail readiness instead of fetching. The
project world must be loaded and primed before a Rust-edit-capable model invocation; decoding uses a
request-local Salsa snapshot and performs no filesystem, Cargo, network, process, or LSP activity.
An epoch lease prevents the cache from revising that database while a decoder can query it. A
concurrent request or unsupported invalidation receives another prepared world. Cold construction
runs in a request-independent in-process worker under the exact-world single-flight lease and a
global one-worker limit. Cancelling the initiating request stops its wait but does not grant the
detached worker new authority: it can only finish the already captured local shadow and publish the
exact result into the bounded process cache.

The executor does not trust generation-time acceptance. It repeats path/capability/schema/read gates,
reconstructs the virtual mutation, verifies current base hashes and syntax, and publishes through
the existing atomic file operations. When a native Rust world was prepared, executor entry first
revalidates the full live-world identity and replays every complete virtual Rust result through a
fresh analyzer stack. This includes known/generated provenance, deletion events, the last patch
hunk, and untouched base tails. Constrained patches additionally require exact in-memory hunk
application and `git apply --check` without recount. The broader recount-compatible Git path is
limited to the unconstrained llama.cpp compatibility backend and is not a fallback.

Strict implementation and repair additionally bind mutation to the active checkpointed work unit.
The model does not need to supply its target path; pb resolves it from the accepted-plan ledger and
records the resolved path in the tool event. Bounded multi-create batches validate every member
before execution and roll back earlier members after an execution failure. Diagnostic previews
cannot mint selected-check evidence and are rejected if they alter repository or Git control state.
Sibling vocabulary candidates run from one request-local parser, mutation, patch, and language-layer
checkpoint and are rolled back before the next candidate. An incomplete UTF-8 token fragment is not
treated as invisible: it remains eligible only when a bounded schema-aware scalar completion can
still satisfy the active protocol state. An empty constrained candidate frontier preserves only a
diagnostic partial transcript, executes no call, and can authorize at most one fresh retry of the
same mutation capability and target.
Optional contract `work_unit_guidance` is size-bounded advisory prompt text for an exact allowed
path; it grants no capability, scope, evidence, progress, or stage transition.
Intrinsic deterministic controller actions preserve the same boundary and cannot be configured or
overridden by stored/API request fields. Production uses one truthful controller user/context
block; there are no prompt-local synthetic assistant calls. Full controller reads may grant
read-before-write only for the exact revalidated fingerprint; diagnostic excerpts grant only
range-confined edits; review observations grant inspection coverage but no verdict authority.
Automatic deletion rejects directories, oversized files, dirty or untracked content, adopted work,
stale fingerprints, forbidden paths, non-required mutation contracts, and ambiguous targets;
eligible tracked deletions remain recoverable from Git. A tracked symlink is removed without
following its target.

Character presentation is not an authority boundary. Profile characters describe who requested a
model tool; Trinity describes deterministic controller and handoff work. The structured event type,
controller receipt, origin, coverage, fingerprints, and recovery metadata remain authoritative and
visible as secondary provenance. UI grouping never converts controller work into a synthetic model
call, and legacy events without model attribution are not assigned to a character by inference.

Configured tasks run in an isolated snapshot. pb rejects Git-control changes, undeclared paths,
unsafe symlinks, and boundedness violations before staging every output. Promotion is transactional:
if one destination fails, previously promoted paths are restored. Unexpected changes do not gain
check or commit credit, and `run_task` never substitutes for a named `run_check` receipt.

### Bounded execution

Step, invocation, token, advisory, plan, and repair budgets cap model-driven work. The no-progress
guard fingerprints repeated failed operations against unchanged workspace/evidence state and can
block a call or terminate the run without spending another model turn.

Host and managed commands drain stdout and stderr concurrently into one bounded result. They have
explicit timeouts, distinguish cancellation from timeout and nonzero exit, and terminate the owned
host process group or managed exec. Discovery, repository reads, integration configuration,
connector responses, task promotion, vision input, web bodies, and durable memory each have byte,
count, or time ceilings at their trust boundary.

### Dynamic tools do not grant themselves authority

An MCP server's annotations are treated as untrusted descriptive hints. A discovered tool is
exposed only when its raw server name matches the operator-audited
`capabilities.read_only_tools` declaration. pb currently has no workflow authority for external
MCP mutation, so undeclared or mutating MCP tools remain hidden in every stage. The server's
workspace/network/container capability declaration controls its environment but does not classify
individual tool effects.

MCP and LSP discovery reject unsupported schema keywords, oversized or unbounded discovery,
normalized-name collisions, and malformed responses. An MCP `isError` result is a failed tool call.
Only an operator-declared read-only tool that the server also marks idempotent can retry, and then
only after a typed transport failure. LSP restarts follow the same typed transport boundary;
protocol/application errors do not trigger blind replay.

Configured LSP diagnostics are also proactive controller inputs. pb selects only changed accepted
task paths whose inferred language matches a configured server; the model cannot expand that
selection. Mid-implementation collection admits only error-severity diagnostics identified as
syntax/parser failures, while settled collection admits error severity only. Passes have path,
call, time, result-count, and message-size ceilings. Every diagnostic records workspace, path, and
content identity. Every matching server/path target is retained as completed, advisory, failed, or
deferred. Only a full pull-diagnostic report completes a target; a push-only server's fresh
publication remains useful advisory evidence but cannot claim a clean result. The same 12-second
proactive deadline starts before workspace observation and bounds Git path discovery, path and
aggregate file bytes, server launch, stdin writes, initialization, the diagnostic request, a typed
restart, shutdown, and final workspace revalidation. A workspace change during collection discards
the whole result. Server
startup, transport, timeout, and response failures remain visible advisory failures and cannot
mint check evidence or bypass configured checks. A blocking current result can grant only
exact-path repair focus after older read/stage evidence is invalidated. pb never executes LSP
workspace edits, commands, formatting, or code actions.

An opt-in semantic mutation gate gives a language server authority to veto a generated boundary or
final publication, never authority to mutate. The controller supplies exact in-memory overlays and
creates a fresh exact bounded shadow tree and isolated provider session for each transaction, and
checks that tree before and after analysis. Shadow creation rejects symlinks and non-regular files,
copy-time live-workspace drift, excess path/byte counts, and provider mutations. Only an LSP process
with read-only workspace authority may substitute this controller-owned analysis root; MCP,
writable, and undeclared services cannot. Required mode trusts only a digest-pinned container
provider with a complete pull-diagnostic baseline/candidate pair; host commands, mutable tags,
push-only publications, restarts, stale versions, timeouts, deletion impact, empty/detached project
graphs, and unclassified diagnostics fail closed. Advisory mode cannot veto. Source, paths,
diagnostic messages, and symbol contents are excluded from constraint telemetry and receipts; only
provider/world hashes, classified outcome counts, guarantee rung, and bounded timing are durable.
Digest-pinned provider images bind embedded toolchains, sysroots, and declarations. A declared
external LSP cache is not yet an authoritative semantic input: until a profile binds its exact
content and mounts it read-only, its presence makes the semantic result unknown.

Marketplace LSP metadata is also untrusted input. Registry manifests, image configurations, and
authentication responses have byte ceilings; the typed LSP package annotation has its own 64 KiB
ceiling and rejects unknown fields, unsupported versions, invalid identifiers, and excessive list
counts. Package fields may supply bounded process arguments, language IDs, initialization options,
and cache requests, but cannot select a host command, environment, runtime, writable workspace, or
network access. A crunchy-pb marketplace LSP without this manifest cannot be installed. These
constraints prevent package metadata from granting its own authority; they do not turn arbitrary
language-server implementation code into trusted code, so the read-only, isolated sidecar remains
the execution boundary.

Registry metadata transport uses the same public-network boundary as built-in research, tightened
to HTTPS. Initial registry URLs, redirects, and bearer-token realms reject credentials, local
names, and any DNS set containing a private or special-use address. Each connection is pinned to a
validated address, ambient proxies are bypassed, redirects are revalidated and bounded, and a
bearer token is forwarded only within the origin for which it was obtained.

Marketplace metadata and executed code share one immutable OCI identity. Installation resolves the
tag, records the registry manifest digest plus the original display tag, re-reads the typed manifest
through that digest, and pulls that exact image when a runtime is available. Registry and repository
identity are canonicalized before marketplace policy is selected. Every digest-addressed child
manifest and image-config blob is hashed locally and must match the digest selected by its parent,
even when the registry omits its optional digest header. Task startup never
pulls a missing digest-pinned image. Tag-only legacy marketplace LSP entries are reported as
`legacy_unverified` and fail closed until explicitly upgraded or reinstalled.

Container-backed connector processes remain resources of the session environment. A retained
per-integration `container_runtime` value must match the owning session runtime or startup fails;
it cannot silently select a second cleanup domain. New UI installs omit this assertion and follow
the runtime that owns each task session.

### Network and media inputs are validated

Public research accepts only HTTP(S). Every initial URL and redirect rejects embedded credentials,
local names, and private or special-use addresses. pb resolves all answers, rejects the target if
any answer is non-public, pins the connection to a validated address, bypasses ambient proxies, and
fails on redirect or response-size ceilings. This controls pb's built-in research client; a shell
command or configured integration can still implement its own network behavior.

Vision accepts only a regular file inside the workspace or a path exactly registered as a session
attachment, with prompt, file-size, and decoded-pixel limits. Durable memory treats entries as
untrusted evidence, bounds repository and entry size, and lets an agent propose only facts, gotchas,
procedures, and debt. A model-supplied field cannot approve a decision or preference.

### Goal control cannot self-amplify

Durable Goal mode adds a Rust-owned controller above strict workflows, not a broader model stage.
`.pb/goal.toml` supplies only validated ceilings and cannot enable Auto, automatic continuation,
network/path/tool expansion, secrets, or publication. The controller snapshots its policy and a
local-repository/no-publication authority envelope into every hashed Goal checkpoint.

Discussion `propose_goal` is read-only. `start_goal` is exposed only to the internal Auto intent,
must cite the current harness turn, and can create only an awaiting-approval Goal. During a Goal,
the model can read bounded status or request pause, amendment, or budget review. Those calls pause
for controller/user handling; they cannot resume, cancel, accept, rewrite the Goal, apply a budget
increase, or grant publishing authority.

All mutating user/API operations use optimistic Goal-checkpoint digests. Initial and replacement
plan approval also names the exact plan digest. Child workflows retain the ordinary stage
capability matrix, managed commit boundary, fingerprints, and policy checks. Total invocation,
token, workflow, milestone, and wall-time limits bind continuation across child workflows.

### Task-plan policy cannot qualify itself

The shipped Task-plan foundation accepts model output only as a proposal. Rust validates its graph,
coverage, Build/Goal contracts, and controller-projected budgets before producing a digest-bound
artifact. Numeric budgets in model proposals are rejected rather than treated as limits.

`.pb/tasks.toml` contains effort presets and aggregate ceilings only. It cannot enable planning,
disable planning, qualify automatic Goal selection, change stage capabilities, or grant network,
credential, mutation, or publication authority. Automatic Goal selection requires a separate
controller-owned qualification record bound to exact model, template, protocol, and evidence
digests; explicit Goal intent still enters the existing approval-gated Goal controller.

The durable multi-Task reducer persists that qualification record with the accepted plan, policy,
repository control/content fingerprints, child checkpoint, and monotonic usage watermarks. It
rejects a second active child, backwards counters, replayed usage inflation, stale restart state,
unreconciled delivery commits, and automatic pending-Task expansion. A changed Build must preserve
its managed commit; a Goal Task must preserve its ordered child commits without a synthetic squash.
Cancellation retains completed history. The controller is connected to the normal user/session
dispatch path, but the embedded qualification catalog is empty in this release. Therefore no
installed model can currently create Goal-shaped Tasks automatically. Default constrained Build
partitioning is available across supported local models; explicit Goal mode remains unchanged.

### Bounded model-source resolution

FlashMoe does not let a model filename select arbitrary tensor behavior during inference. The
DeepSeek V4 source adapter accepts one pinned GGUF profile, bounds metadata and tensor-directory
parsing, validates every required tensor/type/shape and the full compression schedule, and
atomically publishes a source-independent cache. Load then binds the complete typed graph and Metal
kernel surface before inference. Unsupported containers or missing stages fail at that boundary;
the runtime does not probe tensors, invoke a reference executable, or fall back to CPU components.

## Environment isolation

Apple-container-backed sessions give pb ownership of the primary container, task workspace,
services, network, and cache attachments. Lease records support startup reconciliation and terminal
cleanup. Runtime and bootstrap phases are distinct, and service capabilities for workspace,
network, caches, and secrets default to none.

Projects using the local backend or no project environment may create a service-only lease for a
packaged LSP. It has the same ownership labels, durable resource ledger, isolated network,
read-only workspace policy, and terminal cleanup as a sidecar attached to a primary task container;
it does not fabricate a primary environment or grant command execution to the task.

Container-backed MCP/LSP services must declare read-only or read-write workspace access and none,
session-only, or egress network access. Secret values are injected at launch and excluded from
project configuration, arguments, and the session ledger.

Images and approved caches may outlive a session. The primary container, services, networks, and
ephemeral workspaces are session-owned. A cleanup failure changes the durable lease to `Failed` and
retains its resource inventory instead of deleting recovery state. The supervisor retries those
records under the per-session operation lock, includes the associated session workspace in the
retry, and removes a record only after verified container, network, cache, and workspace cleanup
succeeds. A dirty workspace is preserved and keeps the recovery record rather than being erased.

Local inference session caches use separate exact `llamacpp-session-v1` and
`flashmoe-session-v1` namespaces under the typed storage root. FlashMoe checkpoint and manifest
reads reject symlinks, non-directories, oversized records, malformed data, version or model
fingerprint changes, and incompatible state before graph mutation. Prompt-root memory retention is
byte-bounded LRU; dirty eviction attempts atomic owner-only disk publication and reports durable
failure rather than claiming reuse. Disk pruning refreshes recency only after a checkpoint or
manifest has passed validation; failure to update that best-effort hint cannot convert a valid hit
into an inference failure. Sessionless constrained generation may publish only the validated stable
system-prefix checkpoint; dynamic evidence, constraint state, and generated artifacts are not
included. `pb cache clean` is a dry run unless `--yes` is explicit and can remove only a selected
resolved versioned namespace; it never targets the storage root itself.

## Important limits

| Surface | Boundary |
| --- | --- |
| Local execution backend | Commands run on the host with the user's OS permissions. Tool gating and policy still apply, but this is not filesystem, process, or network containment. |
| Broad configured task or `run_command` | Shell authority is intentionally broad within its execution environment. Review tasks and add policy where needed. |
| Unmatched policy call | Allowed. Add explicit rules for operations that need deny or approval behavior. |
| Public research and remote MCP | Data can leave the machine when the tool is invoked. URL validation and read-only tool classification are not confidentiality proofs about the remote service. |
| Web listener | Loopback by default. A non-loopback listener is not automatically protected by authentication or TLS and is advertised through Bonjour for wake-on-HTTP. |
| Container runtime | Isolation depends on the selected runtime and host configuration. Persistent images/caches remain outside the ephemeral resource lifecycle. |
| Publication | Local Ready evidence does not authorize a push, pull request, merge, or provider-side mutation. |
| Goal automatic continuation | Explicit per-Goal user authority inside snapshotted totals. It does not approve new paths, integrations, network access, policy prompts, or publication. |
| Syntax-constrained mutation | Prevents a supported file from completing with parser-invalid syntax. It does not prove type correctness, name resolution, compilation, test success, runtime behavior, or security. Unsupported extensions retain ordinary executor checks. |
| Native Rust v1 constraint | Resolves project targets, imports, and selected callable/literal shapes against one exact pinned rust-analyzer world. Only qualified exact facts may veto. Disabled build scripts/procedural macros, generated APIs, partial dependency state, deeper inference, ownership, and runtime behavior remain unknown; this is not a compilation guarantee. |

## Credentials

The GitHub OAuth flow uses PKCE and a loopback callback. Its token is stored below the user
configuration directory and created with mode `0600` on Unix. Project files can name host
environment variables for service secret injection; they should not contain the secret values.
Parent-process environment access is confined to an audited boundary: operating-system directory
conventions and secret names explicitly declared by configuration. User-visible pb behavior does
not use standalone environment flags.

pb does not turn a secret-bearing host command into a contained operation. Prefer capability-scoped
container integrations when repository, egress, and secret boundaries matter.

The same host-command limit applies to Safari Technology Preview's MCP server. pb does not wrap it
in a second browser-control sandbox; Safari owns the browser session and access controls, while pb's
stage and policy rules govern which discovered MCP tools the model may call.

## Security contract in one sentence

The model may propose an operation only from the capabilities of the current stage; policy may
narrow that proposal; the environment determines where it executes; and only harness-validated,
current evidence can change workflow state.

The [session environment control-plane record](../session-environment-control-plane.md) describes
the implemented resource model and remaining hardening in detail.
