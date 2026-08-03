# Contracts with the user

pb's job is not merely to generate plausible code. It must preserve the user's intent and report
truthfully what happened. The architecture therefore treats several user-facing promises as
machine-enforced contracts.

## Contract hierarchy

```text
explicit user intent and answers
              ↓
trusted project configuration and optional acceptance contract
              ↓
harness-owned workflow and stage capabilities
              ↓
model proposals and tool results
```

Lower layers can supply evidence to higher layers. They cannot rewrite them. A model cannot infer a
missing approval, a repository file cannot grant a forbidden stage capability, and a tool result
cannot redefine the user's acceptance criteria.

## Intent contract

Discussion is read-only. Delivery starts only from explicit Build intent in the web surface or a
delivery-oriented CLI entry point such as `pb queue`. A discussion can propose work but cannot
silently promote itself.

Goal creation is a separate explicit lifecycle action, not a third serialized `TurnIntent`. A
discussion model may propose a Goal. Only the user's Goal action, the Goal API/CLI, or an explicit
Auto turn citing its exact current turn can create one, and creation still stops for exact plan
approval. Project configuration cannot activate a Goal or choose automatic continuation.

The shipped Task controller applies the same authority rule before high-level decomposition.
`.pb/tasks.toml` can narrow budgets but cannot grant authority or authorize automatic Goal
selection. Default Build partitioning uses constrained plain JSON in both supported inference
engines. Its complete model-facing contract is an ordered `tasks` array of request strings. Each
string is the outcome text later given to one normal Build workflow. For a multi-Task result, every
controller source clause must be preserved in exactly one string. Dotted paths and comma-delimited
code/list syntax remain inside their sentence clause; ownership matching ignores only whitespace
adjacent to punctuation. Rust retains the original
objective, recovers exact single ownership, derives UI titles, creates IDs and sequential
dependencies, assigns Build budgets, and keeps tests, documentation, review, and commits inside the
normal Build workflow. Rust also enforces explicit `before`, `after`, and `then` order. Constrained
generation stops at the first complete JSON value; there is no additional model critic in default
routing. Schema validity does not replace the deterministic ownership and ordering checks.

A one-Task proposal is discarded, not projected: the exact original request enters the ordinary
Build workflow. Invalid output gets one revision and then follows that same fail-soft path. Only two
or more accepted Tasks create a durable queue and Tasks UI. Explicit
Goal intent still uses the existing approval-gated Goal lifecycle. Automatic Goal-shaped Tasks need
a separate exact embedded qualification and remain fail-closed.

The accepted high-level plan is not mutation authority. It can only narrow the original objective,
repository, workflow and Goal policies, request cap, aggregate allowance, and no-publication
boundary. Each queued Task becomes a fresh Build or Goal request only when its dependencies are
delivered and the repository matches the prior terminal checkpoint. Task-planning failure cannot
reinterpret the broad request: fallback runs the exact original Build task, and the full constrained
planning transcript plus controller decision remains available in session details.

The ordinary Build planner likewise gets schema constraints only for facts the decoder can prove
locally: a non-empty plan shape and non-empty requirement references. Repository-aware structure,
including requirement coverage and ordered path existence, remains a deterministic Rust acceptance
check. Constraint success is never reported as plan acceptance, and a rejected plan grants no
mutation authority.

When planning discovers a materially missing choice, it may ask the user. The answer becomes part of
the task contract. Guessing would make progress faster at the cost of changing ownership of the
decision, so the workflow pauses instead.

While a session is running, the user may also send an ordinary in-flight message. This is
conversation input to the current primary agent, not a new turn, Task, intent selection, Goal,
approval, or automatic state transition. pb records the message before delivery and injects it at
the next agent loop boundary. The text may clarify or narrow the requested approach, but it cannot
widen repository scope, stage capabilities, an accepted plan, Goal limits, or publication
authority. Changes that require a new approval or controller transition must still use that
explicit lifecycle.

Review agents never consume these messages. If feedback arrives while a plan or code critic is
running, the controller discards any now-stale review submission and routes the pending message to
the planner-side authoring path. Feedback that arrives after implementation but before commit
invalidates downstream check and review evidence and restarts planning against the current bytes.
At the final response or commit boundary, acceptance closes atomically: the message is either
queued before closure and routed, or the send is rejected as a conflict. It is never accepted into
a session that can complete without applying it.

## Scope contract

The repository root, task focus, allowed paths, configured tasks, and expected outputs constrain
where delivery may act. An Apple-container-backed task-owned worktree separates the user's baseline
from in-progress work. Explicit local execution instead operates in the canonical repository and
uses exact baseline fingerprints plus stale-evidence rejection; another local process can still
change those files. Promotion and commit checks prevent undeclared output or unrelated repository
state from being quietly absorbed into the result.

For built-in file mutation, “read before write” means the exact bytes read still match immediately
before atomic replacement. A concurrent edit invalidates that evidence. Configured tasks stage and
validate their complete declared output set before transactional promotion and restore earlier
destinations if promotion fails. The agent cannot request recursive directory deletion.
Structural moves are likewise limited to files and symlinks, so a single allowed path cannot hide
a recursive subtree move.

For constrained FlashMoe and llama.cpp `write_file`, `replace_file`, `edit_file`, and canonical
`apply_patch` output, pb validates the exact virtual result before the model can finish the mutation
and repeats that validation at execution. Rust, Python, TypeScript/TSX, JavaScript/JSX, HTML, and CSS
results must be valid under pb's pinned complete-file syntax profile. A conservative prefix oracle
also rejects promoted local impossibilities such as an unmatched closer. It incrementally advances
new logical payload bytes, rolls candidate branches back to bounded checkpoints, and checks partial
canonical hunk additions/context before their newline; it is not a claim that every accepted prefix
is grammar-extendable. Existing-file and patch validation is bound to the exact controller-observed
base; stale bytes fail rather than being patched approximately. A controller-bound work unit also
binds every patch file to its one accepted path. If the observation covered only ranges, every
old-side hunk must remain wholly inside one of those exact byte ranges.

For a real local backend that may edit Rust in a Cargo project, pb automatically prepares a pinned
rust-analyzer world before inference. Immutable external dependency and sysroot facts can constrain
the stream; project-local facts remain repairable until the complete transaction is known. For an
exact no-build-script/no-procedural-macro profile, closure applies all modifications to already
indexed `.rs` files together in an independently writable database and may reject only an increase
in the promoted unresolved-name/supported-import, missing-field/method, privacy, selected
call/type/trait-bound, mutability, or moved-from-reference diagnostic debt. This check includes
untouched local modules affected by a changed API. New/deleted Rust files, unsupported relative
import contexts, partial profiles, complete borrow checking, trait-implementation completeness,
compiler behavior, and runtime behavior remain unknown. A fresh complete replay against the same
live world runs before executor entry; a generation cache or receipt is not authority. This narrow
claim is not general Rust symbol or type correctness and does not promise compilation.

For a real local backend that may edit Python, pb likewise prepares exact-pinned Astral `ty` state
before inference. The shipped v1 profile uses Python 3.12 as its fallback and binds the host target
platform, first-party sources, bundled typeshed, and configuration/dependency-manifest identities to
a frozen project world. When exactly one conventional project-local `.venv` or `venv` has a safe,
bounded static layout, pb additionally snapshots its Python/stub/package-marker metadata. A
plain-path `.pth` entry can add a fully observed repository directory as a first-party root. An
external environment or exact external editable root can participate only through user-owned
configuration bound to the canonical workspace; repository-owned `.pb/python.toml` can select only
an in-workspace environment. pb derives the supported Python version from `pyvenv.cfg`, primes every
captured dependency module before inference, and includes the dependency image in
cache/single-flight identity. It does not execute an interpreter or contact a package index. Request
overlays resolve newly generated files and imports together, and a deleted candidate
is absent from that transaction's import graph. It can reject a complete
generated string-plus-integer literal operation at a proven statement boundary and newly introduced
promoted `ty` diagnostics at closure. Argument, assignment, return, attribute, and import diagnostics
are never statement-prefix hard masks: later files in the same patch may repair them. The broader
unsupported-operator diagnostic is also closure-only unless the generated-literal proof applies.
Closure applies every candidate together, then checks every
non-deleted frozen first-party file plus newly created Python files, so an API change or deletion can
be rejected when it introduces a promoted error in an untouched in-project dependant. Generated
type-suppression directives cannot hide those errors. Missing external imports can veto only when
the static environment search space is complete. Ambiguous environments, undeclared `.pth` paths,
symlinked layouts, native extensions, import hooks, `Any`, dynamic imports, monkey-patching,
descriptors, runtime dispatch, dependants outside the frozen project, and unpromoted diagnostics
remain partial or unknown. The dependency image is recaptured before execution, then a fresh
complete replay runs against the original immutable world. This is not general Python type or
runtime correctness.

Settled-transaction LSP symbol/type claims are separately opt-in through a server's
`semantic_enforcement` mode.
Required mode accepts only a digest-pinned, complete, exact-overlay diagnostic comparison and repeats
the transaction in a fresh exact bounded shadow workspace and isolated provider session immediately
before publication. Rust evidence also requires a non-empty loaded crate graph and analyzer-confirmed
membership for every affected document. Generation and final-executor receipts are separate and
content-free. The claim is limited to the provider-classified document/target scope and immutable
provider, configuration, and dependency identities; it is not proof that code compiles, tests pass,
all dependants were checked, or dynamic Python/JavaScript behavior is safe. Disabled remains the
default, advisory results do not veto, and unqualified llama.cpp model/template profiles must not be
described as backend-parity qualified.

A broad task command is treated as broad authority, not described as a sandbox. The user remains
responsible for how much authority project configuration grants.

## Capability contract

The active workflow stage is the source of tool authority. Planning and review are read-only;
implementation and repair may mutate; checks and commit are harness-owned. Advisory agents receive
only bounded read-only authority and cannot delegate or advance the caller.

Project policy can make a permitted operation require approval or be denied. It cannot expose an
operation the stage does not allow.

The former model-owned `todo`, `git_commit`, and `git_revert` tools are retired from every current
surface; plans, checkpoints, and managed commit stages own those concerns. Dynamic MCP tools are
also fail-closed: only operator-declared read-only raw tool names are exposed, and server-provided
effect annotations cannot authorize a call. External MCP mutation has no current workflow surface.

Implementation mutation authority is narrower than the stage capability: a checkpointed work-unit
ledger selects one operation and path at a time. pb inserts that path into a target-bound call, so a
model-supplied alternate path cannot widen scope. Native constrained generation receives the same
binding out of band, so target-scoped tool schemas can omit `path` without weakening streaming
syntax or semantic checks. Existing-path operations still require an exact,
current complete-file read; adopted task-owned bytes retain separate provenance. A failed exact-path
diagnostic invalidates older reads and grants repair authority only for that current path.

## Acceptance contract

There are three distinct claims:

| Claim | Meaning |
| --- | --- |
| Model final | The model reached a response it considers final. Useful, but not external verification. |
| Workflow Ready | pb accepted the required plan, current checks and current review, then completed the local workflow. |
| Acceptance satisfied | An explicitly supplied harness contract passed its allowed-path, mutation, check, commit, and completion gates. |

Goal mode adds two higher-level claims without weakening those three:

| Claim | Meaning |
| --- | --- |
| Goal ready for review | Every current criterion has strict-workflow evidence, but at least one criterion is prose or explicitly user-owned. |
| Goal complete | Every current criterion is machine-verified, or the user accepted the exact current Goal checkpoint. |

A multi-Task run adds one orchestration claim:

| Claim | Meaning |
| --- | --- |
| Tasks complete | Every required Task reached committed or verified-no-change delivery, every dependency and repository boundary reconciled, and no pending Task remains. |

A child workflow Ready or Goal Complete is evidence only for its active Task. It cannot skip later
Tasks or make the parent Ready. A Goal Task amendment cannot remove an accepted criterion or change
the parent Task objective, authority, or budget; additions remain an explicit existing Goal user
decision and are recorded in the parent checkpoint.

A workflow Ready result is evidence for a criterion; it is not by itself permission to call a
multi-milestone Goal complete. A reviewer model's prose never converts a subjective criterion into
machine verification.

The hidden harness surface can receive a trusted JSON contract from outside its scratch workspace.
It is parsed before model loading and remains the source of task-specific acceptance facts. Without
one, pb reports contract status as unspecified rather than converting a confident final answer into
verified completion. Strict delivery validates required or forbidden mutation during planning and
against the final task delta, runs every required check, requires the named fresh-review reads and
check evidence, verifies commit requirements, and evaluates final workspace cleanliness. Only a
`Ready` or `NoChange` workflow with all explicit facts satisfied reports
`contract_status=satisfied` and `verified_completed=true`; a contract-free workflow remains
unverified, and an unmet explicit contract terminates as `contract_unsatisfied`.

Contract-required check IDs are controller-owned facts. A planner does not need to copy them into
its submission: pb projects any missing IDs into the accepted plan, recomputes its digest, and sends
that exact artifact to fresh plan review. Additional checks remain model-selected, while paths and
implementation scope remain model-proposed and contract-validated; projection cannot turn an
allowed path into a required mutation or add user intent.

Fresh plan and code reviews still account for every required assessment dimension. Each assessment
contains only its kind and status; a concern or failure records its reason and current evidence once
in the corresponding typed challenge or finding. pb rejects a passing verdict with a concerning
assessment and a revision verdict without both a blocking issue and a non-passing assessment.

An optional bounded `work_unit_guidance` map may give an exact task path a concise advisory hint.
It is trusted prompt context, not an acceptance fact: the hint cannot choose a path, authorize a
mutation, earn progress, satisfy a check or review, or advance the workflow. pb validates guided
paths against nonempty `allowed_paths` and surfaces only the active mutation-ready path's text.

Checks may opt into `diagnostic_eligible`. Such a check can run early only after structural work-unit
completion. Its result is repair feedback, not acceptance evidence: required checks rerun after the
typed implementation artifact, and only that authoritative run can satisfy checking, review,
commit, or verified completion.

A failed check may name a local support file such as a test in its bounded output. pb can include a
small, workspace-confined, read-only excerpt from at most two such regular UTF-8 files in repair
feedback. This does not add mutation authority or acceptance evidence, and it never follows a cited
path outside the workspace.

Configured LSPs add an intrinsic diagnostic contract. pb automatically inspects supported changed
task paths: syntax-classified errors during partial implementation and all error-severity
diagnostics once work is settled or being handed off. Reports are bound to the current workspace
epoch and path fingerprints; concurrent mutation discards them. Every matching server/path target
is accounted as completed, advisory, failed, or deferred. Only an explicit full pull-diagnostic
report completes a target; a fresh push-only publication can report useful errors but cannot prove
an empty target clean. Only complete coverage with no diagnostics is clean. Any content mutation
invalidates settled evidence across the task path set, while syntax
evidence is invalidated only for files whose bytes changed. A blocking report invalidates older read
and staging evidence for only the exact reported paths and requires a fresh read before repair.
Clean, failed, timed-out, or unavailable LSP evidence never satisfies a named check, review,
commit, completion claim, or Goal criterion. The model may still call manual read-only LSP tools
for targeted questions, but it is not responsible for triggering the proactive contract.
The proactive budget begins before repository observation: an oversized workspace, blocked launch,
blocked stdin, or revalidation timeout becomes visible incomplete advisory evidence and cannot delay
the controller beyond the pass deadline or be mistaken for a clean result.

## Evidence contract

A change-bearing Ready build carries evidence that is current for the managed commit:

- the accepted plan and its review;
- the implementation artifact;
- affected configured check receipts;
- the accepted fresh code review;
- content and evidence fingerprints;
- the managed commit OID and evidence-bundle digest;
- bounded usage and terminal outcome records.

Implementation accounting remains model-authored only for step status, bounded summaries, and the
semantic commit type and description. Its model-facing summary fields are capped at 1,024 characters;
the commit type is an enum and the description is bounded so pb's assembled semantic subject cannot
exceed 200 characters. Controller-owned plan identity, fingerprints, touched paths, no-change state,
and the assembled commit subject are not exposed in an inline final mutation completion and are
projected from current trusted state only after that mutation succeeds.

Evidence becomes stale after a relevant mutation. pb refreshes it or stops; it does not silently
reuse a receipt for earlier content. Git staging or committing does not itself change that content
identity: tracked-deletion sentinels are excluded, so a reviewed deletion keeps the same fingerprint
through the managed commit while any actual worktree-byte change invalidates the receipt.

File-read evidence binds the bytes actually returned, not merely a path observed at some point.
An eligible controller-executed complete read creates the same exact evidence intrinsically; its
typed receipt records controller origin, the truthful controller-block representation, coverage,
content-derived action identity, and current fingerprints. The receipt is revalidated immediately
before prompt admission. Partial ranges never claim complete observation or grant whole-file
replacement. A controller observation is one explicit user/context block with no model tool call,
and it cannot supply an approval, check result, review judgment, or semantic completion claim.
In user-facing transcripts, Trinity Walker may speak for this deterministic workflow stewardship,
but the durable event continues to record controller origin, the assisting profile, and its receipt.
Model-requested actions are attributed to that profile's character. The presentation labels these
origins **Model** and **Harness**. The v5 event contract requires the producer to identify every
model invocation, tool action, controller action, and deterministic correction; consumers never
borrow a character from adjacent chat. Harness validation messages summarize the teammate mistake
in plain language and keep the durable structured evidence in a hover-information or
touch-long-press detail sheet. The ordinary message does not expose temporary workspace paths.
Tool results name the exact durable call and batch they answer and provide a typed outcome and
duration; consumers do not correlate repeated calls by tool name or nearby actor. Team messages cite
checks and commits with tagged references, and the envelope embeds the matching typed evidence for
terminal, harness, and web renderers. Session summaries expose commits as structured object IDs and
subjects rather than asking consumers to parse Git log text.
Trinity's visible copy is event-specific: proactive evidence says which code Trinity inspected, and
failures say which artifact, tool, repeated action, diagnostic, or terminal condition needs
attention. Visible copy does not expose prompt/context transfer or tell a teammate to ask the
harness for repository lines.
Generic claims that Trinity "noticed a problem" are not used as a substitute for the durable cause.
When Trinity requests a next action, the visible message directly addresses the responsible teammate
rather than describing that person in the third person. Once the workflow has ended, the responsible
teammate no longer receives an impossible call to action. Trinity instead gives the teammate a final
conversational explanation and says their task is on hold. A second ordinary Trinity bubble is
authored in the durable event projection, addresses the local user by username, and asks for the
available restart, resume, or follow-up context. Terminal and web renderers therefore present the
exact same message without browser-only rewriting. No
terminal-status badge interrupts the chat. Stale repeat-control and duplicate failure corrections
immediately before that terminal handoff are suppressed while the first failure explanation and
intervening teammate action remain. Trinity's lilac identity accent is applied consistently to
provenance, information affordances, and a narrow message/action edge while her avatar remains
unoutlined. Other named teammates use the same neutral-surface treatment with an accent based on
their avatar background. Controller observations are labelled as predicted early repository work
with distinct provenance. Session efficiency copy may count those admitted observations and state an
equal upper bound on potentially avoided model turns, but it must not claim measured inference or
energy savings without a counterfactual run. Control-collar statistics describe rejected generation
candidates, and deterministic repeat or dependency gates describe prevented actions separately;
neither count is allowed to masquerade as the other.
In strict delivery and acceptance contracts, named check evidence comes only from `run_check(id)`;
a similar `run_command` or `run_task` result is diagnostic evidence, not an acceptance receipt. A
restored legacy/direct request may still route an exact configured guard command through the check
runtime for compatibility, but that path cannot satisfy a strict workflow contract.

## Failure contract

pb distinguishes failure modes instead of collapsing them into model prose. Blocked user input,
unavailable executors, failed checks, exhausted repair, control violations, step limits,
cancellation, and unsatisfied acceptance have distinct terminal paths.

Tool failures preserve the distinction too. Command timeout, user cancellation, nonzero exit, and
bounded output are structured results. MCP/LSP transport failures are distinct from remote
application errors, so an application failure cannot cause an unsafe automatic replay. Oversized
inputs and responses fail explicitly rather than being presented as complete evidence.

Durable tool events preserve that distinction for presentation as well: call and batch identities
include the session turn, survive result reordering and worker failure without losing the original
tool identity, and typed outcomes distinguish success, execution failure, validation or
policy rejection, timeout, cancellation, and deterministic cache replay. Older events without these
additive fields stay readable and are shown as unknown rather than upgraded to success.
Missing repository paths are a typed `target_not_found` failure rather than a generic retryable
execution error. pb does not execute an unchanged identical read twice. During plan review, the next
bounded turn offers either the structured review submission or one local discovery action, so the
reviewer can challenge the plan from evidence already held or locate the real symbol without
guessing another path.

Automatic LSP failures are fail-open only with respect to this advisory pre-check: Trinity records
the incomplete evidence and the normal configured checks remain authoritative. Current blocking
diagnostics are fail-closed for handoff until repaired or the bounded workflow stops. pb does not
apply a language server's edits, commands, formatting, or code actions.

Budgets apply across retries and advisors. Recovery can help express an allowed action, but it does
not erase usage, broaden authority, or turn a partial result into success.
When implementation or repair reaches its last ordinary turn with deterministic terminal
preconditions ready, only the typed implementation submission remains exposed. This preserves a
bounded opportunity to account for completed work; it does not infer completion, run a commit, or
bypass the later harness-owned checks and fresh review. If preconditions are not ready, mutation
authority remains unchanged and step exhaustion is still reported as incomplete.
Repeated capped native actions remain one bounded failure sequence while workspace and evidence
fingerprints are unchanged, even if an intervening action parses but then fails. A successful state
transition or executed tool result resets that sequence; a truncated file-write payload is never
treated as a partial file. Native constrained generation stops before an open mutation string can
cross its schema limit, and the same-step compact retry carries the reduced limit in the executable
schema as well as its prompt. Repeated or collapsing decoded prefixes cannot masquerade as output
progress. File mutation tools report success and earn progress only when repository bytes actually
change; an identical replacement is a typed tool failure rather than fresh evidence.
Both constrained inference backends refuse EOS or tool-envelope closure for an invalid supported
complete-file result. Qwen JSON and DeepSeek DSML use the same virtual mutation and syntax gate, and constrained batches
can contain at most one mutation call. A rejected canonical patch is not retried through the broader
llama.cpp/Git compatibility parser.

Goal budgets apply across all child workflows and do not reset between milestones, pause/resume, or
amendments. A project ceiling may narrow a user's request. A model can request budget review but
cannot apply an increase. Budget exhaustion is reported as a typed stopped outcome, never as
completion.

## Persistence contract

Workflow checkpoints preserve structured artifacts, counters, fingerprints, the typed work-unit
ledger, adopted-work provenance, unique progress credits, diagnostic repair focus, and terminal state.
After a service restart, unfinished daemon sessions restore as paused. The user chooses whether to
resume them; pb does not continue mutation merely because a process came back.

Goal checkpoints add the accepted objective, criteria, plan versions, retired criteria, milestone
and child-workflow evidence, total counters, authority and policy hashes, decisions, and terminal
basis. Mutating HTTP/RPC calls carry the current Goal digest; stale approval, pause, edit, cancel, or
accept requests conflict without altering state. A running Goal restores paused. A Goal already
waiting for initial plan approval or final user acceptance remains in that exact review state.

Stopping a Goal is preservation, not rollback: managed commits, current workspace changes, events,
and evidence remain. Editing after work begins similarly cannot rewrite completed history; it
supersedes only unfinished plan material after the replacement digest is approved.

The web collection boundary publishes project, session, usage, and terminal-transition state under
one process identity and monotonic revision clock. Usage totals are derived from the same captured
session projection as the collection rows. Requested calendar-window summaries for the whole
service and every registered project use a bounded server cache that is updated with new metrics
and invalidated atomically with session deletion or project-registry changes; a new window scans
retained turn records once instead of once per publication and subscriber. Collection rows do not
ship cumulative metrics or per-turn usage records for the browser to aggregate. An SSE subscription records its
transition floor inside that same publication boundary. Its first snapshot contains retained
transitions newer than that floor, and each later snapshot advances the connection floor and carries
only the next delta. Work that finishes while the first snapshot is being built is therefore still a
live transition rather than an indistinguishable completed row. Reconnect cursors are scoped to the
originating process identity. HTTP snapshots provide manual recovery data only and cannot replace
the live stream's process generation. Session deletion runs its durable removal, live projection,
usage accounting, revision publication, and cleanup in an owned transaction that continues if the
request disconnects. Its successful HTTP result carries both typed cleanup warnings and the exact
committed project/session snapshot captured at the deletion revision, using the caller's requested
usage window. Project-registry mutations use the same ownership rule: the disk transaction
returns the exact registry it committed, and an independently owned task reconciles sessions, usage
caches, and collection revision under one lock order. There is no fallible post-commit registry
reload and a disconnected HTTP or RPC caller cannot strand memory behind disk. Project mutations
return the same revisioned snapshot shape as the stream, so the browser can apply the authoritative
result even while SSE is reconnecting and can continue displaying its last valid snapshot as stale data.
Existing-session and Goal mutations likewise return the exact revisioned session snapshot used by the
session stream. New-session creation returns identity for navigation; starting a Goal in an existing
session uses the session mutation endpoint and returns the session snapshot. Cancellation requests and
the runner-resolved branch and focus root are explicit session state rather than browser inference or
event-sink-only persistence. Goal and cancellation mutations publish only after their digest checks,
checkpoint rebuild, any parent-Task fold, and the exact durable session write all succeed. The durable
write is serialized with autonomous event snapshots and includes terminal lifecycle events, preventing
an older writer from replacing an acknowledged checkpoint. A rejected or unpersistable mutation cannot
leak partial state or transcript events. Terminal RPC results retain the changed Goal's identity and
committed digest even if the session snapshot has already advanced to another Goal Task. A missing
session on the session-scoped Goal route is a typed `session_not_found` response rather than Goal-input
rejection.
Each session event sender owns the collection publisher, so every state-affecting event appends its
history entry and publishes the matching collection revision synchronously while the caller holds the
session lock. There is no asynchronous watcher reconstructing collection transitions after delivery.
Checkpoint-only projection changes publish through that same boundary. Terminal publication additionally
captures the immutable task, title, handoff, and registered-project projection at that transition.
Session deletion
does not report failure after removing the authoritative record: a
durable-record failure leaves the session intact, while later environment or workspace cleanup
problems are returned as warnings on a successful deletion.

## Publication contract

Ready means “reviewed and checked locally,” not “published.” Pushes, pull requests, remote CI,
review-comment handling, and merges are outside the current delivery contract. This is a deliberate
promise to stop at the external boundary until a separate approval-bearing publication workflow
exists.

Web-service exposure is a separate user-owned contract. Tailscale HTTPS access is disabled by
default and can be enabled immediately only through the Settings action or reconciled from an
explicit typed user preference at service startup. This authority belongs to the user-facing
service controller, not to a model, repository instruction, Build, Goal, or integration tool. pb
may inspect and change only its exact HTTPS port-to-loopback mapping. A conflicting endpoint is a
visible terminal condition for that action, not permission to overwrite, reset, make the service
public with Funnel, or alter tailnet policy.

## The practical promise

pb should always be able to answer four questions without asking you to trust the model's memory:

1. What did the user authorize?
2. What capabilities were active when each action ran?
3. What current evidence supports the outcome?
4. Which consequential action has not yet happened?

That is the connective tissue between workflow reliability, security, local privacy, and a truthful
relationship with the user.
