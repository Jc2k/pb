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
`.pb/tasks.toml` can narrow budgets but cannot qualify a planner or authorize automatic Goal
selection. Qualification is an embedded controller-owned record for the exact model bytes, backend,
prompt template, artifact protocol, and structured-output contract. The planner and critic are
token-constrained to exact controller schemas in both supported inference engines; schema validity
does not replace deterministic coverage, graph, authority, or budget validation, nor the fresh
semantic critique. The controller retains the verbatim objective, splits compound clauses at
punctuation boundaries, and supplies the only behavioral request clauses a Task may claim. It
automatically attaches decomposition-wide constraints to every Task. Each Task separately declares
outcome acceptance, owned tests, and documentation work or impact. In a multi-Task queue, a Build
Task may own at most two behavioral clauses and must have an outcome acceptance fact and test fact
for each. The critic must assess every supplied clause exactly once before its aggregate audits can
pass. A multi-Task proposal cannot create a queue entry that owns only decomposition
constraints; tests, documentation, ordering, and final validation remain with a behavior-owning
Task. An unqualified model stays on the ordinary Build path. A
qualified one-Task result unwraps into the same Build or Goal experience; only two or more Tasks
create a durable queue and Tasks UI.

The accepted high-level plan is not mutation authority. It can only narrow the original objective,
repository, workflow and Goal policies, request cap, aggregate allowance, and no-publication
boundary. Each queued Task becomes a fresh Build or Goal request only when its dependencies are
delivered and the repository matches the prior terminal checkpoint. Task-planning failure cannot
silently reinterpret the broad original request as one Build: retry, edit, and an explicit
run-as-one-Build decision are distinct user actions.

When planning discovers a materially missing choice, it may ask the user. The answer becomes part of
the task contract. Guessing would make progress faster at the cost of changing ownership of the
decision, so the workflow pauses instead.

## Scope contract

The repository root, task focus, allowed paths, configured tasks, and expected outputs constrain
where delivery may act. A task-owned workspace separates the user's baseline from in-progress work.
Promotion and commit checks prevent undeclared output or unrelated repository state from being
quietly absorbed into the result.

For built-in file mutation, “read before write” means the exact bytes read still match immediately
before atomic replacement. A concurrent edit invalidates that evidence. Configured tasks stage and
validate their complete declared output set before transactional promotion and restore earlier
destinations if promotion fails. The agent cannot request recursive directory deletion.
Structural moves are likewise limited to files and symlinks, so a single allowed path cannot hide
a recursive subtree move.

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
model-supplied alternate path cannot widen scope. Existing-path operations still require an exact,
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

An optional bounded `work_unit_guidance` map may give an exact task path a concise advisory hint.
It is trusted prompt context, not an acceptance fact: the hint cannot choose a path, authorize a
mutation, earn progress, satisfy a check or review, or advance the workflow. pb validates guided
paths against nonempty `allowed_paths` and surfaces only the active mutation-ready path's text.

Checks may opt into `diagnostic_eligible`. Such a check can run early only after structural work-unit
completion. Its result is repair feedback, not acceptance evidence: required checks rerun after the
typed implementation artifact, and only that authoritative run can satisfy checking, review,
commit, or verified completion.

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
Model-requested actions are attributed to that profile's character. Actorless legacy tool events
remain labelled as legacy instead of acquiring a guessed character.
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

## Publication contract

Ready means “reviewed and checked locally,” not “published.” Pushes, pull requests, remote CI,
review-comment handling, and merges are outside the current delivery contract. This is a deliberate
promise to stop at the external boundary until a separate approval-bearing publication workflow
exists.

## The practical promise

pb should always be able to answer four questions without asking you to trust the model's memory:

1. What did the user authorize?
2. What capabilities were active when each action ran?
3. What current evidence supports the outcome?
4. Which consequential action has not yet happened?

That is the connective tissue between workflow reliability, security, local privacy, and a truthful
relationship with the user.
