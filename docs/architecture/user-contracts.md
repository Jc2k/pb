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
reuse a receipt for earlier content.

File-read evidence binds the bytes actually returned, not merely a path observed at some point.
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

Workflow checkpoints preserve structured artifacts, counters, fingerprints, and terminal state.
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
