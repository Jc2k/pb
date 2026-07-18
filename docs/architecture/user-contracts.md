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

When planning discovers a materially missing choice, it may ask the user. The answer becomes part of
the task contract. Guessing would make progress faster at the cost of changing ownership of the
decision, so the workflow pauses instead.

## Scope contract

The repository root, task focus, allowed paths, configured tasks, and expected outputs constrain
where delivery may act. A task-owned workspace separates the user's baseline from in-progress work.
Promotion and commit checks prevent undeclared output or unrelated repository state from being
quietly absorbed into the result.

A broad task command is treated as broad authority, not described as a sandbox. The user remains
responsible for how much authority project configuration grants.

## Capability contract

The active workflow stage is the source of tool authority. Planning and review are read-only;
implementation and repair may mutate; checks and commit are harness-owned. Advisory agents receive
only bounded read-only authority and cannot delegate or advance the caller.

Project policy can make a permitted operation require approval or be denied. It cannot expose an
operation the stage does not allow.

## Acceptance contract

There are three distinct claims:

| Claim | Meaning |
| --- | --- |
| Model final | The model reached a response it considers final. Useful, but not external verification. |
| Workflow Ready | pb accepted the required plan, current checks and current review, then completed the local workflow. |
| Acceptance satisfied | An explicitly supplied harness contract passed its allowed-path, mutation, check, commit, and completion gates. |

The hidden harness surface can receive a trusted JSON contract from outside its scratch workspace.
It is parsed before model loading and remains the source of task-specific acceptance facts. Without
one, pb reports contract status as unspecified rather than converting a confident final answer into
verified completion.

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

## Failure contract

pb distinguishes failure modes instead of collapsing them into model prose. Blocked user input,
unavailable executors, failed checks, exhausted repair, control violations, step limits,
cancellation, and unsatisfied acceptance have distinct terminal paths.

Budgets apply across retries and advisors. Recovery can help express an allowed action, but it does
not erase usage, broaden authority, or turn a partial result into success.
Repeated capped native actions remain one bounded failure sequence while workspace and evidence
fingerprints are unchanged, even if an intervening action parses but then fails. A successful state
transition or executed tool result resets that sequence; a truncated file-write payload is never
treated as a partial file.

## Persistence contract

Workflow checkpoints preserve structured artifacts, counters, fingerprints, and terminal state.
After a service restart, unfinished daemon sessions restore as paused. The user chooses whether to
resume them; pb does not continue mutation merely because a process came back.

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
