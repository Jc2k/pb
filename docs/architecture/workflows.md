# Conversation and delivery workflows

pb treats conversation and delivery as different authority scopes. A conversation can build shared
understanding. Delivery can change a repository, but only through an enforced, reviewable sequence.

## Conversation

A discussion starts with read-only capabilities. The model can inspect repository evidence, use
public research tools, consult bounded read-only advisors, answer, or propose delivery. It cannot
edit, run project commands, or commit.

```text
user message → read-only investigation → answer
                                  └────→ delivery proposal → user chooses Build
```

The proposal is not self-authorization. The transition happens only after explicit user intent is
recorded by the calling surface.

## Delivery

Delivery is a harness-owned state machine:

```text
Planning ──→ Plan review ──→ Implementing ──→ Checking ──→ Code review ──→ Committing ──→ Ready
    ↑              │              ↑              │              │
    └─ revision ───┘              └──── repair ──┴──────────────┘
                                      bounded retries
```

Each model-driven stage has one permitted structured terminal action. Checking and committing are
deterministic harness stages and do not run a model.

### 1. Planning

The planning profile reads the repository and submits a structured plan with scope, implementation
steps, and checks. If a user-owned choice is missing, planning can ask a question. The stage cannot
edit files or run a shell.

### 2. Plan review

A review profile receives focused evidence in a fresh context and either accepts the plan or
returns actionable findings. Rejection returns the workflow to bounded plan revision. The reviewer
does not inherit mutation authority.

### 3. Implementation

The build profile receives the accepted plan and mutation capabilities. It can use built-in edits,
configured tasks and checks, or a journaled command escape hatch. It submits an implementation
artifact that identifies what changed and the evidence it produced. It never receives `git_commit`.

### 4. Checks

The harness selects affected configured checks and records their current results. Named acceptance
checks cannot be replaced by an arbitrary command that happens to exit successfully. A mutation
after a check makes older evidence stale.

### 5. Code review

A fresh read-only reviewer inspects the delivered change and current focused evidence. An accepted
review is bound to the content fingerprint it saw. Findings enter a bounded repair cycle; repair
then refreshes checks and review.

### 6. Managed commit

When the accepted plan, checks, and code review all describe the current content, the harness owns
the commit operation. It verifies the repository boundary and builds a durable evidence bundle that
binds the commit OID to the workflow artifacts and check receipts.

## Stage capability matrix

| Capability | Discuss | Plan/review | Implement/repair | Check/commit |
| --- | ---: | ---: | ---: | ---: |
| Read repository | yes | yes | yes | harness only |
| Public research | yes | yes | yes | no |
| Mutate repository | no | no | yes | harness only |
| Run commands/tasks/checks | no | no | yes | harness only |
| Advisory input | yes | yes | yes | no |
| Commit | no | no | no | harness only |

Tool schemas are derived from this matrix for every stage. A request-level allowlist and project
policy can narrow the set further. Neither can broaden it.

## Freshness is part of correctness

Plans, checks, and reviews carry content or evidence fingerprints. A later mutation invalidates
evidence tied to the earlier state. This makes “reviewed” mean reviewed at the delivered content,
not merely that a review-shaped conversation occurred somewhere in the session.

The same rule applies during recovery. Persisted structured artifacts and fingerprints can restore
a stage; discarded or truncated prose cannot become a substitute source of authority.

## Bounded failure and recovery

The workflow budgets stage steps, model invocations, generated tokens, advisory calls, plan cycles,
and repair cycles. It also detects repeated no-progress operations against unchanged state and
stops loops deterministically.

When a model response is truncated before producing a valid action, pb can retry with thinking
disabled and, within the same global budget, use a larger output cap. These recovery mechanics help
the model express an allowed action; they never expose a new capability or waive a transition gate.

## Where the workflow ends

Ready means local delivery is complete under the configured contract. Publication is a separate
workflow because pushing, opening a pull request, and responding to provider state are external
mutations with different approvals and idempotency needs.

For the detailed implementation record and transition invariants, see the
[conversational delivery workflow plan](../conversational-delivery-workflow-plan.md).
