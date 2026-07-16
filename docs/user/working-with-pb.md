# Working with pb

pb separates discussion from delivery. That separation is a user-facing authority boundary, not
just a change of prompt wording.

## Discuss

Use **Discuss** in the web interface for explanation, brainstorming, repository inspection, and
planning. A discussion is read-only. It can consult bounded advisory agents and it can propose a
delivery, but it cannot edit the project or start delivery on its own.

Discussion is useful when the important output is understanding:

- explain a subsystem or unfamiliar diff;
- compare approaches and surface trade-offs;
- investigate a failure without changing anything;
- shape a delivery request before granting mutation authority.

Only your explicit **Build** choice promotes a web conversation into the delivery workflow.

## Build

Use **Build** when the desired outcome is a repository change. A queue task is already explicit:

```bash
pb queue "Implement the accepted retry behavior" --workdir /path/to/project
```

Delivery moves through enforced stages: planning, independent plan review, implementation,
configured checks, independent code review, bounded repair when necessary, and a managed commit.
The active stage controls which tools are visible and which structured submission can advance the
workflow.

You may see pb pause for a planning question when a missing choice would materially change the
work. Answering that question updates the user-owned contract; it does not hand the model a general
permission to improvise.

## Profiles

The primary profile changes how pb approaches the request:

| Profile | Intended use |
| --- | --- |
| `build` | Deliver a concrete, scoped change. |
| `scout` | Inspect the repository and derive an appropriate development environment before delivery. |
| `explore` | Read-only investigation. |
| `plan` | Read-only implementation planning. |
| `review` | Read-only critique of a change or proposal. |
| `ask` | Read-only explanation and questions. |
| `research` | Public research with bounded repository context. |

For example:

```bash
pb queue --profile build "Add the missing test" --workdir /path/to/project
```

Advisory profiles can give the primary session fresh-context input. They cannot mutate the primary
workspace, delegate again, or advance its workflow stage.

## Reading outcomes

A delivery result is deliberately more precise than “the model stopped talking.”

- **Ready** means the enforced workflow reached its terminal success state. For a change-bearing
  build, pb owns the commit and binds it to accepted plan, review, and check evidence.
- **No change** means the request was resolved without a repository delta. It is not a hidden commit.
- **Blocked** means a required user decision, executor, check, or safety gate prevented progress.
- **Failed** means the bounded workflow could not satisfy its stage or repair contract.
- **Cancelled** means the run was explicitly stopped.

A contract-free harness final can still be useful, but it is not called externally verified. See
[Contracts with the user](../architecture/user-contracts.md) for the distinction between a model
answer, a Ready workflow, and a satisfied acceptance contract.

## What Ready does not publish

Ready is local delivery evidence. pb does not automatically push the branch, open a pull request,
merge, wait for provider CI, or respond to remote review comments. Those actions cross a separate
external authority boundary and remain follow-on work. The detailed proposal is preserved in the
[external publication record](../external-publication-workflow-follow-on.md).
