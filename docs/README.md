<div class="pb-hero">
  <div class="pb-kicker">local coding agent</div>
  <h1>pb</h1>
  <p class="pb-lede">A local-first coding agent whose workflow—not the model—owns authority, review, checks, and completion.</p>
</div>

pb combines local model inference, a terminal client, an optional web interface, project-aware
execution environments, and a harness-owned delivery workflow. This site explains both how to use
it and why it behaves the way it does.

<div class="pb-grid">
  <a class="pb-card" href="user/getting-started.html">
    <strong>Use pb</strong>
    <span>Install it, configure a project, start the local service, and run work.</span>
  </a>
  <a class="pb-card" href="architecture/workflows.html">
    <strong>Follow a workflow</strong>
    <span>See how conversation becomes a reviewed, checked, managed commit.</span>
  </a>
  <a class="pb-card" href="architecture/security.html">
    <strong>Understand authority</strong>
    <span>Explore stage capabilities, policy decisions, isolation, and their limits.</span>
  </a>
  <a class="pb-card" href="architecture/local-privacy.html">
    <strong>Trace your data</strong>
    <span>Learn what stays local, what persists, and which choices can cross the machine boundary.</span>
  </a>
</div>

## The product in one loop

```text
you state intent
      ↓
pb selects a bounded workflow and exposes only its current capabilities
      ↓
the local model inspects, proposes, implements, and responds
      ↓
pb validates structured artifacts, runs configured checks, and controls commit ownership
      ↓
you receive the result, its state, and the evidence behind it
```

The model is an important participant, but it is not the authority. pb advances delivery only after
machine-checked stage transitions. A convincing sentence cannot substitute for an accepted plan,
fresh review, passing checks, or a managed commit.

## Two ways into the documentation

The [user guide](user/getting-started.md) is task-oriented. Start there when you want to run pb,
configure a project, manage integrations, or understand where it stores data.

The [architecture](architecture/overview.md) is an exploration organized around product questions:
how work flows, where authority lives, when data can leave the machine, and what pb promises to the
person using it.

## Status language

This site uses three kinds of claims:

- **Shipped** describes behavior enforced by the current source and tests.
- **Configurable** describes behavior that depends on project or user choices.
- **Design record** describes intent, migration history, or follow-on work and is not itself a
  runtime guarantee.

The curated architecture pages focus on shipped behavior and label important limits. The
[engineering records](conversational-delivery-workflow-plan.md) preserve the detailed plans,
invariants, benchmarks, and rollout evidence behind that behavior.
