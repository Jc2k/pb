# External publication workflow follow-on goal

Status: proposed follow-on; intentionally outside local delivery W0-W9

## Objective

Add a provider-owned, approval-bearing workflow that consumes a completed
`ReadyEvidenceBundle`, publishes the exact reviewed commit, waits for remote
checks and review, and routes actionable feedback through a new bounded local
delivery cycle. Do not weaken or bypass the existing plan, check, review, or
managed-commit gates.

This must be a separate goal. Local `Ready` remains useful and truthful with no
network, forge account, or remote branch. Publication failure must not rewrite a
successful local delivery as though its evidence never existed.

## Required control flow

```text
local ReadyEvidenceBundle
  -> explicit publication approval (off | ask | auto policy)
  -> reconcile provider state by idempotency key
  -> push the exact commit OID
  -> create or update one PR/MR
  -> wait for bounded CI and review events
  -> ingest unresolved feedback as structured evidence
  -> fresh-context feedback triage
  -> no action, user decision, or bounded repair delivery
  -> local affected checks and fresh code review
  -> managed commit
  -> approved push and remote reconciliation
  -> terminal published, blocked, failed, or feedback-limit outcome
```

## Non-negotiable guardrails

- Provider operations are typed harness actions, never unrestricted model shell
  commands. Build-stage `run_command` receives no push, PR/MR, CI, approval, or
  feedback-resolution credit.
- The publisher validates the bundle digest and commit OID before any mutation.
  A stale workspace, changed commit, changed remote, or conflicting use of an
  idempotency key stops safely.
- `off` never mutates remote state. `ask` requires a durable user approval tied
  to the exact remote, branch, commit, and intended PR/MR operation. `auto` is
  available only through explicit project policy and the same scoped facts.
- Restart recovery reads remote state before retrying. The same idempotency key
  cannot create a second branch update, PR/MR, comment, or review-resolution
  action.
- CI polling, provider calls, feedback rounds, local repair cycles, model calls,
  tokens, and elapsed time all have independent durable bounds.
- Provider feedback is untrusted input. A fresh-context triage artifact records
  source IDs, applicability, severity, decision, and required user choices.
- A repair caused by feedback re-enters the complete local implementation,
  deterministic-check, fresh-code-review, and managed-commit path. Remote prose
  never grants local review credit.
- Resolution is reconciled against provider thread state. pb must not claim a
  comment resolved, check passed, branch pushed, or PR/MR updated without
  provider evidence.

## Configuration and interfaces

Extend project configuration additively with a publication policy, provider
identity, remote/branch rules, CI wait limits, and feedback-round limits. Keep
secrets in the existing provider credential store, never in workflow config,
events, prompts, or evidence bundles. Update `src/init.rs` with any new
per-project fields.

Evolve `ReadyEvidencePublisher` behind provider-neutral operations for
reconciliation, exact-OID push, PR/MR upsert, status collection, feedback
collection, and resolution. Every mutating request carries a stable
idempotency key and returns a provider receipt suitable for checkpointing.

## Delivery slices

1. Define the durable publication state machine, typed provider contracts,
   approval records, budgets, outcomes, and deterministic fake-provider tests.
2. Add read-only reconciliation and restart behavior with no provider writes.
3. Add exact-OID push behind `ask`, including conflict and duplicate tests.
4. Add idempotent PR/MR upsert and bounded CI waiting.
5. Add structured feedback ingestion and fresh-context triage.
6. Route approved repair through the existing local delivery workflow, then
   reconcile push and review-thread state.
7. Expose truthful controls and progress in web/API and `pb harness agent`;
   evaluate scripted failure/restart matrices before enabling `auto`.

## Completion contract

The follow-on is complete only when fake-provider tests prove at-most-once
external mutations across retries and restarts; no provider action is reachable
without policy and approval; exact local evidence is checked before push; CI and
feedback loops stop deterministically; repairs repeat every local guardrail; web
and harness report the same state; and real-provider smoke tests use a disposable
repository without weakening deterministic acceptance.
