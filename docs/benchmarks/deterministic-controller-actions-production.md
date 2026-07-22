# Deterministic controller actions production qualification

Captured: 2026-07-22

Plan: [Deterministic controller actions](../controller-action-elision-plan.md)

This qualification closes the production decision for intrinsic, truthful controller actions. It
tests two separate properties:

1. deterministic fixtures prove that unsafe or ambiguous cases do not execute; and
2. preserved local-model runs prove both prompt-admission fallback and the admitted positive path
   through the real strict workflow.

Artifact quality is intentionally trivial. The experiment evaluates pb's provenance, authority,
freshness, fallback, checks, review, commit, reporting, and UI/terminal event contract.

## Build identity

| Field | Value |
| --- | --- |
| Source base | `3d36d567f8b76118b61449014dce19138c539ff5` plus the pending deterministic-action implementation |
| Release binary SHA-256 | `b1ea715b5d6fbac26030f27cb74c0cacad650abd89b629095cfae1837e4c48aa` |
| Model | `hf://mlx-community/Qwen3-Coder-Next-4bit` |
| Backend | local FlashMoe `flashmoe-v2-mlxq4`, 48 layers, K=10 |
| Sampling | temperature 0, top-k 1, seed 0 |
| Generation cap | 512 tokens per model turn |
| Contract SHA-256 | `e335ff5f68683d30df6860c8b538acd9d3bf24991a8ba09ae74b465817f1ed89` |

The first sandboxed attempt could not see a Metal device and terminated before inference. It is an
experiment-environment error, not product evidence. Both reported runs used the same release binary
with local Metal access.

## Deterministic safety proof

The complete automated suite passed:

- `cargo test --all-targets`: 1,275 Rust tests passed, 22 ignored; both external environment
  contract tests passed;
- `deno task test:web`: 55 web tests passed;
- strict clippy, Rust formatting, web production build, release build, and rendered documentation
  link/fragment validation passed.

The focused controller fixtures prove:

| Boundary | Evidence |
| --- | --- |
| Truthful prompt shape | One `user` context block, `actual_origin=controller`, no tool calls, no tool-call ID |
| Retired transcript migration | Both old transcript-rendering strings deserialize only as `controller_block` |
| No request/config rollback | Persisted/client fields cannot disable controller actions; retired config keys are ignored, omitted on save, and rejected by `get/set` |
| Read bounds | Small UTF-8 full read succeeds; exact failed-diagnostic ranges are hash-bound; missing, binary, oversized, and unanchored partial inputs fall back |
| Freshness | Workspace/path fingerprints are rechecked immediately before prompt admission; later review mutation invalidates the observation |
| No-change closure | Only a structurally empty accepted plan under a trusted mutation-forbidden contract closes automatically |
| Delete containment | Stale, adopted, dirty, untracked, directory, forbidden, optional-contract, and oversized targets do not execute |
| Delete recovery | A tracked-clean file deletion emits controller provenance; a tracked symlink deletion removes the link without following or changing its target |
| Terminal presentation | Controller reads and deletions format as `pb action`; deletion states Git recovery |
| Web presentation | Contiguous model and controller events form one action group while retaining separate `Model` and `pb` actors |

## Preserved local-model runs

The fixture changed only `answer.txt` from `answer=41` to `answer=42`, required the named command
`grep -qx 'answer=42' answer.txt`, required fresh review evidence, required a semantic commit, and
required a clean workspace.

| Result | 8K fallback run | 32K admitted run |
| --- | --- | --- |
| Run ID | `1784757430558-50564-0` | `1784757864393-52561-0` |
| Controller observations | 0 | 1 full `inspect_change` |
| Controller prompt bytes | 0 | 1,141 |
| Review model invocations | 2 | 1 |
| Total model invocations | 7 | 6 |
| Session tokens | 30,941 | 26,974 |
| Checks | 2 passed, 0 failed | 2 passed, 0 failed |
| Commit | `9e570e3` | `13cd725` |
| Outcome | Ready, contract satisfied, verified completed | Ready, contract satisfied, verified completed |
| Events SHA-256 | `fdec457f8fe5e9aa72be1f4dcbda6f89ddb4fbe09cdd4ce9a590150522dd896f` | `7510e802ba0fb75f806faf31b088a6c0a96d1a3b1f37e37777c8a18fce22e766` |

The 8K run crossed the 55% prompt-admission threshold after adding the candidate inspection. pb
therefore emitted no controller receipt and exposed the normal `inspect_change` tool. The reviewer
used it and completed in two turns. This proves fallback rather than merely absence of a crash.

The 32K run emitted:

```text
[pb action] inspect_change answer.txt · full coverage · 953 bytes
```

Its durable receipt records action ID
`f62b5707d3f8aa6355f5ee8fbce7c9b9a2c4dca8676765b4fd23739c88247cf7`,
`actual_origin=controller`, `prompt_representation=controller_block`, `stage=code_review`, full
coverage, current workspace/path/content hashes, one hash-bound included range, 953 observed bytes,
1,141 prompt bytes, and only `prompt_context` plus `review_coverage` authority. There is no
`inspect_change` model tool call in that run. The next and only review model action is
`submit_code_review`.

## Ranked observations

### P1 — Controller admission and provenance held

- Classification: positive evidence.
- Evidence: the positive run's controller receipt and raw event sequence; the review stage contains
  one `controller_observation`, no `inspect_change` tool call, and then `submit_code_review`.
- Impact: pb saved one local-model invocation without attributing a tool choice to the model or
  granting review-verdict authority.
- Disposition: production-qualified.
- Recommendation: retain the receipt, actor, and no-synthetic-tool regression tests as release
  gates.

### P1 — Context pressure failed closed

- Classification: positive evidence.
- Evidence: the otherwise identical 8K run records zero controller observations and a real
  `inspect_change` call; it still reached a verified Ready outcome.
- Impact: an observation that cannot satisfy the prompt headroom policy does not become partial or
  assumed evidence.
- Disposition: production-qualified fallback.
- Recommendation: keep prompt admission tied to the active tokenizer/template and the current 55%
  bound.

### P1 — Completion, review, and commit remained independent

- Classification: positive evidence.
- Evidence: both runs executed the required check diagnostically and authoritatively, accepted a
  fresh model-authored review, created a semantic task-owned commit, ended with a clean worktree,
  and reported `contract_status=satisfied` plus `verified_completed=true`.
- Impact: controller observation did not bypass acceptance, review, or commit ownership.
- Disposition: production-qualified.
- Recommendation: none beyond retaining the current gates.

### P2 — Local model cost remains material

- Classification: model limitation.
- Evidence: the admitted run still used six model invocations and 26,974 session tokens for a
  deliberately tiny change; the controller observation removed exactly the redundant review
  inspection turn.
- Impact: deterministic actions improve cost but do not erase planning, critique, implementation,
  judgment, or completion costs that remain model-owned.
- Disposition: accepted local-model limitation.
- Recommendation: evaluate additional controller actions only when they meet the same unique,
  bounded, non-judgmental proof; do not weaken workflow stages to chase this cost.

No P0 or P1 pb defect was observed. The automatic journal's P3 progress notes were bounded
diagnostics and did not grant acceptance evidence.

## Preserved artifacts

- Fallback scratch: `/private/tmp/pb-deterministic-actions-qualification-20260722`
- Positive scratch: `/private/tmp/pb-deterministic-actions-positive-20260722`
- Each scratch contains the contract, workspace, cumulative `events.jsonl`, per-run events,
  `journal.md`, workflow checkpoint, run index, checks, diff, and task-owned commit.

## Decision

Ship truthful deterministic controller actions as intrinsic pb behavior across daemon, desktop,
web, queue, terminal, and direct harness execution. Do not ship a gaslight/transcript mode or an
off switch. Continue to fall back to real model tools whenever any eligibility, freshness,
boundedness, or prompt-admission proof fails.
