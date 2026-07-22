# Deterministic controller actions

Status: **Shipped as intrinsic local workflow behavior**

pb spends model invocations on choices and judgments, not on uniquely determined local operations.
Eligible reads, review inspections, no-change closure, and narrowly safe deletions are therefore pb
actions rather than optional model actions. There is no user mode, request field, environment
switch, or production prompt-rendering choice.

This is the production closure of the earlier controller action-elision investigation. The
transcript-shaped experiment—especially the undisclosed compatibility arm—has been retired. pb
never inserts a fabricated assistant tool call or tool result. A controller observation reaches the
model as one explicit user/context message labelled `actual_origin=controller` and
`prompt_representation=controller_block`.

## Product contract

- Normal daemon, desktop, web, queue, and `pb harness agent` workflows all use deterministic
  controller actions when their safety gates pass.
- Ineligible operations fall back to the ordinary model/tool path. The workflow does not weaken a
  gate to save an invocation.
- The hidden evaluator retains `native` only as a control arm for proving result equivalence. It is
  not a product or direct-harness option.
- Legacy `[agent]` action-elision settings remain readable so old configuration files do not break,
  but pb ignores them, omits them when saving, and rejects `pb config get/set` for those keys.
- Old serialized transcript-rendering names deserialize as `controller_block`; they cannot restore
  transcript-shaped prompting.

## Actions and actors

The durable event stream distinguishes model tool calls from pb actions. Web and terminal surfaces
present both in one action timeline without merging their authorship:

| Actor | Examples | Presentation |
| --- | --- | --- |
| Model | `edit_file`, `run_check`, `submit_code_review` | `Model` in the web action list; `tool` in terminal output |
| pb | eligible read, changed-path inspection, no-change closure, safe deletion | `pb` in the web action list; `pb action` in terminal output |

The web session's **Actions** panel contains both categories, and the compact transcript entry uses
the same actor labels. pb actions are also included in the activity history. A controller action
never emits a model `tool_call` or `tool_result` event.

## Eligibility and fallback

| Operation | Controller eligibility | Fallback |
| --- | --- | --- |
| Read an existing planned path | Active modify/delete work unit; regular non-symlink UTF-8 file; at most 8 MiB; complete rendered result fits the bounded prompt and stage-evidence limits | Model receives `read_file`; missing, unreadable, binary, symlinked, oversized, stale, or context-ineligible inputs are not injected |
| Read failed-diagnostic ranges | Active modify work unit; exact diagnostic path/line anchors; every included byte range is hashed; only an edit wholly inside a range is exposed | Model rereads or requests replanning; whole-file replacement remains unavailable |
| Inspect review changes | Code-review stage; every required changed path is reviewable, fresh, and fits the all-or-none prompt admission check | Reviewer uses normal inspection tools |
| Close no-change implementation | Accepted structurally empty plan; trusted contract explicitly forbids mutation; repository still equals the task baseline | Model-owned structured implementation submission |
| Delete a file or symlink | Unique accepted delete work unit; implementing stage; non-adopted; baseline, invocation, ledger, and live fingerprints match; path is allowed, Git-tracked, clean, and no larger than 8 MiB; an attached contract, if any, requires mutation | Model tool path or a blocked workflow; directories, untracked paths, dirty paths, adopted work, stale paths, and oversized files are never automatically deleted |

Tracked symlinks are removed as links without following their targets. Eligible deletion emits a
bounded diff, a controller mutation receipt, and explicit Git-recovery text. Managed commit remains
the later deterministic workflow stage; the action does not claim that a commit, review, or check
has passed.

## Safety invariants

### Truthful provenance

1. The operation executes successfully before pb records or prompts with its result.
2. Every receipt records `actual_origin=controller`, operation, stage, normalized path, coverage,
   observed and prompt bytes, fingerprints, content hashes, included ranges, and authority effects.
3. Action IDs are content-derived. Timestamps and scratch paths do not determine identity.
4. Controller prompt messages have role `user`, contain no tool calls, and have no tool-call ID.
5. Prompt compaction cannot turn controller evidence into a model-authored action.

### Freshness and concurrency

1. Candidate reads bind the current workspace, path, and content fingerprints.
2. pb renders and measures the complete candidate with the active model tokenizer/template.
3. The observation must survive prompt preparation unchanged and keep the prompt at or below 55%
   of usable capacity.
4. pb captures the workspace again immediately before admission. Any mismatch falls back.
5. Mutation tools retain their normal current-content checks; controller evidence does not bypass
   atomic replacement or active-work-unit scope.

### Bounded authority

- A full observation can grant exact read-before-write evidence only for its bound fingerprint.
- A range observation can grant only range-confined edit authority.
- A review observation grants inspection coverage, never assessments, findings, or a verdict.
- No controller observation grants approval, publication, network access, check success, semantic
  summary, completion, or commit authority.
- No-change closure is limited to a trusted mutation-forbidden contract and a verified unchanged
  baseline.
- Deletion is a real mutation and therefore has the strictest structural, Git, size, and freshness
  gates.

## Proof strategy

Production promotion requires evidence at three levels.

### Deterministic unit and integration tests

The test matrix covers:

- truthful single-message controller rendering with no assistant/tool transcript;
- migration of old rendering names without restoring transcript behavior;
- request and persisted-session attempts to disable intrinsic controller actions;
- small complete reads, exact failed-diagnostic ranges, missing files, binary files, oversized
  files, and context admission;
- stable content-derived action identity and stale workspace/path rejection;
- review all-or-none loading and invalidation after a later mutation;
- no-change closure only for a structurally empty mutation-forbidden contract;
- deletion rejection for stale, dirty, untracked, adopted, contract-optional, forbidden, directory,
  and oversized targets;
- tracked-clean file deletion and tracked symlink deletion without following the target;
- durable controller events without model tool-call events; and
- web grouping that preserves `Model` and `pb` actor labels.

### Locked native/controller evaluator

`pb harness action-elision-eval` remains a hidden qualification command for a two-arm experiment:

1. `native`: the local model requests the expected read;
2. `controller_block`: pb performs the same read and supplies a truthful context block.

The evaluator records source, executable, model, fixture, and configuration digests; raw events;
generation inputs; rendered prompt digests; and final artifacts. Protocol version 2 rejects the run
unless:

- fixture inputs are identical;
- native and controller observation-result bytes are identical;
- a controller receipt and matching generation-input action ID are present; and
- the controller input is exactly one controller-owned user/context block with no model tool call
  or tool-call ID.

Behavioral differences remain reported separately. A weak model continuation is not mislabeled as
a provenance defect, but it also cannot weaken these protocol checks.

### Preserved end-to-end harness run

Final qualification uses a blocking `pb harness agent` run from the release binary. The scratch
root, `journal.md`, JSONL events, contract, final repository state, checks, review, and commit are
preserved and audited. Review classifies observations as pb defect, model limitation, experiment
error, or positive evidence.

## Historical experiment decision

The earlier E1/E2 records remain historical evidence:

- [E1 representation screen](benchmarks/controller-action-elision-e1.md)
- [E2 byte-locked qualification](benchmarks/controller-action-elision-e2.md)

They established that controller execution can remove a model turn while retaining exact local
evidence. They did not establish a production benefit for impersonating an assistant tool call. The
product decision is therefore stronger than merely leaving that arm disabled: transcript emulation
has been removed, while truthful deterministic actions are intrinsic.

## Remaining boundary

pb may add other deterministic actions only when their result is unique, local, bounded,
freshness-checkable, non-judgmental, and independently auditable. Editing content, resolving plan
ambiguity, interpreting a review, choosing a semantic claim, approving work, or publishing data
continues to require the model or user actor that owns that decision.
