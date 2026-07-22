# Controller-owned deterministic action elision

Status: **Design record; prompt-rendering experiment required before implementation**

This follow-on explores whether pb can save local-model invocations by executing uniquely determined
observations and bookkeeping in the controller. It extends the
[verified task-completion reliability plan](task-completion-reliability-plan.md) and its typed
work-unit controller. Nothing in this record is a shipped guarantee.

## Outcome

pb should spend model invocations only on choices or judgments that require model intelligence. A
controller may elide an action when its result is locally computable, uniquely determined, bounded,
non-authorizing, and independently auditable. The optimization must preserve verified completion,
fresh evidence, allowed-path restrictions, review independence, managed commit ownership, and local
privacy.

The first question is deliberately empirical: a tool-trained local model may behave better when a
controller-executed read is rendered in the familiar assistant-tool-call/tool-result shape than when
the same bytes appear in a prose controller block. pb must measure that possibility before choosing
the production representation.

## Truth boundary

Controller execution and prompt representation are separate facts.

- The durable event stream records the actual actor: model, controller, deterministic workflow, or
  user.
- A controller action must execute successfully before any result is rendered to the model.
- A prompt-only assistant tool call never becomes a model `ToolCall` event, model-authored intent,
  progress credit, approval, review judgment, or check receipt.
- Every injected result remains bound to its current path and workspace fingerprints.
- A bounded excerpt never claims that the model observed the complete file.
- Prompt compaction cannot convert a controller action into model provenance.
- No compatibility transcript may represent mutation, approval, publication, a passing check, or a
  review verdict.

The implementation must distinguish:

1. **executor evidence** — pb observed enough exact state to enforce concurrency and mutation
   safety; and
2. **model observation coverage** — the exact full content, ranges, metadata, or failure actually
   included in the model prompt.

## Existing bounds

The current controller reads at most 8 MiB through `read_file` and the existing text mutation tools.
A carried complete-file evidence entry is limited to 16 KiB, with 24 entries and 64 KiB total per
bundle. A rendered tool result is limited to 16,000 prompt characters and may be compacted when the
measured prompt crosses its soft context limit. Action elision must respect or tighten these bounds;
it must not silently increase them.

## Eligibility classes

| Class | Examples | Initial policy |
| --- | --- | --- |
| Controller-observable | exact active-path read, metadata, bounded search result, changed-path inspection | eligible after the rendering experiment |
| Controller-derivable | plan digest, current fingerprint, actual touched paths, no-change fact, selected check IDs | eligible when projected as controller-owned facts |
| Controller-executable | uniquely accepted tracked-file deletion, no-change transition | experimental only; requires a later explicit gate |
| Model-required | edit content, resolve ambiguity, plan scope, substantive review, findings, semantic summary | never synthesized by the controller |
| User-required | approval, external publication, destructive expansion, subjective acceptance | never elided |

## E0 — experiment-only provenance and rendering surface

Add a typed harness-fixture field rather than an environment toggle or production setting:

```text
observation_rendering = native | controller_block | disclosed_tool_transcript | compatibility_tool_transcript
```

The field is accepted only by the hidden evaluation surface. It selects prompt representation, not
authority. All controller arms execute the same bounded read, produce the same content hash and
observation coverage, seed the same executor evidence, and differ only in prompt messages.

Every controller arm emits a durable receipt containing at least:

```text
schema_version
action_id
actual_origin = controller
prompt_representation
stage and work_unit_id
operation and normalized path
workspace/path/content fingerprints
coverage = full | ranges | metadata_only | none
observed and prompt byte counts
included ranges and their hashes
included_in_prompt
authority_effects
fallback or invalidation reason
```

Synthetic call IDs live only in the prepared prompt. They cannot collide with real call IDs, enter
the durable model transcript as authored calls, or be replayed as new evidence from the read cache.

Promotion proof:

- serialization, checkpoint, compaction, and event/journal tests preserve actual origin;
- no prompt renderer changes gate state or the result bytes;
- an interrupted or failed controller read creates no successful synthetic transcript;
- the production configuration and tool schemas remain unchanged.

## E1 — compatibility-transcript experiment

### Hypothesis

A tool-trained local model may produce a more correct immediate continuation after a familiar
assistant-tool-call/tool-result sequence than after an equivalently informative prose block. The
benefit, if any, must be separated from the invocation saved by controller execution.

### Locked arms

1. **Native read.** The model selects and calls `read_file`; this is the behavioral and efficiency
   control.
2. **Explicit controller block.** pb executes the read and inserts a canonical labelled observation
   in the dynamic user/context suffix.
3. **Disclosed tool transcript.** A system line says that pb executed the deterministic call on the
   model's behalf, followed by a prompt-only assistant tool call and the exact tool result.
4. **Compatibility tool transcript.** The prompt contains the assistant tool call and result without
   the disclosure line; durable provenance still says controller.

Model, model digest, chat template, tool schemas, context size, generation cap, sampling, seed,
task, contract, workspace bytes, evidence, result truncation, and cache conditions remain locked.
The controller arms must use byte-identical results. The native arm's extra generation is reported
separately rather than normalized away.

### Fixture matrix

The initial screen uses small deterministic tasks where the next read is uniquely selected:

- modify one small existing UTF-8 file;
- delete one tracked unchanged file and modify a second small file;
- repair an exact path named by a failed diagnostic;
- review one small changed file;
- receive a complete but irrelevant allowed-path read; and
- receive a read failure.

Safety fixtures then cover partial large-file content, stale evidence, binary input, more than one
plausible target, and a controller result that is invalidated before generation. These cases test
whether a compatibility transcript causes inappropriate confidence or mutation.

Run the four-arm screen once per fixture in fresh scratch roots. Advance the best two controller
representations to three locked repeats over the qualifying TC3 modify, delete/modify, diagnostic,
and review cases. Preserve every scratch root and independently audit journal, events, exact delta,
checks, review, commit, and clean status.

### Measurements

Functional and safety measures:

- correct first action after the observation;
- redundant reread rate;
- correct target and mutation-tool selection;
- malformed, repeated, or rejected actions;
- verified completion and exact artifact checks;
- unsupported claims after partial, irrelevant, stale, or failed observations;
- false model attribution, false evidence, forbidden mutation, and false completion.

Efficiency measures:

- model invocations and stage steps;
- rendered, cached-prefix, and fresh-prefill tokens;
- generated tokens;
- prompt bytes attributable to the representation;
- wall time and energy; and
- compaction and context-limit incidence.

### Selection rule

No representation advances with any false evidence, false attribution, authority expansion, or
unsafe continuation. Among safe candidates, prefer the disclosed transcript when it is within one
verified completion of the best controller arm in the screen and within five percentage points in
the repeated series. An undisclosed compatibility transcript requires a repeatable improvement of
at least ten percentage points across two independent series before it merits a separate design
decision; it is never promoted implicitly.

If neither transcript materially improves behavior over the explicit controller block, production
uses the explicit block. A model-family-specific win remains evaluation evidence until another
supported local tier confirms it or the rendering is explicitly scoped by typed model capability.

## E2 — production observation receipt

After E1 chooses a representation, introduce a versioned controller observation in workflow state.
It records origin, current fingerprints, coverage, prompt representation, persistence, and
invalidation. The event stream and journal show the receipt independently of its model-facing
rendering.

Likely implementation boundaries:

- `src/workflow/evidence.rs` owns the typed receipt and coverage;
- `src/agent_core.rs` selects, executes, validates, and renders an eligible observation;
- `src/agent_context.rs` measures and reserves context budget;
- `src/events.rs`, harness reports, and web event types expose provenance;
- read-cache replay preserves result provenance without re-executing authority effects.

## E3 — dynamic observation budget

Before injecting content, pb measures the base prompt with the actual model tokenizer. It preserves
the generation reserve, safety margin, and compaction target, then assigns the remaining observation
budget only to the active work unit. Evidence is selected in this order:

1. complete current content;
2. exact failed-diagnostic or changed-hunk ranges;
3. exact accepted symbol/search anchors;
4. metadata-only receipt; or
5. the existing model read/search surface.

Tool schemas remain canonical; all observation content stays in the dynamic suffix. If inclusion
would cross the target, pb reduces coverage or declines elision rather than shrinking the generation
reserve or dropping authoritative anchors.

## E4 — small-file read elision

For an exact accepted-plan path whose complete UTF-8 bytes fit the carried-evidence and measured
prompt budgets, pb:

1. reconciles the work-unit ledger;
2. reads and hashes the complete current bytes;
3. records a full controller observation;
4. includes those exact bytes through the E1-selected rendering;
5. seeds the same fingerprint in the write gate; and
6. begins generation with the unit mutation-ready and without redundant `read_file` exposure.

Any read failure, path-kind mismatch, context shortfall, or stale fingerprint falls back to the
existing evidence-needed state. A successful eligible task must save at least one model invocation
with identical final bytes and verification.

## E5 — localized large-file observation

Files larger than the full-observation budget use explicit coverage rather than a false complete-read
claim. pb may stream/hash the file for executor safety, but the model receives only deterministic
ranges and metadata. A range-bound edit records the file hash, byte interval, interval hash, UTF-8
boundaries, and exact old-text or patch anchor. It cannot authorize replacement outside observed
ranges.

Deterministic range sources are limited to exact diagnostic locations, changed diff hunks, accepted
symbols, or explicit contract anchors. If none exists, pb retains `ripgrep` and ranged `read_file`
instead of guessing relevance.

## E6 — fresh review inspection elision

At isolated code-review entry, pb may compute `inspect_change` material before generation. When every
required changed path fits, the fresh reviewer receives all exact inspections and may submit its own
assessment immediately. If any path has only partial or metadata coverage, `inspect_change` remains
available and the review terminal remains hidden until the existing coverage contract is satisfied.

The controller never supplies assessments, findings, severity, or verdict. Deleted, renamed,
symlink, and binary paths keep their explicit manifest representations.

## E7 — implementation closure fusion

To remove a separate bookkeeping generation without fabricating model judgment, the active mutation
schema may accept optional completion material: accepted-step statuses, summary, and semantic commit
subject. If the atomic mutation succeeds and structurally completes the queue, pb combines those
model-authored fields with controller-projected plan identity, actual paths, no-change fact, and
current fingerprint. Invalid or premature completion material is ignored or rejected without
advancing the workflow.

A contracted no-change task may use a controller-owned structural receipt after plan review, but
checks and any substantive review remain unchanged.

## E8 — conservative automatic deletion

Automatic deletion is last and initially harness-only. It requires a unique active delete in an
accepted and freshly reviewed plan, deliver intent, explicit contract/policy allowance, a tracked
unchanged file or symlink, exact baseline/invocation identity, and no adopted or untracked bytes.
Directories, ambiguous resolution, dirty content, untracked files, and user-owned partial work are
ineligible. Symlink deletion unlinks the link and never follows it.

Product promotion requires a separate approval and recovery decision. A model tool call remains the
fallback; the controller does not broaden destructive authority merely to save a step.

## Edge-case matrix

| Condition | Required behavior |
| --- | --- |
| UTF-8 file at or below 16 KiB and within measured prompt budget | full observation may elide the read |
| Small file that does not fit remaining context | bounded coverage; keep targeted reads available |
| UTF-8 file from 16 KiB through 8 MiB | full executor hash, deterministic ranges only, range-bound edit |
| File above 8 MiB | metadata/streaming hash only; localized command, replan, or explicit action |
| Binary or invalid UTF-8 | metadata and hash; no automatic text mutation |
| Minified file or one huge line | UTF-8-aligned byte windows rather than line-only ranges |
| Symlink | `lstat`, never implicit follow; unlink only when deletion is explicitly eligible |
| File changed after observation | invalidate receipt and mutation evidence before execution |
| File missing or recreated | complete delete, block modify for replan, or re-observe the new identity |
| Several paths exceed total budget | active path only; never eager-load the queue |
| Permission or read failure | structured controller failure; no successful transcript or evidence |
| Prompt compaction | rebuild current observation from workflow state; retain origin and coverage |
| Sensitive or secret-declared path | no eager read unless exact typed task authority permits it |
| Untracked or adopted deletion | never automatically delete |
| Cancellation | preserve prior checkpoint; grant no observation or mutation progress |

## Qualification ladder

1. Scripted tests prove origin, coverage, budget, invalidation, fallback, and terminal gates.
2. E1 selects a prompt representation from preserved native experiments.
3. E4 qualifies one small modify and one delete/modify task 3/3 against non-elided controls.
4. E5 qualifies oversized, partial, binary, UTF-8-boundary, stale, and multi-file containment.
5. E6 qualifies fresh review across small, large, deleted, renamed, symlink, and binary paths.
6. E7 qualifies closure without missing or synthesized semantic fields.
7. E8 remains harness-only until its destructive-action review is separately approved.
8. Run the final-source TC3 corpus and supported model tiers.

Every promotion requires zero false verified completions, forbidden mutations, evidence grants,
review claims, or model attribution; identical independently checked artifacts; clean semantic
commits where required; and a measured wall-time and energy improvement rather than call-count
reduction alone.

## Documentation and configuration obligations

Any shipped behavior updates the curated workflow, user-contract, security, and local-privacy
chapters in the same commit. A new project policy field must also update initialization and user
configuration documentation. No `PB_*` environment switch is permitted. Experiment-only rendering
stays on the typed hidden harness surface until a promotion gate explicitly moves it into supported
configuration.
